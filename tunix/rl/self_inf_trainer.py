# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RL trainer with self-influence-style batch curation."""

from __future__ import annotations

from typing import Any, Callable, Literal

from flax import nnx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
import optax
from tunix.rl import trainer as rl_trainer


class SelfInfTrainer(rl_trainer.Trainer):
  """Filters samples whose gradients anti-align with a mean gradient."""

  def __init__(
      self,
      *args,
      scope: Literal["batch", "group"] = "batch",
      num_generations: int | None = None,
      dot_threshold: float = 0.0,
      **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self.scope = scope
    self.num_generations = num_generations
    self.dot_threshold = dot_threshold
    self._score_loss_fn: Callable[..., Any] | None = None
    self._score_loss_has_aux = False

  def _with_score_loss_fn(
      self, score_loss_fn: Callable[..., Any], *, has_aux: bool
  ) -> None:
    """Configures a score-only loss without changing the update loss."""
    self._score_loss_fn = score_loss_fn
    self._score_loss_has_aux = has_aux

  def with_num_generations(self, num_generations: int) -> None:
    self.num_generations = num_generations

  def _validate_group_config(self, batch_size: int) -> None:
    group_size = self.num_generations
    if group_size is None or group_size <= 0:
      raise ValueError(
          "SelfInfTrainer with group scope requires a positive num_generations."
      )
    if batch_size % group_size != 0:
      raise ValueError(
          "SelfInfTrainer group scope requires batch size divisibility by "
          f"num_generations. Received batch_size={batch_size}, "
          f"num_generations={group_size}."
      )

  def _train_step(
      self, model: nnx.Module, optimizer: nnx.Optimizer, inputs: Any
  ) -> tuple[ArrayLike, Any | None, ArrayLike]:
    inputs = self.gen_model_input_fn(inputs)

    def _restore_batch_dim(x):
      if isinstance(x, (jax.Array, np.ndarray)):
        return jnp.expand_dims(x, axis=0)
      return x

    def per_sample_loss_fn(model, inputs):
      inputs = jax.tree_util.tree_map(_restore_batch_dim, inputs)
      if isinstance(inputs, dict):
        return self.loss_fn(model, **inputs)
      return self.loss_fn(model, inputs)

    wrt = nnx.LoRAParam if self._lora_enabled else nnx.Param
    grad_fn = nnx.value_and_grad(
        per_sample_loss_fn,
        argnums=nnx.DiffState(0, wrt),
        has_aux=self._has_aux,
    )

    def _input_in_axes(x):
      return 0 if isinstance(x, (jax.Array, np.ndarray)) else None

    inputs_in_axes = jax.tree_util.tree_map(_input_in_axes, inputs)
    vmapped_grad_fn = jax.vmap(grad_fn, in_axes=(None, inputs_in_axes))
    out, per_sample_grads = vmapped_grad_fn(model, inputs)

    if self._has_aux:
      per_sample_loss, per_sample_aux = out
    else:
      per_sample_loss = out
      per_sample_aux = None

    score_grads = per_sample_grads
    if self._score_loss_fn is not None:

      def per_sample_score_loss_fn(model, inputs):
        inputs = jax.tree_util.tree_map(_restore_batch_dim, inputs)
        if isinstance(inputs, dict):
          return self._score_loss_fn(model, **inputs)
        return self._score_loss_fn(model, inputs)

      score_grad_fn = nnx.value_and_grad(
          per_sample_score_loss_fn,
          argnums=nnx.DiffState(0, wrt),
          has_aux=self._score_loss_has_aux,
      )
      vmapped_score_grad_fn = jax.vmap(
          score_grad_fn, in_axes=(None, inputs_in_axes)
      )
      if self._score_loss_has_aux:
        (_, _), score_grads = vmapped_score_grad_fn(model, inputs)
      else:
        _, score_grads = vmapped_score_grad_fn(model, inputs)

    leaves = jax.tree_util.tree_leaves(per_sample_grads)
    if not leaves:
      raise ValueError("No gradients found for self-influence curation.")
    batch_size = int(leaves[0].shape[0])

    def _scores_batch_scope(gradient_tree):
      mean_grads = jax.tree_util.tree_map(
          lambda g: jnp.mean(g, axis=0), gradient_tree
      )

      def leaf_score(g, m):
        reduce_axes = tuple(range(1, g.ndim))
        return jnp.sum(g * m, axis=reduce_axes)

      scores = jax.tree_util.tree_map(leaf_score, gradient_tree, mean_grads)
      return jax.tree_util.tree_reduce(lambda a, b: a + b, scores)

    def _scores_group_scope(gradient_tree):
      self._validate_group_config(batch_size)
      group_size = int(self.num_generations)
      num_groups = batch_size // group_size

      def leaf_score(g):
        g2 = g.reshape((num_groups, group_size) + g.shape[1:])
        mean_g = jnp.mean(g2, axis=1)
        reduce_axes = tuple(range(2, g2.ndim))
        return jnp.sum(g2 * mean_g[:, None, ...], axis=reduce_axes)

      scores = jax.tree_util.tree_map(leaf_score, gradient_tree)
      scores = jax.tree_util.tree_reduce(lambda a, b: a + b, scores)
      return scores.reshape((batch_size,))

    if self.scope == "group":
      scores = _scores_group_scope(score_grads)
    else:
      scores = _scores_batch_scope(score_grads)

    mask = scores >= self.dot_threshold
    mask_f = mask.astype(jnp.float32)
    denom = jnp.clip(jnp.sum(mask_f), 1.0)
    num_kept = jnp.sum(mask_f)
    num_skipped = jnp.asarray(batch_size, dtype=jnp.float32) - num_kept

    def masked_mean(pytree):
      def leaf_mean(x):
        reshape = (-1,) + (1,) * (x.ndim - 1)
        return jnp.sum(x * mask_f.reshape(reshape), axis=0) / denom

      return jax.tree_util.tree_map(leaf_mean, pytree)

    final_grads = masked_mean(per_sample_grads)
    grad_norm = optax.global_norm(final_grads)
    final_loss = jnp.sum(per_sample_loss * mask_f) / denom
    optimizer.update(model, final_grads)

    if self._has_aux:
      final_aux = masked_mean(per_sample_aux)
      if isinstance(final_aux, dict):
        final_aux["skipped_samples"] = num_skipped
        final_aux["self_inf_dot_mean"] = jnp.mean(scores)
        final_aux["self_inf_dot_std"] = jnp.std(scores)
        final_aux["self_inf_kept_fraction"] = num_kept / jnp.asarray(
            batch_size, dtype=jnp.float32
        )
      return final_loss, final_aux, grad_norm

    return final_loss, None, grad_norm
