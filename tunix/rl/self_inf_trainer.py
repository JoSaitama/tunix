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

"""Robust RL Trainer with self-influence dynamic batch curation."""

from __future__ import annotations

from typing import Any, Callable, Literal, Tuple

from flax import nnx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from tunix.rl import trainer as rl_trainer


class SelfInfTrainer(rl_trainer.Trainer):
  """Filters samples whose gradients anti-align with the mean gradient.

  Two scopes are supported:
  - "batch": compare each sample gradient against the micro-batch mean gradient.
  - "group": reshape into GRPO groups (num_generations) and compare each sample
    against its group mean gradient.
  """

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
    """Configures an optional score-only loss without changing update loss."""
    self._score_loss_fn = score_loss_fn
    self._score_loss_has_aux = has_aux

  def _train_step(
      self, model: nnx.Module, optimizer: nnx.Optimizer, inputs: Any
  ) -> ArrayLike | Tuple[ArrayLike, Any]:
    inputs = self.gen_model_input_fn(inputs)

    def per_sample_loss_fn(model, inputs):
      if isinstance(inputs, dict):
        return self.loss_fn(model, **inputs)
      return self.loss_fn(model, inputs)

    wrt = nnx.LoRAParam if self._lora_enabled else nnx.Param
    grad_fn = nnx.value_and_grad(
        per_sample_loss_fn,
        argnums=nnx.DiffState(0, wrt),
        has_aux=self._has_aux,
    )

    inputs_in_axes = jax.tree_util.tree_map(lambda _: 0, inputs)
    vmapped_grad_fn = jax.vmap(grad_fn, in_axes=(None, inputs_in_axes))
    (per_sample_loss, per_sample_aux), per_sample_grads = vmapped_grad_fn(
        model, inputs
    )

    score_grads = per_sample_grads
    if self._score_loss_fn is not None:
      def per_sample_score_loss_fn(model, inputs):
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

    def _scores_batch_scope(per_sample_grads):
      mean_grads = jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), per_sample_grads)

      def leaf_score(g, m):
        reduce_axes = tuple(range(1, g.ndim))
        return jnp.sum(g * m, axis=reduce_axes)

      scores = jax.tree_util.tree_map(leaf_score, per_sample_grads, mean_grads)
      return jax.tree_util.tree_reduce(lambda a, b: a + b, scores)

    def _scores_group_scope(per_sample_grads):
      group_size = int(self.num_generations or 0)
      if group_size <= 0 or batch_size % group_size != 0:
        return _scores_batch_scope(per_sample_grads)
      num_groups = batch_size // group_size

      def leaf_score(g):
        g2 = g.reshape((num_groups, group_size) + g.shape[1:])
        m = jnp.mean(g2, axis=1)
        reduce_axes = tuple(range(2, g2.ndim))
        return jnp.sum(g2 * m[:, None, ...], axis=reduce_axes)

      scores = jax.tree_util.tree_map(leaf_score, per_sample_grads)
      scores = jax.tree_util.tree_reduce(lambda a, b: a + b, scores)
      return scores.reshape((batch_size,))

    if self.scope == "group":
      scores = _scores_group_scope(score_grads)
    else:
      scores = _scores_batch_scope(score_grads)

    mask = scores >= self.dot_threshold
    num_kept = jnp.sum(mask.astype(jnp.float32))
    num_skipped = batch_size - num_kept

    def masked_mean(pytree, mask):
      denom = jnp.clip(jnp.sum(mask), 1.0)

      def leaf_mean(x):
        reshape = (-1,) + (1,) * (x.ndim - 1)
        return jnp.sum(x * mask.reshape(reshape), axis=0) / denom

      return jax.tree_util.tree_map(leaf_mean, pytree)

    final_grads = masked_mean(per_sample_grads, mask)
    final_loss = jnp.sum(per_sample_loss * mask) / jnp.clip(
        jnp.sum(mask), 1.0
    )
    optimizer.update(model, final_grads)

    if self._has_aux:
      final_aux = masked_mean(per_sample_aux, mask)
      if isinstance(final_aux, dict):
        final_aux["skipped_samples"] = num_skipped
        final_aux["self_inf_dot_mean"] = jnp.mean(scores)
        final_aux["self_inf_dot_std"] = jnp.std(scores)
        final_aux["self_inf_kept_fraction"] = num_kept / jnp.clip(batch_size, 1.0)
      return final_loss, final_aux

    return final_loss, None
