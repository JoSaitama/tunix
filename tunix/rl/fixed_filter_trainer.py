"""Fixed-ratio random and advantage filtering for Agentic GRPO updates."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Literal

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tunix.rl import trainer as rl_trainer


def _rank(values: jax.Array, tie_breaks: jax.Array) -> jax.Array:
  lower = values[None, :] < values[:, None]
  tied_before = (values[None, :] == values[:, None]) & (
      tie_breaks[None, :] < tie_breaks[:, None]
  )
  return jnp.sum(lower | tied_before, axis=1)


def _selection_mask(quality, tie_breaks, quota_uniform, ratio, method):
  population = int(quality.shape[0])
  target = ratio * population
  base = math.floor(target)
  count = base + (quota_uniform < target - base).astype(jnp.int32)
  ranking_values = tie_breaks if method == "random" else quality
  return _rank(ranking_values, tie_breaks) >= count, count


class FixedFilterTrainer(rl_trainer.Trainer):
  """Applies the GSM8K random/reward fixed-filter Full mask."""

  def __init__(
      self,
      *args,
      method: Literal["random", "reward"],
      scope: Literal["batch", "group"],
      filter_ratio: float,
      num_generations: int,
      decisions_path: str | None = None,
      **kwargs,
  ):
    super().__init__(*args, **kwargs)
    if method not in ("random", "reward") or scope not in ("batch", "group"):
      raise ValueError("fixed filter method/scope is invalid.")
    if not 0.0 <= filter_ratio < 1.0:
      raise ValueError("filter_ratio must be in [0, 1).")
    self.method = method
    self.scope = scope
    self.filter_ratio = filter_ratio
    self.num_generations = num_generations
    self.decisions_path = decisions_path
    if decisions_path:
      os.makedirs(os.path.dirname(os.path.abspath(decisions_path)), exist_ok=True)

  def _train_step(self, model, optimizer, inputs):
    inputs = self.gen_model_input_fn(inputs)
    example = inputs["train_example"]
    quality = jnp.asarray(example.advantages)
    if example.filter_random_values is None:
      raise ValueError("Fixed filtering requires filter_random_values.")
    random_values = jnp.asarray(example.filter_random_values)

    def restore(x):
      return jnp.expand_dims(x, 0) if isinstance(x, (jax.Array, np.ndarray)) else x

    def sample_loss(model, sample):
      sample = jax.tree_util.tree_map(restore, sample)
      return self.loss_fn(model, **sample)

    axes = jax.tree_util.tree_map(
        lambda x: 0 if isinstance(x, (jax.Array, np.ndarray)) else None, inputs
    )
    wrt = nnx.LoRAParam if self._lora_enabled else nnx.Param
    grad_fn = nnx.value_and_grad(
        sample_loss, argnums=nnx.DiffState(0, wrt), has_aux=self._has_aux
    )
    out, grads = jax.vmap(grad_fn, in_axes=(None, axes))(model, inputs)
    losses, per_sample_aux = out if self._has_aux else (out, None)
    batch_size = int(jax.tree_util.tree_leaves(grads)[0].shape[0])
    ties, quotas = random_values[:, 0], random_values[:, 1]
    if self.scope == "group":
      group_size = self.num_generations
      if group_size <= 0 or batch_size % group_size:
        raise ValueError("group fixed filter requires complete prompt groups.")
      num_groups = batch_size // group_size
      masks, counts = jax.vmap(
          lambda q, t, u: _selection_mask(
              q, t, u, self.filter_ratio, self.method
          )
      )(
          quality.reshape(num_groups, group_size),
          ties.reshape(num_groups, group_size),
          quotas.reshape(num_groups, group_size)[:, 0],
      )
      mask, target_count = masks.reshape(batch_size), jnp.sum(counts)
    else:
      mask, target_count = _selection_mask(
          quality, ties, quotas[0], self.filter_ratio, self.method
      )
    mask_f = mask.astype(jnp.float32)
    kept = jnp.sum(mask_f)
    denom = jnp.clip(kept, 1.0)

    def masked_mean(tree):
      return jax.tree_util.tree_map(
          lambda x: jnp.sum(
              x * mask_f.reshape((-1,) + (1,) * (x.ndim - 1)), axis=0
          ) / denom,
          tree,
      )

    final_grads = masked_mean(grads)
    grad_norm = optax.global_norm(final_grads)
    final_loss = jnp.sum(losses * mask_f) / denom
    optimizer.update(model, final_grads)
    if not self._has_aux:
      return final_loss, None, grad_norm
    aux = masked_mean(per_sample_aux)
    if isinstance(aux, dict):
      aux.update({
          "skipped_samples": batch_size - kept,
          "fixed_filter_kept_fraction": kept / float(batch_size),
          "fixed_filter_target_count": target_count,
          "fixed_filter_quality": quality,
          "fixed_filter_mask": mask,
      })
    return final_loss, aux, grad_norm

  def _post_process_train_step(self, aux: Any) -> None:
    if self.decisions_path and isinstance(aux, dict) and "fixed_filter_mask" in aux:
      quality = np.asarray(aux.pop("fixed_filter_quality")).tolist()
      mask = np.asarray(aux.pop("fixed_filter_mask")).tolist()
      record = {
          "train_step": int(self._train_steps) + 1,
          "method": self.method,
          "scope": self.scope,
          "filter_ratio": self.filter_ratio,
          "num_generations": self.num_generations,
          "quality": quality,
          "mask": mask,
      }
      with open(self.decisions_path, "a", encoding="utf-8") as output:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")
    super()._post_process_train_step(aux)
