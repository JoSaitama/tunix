# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Fixed-ratio random and reward filtering for GRPO actor updates."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Literal, Tuple

from flax import nnx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from tunix.rl import trainer as rl_trainer


def _rank(values: jax.Array, tie_breaks: jax.Array) -> jax.Array:
  """Returns zero-based ascending ranks with reproducible tie breaking."""
  lower = values[None, :] < values[:, None]
  tied_before = (values[None, :] == values[:, None]) & (
      tie_breaks[None, :] < tie_breaks[:, None]
  )
  return jnp.sum(lower | tied_before, axis=1)


def _selection_mask(
    quality: jax.Array,
    tie_breaks: jax.Array,
    quota_uniform: jax.Array,
    ratio: float,
    method: Literal["random", "reward"],
) -> tuple[jax.Array, jax.Array]:
  """Filters a stochastically rounded fixed ratio from one population."""
  population = int(quality.shape[0])
  target = ratio * population
  base = math.floor(target)
  filter_count = base + (quota_uniform < (target - base)).astype(jnp.int32)
  ranking_values = tie_breaks if method == "random" else quality
  ranks = _rank(ranking_values, tie_breaks)
  return ranks >= filter_count, filter_count


class FixedFilterTrainer(rl_trainer.Trainer):
  """Applies a random or advantage-based fixed-ratio Full mask."""

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
    if method not in ("random", "reward"):
      raise ValueError(f"method must be random or reward; got {method!r}.")
    if scope not in ("batch", "group"):
      raise ValueError(f"scope must be batch or group; got {scope!r}.")
    if not 0.0 <= filter_ratio < 1.0:
      raise ValueError(f"filter_ratio must be in [0, 1); got {filter_ratio}.")
    self.method = method
    self.scope = scope
    self.filter_ratio = filter_ratio
    self.num_generations = num_generations
    self.decisions_path = decisions_path
    if decisions_path:
      os.makedirs(os.path.dirname(os.path.abspath(decisions_path)), exist_ok=True)

  def _train_step(
      self, model: nnx.Module, optimizer: nnx.Optimizer, inputs: Any
  ) -> ArrayLike | Tuple[ArrayLike, Any]:
    inputs = self.gen_model_input_fn(inputs)
    train_example = inputs["train_example"]
    quality = jnp.asarray(train_example.advantages)
    if train_example.filter_random_values is None:
      raise ValueError("Fixed filtering requires filter_random_values.")
    random_values = jnp.asarray(train_example.filter_random_values)

    def per_sample_loss_fn(model, inputs):
      return self.loss_fn(model, **inputs)

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

    leaves = jax.tree_util.tree_leaves(per_sample_grads)
    if not leaves:
      raise ValueError("No gradients found for fixed-ratio filtering.")
    batch_size = int(leaves[0].shape[0])
    tie_breaks = random_values[:, 0]
    quota_uniforms = random_values[:, 1]

    if self.scope == "group":
      group_size = int(self.num_generations)
      if group_size <= 0 or batch_size % group_size != 0:
        raise ValueError(
            "Group fixed filtering requires batch size divisible by "
            f"num_generations; got {batch_size=} and {group_size=}."
        )
      num_groups = batch_size // group_size
      grouped_quality = quality.reshape(num_groups, group_size)
      grouped_ties = tie_breaks.reshape(num_groups, group_size)
      grouped_quota = quota_uniforms.reshape(num_groups, group_size)[:, 0]
      masks, counts = jax.vmap(
          lambda q, t, u: _selection_mask(
              q, t, u, self.filter_ratio, self.method
          )
      )(grouped_quality, grouped_ties, grouped_quota)
      mask = masks.reshape(batch_size)
      target_filter_count = jnp.sum(counts)
      group_indices = jnp.repeat(jnp.arange(num_groups), group_size)
      generation_indices = jnp.tile(jnp.arange(group_size), num_groups)
    else:
      mask, target_filter_count = _selection_mask(
          quality,
          tie_breaks,
          quota_uniforms[0],
          self.filter_ratio,
          self.method,
      )
      group_size = int(self.num_generations)
      num_groups = batch_size // group_size
      group_indices = jnp.repeat(jnp.arange(num_groups), group_size)
      generation_indices = jnp.tile(jnp.arange(group_size), num_groups)

    num_kept = jnp.sum(mask.astype(jnp.float32))
    num_skipped = batch_size - num_kept

    def masked_mean(pytree):
      denominator = jnp.clip(num_kept, 1.0)

      def leaf_mean(value):
        shape = (-1,) + (1,) * (value.ndim - 1)
        return jnp.sum(value * mask.reshape(shape), axis=0) / denominator

      return jax.tree_util.tree_map(leaf_mean, pytree)

    final_grads = masked_mean(per_sample_grads)
    final_loss = jnp.sum(per_sample_loss * mask) / jnp.clip(num_kept, 1.0)
    optimizer.update(model, final_grads)

    if self._has_aux:
      final_aux = masked_mean(per_sample_aux)
      if isinstance(final_aux, dict):
        final_aux.update({
            "skipped_samples": num_skipped,
            "fixed_filter_kept_fraction": num_kept / float(batch_size),
            "fixed_filter_target_count": target_filter_count,
            "fixed_filter_quality": quality,
            "fixed_filter_mask": mask,
            "fixed_filter_group_indices": group_indices,
            "fixed_filter_generation_indices": generation_indices,
        })
      return final_loss, final_aux
    return final_loss, None

  def _post_process_train_step(self, aux: Any) -> None:
    if self.decisions_path and isinstance(aux, dict):
      array_keys = (
          "fixed_filter_quality",
          "fixed_filter_mask",
          "fixed_filter_group_indices",
          "fixed_filter_generation_indices",
      )
      arrays = {
          key: np.asarray(aux.pop(key)).tolist()
          for key in array_keys
          if key in aux
      }
      if arrays:
        record = {
            "train_step": int(self._train_steps) + 1,
            "method": self.method,
            "scope": self.scope,
            "filter_ratio": self.filter_ratio,
            "num_generations": self.num_generations,
            "actual_filtered": int(np.asarray(aux["skipped_samples"]).item()),
            "target_filtered": int(
                np.asarray(aux["fixed_filter_target_count"]).item()
            ),
            "actual_kept_fraction": float(
                np.asarray(aux["fixed_filter_kept_fraction"]).item()
            ),
            **arrays,
        }
        with open(self.decisions_path, "a", encoding="utf-8") as output:
          output.write(json.dumps(record, separators=(",", ":")) + "\n")
    super()._post_process_train_step(aux)
