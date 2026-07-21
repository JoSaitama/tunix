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

"""Leave-one-out self-influence dynamic batch curation for GRPO."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Callable, Literal, Tuple

from flax import nnx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from tunix.rl import trainer as rl_trainer


_DECISION_ARRAY_KEYS = (
    "loo_scores",
    "loo_raw_self",
    "loo_raw_cross_sum",
    "loo_standard_self_term",
    "loo_standard_cross_term",
    "loo_standard_score",
    "loo_threshold_mask",
    "loo_final_mask",
    "loo_retained_by_cap_mask",
    "loo_group_indices",
    "loo_generation_indices",
    "loo_group_cap_triggered",
)

_DECISION_SCALAR_KEYS = (
    "skipped_samples",
    "loo_pre_cap_kept_samples",
    "loo_post_cap_kept_samples",
    "loo_pre_cap_kept_fraction",
    "loo_post_cap_kept_fraction",
    "loo_retention_cap_triggered",
    "loo_groups_with_cap_triggered",
    "loo_groups_with_all_negative_scores",
    "loo_optimizer_update_applied",
    "loo_effective_update",
    "loo_nonfinite_score_count",
)


def _stable_top_k_mask(scores: jax.Array, k: int) -> jax.Array:
  """Returns a deterministic mask for the highest finite `k` scores."""
  finite = jnp.isfinite(scores)
  safe_scores = jnp.where(finite, scores, -jnp.inf)
  order = jnp.argsort(-safe_scores, stable=True)
  rank = jnp.empty_like(order).at[order].set(jnp.arange(scores.shape[0]))
  return (rank < k) & finite


def _capped_mask(scores: jax.Array, min_keep_fraction: float) -> tuple[jax.Array, ...]:
  """Applies the score threshold and a highest-score minimum-retention cap."""
  population_size = int(scores.shape[0])
  min_keep = max(1, math.ceil(population_size * min_keep_fraction))
  threshold_mask = jnp.isfinite(scores) & (scores >= 0.0)
  top_k_mask = _stable_top_k_mask(scores, min_keep)
  cap_triggered = jnp.sum(threshold_mask) < min_keep
  final_mask = jnp.where(cap_triggered, top_k_mask, threshold_mask)
  retained_by_cap = final_mask & ~threshold_mask
  return final_mask, threshold_mask, retained_by_cap, cap_triggered


def _group_capped_mask(
    grouped_scores: jax.Array, min_keep_fraction: float
) -> tuple[jax.Array, ...]:
  """Applies the retention cap independently to every prompt group."""
  return jax.vmap(
      lambda scores: _capped_mask(scores, min_keep_fraction)
  )(grouped_scores)


def _batch_loo_statistics(per_sample_grads) -> dict[str, jax.Array]:
  """Computes strict N-1 LOO scores and ordinary-DTV decomposition."""
  leaves = jax.tree_util.tree_leaves(per_sample_grads)
  batch_size = int(leaves[0].shape[0])
  if batch_size <= 1:
    raise ValueError("Batch self-influence LOO requires at least two samples.")

  def leaf_statistics(g):
    g = g.astype(jnp.float32)
    reduce_axes = tuple(range(1, g.ndim))
    total = jnp.sum(g, axis=0)
    raw_self = jnp.sum(g * g, axis=reduce_axes)
    raw_total = jnp.sum(g * total, axis=reduce_axes)
    return raw_self, raw_total - raw_self

  leaf_values = [leaf_statistics(g) for g in leaves]
  raw_self = sum((value[0] for value in leaf_values), jnp.zeros(batch_size))
  raw_cross = sum((value[1] for value in leaf_values), jnp.zeros(batch_size))
  denominator = float(batch_size)
  return {
      "raw_self": raw_self,
      "raw_cross_sum": raw_cross,
      "standard_self_term": raw_self / denominator,
      "standard_cross_term": raw_cross / denominator,
      "standard_score": (raw_self + raw_cross) / denominator,
      "loo_score": raw_cross / float(batch_size - 1),
  }


def _group_loo_statistics(
    per_sample_grads, group_size: int
) -> dict[str, jax.Array]:
  """Computes strict G-1 LOO statistics within contiguous prompt groups."""
  leaves = jax.tree_util.tree_leaves(per_sample_grads)
  batch_size = int(leaves[0].shape[0])
  if group_size <= 1:
    raise ValueError("Group self-influence LOO requires num_generations > 1.")
  if batch_size % group_size != 0:
    raise ValueError(
        "Group self-influence LOO requires batch size to be divisible by "
        f"num_generations; got batch_size={batch_size}, group_size={group_size}."
    )
  num_groups = batch_size // group_size

  def leaf_statistics(g):
    g = g.astype(jnp.float32)
    grouped = g.reshape((num_groups, group_size) + g.shape[1:])
    reduce_axes = tuple(range(2, grouped.ndim))
    group_total = jnp.sum(grouped, axis=1)
    raw_self = jnp.sum(grouped * grouped, axis=reduce_axes)
    raw_total = jnp.sum(
        grouped * group_total[:, None, ...], axis=reduce_axes
    )
    return raw_self, raw_total - raw_self

  leaf_values = [leaf_statistics(g) for g in leaves]
  zeros = jnp.zeros((num_groups, group_size))
  raw_self = sum((value[0] for value in leaf_values), zeros)
  raw_cross = sum((value[1] for value in leaf_values), zeros)
  denominator = float(group_size)

  def flatten(x):
    return x.reshape((batch_size,))

  return {
      "raw_self": flatten(raw_self),
      "raw_cross_sum": flatten(raw_cross),
      "standard_self_term": flatten(raw_self / denominator),
      "standard_cross_term": flatten(raw_cross / denominator),
      "standard_score": flatten((raw_self + raw_cross) / denominator),
      "loo_score": flatten(raw_cross / float(group_size - 1)),
  }


class SelfInfLooTrainer(rl_trainer.Trainer):
  """Filters anti-aligned samples using strict leave-one-out gradients."""

  def __init__(
      self,
      *args,
      scope: Literal["batch", "group"] = "batch",
      num_generations: int | None = None,
      min_keep_fraction: float = 0.25,
      decisions_path: str | None = None,
      **kwargs,
  ):
    super().__init__(*args, **kwargs)
    if scope not in ("batch", "group"):
      raise ValueError(f"Unsupported self-influence LOO scope: {scope!r}")
    if not 0.0 < min_keep_fraction <= 1.0:
      raise ValueError("min_keep_fraction must be in (0, 1].")
    self.scope = scope
    self.num_generations = num_generations
    self.min_keep_fraction = min_keep_fraction
    self.decisions_path = decisions_path
    self._score_loss_fn: Callable[..., Any] | None = None
    self._score_loss_has_aux = False
    self.score_objective = "total"
    if decisions_path:
      os.makedirs(os.path.dirname(os.path.abspath(decisions_path)), exist_ok=True)

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
      raise ValueError("No gradients found for self-influence LOO curation.")
    batch_size = int(leaves[0].shape[0])

    if self.scope == "group":
      group_size = int(self.num_generations or 0)
      statistics = _group_loo_statistics(score_grads, group_size)
      num_groups = batch_size // group_size
      grouped_scores = statistics["loo_score"].reshape(
          (num_groups, group_size)
      )
      masks = _group_capped_mask(grouped_scores, self.min_keep_fraction)
      mask, threshold_mask, retained_by_cap, cap_triggered = (
          value.reshape((batch_size,)) if value.ndim == 2 else value
          for value in masks
      )
      group_indices = jnp.repeat(jnp.arange(num_groups), group_size)
      generation_indices = jnp.tile(jnp.arange(group_size), num_groups)
      groups_with_cap = jnp.sum(cap_triggered.astype(jnp.float32))
      groups_all_negative = jnp.sum(
          (
              jnp.sum(
                  threshold_mask.reshape(num_groups, group_size), axis=1
              ) == 0
          ).astype(jnp.float32)
      )
      group_kept_fraction = jnp.mean(
          mask.reshape(num_groups, group_size), axis=1
      )
      min_group_kept_fraction = jnp.min(group_kept_fraction)
      max_group_kept_fraction = jnp.max(group_kept_fraction)
    else:
      statistics = _batch_loo_statistics(score_grads)
      mask, threshold_mask, retained_by_cap, cap_triggered = _capped_mask(
          statistics["loo_score"], self.min_keep_fraction
      )
      group_size = int(self.num_generations or 0)
      if group_size > 0 and batch_size % group_size == 0:
        num_groups = batch_size // group_size
        group_indices = jnp.repeat(jnp.arange(num_groups), group_size)
        generation_indices = jnp.tile(jnp.arange(group_size), num_groups)
      else:
        group_indices = jnp.full((batch_size,), -1, dtype=jnp.int32)
        generation_indices = jnp.arange(batch_size)
      groups_with_cap = cap_triggered.astype(jnp.float32)
      groups_all_negative = (jnp.sum(threshold_mask) == 0).astype(jnp.float32)
      min_group_kept_fraction = jnp.mean(mask)
      max_group_kept_fraction = jnp.mean(mask)

    scores = statistics["loo_score"]
    num_pre_cap_kept = jnp.sum(threshold_mask.astype(jnp.float32))
    num_kept = jnp.sum(mask.astype(jnp.float32))
    num_skipped = batch_size - num_kept

    def masked_mean(pytree, selection_mask):
      denom = jnp.clip(jnp.sum(selection_mask), 1.0)

      def leaf_mean(x):
        reshape = (-1,) + (1,) * (x.ndim - 1)
        return jnp.sum(x * selection_mask.reshape(reshape), axis=0) / denom

      return jax.tree_util.tree_map(leaf_mean, pytree)

    final_grads = masked_mean(per_sample_grads, mask)
    final_loss = jnp.sum(per_sample_loss * mask) / jnp.clip(jnp.sum(mask), 1.0)
    optimizer.update(model, final_grads)

    if self._has_aux:
      final_aux = masked_mean(per_sample_aux, mask)
      if isinstance(final_aux, dict):
        final_aux.update({
            "skipped_samples": num_skipped,
            "self_inf_dot_mean": jnp.mean(scores),
            "self_inf_dot_std": jnp.std(scores),
            "self_inf_kept_fraction": num_kept / float(batch_size),
            "loo_score_mean": jnp.mean(scores),
            "loo_score_std": jnp.std(scores),
            "loo_score_min": jnp.min(scores),
            "loo_score_max": jnp.max(scores),
            "loo_self_term_mean": jnp.mean(statistics["standard_self_term"]),
            "loo_self_term_std": jnp.std(statistics["standard_self_term"]),
            "loo_cross_term_mean": jnp.mean(statistics["standard_cross_term"]),
            "loo_cross_term_std": jnp.std(statistics["standard_cross_term"]),
            "loo_pre_cap_kept_samples": num_pre_cap_kept,
            "loo_post_cap_kept_samples": num_kept,
            "loo_pre_cap_kept_fraction": num_pre_cap_kept / float(batch_size),
            "loo_post_cap_kept_fraction": num_kept / float(batch_size),
            "loo_retention_cap_triggered": jnp.any(cap_triggered).astype(jnp.float32),
            "loo_groups_with_cap_triggered": groups_with_cap,
            "loo_groups_with_all_negative_scores": groups_all_negative,
            "loo_min_group_kept_fraction": min_group_kept_fraction,
            "loo_max_group_kept_fraction": max_group_kept_fraction,
            "loo_optimizer_update_applied": jnp.asarray(1.0),
            "loo_effective_update": (num_kept > 0).astype(jnp.float32),
            "loo_nonfinite_score_count": jnp.sum(~jnp.isfinite(scores)),
            "loo_scores": scores,
            "loo_raw_self": statistics["raw_self"],
            "loo_raw_cross_sum": statistics["raw_cross_sum"],
            "loo_standard_self_term": statistics["standard_self_term"],
            "loo_standard_cross_term": statistics["standard_cross_term"],
            "loo_standard_score": statistics["standard_score"],
            "loo_threshold_mask": threshold_mask,
            "loo_final_mask": mask,
            "loo_retained_by_cap_mask": retained_by_cap,
            "loo_group_indices": group_indices,
            "loo_generation_indices": generation_indices,
            "loo_group_cap_triggered": jnp.atleast_1d(cap_triggered),
        })
      return final_loss, final_aux
    return final_loss, None

  def _post_process_train_step(self, aux: Any) -> None:
    if self.decisions_path and isinstance(aux, dict):
      decision_arrays = {
          key: np.asarray(aux.pop(key)).tolist()
          for key in _DECISION_ARRAY_KEYS
          if key in aux
      }
      if decision_arrays:
        decision_scalars = {
            key: np.asarray(aux[key]).item()
            for key in _DECISION_SCALAR_KEYS
            if key in aux
        }
        record = {
            "train_step": int(self._train_steps) + 1,
            "scope": self.scope,
            "num_generations": self.num_generations,
            "min_keep_fraction": self.min_keep_fraction,
            **decision_scalars,
            **decision_arrays,
        }
        if self.score_objective != "total":
          record["score_objective"] = self.score_objective
        with open(self.decisions_path, "a", encoding="utf-8") as output:
          output.write(json.dumps(record, separators=(",", ":")) + "\n")
    super()._post_process_train_step(aux)
