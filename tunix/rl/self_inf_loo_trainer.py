# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Leave-one-out self-influence dynamic batch curation for Agentic GRPO."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Callable, Literal

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
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


def _stable_top_k_mask(scores: jax.Array, k: int) -> jax.Array:
  finite = jnp.isfinite(scores)
  safe_scores = jnp.where(finite, scores, -jnp.inf)
  order = jnp.argsort(-safe_scores, stable=True)
  rank = jnp.empty_like(order).at[order].set(jnp.arange(scores.shape[0]))
  return (rank < k) & finite


def _capped_mask(
    scores: jax.Array,
    min_keep_fraction: float,
    dot_threshold: float = 0.0,
) -> tuple[jax.Array, ...]:
  population_size = int(scores.shape[0])
  min_keep = max(1, math.ceil(population_size * min_keep_fraction))
  threshold_mask = jnp.isfinite(scores) & (scores >= dot_threshold)
  top_k_mask = _stable_top_k_mask(scores, min_keep)
  cap_triggered = jnp.sum(threshold_mask) < min_keep
  final_mask = jnp.where(cap_triggered, top_k_mask, threshold_mask)
  retained_by_cap = final_mask & ~threshold_mask
  return final_mask, threshold_mask, retained_by_cap, cap_triggered


def _group_capped_mask(
    grouped_scores: jax.Array,
    min_keep_fraction: float,
    dot_threshold: float = 0.0,
) -> tuple[jax.Array, ...]:
  return jax.vmap(
      lambda x: _capped_mask(x, min_keep_fraction, dot_threshold)
  )(grouped_scores)


def _batch_loo_statistics(per_sample_grads) -> dict[str, jax.Array]:
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

  values = [leaf_statistics(g) for g in leaves]
  raw_self = sum((x[0] for x in values), jnp.zeros(batch_size))
  raw_cross = sum((x[1] for x in values), jnp.zeros(batch_size))
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
  leaves = jax.tree_util.tree_leaves(per_sample_grads)
  batch_size = int(leaves[0].shape[0])
  if group_size <= 1:
    raise ValueError("Group self-influence LOO requires num_generations > 1.")
  if batch_size % group_size:
    raise ValueError(
        "Group self-influence LOO requires batch size divisible by "
        f"num_generations; got {batch_size=} and {group_size=}."
    )
  num_groups = batch_size // group_size

  def leaf_statistics(g):
    g = g.astype(jnp.float32)
    grouped = g.reshape((num_groups, group_size) + g.shape[1:])
    reduce_axes = tuple(range(2, grouped.ndim))
    group_total = jnp.sum(grouped, axis=1)
    raw_self = jnp.sum(grouped * grouped, axis=reduce_axes)
    raw_total = jnp.sum(grouped * group_total[:, None, ...], axis=reduce_axes)
    return raw_self, raw_total - raw_self

  values = [leaf_statistics(g) for g in leaves]
  zeros = jnp.zeros((num_groups, group_size))
  raw_self = sum((x[0] for x in values), zeros)
  raw_cross = sum((x[1] for x in values), zeros)
  denominator = float(group_size)
  flatten = lambda x: x.reshape((batch_size,))
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
      dot_threshold: float = 0.0,
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
    self.dot_threshold = dot_threshold
    self.decisions_path = decisions_path
    self._score_loss_fn: Callable[..., Any] | None = None
    self._score_loss_has_aux = False
    self.score_objective = "total"
    if decisions_path:
      os.makedirs(os.path.dirname(os.path.abspath(decisions_path)), exist_ok=True)

  def _with_score_loss_fn(
      self, score_loss_fn: Callable[..., Any], *, has_aux: bool
  ) -> None:
    self._score_loss_fn = score_loss_fn
    self._score_loss_has_aux = has_aux

  def _train_step(self, model, optimizer, inputs):
    inputs = self.gen_model_input_fn(inputs)

    def restore_batch_dim(x):
      if isinstance(x, (jax.Array, np.ndarray)):
        return jnp.expand_dims(x, axis=0)
      return x

    def call_loss(loss_fn, model, sample_inputs):
      sample_inputs = jax.tree_util.tree_map(restore_batch_dim, sample_inputs)
      return loss_fn(model, **sample_inputs)

    wrt = nnx.LoRAParam if self._lora_enabled else nnx.Param
    in_axes = jax.tree_util.tree_map(
        lambda x: 0 if isinstance(x, (jax.Array, np.ndarray)) else None, inputs
    )
    grad_fn = nnx.value_and_grad(
        lambda m, x: call_loss(self.loss_fn, m, x),
        argnums=nnx.DiffState(0, wrt),
        has_aux=self._has_aux,
    )
    out, per_sample_grads = jax.vmap(
        grad_fn, in_axes=(None, in_axes)
    )(model, inputs)
    if self._has_aux:
      per_sample_loss, per_sample_aux = out
    else:
      per_sample_loss, per_sample_aux = out, None

    score_grads = per_sample_grads
    if self._score_loss_fn is not None:
      score_grad_fn = nnx.value_and_grad(
          lambda m, x: call_loss(self._score_loss_fn, m, x),
          argnums=nnx.DiffState(0, wrt),
          has_aux=self._score_loss_has_aux,
      )
      score_out, score_grads = jax.vmap(
          score_grad_fn, in_axes=(None, in_axes)
      )(model, inputs)
      del score_out

    leaves = jax.tree_util.tree_leaves(per_sample_grads)
    if not leaves:
      raise ValueError("No gradients found for self-influence LOO curation.")
    batch_size = int(leaves[0].shape[0])

    if self.scope == "group":
      group_size = int(self.num_generations or 0)
      stats = _group_loo_statistics(score_grads, group_size)
      num_groups = batch_size // group_size
      masks = _group_capped_mask(
          stats["loo_score"].reshape(num_groups, group_size),
          self.min_keep_fraction,
          self.dot_threshold,
      )
      mask, threshold_mask, retained_by_cap, cap_triggered = (
          x.reshape((batch_size,)) if x.ndim == 2 else x for x in masks
      )
      group_indices = jnp.repeat(jnp.arange(num_groups), group_size)
      generation_indices = jnp.tile(jnp.arange(group_size), num_groups)
      group_kept = jnp.mean(mask.reshape(num_groups, group_size), axis=1)
      groups_with_cap = jnp.sum(cap_triggered.astype(jnp.float32))
      groups_all_negative = jnp.sum(
          (
              jnp.sum(
                  threshold_mask.reshape(num_groups, group_size), axis=1
              )
              == 0
          ).astype(jnp.float32)
      )
    else:
      stats = _batch_loo_statistics(score_grads)
      mask, threshold_mask, retained_by_cap, cap_triggered = _capped_mask(
          stats["loo_score"], self.min_keep_fraction, self.dot_threshold
      )
      group_size = int(self.num_generations or 0)
      if group_size > 0 and batch_size % group_size == 0:
        num_groups = batch_size // group_size
        group_indices = jnp.repeat(jnp.arange(num_groups), group_size)
        generation_indices = jnp.tile(jnp.arange(group_size), num_groups)
      else:
        group_indices = jnp.full((batch_size,), -1, dtype=jnp.int32)
        generation_indices = jnp.arange(batch_size)
      group_kept = jnp.asarray([jnp.mean(mask)])
      groups_with_cap = cap_triggered.astype(jnp.float32)
      groups_all_negative = (jnp.sum(threshold_mask) == 0).astype(jnp.float32)

    mask_f = mask.astype(jnp.float32)
    denom = jnp.clip(jnp.sum(mask_f), 1.0)

    def masked_mean(tree):
      def leaf_mean(x):
        shape = (-1,) + (1,) * (x.ndim - 1)
        return jnp.sum(x * mask_f.reshape(shape), axis=0) / denom
      return jax.tree_util.tree_map(leaf_mean, tree)

    final_grads = masked_mean(per_sample_grads)
    grad_norm = optax.global_norm(final_grads)
    final_loss = jnp.sum(per_sample_loss * mask_f) / denom
    optimizer.update(model, final_grads)

    if not self._has_aux:
      return final_loss, None, grad_norm
    final_aux = masked_mean(per_sample_aux)
    if isinstance(final_aux, dict):
      num_kept = jnp.sum(mask_f)
      pre_cap = jnp.sum(threshold_mask.astype(jnp.float32))
      scores = stats["loo_score"]
      final_aux.update({
          "skipped_samples": batch_size - num_kept,
          "self_inf_dot_mean": jnp.mean(scores),
          "self_inf_dot_std": jnp.std(scores),
          "self_inf_kept_fraction": num_kept / float(batch_size),
          "loo_score_mean": jnp.mean(scores),
          "loo_score_std": jnp.std(scores),
          "loo_score_min": jnp.min(scores),
          "loo_score_max": jnp.max(scores),
          "loo_pre_cap_kept_samples": pre_cap,
          "loo_post_cap_kept_samples": num_kept,
          "loo_pre_cap_kept_fraction": pre_cap / float(batch_size),
          "loo_post_cap_kept_fraction": num_kept / float(batch_size),
          "loo_retention_cap_triggered": jnp.any(cap_triggered).astype(jnp.float32),
          "loo_groups_with_cap_triggered": groups_with_cap,
          "loo_groups_with_all_negative_scores": groups_all_negative,
          "loo_min_group_kept_fraction": jnp.min(group_kept),
          "loo_max_group_kept_fraction": jnp.max(group_kept),
          "loo_nonfinite_score_count": jnp.sum(~jnp.isfinite(scores)),
          "loo_scores": scores,
          "loo_raw_self": stats["raw_self"],
          "loo_raw_cross_sum": stats["raw_cross_sum"],
          "loo_standard_self_term": stats["standard_self_term"],
          "loo_standard_cross_term": stats["standard_cross_term"],
          "loo_standard_score": stats["standard_score"],
          "loo_threshold_mask": threshold_mask,
          "loo_final_mask": mask,
          "loo_retained_by_cap_mask": retained_by_cap,
          "loo_group_indices": group_indices,
          "loo_generation_indices": generation_indices,
          "loo_group_cap_triggered": jnp.atleast_1d(cap_triggered),
      })
    return final_loss, final_aux, grad_norm

  def _post_process_train_step(self, aux: Any) -> None:
    if self.decisions_path and isinstance(aux, dict):
      arrays = {
          key: np.asarray(aux.pop(key)).tolist()
          for key in _DECISION_ARRAY_KEYS
          if key in aux
      }
      if arrays:
        scalars = {
            key: np.asarray(value).item()
            for key, value in aux.items()
            if key.startswith("loo_") and np.asarray(value).ndim == 0
        }
        record = {
            "train_step": int(self._train_steps) + 1,
            "scope": self.scope,
            "num_generations": self.num_generations,
            "min_keep_fraction": self.min_keep_fraction,
            "dot_threshold": self.dot_threshold,
            "score_objective": self.score_objective,
            **scalars,
            **arrays,
        }
        with open(self.decisions_path, "a", encoding="utf-8") as output:
          output.write(json.dumps(record, separators=(",", ":")) + "\n")
    super()._post_process_train_step(aux)
