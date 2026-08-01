# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Policy-score, full-update leave-one-out self-influence trainer."""

from __future__ import annotations

from typing import Any, Callable

from flax import nnx
import jax.numpy as jnp
import optax
from tunix.rl import memory_bounded_curation
from tunix.rl import self_inf_loo_trainer


class PolicySelfInfLooTrainer(self_inf_loo_trainer.SelfInfLooTrainer):
  """Uses policy-only gradients for LOO scores and total gradients to update."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.score_objective = "policy"

  def with_policy_score_loss_fn(
      self, score_loss_fn: Callable[..., Any], *, has_aux: bool = False
  ) -> None:
    self.clear_jit_cache()
    self._with_score_loss_fn(score_loss_fn, has_aux=has_aux)

  def _train_step(self, model, optimizer, inputs):
    if self._score_loss_fn is None:
      raise RuntimeError(
          "PolicySelfInfLooTrainer requires a policy score loss function."
      )
    inputs = self.gen_model_input_fn(inputs)
    wrt = nnx.LoRAParam if self._lora_enabled else nnx.Param
    score_grad_fn = memory_bounded_curation.make_grad_fn(
        self._score_loss_fn, wrt=wrt, has_aux=self._score_loss_has_aux
    )
    stats = memory_bounded_curation.dtv_statistics(
        score_grad_fn,
        model,
        inputs,
        scope=self.scope,
        group_size=self.num_generations,
    )
    batch_size = memory_bounded_curation.batch_size(inputs)
    scores = stats["loo_score"]

    if self.scope == "group":
      group_size = int(self.num_generations or 0)
      num_groups = batch_size // group_size
      masks = self_inf_loo_trainer._group_capped_mask(
          scores.reshape(num_groups, group_size),
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
          (jnp.sum(
              threshold_mask.reshape(num_groups, group_size), axis=1
          ) == 0).astype(jnp.float32)
      )
    else:
      mask, threshold_mask, retained_by_cap, cap_triggered = (
          self_inf_loo_trainer._capped_mask(
              scores, self.min_keep_fraction, self.dot_threshold
          )
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

    update_grad_fn = memory_bounded_curation.make_grad_fn(
        self.loss_fn, wrt=wrt, has_aux=self._has_aux
    )
    final_loss, final_aux, final_grads = (
        memory_bounded_curation.masked_value_and_grad(
            update_grad_fn,
            model,
            inputs,
            mask,
            has_aux=self._has_aux,
        )
    )
    grad_norm = optax.global_norm(final_grads)
    optimizer.update(model, final_grads)

    if self._has_aux and isinstance(final_aux, dict):
      mask_f = mask.astype(jnp.float32)
      num_kept = jnp.sum(mask_f)
      pre_cap = jnp.sum(threshold_mask.astype(jnp.float32))
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
