# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Policy-score, full-update leave-one-out self-influence trainer."""

from __future__ import annotations

import functools
from typing import Any, Callable

from flax import nnx
import jax.numpy as jnp
import optax
from tunix.rl import memory_bounded_curation
from tunix.rl import self_inf_loo_trainer


class PolicySelfInfLooTrainer(self_inf_loo_trainer.SelfInfLooTrainer):
  """Uses policy-only gradients for LOO scores and total gradients to update."""

  def clear_jit_cache(self):
    super().clear_jit_cache()
    self._staged_policy_train_step = None

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

    update_grad_fn = memory_bounded_curation.make_masked_aggregate_grad_fn(
        self.loss_fn, wrt=wrt, has_aux=self._has_aux
    )
    out, final_grads = update_grad_fn(model, inputs, mask)
    if self._has_aux:
      final_loss, final_aux = out
    else:
      final_loss, final_aux = out, None
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

  def jit_train_and_eval_step(
      self, skip_jit: bool = False, cache_nnx_graph: bool = False
  ):
    """Builds separate reusable JITs for score, update, and optimizer apply."""
    if skip_jit:
      return super().jit_train_and_eval_step(True, cache_nnx_graph)
    if getattr(self, "_staged_policy_train_step", None) is not None:
      return self._staged_policy_train_step, self._jitted_eval_step_fn

    _, eval_step = super().jit_train_and_eval_step(False, cache_nnx_graph)
    self._jitted_eval_step_fn = eval_step
    wrt = nnx.LoRAParam if self._lora_enabled else nnx.Param

    def score_sample(model, sample):
      grad_fn = memory_bounded_curation.make_grad_fn(
          self._score_loss_fn,
          wrt=wrt,
          has_aux=self._score_loss_has_aux,
      )
      _, gradient = grad_fn(model, sample)
      return gradient

    def update_batch(model, batch_inputs, trajectory_mask):
      grad_fn = memory_bounded_curation.make_masked_aggregate_grad_fn(
          self.loss_fn, wrt=wrt, has_aux=self._has_aux
      )
      return grad_fn(model, batch_inputs, trajectory_mask)

    def apply_update(model, optimizer, gradients):
      grad_norm = optax.global_norm(gradients)
      optimizer.update(model, gradients)
      return grad_norm

    score_step = nnx.jit(score_sample)
    update_step = nnx.jit(update_batch)
    apply_step = nnx.jit(apply_update, donate_argnames=("optimizer",))
    if cache_nnx_graph:
      score_step = nnx.cached_partial(score_step, self.model)
      update_step = nnx.cached_partial(update_step, self.model)
      apply_step = nnx.cached_partial(
          apply_step, self.model, self.optimizer
      )
    else:
      score_step = functools.partial(score_step, self.model)
      update_step = functools.partial(update_step, self.model)
      apply_step = functools.partial(apply_step, self.model, self.optimizer)

    def staged_train_step(raw_inputs):
      inputs = self.gen_model_input_fn(raw_inputs)
      inputs = dict(inputs)
      inputs.pop("algo_config", None)
      stats = memory_bounded_curation.staged_dtv_statistics(
          score_step,
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

      out, final_grads = update_step(inputs, mask)
      if self._has_aux:
        final_loss, final_aux = out
      else:
        final_loss, final_aux = out, None
      grad_norm = apply_step(final_grads)
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
            "loo_retention_cap_triggered": jnp.any(cap_triggered).astype(
                jnp.float32
            ),
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

    self._staged_policy_train_step = staged_train_step
    return staged_train_step, eval_step
