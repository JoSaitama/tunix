# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Policy-score, full-update ordinary DTV self-influence trainer."""

from __future__ import annotations

import functools
from typing import Any, Callable

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tunix.rl import memory_bounded_curation
from tunix.rl import self_inf_trainer


class PolicySelfInfTrainer(self_inf_trainer.SelfInfTrainer):
  """Scores ordinary DTV with policy gradients and updates total gradients."""

  def clear_jit_cache(self):
    super().clear_jit_cache()
    self._staged_policy_train_step = None

  def with_policy_score_loss_fn(
      self, score_loss_fn: Callable[..., Any], *, has_aux: bool = False
  ) -> None:
    """Sets the KL-free policy loss used only for DTV attribution."""
    self.clear_jit_cache()
    self._with_score_loss_fn(score_loss_fn, has_aux=has_aux)

  def _train_step(self, model, optimizer, inputs):
    if self._score_loss_fn is None:
      raise RuntimeError(
          "PolicySelfInfTrainer requires a policy score loss function."
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
    scores = stats["standard_score"]
    mask = scores >= self.dot_threshold

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
      kept = jnp.sum(mask_f)
      batch_size = memory_bounded_curation.batch_size(inputs)
      final_aux.update({
          "skipped_samples": jnp.asarray(batch_size) - kept,
          "self_inf_dot_mean": jnp.mean(scores),
          "self_inf_dot_std": jnp.std(scores),
          "self_inf_kept_fraction": kept / float(batch_size),
      })
    return final_loss, final_aux, grad_norm

  def jit_train_and_eval_step(
      self, skip_jit: bool = False, cache_nnx_graph: bool = False
  ):
    """Builds bounded single-sample JITs with Python orchestration."""
    if skip_jit:
      return super().jit_train_and_eval_step(True, cache_nnx_graph)
    if getattr(self, "_staged_policy_train_step", None) is not None:
      return self._staged_policy_train_step, self._jitted_eval_step_fn

    # Let the base class shard optimizer state and construct the normal eval
    # step. Merely wrapping _train_step does not compile or execute it.
    _, eval_step = super().jit_train_and_eval_step(False, cache_nnx_graph)
    self._jitted_eval_step_fn = eval_step
    wrt = nnx.LoRAParam if self._lora_enabled else nnx.Param

    def score_batch(model, batch_inputs):
      if self.scope == "group":
        size = memory_bounded_curation.batch_size(batch_inputs)
        group_size = self.num_generations
        if group_size is None or group_size <= 1 or size % group_size:
          raise ValueError(
              "Group Policy-DTV requires complete generation groups."
          )

        def sample_loss(model, sample_inputs):
          sample_inputs = jax.tree_util.tree_map(
              lambda x: jnp.expand_dims(x, axis=0)
              if isinstance(x, (jax.Array, np.ndarray))
              else x,
              sample_inputs,
          )
          out = self._score_loss_fn(model, **sample_inputs)
          return out[0] if self._score_loss_has_aux else out

        def input_axis(x):
          return 0 if isinstance(x, (jax.Array, np.ndarray)) else None

        def group_losses(model, group_inputs):
          axes = jax.tree_util.tree_map(input_axis, group_inputs)
          return jax.vmap(sample_loss, in_axes=(None, axes))(
              model, group_inputs
          )

        def mean_group_loss(model, group_inputs):
          # A fixed denominator preserves zero-gradient degenerate rows in the
          # exact DTV group mean rather than renormalizing over active rows.
          return jnp.sum(group_losses(model, group_inputs)) / float(group_size)

        mean_grad_fn = nnx.grad(
            mean_group_loss, argnums=nnx.DiffState(0, wrt)
        )
        scores = []
        for group_index in range(size // group_size):
          start = group_index * group_size
          group_inputs = jax.tree_util.tree_map(
              lambda x: jax.lax.dynamic_slice_in_dim(
                  x, start, group_size, axis=0
              )
              if isinstance(x, (jax.Array, np.ndarray))
              else x,
              batch_inputs,
          )
          mean_gradient = mean_grad_fn(model, group_inputs)
          mean_gradient = jax.tree_util.tree_map(
              jax.lax.stop_gradient, mean_gradient
          )
          _, group_scores = nnx.jvp(
              lambda current_model: group_losses(
                  current_model, group_inputs
              ),
              (model,),
              (mean_gradient,),
          )
          scores.append(group_scores)
        standard_score = jnp.concatenate(scores, axis=0)
        return {"standard_score": standard_score}

      grad_fn = memory_bounded_curation.make_grad_fn(
          self._score_loss_fn,
          wrt=wrt,
          has_aux=self._score_loss_has_aux,
      )

      def score_sample(model, sample_inputs):
        sample_inputs = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0)
            if isinstance(x, (jax.Array, np.ndarray))
            else x,
            sample_inputs,
        )
        return grad_fn(model, sample_inputs)

      def input_axis(x):
        return 0 if isinstance(x, (jax.Array, np.ndarray)) else None

      input_axes = jax.tree_util.tree_map(input_axis, batch_inputs)
      _, gradients = jax.vmap(
          score_sample, in_axes=(None, input_axes)
      )(model, batch_inputs)
      return memory_bounded_curation.statistics_from_gradient_tree(
          gradients,
          scope=self.scope,
          group_size=self.num_generations,
      )

    def update_batch(model, batch_inputs, trajectory_mask):
      grad_fn = memory_bounded_curation.make_masked_aggregate_grad_fn(
          self.loss_fn, wrt=wrt, has_aux=self._has_aux
      )
      return grad_fn(model, batch_inputs, trajectory_mask)

    def apply_update(model, optimizer, gradients):
      grad_norm = optax.global_norm(gradients)
      optimizer.update(model, gradients)
      return grad_norm

    score_step = nnx.jit(score_batch)
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
      # Agentic GRPO loss closures already capture their full and policy-only
      # configs. Do not send the non-array config object through a sample JIT.
      inputs = dict(inputs)
      inputs.pop("algo_config", None)
      stats = score_step(inputs)
      scores = stats["standard_score"]
      mask = scores >= self.dot_threshold
      out, final_grads = update_step(inputs, mask)
      if self._has_aux:
        final_loss, final_aux = out
      else:
        final_loss, final_aux = out, None
      grad_norm = apply_step(final_grads)
      if self._has_aux and isinstance(final_aux, dict):
        mask_f = mask.astype(jnp.float32)
        kept = jnp.sum(mask_f)
        size = memory_bounded_curation.batch_size(inputs)
        final_aux.update({
            "skipped_samples": jnp.asarray(size) - kept,
            "self_inf_dot_mean": jnp.mean(scores),
            "self_inf_dot_std": jnp.std(scores),
            "self_inf_kept_fraction": kept / float(size),
        })
      return final_loss, final_aux, grad_norm

    self._staged_policy_train_step = staged_train_step
    return staged_train_step, eval_step
