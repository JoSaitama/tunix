# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Policy-score, full-update ordinary DTV self-influence trainer."""

from __future__ import annotations

from typing import Any, Callable

from flax import nnx
import jax.numpy as jnp
import optax
from tunix.rl import memory_bounded_curation
from tunix.rl import self_inf_trainer


class PolicySelfInfTrainer(self_inf_trainer.SelfInfTrainer):
  """Scores ordinary DTV with policy gradients and updates total gradients."""

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
      kept = jnp.sum(mask_f)
      batch_size = memory_bounded_curation.batch_size(inputs)
      final_aux.update({
          "skipped_samples": jnp.asarray(batch_size) - kept,
          "self_inf_dot_mean": jnp.mean(scores),
          "self_inf_dot_std": jnp.std(scores),
          "self_inf_kept_fraction": kept / float(batch_size),
      })
    return final_loss, final_aux, grad_norm
