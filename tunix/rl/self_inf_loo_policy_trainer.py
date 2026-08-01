# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Policy-score, full-update leave-one-out self-influence trainer."""

from __future__ import annotations

from typing import Any, Callable

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

  def _train_step(self, *args, **kwargs):
    if self._score_loss_fn is None:
      raise RuntimeError(
          "PolicySelfInfLooTrainer requires a policy score loss function."
      )
    return super()._train_step(*args, **kwargs)
