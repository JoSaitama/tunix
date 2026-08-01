# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Policy-score, full-update ordinary DTV self-influence trainer."""

from __future__ import annotations

from typing import Any, Callable

from tunix.rl import self_inf_trainer


class PolicySelfInfTrainer(self_inf_trainer.SelfInfTrainer):
  """Scores ordinary DTV with policy gradients and updates total gradients."""

  def with_policy_score_loss_fn(
      self, score_loss_fn: Callable[..., Any], *, has_aux: bool = False
  ) -> None:
    """Sets the KL-free policy loss used only for DTV attribution."""
    self.clear_jit_cache()
    self._with_score_loss_fn(score_loss_fn, has_aux=has_aux)

  def _train_step(self, *args, **kwargs):
    if self._score_loss_fn is None:
      raise RuntimeError(
          "PolicySelfInfTrainer requires a policy score loss function."
      )
    return super()._train_step(*args, **kwargs)
