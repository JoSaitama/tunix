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

"""Policy-score LOO trainer that masks policy gradients but preserves KL."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from tunix.rl import self_inf_loo_policy_trainer


class PolicyOnlySelfInfLooTrainer(
    self_inf_loo_policy_trainer.PolicySelfInfLooTrainer
):
  """Masks selected policy gradients while retaining all-sample KL gradients."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.mask_application = "policy_only"

  def _aggregate_update_gradients(
      self,
      per_sample_grads,
      score_grads,
      mask: jax.Array,
      masked_mean: Callable[..., Any],
  ):
    selected_policy_grads = masked_mean(score_grads, mask)
    all_samples = jnp.ones_like(mask)
    all_total_grads = masked_mean(per_sample_grads, all_samples)
    all_policy_grads = masked_mean(score_grads, all_samples)
    return jax.tree_util.tree_map(
        lambda selected_policy, total, policy: (
            selected_policy + total - policy
        ),
        selected_policy_grads,
        all_total_grads,
        all_policy_grads,
    )

  def _aggregate_update_loss(
      self,
      per_sample_loss: jax.Array,
      per_sample_score_loss: jax.Array | None,
      mask: jax.Array,
  ) -> jax.Array:
    if per_sample_score_loss is None:
      raise RuntimeError(
          "PolicyOnlySelfInfLooTrainer requires per-sample policy losses."
      )
    selected_policy_loss = jnp.sum(per_sample_score_loss * mask) / jnp.clip(
        jnp.sum(mask), 1.0
    )
    return (
        selected_policy_loss
        + jnp.mean(per_sample_loss)
        - jnp.mean(per_sample_score_loss)
    )

  def _aggregate_update_aux(
      self,
      per_sample_aux,
      mask: jax.Array,
      masked_mean: Callable[..., Any],
  ):
    return masked_mean(per_sample_aux, jnp.ones_like(mask))
