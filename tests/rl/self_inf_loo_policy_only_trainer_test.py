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

"""Tests for policy-score, policy-only-mask strict LOO configuration."""

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp
import numpy as np

from tunix.rl import self_inf_loo_policy_only_trainer


class PolicyOnlySelfInfLooTrainerTest(absltest.TestCase):

  def test_masks_policy_but_averages_weighted_kl_over_all_samples(self):
    class _Model(nnx.Module):

      def __init__(self):
        self.weight = nnx.Param(jnp.asarray(2.0))

    class _Optimizer:

      def __init__(self):
        self.grads = None

      def update(self, model, grads):
        del model
        self.grads = grads

    trainer = object.__new__(
        self_inf_loo_policy_only_trainer.PolicyOnlySelfInfLooTrainer
    )
    trainer.gen_model_input_fn = lambda inputs: inputs
    trainer.loss_fn = lambda model, policy, weighted_kl: (
        model.weight.get_value() * (policy + weighted_kl),
        {"kl": weighted_kl},
    )
    trainer._score_loss_fn = lambda model, policy, weighted_kl: (
        model.weight.get_value() * policy,
        {"kl": jnp.asarray(0.0)},
    )
    trainer._score_loss_has_aux = True
    trainer._lora_enabled = False
    trainer._has_aux = True
    trainer.scope = "batch"
    trainer.num_generations = 4
    trainer.min_keep_fraction = 0.25

    model = _Model()
    optimizer = _Optimizer()
    loss, aux = trainer._train_step(
        model,
        optimizer,
        {
            "policy": jnp.asarray([1.0, 1.0, -1.0, 1.0]),
            "weighted_kl": jnp.asarray([10.0, 21.0, 30.0, 43.0]),
        },
    )

    # Policy sample 2 is removed: selected policy mean is 1.0. The already
    # beta-weighted KL component uses all samples: mean is 26.0.
    np.testing.assert_allclose(
        optimizer.grads["weight"].get_value(), 27.0
    )
    np.testing.assert_allclose(loss, 54.0)
    np.testing.assert_allclose(aux["kl"], 26.0)
    np.testing.assert_allclose(aux["skipped_samples"], 1.0)

if __name__ == "__main__":
  absltest.main()
