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

"""Tests for policy-score, full-update strict LOO configuration."""

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp
import numpy as np

from tunix.rl import self_inf_loo_policy_trainer
from tunix.rl import self_inf_loo_trainer


class PolicySelfInfLooTrainerTest(absltest.TestCase):

  def test_train_step_scores_policy_but_updates_total_gradients(self):
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
        self_inf_loo_policy_trainer.PolicySelfInfLooTrainer
    )
    trainer.gen_model_input_fn = lambda inputs: inputs
    trainer.loss_fn = lambda model, policy, total: (
        model.weight.value * total,
        {"kl": jnp.asarray(0.0)},
    )
    trainer._score_loss_fn = lambda model, policy, total: (
        model.weight.value * policy,
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
    loss, _ = trainer._train_step(
        model,
        optimizer,
        {
            "policy": jnp.asarray([1.0, 1.0, -1.0, 1.0]),
            "total": jnp.asarray([10.0, 20.0, 30.0, 40.0]),
        },
    )

    np.testing.assert_allclose(loss, 2.0 * 70.0 / 3.0)
    np.testing.assert_allclose(optimizer.grads["weight"].value, 70.0 / 3.0)

  def test_policy_score_loss_setter_is_separate_from_update_loss(self):
    trainer = object.__new__(
        self_inf_loo_policy_trainer.PolicySelfInfLooTrainer
    )
    trainer._score_loss_fn = None
    trainer._score_loss_has_aux = False
    trainer.loss_fn = lambda *_: "total-update-loss"
    policy_loss_fn = lambda *_: "policy-score-loss"

    trainer.with_policy_score_loss_fn(policy_loss_fn, has_aux=True)

    self.assertIs(trainer._score_loss_fn, policy_loss_fn)
    self.assertTrue(trainer._score_loss_has_aux)
    self.assertEqual(trainer.loss_fn(None), "total-update-loss")

  def test_policy_trainer_rejects_missing_score_loss(self):
    trainer = object.__new__(
        self_inf_loo_policy_trainer.PolicySelfInfLooTrainer
    )
    trainer._score_loss_fn = None

    with self.assertRaisesRegex(RuntimeError, "requires a policy score loss"):
      trainer._train_step(None, None, None)

  def test_policy_scores_can_mask_distinct_total_gradients(self):
    policy_grads = {"w": jnp.asarray([[1.0], [1.0], [-1.0], [1.0]])}
    total_grads = {"w": jnp.asarray([[10.0], [20.0], [30.0], [40.0]])}

    statistics = self_inf_loo_trainer._batch_loo_statistics(policy_grads)
    final_mask, _, _, _ = self_inf_loo_trainer._capped_mask(
        statistics["loo_score"], 0.25
    )
    denominator = jnp.clip(jnp.sum(final_mask), 1.0)
    update = jnp.sum(
        total_grads["w"] * final_mask.reshape((-1, 1)), axis=0
    ) / denominator

    np.testing.assert_array_equal(final_mask, [True, True, False, True])
    np.testing.assert_allclose(update, [70.0 / 3.0])


if __name__ == "__main__":
  absltest.main()
