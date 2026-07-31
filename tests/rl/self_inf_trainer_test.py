# Copyright 2026 Google LLC
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

"""Tests for the self-influence RL trainer."""

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp
import optax
from tunix.rl import self_inf_trainer
from tunix.sft import peft_trainer


class ScalarModel(nnx.Module):

  def __init__(self):
    self.weight = nnx.Param(jnp.asarray(1.0, dtype=jnp.float32))

  def __call__(self, feature):
    return self.weight[...] * feature


def _loss_fn(model: ScalarModel, feature):
  loss = jnp.sum(model(feature))
  return loss, {"base_loss": loss}


class StaticConfig:

  def __init__(self, scale: float):
    self.scale = scale


def _loss_fn_with_static(model: ScalarModel, feature, static_config: StaticConfig):
  loss = jnp.sum(model(feature) * static_config.scale)
  return loss, {"base_loss": loss}


class SelfInfTrainerTest(absltest.TestCase):

  def _create_trainer(
      self,
      *,
      scope: str,
      num_generations: int | None = None,
      dot_threshold: float = 0.0,
  ) -> tuple[self_inf_trainer.SelfInfTrainer, ScalarModel]:
    model = ScalarModel()
    trainer = self_inf_trainer.SelfInfTrainer(
        model=model,
        optimizer=optax.sgd(0.1),
        training_config=peft_trainer.TrainingConfig(
            eval_every_n_steps=1,
            max_steps=1,
        ),
        custom_checkpoint_metadata_fn=lambda: {},
        scope=scope,
        num_generations=num_generations,
        dot_threshold=dot_threshold,
    )
    trainer.with_loss_fn(_loss_fn, has_aux=True)
    return trainer, model

  def test_batch_scope_filters_against_batch_mean_gradient(self):
    trainer, model = self._create_trainer(scope="batch")

    loss, aux, grad_norm = trainer._train_step(
        model,
        trainer.optimizer,
        jnp.asarray([3.0, -1.0, -1.0], dtype=jnp.float32),
    )

    self.assertIsNotNone(aux)
    self.assertAlmostEqual(float(loss), 3.0, places=5)
    self.assertAlmostEqual(float(aux["skipped_samples"]), 2.0, places=5)
    self.assertAlmostEqual(
        float(aux["self_inf_kept_fraction"]), 1.0 / 3.0, places=5
    )
    self.assertAlmostEqual(float(model.weight[...]), 0.7, places=5)
    self.assertGreater(float(grad_norm), 0.0)

  def test_group_scope_filters_with_num_generations(self):
    trainer, model = self._create_trainer(scope="group", num_generations=2)

    loss, aux, grad_norm = trainer._train_step(
        model,
        trainer.optimizer,
        jnp.asarray([4.0, -1.0, -3.0, 1.0], dtype=jnp.float32),
    )

    self.assertIsNotNone(aux)
    self.assertAlmostEqual(float(loss), 0.5, places=5)
    self.assertAlmostEqual(float(aux["skipped_samples"]), 2.0, places=5)
    self.assertAlmostEqual(float(aux["self_inf_kept_fraction"]), 0.5, places=5)
    self.assertAlmostEqual(float(model.weight[...]), 0.95, places=5)
    self.assertGreater(float(grad_norm), 0.0)

  def test_group_scope_requires_num_generations(self):
    trainer, model = self._create_trainer(scope="group")

    with self.assertRaisesRegex(
        ValueError, "requires a positive num_generations"
    ):
      trainer._train_step(
          model,
          trainer.optimizer,
          jnp.asarray([1.0, -1.0], dtype=jnp.float32),
      )

  def test_static_non_array_inputs_are_broadcast_in_vmap(self):
    trainer, model = self._create_trainer(scope="batch")
    static_config = StaticConfig(scale=2.0)
    trainer.with_loss_fn(_loss_fn_with_static, has_aux=True)
    trainer.with_gen_model_input_fn(
        lambda x: {"feature": x, "static_config": static_config}
    )

    loss, aux, grad_norm = trainer._train_step(
        model,
        trainer.optimizer,
        jnp.asarray([3.0, -1.0, -1.0], dtype=jnp.float32),
    )

    self.assertIsNotNone(aux)
    self.assertAlmostEqual(float(loss), 6.0, places=5)
    self.assertAlmostEqual(float(aux["skipped_samples"]), 2.0, places=5)
    self.assertGreater(float(grad_norm), 0.0)


if __name__ == "__main__":
  absltest.main()
