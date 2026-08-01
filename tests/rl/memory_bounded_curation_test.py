"""Tests for memory-bounded dynamic trajectory curation primitives."""

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp
import numpy as np
import optax
from tunix.rl import memory_bounded_curation
from tunix.rl import self_inf_policy_trainer
from tunix.sft import peft_trainer


def _synthetic_grad_fn(unused_model, inputs):
  gradient = {"weight": jnp.squeeze(inputs["gradient"], axis=0)}
  return jnp.asarray(0.0), gradient


class _ScalarModel(nnx.Module):

  def __init__(self):
    self.weight = nnx.Param(jnp.asarray(1.0))

  def __call__(self, feature):
    return self.weight[...] * feature


def _update_loss(model, feature):
  loss = jnp.sum(model(feature))
  return loss, {"base_loss": loss}


def _score_loss(model, feature):
  return jnp.sum(model(feature))


class MemoryBoundedCurationTest(absltest.TestCase):

  def test_batch_statistics_match_explicit_gradient_matrix(self):
    gradients = jnp.asarray([[1.0, 2.0], [-3.0, 1.0], [2.0, -4.0]])
    stats = memory_bounded_curation.dtv_statistics(
        _synthetic_grad_fn,
        None,
        {"gradient": gradients},
        scope="batch",
        group_size=None,
    )

    total = np.sum(gradients, axis=0)
    raw_self = np.sum(np.asarray(gradients) ** 2, axis=1)
    raw_cross = np.asarray(gradients) @ total - raw_self
    np.testing.assert_allclose(stats["raw_self"], raw_self)
    np.testing.assert_allclose(stats["raw_cross_sum"], raw_cross)
    np.testing.assert_allclose(
        stats["standard_score"], (raw_self + raw_cross) / 3.0
    )
    np.testing.assert_allclose(stats["loo_score"], raw_cross / 2.0)

  def test_group_statistics_do_not_cross_group_boundaries(self):
    gradients = jnp.asarray([[1.0], [-1.0], [2.0], [3.0]])
    stats = memory_bounded_curation.dtv_statistics(
        _synthetic_grad_fn,
        None,
        {"gradient": gradients},
        scope="group",
        group_size=2,
    )

    np.testing.assert_allclose(stats["standard_score"], [0.0, 0.0, 5.0, 7.5])
    np.testing.assert_allclose(stats["loo_score"], [-1.0, -1.0, 6.0, 6.0])

  def test_policy_trainer_uses_staged_jits(self):
    model = _ScalarModel()
    trainer = self_inf_policy_trainer.PolicySelfInfTrainer(
        model=model,
        optimizer=optax.sgd(0.1),
        training_config=peft_trainer.TrainingConfig(
            eval_every_n_steps=1,
            max_steps=1,
        ),
        custom_checkpoint_metadata_fn=lambda: {},
        scope="group",
        num_generations=2,
    )
    trainer.with_loss_fn(_update_loss, has_aux=True)
    trainer.with_policy_score_loss_fn(_score_loss)
    trainer.with_gen_model_input_fn(lambda x: {"feature": x})
    train_step, _ = trainer.jit_train_and_eval_step(cache_nnx_graph=True)

    loss, aux, grad_norm = train_step(jnp.asarray([4.0, -1.0]))

    self.assertAlmostEqual(float(loss), 4.0)
    self.assertIsNotNone(aux)
    self.assertGreater(float(grad_norm), 0.0)


if __name__ == "__main__":
  absltest.main()
