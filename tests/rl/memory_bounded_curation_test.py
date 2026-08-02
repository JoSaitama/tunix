"""Tests for memory-bounded dynamic trajectory curation primitives."""

from unittest import mock

from absl.testing import absltest
from flax import struct
from flax import nnx
import jax
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


@struct.dataclass
class _SyntheticTrainExample:
  feature: jnp.ndarray
  completion_mask: jnp.ndarray


def _update_loss(model, train_example, algo_config=None):
  del algo_config
  values = model(train_example.feature)
  active = jnp.any(train_example.completion_mask != 0, axis=-1)
  loss = jnp.sum(values * active) / jnp.clip(jnp.sum(active), min=1)
  return loss, {"base_loss": loss}


def _score_loss(model, train_example, algo_config=None):
  del algo_config
  if train_example.completion_mask.ndim != 2:
    raise ValueError("Score loss requires a restored batch dimension.")
  return jnp.sum(model(train_example.feature))


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

  def test_vmapped_statistics_match_memory_bounded_statistics(self):
    gradients = {
        "weight": jnp.asarray([[1.0, 2.0], [-3.0, 1.0], [2.0, -4.0], [1.0, 1.0]])
    }
    expected = memory_bounded_curation.dtv_statistics(
        _synthetic_grad_fn,
        None,
        {"gradient": gradients["weight"]},
        scope="group",
        group_size=2,
    )
    actual = memory_bounded_curation.statistics_from_gradient_tree(
        gradients, scope="group", group_size=2
    )

    for key in expected:
      np.testing.assert_allclose(actual[key], expected[key], rtol=1e-6)

  def _run_policy_trainer(self):
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
    trainer.with_gen_model_input_fn(
        lambda x: {
            "train_example": _SyntheticTrainExample(
                feature=x,
                completion_mask=jnp.ones((x.shape[0], 1), dtype=jnp.int32),
            ),
            "algo_config": object(),
        }
    )
    train_step, _ = trainer.jit_train_and_eval_step(cache_nnx_graph=True)

    loss, aux, grad_norm = train_step(jnp.asarray([4.0, -1.0]))

    self.assertAlmostEqual(float(loss), 4.0)
    self.assertIsNotNone(aux)
    self.assertGreater(float(grad_norm), 0.0)

  def test_policy_trainer_defaults_to_vmapped_scores(self):
    self._run_policy_trainer()

  def test_policy_trainer_keeps_jvp_fallback(self):
    with mock.patch.dict(
        "os.environ", {"TUNIX_POLICY_DTV_SCORE_BACKEND": "jvp"}
    ):
      self._run_policy_trainer()

  def test_masked_aggregate_gradient_matches_explicit_retained_mean(self):
    model = _ScalarModel()
    inputs = {
        "train_example": _SyntheticTrainExample(
            feature=jnp.asarray([2.0, -5.0, 7.0]),
            completion_mask=jnp.ones((3, 2), dtype=jnp.int32),
        )
    }
    mask = jnp.asarray([True, False, True])
    grad_fn = memory_bounded_curation.make_masked_aggregate_grad_fn(
        _update_loss, wrt=nnx.Param, has_aux=True
    )

    (loss, aux), gradient = grad_fn(model, inputs, mask)
    gradient_value = jax.tree_util.tree_leaves(gradient)[0]

    self.assertAlmostEqual(float(loss), 4.5)
    self.assertAlmostEqual(float(aux["base_loss"]), 4.5)
    self.assertAlmostEqual(float(gradient_value), 4.5)

  def test_masked_aggregate_preserves_preexisting_zero_completion_rows(self):
    model = _ScalarModel()
    inputs = {
        "train_example": _SyntheticTrainExample(
            feature=jnp.asarray([2.0, 100.0, 7.0]),
            completion_mask=jnp.asarray([[1, 1], [0, 0], [1, 1]]),
        )
    }
    mask = jnp.asarray([True, True, True])
    grad_fn = memory_bounded_curation.make_masked_aggregate_grad_fn(
        _update_loss, wrt=nnx.Param, has_aux=True
    )

    (loss, _), gradient = grad_fn(model, inputs, mask)
    gradient_value = jax.tree_util.tree_leaves(gradient)[0]

    self.assertAlmostEqual(float(loss), 4.5)
    self.assertAlmostEqual(float(gradient_value), 4.5)


if __name__ == "__main__":
  absltest.main()
