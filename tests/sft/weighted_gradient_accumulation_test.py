"""Tests for effective-count weighted gradient accumulation."""

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
import optax
from tunix.sft import peft_trainer
from tunix.sft import weighted_gradient_accumulation


class WeightedGradientAccumulationTest(absltest.TestCase):

  def test_unequal_counts_match_one_global_mean(self):
    params = {"x": jnp.asarray(0.0)}
    transform = weighted_gradient_accumulation.weighted_multisteps(
        optax.sgd(1.0), 2
    )
    state = transform.init(params)

    updates, state = transform.update(
        {"x": jnp.asarray(2.0)},
        state,
        params,
        effective_count=jnp.asarray(2.0),
    )
    np.testing.assert_allclose(updates["x"], 0.0)
    updates, state = transform.update(
        {"x": jnp.asarray(10.0)},
        state,
        params,
        effective_count=jnp.asarray(8.0),
    )

    expected_mean = (2.0 * 2.0 + 10.0 * 8.0) / 10.0
    np.testing.assert_allclose(updates["x"], -expected_mean)
    np.testing.assert_allclose(state.gradient_step, 1)

  def test_zero_count_window_does_not_advance_inner_optimizer(self):
    params = {"x": jnp.asarray(0.0)}
    schedule = optax.cosine_decay_schedule(1.0, decay_steps=10)
    inner = optax.inject_hyperparams(optax.sgd)(learning_rate=schedule)
    transform = weighted_gradient_accumulation.weighted_multisteps(inner, 2)
    state = transform.init(params)

    for _ in range(2):
      updates, state = transform.update(
          {"x": jnp.asarray(0.0)},
          state,
          params,
          effective_count=jnp.asarray(0.0),
      )

    np.testing.assert_allclose(updates["x"], 0.0)
    np.testing.assert_allclose(state.gradient_step, 0)

  def test_dynamic_schedule_advances_once_per_window_and_is_discoverable(self):
    params = {"x": jnp.asarray(0.0)}
    schedule = optax.cosine_decay_schedule(1.0, decay_steps=314)
    inner = optax.inject_hyperparams(optax.sgd)(learning_rate=schedule)
    transform = weighted_gradient_accumulation.weighted_multisteps(inner, 64)
    state = transform.init(params)

    for call_index in range(65):
      updates, state = transform.update(
          {"x": jnp.asarray(1.0)},
          state,
          params,
          effective_count=jnp.asarray(1.0),
      )
      if call_index < 63:
        np.testing.assert_allclose(updates["x"], 0.0)

    np.testing.assert_allclose(state.gradient_step, 1)
    logged_lr = peft_trainer._find_nested_hyperparam(
        state, "learning_rate"
    )
    self.assertIsNotNone(logged_lr)
    np.testing.assert_allclose(logged_lr, schedule(0), rtol=1e-6)


if __name__ == "__main__":
  absltest.main()
