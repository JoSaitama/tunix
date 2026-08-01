"""Tests for GSM8K-compatible fixed-filter selection."""

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from tunix.rl import fixed_filter_trainer


class FixedFilterTrainerTest(absltest.TestCase):

  def test_reward_filters_lowest_advantage(self):
    mask, count = fixed_filter_trainer._selection_mask(
        jnp.array([-2.0, 1.0, 0.0, 3.0]),
        jnp.array([0.1, 0.2, 0.3, 0.4]),
        jnp.array(0.0),
        0.25,
        "reward",
    )
    self.assertEqual(int(count), 1)
    np.testing.assert_array_equal(mask, [False, True, True, True])

  def test_random_selection_ignores_quality(self):
    ties = jnp.array([0.4, 0.1, 0.3, 0.2])
    first, _ = fixed_filter_trainer._selection_mask(
        jnp.arange(4), ties, jnp.array(0.0), 0.5, "random"
    )
    second, _ = fixed_filter_trainer._selection_mask(
        -jnp.arange(4), ties, jnp.array(0.0), 0.5, "random"
    )
    np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
  absltest.main()
