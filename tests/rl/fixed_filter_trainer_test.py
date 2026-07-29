# Copyright 2025 Google LLC
"""Tests for fixed-ratio random and reward filtering."""

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np

from tunix.rl import fixed_filter_trainer


class FixedFilterTrainerTest(absltest.TestCase):

  def test_reward_filters_lowest_quality_with_reproducible_ties(self):
    mask, count = fixed_filter_trainer._selection_mask(
        jnp.asarray([0.0, 0.0, 1.0, 2.0]),
        jnp.asarray([0.8, 0.2, 0.4, 0.6]),
        jnp.asarray(0.0),
        0.25,
        "reward",
    )
    np.testing.assert_array_equal(mask, [True, False, True, True])
    self.assertEqual(int(count), 1)

  def test_random_filter_uses_stochastic_rounding(self):
    quality = jnp.arange(4, dtype=jnp.float32)
    ties = jnp.asarray([0.4, 0.1, 0.3, 0.2])
    low_mask, low_count = fixed_filter_trainer._selection_mask(
        quality, ties, jnp.asarray(0.39), 0.10, "random"
    )
    high_mask, high_count = fixed_filter_trainer._selection_mask(
        quality, ties, jnp.asarray(0.41), 0.10, "random"
    )
    self.assertEqual(int(low_count), 1)
    self.assertEqual(int(high_count), 0)
    self.assertEqual(int(jnp.sum(~low_mask)), 1)
    self.assertEqual(int(jnp.sum(~high_mask)), 0)

  def test_reward_and_random_differ_only_in_ranking_signal(self):
    quality = jnp.asarray([3.0, 0.0, 2.0, 1.0])
    ties = jnp.asarray([0.0, 0.3, 0.2, 0.1])
    reward_mask, _ = fixed_filter_trainer._selection_mask(
        quality, ties, jnp.asarray(0.0), 0.25, "reward"
    )
    random_mask, _ = fixed_filter_trainer._selection_mask(
        quality, ties, jnp.asarray(0.0), 0.25, "random"
    )
    np.testing.assert_array_equal(reward_mask, [True, False, True, True])
    np.testing.assert_array_equal(random_mask, [False, True, True, True])


if __name__ == "__main__":
  absltest.main()
