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

"""Tests for strict leave-one-out self-influence score helpers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

from tunix.rl import self_inf_loo_trainer


class SelfInfLooTrainerTest(parameterized.TestCase):

  def test_batch_statistics_use_n_minus_one(self):
    grads = {"w": jnp.asarray([
        [1.0, 0.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
    ])}

    statistics = self_inf_loo_trainer._batch_loo_statistics(grads)

    np.testing.assert_allclose(
        statistics["raw_self"], [1.0, 1.0, 1.0, 1.0]
    )
    np.testing.assert_allclose(
        statistics["raw_cross_sum"], [0.0, 0.0, -2.0, 0.0]
    )
    np.testing.assert_allclose(
        statistics["loo_score"], [0.0, 0.0, -2.0 / 3.0, 0.0]
    )
    np.testing.assert_allclose(
        statistics["standard_score"], [0.25, 0.25, -0.25, 0.25]
    )

  def test_group_statistics_are_isolated_and_use_g_minus_one(self):
    grads = {"w": jnp.asarray([
        [1.0, 0.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
    ])}

    statistics = self_inf_loo_trainer._group_loo_statistics(grads, 2)

    np.testing.assert_allclose(statistics["loo_score"], [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        statistics["raw_cross_sum"], [1.0, 1.0, 0.0, 0.0]
    )

  def test_cap_keeps_highest_quarter_without_sample_loop(self):
    scores = jnp.asarray([-4.0, -3.0, -2.0, -1.0])

    final_mask, threshold_mask, retained_by_cap, cap_triggered = (
        self_inf_loo_trainer._capped_mask(scores, 0.25)
    )

    np.testing.assert_array_equal(threshold_mask, [False, False, False, False])
    np.testing.assert_array_equal(final_mask, [False, False, False, True])
    np.testing.assert_array_equal(retained_by_cap, [False, False, False, True])
    self.assertTrue(bool(cap_triggered))

  def test_cap_does_not_remove_nonnegative_samples(self):
    scores = jnp.asarray([2.0, 1.0, 0.0, -1.0])

    final_mask, threshold_mask, retained_by_cap, cap_triggered = (
        self_inf_loo_trainer._capped_mask(scores, 0.25)
    )

    np.testing.assert_array_equal(final_mask, threshold_mask)
    np.testing.assert_array_equal(retained_by_cap, [False, False, False, False])
    self.assertFalse(bool(cap_triggered))

  def test_group_cap_is_applied_independently_per_prompt(self):
    grouped_scores = jnp.asarray([
        [-4.0, -3.0, -2.0, -1.0],
        [-1.0, 2.0, 1.0, -2.0],
    ])

    final_mask, threshold_mask, retained_by_cap, cap_triggered = (
        self_inf_loo_trainer._group_capped_mask(grouped_scores, 0.25)
    )

    np.testing.assert_array_equal(
        final_mask,
        [[False, False, False, True], [False, True, True, False]],
    )
    np.testing.assert_array_equal(
        threshold_mask,
        [[False, False, False, False], [False, True, True, False]],
    )
    np.testing.assert_array_equal(
        retained_by_cap,
        [[False, False, False, True], [False, False, False, False]],
    )
    np.testing.assert_array_equal(cap_triggered, [True, False])

  @parameterized.parameters((0,), (1,), (3,))
  def test_group_statistics_reject_invalid_group_size(self, group_size):
    grads = {"w": jnp.ones((4, 2))}
    with self.assertRaises(ValueError):
      self_inf_loo_trainer._group_loo_statistics(grads, group_size)


if __name__ == "__main__":
  absltest.main()
