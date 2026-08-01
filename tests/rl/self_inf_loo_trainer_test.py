"""Tests for policy DTV leave-one-out score calculations."""

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from tunix.rl import self_inf_loo_trainer


class SelfInfLooTrainerTest(absltest.TestCase):

  def test_batch_loo_excludes_each_samples_self_term(self):
    stats = self_inf_loo_trainer._batch_loo_statistics({
        "weight": jnp.array([[1.0], [2.0], [-1.0]])
    })
    np.testing.assert_allclose(stats["loo_score"], [0.5, 0.0, -1.5])

  def test_group_loo_never_crosses_prompt_group_boundaries(self):
    stats = self_inf_loo_trainer._group_loo_statistics(
        {"weight": jnp.array([[1.0], [-1.0], [2.0], [3.0]])}, 2
    )
    np.testing.assert_allclose(stats["loo_score"], [-1.0, -1.0, 6.0, 6.0])

  def test_retention_cap_keeps_stable_top_scores(self):
    mask, threshold, retained, triggered = self_inf_loo_trainer._capped_mask(
        jnp.array([-3.0, -1.0, -2.0, -4.0]), 0.25
    )
    self.assertTrue(bool(triggered))
    np.testing.assert_array_equal(threshold, [False] * 4)
    np.testing.assert_array_equal(mask, [False, True, False, False])
    np.testing.assert_array_equal(retained, mask)

  def test_configurable_threshold_controls_loo_filtering(self):
    scores = jnp.array([-0.1, 0.0, 0.1, 0.2])
    mask, threshold, _, triggered = self_inf_loo_trainer._capped_mask(
        scores, min_keep_fraction=0.25, dot_threshold=0.1
    )
    self.assertFalse(bool(triggered))
    np.testing.assert_array_equal(threshold, [False, False, True, True])
    np.testing.assert_array_equal(mask, threshold)


if __name__ == "__main__":
  absltest.main()
