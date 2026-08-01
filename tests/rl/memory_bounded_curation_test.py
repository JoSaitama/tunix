"""Tests for memory-bounded dynamic trajectory curation primitives."""

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from tunix.rl import memory_bounded_curation


def _synthetic_grad_fn(unused_model, inputs):
  gradient = {"weight": jnp.squeeze(inputs["gradient"], axis=0)}
  return jnp.asarray(0.0), gradient


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


if __name__ == "__main__":
  absltest.main()
