from __future__ import annotations

import unittest

import numpy as np

from my_example.alignment_baselines.scoring import (
    cosine_scores,
    learnability,
    learnalign_scores,
    stable_top_indices,
)


class AlignmentScoringTest(unittest.TestCase):

    def test_learnability_is_maximal_at_half_success(self):
        outcomes = np.asarray([[0, 0, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
        np.testing.assert_allclose(learnability(outcomes), [0.0, 0.25, 0.0])

    def test_learnalign_fast_row_mean_matches_pairwise_matrix(self):
        features = np.asarray([[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0]])
        weights = np.asarray([0.25, 0.2, 0.1])
        unit = features / np.linalg.norm(features, axis=1, keepdims=True)
        weighted = unit * weights[:, None]
        expected = (weighted @ weighted.T).mean(axis=1)
        np.testing.assert_allclose(
            learnalign_scores(features, weights), expected, atol=1e-12
        )

    def test_gradalign_uses_cosine_direction(self):
        candidates = np.asarray([[2.0, 0.0], [0.0, 4.0], [-3.0, 0.0]])
        scores = cosine_scores(candidates, np.asarray([10.0, 0.0]))
        np.testing.assert_allclose(scores, [1.0, 0.0, -1.0], atol=1e-12)

    def test_stable_top_indices_breaks_ties_by_source_order(self):
        result = stable_top_indices(np.asarray([0.5, 0.8, 0.8, 0.1]), 2)
        np.testing.assert_array_equal(result, [1, 2])


if __name__ == "__main__":
    unittest.main()
