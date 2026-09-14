from __future__ import annotations

import unittest

import numpy as np

from my_example.alignment_baselines.equivalence import (
    compare_learnalign_feature_paths,
    feature_execution_mode,
    prompt_subbatch_completion_slices,
)


class AlignmentEquivalenceTest(unittest.TestCase):

    def test_prompt_subbatches_preserve_complete_rollout_groups(self):
        self.assertEqual(
            prompt_subbatch_completion_slices(
                prompt_count=4,
                num_rollouts=8,
                prompt_subbatch_size=2,
            ),
            (slice(0, 16), slice(16, 32)),
        )
        self.assertEqual(
            prompt_subbatch_completion_slices(
                prompt_count=3,
                num_rollouts=8,
                prompt_subbatch_size=2,
            ),
            (slice(0, 16), slice(16, 24)),
        )

    def test_prompt_subbatches_reject_nonpositive_sizes(self):
        with self.assertRaisesRegex(ValueError, "prompt_count"):
            prompt_subbatch_completion_slices(
                prompt_count=0,
                num_rollouts=8,
                prompt_subbatch_size=2,
            )

    def test_feature_execution_modes_are_mutually_exclusive(self):
        self.assertEqual(
            feature_execution_mode(
                equivalence_enabled=True,
                grouped_feature_estimation=True,
            ),
            "dual",
        )
        self.assertEqual(
            feature_execution_mode(
                equivalence_enabled=False,
                grouped_feature_estimation=True,
            ),
            "grouped",
        )
        self.assertEqual(
            feature_execution_mode(
                equivalence_enabled=False,
                grouped_feature_estimation=False,
            ),
            "legacy",
        )

    def test_identical_features_pass_all_checks(self):
        features = np.arange(6 * 8, dtype=np.float64).reshape((6, 8)) + 1.0
        report = compare_learnalign_feature_paths(
            features,
            features.copy(),
            selection_ratio=4,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["selected_jaccard"], 1.0)
        self.assertAlmostEqual(report["minimum_row_cosine"], 1.0)

    def test_material_feature_change_fails(self):
        legacy = np.eye(4, dtype=np.float64)
        grouped = legacy.copy()
        grouped[0] = -grouped[0]
        report = compare_learnalign_feature_paths(
            legacy,
            grouped,
            selection_ratio=4,
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["feature_allclose"])

    def test_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same"):
            compare_learnalign_feature_paths(
                np.ones((2, 4)),
                np.ones((2, 3)),
                selection_ratio=4,
            )

    def test_selected_set_mode_accepts_same_selection(self):
        legacy = np.arange(8 * 6, dtype=np.float64).reshape((8, 6)) + 1.0
        grouped = legacy * 1.01
        report = compare_learnalign_feature_paths(
            legacy,
            grouped,
            selection_ratio=4,
            score_weights=np.linspace(0.1, 0.25, 8),
            acceptance_mode="selected-set",
        )
        self.assertFalse(report["strict_checks_passed"])
        self.assertEqual(report["selected_jaccard"], 1.0)
        self.assertTrue(report["passed"])

    def test_rejects_wrong_score_weight_count(self):
        with self.assertRaisesRegex(ValueError, "one value per prompt"):
            compare_learnalign_feature_paths(
                np.ones((4, 3)),
                np.ones((4, 3)),
                selection_ratio=4,
                score_weights=np.ones(3),
            )


if __name__ == "__main__":
    unittest.main()
