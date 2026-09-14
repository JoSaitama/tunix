from __future__ import annotations

import unittest

import numpy as np

from my_example.alignment_baselines.equivalence import (
    compare_learnalign_feature_paths,
)


class AlignmentEquivalenceTest(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
