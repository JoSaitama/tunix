from __future__ import annotations

import unittest

import numpy as np

from my_example.alignment_baselines.gradient_batching import (
    feature_completion_slices,
)


class AlignmentGradientBatchingTest(unittest.TestCase):

    def test_learnalign_slices_one_prompt_rollout_group_at_a_time(self):
        slices = feature_completion_slices(
            prompt_count=4,
            num_rollouts=8,
            promptwise=True,
        )
        self.assertEqual(
            [(value.start, value.stop) for value in slices],
            [(0, 8), (8, 16), (16, 24), (24, 32)],
        )

    def test_gradalign_keeps_the_existing_full_chunk_path(self):
        slices = feature_completion_slices(
            prompt_count=4,
            num_rollouts=4,
            promptwise=False,
        )
        self.assertEqual(
            [(value.start, value.stop) for value in slices],
            [(0, 16)],
        )

    def test_promptwise_reassembly_preserves_group_means_and_order(self):
        values = np.arange(32 * 7, dtype=np.float32).reshape((32, 7))
        legacy = values.reshape((4, 8, 7)).mean(axis=1)
        promptwise = np.concatenate(
            [
                values[value].reshape((1, 8, 7)).mean(axis=1)
                for value in feature_completion_slices(
                    prompt_count=4,
                    num_rollouts=8,
                    promptwise=True,
                )
            ],
            axis=0,
        )
        np.testing.assert_array_equal(promptwise, legacy)

    def test_rejects_invalid_sizes(self):
        with self.assertRaisesRegex(ValueError, "prompt_count"):
            feature_completion_slices(
                prompt_count=0,
                num_rollouts=8,
                promptwise=True,
            )
        with self.assertRaisesRegex(ValueError, "num_rollouts"):
            feature_completion_slices(
                prompt_count=4,
                num_rollouts=0,
                promptwise=True,
            )


if __name__ == "__main__":
    unittest.main()
