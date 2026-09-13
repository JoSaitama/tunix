from __future__ import annotations

import unittest

from my_example.alignment_baselines.config import parse_alignment_args


class AlignmentConfigTest(unittest.TestCase):

    def test_baseline_flags_are_removed_before_frozen_parser(self):
        config, remaining = parse_alignment_args(
            [
                "--alignment-method",
                "gradalign",
                "--gradalign-validation-prompts",
                "30",
                "--learning-rate",
                "1e-6",
            ]
        )
        self.assertEqual(config.method, "gradalign")
        self.assertEqual(config.gradalign_validation_prompts, 30)
        self.assertEqual(remaining, ["--learning-rate", "1e-6"])

    def test_selection_micro_batch_must_preserve_training_batch_multiple(self):
        config, _ = parse_alignment_args(
            [
                "--alignment-method",
                "learnalign",
                "--selection-micro-batch-size",
                "3",
            ]
        )
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            config.validate(train_batch_size=4)


if __name__ == "__main__":
    unittest.main()
