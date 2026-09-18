from __future__ import annotations

import unittest
import subprocess
from pathlib import Path

from my_example.alignment_baselines.config import (
    AlignmentBaselineConfig,
    parse_alignment_args,
)


class AlignmentConfigTest(unittest.TestCase):

    def test_method_specific_defaults(self):
        for method, q in (("learnalign", 2), ("gradalign", 4)):
            config, remaining = parse_alignment_args(["--alignment-method", method])
            self.assertEqual(config.selection_ratio, q)
            self.assertEqual(AlignmentBaselineConfig(method=method).selection_ratio, q)
            self.assertEqual(config.to_dict()["selection_ratio"], q)
            self.assertEqual(remaining, [])

    def test_independent_overrides_preserve_frozen_arguments(self):
        for method, q in (("learnalign", 3), ("gradalign", 5)):
            config, remaining = parse_alignment_args([
                "--alignment-method", method,
                "--learnalign-selection-ratio", "3",
                "--gradalign-selection-ratio", "5",
                "--learning-rate", "1e-6",
            ])
            self.assertEqual(config.selection_ratio, q)
            self.assertEqual(remaining, ["--learning-rate", "1e-6"])

    def test_legacy_override_is_supported_for_single_method(self):
        config, _ = parse_alignment_args([
            "--alignment-method", "learnalign", "--selection-ratio", "4",
        ])
        self.assertEqual(config.selection_ratio, 4)

    def test_invalid_effective_ratio_is_rejected(self):
        config, _ = parse_alignment_args([
            "--alignment-method", "learnalign", "--learnalign-selection-ratio", "1",
        ])
        with self.assertRaises(ValueError):
            config.validate(train_batch_size=4)

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


class AlignmentSuiteRoutingTest(unittest.TestCase):

    def run_suite(self, *args):
        script = Path(__file__).resolve().parents[2] / "my_example/run_alignment_baseline_suite.sh"
        return subprocess.run(
            ["bash", str(script), "--seeds", "5", "0", "--mismatch", "0.2",
             "--dry-run", *args], capture_output=True, text=True,
        )

    def test_default_routes_follow_requested_seed_order(self):
        result = self.run_suite()
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 4)
        for line, seed, method, q in zip(
            lines, (5, 5, 0, 0), ("learnalign", "gradalign") * 2, (2, 4) * 2,
        ):
            self.assertIn(f"TUNIX_EXPERIMENT_SEED={seed}", line)
            self.assertIn(f"TUNIX_REWARD_RANK_NOISE_SEED={seed}", line)
            self.assertIn("TUNIX_REWARD_RANK_NOISE_FRACTION=0.2", line)
            self.assertIn(f"{method} --{method}-selection-ratio {q}", line)

    def test_independent_overrides_and_forwarding(self):
        result = self.run_suite(
            "--learnalign-selection-ratio", "3", "--gradalign-selection-ratio", "5",
            "--", "--max-eval-examples", "1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("learnalign --max-eval-examples 1 --learnalign-selection-ratio 3", result.stdout)
        self.assertIn("gradalign --max-eval-examples 1 --gradalign-selection-ratio 5", result.stdout)

    def test_suite_rejects_shared_or_misplaced_ratios(self):
        for option in ("--selection-ratio", "--learnalign-selection-ratio", "--gradalign-selection-ratio"):
            result = self.run_suite("--", option, "2")
            self.assertEqual(result.returncode, 2)
            self.assertIn("before --", result.stderr)

    def test_suite_rejects_invalid_ratio(self):
        for q in ("1", "0", "-2", "abc"):
            result = self.run_suite("--learnalign-selection-ratio", q)
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
