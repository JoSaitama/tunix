"""Server CPU tests: score semantics, controls, reporting, and CLI safeguards."""

import ast
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from my_example.rebuttal import utility_math as m
from my_example.rebuttal.audit_grpo_utility import parser, validate, run


class UtilityMathTest(unittest.TestCase):
    def test_scores_match_explicit_gradient_inner_products(self):
        g = np.random.default_rng(2).normal(size=(16, 7))
        for scope, width in (("group", 4), ("batch", 16)):
            result = m.scores_from_gram(g @ g.T, scope=scope)
            for start in range(0, 16, width):
                block = g[start:start + width]
                dtv = block @ block.mean(axis=0)
                loo = np.sum(block * (block.sum(axis=0) - block), axis=1) / (width - 1)
                np.testing.assert_allclose(result["dtv"][start:start + width], dtv)
                np.testing.assert_allclose(result["loo"][start:start + width], loo)
            np.testing.assert_allclose(result["dtv"], result["self"] + result["cross"])

    def test_group_scope_has_no_other_prompt_leak(self):
        g = np.arange(32).reshape(16, 2).astype(float)
        before = m.scores_from_gram(g @ g.T)
        g[4:] *= -20
        after = m.scores_from_gram(g @ g.T)
        np.testing.assert_allclose(before["dtv"][:4], after["dtv"][:4])

    def test_whole_group_flip_preserves_scores_but_reverses_external_utility(self):
        g = np.array([[1., 2.], [3., -1.], [-1., .5], [2., .3]])
        h = np.array([.8, -.2])
        original, flipped = m.scores_from_gram(g @ g.T), m.scores_from_gram((-g) @ (-g).T)
        np.testing.assert_array_equal(original["dtv"], flipped["dtv"])
        np.testing.assert_array_equal(original["loo"], flipped["loo"])
        np.testing.assert_allclose((-g) @ h, -(g @ h))

    def test_cap_stable_ties_and_threshold_mask(self):
        masks = m.selection_masks(np.ones(8), -np.ones(8))
        np.testing.assert_array_equal(np.flatnonzero(masks["loo"]), [0, 4])
        self.assertFalse(masks["loo_threshold"].any())
        np.testing.assert_array_equal(masks["loo_cap_rescue"], masks["loo"])

    def test_batch_cap_matches_batch_not_prompt(self):
        masks = m.selection_masks(np.ones(8), -np.arange(1, 9), scope="batch")
        np.testing.assert_array_equal(np.flatnonzero(masks["loo"]), [0, 1])

    def test_threshold_only_no_silent_minimum(self):
        masks = m.selection_masks(np.ones(4), -np.ones(4), min_keep_fraction=0.)
        self.assertFalse(masks["loo"].any())

    def test_zero_score_retained(self):
        masks = m.selection_masks(np.zeros(4), np.zeros(4))
        self.assertTrue(masks["dtv"].all())
        self.assertTrue(masks["loo"].all())

    def test_nonfinite_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                m.selection_masks([bad, 1, 1, 1], np.ones(4))
            with self.assertRaises(ValueError):
                m.scores_from_gram(np.full((4, 4), bad))

    def test_matched_random_counts_including_empty_full(self):
        mask = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0], dtype=bool)
        out = m.matched_random(mask, scope="group", group_size=4, seed=1, batch_id=2, repeat=0)
        np.testing.assert_array_equal(out.reshape(-1, 4).sum(axis=1), mask.reshape(-1, 4).sum(axis=1))
        np.testing.assert_array_equal(out, m.matched_random(mask, scope="group", group_size=4,
                                                          seed=1, batch_id=2, repeat=0))

    def test_selection_does_not_use_heldout(self):
        g = np.random.default_rng(0).normal(size=(4, 3))
        scores = m.scores_from_gram(g @ g.T)
        first = m.selection_masks(scores["dtv"], scores["loo"])
        # External labels/gradient may change utility, never the selector mask.
        for h in (np.ones(3), -np.ones(3), np.zeros(3)):
            self.assertEqual((g @ h).shape, (4,))
            second = m.selection_masks(scores["dtv"], scores["loo"])
            np.testing.assert_array_equal(first["dtv"], second["dtv"])
            np.testing.assert_array_equal(first["loo"], second["loo"])

    def test_spearman_ties_and_undefined(self):
        self.assertAlmostEqual(m.spearman([0, 0, 1, 2], [1, 1, 3, 4]), 1.)
        self.assertIsNone(m.spearman([0, 0, 0], [1, 2, 3]))
        self.assertIsNone(m.spearman([1, 2], [1, 2]))

    def test_cluster_intervals_do_not_invent_single_cluster_ci(self):
        self.assertIsNone(m.cluster_interval([1, 2, 3], [0, 0, 0]))
        report = m.correlation_interval([1, 2, 3], [1, 2, 3], [0, 0, 0])
        self.assertIsNone(report["ci95"])
        self.assertEqual(report["valid_bootstrap_draws"], 0)
        self.assertEqual(len(m.cluster_interval([1, 2, 3], [0, 1, 2], repeats=10)), 2)

    def test_first_order_sign_predicts_quadratic_loss_change(self):
        theta, target, g, eta = np.array([2., 1.]), np.array([1., 0.]), np.array([.4, .2]), .001
        h = theta - target
        actual = .5 * np.sum((theta - target) ** 2) - .5 * np.sum((theta - eta * g - target) ** 2)
        expected = eta * (h @ g) - .5 * eta ** 2 * (g @ g)
        self.assertAlmostEqual(actual, expected)
        self.assertGreater(actual, 0)

    def test_conflict_and_useful_negative_advantage_reporting(self):
        rows = [{"score_dtv": 1., "score_loo": [-1., 1., 0., 1.][i],
                 "utility_policy": [1., -1., 0., 2.][i], "utility_full": [2., -.5, .1, 2.][i],
                 "policy_gradient_norm": [.5, .5, 0., 2.][i], "prompt_id": "one_prompt",
                 "keep_dtv": True, "keep_loo": i != 0, "noise_selected": True,
                 "noise_effective": True, "correct": i == 0, "group_correct_count": 1,
                 "unique_correct": i == 0, "advantage": [-1., -.5, 0., 1.][i]}
                for i in range(4)]
        report = m.unit_summary(rows, bootstrap_repeats=0)
        self.assertEqual(report["conflict_actual"]["units"], 1)
        self.assertEqual(report["conflict_actual"]["positive_policy_utility_fraction"], 1.)
        self.assertEqual(report["conflict_actual"]["useful_negative_advantage_units"], 1)
        self.assertEqual(report["loo"]["dropped_fraction_by_stratum"]["unique_correct_in_observed_group"]["drop_fraction"], 1.)

    def test_empty_conflict_is_na_not_a_fabricated_zero_mean(self):
        rows = [{"score_dtv": 0., "score_loo": 0., "utility_policy": 0., "utility_full": 0.,
                 "policy_gradient_norm": 0., "prompt_id": "one_prompt", "keep_dtv": True,
                 "keep_loo": True, "noise_selected": False, "noise_effective": False,
                 "correct": False, "group_correct_count": 0, "unique_correct": False, "advantage": 0.}
                for _ in range(4)]
        report = m.unit_summary(rows, bootstrap_repeats=0)
        self.assertEqual(report["conflict_actual"]["units"], 0)
        self.assertIsNone(report["conflict_actual"]["mean_policy_utility"])
        self.assertIsNone(report["dtv"]["spearman_score_policy_utility_all"])


class UtilityCliTest(unittest.TestCase):
    def args(self, *extra):
        return parser().parse_args(["--input-dir", "/unused/in", "--output-dir", "/unused/out",
                                   "--reward-mode", "accuracy", *extra])

    def test_frozen_dimensions_and_smoke_budget(self):
        args = validate(self.args("--mode", "smoke"))
        self.assertEqual((args.candidate_prompts, args.heldout_prompts), (4, 8))
        self.assertEqual((args.beta, args.completion_length, args.lora_rank), (.08, 768, 64))

    def test_reward_branch_is_explicit(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser().parse_args(["--input-dir", "x", "--output-dir", "y"])

    def test_invalid_configs(self):
        for arguments in (("--candidate-prompts", "3"), ("--candidate-noise", "1.1"),
                          ("--probe-step-sizes", "nan"), ("--finite-random-repeats", "99"),
                          ("--heldout-prompts", "1000"), ("--beta", "-1")):
            with self.assertRaises(ValueError):
                validate(self.args(*arguments))

    def test_preflight_no_jax_and_not_utility_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "in"
            checkpoint = source / "checkpoints/actor/500"
            params = checkpoint / "model_params"
            (params / "d").mkdir(parents=True)
            (params / "d/shard").write_bytes(b"fake payload for metadata test only")
            (params / "_METADATA").write_text("{}")
            (checkpoint / "_CHECKPOINT_METADATA").write_text("{}")
            (source / "git-commit.txt").write_text("a" * 40)
            command = ("import sys; from my_example.rebuttal.audit_grpo_utility import main; "
                       "main(sys.argv[1:]); assert 'jax' not in sys.modules")
            result = subprocess.run([sys.executable, "-c", command, "--input-dir", str(source),
                                     "--output-dir", str(root / "out"), "--reward-mode", "accuracy",
                                     "--expected-commit", "a" * 40], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / "out/audit_summary.json").read_text())
            self.assertTrue(report["execution_passed"])
            self.assertFalse(report["utility_audit_completed"])
            self.assertFalse(report["optimizer_replayed"])

    def test_no_training_or_checkpoint_writes_in_runtime(self):
        path = Path(__file__).parents[1] / "utility_runtime.py"
        tree = ast.parse(path.read_text())
        calls = [node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn("train", calls)
        self.assertNotIn("save", calls)
        self.assertNotIn("update_params", calls)


if __name__ == "__main__":
    unittest.main()
