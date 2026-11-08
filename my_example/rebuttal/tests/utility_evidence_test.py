"""Server CPU evidence/gate tests. No checkpoint or model imports."""

import unittest
import numpy as np

from my_example.rebuttal import utility_math as m


class UtilityEvidenceTest(unittest.TestCase):
    def rows(self):
        return [{"prompt_id": f"p{i // 4}", "batch_id": i // 16,
                 "policy_gradient_norm": 0. if i == 0 else 1.,
                 "score_dtv": float(i), "score_loo": float(i - 5),
                 "utility_policy": float(i - 5), "utility_full": float(i - 5),
                 "utility_half_a": float(i - 5), "utility_half_b": float(i - 4),
                 "keep_dtv": True, "keep_loo": i >= 5,
                 "noise_selected": i // 4 % 2 == 0, "noise_effective": i // 4 % 2 == 0,
                 "correct": i % 4 == 0, "unique_correct": i % 4 == 0,
                 "group_correct_count": 1, "advantage": [-1., 0., .5, 1.][i % 4]}
                for i in range(16)]

    def test_nonzero_gradient_subset_does_not_select_on_target_or_score(self):
        report = m.unit_summary(self.rows(), bootstrap_repeats=0)
        self.assertEqual(report["dtv"]["nonzero_gradient_units"], 15)
        self.assertLess(report["dtv"]["non_near_zero_units"], 15)
        self.assertAlmostEqual(report["dtv"]["spearman_nonzero_gradient"], 1.)

    def test_noise_advantage_joint_denominators_are_counts_not_group_drop_flags(self):
        report = m.unit_summary(self.rows(), bootstrap_repeats=0)["loo"]
        joint = report["noise_advantage_joint"]
        for definition in ("noise_selected", "noise_effective"):
            subset = [row for row in joint if row["noise_definition"] == definition]
            self.assertEqual(sum(row["units"] for row in subset), 16)
            self.assertEqual(sum(row["dropped_units"] for row in subset), 5)
            self.assertTrue(all(row["prompts"] <= 4 for row in subset))

    def test_reference_summary_counts_zeros_and_reports_conditional_intervals(self):
        rows = [{"batch_id": 0, "scope_index": i, "informative": i != 3,
                 "cosine": [1., .5, -.5, None][i], "dot_half_a": 1., "dot_half_b": -1.,
                 "cosine_half_a": 1., "cosine_half_b": -1.} for i in range(4)]
        report = m.reference_summary(rows, repeats=20)
        self.assertEqual(report["zero_or_uninformative_references"], 1)
        self.assertAlmostEqual(report["positive_alignment_fraction_informative"], 2 / 3)
        self.assertEqual(report["half_sign_agreement"], 0.)
        self.assertIn("CONDITIONAL", report["uncertainty"])

    def test_paired_random_controls_average_repeats_within_batch(self):
        rows = [{"batch_id": b, "method": name, "direction_dot_clean_gradient": value,
                 "retained_fraction": .5} for b in (0, 1) for name, value in (
                     ("vanilla", 1.), ("dtv", 2.), ("loo", 1.5),
                     ("random_match_dtv_0", 0.), ("random_match_dtv_1", 2.),
                     ("random_match_loo_0", 1.))]
        report = m.direction_summary(rows, repeats=20)
        pair = report["paired_dtv_minus_random_match_dtv"]
        self.assertEqual(pair["batches"], 2)
        self.assertEqual(pair["mean"], 1.)
        self.assertEqual(pair["batch_cluster_ci95"], [1., 1.])

    def test_numerical_gate_rejects_thousandfold_error_and_wrong_sign(self):
        plus = {"utility": 1e-4, "predicted_first_order_utility": 1e-4}
        minus = {"utility": -1e-4, "predicted_first_order_utility": -1e-4}
        self.assertTrue(m.numerical_pair_check(plus, minus, floor=1e-7)["passed"])
        for utility in (.1, -1e-4, float("nan")):
            self.assertFalse(m.numerical_pair_check(dict(plus, utility=utility), minus, floor=1e-7)["passed"])

    def test_unresolvable_probe_is_not_called_pass(self):
        plus = {"utility": 1e-10, "predicted_first_order_utility": 1e-10}
        minus = {"utility": -1e-10, "predicted_first_order_utility": -1e-10}
        report = m.numerical_pair_check(plus, minus, floor=1e-7)
        self.assertFalse(report["passed"])
        self.assertFalse(report["resolvable"])

    def test_positive_correlations_do_not_prove_assumption_or_choose_method(self):
        units = m.unit_summary(self.rows(), bootstrap_repeats=20)
        report = m.assumption_evidence({"mean_cosine_cluster_ci95": [.1, .5],
                                       "positive_fraction_cluster_ci95": [.6, .9]}, units,
                                      {"split_half": {"positive_alignment": True}})
        self.assertTrue(report["reference_mean_cosine_ci_entirely_positive"])
        self.assertFalse(report["assumption_proven"])
        self.assertFalse(report["method_selection_rule_validated"])

    def test_positive_reference_need_not_make_filtering_beneficial(self):
        g = np.array([[10., -5.], [10., -5.], [-8., 6.], [-8., 6.]])
        h = np.array([0., 1.])
        score = m.scores_from_gram(g @ g.T)
        mask = m.selection_masks(score["dtv"], score["loo"])["dtv"]
        self.assertGreater(g.mean(0) @ h, 0.)
        self.assertLess(g[mask].mean(0) @ h, 0.)
        self.assertGreater(np.linalg.norm(g[mask].mean(0)), np.linalg.norm(g.mean(0)))


if __name__ == "__main__":
    unittest.main()
