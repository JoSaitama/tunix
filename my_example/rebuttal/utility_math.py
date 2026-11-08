"""CPU statistics and controls for the independent checkpoint utility audit.

No JAX imports, models, hidden-label selection or projected gradients here.
"""

from __future__ import annotations

import math
import numpy as np


def scores_from_gram(gram, group_size=4, scope="group"):
    """Independent mathematical oracle for production score reconstruction."""
    gram = np.asarray(gram, dtype=np.float64)
    n = len(gram)
    if gram.shape != (n, n) or not np.isfinite(gram).all() or not n:
        raise ValueError("expected a finite nonempty square Gram matrix")
    if group_size <= 1 or n % group_size or scope not in {"group", "batch"}:
        raise ValueError("invalid group size or scope")
    totals = gram.sum(axis=1) if scope == "batch" else np.concatenate([
        gram[s:s + group_size, s:s + group_size].sum(axis=1)
        for s in range(0, n, group_size)
    ])
    denominator = n if scope == "batch" else group_size
    raw_self = np.diag(gram)
    return {"dtv": totals / denominator,
            "loo": (totals - raw_self) / (denominator - 1),
            "self": raw_self / denominator,
            "cross": (totals - raw_self) / denominator}


def selection_masks(dtv, loo, *, scope="group", group_size=4, min_keep_fraction=.25):
    dtv, loo = np.asarray(dtv), np.asarray(loo)
    if dtv.shape != loo.shape or dtv.ndim != 1 or not len(dtv):
        raise ValueError("scores must be nonempty matching vectors")
    if not np.isfinite(dtv).all() or not np.isfinite(loo).all():
        raise ValueError("nonfinite scores: no silent selection fallback")
    if scope not in {"group", "batch"} or group_size <= 1 or len(dtv) % group_size:
        raise ValueError("invalid scope or group size")
    if not 0 <= min_keep_fraction <= 1:
        raise ValueError("invalid retention cap")
    threshold = loo >= 0
    actual = threshold.copy()
    cap = np.zeros_like(actual)
    width = group_size if scope == "group" else len(dtv)
    for start in range(0, len(dtv), width):
        block = actual[start:start + width]
        # 0 explicitly requests threshold-only diagnostics, not production cap.
        minimum = max(1, math.ceil(width * min_keep_fraction)) if min_keep_fraction else 0
        if block.sum() < minimum:
            order = np.argsort(-loo[start:start + width], kind="stable")
            block[:] = False
            block[order[:minimum]] = True
            cap[start:start + width] = block & ~threshold[start:start + width]
    return {"vanilla": np.ones_like(actual), "dtv": dtv >= 0, "loo": actual,
            "loo_threshold": threshold, "loo_cap_rescue": cap}


def matched_random(mask, *, scope, group_size, seed, batch_id, repeat):
    mask = np.asarray(mask, dtype=bool)
    if scope not in {"group", "batch"} or mask.ndim != 1 or len(mask) % group_size:
        raise ValueError("invalid matching scope")
    result = np.zeros_like(mask)
    width = group_size if scope == "group" else len(mask)
    rng = np.random.default_rng(np.random.SeedSequence([seed, batch_id, repeat]))
    for start in range(0, len(mask), width):
        count = int(mask[start:start + width].sum())
        result[start + rng.choice(width, size=count, replace=False)] = True
    return result


def ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2
        start = end
    return result


def spearman(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("correlation expects matching vectors")
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    rx, ry = ranks(x), ranks(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def correlation_interval(x, y, clusters, *, seed=0, repeats=200):
    x, y, clusters = np.asarray(x), np.asarray(y), np.asarray(clusters)
    if x.shape != y.shape or x.shape != clusters.shape:
        raise ValueError("invalid clustered correlation arrays")
    ids = np.unique(clusters)
    if len(ids) < 2 or repeats < 2:
        return {"ci95": None, "valid_bootstrap_draws": 0}
    blocks = [np.flatnonzero(clusters == key) for key in ids]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        indices = np.concatenate([blocks[i] for i in rng.integers(0, len(ids), len(ids))])
        value = spearman(x[indices], y[indices])
        if value is not None:
            draws.append(value)
    return {"ci95": np.quantile(draws, [.025, .975]).tolist() if len(draws) >= repeats / 2 else None,
            "valid_bootstrap_draws": len(draws)}


def fraction(values):
    values = np.asarray(values)
    return float(np.mean(values)) if values.size else None


def cluster_interval(values, clusters, *, seed=0, repeats=200):
    """Percentile CI of mean, resampling whole candidate prompt/batch clusters.

    Conditional on this checkpoint AND the estimated heldout gradient. This is
    not a training-seed CI or a CI accounting for heldout-gradient uncertainty.
    """
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    if values.ndim != 1 or values.shape != clusters.shape or not np.isfinite(values).all():
        raise ValueError("invalid values/clusters")
    ids = np.unique(clusters)
    if len(ids) < 2 or repeats < 2:
        return None
    blocks = [values[clusters == key] for key in ids]
    rng = np.random.default_rng(seed)
    draws = [float(np.concatenate([blocks[i] for i in rng.integers(0, len(ids), len(ids))]).mean())
             for _ in range(repeats)]
    return np.quantile(draws, [.025, .975]).tolist()


def unit_summary(rows, epsilon=1e-12, bootstrap_repeats=200):
    summaries = {}
    for method in ("dtv", "loo"):
        score = np.array([row[f"score_{method}"] for row in rows])
        pg = np.array([row["utility_policy"] for row in rows])
        full = np.array([row["utility_full"] for row in rows])
        kept = np.array([row[f"keep_{method}"] for row in rows], dtype=bool)
        active = (np.abs(score) > epsilon) & (np.abs(pg) > epsilon)
        gradient_active = np.array([row["policy_gradient_norm"] > epsilon for row in rows])
        clusters = [row["prompt_id"] for row in rows]
        summaries[method] = {
            "units": len(rows), "retained_fraction": fraction(kept),
            "spearman_score_policy_utility_all": spearman(score, pg),
            "spearman_score_full_utility_all": spearman(score, full),
            "spearman_non_near_zero": spearman(score[active], pg[active]),
            "non_near_zero_definition": "conditional on BOTH nonzero score and utility; sensitivity only",
            "nonzero_gradient_units": int(gradient_active.sum()),
            "spearman_nonzero_gradient": spearman(score[gradient_active], pg[gradient_active]),
            "nonzero_gradient_definition": "policy_gradient_norm > epsilon, NOT selected by utility",
            "spearman_gradient_norm_policy_utility": spearman(
                np.array([row["policy_gradient_norm"] for row in rows]), pg),
            "score_utility_prompt_cluster_interval": correlation_interval(
                score, pg, clusters, repeats=bootstrap_repeats),
            "nonzero_gradient_prompt_cluster_interval": correlation_interval(
                score[gradient_active], pg[gradient_active], np.asarray(clusters)[gradient_active],
                repeats=bootstrap_repeats),
            "non_near_zero_units": int(active.sum()),
            "near_zero_score_non_near_zero_policy_utility_units": int(
                ((np.abs(score) <= epsilon) & (np.abs(pg) > epsilon)).sum()),
            "nonzero_advantage_zero_policy_gradient_units": sum(
                row["advantage"] != 0 and row["policy_gradient_norm"] <= epsilon for row in rows),
            "positive_policy_utility_kept": fraction(pg[kept] > epsilon),
            "positive_policy_utility_dropped": fraction(pg[~kept] > epsilon),
            "mean_policy_utility_kept": fraction(pg[kept]),
            "mean_policy_utility_dropped": fraction(pg[~kept]),
            "dropped_fraction_by_stratum": {},
        }
        for half in ("a", "b"):
            key = f"utility_half_{half}"
            if rows and all(row.get(key) is not None for row in rows):
                target = np.array([row[key] for row in rows])
                summaries[method][f"spearman_heldout_half_{half}"] = spearman(score, target)
                summaries[method][f"mean_half_{half}_utility_kept"] = fraction(target[kept])
                summaries[method][f"mean_half_{half}_utility_dropped"] = fraction(target[~kept])
        strata = {
            "noise_selected": [row["noise_selected"] for row in rows],
            "noise_effective": [row["noise_effective"] for row in rows],
            "not_noise_selected": [not row["noise_selected"] for row in rows],
            "correct": [row["correct"] for row in rows],
            "incorrect": [not row["correct"] for row in rows],
            "advantage_positive": [row["advantage"] > 0 for row in rows],
            "advantage_negative": [row["advantage"] < 0 for row in rows],
            "advantage_zero": [row["advantage"] == 0 for row in rows],
            "unique_correct_in_observed_group": [row["unique_correct"] for row in rows],
            "useful_negative_advantage": [row["advantage"] < 0 and row["utility_policy"] > epsilon
                                          for row in rows],
        }
        for name, mask in strata.items():
            mask = np.asarray(mask, dtype=bool)
            summaries[method]["dropped_fraction_by_stratum"][name] = {
                "units": int(mask.sum()), "prompts": len(set(np.asarray(clusters)[mask])),
                "dropped_units": int((~kept[mask]).sum()),
                "drop_fraction": fraction(~kept[mask])}
        summaries[method]["noise_advantage_joint"] = []
        for noise_key in ("noise_selected", "noise_effective"):
            for noisy in (False, True):
                for sign, condition in (("negative", lambda a: a < 0), ("zero", lambda a: a == 0),
                                        ("positive", lambda a: a > 0)):
                    mask = np.array([bool(row[noise_key]) == noisy and condition(row["advantage"])
                                     for row in rows], dtype=bool)
                    summaries[method]["noise_advantage_joint"].append({
                        "noise_definition": noise_key, "corrupted": noisy, "advantage_sign": sign,
                        "units": int(mask.sum()), "prompts": len(set(np.asarray(clusters)[mask])),
                        "dropped_units": int((~kept[mask]).sum()),
                        "drop_fraction": fraction(~kept[mask]),
                        "mean_policy_utility": fraction(pg[mask]),
                        "correct_units": sum(row["correct"] for row, yes in zip(rows, mask) if yes)})
        for count in range(5):
            mask = np.array([row["group_correct_count"] == count for row in rows])
            summaries[method]["dropped_fraction_by_stratum"][f"observed_group_correct_{count}_of_4"] = {
                "units": int(mask.sum()), "drop_fraction": fraction(~kept[mask])}
    conflict = [row for row in rows if row["keep_dtv"] and not row["keep_loo"]]
    summaries["conflict_actual"] = {
        "units": len(conflict), "fraction_all": len(conflict) / len(rows) if rows else None,
        "positive_policy_utility_fraction": fraction([row["utility_policy"] > epsilon for row in conflict]),
        "mean_policy_utility": fraction([row["utility_policy"] for row in conflict]),
        "unique_correct_units": sum(row["unique_correct"] for row in conflict),
        "useful_negative_advantage_units": sum(row["advantage"] < 0 and row["utility_policy"] > epsilon
                                               for row in conflict),
    }
    return summaries


def reference_summary(rows, *, epsilon=1e-12, repeats=200):
    valid = [row for row in rows if row["informative"]]
    cosine = [row["cosine"] for row in valid]
    ids = [row.get("prompt_id", f"{row['batch_id']}:{row['scope_index']}") for row in valid]
    result = {"references": len(rows), "informative_references": len(valid),
              "zero_or_uninformative_references": len(rows) - len(valid),
              "mean_cosine_informative": fraction(cosine),
              "mean_cosine_cluster_ci95": cluster_interval(cosine, ids, repeats=repeats),
              "positive_alignment_fraction_informative": fraction([x > 0 for x in cosine]),
              "positive_fraction_cluster_ci95": cluster_interval([x > 0 for x in cosine], ids, repeats=repeats),
              "uncertainty": "candidate prompt/reference clusters, CONDITIONAL on estimated heldout h"}
    if rows and all("dot_with_heldout" in row for row in rows):
        all_ids = [row.get("prompt_id", f"{row['batch_id']}:{row['scope_index']}") for row in rows]
        dots = [row["dot_with_heldout"] for row in rows]
        result["mean_reference_dot_clean_including_zeros"] = fraction(dots)
        result["mean_reference_dot_cluster_ci95"] = cluster_interval(dots, all_ids, repeats=repeats)
    result["noise_strata"] = {}
    for definition in ("noise_selected", "noise_effective"):
        for noisy in (False, True):
            group = [row for row in rows if row.get(definition) is not None and row[definition] == noisy]
            usable = [row for row in group if row["informative"]]
            result["noise_strata"][f"{definition}_{noisy}"] = {
                "references": len(group), "informative_references": len(usable),
                "mean_dot_clean": fraction([row["dot_with_heldout"] for row in group]),
                "mean_cosine": fraction([row["cosine"] for row in usable]),
                "positive_alignment_fraction_informative": fraction([row["cosine"] > 0 for row in usable])}
    for half in ("a", "b"):
        values = [row[f"cosine_half_{half}"] for row in rows if row.get(f"cosine_half_{half}") is not None]
        result[f"mean_cosine_half_{half}"] = fraction(values)
    both = [row for row in rows if row.get("dot_half_a") is not None and row.get("dot_half_b") is not None]
    active = [row for row in both if abs(row["dot_half_a"]) > epsilon and abs(row["dot_half_b"]) > epsilon]
    result["half_sign_comparison_references"] = len(active)
    result["half_sign_agreement"] = fraction([np.sign(row["dot_half_a"]) == np.sign(row["dot_half_b"])
                                             for row in active])
    return result


def direction_summary(rows, *, repeats=200):
    """Paired across batches; random repetitions averaged WITHIN each batch."""
    methods = sorted({row["method"] for row in rows})
    result = {}
    for name in methods:
        group = [row for row in rows if row["method"] == name]
        values = [row["direction_dot_clean_gradient"] for row in group]
        result[name] = {"batches": len(group), "mean_dot_clean_gradient": fraction(values),
                        "mean_retained_fraction": fraction([row["retained_fraction"] for row in group]),
                        "batch_cluster_ci95": cluster_interval(values, [row["batch_id"] for row in group],
                                                               repeats=repeats)}
    for left, right in (("dtv", "loo"), ("dtv", "vanilla"), ("loo", "vanilla"),
                        ("dtv", "random_match_dtv"), ("loo", "random_match_loo")):
        a = {row["batch_id"]: row["direction_dot_clean_gradient"] for row in rows if row["method"] == left}
        b = {}
        for row in rows:
            if row["method"] == right or row["method"].startswith(right + "_"):
                b.setdefault(row["batch_id"], []).append(row["direction_dot_clean_gradient"])
        ids = sorted(a.keys() & b.keys())
        values = [a[key] - float(np.mean(b[key])) for key in ids]
        result[f"paired_{left}_minus_{right}"] = {
            "batches": len(ids), "mean": fraction(values),
            "batch_cluster_ci95": cluster_interval(values, ids, repeats=repeats)}
    return result


def numerical_pair_check(plus, minus, *, floor, rtol=.1, atol=1e-7):
    """Prespecified self-gradient +/- probe gate, not a method outcome test."""
    expected = [plus["predicted_first_order_utility"], minus["predicted_first_order_utility"]]
    actual = [plus["utility"], minus["utility"]]
    resolvable = all(abs(x) > 10 * floor for x in expected)
    finite = all(np.isfinite(x) for x in expected + actual)
    accurate = finite and all(abs(a - p) <= atol + rtol * abs(p) for a, p in zip(actual, expected))
    signs = expected[0] > 0 and expected[1] < 0 and actual[0] > 0 and actual[1] < 0
    return {"passed": bool(resolvable and accurate and signs), "resolvable": bool(resolvable),
            "taylor_within_tolerance": bool(accurate), "expected_and_actual_signs_correct": bool(signs),
            "max_abs_residual": float(max(abs(a - p) for a, p in zip(actual, expected))) if finite else None,
            "rtol": rtol, "atol": atol, "resolution_floor": floor}


def assumption_evidence(reference, units, heldout):
    """Descriptive evidence, NEVER an automatic certificate or method choice."""
    def positive_interval(interval):
        return interval is not None and interval[0] > 0

    def interval_result(interval):
        if interval is None:
            return "insufficient_clusters_or_undefined"
        if interval[0] > 0:
            return "positive_conditional_interval"
        if interval[1] < 0:
            return "negative_conditional_interval"
        return "interval_includes_zero"

    stable = heldout.get("split_half", {}).get("positive_alignment", False)
    result = {"assumption_proven": False, "method_selection_rule_validated": False,
              "scope": "local independent clean policy-gradient surrogate at this fixed checkpoint",
              "local_informativeness_condition": "<E[r_observed], gradient F_clean> > 0; empirical, conditional estimate",
              "heldout_split_half_positive_alignment": stable,
              "reference_result": interval_result(reference.get("mean_reference_dot_cluster_ci95")),
              "heldout_caveat": "interpret alignment cautiously if split halves are unavailable/nonpositive",
              "mean_reference_dot_ci_entirely_positive": positive_interval(
                  reference.get("mean_reference_dot_cluster_ci95")),
              "reference_mean_cosine_ci_entirely_positive": positive_interval(
                  reference.get("mean_cosine_cluster_ci95")),
              "reference_majority_positive_ci_above_half": (
                  reference.get("positive_fraction_cluster_ci95") is not None and
                  reference["positive_fraction_cluster_ci95"][0] > .5),
              "method_evidence": {}}
    for name in ("dtv", "loo"):
        data = units[name]
        ci = data["score_utility_prompt_cluster_interval"]["ci95"]
        halves = [data.get(f"spearman_heldout_half_{half}") for half in ("a", "b")]
        result["method_evidence"][name] = {
            "score_result": interval_result(ci),
            "score_utility_ci_entirely_positive": positive_interval(ci),
            "score_utility_rho": data["spearman_score_policy_utility_all"],
            "score_utility_ci95": ci,
            "both_half_correlations_positive": all(x is not None and x > 0 for x in halves),
            "kept_mean_policy_utility": data["mean_policy_utility_kept"],
            "dropped_mean_policy_utility": data["mean_policy_utility_dropped"]}
    result["interpretation"] = (
        "All intervals condition on estimated h. Split-half is sensitivity, not a calibrated accuracy guarantee. "
        "Report negative/zero/undefined results; this is not an IF ranking theorem or proof of long-term return. "
        "Whole-group synchronous gradient reversal can preserve internal scores while reversing external utility.")
    return result
