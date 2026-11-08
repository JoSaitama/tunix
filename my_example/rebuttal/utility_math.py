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
        summaries[method] = {
            "units": len(rows), "retained_fraction": fraction(kept),
            "spearman_score_policy_utility_all": spearman(score, pg),
            "spearman_score_full_utility_all": spearman(score, full),
            "spearman_non_near_zero": spearman(score[active], pg[active]),
            "spearman_gradient_norm_policy_utility": spearman(
                np.array([row["policy_gradient_norm"] for row in rows]), pg),
            "score_utility_prompt_cluster_interval": correlation_interval(
                score, pg, [row["prompt_id"] for row in rows], repeats=bootstrap_repeats),
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
                "units": int(mask.sum()), "drop_fraction": fraction(~kept[mask])}
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
