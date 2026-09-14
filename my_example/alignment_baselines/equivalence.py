"""Numerical checks for LearnAlign grouped gradient evaluation."""

from __future__ import annotations

import math

import numpy as np

from .scoring import learnalign_scores, stable_top_indices


def feature_execution_mode(
    *, equivalence_enabled: bool, grouped_feature_estimation: bool
) -> str:
    """Returns one mutually exclusive estimator execution mode."""
    if equivalence_enabled:
        return "dual"
    if grouped_feature_estimation:
        return "grouped"
    return "legacy"


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size != left.size:
        return None
    left_ranks = _rankdata(left)
    right_ranks = _rankdata(right)
    if np.std(left_ranks) == 0.0 or np.std(right_ranks) == 0.0:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def compare_learnalign_feature_paths(
    legacy_features: np.ndarray,
    grouped_features: np.ndarray,
    *,
    selection_ratio: int,
    score_weights: np.ndarray | None = None,
    acceptance_mode: str = "strict",
    rtol: float = 1e-4,
    atol: float = 1e-5,
    min_row_cosine: float = 0.99999,
    min_score_spearman: float = 0.99999,
) -> dict[str, object]:
    """Compares per-completion and grouped-loss feature evaluation.

    By default deterministic non-uniform weights are used for the downstream
    LearnAlign score comparison.  A caller may instead supply the actual
    learnability weights. ``selected-set`` acceptance requires only identical
    top-k prompt indices; the numerical diagnostics remain in the report.
    """
    legacy = np.asarray(legacy_features, dtype=np.float64)
    grouped = np.asarray(grouped_features, dtype=np.float64)
    if legacy.shape != grouped.shape or legacy.ndim != 2:
        raise ValueError(
            "feature paths must have the same [prompt, projection] shape; "
            f"got {legacy.shape} and {grouped.shape}"
        )
    if legacy.shape[0] == 0:
        raise ValueError("at least one prompt feature is required")
    if selection_ratio <= 1:
        raise ValueError("selection_ratio must be greater than one")
    if acceptance_mode not in {"strict", "selected-set"}:
        raise ValueError(f"unsupported acceptance mode: {acceptance_mode}")

    difference = grouped - legacy
    max_abs = float(np.max(np.abs(difference)))
    reference_scale = float(np.max(np.abs(legacy)))
    max_relative = max_abs / max(reference_scale, np.finfo(np.float64).tiny)
    feature_allclose = bool(np.allclose(grouped, legacy, rtol=rtol, atol=atol))

    legacy_norms = np.linalg.norm(legacy, axis=1)
    grouped_norms = np.linalg.norm(grouped, axis=1)
    both_zero = (legacy_norms == 0.0) & (grouped_norms == 0.0)
    denominator = legacy_norms * grouped_norms
    row_cosines = np.divide(
        np.sum(legacy * grouped, axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    row_cosines[both_zero] = 1.0
    minimum_cosine = float(np.min(row_cosines))

    prompt_count = legacy.shape[0]
    if score_weights is None:
        weights = np.linspace(0.125, 0.25, prompt_count, dtype=np.float64)
        score_weights_label = "deterministic_nonuniform_test_weights"
    else:
        weights = np.asarray(score_weights, dtype=np.float64).reshape(-1)
        if weights.shape != (prompt_count,):
            raise ValueError(
                "score_weights must have one value per prompt; "
                f"got {weights.shape} for {prompt_count} prompts"
            )
        score_weights_label = "provided"
    legacy_scores = learnalign_scores(legacy, weights)
    grouped_scores = learnalign_scores(grouped, weights)
    score_difference = grouped_scores - legacy_scores
    score_max_abs = float(np.max(np.abs(score_difference)))
    score_scale = float(np.max(np.abs(legacy_scores)))
    score_max_relative = score_max_abs / max(
        score_scale, np.finfo(np.float64).tiny
    )
    score_spearman = _spearman(legacy_scores, grouped_scores)

    selected_count = max(1, math.ceil(prompt_count / selection_ratio))
    legacy_selected = set(
        int(value) for value in stable_top_indices(legacy_scores, selected_count)
    )
    grouped_selected = set(
        int(value)
        for value in stable_top_indices(grouped_scores, selected_count)
    )
    selected_union = legacy_selected | grouped_selected
    selected_jaccard = (
        len(legacy_selected & grouped_selected) / len(selected_union)
        if selected_union
        else 1.0
    )

    spearman_passed = (
        score_spearman is None
        or score_spearman >= min_score_spearman
    )
    strict_passed = bool(
        feature_allclose
        and minimum_cosine >= min_row_cosine
        and spearman_passed
        and selected_jaccard == 1.0
    )
    passed = (
        selected_jaccard == 1.0
        if acceptance_mode == "selected-set"
        else strict_passed
    )
    return {
        "passed": passed,
        "acceptance_mode": acceptance_mode,
        "strict_checks_passed": strict_passed,
        "prompt_count": prompt_count,
        "projection_dim": legacy.shape[1],
        "feature_allclose": feature_allclose,
        "feature_rtol": rtol,
        "feature_atol": atol,
        "feature_max_abs_error": max_abs,
        "feature_max_relative_error": max_relative,
        "minimum_row_cosine": minimum_cosine,
        "minimum_required_row_cosine": min_row_cosine,
        "score_max_abs_error": score_max_abs,
        "score_max_relative_error": score_max_relative,
        "score_spearman": score_spearman,
        "minimum_required_score_spearman": min_score_spearman,
        "selected_count": selected_count,
        "selected_jaccard": selected_jaccard,
        "legacy_selected_indices": sorted(legacy_selected),
        "grouped_selected_indices": sorted(grouped_selected),
        "score_weights": score_weights_label,
    }
