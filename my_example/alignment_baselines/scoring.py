"""Small, testable scoring functions used by LearnAlign and GradAlign."""

from __future__ import annotations

import numpy as np


def learnability(successes: np.ndarray) -> np.ndarray:
    """Returns LearnAlign's V(x)=p(x)(1-p(x)) for [N, G] outcomes."""
    values = np.asarray(successes, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"successes must have shape [N, G]; got {values.shape}")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("successes must be binary or probabilities in [0, 1]")
    p = values.mean(axis=1)
    return p * (1.0 - p)


def normalize_rows(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"features must have shape [N, D]; got {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, np.maximum(norms, eps))


def learnalign_scores(
    features: np.ndarray,
    learnability_values: np.ndarray,
) -> np.ndarray:
    """Computes the exact row mean of Eq. 8 without materializing NxN.

    If z_i = V_i phi_i / ||phi_i||, then
    mean_j S_ij = z_i^T mean_j(z_j).
    """
    unit = normalize_rows(features)
    weights = np.asarray(learnability_values, dtype=np.float64)
    if weights.shape != (unit.shape[0],):
        raise ValueError(
            f"learnability shape must be {(unit.shape[0],)}; got {weights.shape}"
        )
    weighted = unit * weights[:, None]
    return weighted @ weighted.mean(axis=0)


def cosine_scores(
    features: np.ndarray,
    reference_gradient: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Cosine alignment of candidate gradients with a reference gradient."""
    unit = normalize_rows(features, eps=eps)
    ref = np.asarray(reference_gradient, dtype=np.float64).reshape(-1)
    if ref.shape[0] != unit.shape[1]:
        raise ValueError(
            f"reference dimension {ref.shape[0]} != feature dimension {unit.shape[1]}"
        )
    ref_norm = np.linalg.norm(ref)
    if ref_norm <= eps:
        return np.zeros(unit.shape[0], dtype=np.float64)
    return unit @ (ref / ref_norm)


def stable_top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    """Returns deterministic top-score indices, breaking ties by source order."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not 0 < count <= values.size:
        raise ValueError(f"count must be in [1, {values.size}]; got {count}")
    safe = np.nan_to_num(values, nan=-np.inf)
    order = np.lexsort((np.arange(values.size), -safe))
    return order[:count]
