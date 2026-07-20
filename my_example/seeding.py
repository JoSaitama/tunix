"""Experiment seed helpers that preserve the legacy defaults."""

from __future__ import annotations

import os


EXPERIMENT_SEED_ENV = "TUNIX_EXPERIMENT_SEED"
LEGACY_DATASET_SEED = 42


def experiment_seed() -> int | None:
    """Returns an explicit experiment seed, or None for legacy behavior."""
    raw_value = os.environ.get(EXPERIMENT_SEED_ENV)
    if raw_value is None or not raw_value.strip():
        return None

    try:
        seed = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{EXPERIMENT_SEED_ENV} must be an integer; got {raw_value!r}."
        ) from exc

    if seed < 0:
        raise ValueError(
            f"{EXPERIMENT_SEED_ENV} must be nonnegative; got {seed}."
        )
    return seed


def dataset_shuffle_seed() -> int:
    seed = experiment_seed()
    return LEGACY_DATASET_SEED + (seed or 0)


def seed_summary() -> str:
    seed = experiment_seed()
    if seed is None:
        return "legacy(data_shuffle=42, rollout=implicit-0)"
    return f"{seed}(data_shuffle={dataset_shuffle_seed()}, rollout={seed})"
