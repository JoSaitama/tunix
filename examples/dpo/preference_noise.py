"""Preference-data corruption utilities for DPO experiments.

This module keeps synthetic preference noise transforms in one place so they
can be reused by UltraFeedback, HH-RLHF, SHP, and future DPO datasets.

Main transform:
  - prompt-response mismatch:
      select a corruption subset C, keep prompt_i, but replace the
      chosen/rejected response pair with responses from another corrupted
      example j in C.

This simulates data transmission, shuffle, or join errors where the prompt and
responses no longer match, while still keeping a valid DPO pair structure.

Important design choice:
  Corruption is performed within the selected corruption subset only.
  Clean examples outside the subset are never used as source examples.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import numpy as np


_REQUIRED_COLUMNS = ("prompt", "chosen", "rejected")


def _validate_ratio(name: str, ratio: float) -> None:
  """Validate a corruption ratio."""
  if ratio < 0.0 or ratio > 1.0:
    raise ValueError(f"{name} must be in [0, 1], got {ratio!r}.")


def _select_indices(dataset_size: int, ratio: float, seed: int) -> list[int]:
  """Select deterministic corruption indices."""
  _validate_ratio("ratio", ratio)

  if dataset_size <= 0 or ratio <= 0.0:
    return []

  num_to_corrupt = int(dataset_size * ratio)
  if ratio > 0.0:
    num_to_corrupt = max(1, num_to_corrupt)
  num_to_corrupt = min(num_to_corrupt, dataset_size)

  rng = np.random.default_rng(seed)
  selected = rng.choice(dataset_size, size=num_to_corrupt, replace=False)
  return sorted(int(i) for i in selected)


def _get_column_names(dataset: Any) -> set[str]:
  """Return dataset column names for HuggingFace Dataset or list-like objects."""
  column_names = getattr(dataset, "column_names", None)
  if column_names is not None:
    return set(column_names)

  if len(dataset) == 0:
    return set()

  first_row = dataset[0]
  if isinstance(first_row, dict):
    return set(first_row.keys())

  return set()


def _validate_required_columns(dataset: Any) -> None:
  """Ensure the dataset has normalized DPO fields."""
  columns = _get_column_names(dataset)
  missing = [name for name in _REQUIRED_COLUMNS if name not in columns]
  if missing:
    raise KeyError(
        "Prompt-response mismatch expects a normalized DPO dataset with "
        f"columns {_REQUIRED_COLUMNS}, but missing columns are {missing}. "
        "Please normalize raw UltraFeedback, HH-RLHF, or SHP data to "
        "prompt/chosen/rejected before applying mismatch corruption."
    )


def _make_cycle_source_map(
    selected_indices: list[int],
    rng: np.random.Generator,
    *,
    shift: int = 1,
) -> dict[int, int]:
  """Map each corrupted target index to another corrupted source index.

  The mapping is a derangement over the selected corruption subset:
    - source_idx is always inside selected_indices
    - source_idx != target_idx
    - every selected source is used exactly once for a given field

  For n=2, the only possible derangement is a swap.
  """
  n = len(selected_indices)
  if n < 2:
    raise ValueError(
        "Selected-only mismatch requires at least 2 corrupted examples. "
        "Increase the ratio or use a larger dataset."
    )

  order = list(selected_indices)
  rng.shuffle(order)

  shift = shift % n
  if shift == 0:
    shift = 1

  sources = order[shift:] + order[:shift]
  source_map = {target_idx: source_idx for target_idx, source_idx in zip(order, sources)}

  for target_idx, source_idx in source_map.items():
    if target_idx == source_idx:
      raise AssertionError(
          f"Invalid derangement: target {target_idx} maps to itself."
      )

  return source_map


def apply_prompt_response_mismatch_to_hf_dataset(
    dataset: Any,
    *,
    ratio: float,
    seed: int,
    mode: str = "response_pair",
):
  """Apply selected-only prompt-response mismatch to a HuggingFace Dataset.

  Expected normalized DPO fields:
    - prompt
    - chosen
    - rejected

  Supported modes:
    response_pair:
      Select corruption subset C. For each i in C, keep prompt_i and replace
      chosen_i/rejected_i by chosen_j/rejected_j from another j in C.

    prompt:
      Select corruption subset C. For each i in C, keep chosen_i/rejected_i and
      replace prompt_i by prompt_j from another j in C.

    cross_response:
      Select corruption subset C. For each i in C, keep prompt_i and replace
      chosen_i by chosen_j and rejected_i by rejected_k, where j and k are
      selected-only deranged sources from C. If |C| = 2, j and k are necessarily
      the same source because only one non-self source exists.

    cross_response_flip:
      Select corruption subset C. For each i in C, keep prompt_i and replace
      chosen_i by rejected_j and rejected_i by chosen_k. This is a stronger
      reward-adversarial corruption because the label direction is inverted
      across the selected response pools.

  Important:
    This function should be called after raw datasets such as UltraFeedback,
    HH-RLHF, or SHP are normalized into prompt/chosen/rejected format.
  """
  _validate_ratio("ratio", ratio)

  env_mode = os.environ.get("DPO_MISMATCH_MODE")
  if env_mode:
    mode = env_mode
    print(f"[preference_noise] Using mismatch mode from DPO_MISMATCH_MODE={mode}")

  mode = mode.strip().lower().replace("-", "_")
  valid_modes = {"response_pair", "prompt", "cross_response", "cross_response_flip"}
  if mode not in valid_modes:
    raise ValueError(
        "mismatch mode must be one of response_pair, prompt, cross_response, or "
        f"cross_response_flip; got {mode!r}."
    )

  dataset_size = len(dataset)
  selected_indices = _select_indices(dataset_size, ratio, seed)
  if not selected_indices:
    return dataset

  if len(selected_indices) < 2:
    raise ValueError(
        "Selected-only mismatch selected fewer than 2 examples. "
        f"dataset_size={dataset_size}, ratio={ratio}, "
        f"selected_indices={selected_indices}. Increase ratio or dataset size."
    )

  _validate_required_columns(dataset)

  rng = np.random.default_rng(seed + 7919)

  # All source indices are from the selected corruption subset only.
  pair_source_map = _make_cycle_source_map(selected_indices, rng, shift=1)

  if mode in {"cross_response", "cross_response_flip"}:
    chosen_source_map = pair_source_map
    if len(selected_indices) >= 3:
      rejected_source_map = _make_cycle_source_map(selected_indices, rng, shift=-1)
    else:
      # With two corrupted examples, there is only one valid non-self source.
      rejected_source_map = pair_source_map
  else:
    chosen_source_map = pair_source_map
    rejected_source_map = pair_source_map

  # Precompute source rows from the original dataset, so map() updates do not
  # cascade into later source examples.
  needed_source_indices = set(pair_source_map.values())
  if mode in {"cross_response", "cross_response_flip"}:
    needed_source_indices.update(chosen_source_map.values())
    needed_source_indices.update(rejected_source_map.values())

  source_rows = {
      source_idx: copy.deepcopy(dataset[source_idx])
      for source_idx in needed_source_indices
  }

  selected_set = set(selected_indices)

  def _maybe_mismatch(row, idx):
    if idx not in selected_set:
      return row

    row = dict(row)

    if mode == "response_pair":
      source_idx = pair_source_map[idx]
      source_row = source_rows[source_idx]
      # prompt_i + chosen_j/rejected_j, with j in selected corruption subset.
      row["chosen"] = copy.deepcopy(source_row["chosen"])
      row["rejected"] = copy.deepcopy(source_row["rejected"])

    elif mode == "prompt":
      source_idx = pair_source_map[idx]
      source_row = source_rows[source_idx]
      # prompt_j + chosen_i/rejected_i, with j in selected corruption subset.
      row["prompt"] = copy.deepcopy(source_row["prompt"])

    elif mode == "cross_response":
      chosen_source_idx = chosen_source_map[idx]
      rejected_source_idx = rejected_source_map[idx]
      chosen_source_row = source_rows[chosen_source_idx]
      rejected_source_row = source_rows[rejected_source_idx]
      # prompt_i + chosen_j/rejected_k, with j,k in selected corruption subset.
      row["chosen"] = copy.deepcopy(chosen_source_row["chosen"])
      row["rejected"] = copy.deepcopy(rejected_source_row["rejected"])

    elif mode == "cross_response_flip":
      chosen_source_idx = chosen_source_map[idx]
      rejected_source_idx = rejected_source_map[idx]
      chosen_source_row = source_rows[chosen_source_idx]
      rejected_source_row = source_rows[rejected_source_idx]
      # Reward-adversarial cross response:
      # prompt_i + rejected_j as chosen + chosen_k as rejected,
      # with j,k in the selected corruption subset.
      row["chosen"] = copy.deepcopy(chosen_source_row["rejected"])
      row["rejected"] = copy.deepcopy(rejected_source_row["chosen"])

    return row

  return dataset.map(_maybe_mismatch, with_indices=True)

# BEGIN SAME_POOL_OVERRIDE_APPLY_PROMPT_RESPONSE_MISMATCH
def apply_prompt_response_mismatch_to_hf_dataset(
    dataset,
    ratio,
    seed,
    mode="response_pair",
):
  """Apply selected-only prompt/response mismatch corruption.

  Supported modes:
    response_pair:
      prompt_i + chosen_j/rejected_j.

    prompt:
      prompt_j + chosen_i/rejected_i.

    cross_response:
      prompt_i + chosen_j/rejected_k.

    cross_response_flip:
      prompt_i + rejected_j as chosen + chosen_k as rejected.

    same_pool_cross_response:
      prompt_i + chosen_j/chosen_k or prompt_i + rejected_j/rejected_k.
      The assignment is balanced across corrupted rows. If the number of
      corrupted rows is odd, exactly one mixed chosen_j/rejected_k row is used
      to keep total chosen-pool and rejected-pool response slots balanced.
  """
  import copy as _copy
  import os as _os
  import random as _random

  ratio = float(ratio)
  if ratio < 0.0 or ratio > 1.0:
    raise ValueError(f"ratio must be in [0, 1], got {ratio}")

  env_mode = _os.environ.get("DPO_MISMATCH_MODE")
  if env_mode:
    mode = env_mode
    print(f"[preference_noise] Using mismatch mode from DPO_MISMATCH_MODE={mode}")

  mode = mode.strip().lower().replace("-", "_")
  valid_modes = {
      "response_pair",
      "prompt",
      "cross_response",
      "cross_response_flip",
      "same_pool_cross_response",
  }
  if mode not in valid_modes:
    raise ValueError(
        f"mismatch mode must be one of {sorted(valid_modes)}; got {mode!r}."
    )

  required_columns = ("prompt", "chosen", "rejected")
  column_names = getattr(dataset, "column_names", None)
  if column_names is not None:
    missing = [c for c in required_columns if c not in column_names]
    if missing:
      raise ValueError(
          f"Dataset is missing required columns {missing}; "
          f"available columns are {column_names}."
      )

  dataset_size = len(dataset)
  if dataset_size == 0 or ratio == 0.0:
    return dataset

  corruption_count = int(dataset_size * ratio)
  if corruption_count == 0:
    return dataset
  if corruption_count < 2:
    raise ValueError(
        "prompt/response mismatch requires at least 2 corrupted rows "
        f"to avoid self-mapping, got corruption_count={corruption_count}."
    )

  rng = _random.Random(seed)
  selected_indices = sorted(rng.sample(range(dataset_size), corruption_count))
  selected_set = set(selected_indices)

  source_order = list(selected_indices)
  rng.shuffle(source_order)

  def _source_map_from_order(order, shift):
    n = len(order)
    if n < 2:
      raise ValueError("Need at least 2 selected rows for derangement.")
    shift = shift % n
    if shift == 0:
      shift = 1
    return {
        target_idx: order[(pos + shift) % n]
        for pos, target_idx in enumerate(order)
    }

  pair_source_map = _source_map_from_order(source_order, shift=1)

  if mode in {"cross_response", "cross_response_flip", "same_pool_cross_response"}:
    chosen_source_map = pair_source_map
    if corruption_count >= 3:
      rejected_source_map = _source_map_from_order(source_order, shift=-1)
    else:
      rejected_source_map = pair_source_map
  else:
    chosen_source_map = pair_source_map
    rejected_source_map = pair_source_map

  same_pool_role_map = {}
  if mode == "same_pool_cross_response":
    role_order = list(selected_indices)
    rng.shuffle(role_order)
    half = len(role_order) // 2

    for target_idx in role_order[:half]:
      same_pool_role_map[target_idx] = "chosen_same"
    for target_idx in role_order[half: 2 * half]:
      same_pool_role_map[target_idx] = "rejected_same"
    if len(role_order) % 2 == 1:
      same_pool_role_map[role_order[-1]] = "mixed"

  needed_source_indices = set(pair_source_map.values())
  needed_source_indices.update(chosen_source_map.values())
  needed_source_indices.update(rejected_source_map.values())

  source_rows = {
      idx: _copy.deepcopy(dataset[idx])
      for idx in needed_source_indices
  }

  def _maybe_mismatch(row, idx):
    row = dict(row)
    if idx not in selected_set:
      return row

    if mode == "response_pair":
      source_idx = pair_source_map[idx]
      source_row = source_rows[source_idx]
      row["chosen"] = _copy.deepcopy(source_row["chosen"])
      row["rejected"] = _copy.deepcopy(source_row["rejected"])

    elif mode == "prompt":
      source_idx = pair_source_map[idx]
      source_row = source_rows[source_idx]
      row["prompt"] = _copy.deepcopy(source_row["prompt"])

    elif mode == "cross_response":
      chosen_source_idx = chosen_source_map[idx]
      rejected_source_idx = rejected_source_map[idx]
      chosen_source_row = source_rows[chosen_source_idx]
      rejected_source_row = source_rows[rejected_source_idx]
      row["chosen"] = _copy.deepcopy(chosen_source_row["chosen"])
      row["rejected"] = _copy.deepcopy(rejected_source_row["rejected"])

    elif mode == "cross_response_flip":
      chosen_source_idx = chosen_source_map[idx]
      rejected_source_idx = rejected_source_map[idx]
      chosen_source_row = source_rows[chosen_source_idx]
      rejected_source_row = source_rows[rejected_source_idx]
      row["chosen"] = _copy.deepcopy(chosen_source_row["rejected"])
      row["rejected"] = _copy.deepcopy(rejected_source_row["chosen"])

    elif mode == "same_pool_cross_response":
      chosen_source_idx = chosen_source_map[idx]
      rejected_source_idx = rejected_source_map[idx]
      chosen_source_row = source_rows[chosen_source_idx]
      rejected_source_row = source_rows[rejected_source_idx]
      role = same_pool_role_map[idx]

      if role == "chosen_same":
        # prompt_i + chosen_j/chosen_k
        row["chosen"] = _copy.deepcopy(chosen_source_row["chosen"])
        row["rejected"] = _copy.deepcopy(rejected_source_row["chosen"])
      elif role == "rejected_same":
        # prompt_i + rejected_j/rejected_k
        row["chosen"] = _copy.deepcopy(chosen_source_row["rejected"])
        row["rejected"] = _copy.deepcopy(rejected_source_row["rejected"])
      elif role == "mixed":
        # Rare balancing row when |C| is odd.
        row["chosen"] = _copy.deepcopy(chosen_source_row["chosen"])
        row["rejected"] = _copy.deepcopy(rejected_source_row["rejected"])
      else:
        raise ValueError(f"Unknown same_pool_cross_response role: {role!r}")

    return row

  return dataset.map(_maybe_mismatch, with_indices=True)
# END SAME_POOL_OVERRIDE_APPLY_PROMPT_RESPONSE_MISMATCH
