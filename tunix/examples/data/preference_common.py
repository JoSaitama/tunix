"""Common helpers for preference-pair datasets used by Tunix DPO examples."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np


def extract_response_text(value: Any) -> str:
  if isinstance(value, str):
    return value
  if isinstance(value, list) and value:
    last_message = value[-1]
    if isinstance(last_message, dict) and "content" in last_message:
      return str(last_message["content"])
  raise ValueError(f"Unsupported response format: {value!r}")


def to_preference_record(row: dict[str, Any]) -> dict[str, Any]:
  """Convert normalized prompt/chosen/rejected fields to a Tunix DPO record."""
  prompt = row["prompt"]
  if isinstance(prompt, list):
    prompt_messages = prompt
  else:
    prompt_messages = [{"role": "user", "content": str(prompt)}]

  record = {
      "prompt": prompt_messages,
      "chosen_responses": extract_response_text(row["chosen"]),
      "rejected_responses": extract_response_text(row["rejected"]),
  }

  # Keep lightweight metadata for offline filtering diagnostics.
  # These fields are ignored by the trainer input wrapper but allow us to
  # reconstruct which original examples were dropped by DTV/reward filtering.
  for key in ("example_id", "source_dataset"):
    if key in row:
      record[key] = row[key]

  return record



def normalize_partition(partition: str) -> str:
  normalized = str(partition).strip().lower().replace("-", "_")
  if normalized not in {"all", "sft", "dpo"}:
    raise ValueError(
        "partition must be one of {'all', 'sft', 'dpo'}, "
        f"got {partition!r}."
    )
  return normalized


def normalize_subset(subset: str) -> str:
  normalized = str(subset).strip().lower().replace("-", "_")
  if normalized not in {"all", "train", "eval"}:
    raise ValueError(
        "subset must be one of {'all', 'train', 'eval'}, "
        f"got {subset!r}."
    )
  return normalized


def normalize_corruption_scope(scope: str) -> str:
  normalized = str(scope).strip().lower().replace("-", "_")
  if normalized not in {"none", "global", "tail_fraction"}:
    raise ValueError(
        "scope must be one of {'none', 'global', 'tail_fraction'}, "
        f"got {scope!r}."
    )
  return normalized


def normalize_no_pref_mode(no_pref_mode: str) -> str:
  normalized = str(no_pref_mode).strip().lower().replace("-", "_")
  if normalized not in {"duplicate_chosen", "duplicate_rejected"}:
    raise ValueError(
        "no_pref_mode must be one of "
        "{'duplicate_chosen', 'duplicate_rejected'}, "
        f"got {no_pref_mode!r}."
    )
  return normalized


def validate_ratio(name: str, value: float) -> None:
  if not 0.0 <= float(value) <= 1.0:
    raise ValueError(f"{name} must be between 0 and 1 inclusive; got {value}.")


def prompt_bucket(prompt: str, salt: str) -> float:
  digest = hashlib.sha256(f"{salt}:{prompt}".encode("utf-8")).digest()
  return int.from_bytes(digest[:8], "big") / 2**64


def prompt_is_in_partition(
    prompt: str,
    partition: str,
    sft_fraction: float,
    seed: int,
) -> bool:
  partition = normalize_partition(partition)
  if partition == "all":
    return True
  if not 0.0 < float(sft_fraction) < 1.0:
    raise ValueError(
        f"sft_fraction must be between 0 and 1 exclusive; got {sft_fraction}."
    )
  bucket = prompt_bucket(prompt, salt=f"stage:{seed}")
  is_sft = bucket < float(sft_fraction)
  return is_sft if partition == "sft" else not is_sft


def prompt_is_in_subset(
    prompt: str,
    subset: str,
    eval_fraction: float,
    seed: int,
) -> bool:
  subset = normalize_subset(subset)
  if subset == "all":
    return True
  if not 0.0 < float(eval_fraction) < 1.0:
    raise ValueError(
        f"eval_fraction must be between 0 and 1 exclusive; got {eval_fraction}."
    )
  bucket = prompt_bucket(prompt, salt=f"eval:{seed}")
  is_eval = bucket < float(eval_fraction)
  return is_eval if subset == "eval" else not is_eval


def select_corruption_indices(
    dataset_size: int,
    *,
    scope: str,
    ratio: float,
    tail_fraction: float,
    seed: int,
) -> set[int]:
  """Select deterministic corruption indices after dataset shuffle/filter."""
  scope = normalize_corruption_scope(scope)
  validate_ratio("ratio", ratio)
  validate_ratio("tail_fraction", tail_fraction)

  if dataset_size <= 0 or scope == "none" or float(ratio) <= 0.0:
    return set()

  if scope == "global":
    candidate_indices = list(range(dataset_size))
  else:
    tail_size = int(math.ceil(dataset_size * float(tail_fraction)))
    tail_start = dataset_size - tail_size
    candidate_indices = list(range(max(0, tail_start), dataset_size))

  if not candidate_indices:
    return set()

  num_to_select = int(len(candidate_indices) * float(ratio))
  if ratio > 0.0:
    num_to_select = max(1, num_to_select)
  num_to_select = min(num_to_select, len(candidate_indices))

  rng = np.random.default_rng(seed)
  return set(
      sorted(rng.choice(candidate_indices, size=num_to_select, replace=False))
  )
