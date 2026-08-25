# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""UltraFeedback preference-pair dataset adapter for Tunix DPO."""

from __future__ import annotations

import hashlib
import math
from typing import Any

from datasets import load_dataset

from examples.dpo.preference_noise import apply_prompt_response_mismatch_to_hf_dataset
from grain import python as grain
import numpy as np


_DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"


def _extract_response_text(value: Any) -> str:
  if isinstance(value, str):
    return value
  if isinstance(value, list) and value:
    last_message = value[-1]
    if isinstance(last_message, dict) and "content" in last_message:
      return str(last_message["content"])
  raise ValueError(f"Unsupported response format: {value!r}")


def _to_preference_record(row: dict[str, Any]) -> dict[str, Any]:
  return {
      "prompt": [{"role": "user", "content": row["prompt"]}],
      "chosen_responses": _extract_response_text(row["chosen"]),
      "rejected_responses": _extract_response_text(row["rejected"]),
  }


def _normalize_partition(partition: str) -> str:
  normalized = partition.strip().lower().replace("-", "_")
  if normalized not in {"all", "sft", "dpo"}:
    raise ValueError(
        "partition must be one of {'all', 'sft', 'dpo'}, "
        f"got {partition!r}."
    )
  return normalized


def _normalize_subset(subset: str) -> str:
  normalized = subset.strip().lower().replace("-", "_")
  if normalized not in {"all", "train", "eval"}:
    raise ValueError(
        "subset must be one of {'all', 'train', 'eval'}, "
        f"got {subset!r}."
    )
  return normalized


def _normalize_flip_scope(flip_scope: str) -> str:
  normalized = flip_scope.strip().lower().replace("-", "_")
  if normalized not in {"none", "global", "tail_fraction"}:
    raise ValueError(
        "flip_scope must be one of {'none', 'global', 'tail_fraction'}, "
        f"got {flip_scope!r}."
    )
  return normalized

# jasson's no_pref mode for ablation
def _normalize_no_pref_mode(no_pref_mode: str) -> str:
  normalized = no_pref_mode.strip().lower().replace("-", "_")
  if normalized not in {"duplicate_chosen", "duplicate_rejected"}:
    raise ValueError(
        "no_pref_mode must be one of "
        "{'duplicate_chosen', 'duplicate_rejected'}, "
        f"got {no_pref_mode!r}."
    )
  return normalized

def _validate_flip_ratio(name: str, value: float) -> None:
  if not 0.0 <= value <= 1.0:
    raise ValueError(f"{name} must be between 0 and 1 inclusive; got {value}.")


def _prompt_bucket(prompt: str, salt: str) -> float:
  digest = hashlib.sha256(f"{salt}:{prompt}".encode("utf-8")).digest()
  return int.from_bytes(digest[:8], "big") / 2**64


def _prompt_is_in_partition(
    prompt: str,
    partition: str,
    sft_fraction: float,
    seed: int,
) -> bool:
  partition = _normalize_partition(partition)
  if partition == "all":
    return True
  if not 0.0 < sft_fraction < 1.0:
    raise ValueError(
        f"sft_fraction must be between 0 and 1, exclusive; got {sft_fraction}."
    )
  bucket = _prompt_bucket(prompt, salt=f"stage:{seed}")
  is_sft = bucket < sft_fraction
  return is_sft if partition == "sft" else not is_sft


def _prompt_is_in_subset(
    prompt: str,
    subset: str,
    eval_fraction: float,
    seed: int,
) -> bool:
  subset = _normalize_subset(subset)
  if subset == "all":
    return True
  if not 0.0 < eval_fraction < 1.0:
    raise ValueError(
        f"eval_fraction must be between 0 and 1, exclusive; got {eval_fraction}."
    )

  bucket = _prompt_bucket(prompt, salt=f"eval:{seed}")
  is_eval = bucket < eval_fraction
  return is_eval if subset == "eval" else not is_eval


def _select_flip_indices(
    dataset_size: int,
    *,
    flip_scope: str,
    flip_ratio: float,
    flip_tail_fraction: float,
    flip_seed: int,
) -> set[int]:
  """Returns deterministic indices to flip after dataset shuffle."""
  flip_scope = _normalize_flip_scope(flip_scope)
  _validate_flip_ratio("flip_ratio", flip_ratio)
  _validate_flip_ratio("flip_tail_fraction", flip_tail_fraction)

  if dataset_size <= 0 or flip_scope == "none" or flip_ratio <= 0.0:
    return set()

  if flip_scope == "global":
    candidate_indices = list(range(dataset_size))
  else:
    tail_size = int(math.ceil(dataset_size * flip_tail_fraction))
    tail_start = dataset_size - tail_size
    candidate_indices = list(range(max(0, tail_start), dataset_size))

  if not candidate_indices:
    return set()

  num_to_flip = int(len(candidate_indices) * flip_ratio)
  if flip_ratio > 0.0:
    num_to_flip = max(1, num_to_flip)
  num_to_flip = min(num_to_flip, len(candidate_indices))
  if num_to_flip <= 0:
    return set()

  rng = np.random.default_rng(flip_seed)
  return set(
      sorted(rng.choice(candidate_indices, size=num_to_flip, replace=False))
  )


def create_dataset(
    split: str,
    limit: int | None = None,
    seed: int = 42,
    shuffle_seed: int | None = None,
    partition: str = "all",
    sft_fraction: float = 0.5,
    subset: str = "all",
    eval_fraction: float = 0.0,
    flip_scope: str = "none",
    flip_ratio: float = 0.0,
    flip_tail_fraction: float = 0.5,
    flip_seed: int = 123,
    # jason's no_pref configs for ablation
    no_pref_scope: str = "none",
    no_pref_ratio: float = 0.0,
    no_pref_tail_fraction: float = 0.5,
    no_pref_seed: int = 123,
    no_pref_mode: str = "duplicate_chosen",
  # Jason: prompt-response mismatch configs for ablation.
  mismatch_scope: str = "none",
  mismatch_ratio: float = 0.0,
  mismatch_seed: int = 123,
  mismatch_mode: str = "response_pair",
):
  """Loads UltraFeedback preference pairs as a Grain dataset."""
  dataset = load_dataset(_DATASET_NAME, split=split)
  partition = _normalize_partition(partition)
  subset = _normalize_subset(subset)
  flip_scope = _normalize_flip_scope(flip_scope)
  # jason's no_pref configs for ablation
  no_pref_scope = _normalize_flip_scope(no_pref_scope)
  no_pref_mode = _normalize_no_pref_mode(no_pref_mode)
  _validate_flip_ratio("no_pref_ratio", no_pref_ratio)
  _validate_flip_ratio("no_pref_tail_fraction", no_pref_tail_fraction)
  # Jason: prompt-response mismatch configs for ablation.
  mismatch_scope = _normalize_flip_scope(mismatch_scope)
  mismatch_mode = mismatch_mode.strip().lower().replace("-", "_")
  _validate_flip_ratio("mismatch_ratio", mismatch_ratio)

  active_corruptions = [
      flip_scope != "none" and flip_ratio > 0.0,
      no_pref_scope != "none" and no_pref_ratio > 0.0,
      mismatch_scope != "none" and mismatch_ratio > 0.0,
  ]
  if sum(bool(x) for x in active_corruptions) > 1:
    raise ValueError(
        "Only one corruption type should be enabled at a time: "
        "flip, no_pref, or mismatch."
    )

  if flip_scope != "none" and flip_ratio > 0.0 and no_pref_scope != "none" and no_pref_ratio > 0.0:
    raise ValueError(
        "flip corruption and no-preference corruption should not be enabled together."
    )

  if partition != "all":
    dataset = dataset.filter(
        lambda row: _prompt_is_in_partition(
            row["prompt"],
            partition=partition,
            sft_fraction=sft_fraction,
            seed=seed,
        )
    )
  if subset != "all":
    dataset = dataset.filter(
        lambda row: _prompt_is_in_subset(
            row["prompt"],
            subset=subset,
            eval_fraction=eval_fraction,
            seed=seed,
        )
    )
  effective_shuffle_seed = seed
  if split.startswith("train") and shuffle_seed is not None:
    effective_shuffle_seed = int(shuffle_seed)
  if limit is not None:
    dataset = dataset.shuffle(seed=effective_shuffle_seed).select(
        range(min(limit, len(dataset)))
    )
  elif split.startswith("train"):
    dataset = dataset.shuffle(seed=effective_shuffle_seed)
  if flip_scope != "none" and flip_ratio > 0.0:
    flip_indices = _select_flip_indices(
        len(dataset),
        flip_scope=flip_scope,
        flip_ratio=flip_ratio,
        flip_tail_fraction=flip_tail_fraction,
        flip_seed=flip_seed,
    )

    def _maybe_flip(row, idx):
      if idx in flip_indices:
        return {**row, "chosen": row["rejected"], "rejected": row["chosen"]}
      return row

    dataset = dataset.map(_maybe_flip, with_indices=True)

  if no_pref_scope != "none" and no_pref_ratio > 0.0:
    no_pref_indices = _select_flip_indices(
        len(dataset),
        flip_scope=no_pref_scope,
        flip_ratio=no_pref_ratio,
        flip_tail_fraction=no_pref_tail_fraction,
        flip_seed=no_pref_seed,
    )

    def _maybe_no_pref(row, idx):
      if idx in no_pref_indices:
        if no_pref_mode == "duplicate_chosen":
          return {**row, "rejected": row["chosen"]}
        if no_pref_mode == "duplicate_rejected":
          return {**row, "chosen": row["rejected"]}
        raise ValueError(f"Unsupported no_pref_mode: {no_pref_mode!r}")
      return row

    dataset = dataset.map(_maybe_no_pref, with_indices=True)


  if mismatch_scope != "none" and mismatch_ratio > 0.0:
    if mismatch_scope != "global":
      raise ValueError(
          "prompt-response mismatch currently supports only global scope, "
          f"got {mismatch_scope!r}."
      )
    dataset = apply_prompt_response_mismatch_to_hf_dataset(
        dataset,
        ratio=mismatch_ratio,
        seed=mismatch_seed,
        mode=mismatch_mode,
    )

  return grain.MapDataset.source(dataset).map(_to_preference_record)
