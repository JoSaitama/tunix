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
from typing import Any

from datasets import load_dataset
from grain import python as grain


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


def create_dataset(
    split: str,
    limit: int | None = None,
    seed: int = 42,
    partition: str = "all",
    sft_fraction: float = 0.5,
    subset: str = "all",
    eval_fraction: float = 0.0,
):
  """Loads UltraFeedback preference pairs as a Grain dataset."""
  dataset = load_dataset(_DATASET_NAME, split=split)
  partition = _normalize_partition(partition)
  subset = _normalize_subset(subset)
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
  if limit is not None:
    dataset = dataset.shuffle(seed=seed).select(range(min(limit, len(dataset))))
  elif split.startswith("train"):
    dataset = dataset.shuffle(seed=seed)
  return grain.MapDataset.source(dataset).map(_to_preference_record)
