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


def create_dataset(
    split: str,
    limit: int | None = None,
    seed: int = 42,
):
  """Loads UltraFeedback preference pairs as a Grain dataset."""
  dataset = load_dataset(_DATASET_NAME, split=split)
  if limit is not None:
    dataset = dataset.shuffle(seed=seed).select(range(min(limit, len(dataset))))
  elif split.startswith("train"):
    dataset = dataset.shuffle(seed=seed)
  return grain.MapDataset.source(dataset).map(_to_preference_record)
