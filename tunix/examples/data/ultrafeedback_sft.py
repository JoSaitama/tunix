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

"""UltraFeedback chosen-only dataset adapter for Tunix SFT."""

from __future__ import annotations

from grain import python as grain

from tunix.examples.data import ultrafeedback_dpo


def _to_sft_record(row):
  return {
      "prompt": [{"role": "user", "content": row["prompt"]}],
      "response": ultrafeedback_dpo._extract_response_text(row["chosen"]),
  }


def create_dataset(
    split: str,
    limit: int | None = None,
    seed: int = 42,
    partition: str = "sft",
    sft_fraction: float = 0.5,
    subset: str = "all",
    eval_fraction: float = 0.0,
):
  """Loads UltraFeedback chosen responses for SFT."""
  dataset = ultrafeedback_dpo.load_dataset(
      ultrafeedback_dpo._DATASET_NAME, split=split
  )
  partition = ultrafeedback_dpo._normalize_partition(partition)
  subset = ultrafeedback_dpo._normalize_subset(subset)
  if partition != "all":
    dataset = dataset.filter(
        lambda row: ultrafeedback_dpo._prompt_is_in_partition(
            row["prompt"],
            partition=partition,
            sft_fraction=sft_fraction,
            seed=seed,
        )
    )
  if subset != "all":
    dataset = dataset.filter(
        lambda row: ultrafeedback_dpo._prompt_is_in_subset(
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
  return grain.MapDataset.source(dataset).map(_to_sft_record)
