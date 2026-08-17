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

"""AIME parquet loader for frozen distributed-rollout diagnostics only."""

import datasets as datasets_lib
import fsspec
import grain
import pandas as pd


def create_dataset(
    train_data_path: str,
    eval_data_path: str,
    shuffle: bool = False,
    seed: int = 42,
) -> grain.MapDataset:
  """Loads AIME rows while preserving the DeepScaleR training prompt."""
  del train_data_path
  with fsspec.open(eval_data_path, "rb") as handle:
    frame = pd.read_parquet(handle)

  required = {"problem", "answer"}
  missing = sorted(required - set(frame.columns))
  if missing:
    raise ValueError(f"AIME diagnostic data is missing columns: {missing}")

  frame = frame.loc[
      frame["problem"].fillna("").astype(str).str.strip().ne("")
      & frame["answer"].fillna("").astype(str).str.strip().ne("")
  ].reset_index(drop=True)
  if frame.empty:
    raise ValueError("AIME diagnostic data has no usable rows.")

  dataset = datasets_lib.Dataset.from_pandas(frame)
  if shuffle:
    dataset = dataset.shuffle(seed=seed)

  instruction = (
      "Let's think step by step, and put your final answer within \\boxed{}."
  )

  def _to_prompt(item):
    question = str(item["problem"])
    return {
        "prompts": f"{question} {instruction}",
        "question": question,
        "answer": item["answer"],
    }

  return grain.MapDataset.source(dataset).map(_to_prompt)
