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

"""Dataset loading for the DeepScaleR agentic GRPO recipe.

Returns a raw grain.MapDataset ready for post_init_dataset().
Each element has keys: "prompts", "question", "answer".
"""

from absl import logging
import datasets as datasets_lib
import fsspec
import grain
import pandas as pd


def _filter_training_rows(train_df: pd.DataFrame) -> pd.DataFrame:
  """Deterministically removes rows that cannot be math-rewarded."""
  required = {"problem", "answer"}
  missing = sorted(required - set(train_df.columns))
  if missing:
    raise ValueError(f"DeepScaleR training data is missing columns: {missing}")

  problem_present = train_df["problem"].fillna("").astype(str).str.strip().ne("")
  answer_present = train_df["answer"].fillna("").astype(str).str.strip().ne("")
  empty_problem = int((~problem_present).sum())
  empty_answer = int((problem_present & ~answer_present).sum())
  retained = train_df.loc[problem_present & answer_present].reset_index(drop=True)
  logging.info(
      "DeepScaleR train filtering: input=%d retained=%d "
      "empty_problem=%d empty_answer=%d",
      len(train_df),
      len(retained),
      empty_problem,
      empty_answer,
  )
  if retained.empty:
    raise ValueError("DeepScaleR training data has no non-empty problem/answer rows.")
  return retained


def create_dataset(
    train_data_path: str,
    eval_data_path: str,
    shuffle: bool = False,
    seed: int = 42,
) -> grain.MapDataset:
  """Load DeepScaleR train set and AIME eval set.

  Args:
    train_data_path: Path to deepscaler.json (supports GCS via fsspec).
    eval_data_path: Path to AIME parquet (supports GCS via fsspec).
    shuffle: Whether to shuffle the training set.
    seed: Random seed for shuffling.

  Returns:
    grain.MapDataset where each element has "prompts", "question", "answer".
  """
  instruction = (
      "Let's think step by step, and put your final answer within \\boxed{}."
  )

  def _preprocess(example, index):
    del index
    return {
        "question": example["problem"],
        "ground_truth": example["answer"],
        "data_source": "math",
    }

  def _to_prompt(item):
    question = item.get("question", item.get("problem", ""))
    answer = item.get("answer", item.get("ground_truth", ""))
    return {
        "prompts": f"{question} {instruction}",
        "question": question,
        "answer": answer,
    }

  with fsspec.open(train_data_path) as f:
    train_df = pd.read_json(f)
  train_df = _filter_training_rows(train_df)
  with fsspec.open(eval_data_path, "rb") as f:
    eval_df = pd.read_parquet(f)

  train_ds = datasets_lib.Dataset.from_pandas(train_df).map(
      _preprocess, with_indices=True
  )
  if shuffle:
    train_ds = train_ds.shuffle(seed)

  return grain.MapDataset.source(train_ds).map(_to_prompt)
