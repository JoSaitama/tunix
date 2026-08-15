#!/usr/bin/env python3
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

"""Offline detailed analysis for one completed AIME evaluation."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Iterable

from tunix.utils import math_eval_metrics
from tunix.utils import math_utils


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
  records = []
  with path.open(encoding="utf-8") as source:
    for line_number, line in enumerate(source, start=1):
      if not line.strip():
        continue
      value = json.loads(line)
      if not isinstance(value, dict):
        raise ValueError(f"Line {line_number} is not a JSON object.")
      records.append(value)
  return records


def _answer_key(answer: str | None) -> str | None:
  if answer is None:
    return None
  normalized = math_utils.mathd_normalize_answer(answer)
  return normalized if normalized else None


def analyze_records(
    records: Iterable[dict[str, Any]],
    *,
    k: int,
    max_generation_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  records = list(records)
  metrics = math_eval_metrics.compute_at_k_metrics(
      records, k=k, max_generation_steps=max_generation_steps
  )
  grouped: dict[int | str, list[dict[str, Any]]] = collections.defaultdict(list)
  for record in records:
    grouped[record["problem_id"]].append(record)

  per_problem = []
  for problem_id in sorted(grouped, key=str):
    samples = sorted(grouped[problem_id], key=lambda item: item["sample_index"])
    keys = [
        _answer_key(
            sample.get("extracted_answer")
            or math_eval_metrics.extract_response_answer(sample.get("response"))
        )
        for sample in samples
    ]
    counts = collections.Counter(key for key in keys if key is not None)
    majority_answer = None
    if counts:
      first_index = {key: keys.index(key) for key in counts}
      majority_answer = max(
          counts, key=lambda key: (counts[key], -first_index[key])
      )
    correct = [
        bool(sample.get("is_correct"))
        if "is_correct" in sample
        else math_eval_metrics.is_response_correct(
            sample.get("response"), sample["ground_truth"]
        )
        for sample in samples
    ]
    per_problem.append({
        "problem_id": problem_id,
        "question": samples[0].get("question"),
        "ground_truth": samples[0]["ground_truth"],
        "num_samples": len(samples),
        "correct_count": sum(correct),
        "pass": any(correct),
        "majority_answer": majority_answer,
        "majority_vote_count": counts.get(majority_answer, 0),
        "majority_correct": math_eval_metrics.is_extracted_answer_correct(
            majority_answer, samples[0]["ground_truth"]
        ),
        "unique_answer_count": len(counts),
        "extractable_count": sum(key is not None for key in keys),
        "truncated_count": sum(bool(sample["truncated"]) for sample in samples),
        "token_counts": [int(sample["token_count"]) for sample in samples],
        "answers": keys,
        "correct": correct,
    })

  detailed_summary = {
      "metrics": metrics,
      "num_records": len(records),
      "num_problem_records": len(per_problem),
      "k": k,
      "max_generation_steps": max_generation_steps,
  }
  return detailed_summary, per_problem


def _write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--samples", required=True)
  parser.add_argument("--output_dir", required=True)
  parser.add_argument("--k", type=int, default=16)
  parser.add_argument("--max_generation_steps", type=int, default=8192)
  args = parser.parse_args()

  samples_path = Path(args.samples).expanduser()
  output_dir = Path(args.output_dir).expanduser()
  output_dir.mkdir(parents=True, exist_ok=True)
  summary, per_problem = analyze_records(
      _read_jsonl(samples_path),
      k=args.k,
      max_generation_steps=args.max_generation_steps,
  )
  _write_json(output_dir / "detailed_summary.json", summary)
  with (output_dir / "per_problem.jsonl").open("w", encoding="utf-8") as out:
    for record in per_problem:
      out.write(json.dumps(record, ensure_ascii=False) + "\n")
  print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
