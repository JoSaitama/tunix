#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Paired problem-level comparison for matched AIME evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

from examples.deepscaler import analyze_aime_eval


def _parse_eval(value: str) -> tuple[str, Path]:
  if "=" not in value:
    raise argparse.ArgumentTypeError("--eval requires LABEL=SAMPLES_JSONL.")
  label, path = value.split("=", 1)
  if not label or not path:
    raise argparse.ArgumentTypeError("--eval requires LABEL=SAMPLES_JSONL.")
  return label, Path(path).expanduser()


def _problem_metrics(rows: list[dict[str, Any]], indices: list[int]) -> dict[str, float]:
  selected = [rows[index] for index in indices]
  total_samples = sum(row["num_samples"] for row in selected)
  return {
      "avg": sum(row["correct_count"] for row in selected) / total_samples,
      "pass": sum(bool(row["pass"]) for row in selected) / len(selected),
      "majority": (
          sum(bool(row["majority_correct"]) for row in selected) / len(selected)
      ),
  }


def _percentile(values: list[float], quantile: float) -> float:
  ordered = sorted(values)
  position = (len(ordered) - 1) * quantile
  lower = int(position)
  upper = min(lower + 1, len(ordered) - 1)
  weight = position - lower
  return ordered[lower] * (1 - weight) + ordered[upper] * weight


def compare(
    evaluations: dict[str, list[dict[str, Any]]],
    *,
    reference: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
  if reference not in evaluations:
    raise ValueError(f"Reference {reference!r} is not one of the evaluations.")
  reference_rows = evaluations[reference]
  identities = [
      (row["problem_id"], row["question"], row["ground_truth"])
      for row in reference_rows
  ]
  for label, rows in evaluations.items():
    candidate = [
        (row["problem_id"], row["question"], row["ground_truth"])
        for row in rows
    ]
    if candidate != identities:
      raise ValueError(f"Evaluation {label!r} is not problem-aligned.")

  all_indices = list(range(len(reference_rows)))
  point_metrics = {
      label: _problem_metrics(rows, all_indices)
      for label, rows in evaluations.items()
  }
  rng = random.Random(bootstrap_seed)
  deltas: dict[str, dict[str, list[float]]] = {
      label: {metric: [] for metric in ("avg", "pass", "majority")}
      for label in evaluations
      if label != reference
  }
  for _ in range(bootstrap_samples):
    indices = [rng.randrange(len(reference_rows)) for _ in reference_rows]
    reference_metrics = _problem_metrics(reference_rows, indices)
    for label, rows in evaluations.items():
      if label == reference:
        continue
      candidate_metrics = _problem_metrics(rows, indices)
      for metric in deltas[label]:
        deltas[label][metric].append(
            candidate_metrics[metric] - reference_metrics[metric]
        )

  comparisons = {}
  for label, metric_deltas in deltas.items():
    comparisons[label] = {}
    for metric, values in metric_deltas.items():
      comparisons[label][metric] = {
          "delta": point_metrics[label][metric] - point_metrics[reference][metric],
          "ci95_low": _percentile(values, 0.025),
          "ci95_high": _percentile(values, 0.975),
      }
  return {
      "reference": reference,
      "bootstrap_samples": bootstrap_samples,
      "bootstrap_seed": bootstrap_seed,
      "point_metrics": point_metrics,
      "paired_comparisons": comparisons,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--eval", action="append", required=True, type=_parse_eval)
  parser.add_argument("--reference", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--k", type=int, default=16)
  parser.add_argument("--max_generation_steps", type=int, default=8192)
  parser.add_argument("--bootstrap_samples", type=int, default=10000)
  parser.add_argument("--bootstrap_seed", type=int, default=20260707)
  args = parser.parse_args()

  evaluations = {}
  for label, path in args.eval:
    _, rows = analyze_aime_eval.analyze_records(
        analyze_aime_eval._read_jsonl(path),
        k=args.k,
        max_generation_steps=args.max_generation_steps,
    )
    evaluations[label] = rows
  result = compare(
      evaluations,
      reference=args.reference,
      bootstrap_samples=args.bootstrap_samples,
      bootstrap_seed=args.bootstrap_seed,
  )
  output = Path(args.output).expanduser()
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
