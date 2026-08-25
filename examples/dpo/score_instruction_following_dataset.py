#!/usr/bin/env python3

"""Offline scorer for IFBench-style instruction-following datasets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


DEFAULT_IFBENCH_ROOT = Path("/home/lhf_hongfu_gmail_com/.cache/IFBench")
DEFAULT_LIVEBENCH_ROOT = Path("/home/lhf_hongfu_gmail_com/.cache/LiveBench")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Score IFBench-format instruction-following responses with the "
          "official IFBench evaluation library."
      )
  )
  parser.add_argument(
      "--dataset-format",
      choices=("ifbench", "livebench_if"),
      required=True,
  )
  parser.add_argument("--input-data", required=True)
  parser.add_argument("--input-response-data", required=True)
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--score-prefix", required=True)
  parser.add_argument(
      "--ifbench-root",
      default=os.environ.get("IFBENCH_ROOT", str(DEFAULT_IFBENCH_ROOT)),
  )
  parser.add_argument(
      "--livebench-root",
      default=os.environ.get("LIVEBENCH_ROOT", str(DEFAULT_LIVEBENCH_ROOT)),
  )
  return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  with path.open(encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _compute_scores(
    strict_rows: list[dict[str, Any]],
    loose_rows: list[dict[str, Any]],
    *,
    prefix: str,
) -> dict[str, float]:
  def _prompt_accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
      return 0.0
    return sum(bool(row["follow_all_instructions"]) for row in rows) / len(rows)

  def _instruction_accuracy(rows: list[dict[str, Any]]) -> float:
    total = 0
    correct = 0
    for row in rows:
      follow_list = row.get("follow_instruction_list", [])
      total += len(follow_list)
      correct += sum(bool(value) for value in follow_list)
    if total <= 0:
      return 0.0
    return correct / total

  def _mean_question_score(rows: list[dict[str, Any]]) -> float:
    if not rows:
      return 0.0
    total = 0.0
    for row in rows:
      follow_list = row.get("follow_instruction_list", [])
      prompt_score = 1.0 if row["follow_all_instructions"] else 0.0
      instruction_score = (
          sum(bool(value) for value in follow_list) / len(follow_list)
          if follow_list
          else 0.0
      )
      total += 0.5 * (prompt_score + instruction_score)
    return total / len(rows)

  return {
      f"{prefix}_prompt_strict": _prompt_accuracy(strict_rows),
      f"{prefix}_prompt_loose": _prompt_accuracy(loose_rows),
      f"{prefix}_instruction_strict": _instruction_accuracy(strict_rows),
      f"{prefix}_instruction_loose": _instruction_accuracy(loose_rows),
      f"{prefix}_score": _mean_question_score(strict_rows),
      f"{prefix}_score_loose": _mean_question_score(loose_rows),
  }


def _load_evaluation_lib(
    *,
    dataset_format: str,
    ifbench_root: Path,
    livebench_root: Path,
):
  if dataset_format == "ifbench":
    sys.path.insert(0, str(ifbench_root))
    import evaluation_lib  # pylint: disable=import-error

    return evaluation_lib

  sys.path.insert(0, str(livebench_root))
  from livebench.if_runner.instruction_following_eval import (  # pylint: disable=import-error
      evaluation_main,
  )

  return evaluation_main


def main() -> None:
  args = _parse_args()
  ifbench_root = Path(args.ifbench_root).resolve()
  livebench_root = Path(args.livebench_root).resolve()
  evaluation_lib = _load_evaluation_lib(
      dataset_format=args.dataset_format,
      ifbench_root=ifbench_root,
      livebench_root=livebench_root,
  )

  input_rows = _load_jsonl(Path(args.input_data))
  response_rows = _load_jsonl(Path(args.input_response_data))
  output_dir = Path(args.output_dir).resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  response_by_prompt = {}
  response_by_id = {}
  for row in response_rows:
    prompt = row.get("prompt")
    if prompt is not None:
      response_by_prompt[prompt] = row.get("response", "")
    question_id = row.get("question_id")
    if question_id is not None:
      response_by_id[str(question_id)] = row.get("response", "")

  strict_outputs = []
  loose_outputs = []
  detailed_rows = []
  for row in input_rows:
    if args.dataset_format == "ifbench":
      key = row["key"]
      prompt = row["prompt"]
    else:
      key = row["question_id"]
      prompt = row["turns"][0]
    kwargs = [
        {k: v for k, v in item.items() if v is not None}
        for item in row["kwargs"]
    ]
    response = response_by_id.get(str(key), response_by_prompt.get(prompt, ""))
    example = evaluation_lib.InputExample(
        key=key,
        instruction_id_list=row["instruction_id_list"],
        prompt=prompt,
        kwargs=kwargs,
    )
    prompt_to_response = {prompt: response}
    strict = evaluation_lib.test_instruction_following_strict(
        example, prompt_to_response
    )
    loose = evaluation_lib.test_instruction_following_loose(
        example, prompt_to_response
    )
    strict_outputs.append(
        {
            "key": key,
            "instruction_id_list": strict.instruction_id_list,
            "prompt": strict.prompt,
            "response": strict.response,
            "follow_all_instructions": strict.follow_all_instructions,
            "follow_instruction_list": strict.follow_instruction_list,
        }
    )
    loose_outputs.append(
        {
            "key": key,
            "instruction_id_list": loose.instruction_id_list,
            "prompt": loose.prompt,
            "response": loose.response,
            "follow_all_instructions": loose.follow_all_instructions,
            "follow_instruction_list": loose.follow_instruction_list,
        }
    )
    strict_instruction = (
        sum(strict.follow_instruction_list) / len(strict.follow_instruction_list)
        if strict.follow_instruction_list
        else 0.0
    )
    loose_instruction = (
        sum(loose.follow_instruction_list) / len(loose.follow_instruction_list)
        if loose.follow_instruction_list
        else 0.0
    )
    detailed_rows.append(
        {
            "key": str(key),
            "strict_prompt": float(bool(strict.follow_all_instructions)),
            "strict_instruction": strict_instruction,
            "strict_score": 0.5
            * (float(bool(strict.follow_all_instructions)) + strict_instruction),
            "loose_prompt": float(bool(loose.follow_all_instructions)),
            "loose_instruction": loose_instruction,
            "loose_score": 0.5
            * (float(bool(loose.follow_all_instructions)) + loose_instruction),
        }
    )

  scores = _compute_scores(
      strict_outputs,
      loose_outputs,
      prefix=args.score_prefix,
  )
  _write_jsonl(output_dir / "eval_results_strict.jsonl", strict_outputs)
  _write_jsonl(output_dir / "eval_results_loose.jsonl", loose_outputs)
  _write_jsonl(output_dir / "detailed_scores.jsonl", detailed_rows)
  summary_path = output_dir / "summary.json"
  summary_path.write_text(json.dumps(scores, indent=2, sort_keys=True))
  print(f"Wrote instruction-following summary to {summary_path}")


if __name__ == "__main__":
  main()
