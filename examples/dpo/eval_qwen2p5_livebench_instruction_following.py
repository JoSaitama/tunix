#!/usr/bin/env python3

"""Generate and score LiveBench instruction-following answers for clean runs."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
from typing import Any

from datasets import load_dataset


REPO_ROOT = Path("/home/lhf_hongfu_gmail_com/tunix")
DEFAULT_OFFLINE_EVAL_VENV = Path("/home/lhf_hongfu_gmail_com/.venvs/DPO-EVAL-OFFLINE")
DEFAULT_LIVEBENCH_ROOT = Path("/home/lhf_hongfu_gmail_com/.cache/LiveBench")
HELPER_PATH = REPO_ROOT / "examples/dpo/score_instruction_following_dataset.py"


def _load_clean_benchmark_module():
  module_path = REPO_ROOT / "examples/dpo/eval_qwen2p5_clean_benchmarks.py"
  spec = importlib.util.spec_from_file_location(
      "eval_qwen2p5_clean_benchmarks",
      module_path,
  )
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Run LiveBench instruction-following on the four clean Qwen2.5 DPO "
          "runs using local TPU generation and official IFBench-style scoring."
      )
  )
  parser.add_argument("--repo-root", default=str(REPO_ROOT))
  parser.add_argument("--run-ts", required=True)
  parser.add_argument(
      "--methods",
      nargs="+",
      default=(
          "vanilla_dpo",
          "random_pair_filtering",
          "reward_based_filtering",
          "self_inf",
      ),
  )
  parser.add_argument(
      "--output-root",
      default=None,
      help="Defaults to runs/results/clean_benchmarks_<run-ts>/livebench_if.",
  )
  parser.add_argument("--question-limit", type=int, default=None)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--max-prompt-length", type=int, default=4096)
  parser.add_argument("--max-generation-steps", type=int, default=1024)
  parser.add_argument("--top-k", type=int, default=50)
  parser.add_argument("--top-p", type=float, default=0.95)
  parser.add_argument("--generation-batch-size", type=int, default=8)
  parser.add_argument(
      "--offline-eval-venv",
      default=str(DEFAULT_OFFLINE_EVAL_VENV),
  )
  parser.add_argument(
      "--livebench-root",
      default=str(DEFAULT_LIVEBENCH_ROOT),
  )
  parser.add_argument("--force", action="store_true")
  return parser.parse_args()


def _load_rows(limit: int | None) -> list[dict[str, Any]]:
  rows = list(load_dataset("livebench/instruction_following", split="test"))
  normalized_rows = []
  for row in rows:
    normalized = dict(row)
    for key, value in list(normalized.items()):
      if isinstance(value, datetime):
        normalized[key] = datetime.strftime(value, "%Y-%m-%d")
    normalized_rows.append(normalized)
  rows = normalized_rows
  if limit is not None:
    rows = rows[:limit]
  return rows


def _jsonl_len(path: Path) -> int:
  if not path.exists():
    return 0
  with path.open(encoding="utf-8") as f:
    return sum(1 for line in f if line.strip())


def main() -> None:
  args = _parse_args()
  clean_benchmarks = _load_clean_benchmark_module()
  repo_root = Path(args.repo_root).resolve()
  benchmark_root = (
      Path(args.output_root).resolve()
      if args.output_root
      else repo_root
      / "runs/results"
      / f"clean_benchmarks_{args.run_ts}"
      / "livebench_if"
  )
  benchmark_root.mkdir(parents=True, exist_ok=True)
  summary_path = benchmark_root / "livebench_instruction_following_summary.json"
  data_path = benchmark_root / "data/livebench_instruction_following.jsonl"
  rows = _load_rows(args.question_limit)
  clean_benchmarks._write_jsonl(data_path, rows)  # pylint: disable=protected-access

  runs = clean_benchmarks._discover_clean_runs(  # pylint: disable=protected-access
      repo_root,
      args.run_ts,
      [
          clean_benchmarks._normalize_method(method)  # pylint: disable=protected-access
          for method in args.methods
      ],
  )
  prompts = [row["turns"][0] for row in rows]
  previous_summary = (
      json.loads(summary_path.read_text()) if summary_path.exists() else {}
  )
  summary = {
      "run_ts": args.run_ts,
      "dataset": "livebench/instruction_following",
      "num_examples": len(rows),
      "scores": dict(previous_summary.get("scores", {})),
  }
  offline_python = Path(args.offline_eval_venv).resolve() / "bin/python"
  if not offline_python.exists():
    raise SystemExit(
        "Offline eval venv is missing python: "
        f"{offline_python}. Run setup_qwen2p5_clean_offline_eval_env.sh first."
    )

  for run in runs:
    response_file = benchmark_root / "responses" / f"{run['variant']}.jsonl"
    result_dir = benchmark_root / "results" / run["variant"]
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.force or _jsonl_len(response_file) != len(rows):
      generator = clean_benchmarks.TunixChatGenerator(
          config_path=str(clean_benchmarks.CONFIG_PATH),
          exported_model_path=run["exported_model_path"],
          max_prompt_length=args.max_prompt_length,
          max_generation_steps=args.max_generation_steps,
          top_k=args.top_k,
          top_p=args.top_p,
      )
      try:
        answers = clean_benchmarks._chunked_generate(  # pylint: disable=protected-access
            generator,
            prompts=prompts,
            temperature=0.0,
            seed=args.seed,
            batch_size=args.generation_batch_size,
        )
      finally:
        generator.close()
      response_rows = []
      for row, answer in zip(rows, answers, strict=True):
        response_rows.append(
            {
                "question_id": row["question_id"],
                "prompt": row["turns"][0],
                "response": answer,
            }
        )
      clean_benchmarks._write_jsonl(  # pylint: disable=protected-access
          response_file,
          response_rows,
      )
    subprocess.run(
        [
            str(offline_python),
            str(HELPER_PATH),
            "--dataset-format",
            "livebench_if",
            "--input-data",
            str(data_path),
            "--input-response-data",
            str(response_file),
            "--output-dir",
            str(result_dir),
            "--score-prefix",
            "livebench_if",
            "--livebench-root",
            str(Path(args.livebench_root).resolve()),
        ],
        check=True,
        cwd=str(repo_root),
    )
    method_summary = json.loads((result_dir / "summary.json").read_text())
    summary["scores"][run["variant"]] = {
        **method_summary,
        "response_file": str(response_file),
        "run_name": run["run_name"],
        "variant": run["variant"],
    }

  summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
  print(f"Wrote LiveBench IF summary to {summary_path}")


if __name__ == "__main__":
  main()
