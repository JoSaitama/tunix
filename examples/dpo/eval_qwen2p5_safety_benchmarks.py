#!/usr/bin/env python3

"""Generate and score HarmBench and XSTest for clean Qwen2.5 DPO runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
from typing import Any


REPO_ROOT = Path("/home/lhf_hongfu_gmail_com/tunix")
DEFAULT_OFFLINE_EVAL_VENV = Path("/home/lhf_hongfu_gmail_com/.venvs/DPO-EVAL-OFFLINE")
SAFETY_EVAL_ROOT = Path("/home/lhf_hongfu_gmail_com/.cache/safety-eval")
HELPER_PATH = REPO_ROOT / "examples/dpo/score_safety_generation.py"


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
          "Run XSTest and HarmBench on the four clean Qwen2.5 DPO runs using "
          "local TPU generation and GPT-based safety classifiers."
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
      "--benchmarks",
      nargs="+",
      default=("xstest", "harmbench"),
  )
  parser.add_argument(
      "--output-root",
      default=None,
      help="Defaults to runs/results/clean_benchmarks_<run-ts>/safety_benchmarks.",
  )
  parser.add_argument("--question-limit", type=int, default=None)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--max-prompt-length", type=int, default=4096)
  parser.add_argument("--max-generation-steps", type=int, default=1024)
  parser.add_argument("--top-k", type=int, default=50)
  parser.add_argument("--top-p", type=float, default=0.95)
  parser.add_argument("--generation-batch-size", type=int, default=8)
  parser.add_argument("--judge-model", default="gpt-4.1")
  parser.add_argument("--judge-batch-size", type=int, default=8)
  parser.add_argument(
      "--offline-eval-venv",
      default=str(DEFAULT_OFFLINE_EVAL_VENV),
  )
  parser.add_argument("--force", action="store_true")
  return parser.parse_args()


def _load_xstest_rows(limit: int | None) -> list[dict[str, Any]]:
  path = (
      SAFETY_EVAL_ROOT
      / "evaluation/tasks/generation/xstest/exaggerated_safety.json"
  )
  rows = json.loads(path.read_text())
  if limit is not None:
    rows = rows[:limit]
  return rows


def _load_harmbench_rows(limit: int | None) -> list[dict[str, Any]]:
  import csv

  path = (
      SAFETY_EVAL_ROOT
      / "evaluation/tasks/generation/harmbench/harmbench_behaviors_text_test.csv"
  )
  with path.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
  if limit is not None:
    rows = rows[:limit]
  return rows


def _harmbench_prompt(row: dict[str, Any]) -> str:
  if row["FunctionalCategory"] != "contextual":
    return row["Behavior"]
  return row["ContextString"] + "\n\n---\n\n" + row["Behavior"]


def main() -> None:
  args = _parse_args()
  clean_benchmarks = _load_clean_benchmark_module()
  repo_root = Path(args.repo_root).resolve()
  output_root = (
      Path(args.output_root).resolve()
      if args.output_root
      else repo_root
      / "runs/results"
      / f"clean_benchmarks_{args.run_ts}"
      / "safety_benchmarks"
  )
  output_root.mkdir(parents=True, exist_ok=True)
  offline_python = Path(args.offline_eval_venv).resolve() / "bin/python"
  if not offline_python.exists():
    raise SystemExit(
        "Offline eval venv is missing python: "
        f"{offline_python}. Run setup_qwen2p5_clean_offline_eval_env.sh first."
    )

  runs = clean_benchmarks._discover_clean_runs(  # pylint: disable=protected-access
      repo_root,
      args.run_ts,
      [
          clean_benchmarks._normalize_method(method)  # pylint: disable=protected-access
          for method in args.methods
      ],
  )
  selected_benchmarks = []
  for benchmark in args.benchmarks:
    normalized = benchmark.strip().lower().replace("-", "_")
    if normalized not in {"xstest", "harmbench"}:
      raise ValueError(f"Unsupported safety benchmark: {benchmark}")
    selected_benchmarks.append(normalized)

  benchmark_rows = {}
  benchmark_prompts = {}
  if "xstest" in selected_benchmarks:
    benchmark_rows["xstest"] = _load_xstest_rows(args.question_limit)
    benchmark_prompts["xstest"] = [
        row["prompt"] for row in benchmark_rows["xstest"]
    ]
    clean_benchmarks._write_jsonl(  # pylint: disable=protected-access
        output_root / "data/xstest.jsonl",
        benchmark_rows["xstest"],
    )
  if "harmbench" in selected_benchmarks:
    benchmark_rows["harmbench"] = _load_harmbench_rows(args.question_limit)
    benchmark_prompts["harmbench"] = [
        _harmbench_prompt(row) for row in benchmark_rows["harmbench"]
    ]
    clean_benchmarks._write_jsonl(  # pylint: disable=protected-access
        output_root / "data/harmbench.jsonl",
        benchmark_rows["harmbench"],
    )

  summary: dict[str, Any] = {
      "run_ts": args.run_ts,
      "judge_model": args.judge_model,
      "scores": {},
      "benchmarks": {
          benchmark: {"num_examples": len(benchmark_rows[benchmark])}
          for benchmark in selected_benchmarks
      },
  }

  for run in runs:
    generator = clean_benchmarks.TunixChatGenerator(
        config_path=str(clean_benchmarks.CONFIG_PATH),
        exported_model_path=run["exported_model_path"],
        max_prompt_length=args.max_prompt_length,
        max_generation_steps=args.max_generation_steps,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    method_scores = {}
    try:
      for benchmark in selected_benchmarks:
        response_file = (
            output_root / benchmark / "responses" / f"{run['variant']}.jsonl"
        )
        if not response_file.exists() or args.force:
          answers = clean_benchmarks._chunked_generate(  # pylint: disable=protected-access
              generator,
              prompts=benchmark_prompts[benchmark],
              temperature=0.0,
              seed=args.seed,
              batch_size=args.generation_batch_size,
          )
          response_rows = []
          if benchmark == "xstest":
            for index, (row, answer) in enumerate(
                zip(benchmark_rows[benchmark], answers, strict=True)
            ):
              response_rows.append(
                  {
                      "question_id": str(index),
                      "prompt": row["prompt"],
                      "response": answer,
                  }
              )
          else:
            for row, answer in zip(
                benchmark_rows[benchmark], answers, strict=True
            ):
              response_rows.append(
                  {
                      "question_id": str(row["id"]),
                      "prompt": _harmbench_prompt(row),
                      "response": answer,
                  }
              )
          clean_benchmarks._write_jsonl(  # pylint: disable=protected-access
              response_file,
              response_rows,
          )
        result_dir = output_root / benchmark / "results" / run["variant"]
        subprocess.run(
            [
                str(offline_python),
                str(HELPER_PATH),
                "--benchmark",
                benchmark,
                "--input-data",
                str(output_root / "data" / f"{benchmark}.jsonl"),
                "--input-response-data",
                str(response_file),
                "--output-dir",
                str(result_dir),
                "--judge-model",
                args.judge_model,
                "--batch-size",
                str(args.judge_batch_size),
            ],
            check=True,
            cwd=str(repo_root),
        )
        result_summary = json.loads((result_dir / "summary.json").read_text())
        method_scores.update(result_summary)
    finally:
      generator.close()
    summary["scores"][run["variant"]] = {
        **method_scores,
        "run_name": run["run_name"],
        "variant": run["variant"],
    }

  summary_path = output_root / "safety_benchmarks_summary.json"
  summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
  print(f"Wrote safety benchmark summary to {summary_path}")


if __name__ == "__main__":
  main()
