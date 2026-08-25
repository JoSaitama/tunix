#!/usr/bin/env python3

"""Generate WildBench answers and stage OpenAI batch scoring jobs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
from typing import Any

from datasets import load_dataset


REPO_ROOT = Path("/home/lhf_hongfu_gmail_com/tunix")
DEFAULT_OFFLINE_EVAL_VENV = Path("/home/lhf_hongfu_gmail_com/.venvs/DPO-EVAL-OFFLINE")
WILDBENCH_ROOT = Path("/home/lhf_hongfu_gmail_com/.cache/WildBench")
EVAL_TEMPLATE_PATH = WILDBENCH_ROOT / "evaluation/eval_template.score.v2.md"


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
          "Generate WildBench v2 answers for clean runs and stage WB-Score "
          "OpenAI batch jobs."
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
  parser.add_argument("--wildbench-config", default="v2")
  parser.add_argument("--judge-model", default="gpt-4.1")
  parser.add_argument(
      "--output-root",
      default=None,
      help="Defaults to runs/results/clean_benchmarks_<run-ts>/wildbench.",
  )
  parser.add_argument("--question-limit", type=int, default=None)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--max-words-to-eval", type=int, default=1000)
  parser.add_argument("--max-prompt-length", type=int, default=4096)
  parser.add_argument("--max-generation-steps", type=int, default=1024)
  parser.add_argument("--top-k", type=int, default=50)
  parser.add_argument("--top-p", type=float, default=0.95)
  parser.add_argument("--submit-batch", action="store_true")
  parser.add_argument(
      "--offline-eval-venv",
      default=str(DEFAULT_OFFLINE_EVAL_VENV),
  )
  parser.add_argument("--force", action="store_true")
  return parser.parse_args()


def _load_rows(config_name: str, limit: int | None) -> list[dict[str, Any]]:
  rows = list(load_dataset("allenai/WildBench", config_name, split="test"))
  if limit is not None:
    rows = rows[:limit]
  return rows


def _json_array_len(path: Path) -> int:
  if not path.exists():
    return 0
  try:
    data = json.loads(path.read_text())
  except json.JSONDecodeError:
    return 0
  return len(data) if isinstance(data, list) else 0


def _generate_output(
    generator,
    row: dict[str, Any],
    *,
    seed: int,
) -> str:
  messages = [
      {"role": turn["role"], "content": turn["content"]}
      for turn in row["conversation_input"]
  ]
  return generator.generate(messages, temperature=0.0, seed=seed)


def _shorten(text: str, max_words: int) -> str:
  if max_words > 0 and len(text.split(" ")) > max_words:
    return " ".join(text.split(" ")[:max_words]) + "... (truncated)"
  return text


def _compose_history(row: dict[str, Any]) -> str:
  history = ""
  for turn in row["conversation_input"][:-1]:
    role = turn["role"].upper()
    history += f"{role}: {turn['content']}\n\n"
  return history


def _compose_checklist_markdown(row: dict[str, Any]) -> str:
  return "".join(f"- {item}\n" for item in row["checklist"])


def _render_score_prompt(
    row: dict[str, Any],
    model_output: str,
    *,
    eval_template: str,
    max_words_to_eval: int,
) -> str:
  prompt = eval_template
  prompt = prompt.replace(
      "{$history}",
      _shorten(_compose_history(row), max_words_to_eval),
  )
  prompt = prompt.replace(
      "{$user_query}",
      _shorten(row["conversation_input"][-1]["content"], max_words_to_eval),
  )
  prompt = prompt.replace(
      "{$model_output}",
      _shorten(model_output, max_words_to_eval),
  )
  prompt = prompt.replace(
      "{$checklist}",
      _compose_checklist_markdown(row),
  )
  return prompt


def _build_batch_submit_lines(
    rows: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
    *,
    variant: str,
    judge_model: str,
    eval_template: str,
    max_words_to_eval: int,
) -> list[dict[str, Any]]:
  generated_by_session = {
      row["session_id"]: row.get("output", "")
      for row in generated_rows
  }
  batch_lines = []
  for row in rows:
    session_id = row["session_id"]
    model_output = generated_by_session.get(session_id, "")
    prompt = _render_score_prompt(
        row,
        model_output,
        eval_template=eval_template,
        max_words_to_eval=max_words_to_eval,
    )
    batch_lines.append(
        {
            "custom_id": f"{session_id}||{variant}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": judge_model,
                "temperature": 0,
                "max_tokens": 1024,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}],
            },
        }
    )
  return batch_lines


def _summarize_completed_scores(
    rows: list[dict[str, Any]],
    result_json_path: Path,
) -> dict[str, float]:
  if not result_json_path.exists():
    return {}
  result_rows = json.loads(result_json_path.read_text())
  if not result_rows:
    return {}
  tag_by_session = {row["session_id"]: row["primary_tag"] for row in rows}
  scores_by_tag: dict[str, list[float]] = {}
  lengths: list[int] = []
  all_scores: list[float] = []
  for row in result_rows:
    score = float(row["score"])
    all_scores.append(score)
    tag = tag_by_session.get(row["session_id"], "Unknown")
    scores_by_tag.setdefault(tag, []).append(score)
    lengths.append(len(row.get("model_output", "")))
  tag_adjusted = {
      tag: ((sum(values) / len(values)) - 5.0) * 20.0
      for tag, values in scores_by_tag.items()
      if values
  }
  return {
      "wildbench_raw_score": sum(all_scores) / len(all_scores),
      "wildbench_adjusted_score": ((sum(all_scores) / len(all_scores)) - 5.0)
      * 20.0,
      "wildbench_task_macro_score": (
          sum(tag_adjusted.values()) / len(tag_adjusted) if tag_adjusted else 0.0
      ),
      "wildbench_avg_length": sum(lengths) / len(lengths) if lengths else 0.0,
  }


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
      / "wildbench"
  )
  output_root.mkdir(parents=True, exist_ok=True)
  summary_path = output_root / "wildbench_summary.json"
  offline_python = Path(args.offline_eval_venv).resolve() / "bin/python"
  runs = clean_benchmarks._discover_clean_runs(  # pylint: disable=protected-access
      repo_root,
      args.run_ts,
      [
          clean_benchmarks._normalize_method(method)  # pylint: disable=protected-access
          for method in args.methods
      ],
  )
  rows = _load_rows(args.wildbench_config, args.question_limit)
  previous_summary = (
      json.loads(summary_path.read_text()) if summary_path.exists() else {}
  )
  summary: dict[str, Any] = {
      "run_ts": args.run_ts,
      "dataset": "allenai/WildBench",
      "config": args.wildbench_config,
      "judge_model": args.judge_model,
      "num_examples": len(rows),
      "scores": dict(previous_summary.get("scores", {})),
  }
  eval_template = EVAL_TEMPLATE_PATH.read_text(encoding="utf-8")
  for run in runs:
    result_file = output_root / "result_dirs/wild_bench_v2" / f"{run['variant']}.json"
    batch_submit_path = (
        output_root
        / f"eval_results/score/eval={args.judge_model}/{run['variant']}.batch-submit.jsonl"
    )
    batch_submit_path.parent.mkdir(parents=True, exist_ok=True)
    if args.force or _json_array_len(result_file) != len(rows):
      generator = clean_benchmarks.TunixChatGenerator(
          config_path=str(clean_benchmarks.CONFIG_PATH),
          exported_model_path=run["exported_model_path"],
          max_prompt_length=args.max_prompt_length,
          max_generation_steps=args.max_generation_steps,
          top_k=args.top_k,
          top_p=args.top_p,
      )
      try:
        generated_rows = []
        for index, row in enumerate(rows):
          answer = _generate_output(
              generator,
              row,
              seed=args.seed + index,
          )
          generated_rows.append(
              {
                  "session_id": row["session_id"],
                  "generator": run["variant"],
                  "output": answer,
              }
          )
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(
            json.dumps(generated_rows, indent=2, ensure_ascii=False)
        )
      finally:
        generator.close()

    generated_rows = json.loads(result_file.read_text())
    batch_lines = _build_batch_submit_lines(
        rows,
        generated_rows,
        variant=run["variant"],
        judge_model=args.judge_model,
        eval_template=eval_template,
        max_words_to_eval=args.max_words_to_eval,
    )
    batch_submit_path.write_text(
        "".join(json.dumps(line) + "\n" for line in batch_lines),
        encoding="utf-8",
    )
    if args.submit_batch:
      subprocess.run(
          [
              str(offline_python),
              str(WILDBENCH_ROOT / "src/openai_batch_eval/submit_batch.py"),
              str(batch_submit_path),
          ],
          check=True,
          cwd=str(WILDBENCH_ROOT),
      )
    completed_json = batch_submit_path.with_suffix("").with_suffix(".json")
    summary["scores"][run["variant"]] = {
        "wildbench_batch_submit_file": str(batch_submit_path),
        "wildbench_result_file": str(result_file),
        "wildbench_submitted": bool(args.submit_batch),
        "run_name": run["run_name"],
        "variant": run["variant"],
        **_summarize_completed_scores(rows, completed_json),
    }

  summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
  print(f"Wrote WildBench summary to {summary_path}")


if __name__ == "__main__":
  main()
