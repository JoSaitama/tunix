#!/usr/bin/env python3

"""Build the clean paper table from local metrics and benchmark summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/home/lhf_hongfu_gmail_com/tunix")
METHOD_ORDER = (
    "vanilla_dpo",
    "random_pair_filtering",
    "reward_based_filtering",
    "self_inf",
)
METHOD_DISPLAY = {
    "vanilla_dpo": "Vanilla DPO",
    "random_pair_filtering": "Random Pair Filtering",
    "reward_based_filtering": "Reward-based Filtering",
    "self_inf": "Self-inf (ours)",
}
COLUMN_SPECS = {
    "clean_val_acc_auc": ("Clean Val Acc-AUC $\\uparrow$", 4),
    "clean_test_acc": ("Clean Test Acc $\\uparrow$", 4),
    "rewardbench_overall": ("RewardBench Overall $\\uparrow$", 4),
    "rewardbench_chat": ("RewardBench Chat $\\uparrow$", 4),
    "rewardbench_chat_hard": ("RewardBench Chat Hard $\\uparrow$", 4),
    "rewardbench_safety": ("RewardBench Safety $\\uparrow$", 4),
    "rewardbench_reasoning": ("RewardBench Reasoning $\\uparrow$", 4),
    "rewardbench2_macro_score": ("RewardBench 2 Macro $\\uparrow$", 4),
    "rewardbench2_weighted_score": ("RewardBench 2 Weighted $\\uparrow$", 4),
    "ifeval_prompt_strict": ("IFEval Prompt Strict $\\uparrow$", 4),
    "ifeval_prompt_loose": ("IFEval Prompt Loose $\\uparrow$", 4),
    "ifeval_instruction_strict": ("IFEval Instruction Strict $\\uparrow$", 4),
    "ifeval_instruction_loose": ("IFEval Instruction Loose $\\uparrow$", 4),
    "ifbench_prompt_strict": ("IFBench Prompt Strict $\\uparrow$", 4),
    "ifbench_prompt_loose": ("IFBench Prompt Loose $\\uparrow$", 4),
    "ifbench_instruction_strict": ("IFBench Instruction Strict $\\uparrow$", 4),
    "ifbench_instruction_loose": ("IFBench Instruction Loose $\\uparrow$", 4),
    "livebench_if_score": ("LiveBench IF Score $\\uparrow$", 2),
    "livebench_if_prompt_strict": ("LiveBench IF Prompt Strict $\\uparrow$", 4),
    "xstest_overall_accuracy": ("XSTest Overall Acc. $\\uparrow$", 4),
    "xstest_inverted_rta_all_safe": ("XSTest Safe Compliance $\\uparrow$", 4),
    "xstest_rta_all_contrast": ("XSTest Unsafe Refusal $\\uparrow$", 4),
    "xstest_refusal_f1": ("XSTest Refusal F1 $\\uparrow$", 4),
    "harmbench_inverted_micro_asr_lower": ("HarmBench Inverted ASR $\\uparrow$", 4),
    "wildbench_adjusted_score": ("WildBench Score $\\uparrow$", 2),
    "wildbench_task_macro_score": ("WildBench Task Macro $\\uparrow$", 2),
    "mt_bench_avg_score": ("MT-Bench Avg. Score $\\uparrow$", 2),
    "alpacaeval2_lc_win_rate": ("AlpacaEval 2 LC Win Rate $\\uparrow$", 2),
    "arena_hard_score": ("Arena-Hard Score $\\uparrow$", 2),
}
DEFAULT_TABLE_COLUMNS = (
    "clean_val_acc_auc",
    "clean_test_acc",
    "rewardbench_overall",
    "ifeval_prompt_strict",
    "mt_bench_avg_score",
    "alpacaeval2_lc_win_rate",
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Merge clean local metrics with external benchmark summaries."
  )
  parser.add_argument("--repo-root", default=str(REPO_ROOT))
  parser.add_argument("--run-ts", required=True)
  parser.add_argument(
      "--clean-metrics-json",
      default=None,
      help="Defaults to runs/results/qwen2p5_clean_metrics_<run-ts>.json.",
  )
  parser.add_argument(
      "--test-matrix-json",
      default=None,
      help="Defaults to runs/results/qwen2p5_dpo_test_matrix_<run-ts>.json.",
  )
  parser.add_argument(
      "--benchmark-summary-json",
      default=None,
      help="Defaults to runs/results/clean_benchmarks_<run-ts>/benchmark_summary.json.",
  )
  parser.add_argument(
      "--rewardbench-summary-json",
      default=None,
      help="Defaults to runs/results/clean_benchmarks_<run-ts>/rewardbench_v1/rewardbench_summary.json.",
  )
  parser.add_argument(
      "--rewardbench-v2-summary-json",
      default=None,
      help="Defaults to runs/results/clean_benchmarks_<run-ts>/rewardbench_v2/rewardbench_v2_summary.json.",
  )
  parser.add_argument(
      "--livebench-summary-json",
      default=None,
      help="Defaults to runs/results/clean_benchmarks_<run-ts>/livebench_if/livebench_instruction_following_summary.json.",
  )
  parser.add_argument(
      "--safety-benchmarks-summary-json",
      default=None,
      help="Defaults to runs/results/clean_benchmarks_<run-ts>/safety_benchmarks/safety_benchmarks_summary.json.",
  )
  parser.add_argument(
      "--wildbench-summary-json",
      default=None,
      help="Defaults to runs/results/clean_benchmarks_<run-ts>/wildbench/wildbench_summary.json.",
  )
  parser.add_argument(
      "--output-dir",
      default=None,
      help="Defaults to runs/results/clean_benchmarks_<run-ts>/tables.",
  )
  parser.add_argument(
      "--columns",
      nargs="+",
      default=list(DEFAULT_TABLE_COLUMNS),
      help=(
          "Column ids to include. Supported values: "
          + ", ".join(COLUMN_SPECS)
      ),
  )
  return parser.parse_args()


def load_clean_metrics(path: Path) -> dict[str, dict[str, float]]:
  data = json.loads(path.read_text())
  return data["metrics"]


def load_clean_test_metrics(path: Path) -> dict[str, float]:
  data = json.loads(path.read_text())
  metrics = {}
  for row in data["results"]:
    if row["dataset"] != "clean":
      continue
    metrics[row["variant"]] = float(row["metrics"]["rewards_accuracy"])
  return metrics


def load_benchmark_scores(
    benchmark_summary_path: Path | None,
    rewardbench_summary_path: Path | None,
    rewardbench_v2_summary_path: Path | None,
    livebench_summary_path: Path | None,
    safety_benchmarks_summary_path: Path | None,
    wildbench_summary_path: Path | None,
) -> dict[str, dict[str, float | None]]:
  scores = {
      method: {
          "rewardbench_overall": None,
          "rewardbench_chat": None,
          "rewardbench_chat_hard": None,
          "rewardbench_safety": None,
          "rewardbench_reasoning": None,
          "rewardbench2_macro_score": None,
          "rewardbench2_weighted_score": None,
          "ifeval_prompt_strict": None,
          "ifeval_prompt_loose": None,
          "ifeval_instruction_strict": None,
          "ifeval_instruction_loose": None,
          "ifbench_prompt_strict": None,
          "ifbench_prompt_loose": None,
          "ifbench_instruction_strict": None,
          "ifbench_instruction_loose": None,
          "livebench_if_score": None,
          "livebench_if_prompt_strict": None,
          "xstest_overall_accuracy": None,
          "xstest_inverted_rta_all_safe": None,
          "xstest_rta_all_contrast": None,
          "xstest_refusal_f1": None,
          "harmbench_inverted_micro_asr_lower": None,
          "wildbench_adjusted_score": None,
          "wildbench_task_macro_score": None,
          "mt_bench_avg_score": None,
          "alpacaeval2_lc_win_rate": None,
          "arena_hard_score": None,
      }
      for method in METHOD_ORDER
  }
  if benchmark_summary_path is not None and benchmark_summary_path.exists():
    data = json.loads(benchmark_summary_path.read_text())
    for method, row in data.get("scores", {}).items():
      if method not in scores:
        continue
      for key in scores[method]:
        if key in row and row[key] is not None:
          scores[method][key] = float(row[key])
  if rewardbench_summary_path is not None and rewardbench_summary_path.exists():
    data = json.loads(rewardbench_summary_path.read_text())
    for method, row in data.get("scores", {}).items():
      if method not in scores:
        continue
      for key in (
          "rewardbench_overall",
          "rewardbench_chat",
          "rewardbench_chat_hard",
          "rewardbench_safety",
          "rewardbench_reasoning",
      ):
        if key in row and row[key] is not None:
          scores[method][key] = float(row[key])
  if rewardbench_v2_summary_path is not None and rewardbench_v2_summary_path.exists():
    data = json.loads(rewardbench_v2_summary_path.read_text())
    for method, row in data.get("scores", {}).items():
      if method not in scores:
        continue
      for key in ("rewardbench2_macro_score", "rewardbench2_weighted_score"):
        if key in row and row[key] is not None:
          scores[method][key] = float(row[key])
  if livebench_summary_path is not None and livebench_summary_path.exists():
    data = json.loads(livebench_summary_path.read_text())
    for method, row in data.get("scores", {}).items():
      if method not in scores:
        continue
      for key in ("livebench_if_score", "livebench_if_prompt_strict"):
        if key in row and row[key] is not None:
          scores[method][key] = float(row[key])
  if (
      safety_benchmarks_summary_path is not None
      and safety_benchmarks_summary_path.exists()
  ):
    data = json.loads(safety_benchmarks_summary_path.read_text())
    for method, row in data.get("scores", {}).items():
      if method not in scores:
        continue
      for key in (
          "xstest_overall_accuracy",
          "xstest_inverted_rta_all_safe",
          "xstest_rta_all_contrast",
          "xstest_refusal_f1",
          "harmbench_inverted_micro_asr_lower",
      ):
        if key in row and row[key] is not None:
          scores[method][key] = float(row[key])
  if wildbench_summary_path is not None and wildbench_summary_path.exists():
    data = json.loads(wildbench_summary_path.read_text())
    for method, row in data.get("scores", {}).items():
      if method not in scores:
        continue
      for key in ("wildbench_adjusted_score", "wildbench_task_macro_score"):
        if key in row and row[key] is not None:
          scores[method][key] = float(row[key])
  return scores


def build_table_rows(
    *,
    clean_metrics: dict[str, dict[str, float]],
    clean_test_acc: dict[str, float],
    benchmark_scores: dict[str, dict[str, float | None]],
) -> list[dict[str, Any]]:
  rows = []
  for method in METHOD_ORDER:
    rows.append(
        {
            "method": method,
            "display_name": METHOD_DISPLAY[method],
            "clean_val_acc_auc": float(clean_metrics[method]["clean_val_acc_auc"]),
            "clean_test_acc": float(clean_test_acc[method]),
            "rewardbench_overall": benchmark_scores[method].get(
                "rewardbench_overall"
            ),
            "rewardbench_chat": benchmark_scores[method].get("rewardbench_chat"),
            "rewardbench_chat_hard": benchmark_scores[method].get(
                "rewardbench_chat_hard"
            ),
            "rewardbench_safety": benchmark_scores[method].get(
                "rewardbench_safety"
            ),
            "rewardbench_reasoning": benchmark_scores[method].get(
                "rewardbench_reasoning"
            ),
            "rewardbench2_macro_score": benchmark_scores[method].get(
                "rewardbench2_macro_score"
            ),
            "rewardbench2_weighted_score": benchmark_scores[method].get(
                "rewardbench2_weighted_score"
            ),
            "ifeval_prompt_strict": benchmark_scores[method].get(
                "ifeval_prompt_strict"
            ),
            "ifeval_prompt_loose": benchmark_scores[method].get(
                "ifeval_prompt_loose"
            ),
            "ifeval_instruction_strict": benchmark_scores[method].get(
                "ifeval_instruction_strict"
            ),
            "ifeval_instruction_loose": benchmark_scores[method].get(
                "ifeval_instruction_loose"
            ),
            "ifbench_prompt_strict": benchmark_scores[method].get(
                "ifbench_prompt_strict"
            ),
            "ifbench_prompt_loose": benchmark_scores[method].get(
                "ifbench_prompt_loose"
            ),
            "ifbench_instruction_strict": benchmark_scores[method].get(
                "ifbench_instruction_strict"
            ),
            "ifbench_instruction_loose": benchmark_scores[method].get(
                "ifbench_instruction_loose"
            ),
            "livebench_if_score": benchmark_scores[method].get(
                "livebench_if_score"
            ),
            "livebench_if_prompt_strict": benchmark_scores[method].get(
                "livebench_if_prompt_strict"
            ),
            "xstest_overall_accuracy": benchmark_scores[method].get(
                "xstest_overall_accuracy"
            ),
            "xstest_inverted_rta_all_safe": benchmark_scores[method].get(
                "xstest_inverted_rta_all_safe"
            ),
            "xstest_rta_all_contrast": benchmark_scores[method].get(
                "xstest_rta_all_contrast"
            ),
            "xstest_refusal_f1": benchmark_scores[method].get(
                "xstest_refusal_f1"
            ),
            "harmbench_inverted_micro_asr_lower": benchmark_scores[method].get(
                "harmbench_inverted_micro_asr_lower"
            ),
            "wildbench_adjusted_score": benchmark_scores[method].get(
                "wildbench_adjusted_score"
            ),
            "wildbench_task_macro_score": benchmark_scores[method].get(
                "wildbench_task_macro_score"
            ),
            "mt_bench_avg_score": benchmark_scores[method].get(
                "mt_bench_avg_score"
            ),
            "alpacaeval2_lc_win_rate": benchmark_scores[method].get(
                "alpacaeval2_lc_win_rate"
            ),
            "arena_hard_score": benchmark_scores[method].get("arena_hard_score"),
        }
    )
  return rows


def _rank_marks(
    rows: list[dict[str, Any]],
    key: str,
    *,
    decimals: int,
) -> dict[str, str]:
  values = [round(row[key], decimals) for row in rows if row[key] is not None]
  if not values:
    return {}
  unique = sorted(set(values), reverse=True)
  best = unique[0]
  second = unique[1] if len(unique) > 1 else None
  marks = {}
  for row in rows:
    value = row[key]
    if value is None:
      continue
    rounded = round(value, decimals)
    if rounded == best:
      marks[row["method"]] = "best"
    elif second is not None and rounded == second:
      marks[row["method"]] = "second"
  return marks


def _format_value(value: float | None, *, decimals: int) -> str:
  if value is None:
    return "--"
  return f"{value:.{decimals}f}"


def render_latex_table(
    rows: list[dict[str, Any]],
    columns: tuple[str, ...] = DEFAULT_TABLE_COLUMNS,
) -> str:
  marks_by_column = {
      key: _rank_marks(rows, key, decimals=COLUMN_SPECS[key][1]) for key in columns
  }
  lines = [
      "\\begin{table*}[t]",
      "\\centering",
      "\\small",
      "\\setlength{\\tabcolsep}{6pt}",
      "\\begin{tabular}{l" + ("c" * len(columns)) + "}",
      "\\toprule",
      "Method",
      " & " + " & ".join(COLUMN_SPECS[key][0] for key in columns) + " \\\\",
      "\\midrule",
  ]
  for row in rows:
    values = []
    for key in columns:
      text = _format_value(row[key], decimals=COLUMN_SPECS[key][1])
      mark = marks_by_column[key].get(row["method"])
      if mark == "best":
        text = f"\\textbf{{{text}}}"
      elif mark == "second":
        text = f"\\underline{{{text}}}"
      values.append(text)
    lines.append(f"{row['display_name']} & " + " & ".join(values) + " \\\\")
  lines.extend(
      [
          "\\bottomrule",
          "\\end{tabular}",
          "\\caption{",
          "Clean-setting main table. Clean validation and clean test metrics come from the local DPO runs; external benchmark columns come from the corresponding benchmark pipelines. Bold indicates the best displayed value and underline indicates the second-best displayed value in each column.",
          "}",
          "\\label{tab:clean_main_external}",
          "\\end{table*}",
      ]
  )
  return "\n".join(lines)


def main() -> None:
  args = _parse_args()
  repo_root = Path(args.repo_root).resolve()
  clean_metrics_json = (
      Path(args.clean_metrics_json).resolve()
      if args.clean_metrics_json
      else repo_root / "runs/results" / f"qwen2p5_clean_metrics_{args.run_ts}.json"
  )
  test_matrix_json = (
      Path(args.test_matrix_json).resolve()
      if args.test_matrix_json
      else repo_root / "runs/results" / f"qwen2p5_dpo_test_matrix_{args.run_ts}.json"
  )
  benchmark_summary_json = (
      Path(args.benchmark_summary_json).resolve()
      if args.benchmark_summary_json
      else repo_root
      / "runs/results"
      / f"clean_benchmarks_{args.run_ts}"
      / "benchmark_summary.json"
  )
  rewardbench_summary_json = (
      Path(args.rewardbench_summary_json).resolve()
      if args.rewardbench_summary_json
      else repo_root
      / "runs/results"
      / f"clean_benchmarks_{args.run_ts}"
      / "rewardbench_v1"
      / "rewardbench_summary.json"
  )
  rewardbench_v2_summary_json = (
      Path(args.rewardbench_v2_summary_json).resolve()
      if args.rewardbench_v2_summary_json
      else repo_root
      / "runs/results"
      / f"clean_benchmarks_{args.run_ts}"
      / "rewardbench_v2"
      / "rewardbench_v2_summary.json"
  )
  livebench_summary_json = (
      Path(args.livebench_summary_json).resolve()
      if args.livebench_summary_json
      else repo_root
      / "runs/results"
      / f"clean_benchmarks_{args.run_ts}"
      / "livebench_if"
      / "livebench_instruction_following_summary.json"
  )
  safety_benchmarks_summary_json = (
      Path(args.safety_benchmarks_summary_json).resolve()
      if args.safety_benchmarks_summary_json
      else repo_root
      / "runs/results"
      / f"clean_benchmarks_{args.run_ts}"
      / "safety_benchmarks"
      / "safety_benchmarks_summary.json"
  )
  wildbench_summary_json = (
      Path(args.wildbench_summary_json).resolve()
      if args.wildbench_summary_json
      else repo_root
      / "runs/results"
      / f"clean_benchmarks_{args.run_ts}"
      / "wildbench"
      / "wildbench_summary.json"
  )
  output_dir = (
      Path(args.output_dir).resolve()
      if args.output_dir
      else repo_root / "runs/results" / f"clean_benchmarks_{args.run_ts}" / "tables"
  )
  output_dir.mkdir(parents=True, exist_ok=True)
  columns = tuple(args.columns)
  unsupported_columns = [column for column in columns if column not in COLUMN_SPECS]
  if unsupported_columns:
    raise SystemExit(
        "Unsupported columns: "
        + ", ".join(unsupported_columns)
        + ". Supported columns are: "
        + ", ".join(COLUMN_SPECS)
    )

  rows = build_table_rows(
      clean_metrics=load_clean_metrics(clean_metrics_json),
      clean_test_acc=load_clean_test_metrics(test_matrix_json),
      benchmark_scores=load_benchmark_scores(
          benchmark_summary_json,
          rewardbench_summary_json,
          rewardbench_v2_summary_json,
          livebench_summary_json,
          safety_benchmarks_summary_json,
          wildbench_summary_json,
      ),
  )
  latex = render_latex_table(rows, columns=columns)
  json_path = output_dir / "clean_main_table.json"
  tex_path = output_dir / "clean_main_table.tex"
  json_path.write_text(
      json.dumps({"run_ts": args.run_ts, "columns": columns, "rows": rows}, indent=2)
  )
  tex_path.write_text(latex)
  print(f"Wrote JSON table data to {json_path}")
  print(f"Wrote LaTeX table to {tex_path}")


if __name__ == "__main__":
  main()
