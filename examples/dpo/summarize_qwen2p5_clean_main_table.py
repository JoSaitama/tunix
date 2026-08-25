#!/usr/bin/env python3

"""Aggregate the multi-seed clean DPO main table into mean +- std rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path("/home/lhf_hongfu_gmail_com/tunix")
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from examples.dpo import qwen2p5_clean_main_table_lib as table_lib


COLUMN_SPECS = {
    "clean_val_acc_auc": ("Val Acc-AUC $\\uparrow$", 4),
    "clean_test_acc": ("Test Acc $\\uparrow$", 4),
    "livebench_if_score": ("LiveBench-IF $\\uparrow$", 4),
    "rewardbench2_precise_if": ("RB2 Precise IF $\\uparrow$", 4),
    "ifbench_prompt_strict": ("IFBench P-Strict $\\uparrow$", 4),
}


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Aggregate per-run clean DPO main-table metrics into mean +- std "
          "rows and render Markdown/LaTeX outputs."
      )
  )
  parser.add_argument("--repo-root", default=str(table_lib.REPO_ROOT))
  parser.add_argument("--run-ts", required=True)
  parser.add_argument(
      "--input-json",
      default=None,
      help="Defaults to runs/results/qwen2p5_clean_main_table_<run-ts>/per_run_metrics.json.",
  )
  parser.add_argument(
      "--output-dir",
      default=None,
      help="Defaults to runs/results/qwen2p5_clean_main_table_<run-ts>/tables.",
  )
  parser.add_argument(
      "--columns",
      nargs="+",
      default=list(table_lib.MAIN_TABLE_COLUMNS),
  )
  return parser.parse_args()


def load_per_run_metrics(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text())
  return payload["runs"]


def aggregate_rows(
    *,
    per_run_metrics: dict[str, Any],
    columns: tuple[str, ...],
) -> list[dict[str, Any]]:
  rows = []
  for logical_variant in table_lib.METHOD_ORDER:
    grouped = [
        row
        for row in per_run_metrics.values()
        if row["logical_variant"] == logical_variant
    ]
    if not grouped:
      continue
    grouped.sort(key=lambda row: row["seed"])
    row_summary: dict[str, Any] = {
        "method": logical_variant,
        "display_name": table_lib.METHOD_DISPLAY[logical_variant],
        "seed_count": len(grouped),
        "seeds": [int(row["seed"]) for row in grouped],
        "metrics": {},
    }
    for column in columns:
      values = [float(row["metrics"][column]) for row in grouped if column in row["metrics"]]
      if not values:
        row_summary["metrics"][column] = None
        continue
      row_summary["metrics"][column] = {
          "mean": float(np.mean(values)),
          "std": float(np.std(values)),
          "values": values,
      }
    rows.append(row_summary)
  return rows


def _format_mean_std(mean: float, std: float, digits: int) -> str:
  return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _metric_mean(row: dict[str, Any], column: str) -> float | None:
  metric = row["metrics"].get(column)
  if metric is None:
    return None
  return float(metric["mean"])


def render_latex_table(
    rows: list[dict[str, Any]],
    *,
    columns: tuple[str, ...],
) -> str:
  best_by_column: dict[str, float] = {}
  for column in columns:
    means = [
        _metric_mean(row, column) for row in rows if _metric_mean(row, column) is not None
    ]
    if means:
      best_by_column[column] = max(means)

  header_cells = ["Method"] + [COLUMN_SPECS[column][0] for column in columns]
  latex_rows = []
  for row in rows:
    cells = [row["display_name"]]
    for column in columns:
      metric = row["metrics"].get(column)
      if metric is None:
        cells.append("--")
        continue
      digits = COLUMN_SPECS[column][1]
      rendered = _format_mean_std(metric["mean"], metric["std"], digits)
      if np.isclose(metric["mean"], best_by_column.get(column, float("-inf"))):
        rendered = f"\\textbf{{{rendered}}}"
      cells.append(rendered)
    latex_rows.append(" & ".join(cells) + " \\\\")

  body = "\n".join(latex_rows)
  return (
      "\\begin{table*}[t]\n"
      "\\centering\n"
      "\\footnotesize\n"
      "\\setlength{\\tabcolsep}{4.5pt}\n"
      "\\renewcommand{\\arraystretch}{1.05}\n"
      "\\resizebox{\\textwidth}{!}{%\n"
      f"\\begin{{tabular}}{{l{'c' * len(columns)}}}\n"
      "\\toprule\n"
      + " & ".join(header_cells)
      + " \\\\\n"
      "\\midrule\n"
      + body
      + "\n\\bottomrule\n"
      "\\end{tabular}%\n"
      "}\n"
      "\\caption{\n"
      "Clean-setting main results with three DPO seeds. Reported values are mean $\\pm$ std.\n"
      "Bold indicates the best mean in each column.\n"
      "}\n"
      "\\label{tab:clean_main_selfdtv_multiseed}\n"
      "\\end{table*}\n"
  )


def render_markdown_table(
    rows: list[dict[str, Any]],
    *,
    columns: tuple[str, ...],
) -> str:
  header = ["Method"] + [COLUMN_SPECS[column][0] for column in columns]
  lines = [
      "| " + " | ".join(header) + " |",
      "|" + "---|" * len(header),
  ]
  for row in rows:
    cells = [row["display_name"]]
    for column in columns:
      metric = row["metrics"].get(column)
      if metric is None:
        cells.append("--")
        continue
      digits = COLUMN_SPECS[column][1]
      cells.append(_format_mean_std(metric["mean"], metric["std"], digits))
    lines.append("| " + " | ".join(cells) + " |")
  return "\n".join(lines) + "\n"


def main() -> None:
  args = _parse_args()
  columns = tuple(args.columns)
  repo_root = Path(args.repo_root).resolve()
  input_path = (
      Path(args.input_json).resolve()
      if args.input_json
      else repo_root
      / "runs"
      / "results"
      / f"qwen2p5_clean_main_table_{args.run_ts}"
      / "per_run_metrics.json"
  )
  output_dir = (
      Path(args.output_dir).resolve()
      if args.output_dir
      else repo_root
      / "runs"
      / "results"
      / f"qwen2p5_clean_main_table_{args.run_ts}"
      / "tables"
  )
  output_dir.mkdir(parents=True, exist_ok=True)

  per_run_metrics = load_per_run_metrics(input_path)
  rows = aggregate_rows(per_run_metrics=per_run_metrics, columns=columns)
  payload = {
      "run_ts": args.run_ts,
      "columns": list(columns),
      "rows": rows,
  }
  json_path = output_dir / "clean_main_table_multiseed.json"
  markdown_path = output_dir / "clean_main_table_multiseed.md"
  latex_path = output_dir / "clean_main_table_multiseed.tex"
  json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
  markdown_path.write_text(render_markdown_table(rows, columns=columns))
  latex_path.write_text(render_latex_table(rows, columns=columns))
  print(f"Wrote aggregated JSON to {json_path}")
  print(f"Wrote Markdown table to {markdown_path}")
  print(f"Wrote LaTeX table to {latex_path}")


if __name__ == "__main__":
  main()
