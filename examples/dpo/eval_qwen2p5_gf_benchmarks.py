#!/usr/bin/env python3

"""Evaluate GF20/GF40 DPO runs on offline instruction-following benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from omegaconf import OmegaConf

REPO_ROOT = Path("/home/lhf_hongfu_gmail_com/tunix")
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from examples.dpo import eval_qwen2p5_clean_main_table_runs as eval_lib
from examples.dpo import qwen2p5_clean_main_table_lib as table_lib


DATASET_ORDER = ("global_flip20", "global_flip40")
METHOD_ORDER = (
    "vanilla_dpo",
    "random_pair_filtering_filt5",
    "random_pair_filtering",
    "reward_based_filtering_filt5",
    "reward_based_filtering",
    "self_inf",
)
METHOD_DISPLAY = {
    "vanilla_dpo": "Vanilla DPO",
    "random_pair_filtering_filt5": "Random Pair Filtering (5%)",
    "random_pair_filtering": "Random Pair Filtering (10%)",
    "reward_based_filtering_filt5": "Reward-based Filtering (5%)",
    "reward_based_filtering": "Reward-based Filtering (10%)",
    "self_inf": "Self-DTV (ours)",
}
DATASET_DISPLAY = {
    "global_flip20": "GF20",
    "global_flip40": "GF40",
}
METRIC_COLUMNS = (
    "clean_val_acc_auc",
    "clean_test_acc",
    "livebench_if_score",
    "rewardbench2_precise_if",
    "ifbench_prompt_strict",
)
METRIC_DISPLAY = {
    "clean_val_acc_auc": "Val Acc-AUC",
    "clean_test_acc": "Test Acc",
    "livebench_if_score": "LiveBench-IF",
    "rewardbench2_precise_if": "RB2 Precise IF",
    "ifbench_prompt_strict": "IFBench P-Strict",
}


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run offline benchmark evaluation for GF20/GF40 DPO exports."
  )
  parser.add_argument("--repo-root", default=str(REPO_ROOT))
  parser.add_argument("--run-ts", default=table_lib.LEGACY_RUN_TS)
  parser.add_argument(
      "--datasets",
      nargs="+",
      default=list(DATASET_ORDER),
      choices=list(DATASET_ORDER),
  )
  parser.add_argument(
      "--methods",
      nargs="+",
      default=list(METHOD_ORDER),
      choices=list(METHOD_ORDER),
  )
  parser.add_argument(
      "--benchmarks",
      nargs="+",
      default=("preference", "livebench_if", "rewardbench2", "ifbench"),
      choices=("preference", "livebench_if", "rewardbench2", "ifbench"),
  )
  parser.add_argument(
      "--output-root",
      default=None,
      help="Defaults to runs/results/qwen2p5_gf_benchmarks_<run-ts>.",
  )
  parser.add_argument("--config-path", default=str(table_lib.CONFIG_PATH))
  parser.add_argument("--sft-model-path", default=str(table_lib.SFT_MODEL_PATH))
  parser.add_argument("--force", action="store_true")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--eval-limit", type=int, default=None)
  parser.add_argument("--num-batches", type=int, default=None)
  parser.add_argument("--rewardbench-batch-size", type=int, default=8)
  parser.add_argument("--generation-batch-size", type=int, default=8)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--max-prompt-length", type=int, default=4096)
  parser.add_argument("--max-generation-steps", type=int, default=1024)
  parser.add_argument("--top-k", type=int, default=50)
  parser.add_argument("--top-p", type=float, default=0.95)
  parser.add_argument(
      "--offline-eval-venv",
      default=str(eval_lib.DEFAULT_OFFLINE_EVAL_VENV),
  )
  parser.add_argument("--livebench-root", default=str(eval_lib.LIVEBENCH_ROOT))
  return parser.parse_args()


def _run_key(dataset: str, method: str) -> str:
  return f"{dataset}_{method}"


def _discover_runs(
    *,
    repo_root: Path,
    run_ts: str,
    datasets: list[str],
    methods: list[str],
) -> list[dict[str, Any]]:
  runs_root = repo_root / "runs"
  runs = []
  for dataset in datasets:
    for method in methods:
      run_name = (
          "dpo_qwen2p5_1p5b_ultrafeedback_from_sft_"
          f"{method}_{dataset}_lora_full_{run_ts}"
      )
      run_dir = runs_root / run_name
      exported_model = run_dir / "exported_model"
      if not exported_model.is_dir():
        raise SystemExit(f"Missing exported_model for {run_name}: {exported_model}")
      runs.append(
          {
              "dataset": dataset,
              "dataset_display": DATASET_DISPLAY[dataset],
              "logical_variant": method,
              "display_name": METHOD_DISPLAY[method],
              "run_key": _run_key(dataset, method),
              "run_name": run_name,
              "run_dir": str(run_dir),
              "tensorboard_dir": str(run_dir / "tensorboard"),
              "exported_model_path": str(exported_model),
          }
      )
  return runs


def _load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text()) if path.exists() else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _format_float(value: float | None) -> str:
  return "" if value is None else f"{value:.4f}"


def _write_tables(output_root: Path, summary: dict[str, Any]) -> None:
  tables_dir = output_root / "tables"
  tables_dir.mkdir(parents=True, exist_ok=True)
  header = ["Dataset", "Method"] + [METRIC_DISPLAY[col] for col in METRIC_COLUMNS]
  md_lines = [
      "| " + " | ".join(f"{name} $\\uparrow$" if name not in ("Dataset", "Method") else name for name in header) + " |",
      "|" + "|".join(["---", "---", *("---:" for _ in METRIC_COLUMNS)]) + "|",
  ]
  tex_rows = []
  for dataset in DATASET_ORDER:
    if dataset not in summary["scores"]:
      continue
    for method in METHOD_ORDER:
      if method not in summary["scores"][dataset]:
        continue
      row = summary["scores"][dataset][method]
      values = [_format_float(row.get(col)) for col in METRIC_COLUMNS]
      md_lines.append(
          "| "
          + " | ".join([DATASET_DISPLAY[dataset], METHOD_DISPLAY[method], *values])
          + " |"
      )
      tex_rows.append(
          " & ".join([DATASET_DISPLAY[dataset], METHOD_DISPLAY[method], *values])
          + r" \\"
      )
  (tables_dir / "gf_benchmarks.md").write_text("\n".join(md_lines) + "\n")
  tex = "\n".join(
      [
          r"\begin{table*}[t]",
          r"\centering",
          r"\footnotesize",
          r"\setlength{\tabcolsep}{5pt}",
          r"\resizebox{\textwidth}{!}{%",
          r"\begin{tabular}{llccccc}",
          r"\toprule",
          r"Dataset & Method & Val Acc-AUC $\uparrow$ & Test Acc $\uparrow$ & LiveBench-IF $\uparrow$ & RB2 Precise IF $\uparrow$ & IFBench P-Strict $\uparrow$ \\",
          r"\midrule",
          *tex_rows,
          r"\bottomrule",
          r"\end{tabular}%",
          r"}",
          r"\caption{Offline benchmark results for GF20 and GF40 DPO runs.}",
          r"\label{tab:gf_benchmarks}",
          r"\end{table*}",
          "",
      ]
  )
  (tables_dir / "gf_benchmarks.tex").write_text(tex)


def main() -> None:
  args = _parse_args()
  repo_root = Path(args.repo_root).resolve()
  output_root = (
      Path(args.output_root).resolve()
      if args.output_root
      else repo_root / "runs" / "results" / f"qwen2p5_gf_benchmarks_{args.run_ts}"
  )
  output_root.mkdir(parents=True, exist_ok=True)
  per_run_dir = output_root / "per_run"
  per_run_dir.mkdir(parents=True, exist_ok=True)

  runs = _discover_runs(
      repo_root=repo_root,
      run_ts=args.run_ts,
      datasets=list(args.datasets),
      methods=list(args.methods),
  )
  if args.dry_run:
    for run in runs:
      print(f"{run['run_key']}\t{run['exported_model_path']}")
    return

  clean_benchmarks = eval_lib._load_helper_module(  # pylint: disable=protected-access
      eval_lib.IFEVAL_HELPER_MODULE, "eval_qwen2p5_clean_benchmarks"
  )
  rewardbench_module = eval_lib._load_helper_module(  # pylint: disable=protected-access
      REPO_ROOT / "examples/dpo/eval_qwen2p5_rewardbench_v2.py",
      "eval_qwen2p5_rewardbench_v2",
  )
  base_cfg = OmegaConf.load(args.config_path)
  selected = set(args.benchmarks)

  rewardbench_rows = None
  if "rewardbench2" in selected:
    rewardbench_rows = rewardbench_module._load_rewardbench_v2_rows(  # pylint: disable=protected-access
        args.eval_limit
    )
  livebench_rows = None
  if "livebench_if" in selected:
    livebench_rows = eval_lib._load_livebench_rows(args.eval_limit)  # pylint: disable=protected-access
    livebench_data_path = (
        output_root / "livebench_if" / "data" / "livebench_instruction_following.jsonl"
    )
    clean_benchmarks._write_jsonl(livebench_data_path, livebench_rows)  # pylint: disable=protected-access
  ifbench_prompts = None
  if "ifbench" in selected:
    ifbench_assets = clean_benchmarks._prepare_ifbench_assets(  # pylint: disable=protected-access
        output_root / "ifbench"
    )
    ifbench_prompts = clean_benchmarks._load_jsonl(  # pylint: disable=protected-access
        Path(ifbench_assets["input_path"]),
        args.eval_limit,
    )

  summary_path = output_root / "gf_benchmark_summary.json"
  summary = _load_json(summary_path)
  summary.setdefault("run_ts", args.run_ts)
  summary.setdefault("datasets", list(args.datasets))
  summary.setdefault("methods", list(args.methods))
  summary.setdefault("benchmarks", list(args.benchmarks))
  summary.setdefault("scores", {})

  for run in runs:
    row_path = per_run_dir / f"{run['run_key']}.json"
    cached = _load_json(row_path) if row_path.exists() and not args.force else {}
    metrics = dict(cached.get("metrics", {}))
    if "preference" in selected and (
        "clean_val_acc_auc" not in metrics or args.force
    ):
      metrics["clean_val_acc_auc"] = eval_lib._compute_normalized_scalar_auc(  # pylint: disable=protected-access
          Path(run["tensorboard_dir"]),
          tag="dpo/eval/rewards/accuracy",
      )
    if "preference" in selected and (
        "clean_test_acc" not in metrics or args.force
    ):
      metrics["clean_test_acc"] = eval_lib._evaluate_clean_test_acc(  # pylint: disable=protected-access
          run=run,
          base_cfg=base_cfg,
          sft_model_path=args.sft_model_path,
          eval_limit=args.eval_limit,
          num_batches=args.num_batches,
      )
    if "livebench_if" in selected and (
        "livebench_if_score" not in metrics or args.force
    ):
      metrics["livebench_if_score"] = eval_lib._evaluate_livebench_if(  # pylint: disable=protected-access
          clean_benchmarks=clean_benchmarks,
          repo_root=repo_root,
          run=run,
          output_root=output_root,
          rows=livebench_rows or [],
          args=args,
      )
    if "rewardbench2" in selected and (
        "rewardbench2_precise_if" not in metrics or args.force
    ):
      metrics["rewardbench2_precise_if"] = eval_lib._evaluate_rewardbench2(  # pylint: disable=protected-access
          rewardbench_module=rewardbench_module,
          run=run,
          rewardbench_rows=rewardbench_rows or [],
          base_cfg=base_cfg,
          sft_model_path=args.sft_model_path,
          batch_size=args.rewardbench_batch_size,
          output_root=output_root,
          force=args.force,
      )
    if "ifbench" in selected and (
        "ifbench_prompt_strict" not in metrics or args.force
    ):
      metrics["ifbench_prompt_strict"] = eval_lib._evaluate_ifbench(  # pylint: disable=protected-access
          clean_benchmarks=clean_benchmarks,
          run=run,
          output_root=output_root,
          prompts=ifbench_prompts or [],
          args=args,
      )

    payload = {
        "dataset": run["dataset"],
        "dataset_display": run["dataset_display"],
        "logical_variant": run["logical_variant"],
        "display_name": run["display_name"],
        "run_key": run["run_key"],
        "run_name": run["run_name"],
        "run_dir": run["run_dir"],
        "metrics": metrics,
    }
    _write_json(row_path, payload)
    summary["scores"].setdefault(run["dataset"], {})[run["logical_variant"]] = metrics
    _write_json(summary_path, summary)
    print(
        f"{run['run_key']}: "
        + ", ".join(
            f"{metric}={metrics[metric]:.4f}"
            for metric in METRIC_COLUMNS
            if metric in metrics
        )
    )

  _write_tables(output_root, summary)
  print(f"Wrote GF benchmark summary to {summary_path}")


if __name__ == "__main__":
  main()
