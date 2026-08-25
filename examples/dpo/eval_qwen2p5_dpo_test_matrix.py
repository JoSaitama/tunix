#!/usr/bin/env python3

"""Evaluates exported Qwen2.5 DPO-from-SFT runs on UltraFeedback test_prefs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from absl import logging
from omegaconf import OmegaConf

from examples.dpo import qwen2p5_dpo_eval_lib


RUN_RE = re.compile(
    r"dpo_qwen2p5_1p5b_ultrafeedback_from_sft_"
    r"(?P<variant>.+)_(?P<dataset>clean|global_flip20|global_flip40)_"
    r"lora_full_(?P<run_ts>\d{8}_\d{6})"
)
VARIANT_ORDER = {
    "vanilla_dpo": 0,
    "random_pair_filtering": 1,
    "reward_based_filtering": 2,
    "self_inf": 3,
}
DATASET_ORDER = {
    "clean": 0,
    "global_flip20": 1,
    "global_flip40": 2,
}


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Evaluate exported Qwen2.5 DPO-from-SFT runs on clean test_prefs "
          "without retraining or writing checkpoints."
      )
  )
  parser.add_argument(
      "--repo-root",
      default="/home/lhf_hongfu_gmail_com/tunix",
      help="Repository root containing runs/ and examples/.",
  )
  parser.add_argument(
      "--config-path",
      default=(
          "/home/lhf_hongfu_gmail_com/tunix/examples/dpo/"
          "qwen2p5_1p5b_ultrafeedback_from_sft_lora.yaml"
      ),
      help="Base config used to inherit tokenizer, mesh, and DPO hyperparameters.",
  )
  parser.add_argument(
      "--sft-model-path",
      default=(
          "/home/lhf_hongfu_gmail_com/tunix/runs/"
          "sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/"
          "exported_model"
      ),
      help="Reference SFT exported model path.",
  )
  parser.add_argument(
      "--run-ts",
      default="20260417_013847",
      help="Only evaluate runs whose suffix timestamp matches this value.",
  )
  parser.add_argument(
      "--run-pattern",
      default="dpo_qwen2p5_1p5b_ultrafeedback_from_sft_*",
      help="Glob for run directories under runs/.",
  )
  parser.add_argument(
      "--eval-limit",
      type=int,
      default=None,
      help="Optional example limit for test_prefs, useful for quick validation.",
  )
  parser.add_argument(
      "--num-batches",
      type=int,
      default=None,
      help="Optional cap on the number of eval batches after batching.",
  )
  parser.add_argument(
      "--output-json",
      default=(
          "/home/lhf_hongfu_gmail_com/tunix/runs/results/"
          "qwen2p5_dpo_test_matrix_20260417_013847.json"
      ),
      help="Path to write the full JSON summary.",
  )
  return parser.parse_args()


def _build_test_module_spec(limit: int | None) -> str:
  if limit is None:
    return "examples/data/ultrafeedback_dpo.py:create_dataset(split='test_prefs', seed=42)"
  return (
      "examples/data/ultrafeedback_dpo.py:create_dataset("
      f"split='test_prefs', seed=42, limit={limit})"
  )


def _discover_runs(repo_root: Path, run_pattern: str, run_ts: str) -> list[dict[str, Any]]:
  candidates = []
  for run_dir in sorted((repo_root / "runs").glob(run_pattern)):
    if not run_dir.is_dir():
      continue
    match = RUN_RE.fullmatch(run_dir.name)
    if match is None or match.group("run_ts") != run_ts:
      continue
    exported_model = run_dir / "exported_model"
    if not exported_model.is_dir():
      continue
    variant = match.group("variant")
    dataset = match.group("dataset")
    candidates.append(
        {
            "run_dir": run_dir,
            "run_name": run_dir.name,
            "variant": variant,
            "dataset": dataset,
            "exported_model_path": exported_model,
        }
    )
  return sorted(
      candidates,
      key=lambda x: (
          VARIANT_ORDER.get(x["variant"], 999),
          DATASET_ORDER.get(x["dataset"], 999),
      ),
  )


def _evaluate_run(
    *,
    run_info: dict[str, Any],
    base_cfg: Any,
    sft_model_path: str,
    eval_module_spec: str,
    eval_limit: int | None,
    num_batches: int | None,
) -> dict[str, Any]:
  logging.info(
      "Evaluating %s on test_prefs with actor=%s ref=%s",
      run_info["run_name"],
      run_info["exported_model_path"],
      sft_model_path,
  )

  bundle = qwen2p5_dpo_eval_lib.load_eval_bundle(
      base_cfg=base_cfg,
      actor_model_path=str(run_info["exported_model_path"]),
      reference_model_path=sft_model_path,
      metrics_prefix="dpo_test",
  )
  eval_dataset = qwen2p5_dpo_eval_lib.load_eval_dataset(
      module_spec=eval_module_spec,
      tokenizer=bundle.tokenizer,
      batch_size=int(base_cfg.get("eval_batch_size", base_cfg["batch_size"])),
      num_batches=num_batches,
      dpo_config=bundle.training_config,
  )
  trainer, metrics_logger = qwen2p5_dpo_eval_lib.create_eval_trainer(bundle)
  with bundle.actor_mesh:
    metrics = qwen2p5_dpo_eval_lib.aggregate_eval_metrics(
        trainer=trainer,
        metrics_logger=metrics_logger,
        eval_dataset=eval_dataset,
    )

  summary = {
      "variant": run_info["variant"],
      "dataset": run_info["dataset"],
      "run_name": run_info["run_name"],
      "eval_split": "test_prefs",
      "eval_limit": eval_limit,
      "num_batches": num_batches,
      "metrics": metrics,
  }

  qwen2p5_dpo_eval_lib.close_eval_trainer(trainer, metrics_logger)
  qwen2p5_dpo_eval_lib.close_eval_bundle(bundle)
  return summary


def main() -> None:
  args = _parse_args()
  repo_root = Path(args.repo_root).resolve()
  base_cfg = OmegaConf.load(args.config_path)
  eval_module_spec = _build_test_module_spec(args.eval_limit)
  output_path = Path(args.output_json)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  runs = _discover_runs(repo_root, args.run_pattern, args.run_ts)
  if not runs:
    raise SystemExit(
      f"No run directories matched run_pattern={args.run_pattern!r} run_ts={args.run_ts!r}."
    )

  print(f"Evaluating {len(runs)} runs on {eval_module_spec}")
  results = []
  for index, run_info in enumerate(runs, start=1):
    print(
        f"[{index:02d}/{len(runs):02d}] "
        f"{run_info['variant']} + {run_info['dataset']}"
    )
    results.append(
        _evaluate_run(
            run_info=run_info,
            base_cfg=base_cfg,
            sft_model_path=args.sft_model_path,
            eval_module_spec=eval_module_spec,
            eval_limit=args.eval_limit,
            num_batches=args.num_batches,
        )
    )

  output = {
      "eval_split": "test_prefs",
      "eval_module_spec": eval_module_spec,
      "run_ts": args.run_ts,
      "results": results,
  }
  output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
  print(f"Wrote JSON summary to {output_path}")
  print(
      "variant\tdataset\trewards_accuracy\trewards_margin\tloss\tperplexity"
  )
  for row in results:
    metrics = row["metrics"]
    print(
        "\t".join(
            [
                row["variant"],
                row["dataset"],
                f"{metrics['rewards_accuracy']:.4f}",
                f"{metrics['rewards_margin']:.5f}",
                f"{metrics['loss']:.5f}",
                f"{metrics['perplexity']:.5f}",
            ]
        )
    )


if __name__ == "__main__":
  main()
