#!/usr/bin/env python3

"""Run RewardBench v1 with a Tunix/JAX TPU-native DPO scorer."""

from __future__ import annotations

import argparse
import os
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any

from datasets import load_dataset
from omegaconf import OmegaConf

REPO_ROOT = Path("/home/lhf_hongfu_gmail_com/tunix")
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from examples.dpo import qwen2p5_dpo_eval_lib


CONFIG_PATH = (
    REPO_ROOT / "examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft_lora.yaml"
)
SFT_MODEL_PATH = (
    REPO_ROOT
    / "runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/exported_model"
)
RUN_GLOB = os.environ.get(
    "DPO_EVAL_RUN_GLOB",
    "dpo_qwen2p5_1p5b_ultrafeedback_from_sft_*_clean_lora_full_*",
)
RUN_RE = re.compile(
    os.environ.get(
        "DPO_EVAL_RUN_RE",
        r"dpo_qwen2p5_1p5b_ultrafeedback_from_sft_"
        r"(?P<variant>.+)_clean_lora_full_(?P<run_ts>\\d{8}_\\d{6})",
    )
)
METHOD_ORDER = {
    "vanilla_dpo": 0,
    "random_pair_filtering": 1,
    "reward_based_filtering": 2,
    "self_inf": 3,
    "self_inf_norm": 3.05,
    "self_inf_re_loo": 1.15,
}
METHOD_ALIASES = {
    "vanilla_dpo": "vanilla_dpo",
    "vanilla": "vanilla_dpo",
    "vanilla dpo": "vanilla_dpo",
    "random_pair_filtering": "random_pair_filtering",
    "random pair filtering": "random_pair_filtering",
    "reward_based_filtering": "reward_based_filtering",
    "reward-based filtering": "reward_based_filtering",
    "reward based filtering": "reward_based_filtering",
    "self_inf_norm": "self_inf_norm",
    "dtv norm": "self_inf_norm",
    "self dtv norm": "self_inf_norm",
    "self-dtv norm": "self_inf_norm",
    "self-dtv-norm": "self_inf_norm",
    "self influence norm": "self_inf_norm",
    "self-influence norm": "self_inf_norm",
    "self-influence-norm": "self_inf_norm",
    "self inf norm": "self_inf_norm",
    "self-inf-norm": "self_inf_norm",
    "self-inf norm": "self_inf_norm",
    "self_dtv_norm": "self_inf_norm",
    "dtv_norm": "self_inf_norm",
    "self_inf_norm_batch": "self_inf_norm",
    "self_inf": "self_inf",
    "self_inf_re_loo": "self_inf_re_loo",
    "self_inf_re_loo_batch": "self_inf_re_loo",
    "self inf re loo": "self_inf_re_loo",
    "self-inf-re-loo": "self_inf_re_loo",
    "self-inf re loo": "self_inf_re_loo",
    "self_inf_reliable_loo": "self_inf_re_loo",
    "self inf reliable loo": "self_inf_re_loo",
    "self-inf reliable loo": "self_inf_re_loo",
    "self-inf-reliable-loo": "self_inf_re_loo",
    "self inf reliable loo batch": "self_inf_re_loo",
    "self-inf-reliable-loo-batch": "self_inf_re_loo",
    "self_inf_reliable_loo_batch": "self_inf_re_loo",
    "self influence reliable loo": "self_inf_re_loo",
    "self-influence-reliable-loo": "self_inf_re_loo",
    "self_dtv_re_loo": "self_inf_re_loo",
    "dtv_re_loo": "self_inf_re_loo",
    "dtv re loo": "self_inf_re_loo",
    "reliable dtv loo": "self_inf_re_loo",
    "reliable-dtv-loo": "self_inf_re_loo",
    "self inf": "self_inf",
    "self-inf": "self_inf",
    "self-inf (ours)": "self_inf",
}
REWARDBENCH_REPO = Path("/home/lhf_hongfu_gmail_com/.cache/reward-bench")
REWARDBENCH_DATASET = "allenai/reward-bench"
REWARDBENCH_SPLIT = "filtered"
SCORING_BACKEND = "tunix_tpu_rewardbench_adapter"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Evaluate the four clean Qwen2.5 DPO variants on RewardBench v1 "
          "using local DPO implicit rewards."
      )
  )
  parser.add_argument("--repo-root", default=str(REPO_ROOT))
  parser.add_argument("--config-path", default=str(CONFIG_PATH))
  parser.add_argument("--sft-model-path", default=str(SFT_MODEL_PATH))
  parser.add_argument("--run-ts", required=True)
  parser.add_argument(
      "--methods",
      nargs="+",
      default=list(METHOD_ORDER),
      help="Subset of clean methods to include.",
  )
  parser.add_argument(
      "--rewardbench-repo",
      default=str(REWARDBENCH_REPO),
      help="Local clone used to source official RewardBench constants.",
  )
  parser.add_argument(
      "--output-root",
      default=None,
      help=(
          "Defaults to runs/results/clean_benchmarks_<run-ts>/rewardbench_v1 "
          "under repo root."
      ),
  )
  parser.add_argument(
      "--batch-size",
      type=int,
      default=8,
      help="Pair batch size for local TPU scoring.",
  )
  parser.add_argument(
      "--eval-limit",
      type=int,
      default=None,
      help="Optional debug cap on RewardBench examples.",
  )
  parser.add_argument(
      "--force",
      action="store_true",
      help="Recompute per-model outputs even if score files already exist.",
  )
  return parser.parse_args()



# Self-Inf-lambda compatibility.
# This method is implemented as a unified DTV-family score:
#   S_j(lambda) = g_j^T G_{-j} + lambda * ||g_j||^2
# The default experiment uses lambda = 0.5.
METHOD_ORDER["self_inf_lambda"] = 1.25
METHOD_ALIASES.update({
    "self_inf_lambda": "self_inf_lambda",
    "self inf lambda": "self_inf_lambda",
    "self-inf-lambda": "self_inf_lambda",
    "self-inf lambda": "self_inf_lambda",
    "self_inf_lambda_batch": "self_inf_lambda",
    "self inf lambda batch": "self_inf_lambda",
    "self-inf lambda batch": "self_inf_lambda",
    "self-inf-lambda-batch": "self_inf_lambda",
    "self influence lambda batch": "self_inf_lambda",
    "self-influence-lambda-batch": "self_inf_lambda",
    "dtv lambda batch": "self_inf_lambda",
    "dtv-lambda-batch": "self_inf_lambda",
    "self influence lambda": "self_inf_lambda",
    "self-influence-lambda": "self_inf_lambda",
    "self_dtv_lambda": "self_inf_lambda",
    "dtv_lambda": "self_inf_lambda",
    "dtv lambda": "self_inf_lambda",
})
if "METHOD_DISPLAY" in globals():
    METHOD_DISPLAY["self_inf_lambda"] = "Self-Inf-Lambda"
if "METHOD_DISPLAY_NAMES" in globals():
    METHOD_DISPLAY_NAMES["self_inf_lambda"] = "Self-Inf-Lambda"
if "DISPLAY_NAMES" in globals():
    DISPLAY_NAMES["self_inf_lambda"] = "Self-Inf-Lambda"
if "METHOD_NAMES" in globals():
    METHOD_NAMES["self_inf_lambda"] = "Self-Inf-Lambda"

def _normalize_method(method: str) -> str:
  normalized = method.strip().lower().replace("_", " ").replace("-", " ")
  normalized = re.sub(r"\s+", " ", normalized)
  if normalized not in METHOD_ALIASES:
    raise ValueError(
        f"Unsupported method {method!r}. Expected one of: "
        f"{', '.join(sorted(METHOD_ALIASES))}"
    )
  return METHOD_ALIASES[normalized]


def _discover_clean_runs(
    repo_root: Path,
    *,
    run_ts: str,
    methods: list[str],
) -> list[dict[str, Any]]:
  wanted = {_normalize_method(method) for method in methods}
  candidates = []
  for run_dir in sorted((repo_root / "runs").glob(RUN_GLOB)):
    if not run_dir.is_dir():
      continue
    match = RUN_RE.fullmatch(run_dir.name)
    if match is None or match.group("run_ts") != run_ts:
      continue
    variant = match.group("variant")
    if variant not in wanted:
      continue
    exported_model = run_dir / "exported_model"
    if not exported_model.is_dir():
      continue
    candidates.append(
        {
            "variant": variant,
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "exported_model_path": str(exported_model),
        }
    )
  candidates.sort(key=lambda item: METHOD_ORDER[item["variant"]])
  missing = sorted(wanted - {row["variant"] for row in candidates})
  if missing:
    raise SystemExit(
      "Missing clean exported_model directories for: "
      + ", ".join(missing)
      + f" (run_ts={run_ts})"
    )
  return candidates


def _load_rewardbench_reference_constants(
    repo_root: Path,
) -> tuple[dict[str, int], dict[str, list[str]]]:
  constants_path = repo_root / "rewardbench/constants.py"
  if not constants_path.exists():
    raise SystemExit(
        "RewardBench constants file is missing: "
        f"{constants_path}. Clone the official reward-bench repo first."
    )
  spec = importlib.util.spec_from_file_location(
      "rewardbench_constants", constants_path
  )
  if spec is None or spec.loader is None:
    raise SystemExit(f"Failed to load RewardBench constants from {constants_path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return dict(module.EXAMPLE_COUNTS), dict(module.SUBSET_MAPPING)


def _slugify_section(section: str) -> str:
  return section.strip().lower().replace(" ", "_")


def _calculate_section_scores(
    *,
    subset_accuracies: dict[str, float],
    example_counts: dict[str, int],
    subset_mapping: dict[str, list[str]],
) -> dict[str, float]:
  output = {}
  for section, subsets in subset_mapping.items():
    total_examples = 0
    total_weighted = 0.0
    for subset in subsets:
      if subset not in subset_accuracies or subset not in example_counts:
        continue
      total_examples += int(example_counts[subset])
      total_weighted += float(subset_accuracies[subset]) * float(
          example_counts[subset]
      )
    if total_examples <= 0:
      output[_slugify_section(section)] = 0.0
    else:
      output[_slugify_section(section)] = total_weighted / total_examples
  return output


def _load_rewardbench_rows(limit: int | None) -> list[dict[str, Any]]:
  dataset = load_dataset(REWARDBENCH_DATASET, split=REWARDBENCH_SPLIT)
  if limit is not None:
    dataset = dataset.select(range(min(limit, len(dataset))))
  return list(dataset)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _evaluate_method(
    *,
    run: dict[str, Any],
    rewardbench_rows: list[dict[str, Any]],
    base_cfg: Any,
    sft_model_path: str,
    batch_size: int,
    output_root: Path,
    force: bool,
    example_counts: dict[str, int],
    subset_mapping: dict[str, list[str]],
) -> dict[str, Any]:
  per_example_path = output_root / f"{run['variant']}_per_example.jsonl"
  summary_path = output_root / f"{run['variant']}_summary.json"
  if per_example_path.exists() and summary_path.exists() and not force:
    return json.loads(summary_path.read_text())

  bundle = qwen2p5_dpo_eval_lib.load_eval_bundle(
      base_cfg=base_cfg,
      actor_model_path=run["exported_model_path"],
      reference_model_path=sft_model_path,
      metrics_prefix=f"rewardbench_{run['variant']}",
  )
  trainer, metrics_logger = qwen2p5_dpo_eval_lib.create_eval_trainer(bundle)

  per_example_rows: list[dict[str, Any]] = []
  try:
    for batch_rows in qwen2p5_dpo_eval_lib.iter_chunks(rewardbench_rows, batch_size):
      with bundle.actor_mesh:
        batch_metrics = qwen2p5_dpo_eval_lib.iter_preference_batch_metrics(
            trainer=trainer,
            prompts=[row["prompt"] for row in batch_rows],
            chosen_responses=[row["chosen"] for row in batch_rows],
            rejected_responses=[row["rejected"] for row in batch_rows],
        )
      for row, metrics in zip(batch_rows, batch_metrics, strict=True):
        per_example_rows.append(
            {
                "id": row["id"],
                "subset": row["subset"],
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                **metrics,
            }
        )
  finally:
    qwen2p5_dpo_eval_lib.close_eval_trainer(trainer, metrics_logger)
    qwen2p5_dpo_eval_lib.close_eval_bundle(bundle)

  _write_jsonl(per_example_path, per_example_rows)

  subset_totals: dict[str, int] = {}
  subset_correct: dict[str, int] = {}
  for row in per_example_rows:
    subset = row["subset"]
    subset_totals[subset] = subset_totals.get(subset, 0) + 1
    subset_correct[subset] = subset_correct.get(subset, 0) + int(
        row["rewards_accuracy"] > 0.5
    )
  subset_accuracies = {
      subset: subset_correct[subset] / subset_totals[subset]
      for subset in sorted(subset_totals)
  }
  section_scores = _calculate_section_scores(
      subset_accuracies=subset_accuracies,
      example_counts=example_counts,
      subset_mapping=subset_mapping,
  )
  overall = sum(int(row["rewards_accuracy"] > 0.5) for row in per_example_rows) / max(
      len(per_example_rows), 1
  )
  summary = {
      "variant": run["variant"],
      "run_name": run["run_name"],
      "num_examples": len(per_example_rows),
      "scoring_backend": SCORING_BACKEND,
      "rewardbench_overall": overall,
      "rewardbench_chat": section_scores["chat"],
      "rewardbench_chat_hard": section_scores["chat_hard"],
      "rewardbench_safety": section_scores["safety"],
      "rewardbench_reasoning": section_scores["reasoning"],
      "subset_accuracies": subset_accuracies,
      "per_example_path": str(per_example_path),
  }
  summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
  return summary


def main() -> None:
  args = _parse_args()
  repo_root = Path(args.repo_root).resolve()
  output_root = (
      Path(args.output_root).resolve()
      if args.output_root
      else repo_root
      / "runs/results"
      / f"clean_benchmarks_{args.run_ts}"
      / "rewardbench_v1"
  )
  output_root.mkdir(parents=True, exist_ok=True)

  example_counts, subset_mapping = _load_rewardbench_reference_constants(
      Path(args.rewardbench_repo).resolve()
  )
  rewardbench_rows = _load_rewardbench_rows(args.eval_limit)
  runs = _discover_clean_runs(
      repo_root, run_ts=args.run_ts, methods=list(args.methods)
  )
  base_cfg = OmegaConf.load(args.config_path)

  model_summaries = {}
  for index, run in enumerate(runs, start=1):
    print(f"[{index:02d}/{len(runs):02d}] RewardBench v1: {run['variant']}")
    model_summaries[run["variant"]] = _evaluate_method(
        run=run,
        rewardbench_rows=rewardbench_rows,
        base_cfg=base_cfg,
        sft_model_path=str(Path(args.sft_model_path).resolve()),
        batch_size=args.batch_size,
        output_root=output_root,
        force=args.force,
        example_counts=example_counts,
        subset_mapping=subset_mapping,
    )

  summary = {
      "run_ts": args.run_ts,
      "dataset": REWARDBENCH_DATASET,
      "split": REWARDBENCH_SPLIT,
      "num_examples": len(rewardbench_rows),
      "scoring_backend": SCORING_BACKEND,
      "rewardbench_repo": str(Path(args.rewardbench_repo).resolve()),
      "scores": model_summaries,
  }
  summary_path = output_root / "rewardbench_summary.json"
  summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

  print(f"Wrote RewardBench summary to {summary_path}")
  for method in sorted(model_summaries, key=lambda item: METHOD_ORDER[item]):
    row = model_summaries[method]
    print(
        "\t".join(
            [
                method,
                f"overall={row['rewardbench_overall']:.4f}",
                f"chat={row['rewardbench_chat']:.4f}",
                f"chat_hard={row['rewardbench_chat_hard']:.4f}",
                f"safety={row['rewardbench_safety']:.4f}",
                f"reasoning={row['rewardbench_reasoning']:.4f}",
            ]
        )
    )


if __name__ == "__main__":
  main()
