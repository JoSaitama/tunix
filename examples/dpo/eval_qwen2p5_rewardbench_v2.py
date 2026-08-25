#!/usr/bin/env python3

"""Run RewardBench v2 with a Tunix/JAX TPU-native DPO scorer."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any

from datasets import Dataset, load_dataset
import numpy as np
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
REWARDBENCH_DATASET = "allenai/reward-bench-2"
REWARDBENCH_SPLIT = "test"
SCORING_BACKEND = "tunix_tpu_rewardbench_v2_adapter"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Evaluate the four clean Qwen2.5 DPO variants on RewardBench v2 "
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
      "--output-root",
      default=None,
      help=(
          "Defaults to runs/results/clean_benchmarks_<run-ts>/rewardbench_v2 "
          "under repo root."
      ),
  )
  parser.add_argument(
      "--batch-size",
      type=int,
      default=8,
      help="Candidate batch size for local TPU scoring.",
  )
  parser.add_argument(
      "--eval-limit",
      type=int,
      default=None,
      help="Optional debug cap on RewardBench v2 examples.",
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


def _load_rewardbench_v2_rows(limit: int | None) -> list[dict[str, Any]]:
  dataset = load_dataset(REWARDBENCH_DATASET, split=REWARDBENCH_SPLIT)
  if limit is not None:
    dataset = dataset.select(range(min(limit, len(dataset))))
  return list(dataset)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _compute_prompt_stats(
    samples: list[tuple[bool, float]],
) -> tuple[bool, float | None, float | None]:
  """Official RewardBench v2 ties prompt statistic helper."""
  correct_scores = [score for is_correct, score in samples if is_correct]
  incorrect_scores = [score for is_correct, score in samples if not is_correct]
  best_correct = max(correct_scores)
  worst_correct = min(correct_scores)
  best_incorrect = max(incorrect_scores)
  different_correct_margin = (
      best_correct - worst_correct if len(correct_scores) > 1 else None
  )
  correct_incorrect_margin = worst_correct - best_incorrect
  accurate = correct_incorrect_margin > 0
  return accurate, different_correct_margin, correct_incorrect_margin


def _process_single_model_ties(
    dataset: Dataset,
) -> tuple[Dataset, float]:
  """Local copy of official RewardBench v2 ties aggregation."""
  grouped_samples: dict[tuple[str, int], list[tuple[bool, float]]] = defaultdict(
      list
  )
  for sample in dataset:
    sample_type, prompt_id_str = sample["id"].split(":")
    prompt_id = int(prompt_id_str)
    for index, raw_score in enumerate(sample["scores"]):
      score = raw_score[0] if isinstance(raw_score, list) else raw_score
      grouped_samples[(sample_type, prompt_id)].append(
          (index < sample["num_correct"], score)
      )

  ref_stats = {}
  tied_stats = {}
  for (sample_type, prompt_id), samples in grouped_samples.items():
    stats = _compute_prompt_stats(samples)
    if sample_type == "ref":
      ref_stats[prompt_id] = stats
    else:
      tied_stats[prompt_id] = stats

  ref_accuracy = float(np.mean([stats[0] for stats in ref_stats.values()])) if ref_stats else 0.0
  tied_accuracy = float(np.mean([stats[0] for stats in tied_stats.values()])) if tied_stats else 0.0
  all_prompts = set(ref_stats) & set(tied_stats)
  if all_prompts:
    diff_corr_margin = np.array([tied_stats[pid][1] for pid in all_prompts])
    corr_incorrect_ties = np.array([tied_stats[pid][2] for pid in all_prompts])
    corr_incorrect_ref = np.array([ref_stats[pid][2] for pid in all_prompts])
    correctness_preferred = float(np.mean(corr_incorrect_ties > diff_corr_margin))
    correctness_preferred_hard = float(
        np.mean(
            np.minimum(corr_incorrect_ref, corr_incorrect_ties) > diff_corr_margin
        )
    )
    margin_scores = np.tanh(
        np.minimum(corr_incorrect_ref, corr_incorrect_ties) / diff_corr_margin - 1
    )
    margin_scores = np.nan_to_num(margin_scores, nan=0.0)
    correctness_margin_score = float(np.mean(margin_scores))
  else:
    correctness_preferred = 0.0
    correctness_preferred_hard = 0.0
    correctness_margin_score = 0.0
  overall_score = (
      0.30 * tied_accuracy
      + 0.30 * ref_accuracy
      + 0.20 * correctness_preferred
      + 0.20 * correctness_preferred_hard
      + 0.01 * correctness_margin_score
  )
  if "results" in dataset.column_names:
    dataset = dataset.remove_columns(["results"])
  return dataset.add_column("results", [None] * len(dataset)), float(overall_score)


def _score_candidate_group(
    *,
    bundle: qwen2p5_dpo_eval_lib.DPOEvalBundle,
    trainer: Any,
    row: dict[str, Any],
    batch_size: int,
) -> list[float]:
  candidates = list(row["chosen"]) + list(row["rejected"])
  prompts = [row["prompt"]] * len(candidates)
  scores: list[float] = []
  for batch_rows in qwen2p5_dpo_eval_lib.iter_chunks(candidates, batch_size):
    batch_prompts = prompts[len(scores) : len(scores) + len(batch_rows)]
    with bundle.actor_mesh:
      batch_scores = qwen2p5_dpo_eval_lib.iter_response_reward_scores(
          trainer=trainer,
          prompts=batch_prompts,
          responses=list(batch_rows),
      )
    scores.extend(batch_scores)
  return scores


def _summarize_rewardbench_v2_rows(
    *,
    scored_rows: list[dict[str, Any]],
) -> dict[str, Any]:
  subset_scores: dict[str, float] = {}
  subset_counts: dict[str, int] = {}
  non_ties_rows = [row for row in scored_rows if row["subset"].lower() != "ties"]
  ties_rows = [row for row in scored_rows if row["subset"].lower() == "ties"]

  grouped: dict[str, list[dict[str, Any]]] = {}
  for row in non_ties_rows:
    grouped.setdefault(row["subset"], []).append(row)
  for subset, rows in grouped.items():
    correct = 0
    for row in rows:
      predicted_index = int(np.argmax(row["scores"]))
      correct += int(predicted_index < int(row["num_correct"]))
    subset_scores[subset] = correct / len(rows) if rows else 0.0
    subset_counts[subset] = len(rows)

  if ties_rows:
    ties_dataset = Dataset.from_list(
        [
            {
                "id": row["id"],
                "scores": row["scores"],
                "num_correct": row["num_correct"],
            }
            for row in ties_rows
        ]
    )
    _, ties_score = _process_single_model_ties(ties_dataset)
    subset_scores["Ties"] = float(ties_score)
    subset_counts["Ties"] = len(ties_rows)

  macro_score = float(np.mean(list(subset_scores.values()))) if subset_scores else 0.0
  weighted_numerator = sum(
      subset_scores[subset] * subset_counts[subset] for subset in subset_scores
  )
  weighted_score = (
      weighted_numerator / sum(subset_counts.values()) if subset_counts else 0.0
  )
  flat_scores = {
      f"rewardbench2_{subset.lower().replace('-', '_').replace(' ', '_')}": score
      for subset, score in subset_scores.items()
  }
  return {
      "rewardbench2_macro_score": macro_score,
      "rewardbench2_weighted_score": float(weighted_score),
      "rewardbench2_subset_scores": subset_scores,
      **flat_scores,
  }


def _evaluate_method(
    *,
    run: dict[str, Any],
    rewardbench_rows: list[dict[str, Any]],
    base_cfg: Any,
    sft_model_path: str,
    batch_size: int,
    output_root: Path,
    force: bool,
) -> dict[str, Any]:
  per_example_path = output_root / f"{run['variant']}_per_prompt.jsonl"
  summary_path = output_root / f"{run['variant']}_summary.json"
  if per_example_path.exists() and summary_path.exists() and not force:
    cached_summary = json.loads(summary_path.read_text())
    if int(cached_summary.get("num_examples", -1)) == len(rewardbench_rows):
      return cached_summary

  bundle = qwen2p5_dpo_eval_lib.load_eval_bundle(
      base_cfg=base_cfg,
      actor_model_path=run["exported_model_path"],
      reference_model_path=sft_model_path,
      metrics_prefix=f"rewardbench2_{run['variant']}",
  )
  trainer, metrics_logger = qwen2p5_dpo_eval_lib.create_eval_trainer(bundle)

  scored_rows: list[dict[str, Any]] = []
  try:
    for row in rewardbench_rows:
      scores = _score_candidate_group(
          bundle=bundle,
          trainer=trainer,
          row=row,
          batch_size=batch_size,
      )
      scored_rows.append(
          {
              "id": row["id"],
              "prompt": row["prompt"],
              "subset": row["subset"],
              "num_correct": int(row["num_correct"]),
              "num_incorrect": int(row["num_incorrect"]),
              "total_completions": int(row["total_completions"]),
              "scores": scores,
          }
      )
  finally:
    qwen2p5_dpo_eval_lib.close_eval_trainer(trainer, metrics_logger)
    qwen2p5_dpo_eval_lib.close_eval_bundle(bundle)

  summary = _summarize_rewardbench_v2_rows(
      scored_rows=scored_rows,
  )
  summary.update(
      {
          "variant": run["variant"],
          "run_name": run["run_name"],
          "num_examples": len(scored_rows),
          "scoring_backend": SCORING_BACKEND,
      }
  )
  _write_jsonl(per_example_path, scored_rows)
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
      / "rewardbench_v2"
  )
  output_root.mkdir(parents=True, exist_ok=True)
  output_path = output_root / "rewardbench_v2_summary.json"

  runs = _discover_clean_runs(
      repo_root,
      run_ts=args.run_ts,
      methods=list(args.methods),
  )
  rewardbench_rows = _load_rewardbench_v2_rows(args.eval_limit)
  base_cfg = OmegaConf.load(args.config_path)
  previous_payload = json.loads(output_path.read_text()) if output_path.exists() else {}
  scores = dict(previous_payload.get("scores", {}))
  for run in runs:
    summary = _evaluate_method(
        run=run,
        rewardbench_rows=rewardbench_rows,
        base_cfg=base_cfg,
        sft_model_path=args.sft_model_path,
        batch_size=args.batch_size,
        output_root=output_root,
        force=args.force,
    )
    scores[run["variant"]] = summary
    print(
        f"{run['variant']}: macro={summary['rewardbench2_macro_score']:.4f} "
        f"weighted={summary['rewardbench2_weighted_score']:.4f}"
    )

  payload = {
      "run_ts": args.run_ts,
      "dataset": REWARDBENCH_DATASET,
      "split": REWARDBENCH_SPLIT,
      "scoring_backend": SCORING_BACKEND,
      "scores": scores,
  }
  output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
  print(f"Wrote RewardBench v2 summary to {output_path}")


if __name__ == "__main__":
  main()
