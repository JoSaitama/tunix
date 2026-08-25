#!/usr/bin/env python3

"""Evaluate all metrics needed by the multi-seed clean DPO main table."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import os
import json
import re
from pathlib import Path
import subprocess
import sys
from typing import Any

from datasets import load_dataset
import numpy as np
from omegaconf import OmegaConf
from tensorboard.backend.event_processing import event_accumulator

REPO_ROOT = Path("/home/lhf_hongfu_gmail_com/tunix")
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from examples.dpo import qwen2p5_clean_main_table_lib as table_lib
from examples.dpo import qwen2p5_dpo_eval_lib


CONFIG_PATH = table_lib.CONFIG_PATH
SFT_MODEL_PATH = table_lib.SFT_MODEL_PATH
DEFAULT_OFFLINE_EVAL_VENV = Path("/home/lhf_hongfu_gmail_com/.venvs/DPO-EVAL-OFFLINE")
LIVEBENCH_HELPER_PATH = REPO_ROOT / "examples/dpo/score_instruction_following_dataset.py"
IFEVAL_HELPER_MODULE = REPO_ROOT / "examples/dpo/eval_qwen2p5_clean_benchmarks.py"
LIVEBENCH_ROOT = Path("/home/lhf_hongfu_gmail_com/.cache/LiveBench")



# FIX_REWARD_BASED_SPACE_NORMALIZED_ALIASES_MAIN_TABLE
REWARD_BASED_EXTRA_ALIASES = {
    "reward based filtering filt5": "reward_based_filtering_filt5",
    "reward based filtering 5": "reward_based_filtering_filt5",
    "reward based filtering 5pct": "reward_based_filtering_filt5",
    "reward based filtering filt10": "reward_based_filtering_filt10",
    "reward based filtering 10": "reward_based_filtering_filt10",
    "reward based filtering 10pct": "reward_based_filtering_filt10",
    "reward based filtering filt40": "reward_based_filtering_filt40",
    "reward based filtering 40": "reward_based_filtering_filt40",
    "reward based filtering 40pct": "reward_based_filtering_filt40",
    "reward_based_filtering_filt20": "reward_based_filtering_filt20",
    "reward based filtering filt20": "reward_based_filtering_filt20",
    "reward_based_filtering_20": "reward_based_filtering_filt20",
    "reward based filtering 20": "reward_based_filtering_filt20",
    "reward_based_filtering_20pct": "reward_based_filtering_filt20",
    "reward based filtering 20pct": "reward_based_filtering_filt20",
    "reward-based filtering (20%)": "reward_based_filtering_filt20",
    "reward_based_filtering_filt40": "reward_based_filtering_filt40",
    "reward_based_filtering_40": "reward_based_filtering_filt40",
    "reward_based_filtering_40pct": "reward_based_filtering_filt40",
}

def _load_helper_module(module_path: Path, module_name: str):
  spec = importlib.util.spec_from_file_location(module_name, module_path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Evaluate the 3-seed clean DPO main-table runs on clean held-out "
          "preference accuracy plus LiveBench-IF, RewardBench 2 Precise IF, "
          "and IFBench."
      )
  )
  parser.add_argument("--repo-root", default=str(REPO_ROOT))
  parser.add_argument("--run-ts", required=True)
  parser.add_argument(
      "--legacy-run-ts",
      default=table_lib.LEGACY_RUN_TS,
      help="Reuse existing seed-0 clean results from this legacy timestamp.",
  )
  parser.add_argument(
      "--no-legacy",
      action="store_true",
      help="Disable legacy seed-0 fallback and require all runs to come from --run-ts.",
  )
  parser.add_argument(
      "--methods",
      nargs="+",
      default=list(table_lib.METHOD_ORDER),
      help="Subset of logical clean main-table methods to evaluate.",
  )
  parser.add_argument(
      "--seeds",
      nargs="+",
      type=int,
      default=list(table_lib.DEFAULT_SEEDS),
      help="Seeds to include when discovering seeded runs.",
  )
  parser.add_argument(
      "--benchmarks",
      nargs="+",
      default=("clean_test", "livebench_if", "rewardbench2", "ifbench"),
      help="Subset of metrics/benchmarks to evaluate.",
  )
  parser.add_argument(
      "--profile",
      default="full",
      help="Run profile to discover (for example: full or smoke).",
  )
  parser.add_argument(
      "--output-root",
      default=None,
      help="Defaults to runs/results/qwen2p5_clean_main_table_<run-ts>.",
  )
  parser.add_argument(
      "--config-path",
      default=str(CONFIG_PATH),
  )
  parser.add_argument(
      "--sft-model-path",
      default=str(SFT_MODEL_PATH),
  )
  parser.add_argument("--force", action="store_true")
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
      default=str(DEFAULT_OFFLINE_EVAL_VENV),
  )
  parser.add_argument(
      "--livebench-root",
      default=str(LIVEBENCH_ROOT),
  )
  return parser.parse_args()


def _normalize_benchmarks(names: list[str]) -> set[str]:
  aliases = {
      "clean_test": "clean_test",
      "test": "clean_test",
      "livebench_if": "livebench_if",
      "livebench": "livebench_if",
      "rewardbench2": "rewardbench2",
      "rewardbench_v2": "rewardbench2",
      "ifbench": "ifbench",
  }
  normalized = set()
  for name in names:
    key = name.strip().lower().replace("-", "_")
    if key not in aliases:
      raise ValueError(f"Unsupported benchmark selector {name!r}.")
    normalized.add(aliases[key])
  return normalized


def _load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text()) if path.exists() else {}


def _build_legacy_payloads(repo_root: Path, legacy_run_ts: str | None) -> dict[str, Any]:
  if not legacy_run_ts:
    return {}
  results_root = repo_root / "runs" / "results"
  clean_benchmark_root = results_root / f"clean_benchmarks_{legacy_run_ts}"
  return {
      "test_matrix": _load_json(
          results_root / f"qwen2p5_dpo_test_matrix_{legacy_run_ts}.json"
      ),
      "benchmark_summary": _load_json(
          clean_benchmark_root / "benchmark_summary.json"
      ),
      "rewardbench_v2": _load_json(
          clean_benchmark_root / "rewardbench_v2" / "rewardbench_v2_summary.json"
      ),
  }


def _lookup_legacy_metric(
    *,
    run: dict[str, Any],
    legacy_payloads: dict[str, Any],
    metric_name: str,
) -> float | None:
  if run["source"] != "legacy_seed0":
    return None
  legacy_variant = run["source_variant"]
  if metric_name == "clean_test_acc":
    for row in legacy_payloads.get("test_matrix", {}).get("results", []):
      if row.get("dataset") != "clean" or row.get("variant") != legacy_variant:
        continue
      return float(row["metrics"]["rewards_accuracy"])
    return None
  if metric_name in ("livebench_if_score", "ifbench_prompt_strict"):
    row = legacy_payloads.get("benchmark_summary", {}).get("scores", {}).get(
        legacy_variant, {}
    )
    value = row.get(metric_name)
    return float(value) if value is not None else None
  if metric_name == "rewardbench2_precise_if":
    row = legacy_payloads.get("rewardbench_v2", {}).get("scores", {}).get(
        legacy_variant, {}
    )
    value = row.get(metric_name)
    return float(value) if value is not None else None
  return None


def _compute_normalized_scalar_auc(tensorboard_dir: Path, tag: str) -> float:
  accumulator = event_accumulator.EventAccumulator(str(tensorboard_dir))
  accumulator.Reload()
  scalar_points = accumulator.Scalars(tag)
  if not scalar_points:
    raise SystemExit(
        f"No scalar history for tag={tag!r} under tensorboard_dir={tensorboard_dir}"
    )
  steps = np.asarray([point.step for point in scalar_points], dtype=np.float64)
  values = np.asarray([point.value for point in scalar_points], dtype=np.float64)
  if len(steps) == 1 or np.allclose(steps[0], steps[-1]):
    return float(values[-1])
  area = np.trapezoid(values, steps)
  return float(area / (steps[-1] - steps[0]))


def _append_limit_to_module_spec(module_spec: str, limit: int | None) -> str:
  """Append limit=... to a create_dataset(...) module spec when requested."""
  if limit is None:
    return module_spec
  if "limit=" in module_spec:
    return module_spec
  stripped = module_spec.strip()
  if stripped.endswith(")"):
    return stripped[:-1] + f", limit={limit})"
  return stripped + f"(limit={limit})"


def _build_test_module_spec(limit: int | None) -> str:
  env_spec = os.environ.get("DPO_CLEAN_TEST_DATA_MODULE", "").strip()
  if env_spec:
    return _append_limit_to_module_spec(env_spec, limit)

  if limit is None:
    return "examples/data/ultrafeedback_dpo.py:create_dataset(split='test_prefs', seed=42)"
  return (
      "examples/data/ultrafeedback_dpo.py:create_dataset("
      f"split='test_prefs', seed=42, limit={limit})"
  )
def _evaluate_clean_test_acc(
    *,
    run: dict[str, Any],
    base_cfg: Any,
    sft_model_path: str,
    eval_limit: int | None,
    num_batches: int | None,
) -> float:
  bundle = qwen2p5_dpo_eval_lib.load_eval_bundle(
      base_cfg=base_cfg,
      actor_model_path=run["exported_model_path"],
      reference_model_path=sft_model_path,
      metrics_prefix=f"dpo_clean_test_{run['run_key']}",
  )
  eval_dataset = qwen2p5_dpo_eval_lib.load_eval_dataset(
      module_spec=_build_test_module_spec(eval_limit),
      tokenizer=bundle.tokenizer,
      batch_size=int(base_cfg.get("eval_batch_size", base_cfg["batch_size"])),
      num_batches=num_batches,
      dpo_config=bundle.training_config,
  )
  trainer, metrics_logger = qwen2p5_dpo_eval_lib.create_eval_trainer(bundle)
  try:
    with bundle.actor_mesh:
      metrics = qwen2p5_dpo_eval_lib.aggregate_eval_metrics(
          trainer=trainer,
          metrics_logger=metrics_logger,
          eval_dataset=eval_dataset,
      )
  finally:
    qwen2p5_dpo_eval_lib.close_eval_trainer(trainer, metrics_logger)
    qwen2p5_dpo_eval_lib.close_eval_bundle(bundle)
  return float(metrics["rewards_accuracy"])


def _load_livebench_rows(limit: int | None) -> list[dict[str, Any]]:
  rows = list(load_dataset("livebench/instruction_following", split="test"))
  normalized_rows = []
  for row in rows:
    normalized = dict(row)
    for key, value in list(normalized.items()):
      if isinstance(value, datetime):
        normalized[key] = datetime.strftime(value, "%Y-%m-%d")
    normalized_rows.append(normalized)
  return normalized_rows[:limit] if limit is not None else normalized_rows


def _jsonl_len(path: Path) -> int:
  if not path.exists():
    return 0
  with path.open(encoding="utf-8") as f:
    return sum(1 for line in f if line.strip())


def _evaluate_livebench_if(
    *,
    clean_benchmarks: Any,
    repo_root: Path,
    run: dict[str, Any],
    output_root: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> float:
  response_file = output_root / "livebench_if" / "responses" / f"{run['run_key']}.jsonl"
  result_dir = output_root / "livebench_if" / "results" / run["run_key"]
  result_dir.mkdir(parents=True, exist_ok=True)
  prompts = [row["turns"][0] for row in rows]
  if args.force or _jsonl_len(response_file) != len(rows):
    generator = clean_benchmarks.TunixChatGenerator(
        config_path=str(CONFIG_PATH),
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
    response_rows = [
        {
            "question_id": row["question_id"],
            "prompt": row["turns"][0],
            "response": answer,
        }
        for row, answer in zip(rows, answers, strict=True)
    ]
    clean_benchmarks._write_jsonl(response_file, response_rows)  # pylint: disable=protected-access

  offline_python = Path(args.offline_eval_venv).resolve() / "bin/python"
  subprocess.run(
      [
          str(offline_python),
          str(LIVEBENCH_HELPER_PATH),
          "--dataset-format",
          "livebench_if",
          "--input-data",
          str(output_root / "livebench_if" / "data" / "livebench_instruction_following.jsonl"),
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
  summary = json.loads((result_dir / "summary.json").read_text())
  return float(summary["livebench_if_score"])


def _evaluate_ifbench(
    *,
    clean_benchmarks: Any,
    run: dict[str, Any],
    output_root: Path,
    prompts: list[dict[str, Any]],
    args: argparse.Namespace,
) -> float:
  assets = clean_benchmarks._prepare_ifbench_assets(  # pylint: disable=protected-access
      output_root / "ifbench"
  )
  response_file = Path(assets["response_dir"]) / f"{run['run_key']}.jsonl"
  expected_count = len(prompts)
  if args.force or _jsonl_len(response_file) != expected_count:
    generator = clean_benchmarks.TunixChatGenerator(
        config_path=str(CONFIG_PATH),
        exported_model_path=run["exported_model_path"],
        max_prompt_length=args.max_prompt_length,
        max_generation_steps=args.max_generation_steps,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    try:
      clean_benchmarks._generate_ifbench(  # pylint: disable=protected-access
          generator,
          prompts=prompts,
          output_file=response_file,
          seed=args.seed,
          batch_size=args.generation_batch_size,
      )
    finally:
      generator.close()
  scores = clean_benchmarks._run_ifbench_evaluator(  # pylint: disable=protected-access
      assets=assets,
      model_name=run["run_key"],
      offline_eval_venv=Path(args.offline_eval_venv).resolve(),
  )
  return float(scores["ifbench_prompt_strict"])


def _evaluate_rewardbench2(
    *,
    rewardbench_module: Any,
    run: dict[str, Any],
    rewardbench_rows: list[dict[str, Any]],
    base_cfg: Any,
    sft_model_path: str,
    batch_size: int,
    output_root: Path,
    force: bool,
) -> float:
  summary = rewardbench_module._evaluate_method(  # pylint: disable=protected-access
      run={
          "variant": run["run_key"],
          "run_name": run["run_name"],
          "exported_model_path": run["exported_model_path"],
      },
      rewardbench_rows=rewardbench_rows,
      base_cfg=base_cfg,
      sft_model_path=sft_model_path,
      batch_size=batch_size,
      output_root=output_root / "rewardbench_v2",
      force=force,
  )
  return float(summary["rewardbench2_precise_if"])



def _normalize_no_pref_method(method: str) -> str:
  aliases = {
      "vanilla_dpo": "vanilla_dpo",
      "vanilla": "vanilla_dpo",
      "self_inf": "self_inf",
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
      "self_inf_loo": "self_inf_loo",
      "self_inf_re_loo": "self_inf_re_loo",
      "self_inf_lambda": "self_inf_lambda",
      "self_inf_lambda_batch": "self_inf_lambda",
      "self_influence_lambda": "self_inf_lambda",
      "self_influence_lambda_batch": "self_inf_lambda",
      "dtv_lambda_batch": "self_inf_lambda",
    "self inf lambda batch": "self_inf_lambda",
    "self-inf lambda batch": "self_inf_lambda",
    "self-inf-lambda-batch": "self_inf_lambda",
    "self influence lambda batch": "self_inf_lambda",
    "self-influence-lambda-batch": "self_inf_lambda",
    "dtv lambda batch": "self_inf_lambda",
    "dtv-lambda-batch": "self_inf_lambda",
      "self inf lambda": "self_inf_lambda",
      "self influence lambda batch": "self_inf_lambda",
      "self-influence-lambda-batch": "self_inf_lambda",
      "self-inf-lambda": "self_inf_lambda",
      "self-inf lambda": "self_inf_lambda",
      "self influence lambda": "self_inf_lambda",
      "self-influence-lambda": "self_inf_lambda",
      "self_dtv_lambda": "self_inf_lambda",
      "dtv_lambda": "self_inf_lambda",
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
      "self_inf_re_loo": "self_inf_re_loo",
      "self inf loo": "self_inf_loo",
      "self-inf-loo": "self_inf_loo",
      "self-inf loo": "self_inf_loo",
      "self_dtv_loo": "self_inf_loo",
      "self_inf_loo_batch": "self_inf_loo",
      "self_inf_loo_cos": "self_inf_loo_cos",
      "self inf loo cos": "self_inf_loo_cos",
      "self-inf-loo-cos": "self_inf_loo_cos",
      "self-inf loo cos": "self_inf_loo_cos",
      "self_dtv_loo_cos": "self_inf_loo_cos",
      "self_inf_loo_cos_batch": "self_inf_loo_cos",
      "self-dtv": "self_inf",
      "random_pair_filtering_filt5": "random_pair_filtering_filt5",
      "random_pair_filtering_5": "random_pair_filtering_filt5",
      "random_pair_filtering_5pct": "random_pair_filtering_filt5",
      "random_pair_filtering_filt10": "random_pair_filtering_filt10",
      "reward_based_filtering_filt5": "reward_based_filtering_filt5",
      "reward_based_filtering_5": "reward_based_filtering_filt5",
      "reward_based_filtering_5pct": "reward_based_filtering_filt5",
      "reward_based_filtering_filt10": "reward_based_filtering_filt10",
      "reward_based_filtering_10": "reward_based_filtering_filt10",
      "reward_based_filtering_10pct": "reward_based_filtering_filt10",
      "reward_based_filtering_filt40": "reward_based_filtering_filt40",
      "reward_based_filtering_40": "reward_based_filtering_filt40",
      "reward_based_filtering_40pct": "reward_based_filtering_filt40",
      "random_pair_filtering_10": "random_pair_filtering_filt10",
      "random_pair_filtering_10pct": "random_pair_filtering_filt10",
      "random_pair_filtering_filt20": "random_pair_filtering_filt20",
    "random pair filtering filt20": "random_pair_filtering_filt20",
    "random_pair_filtering_20": "random_pair_filtering_filt20",
    "random pair filtering 20": "random_pair_filtering_filt20",
    "random_pair_filtering_20pct": "random_pair_filtering_filt20",
    "random pair filtering 20pct": "random_pair_filtering_filt20",
    "random pair filtering (20%)": "random_pair_filtering_filt20",
    "random_pair_filtering_filt40": "random_pair_filtering_filt40",
      "random_pair_filtering_40": "random_pair_filtering_filt40",
      "random_pair_filtering_40pct": "random_pair_filtering_filt40",
      "dtv": "self_inf",
      "self_dtv": "self_inf",
  }
  key = method.strip().lower().replace(" ", "_").replace("-", "_")
  if key not in aliases:
    raise ValueError(
        f"Unsupported no-pref method {method!r}. "
        f"Supported methods: {sorted(aliases)}"
    )
  return aliases[key]


def _discover_no_pref_main_table_runs(
    *,
    repo_root: Path,
    methods: list[str],
    profile: str,
) -> list[dict[str, Any]]:
  wanted = {_normalize_no_pref_method(method) for method in methods}

  runs_root = Path(
      os.environ.get(
          "DPO_EVAL_RUNS_ROOT",
          str(repo_root / "runs_xuesong"),
      )
  ).resolve()
  run_glob = os.environ.get(
      "DPO_EVAL_RUN_GLOB",
      f"dpo_qwen2p5_*_global_no_pref*_lora_{profile}_*",
  )
  run_re = re.compile(
      r"dpo_qwen2p5_"
      r"(?P<source_variant>.+)_"
      r"(?P<corruption>clean|global_no_pref10|global_no_pref20|global_no_pref40|global_mismatch\d+)_"
      r"lora_"
      + re.escape(profile)
      + r"_(?P<source_run_ts>\d{8}_\d{6})$"
  )

  print(f"Discovering no-pref runs under {runs_root} with glob {run_glob}")

  display_names = {
      "vanilla_dpo": "Vanilla DPO",
      "self_inf": "Self-DTV",
    "self_inf_re_loo": "Self-DTV-Re-LOO",
    "self_inf_lambda": "Self-Inf-Lambda",
      "self_inf_norm": "Self-DTV-Norm",
      "random_pair_filtering_filt5": "Random Pair Filtering (5%)",
      "random_pair_filtering_filt10": "Random Pair Filtering (10%)",
      "random_pair_filtering_filt40": "Random Pair Filtering (40%)",
      "reward_based_filtering_filt5": "Reward-based Filtering (5%)",
      "reward_based_filtering_filt10": "Reward-based Filtering (10%)",
      "reward_based_filtering_filt40": "Reward-based Filtering (40%)",
  }
  method_order = {
      "vanilla_dpo": 0,
      "self_inf": 1,
      "self_inf_norm": 1.05,
      "self_inf_loo": 1.10,
      "self_inf_re_loo": 1.15,
      "self_inf_loo_cos": 1.20,
      "self_inf_lambda": 1.25,
      "random_pair_filtering_filt5": 2,
      "random_pair_filtering_filt10": 3,
      "random_pair_filtering_filt20": 3.05,
    "random_pair_filtering_filt40": 3.1,
      "reward_based_filtering_filt5": 4,
      "reward_based_filtering_filt10": 5,
      "reward_based_filtering_filt20": 5.05,
    "reward_based_filtering_filt40": 5.1,
  }
  corruption_order = {
      "clean": -1,
      "global_no_pref10": 0,
      "global_no_pref20": 1,
      "global_no_pref40": 2,
  }

  discovered = []
  for run_dir in sorted(runs_root.glob(run_glob)):
    if not run_dir.is_dir():
      continue

    match = run_re.fullmatch(run_dir.name)
    if match is None:
      continue

    source_variant = match.group("source_variant")
    corruption = match.group("corruption")
    source_run_ts = match.group("source_run_ts")

    try:
      logical_variant = _normalize_no_pref_method(source_variant)
    except ValueError:
      continue

    if logical_variant not in wanted:
      continue

    exported_model_path = run_dir / "exported_model"
    tensorboard_dir = run_dir / "tensorboard"

    if not (exported_model_path / "model.safetensors").exists():
      print(f"Skipping {run_dir.name}: missing exported_model/model.safetensors")
      continue
    if not tensorboard_dir.exists():
      print(f"Skipping {run_dir.name}: missing tensorboard dir")
      continue

    run_key = f"{logical_variant}_{corruption}"

    discovered.append({
        "source": "no_pref",
        "seed": 0,
        "run_name": run_dir.name,
        "run_key": run_key,
        "logical_variant": logical_variant,
        "display_name": display_names.get(logical_variant, logical_variant),
        "source_variant": source_variant,
        "source_run_ts": source_run_ts,
        "corruption": corruption,
        "run_dir": str(run_dir),
        "exported_model_path": str(exported_model_path),
        "tensorboard_dir": str(tensorboard_dir),
    })

  discovered.sort(
      key=lambda row: (
          corruption_order.get(row["corruption"], 99),
          method_order.get(row["logical_variant"], 99),
          row["source_run_ts"],
      )
  )

  if not discovered:
    raise SystemExit(
        f"No no-pref runs discovered under {runs_root} with glob {run_glob}"
    )

  print("Discovered no-pref runs:")
  for row in discovered:
    print(
        f"  {row['run_key']}: "
        f"{row['run_dir']} "
        f"model={row['exported_model_path']}"
    )

  return discovered


def main() -> None:
  args = _parse_args()
  repo_root = Path(args.repo_root).resolve()
  output_root = (
      Path(args.output_root).resolve()
      if args.output_root
      else repo_root / "runs" / "results" / f"qwen2p5_clean_main_table_{args.run_ts}"
  )
  output_root.mkdir(parents=True, exist_ok=True)
  per_run_dir = output_root / "per_run"
  per_run_dir.mkdir(parents=True, exist_ok=True)
  per_run_summary_path = output_root / "per_run_metrics.json"

  clean_benchmarks = _load_helper_module(
      IFEVAL_HELPER_MODULE, "eval_qwen2p5_clean_benchmarks"
  )
  rewardbench_module = _load_helper_module(
      REPO_ROOT / "examples/dpo/eval_qwen2p5_rewardbench_v2.py",
      "eval_qwen2p5_rewardbench_v2",
  )
  base_cfg = OmegaConf.load(args.config_path)
  legacy_payloads = _build_legacy_payloads(repo_root, args.legacy_run_ts)
  if args.no_legacy:
    legacy_payloads = {}
  selected_benchmarks = _normalize_benchmarks(list(args.benchmarks))
  runs = _discover_no_pref_main_table_runs(
      repo_root=repo_root,
      methods=args.methods,
      profile=args.profile,
  )

  rewardbench_rows = None
  if "rewardbench2" in selected_benchmarks:
    rewardbench_rows = rewardbench_module._load_rewardbench_v2_rows(  # pylint: disable=protected-access
        args.eval_limit
    )
  livebench_rows = None
  if "livebench_if" in selected_benchmarks:
    livebench_rows = _load_livebench_rows(args.eval_limit)
    livebench_data_path = (
        output_root / "livebench_if" / "data" / "livebench_instruction_following.jsonl"
    )
    clean_benchmarks._write_jsonl(  # pylint: disable=protected-access
        livebench_data_path,
        livebench_rows,
    )
  ifbench_prompts = None
  if "ifbench" in selected_benchmarks:
    ifbench_assets = clean_benchmarks._prepare_ifbench_assets(  # pylint: disable=protected-access
        output_root / "ifbench"
    )
    ifbench_prompts = clean_benchmarks._load_jsonl(  # pylint: disable=protected-access
        Path(ifbench_assets["input_path"]),
        args.eval_limit,
    )

  previous_payload = _load_json(per_run_summary_path)
  per_run_payload = {
      "run_ts": args.run_ts,
      "legacy_run_ts": None if args.no_legacy else args.legacy_run_ts,
      "columns": list(table_lib.MAIN_TABLE_COLUMNS),
      "runs": dict(previous_payload.get("runs", {})),
  }

  for run in runs:
    row_path = per_run_dir / f"{run['run_key']}.json"
    cached = _load_json(row_path) if row_path.exists() and not args.force else {}
    metrics = dict(cached.get("metrics", {}))

    if "clean_val_acc_auc" not in metrics or args.force:
      metrics["clean_val_acc_auc"] = _compute_normalized_scalar_auc(
          Path(run["tensorboard_dir"]),
          tag="dpo/eval/rewards/accuracy",
      )
    if "clean_test" in selected_benchmarks and (
        "clean_test_acc" not in metrics or args.force
    ):
      legacy_value = _lookup_legacy_metric(
          run=run,
          legacy_payloads=legacy_payloads,
          metric_name="clean_test_acc",
      )
      metrics["clean_test_acc"] = (
          legacy_value
          if legacy_value is not None and not args.force
          else _evaluate_clean_test_acc(
              run=run,
              base_cfg=base_cfg,
              sft_model_path=args.sft_model_path,
              eval_limit=args.eval_limit,
              num_batches=args.num_batches,
          )
      )
    if "livebench_if" in selected_benchmarks and (
        "livebench_if_score" not in metrics or args.force
    ):
      legacy_value = _lookup_legacy_metric(
          run=run,
          legacy_payloads=legacy_payloads,
          metric_name="livebench_if_score",
      )
      metrics["livebench_if_score"] = (
          legacy_value
          if legacy_value is not None and not args.force
          else _evaluate_livebench_if(
              clean_benchmarks=clean_benchmarks,
              repo_root=repo_root,
              run=run,
              output_root=output_root,
              rows=livebench_rows or [],
              args=args,
          )
      )
    if "rewardbench2" in selected_benchmarks and (
        "rewardbench2_precise_if" not in metrics or args.force
    ):
      legacy_value = _lookup_legacy_metric(
          run=run,
          legacy_payloads=legacy_payloads,
          metric_name="rewardbench2_precise_if",
      )
      metrics["rewardbench2_precise_if"] = (
          legacy_value
          if legacy_value is not None and not args.force
          else _evaluate_rewardbench2(
              rewardbench_module=rewardbench_module,
              run=run,
              rewardbench_rows=rewardbench_rows or [],
              base_cfg=base_cfg,
              sft_model_path=args.sft_model_path,
              batch_size=args.rewardbench_batch_size,
              output_root=output_root,
              force=args.force,
          )
      )
    if "ifbench" in selected_benchmarks and (
        "ifbench_prompt_strict" not in metrics or args.force
    ):
      legacy_value = _lookup_legacy_metric(
          run=run,
          legacy_payloads=legacy_payloads,
          metric_name="ifbench_prompt_strict",
      )
      metrics["ifbench_prompt_strict"] = (
          legacy_value
          if legacy_value is not None and not args.force
          else _evaluate_ifbench(
              clean_benchmarks=clean_benchmarks,
              run=run,
              output_root=output_root,
              prompts=ifbench_prompts or [],
              args=args,
          )
      )

    payload = {
        "run_key": run["run_key"],
        "logical_variant": run["logical_variant"],
        "display_name": run["display_name"],
        "seed": run["seed"],
        "source": run["source"],
        "source_variant": run["source_variant"],
        "source_run_ts": run["source_run_ts"],
        "run_name": run["run_name"],
        "run_dir": run["run_dir"],
        "metrics": metrics,
    }
    row_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    per_run_payload["runs"][run["run_key"]] = payload
    print(
        f"{run['run_key']}: "
        + ", ".join(
            f"{key}={metrics[key]:.4f}"
            for key in table_lib.MAIN_TABLE_COLUMNS
            if key in metrics
        )
    )

  per_run_summary_path.write_text(json.dumps(per_run_payload, indent=2, sort_keys=True))
  print(f"Wrote per-run clean main-table metrics to {per_run_summary_path}")

# === DYNAMIC_FILTER20_40_EVAL_PATCH_BEGIN ===
# Add filter20/filter40 aliases, display names, and stable method ordering.
# This is intentionally eval-side only; training keep ratios are parsed in
# qwen2p5_dpo_experiments.py.

_DYNAMIC_FILTER_EVAL_ALIASES = {
    "random_pair_filtering_filt20": "random_pair_filtering_filt20",
    "random pair filtering filt20": "random_pair_filtering_filt20",
    "random_pair_filtering_20": "random_pair_filtering_filt20",
    "random pair filtering 20": "random_pair_filtering_filt20",
    "random_pair_filtering_20pct": "random_pair_filtering_filt20",
    "random pair filtering 20pct": "random_pair_filtering_filt20",
    "random pair filtering (20%)": "random_pair_filtering_filt20",

    "random_pair_filtering_filt40": "random_pair_filtering_filt40",
    "random pair filtering filt40": "random_pair_filtering_filt40",
    "random_pair_filtering_40": "random_pair_filtering_filt40",
    "random pair filtering 40": "random_pair_filtering_filt40",
    "random_pair_filtering_40pct": "random_pair_filtering_filt40",
    "random pair filtering 40pct": "random_pair_filtering_filt40",
    "random pair filtering (40%)": "random_pair_filtering_filt40",

    "reward_based_filtering_filt20": "reward_based_filtering_filt20",
    "reward based filtering filt20": "reward_based_filtering_filt20",
    "reward_based_filtering_20": "reward_based_filtering_filt20",
    "reward based filtering 20": "reward_based_filtering_filt20",
    "reward_based_filtering_20pct": "reward_based_filtering_filt20",
    "reward based filtering 20pct": "reward_based_filtering_filt20",
    "reward-based filtering (20%)": "reward_based_filtering_filt20",

    "reward_based_filtering_filt40": "reward_based_filtering_filt40",
    "reward based filtering filt40": "reward_based_filtering_filt40",
    "reward_based_filtering_40": "reward_based_filtering_filt40",
    "reward based filtering 40": "reward_based_filtering_filt40",
    "reward_based_filtering_40pct": "reward_based_filtering_filt40",
    "reward based filtering 40pct": "reward_based_filtering_filt40",
    "reward-based filtering (40%)": "reward_based_filtering_filt40",
}

_DYNAMIC_FILTER_EVAL_DISPLAY = {
    "random_pair_filtering_filt20": "Random Pair Filtering (20%)",
    "random_pair_filtering_filt40": "Random Pair Filtering (40%)",
    "reward_based_filtering_filt20": "Reward-based Filtering (20%)",
    "reward_based_filtering_filt40": "Reward-based Filtering (40%)",
}

_DYNAMIC_FILTER_EVAL_ORDER = {
    "random_pair_filtering_filt20": 3.05,
    "random_pair_filtering_filt40": 3.10,
    "reward_based_filtering_filt20": 5.05,
    "reward_based_filtering_filt40": 5.10,
}

if "METHOD_ALIASES" in globals() and isinstance(METHOD_ALIASES, dict):
  METHOD_ALIASES.update(_DYNAMIC_FILTER_EVAL_ALIASES)

for _name in [
    "METHOD_DISPLAY",
    "METHOD_DISPLAY_NAMES",
    "DISPLAY_NAMES",
    "METHOD_NAMES",
]:
  if _name in globals() and isinstance(globals()[_name], dict):
    globals()[_name].update(_DYNAMIC_FILTER_EVAL_DISPLAY)

for _name in [
    "METHOD_ORDER",
    "METHOD_SORT_ORDER",
    "ORDER_BY_METHOD",
]:
  if _name in globals():
    _order_obj = globals()[_name]
    if isinstance(_order_obj, dict):
      _order_obj.update(_DYNAMIC_FILTER_EVAL_ORDER)
    elif isinstance(_order_obj, list):
      for _method in _DYNAMIC_FILTER_EVAL_DISPLAY:
        if _method not in _order_obj:
          _order_obj.append(_method)
    elif isinstance(_order_obj, tuple):
      for _method in _DYNAMIC_FILTER_EVAL_DISPLAY:
        if _method not in _order_obj:
          _order_obj = _order_obj + (_method,)
      globals()[_name] = _order_obj
# === DYNAMIC_FILTER20_40_EVAL_PATCH_END ===

# === NO_PREF_MAIN_EVAL_FILTER20_PATCH_BEGIN ===
# Add missing 20% fixed-ratio filtering aliases for no-pref main-table eval.
# The script already supports filt5/filt10/filt40; this fills the filt20 gap.

_FILTER20_ALIASES = {
    "reward_based_filtering_filt20": "reward_based_filtering_filt20",
    "reward based filtering filt20": "reward_based_filtering_filt20",
    "reward_based_filtering_20": "reward_based_filtering_filt20",
    "reward based filtering 20": "reward_based_filtering_filt20",
    "reward_based_filtering_20pct": "reward_based_filtering_filt20",
    "reward based filtering 20pct": "reward_based_filtering_filt20",
    "reward-based filtering (20%)": "reward_based_filtering_filt20",

    "random_pair_filtering_filt20": "random_pair_filtering_filt20",
    "random pair filtering filt20": "random_pair_filtering_filt20",
    "random_pair_filtering_20": "random_pair_filtering_filt20",
    "random pair filtering 20": "random_pair_filtering_filt20",
    "random_pair_filtering_20pct": "random_pair_filtering_filt20",
    "random pair filtering 20pct": "random_pair_filtering_filt20",
    "random pair filtering (20%)": "random_pair_filtering_filt20",
}

_FILTER20_DISPLAY = {
    "reward_based_filtering_filt20": "Reward-based Filtering (20%)",
    "random_pair_filtering_filt20": "Random Pair Filtering (20%)",
}

_FILTER20_ORDER = {
    "random_pair_filtering_filt20": 3.05,
    "reward_based_filtering_filt20": 5.05,
}

# Be robust to this file's historical naming choices.
for _dict_name in [
    "NO_PREF_METHOD_ALIASES",
    "METHOD_ALIASES",
    "METHOD_ALIAS",
    "VARIANT_ALIASES",
]:
  if _dict_name in globals() and isinstance(globals()[_dict_name], dict):
    globals()[_dict_name].update(_FILTER20_ALIASES)

for _dict_name in [
    "NO_PREF_METHOD_DISPLAY",
    "METHOD_DISPLAY",
    "METHOD_DISPLAY_NAMES",
    "DISPLAY_NAMES",
    "METHOD_NAMES",
]:
  if _dict_name in globals() and isinstance(globals()[_dict_name], dict):
    globals()[_dict_name].update(_FILTER20_DISPLAY)

for _order_name in [
    "NO_PREF_METHOD_ORDER",
    "METHOD_ORDER",
    "METHOD_SORT_ORDER",
    "ORDER_BY_METHOD",
]:
  if _order_name in globals():
    _obj = globals()[_order_name]
    if isinstance(_obj, dict):
      _obj.update(_FILTER20_ORDER)
    elif isinstance(_obj, list):
      for _method in _FILTER20_DISPLAY:
        if _method not in _obj:
          _obj.append(_method)
    elif isinstance(_obj, tuple):
      for _method in _FILTER20_DISPLAY:
        if _method not in _obj:
          _obj = _obj + (_method,)
      globals()[_order_name] = _obj
# === NO_PREF_MAIN_EVAL_FILTER20_PATCH_END ===

# === NO_PREF_MAIN_DYNAMIC_FILTER_NORMALIZE_PATCH_BEGIN ===
# Robust dynamic support for fixed-ratio reward/random filtering methods.
# This avoids manually adding every filtXX alias to the no-pref eval script.

_ORIGINAL_NORMALIZE_NO_PREF_METHOD_DYNAMIC_FILTER = _normalize_no_pref_method


def _dynamic_no_pref_filter_token_to_float(token: str) -> float:
  value = float(token.replace("p", "."))
  if not (0.0 <= value < 100.0):
    raise ValueError(
        f"Filtering percentage must be in [0, 100), got {value!r}."
    )
  return value


def _dynamic_no_pref_filter_percent_to_token(value: float) -> str:
  if float(value).is_integer():
    return str(int(value))
  return str(value).replace(".", "p")


def _dynamic_no_pref_filter_method(method: str) -> str | None:
  normalized = method.strip().lower()
  normalized = normalized.replace("-", "_").replace(" ", "_")
  normalized = normalized.replace("(", "").replace(")", "")
  normalized = normalized.replace("%", "pct")
  normalized = "_".join(part for part in normalized.split("_") if part)

  prefixes = (
      "reward_based_filtering",
      "random_pair_filtering",
  )

  for prefix in prefixes:
    filt_prefix = f"{prefix}_filt"
    if normalized.startswith(filt_prefix):
      token = normalized[len(filt_prefix):]
      if token:
        percent = _dynamic_no_pref_filter_token_to_float(token)
        token = _dynamic_no_pref_filter_percent_to_token(percent)
        return f"{prefix}_filt{token}"

    alias_prefix = f"{prefix}_"
    if normalized.startswith(alias_prefix):
      token = normalized[len(alias_prefix):]
      if token.endswith("pct"):
        token = token[:-3]
      if token:
        try:
          percent = _dynamic_no_pref_filter_token_to_float(token)
        except ValueError:
          continue
        token = _dynamic_no_pref_filter_percent_to_token(percent)
        return f"{prefix}_filt{token}"

  return None


def _normalize_no_pref_method(method: str) -> str:
  dynamic = _dynamic_no_pref_filter_method(method)
  if dynamic is not None:
    return dynamic
  return _ORIGINAL_NORMALIZE_NO_PREF_METHOD_DYNAMIC_FILTER(method)


_DYNAMIC_NO_PREF_FILTER_DISPLAY = {
    "reward_based_filtering_filt20": "Reward-based Filtering (20%)",
    "reward_based_filtering_filt40": "Reward-based Filtering (40%)",
    "random_pair_filtering_filt20": "Random Pair Filtering (20%)",
    "random_pair_filtering_filt40": "Random Pair Filtering (40%)",
}

_DYNAMIC_NO_PREF_FILTER_ORDER = {
    "random_pair_filtering_filt20": 3.05,
    "random_pair_filtering_filt40": 3.10,
    "reward_based_filtering_filt20": 5.05,
    "reward_based_filtering_filt40": 5.10,
}

for _dict_name in [
    "METHOD_DISPLAY",
    "METHOD_DISPLAY_NAMES",
    "DISPLAY_NAMES",
    "METHOD_NAMES",
    "NO_PREF_METHOD_DISPLAY",
]:
  if _dict_name in globals() and isinstance(globals()[_dict_name], dict):
    globals()[_dict_name].update(_DYNAMIC_NO_PREF_FILTER_DISPLAY)

for _order_name in [
    "METHOD_ORDER",
    "METHOD_SORT_ORDER",
    "ORDER_BY_METHOD",
    "NO_PREF_METHOD_ORDER",
]:
  if _order_name in globals():
    _obj = globals()[_order_name]
    if isinstance(_obj, dict):
      _obj.update(_DYNAMIC_NO_PREF_FILTER_ORDER)
    elif isinstance(_obj, list):
      for _method in _DYNAMIC_NO_PREF_FILTER_DISPLAY:
        if _method not in _obj:
          _obj.append(_method)
    elif isinstance(_obj, tuple):
      for _method in _DYNAMIC_NO_PREF_FILTER_DISPLAY:
        if _method not in _obj:
          _obj = _obj + (_method,)
      globals()[_order_name] = _obj
# === NO_PREF_MAIN_DYNAMIC_FILTER_NORMALIZE_PATCH_END ===

if __name__ == "__main__":
  main()
