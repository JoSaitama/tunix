#!/usr/bin/env python3

"""Helpers for the multi-seed clean DPO main table."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys
import os
from typing import Any

REPO_ROOT = Path("/home/lhf_hongfu_gmail_com/tunix")
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from examples.dpo import qwen2p5_dpo_experiments as exp_lib

CONFIG_PATH = (
    REPO_ROOT / "examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft_lora.yaml"
)
SFT_MODEL_PATH = (
    REPO_ROOT
    / "runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/exported_model"
)
LEGACY_RUN_TS = "20260417_013847"
METHOD_ORDER = (
    "vanilla_dpo",
    "random_pair_filtering_filt5",
    "random_pair_filtering_filt10",
    "reward_based_filtering_filt5",
    "reward_based_filtering_filt10",
    "self_inf",
    "self_inf_loo",
    "self_inf_loo_cos",
)
METHOD_DISPLAY = {
    method: exp_lib.get_method_display_name(method) for method in METHOD_ORDER
}
DEFAULT_SEEDS = (0, 1, 2)
MAIN_TABLE_COLUMNS = (
    "clean_val_acc_auc",
    "clean_test_acc",
    "livebench_if_score",
    "rewardbench2_precise_if",
    "ifbench_prompt_strict",
)
LEGACY_VARIANT_MAP = {
    "vanilla_dpo": "vanilla_dpo",
    "random_pair_filtering_filt10": "random_pair_filtering",
    "reward_based_filtering_filt10": "reward_based_filtering",
    "self_inf": "self_inf",
}
METHOD_ALIASES = {
    "vanilla_dpo": "vanilla_dpo",
    "vanilla": "vanilla_dpo",
    "vanilla dpo": "vanilla_dpo",
    "random_pair_filtering": "random_pair_filtering_filt10",
    "random pair filtering": "random_pair_filtering_filt10",
    "random_pair_filtering_filt5": "random_pair_filtering_filt5",
    "random_pair_filtering_filt10": "random_pair_filtering_filt10",
    "random pair filtering (5%)": "random_pair_filtering_filt5",
    "random pair filtering (10%)": "random_pair_filtering_filt10",
    "reward_based_filtering": "reward_based_filtering_filt10",
    "reward based filtering": "reward_based_filtering_filt10",
    "reward-based filtering": "reward_based_filtering_filt10",
    "reward_based_filtering_filt5": "reward_based_filtering_filt5",
    "reward_based_filtering_filt10": "reward_based_filtering_filt10",
    "reward-based filtering (5%)": "reward_based_filtering_filt5",
    "reward-based filtering (10%)": "reward_based_filtering_filt10",
    "self_inf": "self_inf",
    "self_dtv": "self_inf",
    "self-dtv": "self_inf",
    "self-dtv (ours)": "self_inf",
}


def normalize_method(method: str) -> str:
  normalized = method.strip().lower()
  normalized_space = " ".join(normalized.replace("_", " ").replace("-", " ").split())
  candidates = (
      normalized,
      normalized_space,
      normalized_space.replace(" ", "_"),
  )
  for candidate in candidates:
    if candidate in METHOD_ALIASES:
      return METHOD_ALIASES[candidate]
  if normalized_space not in METHOD_ALIASES:
    raise ValueError(
        f"Unsupported clean main-table method {method!r}. Expected one of: "
        + ", ".join(sorted(METHOD_ALIASES))
    )
  return METHOD_ALIASES[normalized_space]


def run_key(logical_variant: str, seed: int) -> str:
  return f"{normalize_method(logical_variant)}_seed{int(seed)}"


def expected_run_specs(
    methods: Sequence[str] = METHOD_ORDER,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> list[dict[str, Any]]:
  return [
      {
          "logical_variant": normalize_method(method),
          "seed": int(seed),
          "run_key": run_key(method, seed),
      }
      for method in methods
      for seed in seeds
  ]


def discover_clean_main_table_runs(
    *,
    repo_root: Path,
    run_ts: str,
    legacy_run_ts: str | None = LEGACY_RUN_TS,
    allow_legacy_fallback: bool = True,
    methods: Sequence[str] = METHOD_ORDER,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    profile: str = "full",
    allow_missing: bool = False,
) -> list[dict[str, Any]]:
  """Discover seeded clean DPO runs and legacy seed-0 fallbacks."""
  repo_root = repo_root.resolve()
  runs_root = Path(
      os.environ.get("DPO_EVAL_RUNS_ROOT", str(repo_root / "runs_xuesong" / "model"))
  )
  wanted_methods = tuple(normalize_method(method) for method in methods)
  wanted_seeds = tuple(int(seed) for seed in seeds)

  parsed_runs: list[dict[str, Any]] = []
  for run_dir in sorted(
      runs_root.glob(
          os.environ.get(
              "DPO_EVAL_RUN_GLOB",
              "dpo_qwen2p5_1p5b_ultrafeedback_from_sft_*_clean_lora_*_*",
          )
      )
  ):
    if not run_dir.is_dir():
      continue
    parsed = exp_lib.parse_run_dir_name(run_dir.name)
    if (
        parsed is None
        or parsed["corruption_config"] != "clean"
        or parsed["profile"] != profile
    ):
      continue
    exported_model = run_dir / "exported_model"
    if not exported_model.is_dir():
      continue
    parsed_runs.append(
        {
            **parsed,
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "tensorboard_dir": str(run_dir / "tensorboard"),
            "exported_model_path": str(exported_model),
        }
    )

  seeded_index = {
      (row["variant"], row["seed"], row["run_ts"]): row
      for row in parsed_runs
      if row["seed"] is not None
  }
  legacy_index = {
      (row["variant"], row["run_ts"]): row
      for row in parsed_runs
      if row["seed"] is None
  }

  discovered: list[dict[str, Any]] = []
  for logical_variant in wanted_methods:
    for seed in wanted_seeds:
      selected = seeded_index.get((logical_variant, seed, run_ts))
      source = "seeded"
      if selected is None and seed == 0 and legacy_run_ts is not None:
        if allow_legacy_fallback:
          legacy_variant = LEGACY_VARIANT_MAP.get(logical_variant)
          if legacy_variant is not None:
            selected = legacy_index.get((legacy_variant, legacy_run_ts))
            if selected is not None:
              source = "legacy_seed0"
      if selected is None:
        if allow_missing:
          continue
        raise SystemExit(
            "Missing clean main-table run for "
            f"{logical_variant}@seed{seed} "
            f"(run_ts={run_ts}, legacy_run_ts={legacy_run_ts})"
        )
      discovered.append(
          {
              "logical_variant": logical_variant,
              "display_name": METHOD_DISPLAY[logical_variant],
              "seed": seed,
              "run_key": run_key(logical_variant, seed),
              "source": source,
              "source_variant": selected["variant"],
              "source_run_ts": selected["run_ts"],
              "run_name": selected["run_name"],
              "run_dir": selected["run_dir"],
              "tensorboard_dir": selected["tensorboard_dir"],
              "exported_model_path": selected["exported_model_path"],
          }
      )
  discovered.sort(
      key=lambda row: (
          METHOD_ORDER.index(row["logical_variant"]),
          row["seed"],
      )
  )
  return discovered


# ---------------------------------------------------------------------
# Self-Inf-lambda compatibility.
#
# Self-Inf-lambda uses the unified DTV-family score:
#   S_j(lambda) = g_j^T G_{-j} + lambda * ||g_j||^2
#
# The default experiment uses lambda = 0.5. This compatibility block only
# registers the method for discovery/evaluation. It does not affect any
# existing method or require YAML changes.
# ---------------------------------------------------------------------
if "self_inf_lambda" not in METHOD_ORDER:
  if isinstance(METHOD_ORDER, tuple):
    METHOD_ORDER = METHOD_ORDER + ("self_inf_lambda",)
  elif isinstance(METHOD_ORDER, list):
    METHOD_ORDER.append("self_inf_lambda")
  else:
    raise TypeError(f"Unsupported METHOD_ORDER type: {type(METHOD_ORDER)}")

if "METHOD_DISPLAY" in globals():
  METHOD_DISPLAY["self_inf_lambda"] = "Self-Inf-Lambda"
if "METHOD_DISPLAY_NAMES" in globals():
  METHOD_DISPLAY_NAMES["self_inf_lambda"] = "Self-Inf-Lambda"
if "DISPLAY_NAMES" in globals():
  DISPLAY_NAMES["self_inf_lambda"] = "Self-Inf-Lambda"
if "METHOD_NAMES" in globals():
  METHOD_NAMES["self_inf_lambda"] = "Self-Inf-Lambda"

if "METHOD_ALIASES" in globals():
  METHOD_ALIASES.update({
      "self_inf_lambda": "self_inf_lambda",
      "self_inf_lambda_batch": "self_inf_lambda",
    "self inf lambda batch": "self_inf_lambda",
    "self-inf lambda batch": "self_inf_lambda",
    "self-inf-lambda-batch": "self_inf_lambda",
    "self influence lambda batch": "self_inf_lambda",
    "self-influence-lambda-batch": "self_inf_lambda",
    "dtv lambda batch": "self_inf_lambda",
    "dtv-lambda-batch": "self_inf_lambda",
      "self inf lambda": "self_inf_lambda",
      "self-inf-lambda": "self_inf_lambda",
      "self-inf lambda": "self_inf_lambda",
      "self influence lambda": "self_inf_lambda",
      "self-influence-lambda": "self_inf_lambda",
      "self_dtv_lambda": "self_inf_lambda",
      "dtv_lambda": "self_inf_lambda",
      "dtv lambda": "self_inf_lambda",
  })

# === DYNAMIC_FILTER40_TABLE_PATCH_BEGIN ===
# Explicit table/eval support for the budget-matched 40% baselines.
# The training runner supports arbitrary filtXX dynamically; table ordering
# is kept explicit here so f40 appears in a stable position.

_DYNAMIC_FILTER40_METHODS = {
    "random_pair_filtering_filt40": "Random Pair Filtering (40%)",
    "reward_based_filtering_filt40": "Reward-based Filtering (40%)",
}

if isinstance(METHOD_ORDER, tuple):
  for _method in _DYNAMIC_FILTER40_METHODS:
    if _method not in METHOD_ORDER:
      METHOD_ORDER = METHOD_ORDER + (_method,)
elif isinstance(METHOD_ORDER, list):
  for _method in _DYNAMIC_FILTER40_METHODS:
    if _method not in METHOD_ORDER:
      METHOD_ORDER.append(_method)
else:
  raise TypeError(f"Unsupported METHOD_ORDER type: {type(METHOD_ORDER)}")

METHOD_DISPLAY.update(_DYNAMIC_FILTER40_METHODS)

METHOD_ALIASES.update({
    "random_pair_filtering_filt40": "random_pair_filtering_filt40",
    "random pair filtering filt40": "random_pair_filtering_filt40",
    "random_pair_filtering_40": "random_pair_filtering_filt40",
    "random pair filtering 40": "random_pair_filtering_filt40",
    "random_pair_filtering_40pct": "random_pair_filtering_filt40",
    "random pair filtering 40pct": "random_pair_filtering_filt40",
    "random pair filtering (40%)": "random_pair_filtering_filt40",
    "reward_based_filtering_filt40": "reward_based_filtering_filt40",
    "reward based filtering filt40": "reward_based_filtering_filt40",
    "reward_based_filtering_40": "reward_based_filtering_filt40",
    "reward based filtering 40": "reward_based_filtering_filt40",
    "reward_based_filtering_40pct": "reward_based_filtering_filt40",
    "reward based filtering 40pct": "reward_based_filtering_filt40",
    "reward-based filtering (40%)": "reward_based_filtering_filt40",
})
# === DYNAMIC_FILTER40_TABLE_PATCH_END ===
