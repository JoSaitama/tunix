# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for Qwen2.5 UltraFeedback DPO experiment launchers."""

from __future__ import annotations

import os

import re
from collections.abc import Mapping
from typing import Any

from omegaconf import OmegaConf


SUPPORTED_VARIANTS = (
    "vanilla_dpo",
    "random_pair_filtering",
    "random_pair_filtering_filt5",
    "random_pair_filtering_filt10",
    "reward_based_filtering",
    "reward_based_filtering_filt5",
    "reward_based_filtering_filt10",
    "self_inf",
    "self_inf_norm",
    "self_inf_loo",
    "self_inf_re_loo",
    "self_inf_lambda",
    "self_inf_loo_cos",
    "outlier_l2",
)
SUPPORTED_CORRUPTION_CONFIGS = (
    "clean",
    "tail50_flip20",
    "tail50_flip40",
    "global_flip10",
    "global_flip20",
    "global_flip30",
    "global_flip40",
    # jason's no_pref configs for ablation
    "global_no_pref10",
    "global_no_pref20",
    "global_no_pref40",
) + tuple(f"global_mismatch{pct}" for pct in range(1, 101))

_DATASET_MODULE = os.environ.get("DPO_DATASET_MODULE", "tunix/examples/data/ultrafeedback_dpo.py:create_dataset")
_DEFAULT_CURATION_KEEP_RATIO = 0.9
_FILTER_KEEP_RATIO_BY_LOGICAL_VARIANT = {
    "random_pair_filtering": 0.9,
    "random_pair_filtering_filt10": 0.9,
    "random_pair_filtering_filt5": 0.95,
    "reward_based_filtering": 0.9,
    "reward_based_filtering_filt10": 0.9,
    "reward_based_filtering_filt5": 0.95,
}
METHOD_DISPLAY = {
    "vanilla_dpo": "Vanilla DPO",
    "random_pair_filtering": "Random Pair Filtering (10%)",
    "random_pair_filtering_filt10": "Random Pair Filtering (10%)",
    "random_pair_filtering_filt5": "Random Pair Filtering (5%)",
    "reward_based_filtering": "Reward-based Filtering (10%)",
    "reward_based_filtering_filt10": "Reward-based Filtering (10%)",
    "reward_based_filtering_filt5": "Reward-based Filtering (5%)",
    "self_inf": "Self-DTV (ours)",
    "self_inf_norm": "Self-DTV-Norm",
    "self_inf_loo": "Self-DTV-LOO",
    "self_inf_re_loo": "Self-DTV-Re-LOO",
    "self_inf_lambda": "Self-Inf-Lambda",
    "self_inf_loo_cos": "Self-DTV-LOO-Cos",
    "outlier_l2": "Outlier L2",
}
_BASE_DATASET_KWARGS: dict[str, Any] = {
    "split": "train_prefs",
    "partition": "dpo",
    "sft_fraction": 0.25,
    "eval_fraction": 0.1,
    "seed": 42,
}
_CORRUPTION_SPECS: dict[str, dict[str, Any]] = {
    "clean": {
        "flip_scope": "none",
        "flip_ratio": 0.0,
        "flip_tail_fraction": 0.5,
        "flip_seed": 123,
    },
    "tail50_flip20": {
        "flip_scope": "tail_fraction",
        "flip_ratio": 0.2,
        "flip_tail_fraction": 0.5,
        "flip_seed": 123,
    },
    "tail50_flip40": {
        "flip_scope": "tail_fraction",
        "flip_ratio": 0.4,
        "flip_tail_fraction": 0.5,
        "flip_seed": 123,
    },
    "global_flip10": {
        "flip_scope": "global",
        "flip_ratio": 0.1,
        "flip_tail_fraction": 0.5,
        "flip_seed": 123,
    },
    "global_flip20": {
        "flip_scope": "global",
        "flip_ratio": 0.2,
        "flip_tail_fraction": 0.5,
        "flip_seed": 123,
    },
    "global_flip30": {
        "flip_scope": "global",
        "flip_ratio": 0.3,
        "flip_tail_fraction": 0.5,
        "flip_seed": 123,
    },
    "global_flip40": {
        "flip_scope": "global",
        "flip_ratio": 0.4,
        "flip_tail_fraction": 0.5,
        "flip_seed": 123,
    },
    #jason's no_pref configs for ablation
    "global_no_pref10": {
        "flip_scope": "none",
        "flip_ratio": 0.0,
        "flip_tail_fraction": 0.5,
        "flip_seed": 123,
        "no_pref_scope": "global",
        "no_pref_ratio": 0.1,
        "no_pref_tail_fraction": 0.5,
        "no_pref_seed": 123,
        "no_pref_mode": "duplicate_chosen",
    },
    "global_no_pref20": {
        "flip_scope": "none",
        "flip_ratio": 0.0,
        "flip_tail_fraction": 0.5,
        "flip_seed": 123,
        "no_pref_scope": "global",
        "no_pref_ratio": 0.2,
        "no_pref_tail_fraction": 0.5,
        "no_pref_seed": 123,
        "no_pref_mode": "duplicate_chosen",
    },
    "global_no_pref40": {
        "flip_scope": "none",
        "flip_ratio": 0.0,
        "flip_tail_fraction": 0.5,
        "flip_seed": 123,
        "no_pref_scope": "global",
        "no_pref_ratio": 0.4,
        "no_pref_tail_fraction": 0.5,
        "no_pref_seed": 123,
        "no_pref_mode": "duplicate_chosen",
    },
}

def _is_global_mismatch_config(corruption_config: str) -> bool:
  return re.fullmatch(r"global_mismatch([1-9][0-9]?|100)", corruption_config) is not None


def _build_global_mismatch_spec(corruption_config: str) -> dict[str, Any]:
  match = re.fullmatch(r"global_mismatch([1-9][0-9]?|100)", corruption_config)
  if match is None:
    raise ValueError(f"Invalid global mismatch corruption config: {corruption_config!r}.")

  pct = int(match.group(1))
  return {
      "flip_scope": "none",
      "flip_ratio": 0.0,
      "flip_tail_fraction": 0.5,
      "flip_seed": 123,
      "no_pref_scope": "none",
      "no_pref_ratio": 0.0,
      "no_pref_tail_fraction": 0.5,
      "no_pref_seed": 123,
      "no_pref_mode": "duplicate_chosen",
      "mismatch_scope": "global",
      "mismatch_ratio": pct / 100.0,
      "mismatch_seed": 123,
      "mismatch_mode": "response_pair",
  }


def _get_corruption_spec(corruption_config: str) -> dict[str, Any]:
  if corruption_config in _CORRUPTION_SPECS:
    return _CORRUPTION_SPECS[corruption_config]
  if _is_global_mismatch_config(corruption_config):
    return _build_global_mismatch_spec(corruption_config)
  raise ValueError(f"Unknown corruption config: {corruption_config!r}.")



RUN_DIR_RE = re.compile(
    rf"{re.escape(os.environ.get('DPO_RUN_DIR_PREFIX', 'dpo_qwen2p5_1p5b_ultrafeedback_from_sft'))}_"
    r"(?P<variant>.+)_(?P<corruption_config>clean|tail50_flip20|tail50_flip40|global_flip10|global_flip20|global_flip30|global_flip40|global_no_pref10|global_no_pref20|global_no_pref40|global_mismatch\d+)_"
    r"(?P<ft_mode>full|lora)_(?P<profile>full|smoke)"
    r"(?:_seed(?P<seed>\d+))?_(?P<run_ts>[A-Za-z0-9_]+)"
)


def _format_module_spec(path: str, kwargs: Mapping[str, Any]) -> str:
  args = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
  return f"{path}({args})"


def _normalize_variant(variant: str) -> str:
  normalized = variant.strip().lower()
  normalized = normalized.replace("%", "")
  normalized = normalized.replace("(", " ").replace(")", " ")
  normalized = re.sub(r"[\s\-]+", "_", normalized)
  normalized = re.sub(r"_+", "_", normalized).strip("_")
  alias_map = {
      "baseline": "vanilla_dpo",
      "vanilla_dpo": "vanilla_dpo",
      "outlier_l2": "outlier_l2",
      "random_pair_filtering": "random_pair_filtering",
      "random_pair_filtering_filt10": "random_pair_filtering_filt10",
      "random_pair_filtering_10": "random_pair_filtering_filt10",
      "random_pair_filtering_10pct": "random_pair_filtering_filt10",
      "random_pair_filtering_5": "random_pair_filtering_filt5",
      "random_pair_filtering_filt5": "random_pair_filtering_filt5",
      "random_pair_filtering_5pct": "random_pair_filtering_filt5",
      "reward_based_filtering": "reward_based_filtering",
      "reward_based_filtering_filt10": "reward_based_filtering_filt10",
      "reward_based_filtering_10": "reward_based_filtering_filt10",
      "reward_based_filtering_10pct": "reward_based_filtering_filt10",
      "reward_based_filtering_filt5": "reward_based_filtering_filt5",
      "reward_based_filtering_5": "reward_based_filtering_filt5",
      "reward_based_filtering_5pct": "reward_based_filtering_filt5",
      "self_inf": "self_inf",
      "self_inf_batch": "self_inf",
      "self_influence_batch": "self_inf",
      "self_dtv": "self_inf",
      "self_dtv_ours": "self_inf",
      "self_inf_norm": "self_inf_norm",
      "self_inf_norm_batch": "self_inf_norm",
      "self_influence_norm": "self_inf_norm",
      "self_influence_norm_batch": "self_inf_norm",
      "self_dtv_norm": "self_inf_norm",
      "dtv_norm": "self_inf_norm",
      "self_inf_loo": "self_inf_loo",
      "self_inf_re_loo": "self_inf_re_loo",
      "self_inf_re_loo_batch": "self_inf_re_loo",
      "self_inf_reliable_loo": "self_inf_re_loo",
      "self_inf_reliable_loo_batch": "self_inf_re_loo",
      "self_dtv_re_loo": "self_inf_re_loo",
      "dtv_re_loo": "self_inf_re_loo",
      "self_inf_lambda": "self_inf_lambda",
      "self_inf_lambda_batch": "self_inf_lambda",
      "self_influence_lambda": "self_inf_lambda",
      "self_influence_lambda_batch": "self_inf_lambda",
      "self influence lambda batch": "self_inf_lambda",
      "self-influence-lambda-batch": "self_inf_lambda",
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
      "self_inf_loo_batch": "self_inf_loo",
      "self_dtv_loo": "self_inf_loo",
      "self_inf_loo_cos": "self_inf_loo_cos",
      "self_inf_loo_cos_batch": "self_inf_loo_cos",
      "self_dtv_loo_cos": "self_inf_loo_cos",
  }
  if normalized not in alias_map:
    raise ValueError(
        "variant must be one of "
        f"{SUPPORTED_VARIANTS} or their legacy aliases; got {variant!r}."
    )
  return alias_map[normalized]


def _normalize_corruption_config(corruption_config: str) -> str:
  normalized = corruption_config.strip().lower()
  if normalized in _CORRUPTION_SPECS or _is_global_mismatch_config(normalized):
    return normalized
  raise ValueError(
      "corruption_config must be one of "
      f"{SUPPORTED_CORRUPTION_CONFIGS}; got {corruption_config!r}."
  )


def get_method_display_name(variant: str) -> str:
  return METHOD_DISPLAY[_normalize_variant(variant)]


def get_underlying_trainer_variant(variant: str) -> str:
  normalized = _normalize_variant(variant)
  if normalized.startswith("random_pair_filtering"):
    return "random_pair_filtering"
  if normalized.startswith("reward_based_filtering"):
    return "reward_based_filtering"
  return normalized


def get_curation_keep_ratio_for_variant(
    variant: str,
    override_keep_ratio: str | None = None,
) -> float | None:
  normalized = _normalize_variant(variant)
  if normalized not in _FILTER_KEEP_RATIO_BY_LOGICAL_VARIANT:
    return (
        float(override_keep_ratio)
        if override_keep_ratio not in (None, "")
        else None
    )
  if override_keep_ratio not in (None, ""):
    return float(override_keep_ratio)
  return _FILTER_KEEP_RATIO_BY_LOGICAL_VARIANT[normalized]


def parse_run_dir_name(run_name: str) -> dict[str, Any] | None:
  match = RUN_DIR_RE.fullmatch(run_name)
  if match is None:
    return None
  raw_variant = match.group("variant")
  normalized_variant = _normalize_variant(raw_variant)
  seed_text = match.group("seed")
  parsed = {
      "raw_variant": raw_variant,
      "variant": normalized_variant,
      "corruption_config": match.group("corruption_config"),
      "ft_mode": match.group("ft_mode"),
      "profile": match.group("profile"),
      "seed": int(seed_text) if seed_text is not None else None,
      "run_ts": match.group("run_ts"),
      "display_name": get_method_display_name(normalized_variant),
  }
  return parsed


def build_dataset_module_spec(
    *,
    subset: str,
    corruption_config: str,
    profile: str,
    train_shuffle_seed: int | None = None,
) -> str:
  """Builds a dataset module spec for the requested corruption setting."""
  corruption_config = _normalize_corruption_config(corruption_config)
  kwargs = dict(_BASE_DATASET_KWARGS)
  kwargs["subset"] = subset
  if subset == "train":
    if train_shuffle_seed is not None:
      kwargs["shuffle_seed"] = int(train_shuffle_seed)
    kwargs.update(_get_corruption_spec(corruption_config))
    if profile == "smoke":
      kwargs["limit"] = 512
  else:
    kwargs["flip_scope"] = "none"
    kwargs["flip_ratio"] = 0.0
    kwargs["flip_tail_fraction"] = 0.5
    kwargs["flip_seed"] = 123
    if profile == "smoke":
      kwargs["limit"] = 64
  return _format_module_spec(_DATASET_MODULE, kwargs)


def build_run_root(
    *,
    repo_root: str,
    variant: str,
    corruption_config: str,
    ft_mode: str,
    profile: str,
    run_ts: str,
    seed: int | None = None,
) -> str:
  variant = _normalize_variant(variant)
  corruption_config = _normalize_corruption_config(corruption_config)
  seed_suffix = f"_seed{seed}" if seed is not None else ""
  return (
      f"{repo_root}/runs/{os.environ.get('DPO_RUN_DIR_PREFIX', 'dpo_qwen2p5_1p5b_ultrafeedback_from_sft')}_"
      f"{variant}_{corruption_config}_{ft_mode}_{profile}{seed_suffix}_{run_ts}"
  )


def build_run_name(
    *,
    variant: str,
    corruption_config: str,
    ft_mode: str,
    profile: str,
    run_ts: str,
    seed: int | None = None,
) -> str:
  variant = _normalize_variant(variant)
  corruption_config = _normalize_corruption_config(corruption_config)
  seed_fragment = f"-seed{seed}" if seed is not None else ""
  return (
      f"{os.environ.get('DPO_RUN_DATASET_TAG', 'qwen2p5-1p5b-ultrafeedback-from-sft')}-"
      f"{variant}-{corruption_config}-{ft_mode}-{profile}{seed_fragment}-{run_ts}"
  )


def prepare_launch_config(
    *,
    config_path: str,
    output_path: str,
    run_root: str,
    run_name: str,
    profile: str,
    variant: str,
    corruption_config: str,
    sft_model_path: str,
    curation_threshold: str,
    curation_keep_ratio: str,
    self_influence_dot_threshold: str,
    dpo_seed: int | None = None,
    curation_seed: int | None = None,
    train_shuffle_seed: int | None = None,
) -> None:
  """Creates a temp config for a Qwen2.5 DPO launch."""
  variant = _normalize_variant(variant)
  corruption_config = _normalize_corruption_config(corruption_config)

  cfg = OmegaConf.load(config_path)
  cfg.actor_model_config.model_path = sft_model_path
  cfg.reference_model_config.model_path = sft_model_path
  if dpo_seed is not None:
    cfg.actor_model_config.rng_seed = int(dpo_seed)
    cfg.reference_model_config.rng_seed = int(dpo_seed)
  cfg.training_config.checkpoint_root_directory = f"{run_root}/checkpoints"
  cfg.training_config.metrics_logging_options.run_name = run_name
  cfg.training_config.metrics_logging_options.log_dir = f"{run_root}/tensorboard"
  cfg.exported_model_output_dir = f"{run_root}/exported_model"

  if os.environ.get("DPO_TIMING_ONLY", "0") == "1":
    cfg.training_config.checkpoint_root_directory = None
    cfg.training_config.checkpointing_options = None
    cfg.exported_model_output_dir = None

    if "merged_model_output_dir" in cfg:
      cfg.merged_model_output_dir = None

    print(
        "[DPO_TIMING_ONLY] Checkpointing and model export are disabled.",
        flush=True,
    )

  cfg.train_data_module = build_dataset_module_spec(
      subset="train",
      corruption_config=corruption_config,
      profile=profile,
      train_shuffle_seed=train_shuffle_seed,
  )
  cfg.eval_data_module = build_dataset_module_spec(
      subset="eval",
      corruption_config="clean",
      profile=profile,
  )
  cfg.dpo_config.late_flip_ratio = 0.0
  cfg.dpo_config.late_flip_start_step = None
  if curation_seed is not None:
    cfg.dpo_config.curation_seed = int(curation_seed)

  trainer_variant = get_underlying_trainer_variant(variant)
  if trainer_variant == "vanilla_dpo":
    cfg.dpo_config.use_dynamic_batch_curation = False
  elif trainer_variant == "outlier_l2":
    cfg.dpo_config.use_dynamic_batch_curation = True
    cfg.dpo_config.curation_variant = "outlier_l2"
    if curation_threshold:
      cfg.dpo_config.curation_threshold = float(curation_threshold)
  elif trainer_variant in {"self_inf", "self_inf_norm", "self_inf_loo", "self_inf_re_loo", "self_inf_lambda", "self_inf_loo_cos"}:
    cfg.dpo_config.use_dynamic_batch_curation = True
    cfg.dpo_config.curation_variant = {
        "self_inf": "self_inf_batch",
        "self_inf_norm": "self_inf_norm_batch",
        "self_inf_loo": "self_inf_loo_batch",
        "self_inf_re_loo": "self_inf_re_loo_batch",
        "self_inf_lambda": "self_inf_lambda_batch",
        "self_inf_loo_cos": "self_inf_loo_cos_batch",
    }[trainer_variant]
    if self_influence_dot_threshold:
      cfg.dpo_config.self_influence_dot_threshold = float(
          self_influence_dot_threshold
      )
  else:
    cfg.dpo_config.use_dynamic_batch_curation = True
    cfg.dpo_config.curation_variant = trainer_variant
    resolved_keep_ratio = get_curation_keep_ratio_for_variant(
        variant, curation_keep_ratio
    )
    cfg.dpo_config.curation_keep_ratio = (
        resolved_keep_ratio
        if resolved_keep_ratio is not None
        else _DEFAULT_CURATION_KEEP_RATIO
    )

  if profile == "smoke":
    cfg.batch_size = 2
    cfg.eval_batch_size = 2
    cfg.training_config.max_steps = 20
    cfg.training_config.eval_every_n_steps = 10
    cfg.training_config.gradient_accumulation_steps = 2
    cfg.optimizer_config.warmup_steps = 2
    cfg.optimizer_config.decay_steps = 20

  OmegaConf.save(cfg, output_path)

# === DYNAMIC_FILTER_PERCENT_PATCH_BEGIN ===
# Support dynamic fixed-ratio filtering variants such as:
#   random_pair_filtering_filt20  -> keep 80%
#   random_pair_filtering_filt40  -> keep 60%
#   reward_based_filtering_filt20 -> keep 80%
#   reward_based_filtering_filt40 -> keep 60%
#
# This keeps existing filt5/filt10 behavior unchanged and avoids hard-coding
# every new filtering percentage in the trainer.

_ORIGINAL_NORMALIZE_VARIANT_FOR_DYNAMIC_FILTER = _normalize_variant
_ORIGINAL_GET_METHOD_DISPLAY_NAME_FOR_DYNAMIC_FILTER = get_method_display_name


def _dynamic_filter_percent_token_to_float(token: str) -> float:
  value = float(token.replace("p", "."))
  if not (0.0 <= value < 100.0):
    raise ValueError(
        f"Filtering percentage must be in [0, 100), got {value!r}."
    )
  return value


def _dynamic_filter_percent_to_token(value: float) -> str:
  if float(value).is_integer():
    return str(int(value))
  return str(value).replace(".", "p")


def _dynamic_filter_variant_parts(variant: str):
  normalized = variant.strip().lower().replace("-", "_").replace(" ", "_")
  normalized = "_".join(part for part in normalized.split("_") if part)

  prefixes = (
      "random_pair_filtering",
      "reward_based_filtering",
  )
  for prefix in prefixes:
    filt_prefix = f"{prefix}_filt"
    if normalized.startswith(filt_prefix):
      token = normalized[len(filt_prefix):]
      if token:
        percent = _dynamic_filter_percent_token_to_float(token)
        canonical = f"{prefix}_filt{_dynamic_filter_percent_to_token(percent)}"
        return prefix, percent, canonical

    alias_prefix = f"{prefix}_"
    if normalized.startswith(alias_prefix):
      token = normalized[len(alias_prefix):]
      if token.endswith("pct"):
        token = token[:-3]
      if token:
        try:
          percent = _dynamic_filter_percent_token_to_float(token)
        except ValueError:
          continue
        canonical = f"{prefix}_filt{_dynamic_filter_percent_to_token(percent)}"
        return prefix, percent, canonical

  return None


def _normalize_variant(variant: str) -> str:
  parts = _dynamic_filter_variant_parts(variant)
  if parts is not None:
    return parts[2]
  return _ORIGINAL_NORMALIZE_VARIANT_FOR_DYNAMIC_FILTER(variant)


def _dynamic_filter_keep_ratio_for_variant(variant: str) -> float | None:
  parts = _dynamic_filter_variant_parts(variant)
  if parts is None:
    return None
  _, percent, _ = parts
  return 1.0 - percent / 100.0


def get_curation_keep_ratio_for_variant(
    variant: str,
    override_keep_ratio: str | None = None,
) -> float | None:
  if override_keep_ratio not in (None, ""):
    return float(override_keep_ratio)

  normalized = _normalize_variant(variant)

  dynamic_keep_ratio = _dynamic_filter_keep_ratio_for_variant(normalized)
  if dynamic_keep_ratio is not None:
    return dynamic_keep_ratio

  if normalized not in _FILTER_KEEP_RATIO_BY_LOGICAL_VARIANT:
    return None
  return _FILTER_KEEP_RATIO_BY_LOGICAL_VARIANT[normalized]


def get_method_display_name(variant: str) -> str:
  normalized = _normalize_variant(variant)
  parts = _dynamic_filter_variant_parts(normalized)
  if parts is not None:
    prefix, percent, _ = parts
    pct_text = _dynamic_filter_percent_to_token(percent).replace("p", ".")
    if prefix == "random_pair_filtering":
      return f"Random Pair Filtering ({pct_text}%)"
    if prefix == "reward_based_filtering":
      return f"Reward-based Filtering ({pct_text}%)"
  return _ORIGINAL_GET_METHOD_DISPLAY_NAME_FOR_DYNAMIC_FILTER(variant)


for _variant in (
    "random_pair_filtering_filt20",
    "random_pair_filtering_filt40",
    "reward_based_filtering_filt20",
    "reward_based_filtering_filt40",
):
  if _variant not in SUPPORTED_VARIANTS:
    SUPPORTED_VARIANTS = tuple(SUPPORTED_VARIANTS) + (_variant,)
# === DYNAMIC_FILTER_PERCENT_PATCH_END ===
