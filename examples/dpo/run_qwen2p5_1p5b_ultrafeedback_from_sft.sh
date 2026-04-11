#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="/home/lhf_hongfu_gmail_com/.venvs/DPO"
ROOT_ENV_FILE="${REPO_ROOT}/.env"
EXAMPLE_ENV_FILE="${REPO_ROOT}/my_example/.env"
PROFILE="${1:-full}"

if [[ "${PROFILE}" == "full" || "${PROFILE}" == "smoke" ]]; then
  shift
else
  PROFILE="full"
fi

if [[ -n "${1:-}" && "$1" != --* ]]; then
  VARIANT="$1"
  shift
else
  VARIANT="baseline"
fi
FT_MODE="full"
SFT_MODEL_PATH=""
declare -a USER_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ft-mode)
      FT_MODE="${2:-}"
      shift 2
      ;;
    --ft-mode=*)
      FT_MODE="${1#*=}"
      shift
      ;;
    *)
      if [[ -z "${SFT_MODEL_PATH}" && "$1" != *=* ]]; then
        SFT_MODEL_PATH="$1"
      else
        USER_OVERRIDES+=("$1")
      fi
      shift
      ;;
  esac
done

case "${FT_MODE}" in
  full|lora)
    ;;
  *)
    echo "Unsupported --ft-mode: ${FT_MODE}" >&2
    exit 1
    ;;
esac

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
fi

if [[ -f "${ROOT_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ROOT_ENV_FILE}"
  set +a
elif [[ -f "${EXAMPLE_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${EXAMPLE_ENV_FILE}"
  set +a
fi

cd "${REPO_ROOT}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required to load the Qwen tokenizer." >&2
  exit 1
fi

if [[ -z "${SFT_MODEL_PATH}" || ! -d "${SFT_MODEL_PATH}" ]]; then
  echo "Usage: $0 [full|smoke] [baseline|outlier_l2|self_inf_batch] /path/to/sft_exported_model [--ft-mode full|lora] [overrides...]" >&2
  exit 1
fi

if [[ "${FT_MODE}" == "lora" ]]; then
  CONFIG_PATH="${REPO_ROOT}/examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft_lora.yaml"
else
  CONFIG_PATH="${REPO_ROOT}/examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft.yaml"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/dpo_qwen2p5_1p5b_ultrafeedback_from_sft_${VARIANT}_${FT_MODE}_${PROFILE}_${RUN_TS}}"
RUN_NAME="qwen2p5-1p5b-ultrafeedback-from-sft-${VARIANT}-${FT_MODE}-${PROFILE}-${RUN_TS}"
TMP_CONFIG="$(mktemp "/mnt/qwen2p5_1p5b_ultrafeedback_dpo_${VARIANT}_${FT_MODE}_${PROFILE}_XXXX.yaml")"
trap 'rm -f "${TMP_CONFIG}"' EXIT

"${PYTHON_BIN}" - <<'PY' \
  "${CONFIG_PATH}" \
  "${TMP_CONFIG}" \
  "${RUN_ROOT}" \
  "${RUN_NAME}" \
  "${PROFILE}" \
  "${VARIANT}" \
  "${SFT_MODEL_PATH}" \
  "${CURATION_THRESHOLD:-}" \
  "${SELF_INFLUENCE_DOT_THRESHOLD:-}"
from omegaconf import OmegaConf
import sys

(
    config_path,
    output_path,
    run_root,
    run_name,
    profile,
    variant,
    sft_model_path,
    curation_threshold_env,
    self_influence_dot_threshold_env,
) = sys.argv[1:]

cfg = OmegaConf.load(config_path)
cfg.actor_model_config.model_path = sft_model_path
cfg.reference_model_config.model_path = sft_model_path
cfg.training_config.checkpoint_root_directory = f"{run_root}/checkpoints"
cfg.training_config.metrics_logging_options.run_name = run_name
cfg.training_config.metrics_logging_options.log_dir = f"{run_root}/tensorboard"
cfg.exported_model_output_dir = f"{run_root}/exported_model"

if variant == "baseline":
  cfg.dpo_config.use_dynamic_batch_curation = False
elif variant == "outlier_l2":
  cfg.dpo_config.use_dynamic_batch_curation = True
  cfg.dpo_config.curation_variant = "outlier_l2"
  if curation_threshold_env:
    cfg.dpo_config.curation_threshold = float(curation_threshold_env)
elif variant == "self_inf_batch":
  cfg.dpo_config.use_dynamic_batch_curation = True
  cfg.dpo_config.curation_variant = "self_inf_batch"
  if self_influence_dot_threshold_env:
    cfg.dpo_config.self_influence_dot_threshold = float(
        self_influence_dot_threshold_env
    )
else:
  raise SystemExit(f"Unsupported variant: {variant}")

if profile == "smoke":
  cfg.train_data_module = (
      "examples/data/ultrafeedback_dpo.py:create_dataset("
      "split='train_prefs', partition='dpo', subset='train', "
      "sft_fraction=0.25, eval_fraction=0.1, seed=42, limit=512)"
  )
  cfg.eval_data_module = (
      "examples/data/ultrafeedback_dpo.py:create_dataset("
      "split='train_prefs', partition='dpo', subset='eval', "
      "sft_fraction=0.25, eval_fraction=0.1, seed=42, limit=64)"
  )
  cfg.batch_size = 2
  cfg.eval_batch_size = 2
  cfg.training_config.max_steps = 20
  cfg.training_config.eval_every_n_steps = 10
  cfg.training_config.gradient_accumulation_steps = 2
  cfg.optimizer_config.warmup_steps = 2
  cfg.optimizer_config.decay_steps = 20

OmegaConf.save(cfg, output_path)
PY

"${PYTHON_BIN}" -m tunix.cli.dpo_main \
  "${TMP_CONFIG}" \
  "${USER_OVERRIDES[@]}"
