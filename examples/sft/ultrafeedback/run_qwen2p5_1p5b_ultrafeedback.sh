#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PATH="/home/lhf_hongfu_gmail_com/.venvs/DPO"
ROOT_ENV_FILE="${REPO_ROOT}/.env"
EXAMPLE_ENV_FILE="${REPO_ROOT}/my_example/.env"
PROFILE="${1:-full}"

if [[ "${PROFILE}" == "full" || "${PROFILE}" == "smoke" ]]; then
  shift
else
  PROFILE="full"
fi

FT_MODE="full"
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
      USER_OVERRIDES+=("$1")
      shift
      ;;
  esac
done
set -- "${USER_OVERRIDES[@]}"

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
  echo "HF_TOKEN is required to download Qwen/Qwen2.5-1.5B." >&2
  exit 1
fi

if [[ "${FT_MODE}" == "lora" ]]; then
  CONFIG_PATH="${REPO_ROOT}/examples/sft/ultrafeedback/qwen2p5_1p5b_ultrafeedback_lora.yaml"
else
  CONFIG_PATH="${REPO_ROOT}/examples/sft/ultrafeedback/qwen2p5_1p5b_ultrafeedback.yaml"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/sft_qwen2p5_1p5b_ultrafeedback_${FT_MODE}_${PROFILE}_${RUN_TS}}"
RUN_NAME="qwen2p5-1p5b-ultrafeedback-sft-${FT_MODE}-${PROFILE}-${RUN_TS}"
TMP_CONFIG="$(mktemp "/tmp/qwen2p5_1p5b_ultrafeedback_${FT_MODE}_${PROFILE}_XXXX.yaml")"
trap 'rm -f "${TMP_CONFIG}"' EXIT

"${PYTHON_BIN}" - <<'PY' "${CONFIG_PATH}" "${TMP_CONFIG}" "${RUN_ROOT}" "${RUN_NAME}" "${PROFILE}"
from omegaconf import OmegaConf
import sys

config_path, output_path, run_root, run_name, profile = sys.argv[1:]
cfg = OmegaConf.load(config_path)

cfg.training_config.checkpoint_root_directory = f"{run_root}/checkpoints"
cfg.training_config.metrics_logging_options.run_name = run_name
cfg.training_config.metrics_logging_options.log_dir = f"{run_root}/tensorboard"
cfg.exported_model_output_dir = f"{run_root}/exported_model"

if profile == "smoke":
  cfg.train_data_module = (
      "examples/data/ultrafeedback_sft.py:create_dataset("
      "split='train_prefs', partition='sft', subset='train', "
      "sft_fraction=0.25, eval_fraction=0.1, seed=42, limit=256)"
  )
  cfg.eval_data_module = (
      "examples/data/ultrafeedback_sft.py:create_dataset("
      "split='train_prefs', partition='sft', subset='eval', "
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

"${PYTHON_BIN}" -m tunix.cli.peft_main \
  "${TMP_CONFIG}" \
  "${USER_OVERRIDES[@]}"
