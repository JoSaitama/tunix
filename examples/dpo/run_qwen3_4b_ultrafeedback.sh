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
  echo "HF_TOKEN is required to download Qwen/Qwen3-4B-Instruct-2507." >&2
  exit 1
fi

CONFIG_PATH="${REPO_ROOT}/examples/dpo/qwen3_4b_ultrafeedback.yaml"
PYTHON_BIN="${PYTHON_BIN:-python3}"
declare -a EXTRA_ARGS=()

if [[ "${PROFILE}" == "smoke" ]]; then
  EXTRA_ARGS+=(
    "train_data_module=examples/data/ultrafeedback_dpo.py:create_dataset(split='train_prefs', limit=512, seed=42)"
    "eval_data_module=examples/data/ultrafeedback_dpo.py:create_dataset(split='test_prefs', limit=64, seed=42)"
    "training_config.max_steps=20"
    "training_config.eval_every_n_steps=10"
    "training_config.gradient_accumulation_steps=8"
    "training_config.checkpoint_root_directory=/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen3_4b_ultrafeedback_smoke/checkpoints"
    "training_config.checkpointing_options.save_interval_steps=250"
    "training_config.checkpointing_options.max_to_keep=4"
    "training_config.metrics_logging_options.project_name=tunix"
    "training_config.metrics_logging_options.run_name=qwen3-4b-ultrafeedback-dpo-smoke"
    "training_config.metrics_logging_options.log_dir=/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen3_4b_ultrafeedback_smoke/tensorboard"
    "training_config.metrics_logging_options.flush_every_n_steps=20"
    "training_config.data_sharding_axis=['fsdp']"
    "training_config.max_inflight_computations=2"
    "training_config.metrics_prefix=dpo"
    "training_config.pbar_description=DPO"
    "optimizer_config.opt_type=adamw"
    "optimizer_config.schedule_type=warmup_cosine_decay_schedule"
    "optimizer_config.init_value=0.0"
    "optimizer_config.peak_value=5e-6"
    "optimizer_config.end_value=0.0"
    "optimizer_config.warmup_steps=2"
    "optimizer_config.decay_steps=20"
    "optimizer_config.b1=0.9"
    "optimizer_config.b2=0.99"
    "optimizer_config.weight_decay=0.1"
    "optimizer_config.max_grad_norm=0.1"
    "dpo_config.beta=0.01"
    "dpo_config.label_smoothing=0.0"
    "dpo_config.max_prompt_length=256"
    "dpo_config.max_response_length=256"
    "merged_model_output_dir=/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen3_4b_ultrafeedback_smoke/merged_lora"
  )
fi

"${PYTHON_BIN}" -m tunix.cli.dpo_main \
  "${CONFIG_PATH}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
