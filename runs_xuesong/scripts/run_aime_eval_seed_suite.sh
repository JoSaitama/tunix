#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/jason_chia925_gmail_com/Project/tunix}"
VENV="${VENV:-${REPO}/.venv}"
TPU_NAME="${TPU_NAME:-node-v5p-16-ziao1}"
ZONE="${ZONE:-us-central1-a}"
MODEL_PATH="${MODEL_PATH:-/home/lhf_hongfu_gmail_com/models/deepseek-r1-distill-qwen-1.5b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-/home/lhf_hongfu_gmail_com/tunix-hf-data/aime_eval.parquet}"
LOO_RUN_ROOT="${LOO_RUN_ROOT:-${REPO}/runs_xuesong/runs/grpo_aime_dtv_selfinf_group_loo_policy_seed0_clean_20260805_095513}"
DTV_RUN_ROOT="${DTV_RUN_ROOT:-${REPO}/runs_xuesong/runs/grpo_aime_dtv_selfinf_group_policy_seed0_clean_20260805_101839}"
BASELINE_RUN_ROOT="${BASELINE_RUN_ROOT:-${REPO}/runs_xuesong/runs/grpo_aime_baseline_seed0_clean_20260813_020931}"
BASE_SOURCE_ROOT="${BASE_SOURCE_ROOT:-${REPO}/runs_xuesong/eval_sources/deepseek_r1_distill_qwen_1p5b}"
SUITE_ROOT="${SUITE_ROOT:-${REPO}/runs_xuesong/evals/aime2024_k16_multiseed_fsdp_tp}"
EVAL_MODELS="${EVAL_MODELS:-dtv_loo pretrain}"
EVAL_PROTOCOL_NAME="${EVAL_PROTOCOL_NAME:-aime2024_8k_constrained}"
EVAL_BASE_MODEL_LOAD_MODE="${EVAL_BASE_MODEL_LOAD_MODE:-nnx_sync}"
EVAL_CHECKPOINT_STEP="${EVAL_CHECKPOINT_STEP:-314}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-16}"
EVAL_MAX_PROMPT_LENGTH="${EVAL_MAX_PROMPT_LENGTH:-2048}"
EVAL_MAX_GENERATION_STEPS="${EVAL_MAX_GENERATION_STEPS:-8192}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.6}"
EVAL_TOP_P="${EVAL_TOP_P:-0.95}"
EVAL_PROBLEM_BATCH_SIZE="${EVAL_PROBLEM_BATCH_SIZE:-16}"
EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-16}"
EVAL_MAX_NUM_BATCHED_TOKENS="${EVAL_MAX_NUM_BATCHED_TOKENS:-38400}"
EVAL_TENSOR_PARALLEL_SIZE="${EVAL_TENSOR_PARALLEL_SIZE:--1}"
EVAL_DATA_PARALLEL_SIZE="${EVAL_DATA_PARALLEL_SIZE:--1}"
EVAL_VLLM_HBM_UTILIZATION="${EVAL_VLLM_HBM_UTILIZATION:-0.8}"
EVAL_VLLM_SERVER_MODE="${EVAL_VLLM_SERVER_MODE:-false}"
EVAL_VLLM_ASYNC_SCHEDULING="${EVAL_VLLM_ASYNC_SCHEDULING:-false}"
EVAL_VLLM_ENABLE_PREFIX_CACHING="${EVAL_VLLM_ENABLE_PREFIX_CACHING:-false}"
TPU_RELEASE_ATTEMPTS="${TPU_RELEASE_ATTEMPTS:-30}"
TPU_RELEASE_INTERVAL_SECONDS="${TPU_RELEASE_INTERVAL_SECONDS:-10}"

for value_name in \
  EVAL_NUM_SAMPLES EVAL_MAX_PROMPT_LENGTH EVAL_MAX_GENERATION_STEPS \
  EVAL_PROBLEM_BATCH_SIZE EVAL_MAX_NUM_SEQS EVAL_MAX_NUM_BATCHED_TOKENS \
  EVAL_CHECKPOINT_STEP; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${value_name} must be a positive integer; received ${value}." >&2
    exit 2
  fi
done
if [[ -n "$EVAL_LIMIT" && ! "$EVAL_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EVAL_LIMIT must be empty or a positive integer." >&2
  exit 2
fi
for value_name in \
  EVAL_VLLM_SERVER_MODE EVAL_VLLM_ASYNC_SCHEDULING \
  EVAL_VLLM_ENABLE_PREFIX_CACHING; do
  value="${!value_name}"
  if [[ "$value" != "true" && "$value" != "false" ]]; then
    echo "ERROR: ${value_name} must be true or false; received ${value}." >&2
    exit 2
  fi
done
if [[ "$EVAL_VLLM_ASYNC_SCHEDULING" == "true" && "$EVAL_VLLM_SERVER_MODE" != "true" ]]; then
  echo "ERROR: async scheduling requires server mode." >&2
  exit 2
fi

EXPECTED_PROBLEMS="${EVAL_LIMIT:-30}"
EXPECTED_SAMPLES=$((EXPECTED_PROBLEMS * EVAL_NUM_SAMPLES))

SEEDS=(0 5 13 21 42)
if [[ $# -gt 0 ]]; then
  SEEDS=("$@")
fi

mkdir -p "$SUITE_ROOT" "$BASE_SOURCE_ROOT"
STATUS_FILE="${SUITE_ROOT}/status.tsv"
if [[ ! -f "$STATUS_FILE" ]]; then
  printf 'seed\tmodel\tstatus\texit_code\toutput_dir\n' > "$STATUS_FILE"
fi

worker0_is_free() {
  sudo -n true >/dev/null 2>&1 &&
    ! sudo -n fuser /dev/vfio/0 >/dev/null 2>&1 &&
    ! pgrep -f '[e]val_final_checkpoint_metrics|[v]llm' >/dev/null 2>&1
}

worker1_is_free() {
  local result
  result="$(gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --worker=1 \
    --internal-ip \
    --command='if sudo -n true >/dev/null 2>&1 && ! sudo -n fuser /dev/vfio/0 >/dev/null 2>&1 && ! pgrep -f "[e]val_final_checkpoint_metrics|[v]llm" >/dev/null 2>&1; then echo WORKER1_FREE; else echo WORKER1_BUSY; fi' \
    2>/dev/null)" || return 1
  grep -q '^WORKER1_FREE$' <<< "$result"
}

clear_stale_locks() {
  if worker0_is_free; then
    sudo -n rm -f /tmp/libtpu_lockfile
  else
    return 1
  fi
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --worker=1 \
    --internal-ip \
    --command='sudo -n rm -f /tmp/libtpu_lockfile' \
    >/dev/null 2>&1
}

wait_for_tpu_release() {
  local attempt
  for ((attempt = 1; attempt <= TPU_RELEASE_ATTEMPTS; attempt++)); do
    if worker0_is_free && worker1_is_free; then
      clear_stale_locks
      echo "TPU_RELEASED attempt=${attempt}"
      return 0
    fi
    echo "WAITING_FOR_TPU_RELEASE attempt=${attempt}/${TPU_RELEASE_ATTEMPTS}"
    sleep "$TPU_RELEASE_INTERVAL_SECONDS"
  done
  echo "ERROR: TPU workers did not become free within the bounded wait." >&2
  return 1
}

completed_output() {
  local output_dir="$1"
  [[ -f "${output_dir}/logs/launcher.status" ]] &&
    grep -qx 'EXIT_CODE=0' "${output_dir}/logs/launcher.status" &&
    [[ -f "${output_dir}/samples.jsonl" ]] &&
    [[ "$(wc -l < "${output_dir}/samples.jsonl")" -eq "$EXPECTED_SAMPLES" ]] &&
    [[ -f "${output_dir}/summary.json" ]] &&
    [[ -f "${output_dir}/.primary_done" ]] &&
    "${VENV}/bin/python" - \
      "${output_dir}/eval_config.json" \
      "$EVAL_PROTOCOL_NAME" \
      "$EXPECTED_PROBLEMS" \
      "$EVAL_NUM_SAMPLES" \
      "$EVAL_MAX_PROMPT_LENGTH" \
      "$EVAL_MAX_GENERATION_STEPS" \
      "$EVAL_PROBLEM_BATCH_SIZE" \
      "$EVAL_MAX_NUM_SEQS" \
      "$EVAL_MAX_NUM_BATCHED_TOKENS" \
      "$EVAL_TENSOR_PARALLEL_SIZE" \
      "$EVAL_DATA_PARALLEL_SIZE" \
      "$EVAL_VLLM_SERVER_MODE" \
      "$EVAL_VLLM_ASYNC_SCHEDULING" \
      "$EVAL_VLLM_ENABLE_PREFIX_CACHING" <<'PY'
import json
import sys

path = sys.argv[1]
try:
  with open(path, encoding="utf-8") as handle:
    config = json.load(handle)
except (OSError, json.JSONDecodeError):
  raise SystemExit(1)

expected = {
    "protocol_name": sys.argv[2],
    "num_problems": int(sys.argv[3]),
    "num_samples": int(sys.argv[4]),
    "max_prompt_length": int(sys.argv[5]),
    "max_generation_steps": int(sys.argv[6]),
    "problem_batch_size": int(sys.argv[7]),
    "max_num_seqs": int(sys.argv[8]),
    "max_num_batched_tokens": int(sys.argv[9]),
    "tensor_parallel_size": int(sys.argv[10]),
    "data_parallel_size": int(sys.argv[11]),
    "vllm_server_mode": sys.argv[12] == "true",
    "vllm_async_scheduling": sys.argv[13] == "true",
    "vllm_enable_prefix_caching": sys.argv[14] == "true",
}
# Results created before protocol_name/max_prompt_length were recorded belong
# to the unchanged legacy 8K protocol.
if "protocol_name" not in config:
  config["protocol_name"] = "aime2024_8k_constrained"
if "max_prompt_length" not in config:
  config["max_prompt_length"] = 2048
for legacy_false_key in (
    "vllm_server_mode",
    "vllm_async_scheduling",
    "vllm_enable_prefix_caching",
):
  if legacy_false_key not in config:
    config[legacy_false_key] = False
raise SystemExit(0 if all(config.get(k) == v for k, v in expected.items()) else 1)
PY
}

run_eval() {
  local seed="$1"
  local label="$2"
  local checkpoint_source="$3"
  local checkpoint_step="$4"
  local run_root="$5"
  local output_dir="${SUITE_ROOT}/seed${seed}/${label}"
  local nohup_log="${SUITE_ROOT}/seed${seed}/${label}.launcher.log"
  local optional_args=()

  if [[ -n "$EVAL_LIMIT" ]]; then
    optional_args+=(--limit "$EVAL_LIMIT")
  fi
  if [[ "$EVAL_VLLM_SERVER_MODE" == "true" ]]; then
    optional_args+=(--vllm-server-mode)
  fi
  if [[ "$EVAL_VLLM_ASYNC_SCHEDULING" == "true" ]]; then
    optional_args+=(--vllm-async-scheduling)
  fi
  if [[ "$EVAL_VLLM_ENABLE_PREFIX_CACHING" == "true" ]]; then
    optional_args+=(--vllm-enable-prefix-caching)
  fi

  mkdir -p "${SUITE_ROOT}/seed${seed}"
  if completed_output "$output_dir"; then
    echo "SKIP_COMPLETED seed=${seed} model=${label} output=${output_dir}"
    printf '%s\t%s\tskipped_complete\t0\t%s\n' \
      "$seed" "$label" "$output_dir" >> "$STATUS_FILE"
    return 0
  fi
  if [[ -e "$output_dir" ]]; then
    echo "ERROR: incomplete output already exists: ${output_dir}" >&2
    return 1
  fi
  if [[ "$checkpoint_source" == "checkpoint" && ! -d "${run_root}/checkpoints/actor/${checkpoint_step}" ]]; then
    echo "ERROR: checkpoint directory is missing: ${run_root}/checkpoints/actor/${checkpoint_step}" >&2
    return 1
  fi

  wait_for_tpu_release
  echo "START seed=${seed} model=${label} output=${output_dir}"
  printf '%s\t%s\tstarted\t-\t%s\n' \
    "$seed" "$label" "$output_dir" >> "$STATUS_FILE"

  set +e
  env \
    REPO="$REPO" \
    VENV="$VENV" \
    TPU_NAME="$TPU_NAME" \
    ZONE="$ZONE" \
    EVAL_DATA_PATH="$EVAL_DATA_PATH" \
    MODEL_PATH="$MODEL_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    "${REPO}/runs_xuesong/scripts/run_aime_final_eval.sh" \
      --run-root "$run_root" \
      --output-dir "$output_dir" \
      --protocol-name "$EVAL_PROTOCOL_NAME" \
      --checkpoint-source "$checkpoint_source" \
      --base-model-load-mode "$EVAL_BASE_MODEL_LOAD_MODE" \
      --checkpoint-step "$checkpoint_step" \
      --eval-seed "$seed" \
      --num-samples "$EVAL_NUM_SAMPLES" \
      --max-prompt-length "$EVAL_MAX_PROMPT_LENGTH" \
      --max-generation-steps "$EVAL_MAX_GENERATION_STEPS" \
      --temperature "$EVAL_TEMPERATURE" \
      --top-p "$EVAL_TOP_P" \
      --problem-batch-size "$EVAL_PROBLEM_BATCH_SIZE" \
      --max-num-seqs "$EVAL_MAX_NUM_SEQS" \
      --max-num-batched-tokens "$EVAL_MAX_NUM_BATCHED_TOKENS" \
      --tensor-parallel-size "$EVAL_TENSOR_PARALLEL_SIZE" \
      --data-parallel-size "$EVAL_DATA_PARALLEL_SIZE" \
      --vllm-hbm-utilization "$EVAL_VLLM_HBM_UTILIZATION" \
      "${optional_args[@]}" \
    > "$nohup_log" 2>&1
  local status=$?
  set -e

  if [[ "$status" -eq 0 ]] && completed_output "$output_dir"; then
    echo "COMPLETE seed=${seed} model=${label} output=${output_dir}"
    printf '%s\t%s\tcomplete\t0\t%s\n' \
      "$seed" "$label" "$output_dir" >> "$STATUS_FILE"
  else
    echo "FAILED seed=${seed} model=${label} exit_code=${status}" >&2
    printf '%s\t%s\tfailed\t%s\t%s\n' \
      "$seed" "$label" "$status" "$output_dir" >> "$STATUS_FILE"
    return 1
  fi

  wait_for_tpu_release
}

read -r -a SELECTED_MODELS <<< "$EVAL_MODELS"
if [[ "${#SELECTED_MODELS[@]}" -eq 0 ]]; then
  echo "ERROR: EVAL_MODELS must contain at least one model label." >&2
  exit 2
fi

echo "EVAL_PROTOCOL_NAME=${EVAL_PROTOCOL_NAME}"
echo "EVAL_MODELS=${SELECTED_MODELS[*]}"
echo "EVAL_SEEDS=${SEEDS[*]}"
echo "EVAL_CARDINALITY=${EXPECTED_PROBLEMS}x${EVAL_NUM_SAMPLES}=${EXPECTED_SAMPLES}"
echo "EVAL_LENGTHS=prompt:${EVAL_MAX_PROMPT_LENGTH},generation:${EVAL_MAX_GENERATION_STEPS}"
echo "EVAL_EXECUTION=base_load:${EVAL_BASE_MODEL_LOAD_MODE},tp:${EVAL_TENSOR_PARALLEL_SIZE},dp:${EVAL_DATA_PARALLEL_SIZE},server:${EVAL_VLLM_SERVER_MODE},async:${EVAL_VLLM_ASYNC_SCHEDULING},prefix_cache:${EVAL_VLLM_ENABLE_PREFIX_CACHING}"

for seed in "${SEEDS[@]}"; do
  for model in "${SELECTED_MODELS[@]}"; do
    case "$model" in
      dtv)
        run_eval "$seed" dtv checkpoint "$EVAL_CHECKPOINT_STEP" "$DTV_RUN_ROOT"
        ;;
      dtv_loo)
        run_eval "$seed" dtv_loo checkpoint "$EVAL_CHECKPOINT_STEP" "$LOO_RUN_ROOT"
        ;;
      baseline)
        run_eval "$seed" baseline checkpoint "$EVAL_CHECKPOINT_STEP" "$BASELINE_RUN_ROOT"
        ;;
      pretrain)
        run_eval "$seed" pretrain base_model 0 "$BASE_SOURCE_ROOT"
        ;;
      *)
        echo "ERROR: unsupported EVAL_MODELS entry: $model" >&2
        exit 2
        ;;
    esac
  done
done

echo "EVAL_SEED_SUITE_COMPLETE models=${SELECTED_MODELS[*]} seeds=${SEEDS[*]}"
echo "STATUS_FILE=${STATUS_FILE}"
