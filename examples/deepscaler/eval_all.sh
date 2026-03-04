#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EVAL_SH="$ROOT_DIR/run_eval.sh"

if [[ ! -x "$RUN_EVAL_SH" ]]; then
  echo "Missing executable: $RUN_EVAL_SH" >&2
  exit 1
fi

DATA_PATH="${DATA_PATH:-/home/lhf_hongfu_gmail_com/.cache/huggingface/hub/datasets--HuggingFaceH4--aime_2024/snapshots/2fe88a2f1091d5048c0f36abc874fb997b3dd99a/data/train-00000-of-00001.parquet}"
MESH_FSDP="${MESH_FSDP:-2}"
MESH_TP="${MESH_TP:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_BATCHES="${EVAL_NUM_BATCHES:-30}"
EVAL_MAX_PROMPT_LENGTH="${EVAL_MAX_PROMPT_LENGTH:-2048}"
EVAL_MAX_GENERATION_STEPS="${EVAL_MAX_GENERATION_STEPS:-32768}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.6}"
EVAL_TOP_K="${EVAL_TOP_K:--1}"
EVAL_TOP_P="${EVAL_TOP_P:-0.95}"
EVAL_DEBUG_FIRST_N="${EVAL_DEBUG_FIRST_N:-0}"
LOG_DIR="${LOG_DIR:-/tmp/deepscaler_eval_logs_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -e "$DATA_PATH" ]]; then
  echo "Dataset not found: $DATA_PATH" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

MODELS=(
  "Qwen/Qwen2.5-1.5B-Instruct|/home/lhf_hongfu_gmail_com/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306|qwen2p5_1p5b_instruct"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B|/home/lhf_hongfu_gmail_com/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562|deepseek_r1_distill_qwen_1p5b"
  "agentica-org/DeepScaleR-1.5B-Preview|/home/lhf_hongfu_gmail_com/.cache/huggingface/hub/models--agentica-org--DeepScaleR-1.5B-Preview/snapshots/e3f524ce413a296b4d388e7560dd5c82c1c56725|deepscaler_1p5b_preview"
)

run_one() {
  local model_version="$1"
  local model_path="$2"
  local slug="$3"
  local passes="$4"
  local log_file="$LOG_DIR/eval_${slug}_p${passes}.log"

  if [[ ! -e "$model_path" ]]; then
    echo "Model path not found: $model_path" >&2
    return 1
  fi

  rm -f /tmp/libtpu_lockfile || true
  echo "===== START model=${model_version} num_passes=${passes} ====="

  MODEL_VERSION="$model_version" \
  MODEL_PATH="$model_path" \
  TEST_DATA_PATH="$DATA_PATH" \
  MESH_FSDP="$MESH_FSDP" \
  MESH_TP="$MESH_TP" \
  EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
  EVAL_NUM_BATCHES="$EVAL_NUM_BATCHES" \
  EVAL_MAX_PROMPT_LENGTH="$EVAL_MAX_PROMPT_LENGTH" \
  EVAL_MAX_GENERATION_STEPS="$EVAL_MAX_GENERATION_STEPS" \
  EVAL_TEMPERATURE="$EVAL_TEMPERATURE" \
  EVAL_TOP_K="$EVAL_TOP_K" \
  EVAL_TOP_P="$EVAL_TOP_P" \
  EVAL_DEBUG_FIRST_N="$EVAL_DEBUG_FIRST_N" \
  EVAL_NUM_PASSES="$passes" \
  "$RUN_EVAL_SH" | tee "$log_file"

  echo "===== END model=${model_version} num_passes=${passes} ====="
}

for passes in 1 16; do
  for item in "${MODELS[@]}"; do
    IFS='|' read -r model_version model_path slug <<<"$item"
    run_one "$model_version" "$model_path" "$slug" "$passes"
  done
done

echo
echo "===== FINAL SUMMARY ====="
for passes in 1 16; do
  for item in "${MODELS[@]}"; do
    IFS='|' read -r model_version _ slug <<<"$item"
    log_file="$LOG_DIR/eval_${slug}_p${passes}.log"
    correct_line="$(rg "Correct:" "$log_file" | tail -1 || true)"
    accuracy_line="$(rg "Accuracy:" "$log_file" | tail -1 || true)"
    printf "%-45s num_passes=%-2s | %s | %s\n" "$model_version" "$passes" "$correct_line" "$accuracy_line"
  done
done

echo
echo "Logs saved to: $LOG_DIR"
