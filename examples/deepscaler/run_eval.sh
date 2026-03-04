#!/usr/bin/env bash
set -euo pipefail

DEFAULT_MODEL_PATH="/home/lhf_hongfu_gmail_com/.cache/huggingface/hub/models--agentica-org--DeepScaleR-1.5B-Preview/snapshots/e3f524ce413a296b4d388e7560dd5c82c1c56725"
DEFAULT_TEST_DATA_PATH="/home/lhf_hongfu_gmail_com/.cache/huggingface/hub/datasets--HuggingFaceH4--aime_2024/snapshots/2fe88a2f1091d5048c0f36abc874fb997b3dd99a/data/train-00000-of-00001.parquet"

MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
TEST_DATA_PATH="${TEST_DATA_PATH:-$DEFAULT_TEST_DATA_PATH}"
MODEL_VERSION="${MODEL_VERSION:-agentica-org/DeepScaleR-1.5B-Preview}"
MESH_FSDP="${MESH_FSDP:-2}"
MESH_TP="${MESH_TP:-2}"
SMOKE_TEST="${SMOKE_TEST:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_BATCHES="${EVAL_NUM_BATCHES:-30}"
EVAL_MAX_PROMPT_LENGTH="${EVAL_MAX_PROMPT_LENGTH:-2048}"
EVAL_MAX_GENERATION_STEPS="${EVAL_MAX_GENERATION_STEPS:-32768}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.6}"
EVAL_TOP_K="${EVAL_TOP_K:--1}"
EVAL_TOP_P="${EVAL_TOP_P:-0.95}"
EVAL_DEBUG_FIRST_N="${EVAL_DEBUG_FIRST_N:-0}"
EVAL_NUM_PASSES="${EVAL_NUM_PASSES:-16}"

for p in "$MODEL_PATH" "$TEST_DATA_PATH"; do
  if [[ ! -e "$p" ]]; then
    echo "Missing required path: $p" >&2
    echo "Set MODEL_PATH / TEST_DATA_PATH to override defaults." >&2
    exit 1
  fi
done

cmd=(
  python examples/deepscaler/math_eval_nb.py
  --model-version "$MODEL_VERSION"
  --model-path "$MODEL_PATH"
  --dataset-path "$TEST_DATA_PATH"
  --mesh-fsdp "$MESH_FSDP"
  --mesh-tp "$MESH_TP"
  --batch-size "$EVAL_BATCH_SIZE"
  --num-batches "$EVAL_NUM_BATCHES"
  --max-prompt-length "$EVAL_MAX_PROMPT_LENGTH"
  --max-generation-steps "$EVAL_MAX_GENERATION_STEPS"
  --temperature "$EVAL_TEMPERATURE"
  --top-k "$EVAL_TOP_K"
  --top-p "$EVAL_TOP_P"
  --num-passes "$EVAL_NUM_PASSES"
  --debug-first-n "$EVAL_DEBUG_FIRST_N"
)

if [[ "$SMOKE_TEST" == "1" ]]; then
  cmd+=(--smoke-test)
fi

cmd+=("$@")
"${cmd[@]}"
