#!/usr/bin/env bash
set -euo pipefail

DEFAULT_MODEL_PATH="/home/lhf_hongfu_gmail_com/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
DEFAULT_TRAIN_DATA_PATH="/home/lhf_hongfu_gmail_com/.cache/huggingface/hub/datasets--agentica-org--DeepScaleR-Preview-Dataset/snapshots/b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29/deepscaler.json"
DEFAULT_TEST_DATA_PATH="/home/lhf_hongfu_gmail_com/.cache/huggingface/hub/datasets--HuggingFaceH4--aime_2024/snapshots/2fe88a2f1091d5048c0f36abc874fb997b3dd99a/data/train-00000-of-00001.parquet"

MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$DEFAULT_TRAIN_DATA_PATH}"
TEST_DATA_PATH="${TEST_DATA_PATH:-$DEFAULT_TEST_DATA_PATH}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/tmp/deepscaler_ckpt_${RUN_TS}}"
METRICS_LOG_DIR="${METRICS_LOG_DIR:-/tmp/deepscaler_tb_${RUN_TS}}"
MESH_FSDP="${MESH_FSDP:-2}"
MESH_TP="${MESH_TP:-2}"
GRPO_MAX_CONCURRENCY="${GRPO_MAX_CONCURRENCY:-1}"

# Training defaults for the current DeepScaler full-dataset setup.
ROLLOUT_ENGINE="${ROLLOUT_ENGINE:-sglang_jax}"
ROLLOUT_TP="${ROLLOUT_TP:-2}"
ROLLOUT_PROMPT_BATCH_SIZE="${ROLLOUT_PROMPT_BATCH_SIZE:-4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
TOTAL_GENERATION_STEPS="${TOTAL_GENERATION_STEPS:-8192}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-128}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
NUM_BATCHES="${NUM_BATCHES:-315}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
TRAIN_FRACTION="${TRAIN_FRACTION:-1.0}"
MAX_STEPS="${MAX_STEPS:-$NUM_BATCHES}"
SAVE_INTERVAL_STEPS="${SAVE_INTERVAL_STEPS:-158}"
MAX_TO_KEEP="${MAX_TO_KEEP:-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"

# Dtype knobs:
# - TRAIN_DTYPE options: fp32 | bf16
# - REWARD_ADVANTAGE_DTYPE options: fp32 | bf16
# - ROLLOUT_SGLANG_JAX_DTYPE options: auto | float32 | bfloat16 | float16
#   (also accepts aliases: fp32, bf16, half, float)
# - ROLLOUT_SGLANG_JAX_KV_CACHE_DTYPE options: auto | bf16 | fp8_e5m2 | fp8_e4m3
# - OFFLOAD_TO_CPU options: true | false
TRAIN_DTYPE="${TRAIN_DTYPE:-bf16}"
REWARD_ADVANTAGE_DTYPE="${REWARD_ADVANTAGE_DTYPE:-bf16}"
ROLLOUT_SGLANG_JAX_DTYPE="${ROLLOUT_SGLANG_JAX_DTYPE:-auto}"
ROLLOUT_SGLANG_JAX_KV_CACHE_DTYPE="${ROLLOUT_SGLANG_JAX_KV_CACHE_DTYPE:-auto}"
OFFLOAD_TO_CPU="${OFFLOAD_TO_CPU:-false}"

case "$(printf '%s' "$OFFLOAD_TO_CPU" | tr '[:upper:]' '[:lower:]')" in
  true|1|yes|y)
    OFFLOAD_TO_CPU_FLAG="--offload-to-cpu"
    ;;
  false|0|no|n)
    OFFLOAD_TO_CPU_FLAG="--no-offload-to-cpu"
    ;;
  *)
    echo "Invalid OFFLOAD_TO_CPU: $OFFLOAD_TO_CPU (expected true/false)" >&2
    exit 1
    ;;
esac

for p in "$MODEL_PATH" "$TRAIN_DATA_PATH" "$TEST_DATA_PATH"; do
  if [[ ! -e "$p" ]]; then
    echo "Missing required path: $p" >&2
    echo "Set MODEL_PATH / TRAIN_DATA_PATH / TEST_DATA_PATH to override defaults." >&2
    exit 1
  fi
done

python examples/deepscaler/train_deepscaler_nb.py \
  --model-path "$MODEL_PATH" \
  --train-dataset-path "$TRAIN_DATA_PATH" \
  --test-dataset-path "$TEST_DATA_PATH" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --metrics-log-dir "$METRICS_LOG_DIR" \
  --mesh-fsdp "$MESH_FSDP" \
  --mesh-tp "$MESH_TP" \
  --grpo-max-concurrency "$GRPO_MAX_CONCURRENCY" \
  --train-dtype "$TRAIN_DTYPE" \
  --reward-advantage-dtype "$REWARD_ADVANTAGE_DTYPE" \
  --rollout-sglang-jax-dtype "$ROLLOUT_SGLANG_JAX_DTYPE" \
  --rollout-sglang-jax-kv-cache-dtype "$ROLLOUT_SGLANG_JAX_KV_CACHE_DTYPE" \
  --rollout-engine "$ROLLOUT_ENGINE" \
  --rollout-tp "$ROLLOUT_TP" \
  --enable-rollout-fast-path \
  --rollout-prompt-batch-size "$ROLLOUT_PROMPT_BATCH_SIZE" \
  --num-generations "$NUM_GENERATIONS" \
  --max-prompt-length "$MAX_PROMPT_LENGTH" \
  --total-generation-steps "$TOTAL_GENERATION_STEPS" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --batch-size "$BATCH_SIZE" \
  --mini-batch-size "$MINI_BATCH_SIZE" \
  --train-micro-batch-size "$TRAIN_MICRO_BATCH_SIZE" \
  --num-batches "$NUM_BATCHES" \
  --num-epochs "$NUM_EPOCHS" \
  --train-fraction "$TRAIN_FRACTION" \
  --max-steps "$MAX_STEPS" \
  --weight-decay "$WEIGHT_DECAY" \
  --max-grad-norm "$MAX_GRAD_NORM" \
  --save-interval-steps "$SAVE_INTERVAL_STEPS" \
  --max-to-keep "$MAX_TO_KEEP" \
  "$OFFLOAD_TO_CPU_FLAG" \
  "$@"
