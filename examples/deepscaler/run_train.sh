#!/usr/bin/env bash
set -euo pipefail

DEFAULT_MODEL_PATH="/home/lhf_hongfu_gmail_com/.cache/huggingface/hub/models--agentica-org--DeepScaleR-1.5B-Preview/snapshots/e3f524ce413a296b4d388e7560dd5c82c1c56725"
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

# Dtype knobs:
# - TRAIN_DTYPE options: fp32 | bf16
# - REWARD_ADVANTAGE_DTYPE options: fp32 | bf16
# - ROLLOUT_SGLANG_JAX_DTYPE options: auto | float32 | bfloat16 | float16
#   (also accepts aliases: fp32, bf16, half, float)
# - ROLLOUT_SGLANG_JAX_KV_CACHE_DTYPE options: auto | bf16 | fp8_e5m2 | fp8_e4m3
TRAIN_DTYPE="${TRAIN_DTYPE:-bf16}"
REWARD_ADVANTAGE_DTYPE="${REWARD_ADVANTAGE_DTYPE:-bf16}"
ROLLOUT_SGLANG_JAX_DTYPE="${ROLLOUT_SGLANG_JAX_DTYPE:-auto}"
ROLLOUT_SGLANG_JAX_KV_CACHE_DTYPE="${ROLLOUT_SGLANG_JAX_KV_CACHE_DTYPE:-auto}"

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
  "$@"
