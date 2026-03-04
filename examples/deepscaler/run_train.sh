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
  "$@"
