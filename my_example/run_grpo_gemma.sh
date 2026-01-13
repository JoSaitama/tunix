#!/usr/bin/env bash
set -euo pipefail

# Load local env file if present.
ENV_FILE="$(dirname "$0")/.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

# Example usage:
#   HF_TOKEN=... KAGGLE_USERNAME=... KAGGLE_KEY=... WANDB_API_KEY=... \
#   HF_TOKEN=... KAGGLE_USERNAME=... KAGGLE_KEY=... WANDB_API_KEY=... \
#   ./my_example/run_grpo_gemma.sh \
#   --use-dynamic-batch-curation \
#   --curation-threshold 3.0

RUN_TS="$(date +%Y%m%d_%H%M%S)"
METRICS_LOG_DIR="${METRICS_LOG_DIR:-/tmp/content/tmp/tensorboard/grpo_${RUN_TS}}"

# Use a shared checkpoint location for multi-host runs. Set CKPT_BUCKET to something
# like gs://your-bucket. You can also override CHECKPOINT_ROOT directly.
if [ -n "${CKPT_BUCKET:-}" ]; then
  CKPT_BUCKET="${CKPT_BUCKET%/}"
  DEFAULT_CHECKPOINT_ROOT="${CKPT_BUCKET}/tunix/ckpts/grpo_${RUN_TS}"
else
  DEFAULT_CHECKPOINT_ROOT="/tmp/content/ckpts_run2_${RUN_TS}"
  echo "WARNING: CKPT_BUCKET not set; defaulting to local ${DEFAULT_CHECKPOINT_ROOT}. Multi-host checkpointing will likely fail." >&2
fi
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${DEFAULT_CHECKPOINT_ROOT}}"

python -m my_example.main \
  --source tfds \
  --train-data-dir ./data/train \
  --test-data-dir ./data/test \
  --train-fraction 0.9 \
  --train-micro-batch-size 16 \
  --test-micro-batch-size 1 \
  --num-generations 4 \
  --max-train-examples 29888 \
  --max-eval-examples 1319 \
  --num-epochs 1 \
  --no-wandb \
  --metrics-log-dir "${METRICS_LOG_DIR}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --save-interval-steps 100000 \
  --model-id google/gemma-3-1b-it \
  --tokenizer-path gs://gemma-data/tokenizers/tokenizer_gemma3.model \
  --mesh-counts 4,1 \
  "$@"
