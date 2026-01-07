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
METRICS_LOG_DIR="/tmp/content/tmp/tensorboard/grpo_${RUN_TS}"
CHECKPOINT_ROOT="/tmp/content/ckpts_run2_${RUN_TS}"

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
  --model-id google/gemma-3-1b-it \
  --tokenizer-path gs://gemma-data/tokenizers/tokenizer_gemma3.model \
  --mesh-counts 4,1 \
  "$@"
