#!/usr/bin/env bash
set -euo pipefail

RUN_TS="$(date +%Y%m%d_%H%M%S)"
METRICS_LOG_DIR="/tmp/content/tmp/tensorboard/grpo_${RUN_TS}"
CHECKPOINT_ROOT="/tmp/content/ckpts_run2_${RUN_TS}"

./my_example/run_grpo_gemma.sh \
  --use-dynamic-batch-curation \
  --curation-threshold 3.0 \
  --no-wandb \
  --metrics-log-dir "${METRICS_LOG_DIR}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  "$@"
