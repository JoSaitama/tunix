#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
METRICS_LOG_DIR="/tmp/content/tmp/tensorboard/grpo_${RUN_TS}"
CHECKPOINT_ROOT="/tmp/content/ckpts_run2_${RUN_TS}"

# Run from repo root so run_grpo_gemma.sh relative paths resolve.
cd "${ROOT_DIR}"

./my_example/run_grpo_gemma.sh \
  --use-dbc-outlier-l2 \
  --curation-threshold 3.0 \
  --no-wandb \
  --metrics-log-dir "${METRICS_LOG_DIR}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --train-micro-batch-size 4 \
  "$@"

# Alternative DBC variant:
#   --use-dbc-self-inf-batch
#   --use-dbc-self-inf-group
