#!/usr/bin/env bash
# Standalone GRPO+DS; no changes to the original GRPO/PODS launchers.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${ROOT_DIR}/my_example/.env" ]; then
  set -a
  source "${ROOT_DIR}/my_example/.env"
  set +a
fi
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
export TPU_VISIBLE_CHIPS=0,1,2,3
export CLOUD_TPU_TASK_ID=0
CAP="${TUNIX_DS_MAX_GENERATION_BATCHES:-20}"
DRY_RUN=0
ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --ds-max-generation-batches)
      [ "$#" -ge 2 ] || { echo "missing DS cap" >&2; exit 2; }
      CAP="$2"; shift 2 ;;
    --ds-max-generation-batches=*) CAP="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h)
      echo "usage: bash my_example/run_ds_baseline.sh [--ds-max-generation-batches 20] [--dry-run] [frozen GRPO args...]"
      exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[[ "$CAP" =~ ^[0-9]+$ ]] && [ "$CAP" -ge 1 ] || { echo "DS cap must be a positive integer" >&2; exit 2; }
if [ -n "${TUNIX_PODS_KEEP_COUNT:-}" ]; then
  echo "Clear TUNIX_PODS_KEEP_COUNT: DS cannot be combined with PODS" >&2; exit 2
fi
RUN_TS="$(date +%Y%m%d_%H%M%S)"
SEED="${TUNIX_EXPERIMENT_SEED:-legacy}"
NOISE="${TUNIX_REWARD_RANK_NOISE_FRACTION:-0}"
RUN_ROOT="${TUNIX_DS_RUN_ROOT:-${ROOT_DIR}/logs/rebuttal_grpo_gsm8k_ds_seed${SEED}_mismatch${NOISE}_${RUN_TS}}"
RESULTS_DIR="${TUNIX_MY_RESULT_DIR:-${RUN_ROOT}/results}"
METRICS_LOG_DIR="${RUN_ROOT}/tensorboard"
LABEL="ds__grpo_${RUN_TS}"
RUN_LOG="${RESULTS_DIR}/${LABEL}__stdout.log"
for ((i=0; i<${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --metrics-log-dir) [ "$((i+1))" -lt "${#ARGS[@]}" ] || exit 2
      METRICS_LOG_DIR="${ARGS[$((i+1))]}" ;;
    --metrics-log-dir=*) METRICS_LOG_DIR="${ARGS[$i]#*=}" ;;
  esac
done
COMMAND=(python -m my_example.main
  --source tfds --train-data-dir ./data/train --test-data-dir ./data/test
  --train-fraction 0.9 --train-micro-batch-size 4 --test-micro-batch-size 1
  --num-generations 4 --max-train-examples 3072 --max-eval-examples 1319
  --num-epochs 1 --learning-rate 1e-6 --no-wandb
  --metrics-log-dir "$METRICS_LOG_DIR" --checkpoint-root "${RUN_ROOT}/checkpoints"
  --output-dir "${RUN_ROOT}/model" --model-id google/gemma-3-1b-it
  --tokenizer-path gs://gemma-data/tokenizers/tokenizer_gemma3.model --mesh-counts 4,1)
COMMAND+=( ${ARGS[@]+"${ARGS[@]}"} )
if [ "$DRY_RUN" -eq 1 ]; then
  printf '%q ' env TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
    TPU_VISIBLE_CHIPS=0,1,2,3 CLOUD_TPU_TASK_ID=0 TUNIX_DYNAMIC_SAMPLING=1 \
    "TUNIX_DS_MAX_GENERATION_BATCHES=$CAP" "TUNIX_DS_ARTIFACT_DIR=${RUN_ROOT}/selection" "${COMMAND[@]}"
  printf '\n'; exit 0
fi
cd "$ROOT_DIR"
mkdir -p "$RESULTS_DIR"
set +e
env TUNIX_DYNAMIC_SAMPLING=1 "TUNIX_DS_MAX_GENERATION_BATCHES=$CAP" \
  "TUNIX_DS_ARTIFACT_DIR=${RUN_ROOT}/selection" "${COMMAND[@]}" 2>&1 | tee "$RUN_LOG"
PIPE_CODES=("${PIPESTATUS[@]}")
CODE=${PIPE_CODES[0]}
if [ "$CODE" -eq 0 ]; then CODE=${PIPE_CODES[1]}; fi
set -e
printf '%s\n' "$CODE" > "${RUN_ROOT}/exit_code"
python ./my_example/save_results_to_my_result.py \
  --tb-logdir "$METRICS_LOG_DIR" --label "$LABEL" --outdir "$RESULTS_DIR" \
  --stdout-log "$RUN_LOG" || echo "[warn] result export failed" >&2
exit "$CODE"
