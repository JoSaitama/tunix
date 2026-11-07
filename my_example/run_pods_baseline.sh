#!/usr/bin/env bash
# PODS uses the existing frozen GSM8K entrypoint and result exporter.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${ROOT_DIR}/my_example/.env" ]; then
  set -a
  source "${ROOT_DIR}/my_example/.env"
  set +a
fi
KEEP_COUNT="${TUNIX_PODS_KEEP_COUNT:-2}"
DRY_RUN=0
ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --pods-keep-count)
      [ "$#" -ge 2 ] || { echo "missing --pods-keep-count value" >&2; exit 2; }
      KEEP_COUNT="$2"; shift 2 ;;
    --pods-keep-count=*) KEEP_COUNT="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h)
      echo "usage: bash my_example/run_pods_baseline.sh [--pods-keep-count 2] [frozen GRPO args...]"
      exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[[ "$KEEP_COUNT" =~ ^[0-9]+$ ]] && [ "$KEEP_COUNT" -ge 2 ] || {
  echo "PODS keep count must be an integer >= 2" >&2; exit 2;
}
RUN_TS="$(date +%Y%m%d_%H%M%S)"
SEED="${TUNIX_EXPERIMENT_SEED:-legacy}"
NOISE="${TUNIX_REWARD_RANK_NOISE_FRACTION:-0}"
RUN_ROOT="${TUNIX_PODS_RUN_ROOT:-${ROOT_DIR}/logs/pods_m${KEEP_COUNT}_seed${SEED}_mismatch${NOISE}_${RUN_TS}}"
METRICS_LOG_DIR="${RUN_ROOT}/tensorboard"
CHECKPOINT_ROOT="${RUN_ROOT}/checkpoints"
OUTPUT_DIR="${RUN_ROOT}/model"
RESULTS_DIR="${TUNIX_MY_RESULT_DIR:-${RUN_ROOT}/results}"
LABEL="pods__grpo_${RUN_TS}"
RUN_LOG="${RESULTS_DIR}/${LABEL}__stdout.log"
# Track overrides for the exporter. The Python parser uses the last argument.
for ((i=0; i<${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --metrics-log-dir)
      [ "$((i+1))" -lt "${#ARGS[@]}" ] || exit 2
      METRICS_LOG_DIR="${ARGS[$((i+1))]}" ;;
    --metrics-log-dir=*) METRICS_LOG_DIR="${ARGS[$i]#*=}" ;;
  esac
done
COMMAND=(python -m my_example.main
  --source tfds --train-data-dir ./data/train --test-data-dir ./data/test
  --train-fraction 0.9 --train-micro-batch-size 4 --test-micro-batch-size 1
  --num-generations 4 --max-train-examples 3072 --max-eval-examples 1319
  --num-epochs 1 --learning-rate 1e-6 --no-wandb
  --metrics-log-dir "$METRICS_LOG_DIR" --checkpoint-root "$CHECKPOINT_ROOT"
  --output-dir "$OUTPUT_DIR" --model-id google/gemma-3-1b-it
  --tokenizer-path gs://gemma-data/tokenizers/tokenizer_gemma3.model
  --mesh-counts 4,1)
COMMAND+=( ${ARGS[@]+"${ARGS[@]}"} )
if [ "$DRY_RUN" -eq 1 ]; then
  printf '%q ' env "TUNIX_PODS_KEEP_COUNT=${KEEP_COUNT}" \
    "TUNIX_PODS_ARTIFACT_DIR=${RUN_ROOT}/selection" "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
cd "$ROOT_DIR"
mkdir -p "$RESULTS_DIR"
set +e
env TUNIX_PODS_KEEP_COUNT="$KEEP_COUNT" TUNIX_PODS_ARTIFACT_DIR="${RUN_ROOT}/selection" \
  "${COMMAND[@]}" 2>&1 | tee "$RUN_LOG"
PIPE_CODES=("${PIPESTATUS[@]}")
CODE=${PIPE_CODES[0]}
if [ "$CODE" -eq 0 ]; then CODE=${PIPE_CODES[1]}; fi
set -e
printf '%s\n' "$CODE" > "${RUN_ROOT}/exit_code"
python ./my_example/save_results_to_my_result.py \
  --tb-logdir "$METRICS_LOG_DIR" --label "$LABEL" --outdir "$RESULTS_DIR" \
  --stdout-log "$RUN_LOG" || echo "[warn] result export failed" >&2
if [ "$CODE" -eq 0 ]; then
  python "./my_example/my result/plot_global_eval_rewards_sum.py" \
    --tag global/eval/rewards/sum --outdir "$RESULTS_DIR" --min-points 1 \
    || echo "[warn] reward plot export failed" >&2
fi
exit "$CODE"
