#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Usage: run_dtv_flip.sh [dtv|dtv-loo] [batch|group] [existing GRPO flags...]
SCORE="${1:-dtv}"
if [ "$#" -gt 0 ]; then shift; fi
SCOPE="${1:-batch}"
if [ "$#" -gt 0 ]; then shift; fi
case "$SCORE" in
  dtv) LOO=0 ;;
  dtv-loo) LOO=1 ;;
  *) echo "error: score must be dtv or dtv-loo" >&2; exit 2 ;;
esac
case "$SCOPE" in
  batch|group) ;;
  *) echo "error: scope must be batch or group" >&2; exit 2 ;;
esac
# Keep this experiment's method unambiguous; other GRPO flags are forwarded.
for arg in "$@"; do
  case "$arg" in
    --use-dbc-*|--use-dynamic-batch-curation*)
      echo "error: dtv-flip already selects the curation method" >&2; exit 2 ;;
  esac
done

MAX_FLIP_FRACTION="${TUNIX_DTV_FLIP_MAX_FRACTION:-0.1}"
if ! [[ "$MAX_FLIP_FRACTION" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]]; then
  echo "error: TUNIX_DTV_FLIP_MAX_FRACTION must be a decimal in [0, 1]" >&2
  exit 2
fi
CAP_SLUG="${MAX_FLIP_FRACTION//./p}"

RUN_TS="$(date +%Y%m%d_%H%M%S)_$$"
METRICS_LOG_DIR="/tmp/content/tmp/tensorboard/dtv-flip_${SCORE}_${SCOPE}_cap${CAP_SLUG}_${RUN_TS}"
CHECKPOINT_ROOT="/tmp/content/ckpts_dtv-flip_${SCORE}_${SCOPE}_cap${CAP_SLUG}_${RUN_TS}"
RESULTS_DIR="${TUNIX_MY_RESULT_DIR:-${ROOT_DIR}/my_example/my result}"
LABEL="dtv-flip_${SCORE}_${SCOPE}_cap${CAP_SLUG}_${RUN_TS}"
RUN_LOG="${RESULTS_DIR}/${LABEL}__stdout.log"

# Respect any explicit overrides passed via "$@" so export reads the correct
# logdir/root (argparse uses the last occurrence).
ARGS=("$@")
i=0
while [ "${i}" -lt "${#ARGS[@]}" ]; do
  case "${ARGS[$i]}" in
    --metrics-log-dir)
      if [ "$((i + 1))" -ge "${#ARGS[@]}" ]; then
        echo "error: --metrics-log-dir requires a value" >&2
        exit 2
      fi
      METRICS_LOG_DIR="${ARGS[$((i + 1))]}"
      i="$((i + 2))"
      continue
      ;;
    --metrics-log-dir=*)
      METRICS_LOG_DIR="${ARGS[$i]#*=}"
      i="$((i + 1))"
      continue
      ;;
    --checkpoint-root)
      if [ "$((i + 1))" -ge "${#ARGS[@]}" ]; then
        echo "error: --checkpoint-root requires a value" >&2
        exit 2
      fi
      CHECKPOINT_ROOT="${ARGS[$((i + 1))]}"
      i="$((i + 2))"
      continue
      ;;
    --checkpoint-root=*)
      CHECKPOINT_ROOT="${ARGS[$i]#*=}"
      i="$((i + 1))"
      continue
      ;;
  esac
  i="$((i + 1))"
done

# Run from repo root so run_grpo_gemma.sh relative paths resolve.
cd "${ROOT_DIR}"

mkdir -p "${RESULTS_DIR}"

TUNIX_DTV_FLIP_MAX_FRACTION="$MAX_FLIP_FRACTION" \
TUNIX_DBC_SELF_INF_FLIP=1 \
TUNIX_DBC_SELF_INF_LOO="$LOO" \
TUNIX_DTV_FLIP_DECISIONS_PATH="${RESULTS_DIR}/${LABEL}__decisions.jsonl" \
./my_example/run_grpo_gemma.sh \
  "--use-dbc-self-inf-${SCOPE}" \
  --no-wandb \
  --metrics-log-dir "${METRICS_LOG_DIR}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --train-micro-batch-size 4 \
  "$@" 2>&1 | tee "${RUN_LOG}"

python ./my_example/save_results_to_my_result.py \
  --tb-logdir "${METRICS_LOG_DIR}" \
  --label "${LABEL}" \
  --outdir "${RESULTS_DIR}" \
  --stdout-log "${RUN_LOG}" \
  || echo "[warn] failed to export results to my_example/my result"

python "./my_example/my result/plot_global_eval_rewards_sum.py" \
  --tag "global/eval/rewards/sum" \
  --outdir "${RESULTS_DIR}" \
  --min-points 1 \
  || echo "[warn] failed to update overlay plot in my_example/my result"
