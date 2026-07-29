#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$#" -lt 3 ]; then
  echo "usage: $0 {random|reward} {batch|group} RATIO [GRPO arguments...]" >&2
  exit 2
fi

METHOD="$1"
SCOPE="$2"
RATIO="$3"
shift 3

case "${METHOD}:${SCOPE}" in
  random:batch|random:group|reward:batch|reward:group) ;;
  *) echo "error: invalid method/scope: ${METHOD}/${SCOPE}" >&2; exit 2 ;;
esac

if ! awk -v value="${RATIO}" 'BEGIN {
  scaled = value * 100;
  rounded = int(scaled + 0.5);
  exit !(value ~ /^[0-9]+([.][0-9]+)?$/ &&
         value > 0 && value < 1 &&
         scaled == rounded && rounded % 5 == 0)
}'; then
  echo "error: RATIO must be in (0,1) and a 5% step, e.g. 0.05, 0.10, 0.35" >&2
  exit 2
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RATIO_SLUG="$(awk -v value="${RATIO}" 'BEGIN { printf "0p%02d", value * 100 }')"
METRICS_LOG_DIR="/tmp/content/tmp/tensorboard/grpo_${RUN_TS}"
CHECKPOINT_ROOT="/tmp/content/ckpts_run2_${RUN_TS}"
RESULTS_DIR="${TUNIX_MY_RESULT_DIR:-${ROOT_DIR}/my_example/my result}"
LABEL="${METHOD}_${SCOPE}_ratio${RATIO_SLUG}__grpo_${RUN_TS}"
RUN_LOG="${RESULTS_DIR}/${LABEL}__stdout.log"
DECISIONS_LOG="${RESULTS_DIR}/${LABEL}__selection.jsonl"

ARGS=("$@")
i=0
while [ "${i}" -lt "${#ARGS[@]}" ]; do
  case "${ARGS[$i]}" in
    --metrics-log-dir) METRICS_LOG_DIR="${ARGS[$((i + 1))]}"; i="$((i + 2))"; continue ;;
    --metrics-log-dir=*) METRICS_LOG_DIR="${ARGS[$i]#*=}" ;;
    --checkpoint-root) CHECKPOINT_ROOT="${ARGS[$((i + 1))]}"; i="$((i + 2))"; continue ;;
    --checkpoint-root=*) CHECKPOINT_ROOT="${ARGS[$i]#*=}" ;;
  esac
  i="$((i + 1))"
done

cd "${ROOT_DIR}"
mkdir -p "${RESULTS_DIR}"

TUNIX_DBC_VARIANT=fixed_filter \
TUNIX_FIXED_FILTER_METHOD="${METHOD}" \
TUNIX_FIXED_FILTER_SCOPE="${SCOPE}" \
TUNIX_FIXED_FILTER_RATIO="${RATIO}" \
TUNIX_FIXED_FILTER_DECISIONS_PATH="${DECISIONS_LOG}" \
TUNIX_GRPO_NUM_GENERATIONS=4 \
./my_example/run_grpo_gemma.sh \
  --use-dynamic-batch-curation \
  --num-generations 4 \
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
  || echo "[warn] failed to export results"
