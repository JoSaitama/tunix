#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
METRICS_LOG_DIR="/tmp/content/tmp/tensorboard/grpo_${RUN_TS}"
CHECKPOINT_ROOT="/tmp/content/ckpts_run2_${RUN_TS}"
RESULTS_DIR="${TUNIX_MY_RESULT_DIR:-${ROOT_DIR}/my_example/my result}"
LABEL="selfinf-group_loo_policy__grpo_${RUN_TS}"
RUN_LOG="${RESULTS_DIR}/${LABEL}__stdout.log"
DECISIONS_LOG="${RESULTS_DIR}/${LABEL}__selection.jsonl"

# Respect explicit overrides so result export reads the effective locations.
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

cd "${ROOT_DIR}"
mkdir -p "${RESULTS_DIR}"

TUNIX_DBC_SELF_INF_LOO=1 \
TUNIX_DBC_SELF_INF_LOO_POLICY=1 \
TUNIX_DBC_SELF_INF_DECISIONS_PATH="${DECISIONS_LOG}" \
./my_example/run_grpo_gemma.sh \
  --use-dbc-self-inf-group \
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
