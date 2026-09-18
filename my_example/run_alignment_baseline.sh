#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_FILE="$(dirname "$0")/.env"
if [ -f "${ENV_FILE}" ]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

if [ "$#" -lt 1 ]; then
  echo "usage: $0 learnalign|gradalign [alignment/frozen GRPO args...]" >&2
  echo "Independent q defaults: --learnalign-selection-ratio 2; --gradalign-selection-ratio 4" >&2
  exit 2
fi

METHOD="$1"
shift
case "${METHOD}" in
  learnalign|gradalign) ;;
  *)
    echo "error: method must be learnalign or gradalign; got '${METHOD}'" >&2
    exit 2
    ;;
esac

RUN_TS="$(date +%Y%m%d_%H%M%S)"
SEED="${TUNIX_EXPERIMENT_SEED:-legacy}"
NOISE="${TUNIX_REWARD_RANK_NOISE_FRACTION:-0}"
RUN_STEM="${METHOD}_seed${SEED}_mismatch${NOISE}_${RUN_TS}"
RUN_ROOT="${ROOT_DIR}/logs/${RUN_STEM}"
ALIGNMENT_ARTIFACT_DIR="${RUN_ROOT}/selection"
METRICS_LOG_DIR="${RUN_ROOT}/tensorboard"
CHECKPOINT_ROOT="${RUN_ROOT}/checkpoints"
OUTPUT_DIR="${RUN_ROOT}/model"
RESULTS_DIR="${TUNIX_MY_RESULT_DIR:-${RUN_ROOT}/results}"
LABEL="${METHOD}__grpo_${RUN_TS}"
RUN_LOG="${RESULTS_DIR}/${LABEL}__stdout.log"

# Keep the result exporter pointed at the effective TensorBoard directory when
# a caller overrides the default. argparse accepts the last occurrence.
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
      ;;
  esac
  i="$((i + 1))"
done

cd "${ROOT_DIR}"
mkdir -p "${RESULTS_DIR}"

python -m my_example.alignment_main \
  --alignment-method "${METHOD}" \
  --alignment-artifact-dir "${ALIGNMENT_ARTIFACT_DIR}" \
  --source tfds \
  --train-data-dir ./data/train \
  --test-data-dir ./data/test \
  --train-fraction 0.9 \
  --train-micro-batch-size 4 \
  --test-micro-batch-size 1 \
  --num-generations 4 \
  --max-train-examples 3072 \
  --max-eval-examples 1319 \
  --num-epochs 1 \
  --learning-rate 1e-6 \
  --no-wandb \
  --metrics-log-dir "${METRICS_LOG_DIR}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-id google/gemma-3-1b-it \
  --tokenizer-path gs://gemma-data/tokenizers/tokenizer_gemma3.model \
  --mesh-counts 4,1 \
  "$@" 2>&1 | tee "${RUN_LOG}"

python ./my_example/save_results_to_my_result.py \
  --tb-logdir "${METRICS_LOG_DIR}" \
  --label "${LABEL}" \
  --outdir "${RESULTS_DIR}" \
  --stdout-log "${RUN_LOG}" \
  || echo "[warn] failed to export alignment baseline results"

python "./my_example/my result/plot_global_eval_rewards_sum.py" \
  --tag "global/eval/rewards/sum" \
  --outdir "${RESULTS_DIR}" \
  --min-points 1 \
  || echo "[warn] failed to update alignment baseline reward plot"
