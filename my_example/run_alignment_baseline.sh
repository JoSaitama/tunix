#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="$(dirname "$0")/.env"
if [ -f "${ENV_FILE}" ]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

if [ "$#" -lt 1 ]; then
  echo "usage: $0 learnalign|gradalign [alignment/frozen GRPO args...]" >&2
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

python -m my_example.alignment_main \
  --alignment-method "${METHOD}" \
  --alignment-artifact-dir "${PWD}/logs/${RUN_STEM}/selection" \
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
  --metrics-log-dir "${PWD}/logs/${RUN_STEM}/tensorboard" \
  --checkpoint-root "${PWD}/logs/${RUN_STEM}/checkpoints" \
  --output-dir "${PWD}/logs/${RUN_STEM}/model" \
  --model-id google/gemma-3-1b-it \
  --tokenizer-path gs://gemma-data/tokenizers/tokenizer_gemma3.model \
  --mesh-counts 4,1 \
  "$@"
