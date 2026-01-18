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
#   --use-dbc-outlier-l2 \
#   --curation-threshold 3.0

RUN_TS="$(date +%Y%m%d_%H%M%S)"
METRICS_LOG_DIR="/tmp/content/tmp/tensorboard/grpo_${RUN_TS}"
CHECKPOINT_ROOT="/tmp/content/ckpts_run2_${RUN_TS}"

ARGS=("$@")

# Convenience flag: allow callers to specify evaluation size in batches without
# touching the python CLI. This keeps backward compatibility with older runs.
NUM_TEST_BATCHES=""
USER_MAX_EVAL_EXAMPLES=""
USER_TEST_MICRO_BATCH_SIZE=""
FORWARD_ARGS=()

i=0
while [ "${i}" -lt "${#ARGS[@]}" ]; do
  case "${ARGS[$i]}" in
    --num-test-batches)
      if [ "$((i + 1))" -ge "${#ARGS[@]}" ]; then
        echo "error: --num-test-batches requires a value" >&2
        exit 2
      fi
      NUM_TEST_BATCHES="${ARGS[$((i + 1))]}"
      i="$((i + 2))"
      continue
      ;;
    --num-test-batches=*)
      NUM_TEST_BATCHES="${ARGS[$i]#*=}"
      i="$((i + 1))"
      continue
      ;;
    --test-micro-batch-size)
      if [ "$((i + 1))" -ge "${#ARGS[@]}" ]; then
        echo "error: --test-micro-batch-size requires a value" >&2
        exit 2
      fi
      USER_TEST_MICRO_BATCH_SIZE="${ARGS[$((i + 1))]}"
      FORWARD_ARGS+=("${ARGS[$i]}" "${ARGS[$((i + 1))]}")
      i="$((i + 2))"
      continue
      ;;
    --test-micro-batch-size=*)
      USER_TEST_MICRO_BATCH_SIZE="${ARGS[$i]#*=}"
      FORWARD_ARGS+=("${ARGS[$i]}")
      i="$((i + 1))"
      continue
      ;;
    --max-eval-examples)
      if [ "$((i + 1))" -ge "${#ARGS[@]}" ]; then
        echo "error: --max-eval-examples requires a value" >&2
        exit 2
      fi
      USER_MAX_EVAL_EXAMPLES="${ARGS[$((i + 1))]}"
      FORWARD_ARGS+=("${ARGS[$i]}" "${ARGS[$((i + 1))]}")
      i="$((i + 2))"
      continue
      ;;
    --max-eval-examples=*)
      USER_MAX_EVAL_EXAMPLES="${ARGS[$i]#*=}"
      FORWARD_ARGS+=("${ARGS[$i]}")
      i="$((i + 1))"
      continue
      ;;
    *)
      FORWARD_ARGS+=("${ARGS[$i]}")
      i="$((i + 1))"
      continue
      ;;
  esac
done

EXTRA_EVAL_ARGS=()
if [ -n "${NUM_TEST_BATCHES}" ] && [ -z "${USER_MAX_EVAL_EXAMPLES}" ]; then
  TEST_MICRO_BATCH_SIZE="${USER_TEST_MICRO_BATCH_SIZE:-32}"
  if ! [[ "${NUM_TEST_BATCHES}" =~ ^[0-9]+$ ]]; then
    echo "error: --num-test-batches must be an integer; got: ${NUM_TEST_BATCHES}" >&2
    exit 2
  fi
  if ! [[ "${TEST_MICRO_BATCH_SIZE}" =~ ^[0-9]+$ ]]; then
    echo "error: --test-micro-batch-size must be an integer; got: ${TEST_MICRO_BATCH_SIZE}" >&2
    exit 2
  fi
  EXTRA_EVAL_ARGS=(--max-eval-examples "$((NUM_TEST_BATCHES * TEST_MICRO_BATCH_SIZE))")
fi

python -m my_example.main \
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
  --model-id google/gemma-3-1b-it \
  --tokenizer-path gs://gemma-data/tokenizers/tokenizer_gemma3.model \
  --mesh-counts 4,1 \
  "${EXTRA_EVAL_ARGS[@]}" \
  "${FORWARD_ARGS[@]}"
