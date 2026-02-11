#!/usr/bin/env bash
set -euo pipefail

# Load shared credentials if present.
ENV_FILE="$(dirname "$0")/../my_example/.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

# Example:
#   ./my_example_qwen_aime/run_grpo_qwen_aime.sh \
#   --train-data-path ./data/deepscaler/deepscaler.json \
#   --test-data-path ./data/aime_2024/train-00000-of-00001.parquet \
#   --num-test-batches 2 \
#   --use-dynamic-batch-curation \
#   --curation-threshold 3.0

RUN_TS="$(date +%Y%m%d_%H%M%S)"
METRICS_LOG_DIR="/tmp/content/tmp/tensorboard/grpo_qwen_aime_${RUN_TS}"
CHECKPOINT_ROOT="/tmp/content/ckpts_qwen_aime_${RUN_TS}"

ARGS=("$@")
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
  TEST_MICRO_BATCH_SIZE="${USER_TEST_MICRO_BATCH_SIZE:-1}"
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

python -m my_example_qwen_aime.main \
  --train-data-path ./data/deepscaler/deepscaler.json \
  --test-data-path ./data/aime_2024/train-00000-of-00001.parquet \
  --train-fraction 0.9 \
  --train-micro-batch-size 1 \
  --test-micro-batch-size 1 \
  --num-generations 2 \
  --max-train-examples 512 \
  --max-eval-examples 64 \
  --num-epochs 1 \
  --learning-rate 1e-6 \
  --no-wandb \
  --metrics-log-dir "${METRICS_LOG_DIR}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  "${EXTRA_EVAL_ARGS[@]}" \
  "${FORWARD_ARGS[@]}"
