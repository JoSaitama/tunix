#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EVAL_SH="$ROOT_DIR/run_eval.sh"

if [[ ! -x "$RUN_EVAL_SH" ]]; then
  echo "Missing executable: $RUN_EVAL_SH" >&2
  exit 1
fi

if command -v rg >/dev/null 2>&1; then
  FIND_LAST_LINE_CMD=(rg)
else
  FIND_LAST_LINE_CMD=(grep)
fi

NUM_RUNS="${NUM_RUNS:-16}"
LOG_DIR="${LOG_DIR:-/tmp/deepscaler_pass1_avg16_$(date +%Y%m%d_%H%M%S)}"
SAMPLER_TYPE="sglang-jax"

mkdir -p "$LOG_DIR"

sum_accuracy="0"
sum_correct="0"
sum_total="0"
completed_runs=0

for run_idx in $(seq 1 "$NUM_RUNS"); do
  log_file="$LOG_DIR/run_${run_idx}.log"
  run_seed=$((run_idx - 1))

  echo "===== START run=${run_idx}/${NUM_RUNS} num_passes=1 sampler=${SAMPLER_TYPE} seed=${run_seed} ====="

  rm -f /tmp/libtpu_lockfile || true

  EVAL_NUM_PASSES=1 \
  EVAL_SEED="$run_seed" \
  "$RUN_EVAL_SH" \
    --sampler-type "$SAMPLER_TYPE" \
    "$@" | tee "$log_file"

  correct_line="$("${FIND_LAST_LINE_CMD[@]}" "Correct:" "$log_file" | tail -1 || true)"
  accuracy_line="$("${FIND_LAST_LINE_CMD[@]}" "Accuracy:" "$log_file" | tail -1 || true)"

  if [[ -z "$correct_line" || -z "$accuracy_line" ]]; then
    echo "Failed to parse metrics from $log_file" >&2
    exit 1
  fi

  correct_value="$(sed -E 's/.*Correct: ([0-9]+)\/([0-9]+).*/\1/' <<<"$correct_line")"
  total_value="$(sed -E 's/.*Correct: ([0-9]+)\/([0-9]+).*/\2/' <<<"$correct_line")"
  accuracy_value="$(sed -E 's/.*Accuracy: ([0-9.]+)%.*/\1/' <<<"$accuracy_line")"

  sum_correct=$((sum_correct + correct_value))
  sum_total=$((sum_total + total_value))
  sum_accuracy="$(awk -v a="$sum_accuracy" -v b="$accuracy_value" 'BEGIN { printf "%.10f", a + b }')"
  completed_runs=$((completed_runs + 1))

  echo "===== END run=${run_idx}/${NUM_RUNS} sampler=${SAMPLER_TYPE} seed=${run_seed} | Correct: ${correct_value}/${total_value} | Accuracy: ${accuracy_value}% ====="
done

avg_accuracy="$(awk -v total="$sum_accuracy" -v runs="$completed_runs" 'BEGIN { printf "%.4f", total / runs }')"
avg_correct="$(awk -v total="$sum_correct" -v runs="$completed_runs" 'BEGIN { printf "%.4f", total / runs }')"
avg_total="$(awk -v total="$sum_total" -v runs="$completed_runs" 'BEGIN { printf "%.4f", total / runs }')"

echo
echo "===== FINAL SUMMARY ====="
echo "Runs: $completed_runs"
echo "Sampler: $SAMPLER_TYPE"
echo "Seeds: 0..$((completed_runs - 1))"
echo "Metric: Pass@1 averaged over ${completed_runs} independent runs"
echo "Average Correct: ${avg_correct}/${avg_total}"
echo "Average Accuracy: ${avg_accuracy}%"
echo "Logs saved to: $LOG_DIR"
