#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="dbc_sweep_${RUN_TS}"
RESULTS_DIR="${ROOT_DIR}/my_example/my result/sweeps/${SWEEP_ID}"

mkdir -p "${RESULTS_DIR}"

cd "${ROOT_DIR}"

for arg in "$@"; do
  case "${arg}" in
    --metrics-log-dir|--metrics-log-dir=*|--checkpoint-root|--checkpoint-root=*)
      echo "error: do not pass --metrics-log-dir/--checkpoint-root to this sweep script." >&2
      echo "it auto-creates per-run log dirs and checkpoint roots." >&2
      exit 2
      ;;
  esac
done

echo "Sweep: ${SWEEP_ID}"
echo "Results: ${RESULTS_DIR}"

N=5
export TUNIX_MY_RESULT_DIR="${RESULTS_DIR}"

for i in $(seq 1 "${N}"); do
  echo
  echo "===== baseline (${i}/${N}) ====="
  ./my_example/run_baseline.sh "$@"
done

for i in $(seq 1 "${N}"); do
  echo
  echo "===== selfinf-batch (${i}/${N}) ====="
  ./my_example/run_dbc_self_inf_batch.sh "$@"
done

for i in $(seq 1 "${N}"); do
  echo
  echo "===== selfinf-group (${i}/${N}) ====="
  ./my_example/run_dbc_self_inf_group.sh "$@"
done

for i in $(seq 1 "${N}"); do
  echo
  echo "===== outlier-l2 (${i}/${N}) ====="
  ./my_example/run_dbc_outlier_l2.sh "$@"
done

echo
echo "Sweep complete."
echo "To analyze:"
echo "  python ./my_example/analyze_my_result.py --results-dir \"${RESULTS_DIR}\" --output-prefix \"${SWEEP_ID}\" --tag \"global/eval/rewards/sum\""
