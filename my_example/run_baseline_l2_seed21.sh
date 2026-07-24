#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE_TS="$(date +%Y%m%d_%H%M%S)"
QUEUE_NAME="baseline_l2_seed21_${QUEUE_TS}"
QUEUE_ROOT="${ROOT_DIR}/logs/${QUEUE_NAME}"
QUEUE_LOG="${QUEUE_ROOT}/queue.log"
QUEUE_STATUS="${QUEUE_ROOT}/status.tsv"

mkdir -p "${QUEUE_ROOT}"
exec > >(tee "${QUEUE_LOG}") 2>&1

printf '%s\n' "$$" > "${QUEUE_ROOT}/pid"
printf 'timestamp\tmethod\tseed\tstatus\texit_code\n' > "${QUEUE_STATUS}"

if pgrep -af '[p]ython.*my_example' >/dev/null 2>&1; then
  echo "error: another my_example Python task is already running:"
  pgrep -af '[p]ython.*my_example'
  printf '%s\t-\t-\tblocked\t3\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" \
    >> "${QUEUE_STATUS}"
  printf '3\n' > "${QUEUE_ROOT}/exit_code"
  exit 3
fi

METHODS=("baseline" "l2")
SEEDS=(21 21)

echo "Queue: ${QUEUE_NAME}"
echo "Queue root: ${QUEUE_ROOT}"
echo "Runs: baseline/21, l2/21"

for index in "${!METHODS[@]}"; do
  method="${METHODS[$index]}"
  seed="${SEEDS[$index]}"
  timestamp="$(date '+%Y-%m-%dT%H:%M:%S%z')"

  echo
  echo "================================================================"
  echo "Starting method=${method} seed=${seed} at ${timestamp}"
  echo "================================================================"
  printf '%s\t%s\t%s\tstarted\t-\n' \
    "${timestamp}" "${method}" "${seed}" >> "${QUEUE_STATUS}"

  "${ROOT_DIR}/my_example/run_seeded_full.sh" "${method}" "${seed}"
  status=$?

  if [ "${status}" -ne 0 ]; then
    timestamp="$(date '+%Y-%m-%dT%H:%M:%S%z')"
    printf '%s\t%s\t%s\tfailed\t%s\n' \
      "${timestamp}" "${method}" "${seed}" "${status}" >> "${QUEUE_STATUS}"
    printf '%s\n' "${status}" > "${QUEUE_ROOT}/exit_code"
    echo "Queue stopped after method=${method} seed=${seed}, exit=${status}"
    exit "${status}"
  fi

  timestamp="$(date '+%Y-%m-%dT%H:%M:%S%z')"
  printf '%s\t%s\t%s\tcompleted\t0\n' \
    "${timestamp}" "${method}" "${seed}" >> "${QUEUE_STATUS}"
  echo "Completed method=${method} seed=${seed} at ${timestamp}"
done

printf '0\n' > "${QUEUE_ROOT}/exit_code"
echo
echo "Baseline and L2 seed 21 runs completed successfully."
