#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$#" -lt 1 ]; then
  echo "usage: $0 SEED [NOISE_FRACTION=0] [FILTER_RATIO=0.10] [GRPO arguments...]" >&2
  echo "example clean:    $0 0 0 0.10" >&2
  echo "example mismatch: $0 5 0.2 0.20" >&2
  exit 2
fi

SEED="$1"
FRACTION="${2:-0}"
FILTER_RATIO="${3:-0.10}"
shift "$(( $# >= 3 ? 3 : $# ))"

case "${SEED}" in
  ''|*[!0-9]*)
    echo "error: SEED must be a nonnegative integer; got '${SEED}'" >&2
    exit 2
    ;;
esac

if ! awk -v value="${FRACTION}" \
  'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value >= 0 && value <= 1) }'
then
  echo "error: NOISE_FRACTION must be in [0, 1]; got '${FRACTION}'" >&2
  exit 2
fi

if ! awk -v value="${FILTER_RATIO}" 'BEGIN {
  scaled = value * 100; rounded = int(scaled + 0.5);
  exit !(value ~ /^[0-9]+([.][0-9]+)?$/ &&
         value > 0 && value < 1 &&
         scaled == rounded && rounded % 5 == 0)
}'; then
  echo "error: FILTER_RATIO must be a 5% step in (0,1); got '${FILTER_RATIO}'" >&2
  exit 2
fi

QUEUE_TS="$(date +%Y%m%d_%H%M%S)"
FRACTION_SLUG="${FRACTION//./p}"
QUEUE_NAME="reward_rank_noise_${FRACTION_SLUG}_seed${SEED}_${QUEUE_TS}"
QUEUE_ROOT="${ROOT_DIR}/logs/${QUEUE_NAME}"
QUEUE_LOG="${QUEUE_ROOT}/queue.log"
QUEUE_STATUS="${QUEUE_ROOT}/status.tsv"

mkdir -p "${QUEUE_ROOT}"
exec > >(tee "${QUEUE_LOG}") 2>&1

printf '%s\n' "$$" > "${QUEUE_ROOT}/pid"
printf 'timestamp\tmethod\tseed\tfraction\tstatus\texit_code\n' \
  > "${QUEUE_STATUS}"

if pgrep -af '[p]ython.*my_example' >/dev/null 2>&1; then
  echo "error: another my_example Python task is already running:"
  pgrep -af '[p]ython.*my_example'
  printf '%s\t-\t%s\t%s\tblocked\t3\n' \
    "$(date '+%Y-%m-%dT%H:%M:%S%z')" "${SEED}" "${FRACTION}" \
    >> "${QUEUE_STATUS}"
  printf '3\n' > "${QUEUE_ROOT}/exit_code"
  exit 3
fi

METHODS=(
  "baseline"
  "random_group"
  "reward_group"
  "group_policy"
  "group_loo_policy"
  # "l2"
  # "random_batch"
  # "reward_batch"
  # "batch_policy"
  # "batch_loo_policy"
)

echo "Queue:          ${QUEUE_NAME}"
echo "Queue root:     ${QUEUE_ROOT}"
echo "Seed:           ${SEED}"
echo "Noise fraction: ${FRACTION}"
echo "Noise seed:     ${SEED}"
echo "Filter ratio:   ${FILTER_RATIO}"
echo "Methods:        ${METHODS[*]}"

for method in "${METHODS[@]}"; do
  timestamp="$(date '+%Y-%m-%dT%H:%M:%S%z')"

  echo
  echo "================================================================"
  echo "Starting method=${method} seed=${SEED} noise=${FRACTION}"
  echo "================================================================"
  printf '%s\t%s\t%s\t%s\tstarted\t-\n' \
    "${timestamp}" "${method}" "${SEED}" "${FRACTION}" \
    >> "${QUEUE_STATUS}"

  TUNIX_REWARD_RANK_NOISE_FRACTION="${FRACTION}" \
  TUNIX_REWARD_RANK_NOISE_SEED="${SEED}" \
  TUNIX_FILTER_RATIO="${FILTER_RATIO}" \
    "${ROOT_DIR}/my_example/run_seeded_full.sh" \
      "${method}" "${SEED}" "$@"
  status=$?

  timestamp="$(date '+%Y-%m-%dT%H:%M:%S%z')"
  if [ "${status}" -ne 0 ]; then
    printf '%s\t%s\t%s\t%s\tfailed\t%s\n' \
      "${timestamp}" "${method}" "${SEED}" "${FRACTION}" "${status}" \
      >> "${QUEUE_STATUS}"
    printf '%s\n' "${status}" > "${QUEUE_ROOT}/exit_code"
    echo "Queue stopped after method=${method}, exit=${status}"
    exit "${status}"
  fi

  printf '%s\t%s\t%s\t%s\tcompleted\t0\n' \
    "${timestamp}" "${method}" "${SEED}" "${FRACTION}" \
    >> "${QUEUE_STATUS}"
  echo "Completed method=${method} at ${timestamp}"
done

printf '0\n' > "${QUEUE_ROOT}/exit_code"
echo
echo "All configured runs completed successfully."
