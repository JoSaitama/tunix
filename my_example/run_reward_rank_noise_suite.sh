#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$#" -lt 2 ]; then
  echo "usage: $0 SEED FRACTION [extra run_grpo_gemma.sh arguments...]" >&2
  echo "example: $0 0 0.2" >&2
  exit 2
fi

SEED="$1"
FRACTION="$2"
shift 2

case "${SEED}" in
  ''|*[!0-9]*)
    echo "error: SEED must be a nonnegative integer; got '${SEED}'" >&2
    exit 2
    ;;
esac

if ! awk -v value="${FRACTION}" \
  'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0 && value <= 1) }'
then
  echo "error: FRACTION must be a number in (0, 1]; got '${FRACTION}'" >&2
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
  "l2"
  "group_policy"
  "batch_policy"
  "group_loo_policy"
  "batch_loo_policy"
)

echo "Queue:          ${QUEUE_NAME}"
echo "Queue root:     ${QUEUE_ROOT}"
echo "Seed:           ${SEED}"
echo "Noise fraction: ${FRACTION}"
echo "Noise seed:     ${SEED}"
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
echo "All six reward-rank-noise runs completed successfully."
