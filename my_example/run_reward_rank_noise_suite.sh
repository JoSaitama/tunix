#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
usage:
  run_reward_rank_noise_suite.sh \
    --seeds SEED [SEED ...] \
    [--mismatch FRACTION] \
    [--filter RATIO] \
    [--dataset NAME] \
    [--methods METHOD [METHOD ...]] \
    [-- extra run_grpo_gemma.sh arguments...]

examples:
  # 12 clean runs: 3 seeds x 4 Random/Reward methods
  run_reward_rank_noise_suite.sh \
    --seeds 0 5 21 \
    --mismatch 0 \
    --filter 0.20 \
    --methods random_batch random_group reward_batch reward_group

  # Two methods under mismatch40
  run_reward_rank_noise_suite.sh \
    --seeds 0 5 \
    --mismatch 0.4 \
    --filter 0.35 \
    --methods random_group reward_group

legacy positional form remains supported:
  run_reward_rank_noise_suite.sh SEED [NOISE_FRACTION=0] [FILTER_RATIO=0.10]
EOF
}

DEFAULT_METHODS=(
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

VALID_METHODS=(
  "baseline"
  "l2"
  "random_batch"
  "random_group"
  "reward_batch"
  "reward_group"
  "batch_policy"
  "group_policy"
  "batch_loo_policy"
  "group_loo_policy"
)

SEEDS=()
METHODS=()
FRACTION="0"
FILTER_RATIO="0.10"
DATASET="gsm8k"
EXTRA_ARGS=()

if [ "$#" -eq 0 ]; then
  usage >&2
  exit 2
fi

# Preserve the previous positional interface for existing experiment commands.
if [[ "$1" != --* ]]; then
  SEEDS=("$1")
  FRACTION="${2:-0}"
  FILTER_RATIO="${3:-0.10}"
  shift "$(( $# >= 3 ? 3 : $# ))"
  METHODS=("${DEFAULT_METHODS[@]}")
  EXTRA_ARGS=("$@")
else
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --help|-h)
        usage
        exit 0
        ;;
      --seeds|--seed)
        shift
        while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
          SEEDS+=("$1")
          shift
        done
        ;;
      --methods|--method)
        shift
        while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
          METHODS+=("$1")
          shift
        done
        ;;
      --mismatch)
        if [ "$#" -lt 2 ]; then
          echo "error: --mismatch requires a value" >&2
          exit 2
        fi
        FRACTION="$2"
        shift 2
        ;;
      --filter)
        if [ "$#" -lt 2 ]; then
          echo "error: --filter requires a value" >&2
          exit 2
        fi
        FILTER_RATIO="$2"
        shift 2
        ;;
      --dataset)
        if [ "$#" -lt 2 ]; then
          echo "error: --dataset requires a value" >&2
          exit 2
        fi
        DATASET="$2"
        shift 2
        ;;
      --)
        shift
        EXTRA_ARGS=("$@")
        break
        ;;
      *)
        echo "error: unknown argument '$1'" >&2
        usage >&2
        exit 2
        ;;
    esac
  done
fi

if [ "${#SEEDS[@]}" -eq 0 ]; then
  echo "error: at least one seed is required via --seeds" >&2
  exit 2
fi
if [ "${#METHODS[@]}" -eq 0 ]; then
  METHODS=("${DEFAULT_METHODS[@]}")
fi

for seed in "${SEEDS[@]}"; do
  case "${seed}" in
    ''|*[!0-9]*)
      echo "error: seeds must be nonnegative integers; got '${seed}'" >&2
      exit 2
      ;;
  esac
done

if ! awk -v value="${FRACTION}" \
  'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value >= 0 && value <= 1) }'
then
  echo "error: --mismatch must be in [0, 1]; got '${FRACTION}'" >&2
  exit 2
fi

if ! awk -v value="${FILTER_RATIO}" 'BEGIN {
  scaled = value * 100;
  rounded = int(scaled + 0.5);
  exit !(value ~ /^[0-9]+([.][0-9]+)?$/ &&
         value > 0 && value < 1 &&
         scaled == rounded && rounded % 5 == 0)
}'; then
  echo "error: --filter must be a 5% step in (0,1); got '${FILTER_RATIO}'" >&2
  exit 2
fi

case "${DATASET}" in
  ''|*[!A-Za-z0-9_-]*)
    echo "error: --dataset may contain only letters, digits, '_' and '-'" >&2
    exit 2
    ;;
esac

for method in "${METHODS[@]}"; do
  valid=false
  for candidate in "${VALID_METHODS[@]}"; do
    if [ "${method}" = "${candidate}" ]; then
      valid=true
      break
    fi
  done
  if [ "${valid}" != true ]; then
    echo "error: unsupported method '${method}'" >&2
    echo "valid methods: ${VALID_METHODS[*]}" >&2
    exit 2
  fi
done

SUITE_TS="$(date +%Y%m%d_%H%M%S)"
SEED_SLUG="$(IFS=_; echo "${SEEDS[*]}")"
FRACTION_CANONICAL="$(awk -v value="${FRACTION}" 'BEGIN { printf "%g", value }')"
FRACTION_SLUG="${FRACTION_CANONICAL//./p}"
if [ "${FRACTION_CANONICAL}" = "0" ]; then
  DATA_MODE="clean"
else
  DATA_MODE="mismatch${FRACTION_SLUG}"
fi
FILTER_PERCENT="$(awk -v value="${FILTER_RATIO}" 'BEGIN { printf "%02d", value * 100 }')"
FILTER_SLUG="0p${FILTER_PERCENT}"

LOG_DIR="${ROOT_DIR}/logs"
SUITE_STEM="grpo_${DATASET}_seed_${SEED_SLUG}_${DATA_MODE}_${SUITE_TS}"
SUITE_DIR="${LOG_DIR}/${SUITE_STEM}"
SUITE_LOG="${SUITE_DIR}/nohup.log"
SUITE_STATUS="${SUITE_DIR}/status.tsv"
SUITE_PID="${SUITE_DIR}/pid"
SUITE_EXIT_CODE="${SUITE_DIR}/exit_code"

mkdir -p "${SUITE_DIR}"
exec > >(tee "${SUITE_LOG}") 2>&1

printf '%s\n' "$$" > "${SUITE_PID}"
printf 'timestamp\tmethod\tseed\tmismatch\tfilter_ratio\tstatus\texit_code\tlog\n' \
  > "${SUITE_STATUS}"

if pgrep -af '[p]ython.*my_example' >/dev/null 2>&1; then
  echo "error: another my_example Python task is already running:"
  pgrep -af '[p]ython.*my_example'
  printf '%s\n' "3" > "${SUITE_EXIT_CODE}"
  exit 3
fi

TOTAL_RUNS=$(( ${#SEEDS[@]} * ${#METHODS[@]} ))
echo "Suite directory: ${SUITE_DIR}"
echo "Suite log:      ${SUITE_LOG}"
echo "Status:         ${SUITE_STATUS}"
echo "Dataset:        ${DATASET}"
echo "Seeds:          ${SEEDS[*]}"
echo "Data mode:      ${DATA_MODE}"
echo "Mismatch:       ${FRACTION_CANONICAL}"
echo "Filter ratio:   ${FILTER_RATIO}"
echo "Methods:        ${METHODS[*]}"
echo "Total runs:     ${TOTAL_RUNS}"

for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
    RUN_TS="$(date +%Y%m%d_%H%M%S)"
    FILTER_SUFFIX=""
    METHOD_FILTER_RATIO="n/a"
    case "${method}" in
      baseline)
        METHOD_OUTPUT_SLUG="baseline"
        ;;
      l2)
        METHOD_OUTPUT_SLUG="dtv_outlier_l2"
        ;;
      random_batch|random_group|reward_batch|reward_group)
        METHOD_OUTPUT_SLUG="${method}"
        ;;
      *)
        METHOD_OUTPUT_SLUG="dtv_selfinf_${method}"
        ;;
    esac
    case "${method}" in
      random_batch|random_group|reward_batch|reward_group)
        FILTER_SUFFIX="_filter${FILTER_SLUG}"
        METHOD_FILTER_RATIO="${FILTER_RATIO}"
        ;;
    esac
    METHOD_STEM="grpo_${DATASET}_${METHOD_OUTPUT_SLUG}${FILTER_SUFFIX}_seed${seed}_${DATA_MODE}_${RUN_TS}"
    METHOD_LOG="${LOG_DIR}/${METHOD_STEM}/nohup.log"
    timestamp="$(date '+%Y-%m-%dT%H:%M:%S%z')"

    echo
    echo "================================================================"
    echo "Starting method=${method} seed=${seed} mode=${DATA_MODE}"
    echo "Method log: ${METHOD_LOG}"
    echo "================================================================"
    printf '%s\t%s\t%s\t%s\t%s\tstarted\t-\t%s\n' \
      "${timestamp}" "${method}" "${seed}" "${FRACTION_CANONICAL}" \
      "${METHOD_FILTER_RATIO}" "${METHOD_LOG}" >> "${SUITE_STATUS}"

    TUNIX_REWARD_RANK_NOISE_FRACTION="${FRACTION_CANONICAL}" \
    TUNIX_REWARD_RANK_NOISE_SEED="${seed}" \
    TUNIX_FILTER_RATIO="${FILTER_RATIO}" \
    TUNIX_DATASET_NAME="${DATASET}" \
    TUNIX_RUN_TIMESTAMP="${RUN_TS}" \
      "${ROOT_DIR}/my_example/run_seeded_full.sh" \
        "${method}" "${seed}" "${EXTRA_ARGS[@]}"
    status=$?

    timestamp="$(date '+%Y-%m-%dT%H:%M:%S%z')"
    if [ "${status}" -ne 0 ]; then
      printf '%s\t%s\t%s\t%s\t%s\tfailed\t%s\t%s\n' \
        "${timestamp}" "${method}" "${seed}" "${FRACTION_CANONICAL}" \
        "${METHOD_FILTER_RATIO}" "${status}" "${METHOD_LOG}" >> "${SUITE_STATUS}"
      printf '%s\n' "${status}" > "${SUITE_EXIT_CODE}"
      echo "Suite stopped after method=${method}, seed=${seed}, exit=${status}"
      exit "${status}"
    fi

    printf '%s\t%s\t%s\t%s\t%s\tcompleted\t0\t%s\n' \
      "${timestamp}" "${method}" "${seed}" "${FRACTION_CANONICAL}" \
      "${METHOD_FILTER_RATIO}" "${METHOD_LOG}" >> "${SUITE_STATUS}"
    echo "Completed method=${method}, seed=${seed} at ${timestamp}"
  done
done

printf '0\n' > "${SUITE_EXIT_CODE}"
echo
echo "All ${TOTAL_RUNS} configured runs completed successfully."
