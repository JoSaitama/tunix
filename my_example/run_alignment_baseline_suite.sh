#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
usage:
  run_alignment_baseline_suite.sh \
    --seeds 0 5 13 21 42 \
    --mismatch 0|0.2 \
    [--methods learnalign gradalign] \
    [-- extra alignment_main.py arguments...]

The selector starts from binary exact correctness.  Under --mismatch, the same
deterministic prompt-group rank reversal is applied to LearnAlign candidates
and GradAlign candidates; GradAlign's held-out validation direction stays clean.
Actual GRPO updates use the existing dense-reward rank reversal.
EOF
}

SEEDS=()
METHODS=(learnalign gradalign)
MISMATCH=""
EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --seeds)
      shift
      SEEDS=()
      while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
        SEEDS+=("$1")
        shift
      done
      ;;
    --methods)
      shift
      METHODS=()
      while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
        METHODS+=("$1")
        shift
      done
      ;;
    --mismatch)
      MISMATCH="$2"
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

if [ "${#SEEDS[@]}" -eq 0 ] || [ -z "${MISMATCH}" ]; then
  usage >&2
  exit 2
fi
if ! awk -v value="${MISMATCH}" 'BEGIN {
  exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value >= 0 && value <= 1)
}'; then
  echo "error: --mismatch must be in [0,1]" >&2
  exit 2
fi
for method in "${METHODS[@]}"; do
  case "${method}" in
    learnalign|gradalign) ;;
    *) echo "error: unsupported method '${method}'" >&2; exit 2 ;;
  esac
done

SUITE_TS="$(date +%Y%m%d_%H%M%S)"
SUITE_DIR="${ROOT_DIR}/logs/alignment_suite_mismatch${MISMATCH}_${SUITE_TS}"
mkdir -p "${SUITE_DIR}"
STATUS_FILE="${SUITE_DIR}/status.tsv"
printf 'method\tseed\tmismatch\tstatus\texit_code\n' > "${STATUS_FILE}"

for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
    echo "Starting ${method}: seed=${seed}, mismatch=${MISMATCH}"
    printf '%s\t%s\t%s\tstarted\t-\n' \
      "${method}" "${seed}" "${MISMATCH}" >> "${STATUS_FILE}"
    if TUNIX_EXPERIMENT_SEED="${seed}" \
      TUNIX_REWARD_RANK_NOISE_SEED="${seed}" \
      TUNIX_REWARD_RANK_NOISE_FRACTION="${MISMATCH}" \
      "${ROOT_DIR}/my_example/run_alignment_baseline.sh" \
        "${method}" "${EXTRA_ARGS[@]}"; then
      printf '%s\t%s\t%s\tcompleted\t0\n' \
        "${method}" "${seed}" "${MISMATCH}" >> "${STATUS_FILE}"
    else
      code=$?
      printf '%s\t%s\t%s\tfailed\t%s\n' \
        "${method}" "${seed}" "${MISMATCH}" "${code}" >> "${STATUS_FILE}"
      exit "${code}"
    fi
  done
done

echo "All runs completed. Status: ${STATUS_FILE}"
