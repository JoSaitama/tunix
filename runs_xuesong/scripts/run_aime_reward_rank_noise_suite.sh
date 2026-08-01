#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-/home/jason_chia925_gmail_com/Project/tunix}"
ALL_METHODS=(baseline batch_policy group_policy batch_loo_policy group_loo_policy random_batch random_group reward_batch reward_group)
SEEDS=(); METHODS=(); MISMATCH=0; FILTER=0.10; EXTRA=()
usage() {
  echo "usage: $0 --seeds SEED... [--mismatch FRACTION] [--filter RATIO] [--methods METHOD...] [-- overrides...]"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds|--seed) shift; while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done ;;
    --methods|--method) shift; while [[ $# -gt 0 && "$1" != --* ]]; do METHODS+=("$1"); shift; done ;;
    --mismatch) MISMATCH="$2"; shift 2 ;;
    --filter) FILTER="$2"; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ ${#SEEDS[@]} -gt 0 ]] || { echo "--seeds is required" >&2; exit 2; }
[[ ${#METHODS[@]} -gt 0 ]] || METHODS=("${ALL_METHODS[@]}")
for seed in "${SEEDS[@]}"; do [[ "$seed" =~ ^[0-9]+$ ]] || exit 2; done
awk -v n="$MISMATCH" 'BEGIN {
  exit !(n ~ /^[0-9]+([.][0-9]+)?$/ && n >= 0 && n <= 1)
}' || { echo "--mismatch must be in [0,1]" >&2; exit 2; }
awk -v n="$FILTER" 'BEGIN {
  exit !(n ~ /^[0-9]+([.][0-9]+)?$/ && n > 0 && n < 1)
}' || { echo "--filter must be in (0,1)" >&2; exit 2; }
for method in "${METHODS[@]}"; do
  [[ " ${ALL_METHODS[*]} " == *" ${method} "* ]] || { echo "unsupported method: $method" >&2; exit 2; }
done

SUITE_TS="$(date -u +%Y%m%d_%H%M%S)"
SUITE_DIR="${REPO}/runs_xuesong/logs/suites/grpo_aime_suite_${SUITE_TS}"
mkdir -p "$SUITE_DIR"
exec > >(tee "${SUITE_DIR}/nohup.log") 2>&1
printf '%s\n' "$$" > "${SUITE_DIR}/pid"
printf 'method\tseed\tstatus\texit_code\n' > "${SUITE_DIR}/status.tsv"

for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
    printf '%s\t%s\tstarted\t-\n' "$method" "$seed" >> "${SUITE_DIR}/status.tsv"
    TUNIX_REWARD_RANK_NOISE_FRACTION="$MISMATCH" \
    TUNIX_REWARD_RANK_NOISE_SEED="$seed" \
    TUNIX_FILTER_RATIO="$FILTER" \
    TUNIX_RUN_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)" \
      "${REPO}/runs_xuesong/scripts/run_aime_seeded_full.sh" \
      "$method" "$seed" "${EXTRA[@]}"
    status=$?
    if [[ $status -ne 0 ]]; then
      printf '%s\t%s\tfailed\t%s\n' "$method" "$seed" "$status" >> "${SUITE_DIR}/status.tsv"
      printf '%s\n' "$status" > "${SUITE_DIR}/exit_code"
      exit "$status"
    fi
    printf '%s\t%s\tcompleted\t0\n' "$method" "$seed" >> "${SUITE_DIR}/status.tsv"
  done
done
printf '0\n' > "${SUITE_DIR}/exit_code"
