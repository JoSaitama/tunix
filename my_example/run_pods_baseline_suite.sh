#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS=()
NOISE=""
KEEP_COUNT=2
DRY_RUN=0
EXTRA=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --seeds)
      shift
      while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do SEEDS+=("$1"); shift; done ;;
    --mismatch|--pods-keep-count)
      [ "$#" -ge 2 ] || { echo "missing $1 value" >&2; exit 2; }
      if [ "$1" = --mismatch ]; then NOISE="$2"; else KEEP_COUNT="$2"; fi
      shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    --help|-h)
      echo "usage: bash my_example/run_pods_baseline_suite.sh --seeds 5 0 13 21 42 --mismatch 0|0.2 [--pods-keep-count 2] [--dry-run] [-- GRPO args...]"
      exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ "${#SEEDS[@]}" -gt 0 ] && [ -n "$NOISE" ] || { echo "--seeds and --mismatch are required" >&2; exit 2; }
awk -v x="$NOISE" 'BEGIN {exit !(x ~ /^[0-9]+([.][0-9]+)?$/ && x>=0 && x<=1)}' || exit 2
[[ "$KEEP_COUNT" =~ ^[0-9]+$ ]] && [ "$KEEP_COUNT" -ge 2 ] || exit 2
for seed in "${SEEDS[@]}"; do [[ "$seed" =~ ^[0-9]+$ ]] || exit 2; done
if [ "$DRY_RUN" -eq 0 ]; then
  SUITE_DIR="${ROOT_DIR}/logs/pods_suite_m${KEEP_COUNT}_mismatch${NOISE}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$SUITE_DIR"
  printf 'seed\tmismatch\tkeep_count\tstatus\texit_code\n' > "${SUITE_DIR}/status.tsv"
fi
for seed in "${SEEDS[@]}"; do
  COMMAND=(env "TUNIX_EXPERIMENT_SEED=$seed" "TUNIX_REWARD_RANK_NOISE_SEED=$seed"
    "TUNIX_REWARD_RANK_NOISE_FRACTION=$NOISE" bash "${ROOT_DIR}/my_example/run_pods_baseline.sh"
    --pods-keep-count "$KEEP_COUNT")
  COMMAND+=( ${EXTRA[@]+"${EXTRA[@]}"} )
  if [ "$DRY_RUN" -eq 1 ]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; continue; fi
  printf '%s\t%s\t%s\tstarted\t-\n' "$seed" "$NOISE" "$KEEP_COUNT" >> "${SUITE_DIR}/status.tsv"
  if "${COMMAND[@]}"; then
    printf '%s\t%s\t%s\tcompleted\t0\n' "$seed" "$NOISE" "$KEEP_COUNT" >> "${SUITE_DIR}/status.tsv"
  else
    CODE=$?
    printf '%s\t%s\t%s\tfailed\t%s\n' "$seed" "$NOISE" "$KEEP_COUNT" "$CODE" >> "${SUITE_DIR}/status.tsv"
    exit "$CODE"
  fi
done
