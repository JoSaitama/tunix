#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEEDS=()
NOISE=""
DRY_RUN=0
EXTRA=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --seeds) shift
      while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do SEEDS+=("$1"); shift; done ;;
    --mismatch) [ "$#" -ge 2 ] || exit 2; NOISE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    --help|-h)
      echo "usage: bash $0 --seeds 5 0 13 21 42 --mismatch 0|0.2 [--dry-run] [-- GRPO args...]"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ "${#SEEDS[@]}" -gt 0 ] && [ -n "$NOISE" ] || exit 2
awk -v x="$NOISE" 'BEGIN {exit !(x ~ /^[0-9]+([.][0-9]+)?$/ && x>=0 && x<=1)}' || exit 2
for seed in "${SEEDS[@]}"; do [[ "$seed" =~ ^[0-9]+$ ]] || exit 2; done
if [ "$DRY_RUN" -eq 0 ]; then
  SUITE_DIR="$ROOT_DIR/logs/rebuttal_grpo_gsm8k_gradient_norm_top50_suite_mismatch${NOISE}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$SUITE_DIR"
  printf 'seed\tmismatch\tstatus\texit_code\n' > "$SUITE_DIR/status.tsv"
fi
for seed in "${SEEDS[@]}"; do
  COMMAND=(env "TUNIX_EXPERIMENT_SEED=$seed" "TUNIX_REWARD_RANK_NOISE_SEED=$seed"
    "TUNIX_REWARD_RANK_NOISE_FRACTION=$NOISE" bash "$ROOT_DIR/my_example/rebuttal/run_gradient_norm_baseline.sh")
  COMMAND+=( ${EXTRA[@]+"${EXTRA[@]}"} )
  if [ "$DRY_RUN" -eq 1 ]; then
    COMMAND+=(--dry-run)
    "${COMMAND[@]}"
    continue
  fi
  printf '%s\t%s\tstarted\t-\n' "$seed" "$NOISE" >> "$SUITE_DIR/status.tsv"
  if "${COMMAND[@]}"; then CODE=0; else CODE=$?; fi
  printf '%s\t%s\tfinished\t%s\n' "$seed" "$NOISE" "$CODE" >> "$SUITE_DIR/status.tsv"
  if [ "$CODE" -ne 0 ]; then exit "$CODE"; fi
done
