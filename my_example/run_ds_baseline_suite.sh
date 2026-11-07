#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS=()
NOISE=""
CAP=20
DRY_RUN=0
EXTRA=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --seeds) shift
      while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do SEEDS+=("$1"); shift; done ;;
    --mismatch|--ds-max-generation-batches)
      [ "$#" -ge 2 ] || exit 2
      if [ "$1" = --mismatch ]; then NOISE="$2"; else CAP="$2"; fi
      shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    --help|-h)
      echo "usage: bash my_example/run_ds_baseline_suite.sh --seeds 5 0 13 21 42 --mismatch 0|0.2 [--ds-max-generation-batches 20] [--dry-run] [-- GRPO args...]"
      exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ "${#SEEDS[@]}" -gt 0 ] && [ -n "$NOISE" ] || { echo "--seeds and --mismatch required" >&2; exit 2; }
awk -v x="$NOISE" 'BEGIN {exit !(x ~ /^[0-9]+([.][0-9]+)?$/ && x>=0 && x<=1)}' || exit 2
[[ "$CAP" =~ ^[0-9]+$ ]] && [ "$CAP" -ge 1 ] || exit 2
for seed in "${SEEDS[@]}"; do [[ "$seed" =~ ^[0-9]+$ ]] || exit 2; done
if [ "$DRY_RUN" -eq 0 ]; then
  SUITE_DIR="${ROOT_DIR}/logs/rebuttal_grpo_gsm8k_ds_suite_mismatch${NOISE}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$SUITE_DIR"
  printf 'seed\tmismatch\tstatus\texit_code\n' > "${SUITE_DIR}/status.tsv"
fi
for seed in "${SEEDS[@]}"; do
  COMMAND=(env TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
    TPU_VISIBLE_CHIPS=0,1,2,3 CLOUD_TPU_TASK_ID=0
    "TUNIX_EXPERIMENT_SEED=$seed" "TUNIX_REWARD_RANK_NOISE_SEED=$seed"
    "TUNIX_REWARD_RANK_NOISE_FRACTION=$NOISE" bash "${ROOT_DIR}/my_example/run_ds_baseline.sh"
    --ds-max-generation-batches "$CAP")
  COMMAND+=( ${EXTRA[@]+"${EXTRA[@]}"} )
  if [ "$DRY_RUN" -eq 1 ]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; continue; fi
  printf '%s\t%s\tstarted\t-\n' "$seed" "$NOISE" >> "${SUITE_DIR}/status.tsv"
  if "${COMMAND[@]}"; then CODE=0; else CODE=$?; fi
  printf '%s\t%s\tfinished\t%s\n' "$seed" "$NOISE" "$CODE" >> "${SUITE_DIR}/status.tsv"
  if [ "$CODE" -ne 0 ]; then exit "$CODE"; fi
done
