#!/usr/bin/env bash
# GSM8K GRPO with a groupwise PCGrad/CAGrad actor update.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
usage() {
  echo "usage: $0 {pcgrad|cagrad} [--include-kl] [--cagrad-c C] [--dry-run] [GRPO arguments...]"
}
METHOD="${1:-}"
case "$METHOD" in
  pcgrad|cagrad) shift ;;
  --help|-h) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
if [ -f my_example/.env ]; then
  set -a
  source my_example/.env
  set +a
fi
INCLUDE_KL=0
C=0.5
DRY_RUN=0
ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --include-kl) INCLUDE_KL=1; shift ;;
    --cagrad-c)
      [ "$METHOD" = cagrad ] && [ "$#" -ge 2 ] || { usage >&2; exit 2; }
      C="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    --use-dbc-*|--use-dynamic-batch-curation*)
      echo 'PCGrad/CAGrad already select the actor trainer.' >&2; exit 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if [ -n "${TUNIX_PODS_KEEP_COUNT:-}" ] || [ "${TUNIX_DYNAMIC_SAMPLING:-0}" = 1 ]; then
  echo 'PCGrad/CAGrad cannot be combined with PODS or dynamic sampling.' >&2
  exit 2
fi
if ! awk -v c="$C" 'BEGIN {exit !(c ~ /^([0-9]+([.][0-9]+)?|[.][0-9]+)$/ && c>=0 && c<1)}'; then
  echo 'CAGrad c must be in [0, 1).' >&2
  exit 2
fi
SEED="${TUNIX_EXPERIMENT_SEED:-0}"
NOISE="${TUNIX_REWARD_RANK_NOISE_FRACTION:-0}"
NOISE_SEED="${TUNIX_REWARD_RANK_NOISE_SEED:-$SEED}"
[[ "$SEED" =~ ^[0-9]+$ && "$NOISE_SEED" =~ ^[0-9]+$ ]] || { echo 'Seeds must be nonnegative integers.' >&2; exit 2; }
awk -v x="$NOISE" 'BEGIN {exit !(x ~ /^[0-9]+([.][0-9]+)?$/ && x>=0 && x<=1)}' || {
  echo 'Mismatch fraction must be in [0, 1].' >&2; exit 2;
}
MODE=policy
if [ "$INCLUDE_KL" -eq 1 ]; then MODE=full; fi
LABEL="${METHOD}_group_${MODE}"
if [ "$METHOD" = cagrad ]; then LABEL="${LABEL}_c${C//./p}"; fi
RUN_TS="$(date +%Y%m%d_%H%M%S)_$$"
RUN_ROOT="${TUNIX_GRADIENT_CONFLICT_RUN_ROOT:-$ROOT_DIR/logs/rebuttal_grpo_gsm8k_${LABEL}_seed${SEED}_mismatch${NOISE}_${RUN_TS}}"
METRICS_LOG_DIR="$RUN_ROOT/tensorboard"
for ((i=0; i<${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --metrics-log-dir)
      [ "$((i+1))" -lt "${#ARGS[@]}" ] || exit 2
      METRICS_LOG_DIR="${ARGS[$((i+1))]}" ;;
    --metrics-log-dir=*) METRICS_LOG_DIR="${ARGS[$i]#*=}" ;;
    --num-generations)
      [ "${ARGS[$((i+1))]:-}" = 4 ] || { echo 'PCGrad/CAGrad require G=4.' >&2; exit 2; } ;;
    --num-generations=*)
      [ "${ARGS[$i]#*=}" = 4 ] || { echo 'PCGrad/CAGrad require G=4.' >&2; exit 2; } ;;
  esac
done
COMMAND=(env -u JAX_PLATFORM_NAME -u TUNIX_GRPO_BENCHMARK
  JAX_PLATFORMS=tpu TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
  TPU_VISIBLE_CHIPS=0,1,2,3 CLOUD_TPU_TASK_ID=0
  TUNIX_DBC_VARIANT=gradient_conflict TUNIX_REWARD_MODE=accuracy
  "TUNIX_GRADIENT_CONFLICT_METHOD=$METHOD" "TUNIX_GRADIENT_CONFLICT_INCLUDE_KL=$INCLUDE_KL"
  "TUNIX_CAGRAD_C=$C" "TUNIX_EXPERIMENT_SEED=$SEED"
  "TUNIX_REWARD_RANK_NOISE_FRACTION=$NOISE" "TUNIX_REWARD_RANK_NOISE_SEED=$NOISE_SEED"
  python -m my_example.main
  --source tfds --train-data-dir ./data/train --test-data-dir ./data/test
  --train-fraction 0.9 --train-micro-batch-size 4 --test-micro-batch-size 1
  --num-generations 4 --max-train-examples 3072 --max-eval-examples 1319
  --num-epochs 1 --learning-rate 1e-6 --no-wandb --use-dynamic-batch-curation
  --metrics-log-dir "$METRICS_LOG_DIR" --checkpoint-root "$RUN_ROOT/checkpoints"
  --output-dir "$RUN_ROOT/model" --model-id google/gemma-3-1b-it
  --tokenizer-path gs://gemma-data/tokenizers/tokenizer_gemma3.model --mesh-counts 4,1)
COMMAND+=( ${ARGS[@]+"${ARGS[@]}"} )
if [ "$DRY_RUN" -eq 1 ]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
if pgrep -af '[p]ython.*( -m my_example\.|/my_example/)'; then
  echo 'Another my_example task is active on this worker.' >&2
  exit 3
fi
mkdir -p "$(dirname "$RUN_ROOT")"
mkdir "$RUN_ROOT"  # Refuse reuse of a previous experiment's checkpoint root.
mkdir "$RUN_ROOT/results"
printf '%q ' "${COMMAND[@]}" > "$RUN_ROOT/launch_command.txt"
printf '\n' >> "$RUN_ROOT/launch_command.txt"
trap 'code=$?; printf "%s\n" "$code" > "$RUN_ROOT/exit_code"' EXIT
printf 'Run directory %s\n' "$RUN_ROOT"
"${COMMAND[@]}" 2>&1 | tee "$RUN_ROOT/nohup.log"
python my_example/save_results_to_my_result.py \
  --tb-logdir "$METRICS_LOG_DIR" --label "$LABEL" --outdir "$RUN_ROOT/results" \
  --stdout-log "$RUN_ROOT/nohup.log" || echo '[warn] result export failed' >&2
