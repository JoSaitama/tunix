#!/usr/bin/env bash
# Frozen GSM8K entrypoint/exporter; opt-in actor trainer, no new training CLI.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
if [ -f my_example/.env ]; then set -a; source my_example/.env; set +a; fi
ARGS=()
DRY_RUN=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) echo "usage: bash $0 [--dry-run] [existing frozen GRPO args...]"; exit 0 ;;
    --use-dbc-*|--use-dynamic-batch-curation*)
      echo 'Norm top50 already selects the actor trainer; do not combine filtering methods.' >&2; exit 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if [ -n "${TUNIX_PODS_KEEP_COUNT:-}" ] || [ "${TUNIX_DYNAMIC_SAMPLING:-0}" = 1 ]; then
  echo 'Clear PODS/DS environment flags before launching gradient-norm top50.' >&2; exit 2
fi
SEED="${TUNIX_EXPERIMENT_SEED:-0}"
NOISE="${TUNIX_REWARD_RANK_NOISE_FRACTION:-0}"
NOISE_SEED="${TUNIX_REWARD_RANK_NOISE_SEED:-$SEED}"
[[ "$SEED" =~ ^[0-9]+$ && "$NOISE_SEED" =~ ^[0-9]+$ ]] || exit 2
awk -v x="$NOISE" 'BEGIN {exit !(x ~ /^[0-9]+([.][0-9]+)?$/ && x>=0 && x<=1)}' || exit 2
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${TUNIX_NORM_RUN_ROOT:-$ROOT_DIR/logs/rebuttal_grpo_gsm8k_gradient_norm_top50_seed${SEED}_mismatch${NOISE}_${RUN_TS}}"
RESULTS_DIR="${TUNIX_MY_RESULT_DIR:-$RUN_ROOT/results}"
METRICS_LOG_DIR="$RUN_ROOT/tensorboard"
LABEL="gradient_norm_top50__grpo_${RUN_TS}"
RUN_LOG="$RESULTS_DIR/${LABEL}__stdout.log"
for ((i=0; i<${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --metrics-log-dir) [ "$((i+1))" -lt "${#ARGS[@]}" ] || exit 2
      METRICS_LOG_DIR="${ARGS[$((i+1))]}" ;;
    --metrics-log-dir=*) METRICS_LOG_DIR="${ARGS[$i]#*=}" ;;
    --num-generations) [ "$((i+1))" -lt "${#ARGS[@]}" ] || exit 2
      [ "${ARGS[$((i+1))]}" = 4 ] || { echo 'Frozen norm baseline requires G=4.' >&2; exit 2; } ;;
    --num-generations=*) [ "${ARGS[$i]#*=}" = 4 ] || exit 2 ;;
  esac
done
COMMAND=(env -u JAX_PLATFORMS -u JAX_PLATFORM_NAME
  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
  TPU_VISIBLE_CHIPS=0,1,2,3 CLOUD_TPU_TASK_ID=0
  TUNIX_DBC_VARIANT=gradient_norm_top50 TUNIX_GRPO_NUM_GENERATIONS=4
  "TUNIX_NORM_ARTIFACT_DIR=$RUN_ROOT/selection"
  "TUNIX_EXPERIMENT_SEED=$SEED" "TUNIX_REWARD_RANK_NOISE_FRACTION=$NOISE"
  "TUNIX_REWARD_RANK_NOISE_SEED=$NOISE_SEED"
  python -m my_example.main
  --source tfds --train-data-dir ./data/train --test-data-dir ./data/test
  --train-fraction .9 --train-micro-batch-size 4 --test-micro-batch-size 1
  --num-generations 4 --max-train-examples 3072 --max-eval-examples 1319
  --num-epochs 1 --learning-rate 1e-6 --no-wandb --use-dynamic-batch-curation
  --metrics-log-dir "$METRICS_LOG_DIR" --checkpoint-root "$RUN_ROOT/checkpoints"
  --output-dir "$RUN_ROOT/model" --model-id google/gemma-3-1b-it
  --tokenizer-path gs://gemma-data/tokenizers/tokenizer_gemma3.model --mesh-counts 4,1)
COMMAND+=( ${ARGS[@]+"${ARGS[@]}"} )
if [ "$DRY_RUN" -eq 1 ]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
if pgrep -af '[p]ython.*( -m my_example\.|/my_example/)'; then
  echo 'STOP: another model/test task is active on this worker.' >&2; exit 3
fi
if [ -e "$RUN_ROOT" ]; then
  echo "Use a fresh run directory; refusing to overwrite $RUN_ROOT" >&2; exit 2
fi
mkdir -p "$RESULTS_DIR"
printf '%q ' "${COMMAND[@]}" > "$RUN_ROOT/launch_command.txt"
printf '\n' >> "$RUN_ROOT/launch_command.txt"
python - "$RUN_ROOT/runtime_config.json" "${COMMAND[@]}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from my_example.config import build_parser
command = sys.argv[2:]
start = command.index('my_example.main') + 1
arguments = vars(build_parser().parse_args(command[start:]))
Path(sys.argv[1]).write_text(json.dumps({
    'training_cli_including_current_parser_defaults': arguments,
    'config_source_sha256': hashlib.sha256(Path('my_example/config.py').read_bytes()).hexdigest(),
    'provenance': 'actual current launch, not an assertion about historical archived overrides',
}, indent=2, allow_nan=False) + '\n', encoding='utf-8')
PY
set +e
"${COMMAND[@]}" 2>&1 | tee "$RUN_LOG"
PIPE_CODES=("${PIPESTATUS[@]}")
CODE=${PIPE_CODES[0]}
if [ "$CODE" -eq 0 ]; then CODE=${PIPE_CODES[1]}; fi
set -e
printf '%s\n' "$CODE" > "$RUN_ROOT/exit_code"
python my_example/save_results_to_my_result.py \
  --tb-logdir "$METRICS_LOG_DIR" --label "$LABEL" --outdir "$RESULTS_DIR" \
  --stdout-log "$RUN_LOG" || echo '[warn] result export failed' >&2
exit "$CODE"
