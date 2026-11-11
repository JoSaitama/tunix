#!/usr/bin/env bash
# No training or TPU access for --dry-run. The ordinary launchers are unchanged.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ ${1:-} == --help || $# -lt 3 ]]; then
  echo "usage: bash $0 {vanilla|dtv|dtv-loo|pods|grpo-ds|norm-top50|pcgrad|cagrad|gradalign|random10|reward10|learnalign} {timing|memory} OUT [WARMUP] [STEPS] [off|on] [--dry-run]"
  echo 'LearnAlign: timing 10+50; memory 1+2 (prelearn) or 10+2 (postselect).'
  [[ ${1:-} == --help ]] && exit 0
  exit 2
fi
METHOD="$1"
MODE="$2"
OUT="$3"
DEFAULT_WARMUP=15 DEFAULT_STEPS=50
if [[ "$METHOD" == gradalign ]]; then
  if [[ "$MODE" == memory ]]; then DEFAULT_WARMUP=0 DEFAULT_STEPS=1
  else DEFAULT_WARMUP=20; fi
elif [[ "$MODE" == memory ]]; then
  DEFAULT_WARMUP=1 DEFAULT_STEPS=2
elif [[ "$METHOD" == learnalign ]]; then
  DEFAULT_WARMUP=10
fi
WARMUP="${4:-$DEFAULT_WARMUP}"
STEPS="${5:-$DEFAULT_STEPS}"
DIAGNOSTICS="${6:-off}"
DRY_RUN="${7:-}"
[[ -n "$OUT" && $# -le 7 && ( -z "$DRY_RUN" || "$DRY_RUN" == --dry-run ) ]] || exit 2
[[ "$MODE" == timing || "$MODE" == memory ]] || exit 2
[[ "$DIAGNOSTICS" == off || ( "$DIAGNOSTICS" == on && "$MODE" == timing ) ]] || exit 2
[[ "$WARMUP" =~ ^(0|[1-9][0-9]{0,2})$ && "$STEPS" =~ ^[1-9][0-9]{0,2}$ ]] || exit 2
(( WARMUP + STEPS <= 691 )) || exit 2
if [[ "$METHOD" == gradalign ]]; then
  if [[ "$MODE" == memory ]]; then
    [[ "$WARMUP" == 0 && "$STEPS" == 1 ]] || { echo 'GradAlign memory uses 0 + 1.' >&2; exit 2; }
  else
    (( WARMUP >= 10 && WARMUP % 10 == 0 && STEPS % 10 == 0 )) || {
      echo 'GradAlign timing needs aligned ten-update cycles.' >&2; exit 2;
    }
  fi
elif [[ "$METHOD" == learnalign ]]; then
  if [[ "$MODE" == memory ]]; then
    [[ "$STEPS" == 2 && ( "$WARMUP" == 1 || "$WARMUP" == 10 ) ]] || exit 2
  else
    [[ "$WARMUP" == 10 && "$STEPS" == 50 ]] || exit 2
  fi
else
  (( WARMUP > 0 )) || exit 2
fi
[[ "$OUT" == /* ]] || OUT="$ROOT/$OUT"
if [[ "$DRY_RUN" != --dry-run && -f my_example/.env ]]; then
  set -a; source my_example/.env; set +a
fi

unset JAX_PLATFORMS JAX_PLATFORM_NAME
unset TUNIX_PODS_KEEP_COUNT TUNIX_DYNAMIC_SAMPLING TUNIX_FIXED_FILTER_METHOD
unset TUNIX_FIXED_FILTER_SCOPE TUNIX_FIXED_FILTER_RATIO TUNIX_FIXED_FILTER_DECISIONS_PATH
unset TUNIX_DBC_VARIANT TUNIX_DBC_SELF_INF_SCOPE TUNIX_DBC_SELF_INF_MIN_KEEP_FRACTION
unset TUNIX_DBC_SELF_INF_POLICY TUNIX_DBC_SELF_INF_LOO TUNIX_DBC_SELF_INF_LOO_POLICY
unset TUNIX_DBC_SELF_INF_LOO_POLICY_ONLY TUNIX_DBC_SELF_INF_FLIP
unset TUNIX_DBC_SELF_INF_DECISIONS_PATH TUNIX_ACTOR_GRAD_ACC_FACTOR
unset TUNIX_ACTOR_PROMPT_GROUP_COALESCE TUNIX_ENABLE_ACTOR_PHASE_TIMING
unset TUNIX_ENABLE_ACTOR_HEARTBEAT TUNIX_ENABLE_ACTOR_PROGRESS_METRICS
unset TUNIX_PODS_ARTIFACT_DIR TUNIX_DS_ARTIFACT_DIR TUNIX_NORM_ARTIFACT_DIR
unset TUNIX_GRADIENT_CONFLICT_METHOD TUNIX_GRADIENT_CONFLICT_INCLUDE_KL TUNIX_CAGRAD_C
unset TUNIX_LEARNALIGN_EQUIVALENCE_REPORT

METHOD_ARGS=()
MODULE=my_example.main
case "$METHOD" in
  vanilla) ;;
  dtv)
    export TUNIX_DBC_SELF_INF_POLICY=1
    METHOD_ARGS=(--use-dbc-self-inf-group)
    ;;
  dtv-loo)
    export TUNIX_DBC_SELF_INF_LOO=1 TUNIX_DBC_SELF_INF_LOO_POLICY=1
    METHOD_ARGS=(--use-dbc-self-inf-group)
    if [[ "$DIAGNOSTICS" == on ]]; then
      export TUNIX_DBC_SELF_INF_DECISIONS_PATH="$OUT/selection.jsonl"
    fi
    ;;
  pods) export TUNIX_PODS_KEEP_COUNT=2 ;;
  grpo-ds) export TUNIX_DYNAMIC_SAMPLING=1 TUNIX_DS_MAX_GENERATION_BATCHES=20 ;;
  norm-top50)
    export TUNIX_DBC_VARIANT=gradient_norm_top50 TUNIX_GRPO_NUM_GENERATIONS=4
    METHOD_ARGS=(--use-dynamic-batch-curation)
    ;;
  pcgrad|cagrad)
    export TUNIX_DBC_VARIANT=gradient_conflict TUNIX_GRADIENT_CONFLICT_METHOD="$METHOD"
    export TUNIX_GRADIENT_CONFLICT_INCLUDE_KL=0 TUNIX_CAGRAD_C=0.5
    METHOD_ARGS=(--use-dynamic-batch-curation)
    ;;
  gradalign)
    MODULE=my_example.alignment_main
    METHOD_ARGS=(--alignment-method gradalign --gradalign-selection-ratio 4
      --gradalign-validation-prompts 30 --gradalign-candidate-rollouts 4
      --gradalign-validation-rollouts 4 --gradalign-selection-interval 10)
    ;;
  learnalign)
    MODULE=my_example.alignment_main
    METHOD_ARGS=(--alignment-method learnalign --learnalign-selection-ratio 2
      --learnalign-warmup-prompts 40 --learnalign-estimation-rollouts 8
      --learnalign-warmup-counts-toward-budget)
    ;;
  random10|reward10)
    export TUNIX_DBC_VARIANT=fixed_filter TUNIX_GRPO_NUM_GENERATIONS=4
    export TUNIX_FIXED_FILTER_METHOD="${METHOD%10}" TUNIX_FIXED_FILTER_SCOPE=group
    export TUNIX_FIXED_FILTER_RATIO=0.10
    METHOD_ARGS=(--use-dynamic-batch-curation)
    ;;
  *) echo "Unsupported benchmark method: $METHOD" >&2; exit 2 ;;
esac
[[ "$DIAGNOSTICS" == off || "$METHOD" =~ ^(vanilla|dtv|dtv-loo)$ ]] || exit 2
export TUNIX_EXPERIMENT_SEED=13 TUNIX_REWARD_RANK_NOISE_FRACTION=0
export TUNIX_REWARD_RANK_NOISE_SEED=13 TUNIX_REWARD_MODE=accuracy
export TUNIX_DISABLE_TFLOPS_MEASUREMENT=1 PYTHONUNBUFFERED=1
export JAX_ENABLE_COMPILATION_CACHE=false
export TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
export TPU_VISIBLE_CHIPS=0,1,2,3 CLOUD_TPU_TASK_ID=0
export TUNIX_GRPO_BENCHMARK="$MODE" TUNIX_BENCH_METHOD="$METHOD"
export TUNIX_BENCH_DIAGNOSTICS="$DIAGNOSTICS"
export TUNIX_BENCH_WARMUP="$WARMUP" TUNIX_BENCH_STEPS="$STEPS" TUNIX_BENCH_OUT="$OUT"
export TUNIX_BENCH_RESOURCE="${TUNIX_BENCH_RESOURCE:-v5p-16}"
HOST="$(hostname -s)"
export TUNIX_BENCH_PHYSICAL_WORKER="${TUNIX_BENCH_PHYSICAL_WORKER:-${HOST##*-w-}}"

COMMAND=(python -m "$MODULE" --skip-eval-before --skip-eval-after
  --source tfds --train-data-dir ./data/train --test-data-dir ./data/test
  --train-fraction 0.9 --train-micro-batch-size 4 --test-micro-batch-size 1
  --num-generations 4 --max-train-examples 3072 --max-eval-examples 1319
  --num-epochs 1 --learning-rate 1e-6 --no-wandb
  --model-id google/gemma-3-1b-it --mesh-counts 4,1
  --metrics-log-dir "$OUT/tensorboard"
  --checkpoint-root "$OUT/checkpoints" --output-dir "$OUT/model")
COMMAND+=( ${METHOD_ARGS[@]+"${METHOD_ARGS[@]}"} )
if [[ "$DRY_RUN" == --dry-run ]]; then
  printf 'method=%s mode=%s diagnostics=%s warmup=%s steps=%s seed=13 noise=0 topology=single-worker-4-TPU\n' \
    "$METHOD" "$MODE" "$DIAGNOSTICS" "$WARMUP" "$STEPS"
  printf 'score_policy=%s loo=%s loo_policy=%s selection=%s\n' \
    "${TUNIX_DBC_SELF_INF_POLICY:-off}" "${TUNIX_DBC_SELF_INF_LOO:-off}" \
    "${TUNIX_DBC_SELF_INF_LOO_POLICY:-off}" "${TUNIX_DBC_SELF_INF_DECISIONS_PATH:-off}"
  printf 'actor_variant=%s conflict_method=%s pods_keep=%s ds=%s\n' \
    "${TUNIX_DBC_VARIANT:-off}" "${TUNIX_GRADIENT_CONFLICT_METHOD:-off}" \
    "${TUNIX_PODS_KEEP_COUNT:-off}" "${TUNIX_DYNAMIC_SAMPLING:-off}"
  printf 'fixed_filter=%s scope=%s ratio=%s\n' \
    "${TUNIX_FIXED_FILTER_METHOD:-off}" "${TUNIX_FIXED_FILTER_SCOPE:-off}" "${TUNIX_FIXED_FILTER_RATIO:-off}"
  printf '%q ' "${COMMAND[@]}"; printf '\n'
  exit 0
fi
if pgrep -af '[p]ython[^ ]* -m my_example\.'; then
  echo 'STOP: another Tunix Python job is active on this worker' >&2
  exit 3
fi
git diff --quiet HEAD -- my_example tunix || {
  echo 'STOP: tracked training code has local changes; record a clean benchmark revision.' >&2
  exit 3
}
[[ ! -e "$OUT" && ! -L "$OUT" ]] || { echo "Refusing existing output: $OUT" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"
if [[ "$MODE" == memory ]]; then
  export TUNIX_BENCH_MIN_FREE_GIB="${TUNIX_BENCH_MIN_FREE_GIB:-6}"
  python -m my_example.rebuttal.grpo_benchmark disk-check "$(dirname "$OUT")" \
    --min-free-gib "$TUNIX_BENCH_MIN_FREE_GIB"
fi
mkdir "$OUT"
export TUNIX_BENCH_LAUNCH_MONOTONIC="$(python -c 'import time; print(time.perf_counter())')"
set +e
"${COMMAND[@]}" 2>&1 | tee "$OUT/stdout.log"
CODES=("${PIPESTATUS[@]}")
set -e
CODE="${CODES[0]}"
if [[ "$CODE" == 0 ]]; then CODE="${CODES[1]}"; fi
printf '%s\n' "$CODE" > "$OUT/exit_code"
exit "$CODE"
