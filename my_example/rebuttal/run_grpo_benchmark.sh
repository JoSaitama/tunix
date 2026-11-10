#!/usr/bin/env bash
# No training or TPU access for --dry-run. The ordinary launchers are unchanged.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ ${1:-} == --help || $# -lt 3 ]]; then
  echo "usage: bash $0 {vanilla|dtv|dtv-loo} {timing|memory} OUT [WARMUP=15] [STEPS=50] [off|on] [--dry-run]"
  [[ ${1:-} == --help ]] && exit 0
  exit 2
fi
METHOD="$1"
MODE="$2"
OUT="$3"
WARMUP="${4:-15}"
STEPS="${5:-50}"
DIAGNOSTICS="${6:-off}"
DRY_RUN="${7:-}"
[[ -n "$OUT" && $# -le 7 && ( -z "$DRY_RUN" || "$DRY_RUN" == --dry-run ) ]] || exit 2
[[ "$MODE" == timing || "$MODE" == memory ]] || exit 2
[[ "$DIAGNOSTICS" == off || ( "$DIAGNOSTICS" == on && "$MODE" == timing ) ]] || exit 2
[[ "$WARMUP" =~ ^[1-9][0-9]{0,2}$ && "$STEPS" =~ ^[1-9][0-9]{0,2}$ ]] || exit 2
(( WARMUP + STEPS <= 691 )) || exit 2
[[ "$OUT" == /* ]] || OUT="$ROOT/$OUT"

unset JAX_PLATFORMS JAX_PLATFORM_NAME
unset TUNIX_PODS_KEEP_COUNT TUNIX_DYNAMIC_SAMPLING TUNIX_FIXED_FILTER_METHOD
unset TUNIX_DBC_VARIANT TUNIX_DBC_SELF_INF_SCOPE TUNIX_DBC_SELF_INF_MIN_KEEP_FRACTION
unset TUNIX_DBC_SELF_INF_POLICY TUNIX_DBC_SELF_INF_LOO TUNIX_DBC_SELF_INF_LOO_POLICY
unset TUNIX_DBC_SELF_INF_LOO_POLICY_ONLY TUNIX_DBC_SELF_INF_FLIP
unset TUNIX_DBC_SELF_INF_DECISIONS_PATH TUNIX_ACTOR_GRAD_ACC_FACTOR
unset TUNIX_ACTOR_PROMPT_GROUP_COALESCE TUNIX_ENABLE_ACTOR_PHASE_TIMING
unset TUNIX_ENABLE_ACTOR_HEARTBEAT TUNIX_ENABLE_ACTOR_PROGRESS_METRICS

METHOD_ARGS=()
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
  *) echo "Unsupported benchmark method: $METHOD" >&2; exit 2 ;;
esac
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

COMMAND=(bash my_example/run_grpo_gemma.sh --skip-eval-before --skip-eval-after
  --metrics-log-dir "$OUT/tensorboard"
  --checkpoint-root "$OUT/checkpoints" --output-dir "$OUT/model")
COMMAND+=( ${METHOD_ARGS[@]+"${METHOD_ARGS[@]}"} )
if [[ "$DRY_RUN" == --dry-run ]]; then
  printf 'method=%s mode=%s diagnostics=%s warmup=%s steps=%s seed=13 noise=0 topology=single-worker-4-TPU\n' \
    "$METHOD" "$MODE" "$DIAGNOSTICS" "$WARMUP" "$STEPS"
  printf 'score_policy=%s loo=%s loo_policy=%s selection=%s\n' \
    "${TUNIX_DBC_SELF_INF_POLICY:-off}" "${TUNIX_DBC_SELF_INF_LOO:-off}" \
    "${TUNIX_DBC_SELF_INF_LOO_POLICY:-off}" "${TUNIX_DBC_SELF_INF_DECISIONS_PATH:-off}"
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
