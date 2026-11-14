#!/usr/bin/env bash
# Sequential audit on the current TPU worker. Dry-run never starts Python.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  printf '%s\n' \
    "usage: bash $0 {smoke|full} OUTPUT_ROOT [--dry-run]" \
    "       bash $0 correctness OUTPUT_ROOT {clean|mismatch0.2} [--dry-run]" >&2
  exit 2
}
[[ $# -ge 2 ]] || usage
mode="$1"
root="$2"
shift 2
conditions=(clean mismatch0.2)
case "$mode" in
  smoke) steps=1; seeds=(5); summary_args=(--smoke) ;;
  full) steps=32; seeds=(5 0 13 21 42); summary_args=() ;;
  correctness)
    [[ "${1:-}" == clean || "${1:-}" == mismatch0.2 ]] || usage
    conditions=("$1")
    steps=32; seeds=(5); summary_args=(--correctness --condition "$1")
    shift
    ;;
  *) usage ;;
esac
dry="${1:-}"
[[ -n "$root" ]] || usage
[[ $# -eq 0 || ( $# -eq 1 && "$dry" == --dry-run ) ]] || usage
[[ "$root" == /* ]] || root="$PWD/$root"
if [[ -z "$dry" ]]; then
  [[ ! -e "$root" && ! -L "$root" ]] || { echo "Refusing existing output: $root" >&2; exit 2; }
  test -d data/train
  if pgrep -af '[p]ython[^ ]* -m my_example\.'; then
    echo "Another Tunix job is active on this worker" >&2
    exit 3
  fi
  git diff --quiet HEAD -- my_example tunix || {
    echo "Commit local training code before running a versioned audit" >&2
    exit 3
  }
  if [[ -f my_example/.env ]]; then
    set -a
    source my_example/.env
    set +a
  fi
  mkdir -p "$(dirname "$root")"
  mkdir "$root"
  trap 'rc=$?; printf "%s\n" "$rc" >"$root/exit_code"' EXIT
fi

unset JAX_PLATFORM_NAME TUNIX_GRPO_BENCHMARK TUNIX_DYNAMIC_SAMPLING TUNIX_PODS_KEEP_COUNT
unset TUNIX_DBC_SELF_INF_LOO_POLICY_ONLY TUNIX_DBC_SELF_INF_MIN_KEEP_FRACTION
unset TUNIX_DBC_SELF_INF_DECISIONS_PATH TUNIX_ACTOR_GRAD_ACC_FACTOR TUNIX_ACTOR_PROMPT_GROUP_COALESCE
unset TUNIX_ENABLE_ACTOR_PHASE_TIMING TUNIX_ENABLE_ACTOR_HEARTBEAT TUNIX_ENABLE_ACTOR_PROGRESS_METRICS
export JAX_PLATFORMS=tpu TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
export TPU_VISIBLE_CHIPS=0,1,2,3 CLOUD_TPU_TASK_ID=0
export TUNIX_GRPO_FILTER_AUDIT=1 TUNIX_FILTER_AUDIT_STEPS="$steps"
export TUNIX_REWARD_MODE=accuracy TUNIX_DISABLE_TFLOPS_MEASUREMENT=1 PYTHONUNBUFFERED=1
unset TUNIX_FILTER_AUDIT_CORRECTNESS
if [[ "$mode" == correctness ]]; then
  export TUNIX_FILTER_AUDIT_CORRECTNESS=1
fi

for condition in "${conditions[@]}"; do
  export TUNIX_FILTER_AUDIT_CONDITION="$condition"
  fraction=0
  [[ "$condition" == clean ]] || fraction=0.2
  export TUNIX_REWARD_RANK_NOISE_FRACTION="$fraction"
  for seed in "${seeds[@]}"; do
    export TUNIX_EXPERIMENT_SEED="$seed" TUNIX_REWARD_RANK_NOISE_SEED="$seed"
    for method in dtv dtv-loo; do
      export TUNIX_FILTER_AUDIT_METHOD="$method"
      if [[ "$method" == dtv ]]; then
        export TUNIX_DBC_SELF_INF_POLICY=1 TUNIX_DBC_SELF_INF_LOO=0 TUNIX_DBC_SELF_INF_LOO_POLICY=0
      else
        export TUNIX_DBC_SELF_INF_POLICY=0 TUNIX_DBC_SELF_INF_LOO=1 TUNIX_DBC_SELF_INF_LOO_POLICY=1
      fi
      run="$root/${method}_seed${seed}_${condition}"
      export TUNIX_FILTER_AUDIT_OUT="$run"
      command=(python -m my_example.main --source tfds --train-data-dir ./data/train
        --train-fraction 0.9 --train-micro-batch-size 4 --num-generations 4 --max-train-examples 3072
        --num-epochs 1 --learning-rate 1e-6 --use-dbc-self-inf-group
        --no-wandb --skip-eval-before --skip-eval-after
        --model-id google/gemma-3-1b-it
        --tokenizer-path gs://gemma-data/tokenizers/tokenizer_gemma3.model --mesh-counts 4,1)
      printf '%s method=%s seed=%s condition=%s fraction=%s updates=%s output=%s\n' \
        "${dry:-START}" "$method" "$seed" "$condition" "$fraction" "$steps" "$run"
      if [[ -n "$dry" ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
      else
        mkdir "$run"
        "${command[@]}" 2>&1 | tee "$run/stdout.log"
      fi
    done
  done
done

if [[ -z "$dry" ]]; then
  python -m my_example.rebuttal.grpo_filter_audit "$root" "${summary_args[@]+${summary_args[@]}}" | tee "$root/summary.md"
  python -m my_example.rebuttal.grpo_filter_audit "$root" "${summary_args[@]+${summary_args[@]}}" --json >"$root/summary.json"
fi
