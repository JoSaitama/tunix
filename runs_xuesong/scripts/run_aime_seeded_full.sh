#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/jason_chia925_gmail_com/Project/tunix}"
if [[ $# -lt 2 ]]; then
  echo "usage: $0 METHOD SEED [extra config overrides...]" >&2
  exit 2
fi
METHOD="$1"; SEED="$2"; shift 2
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be nonnegative" >&2; exit 2; }

VARIANT=""
case "$METHOD" in
  baseline) METHOD_SLUG=baseline ;;
  batch_policy) METHOD_SLUG=dtv_selfinf_batch_policy; VARIANT=self_inf_batch_policy ;;
  group_policy) METHOD_SLUG=dtv_selfinf_group_policy; VARIANT=self_inf_group_policy ;;
  batch_loo_policy) METHOD_SLUG=dtv_selfinf_batch_loo_policy; VARIANT=self_inf_batch_loo_policy ;;
  group_loo_policy) METHOD_SLUG=dtv_selfinf_group_loo_policy; VARIANT=self_inf_group_loo_policy ;;
  random_batch|random_group|reward_batch|reward_group)
    METHOD_SLUG="$METHOD"; VARIANT=fixed_filter
    export TUNIX_FIXED_FILTER_METHOD="${METHOD%%_*}"
    export TUNIX_FIXED_FILTER_SCOPE="${METHOD##*_}"
    export TUNIX_FIXED_FILTER_RATIO="${TUNIX_FILTER_RATIO:?TUNIX_FILTER_RATIO is required}"
    ;;
  *) echo "unsupported METHOD=$METHOD" >&2; exit 2 ;;
esac

NOISE="${TUNIX_REWARD_RANK_NOISE_FRACTION:-0}"
if awk -v n="$NOISE" 'BEGIN { exit !(n > 0) }'; then
  NOISE_CANONICAL="$(awk -v n="$NOISE" 'BEGIN { printf "%g", n }')"
  DATA_MODE="mismatch${NOISE_CANONICAL//./p}"
else
  DATA_MODE=clean
fi
FILTER_SUFFIX=""
case "$METHOD" in
  random_*|reward_*)
    FILTER_PERCENT="$(awk -v n="$TUNIX_FILTER_RATIO" 'BEGIN { printf "%02d", n*100 }')"
    FILTER_SUFFIX="_filter0p${FILTER_PERCENT}"
    ;;
esac

RUN_TS="${TUNIX_RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_NAME="grpo_aime_${METHOD_SLUG}${FILTER_SUFFIX}_seed${SEED}_${DATA_MODE}_${RUN_TS}"
export RUN_NAME
export RUN_ROOT="${REPO}/runs_xuesong/runs/${RUN_NAME}"
export LOG_ROOT="${REPO}/runs_xuesong/logs/${RUN_NAME}"
export CACHE_ROOT="${REPO}/runs_xuesong/cache/${RUN_NAME}"
export RUN_SCRIPT=examples/deepscaler/run_deepscaler_disagg_v5p16_1epoch.sh
export NUM_BATCHES="${NUM_BATCHES:-64}"
export MAX_STEPS_OVERRIDE="${MAX_STEPS_OVERRIDE:-$NUM_BATCHES}"
if [[ "$NUM_BATCHES" -le 64 ]]; then
  CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-$NUM_BATCHES}"
else
  CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-64}"
fi
export TUNIX_EXPERIMENT_SEED="$SEED"
export TUNIX_REWARD_RANK_NOISE_SEED="${TUNIX_REWARD_RANK_NOISE_SEED:-$SEED}"
export TUNIX_GRPO_NUM_GENERATIONS=8
export TUNIX_DBC_DECISIONS_PATH="${LOG_ROOT}/selection.jsonl"
export TUNIX_DATA_MANIFEST_PATH="${LOG_ROOT}/data_manifest.json"
export TUNIX_TRAINING_SUMMARY_PATH="${LOG_ROOT}/training_summary.json"
mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$CACHE_ROOT"
exec > >(tee "${LOG_ROOT}/nohup.log") 2>&1

RESUME_ACTOR_CHECKPOINT_ROOT="${TUNIX_RESUME_ACTOR_CHECKPOINT_ROOT:-}"
RESUME_CHECKPOINT_STEP="${TUNIX_RESUME_CHECKPOINT_STEP:-}"
if [[ -n "$RESUME_ACTOR_CHECKPOINT_ROOT" || -n "$RESUME_CHECKPOINT_STEP" ]]; then
  [[ -n "$RESUME_ACTOR_CHECKPOINT_ROOT" && -n "$RESUME_CHECKPOINT_STEP" ]] || {
    echo "TUNIX_RESUME_ACTOR_CHECKPOINT_ROOT and TUNIX_RESUME_CHECKPOINT_STEP must be set together" >&2
    exit 2
  }
  [[ "$RESUME_CHECKPOINT_STEP" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid TUNIX_RESUME_CHECKPOINT_STEP=$RESUME_CHECKPOINT_STEP" >&2
    exit 2
  }

  RESUME_SOURCE="${RESUME_ACTOR_CHECKPOINT_ROOT}/${RESUME_CHECKPOINT_STEP}"
  RESUME_DEST_ROOT="${RUN_ROOT}/checkpoints/actor"
  RESUME_DEST="${RESUME_DEST_ROOT}/${RESUME_CHECKPOINT_STEP}"
  [[ -d "$RESUME_SOURCE" ]] || {
    echo "resume checkpoint is missing: $RESUME_SOURCE" >&2
    exit 1
  }
  mkdir -p "$RESUME_DEST_ROOT"
  if [[ -e "$RESUME_DEST" || -L "$RESUME_DEST" ]]; then
    [[ "$(realpath "$RESUME_DEST")" == "$(realpath "$RESUME_SOURCE")" ]] || {
      echo "resume destination already points elsewhere: $RESUME_DEST" >&2
      exit 1
    }
  else
    ln -s "$RESUME_SOURCE" "$RESUME_DEST"
  fi
  echo "Resume checkpoint: $RESUME_DEST -> $(realpath "$RESUME_DEST")"
fi

SUMMARY_MINI_BATCH_SIZE=128
SUMMARY_TRAIN_MICRO_BATCH_SIZE=2
SUMMARY_MAX_RESPONSE_LENGTH=8192
SUMMARY_LR_SCHEDULE=cosine_decay_schedule
SUMMARY_LR_INIT=1e-6
SUMMARY_LR_DECAY_STEPS="$MAX_STEPS_OVERRIDE"
for override in "$@"; do
  case "$override" in
    rl_training_config.mini_batch_size=*)
      SUMMARY_MINI_BATCH_SIZE="${override#*=}"
      ;;
    rl_training_config.train_micro_batch_size=*)
      SUMMARY_TRAIN_MICRO_BATCH_SIZE="${override#*=}"
      ;;
    agentic_grpo_config.max_response_length=*)
      SUMMARY_MAX_RESPONSE_LENGTH="${override#*=}"
      ;;
    rl_training_config.actor_optimizer_config.schedule_type=*)
      SUMMARY_LR_SCHEDULE="${override#*=}"
      ;;
    rl_training_config.actor_optimizer_config.init_value=*)
      SUMMARY_LR_INIT="${override#*=}"
      ;;
    rl_training_config.actor_optimizer_config.decay_steps=*)
      SUMMARY_LR_DECAY_STEPS="${override#*=}"
      ;;
  esac
done
[[ "$SUMMARY_MINI_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "invalid effective mini_batch_size: $SUMMARY_MINI_BATCH_SIZE" >&2
  exit 2
}
[[ "$SUMMARY_TRAIN_MICRO_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "invalid effective train_micro_batch_size: $SUMMARY_TRAIN_MICRO_BATCH_SIZE" >&2
  exit 2
}
(( SUMMARY_MINI_BATCH_SIZE % SUMMARY_TRAIN_MICRO_BATCH_SIZE == 0 )) || {
  echo "mini_batch_size must be divisible by train_micro_batch_size" >&2
  exit 2
}
SUMMARY_GRADIENT_ACCUMULATION=$((
  SUMMARY_MINI_BATCH_SIZE / SUMMARY_TRAIN_MICRO_BATCH_SIZE
))

ARGS=(
  "model_config.rng_seed=${SEED}"
  "data_config.seed=$((42 + SEED))"
  require_complete_num_batches=true
  rollout_config.max_prompt_length=1024
  rollout_config.total_generation_steps=8192
  rollout_config.max_tokens_to_generate=8192
  agentic_grpo_config.num_generations=8
  agentic_grpo_config.max_response_length=8192
  agentic_grpo_config.beta=0.001
  agentic_grpo_config.loss_agg_mode=sequence-mean-token-mean
  agentic_grpo_config.degenerate_group_masking=true
  "rl_training_config.checkpointing_options.save_interval_steps=${CHECKPOINT_INTERVAL}"
  rl_training_config.checkpointing_options.max_to_keep=1
)
[[ -z "$VARIANT" ]] || ARGS+=("rl_training_config.dynamic_batch_curation_variant=${VARIANT}")

echo "Method: $METHOD"
echo "Run: $RUN_ROOT"
echo "Logs: $LOG_ROOT"
set +e
bash "${REPO}/runs_xuesong/scripts/run_official_like_dual_worker.sh" "${ARGS[@]}" "$@"
status=$?
set -e
printf '%s\n' "$status" > "${LOG_ROOT}/exit_code"
cat > "${LOG_ROOT}/run_summary.env" <<EOF
METHOD=${METHOD}
SEED=${SEED}
DATA_MODE=${DATA_MODE}
RUN_NAME=${RUN_NAME}
RUN_ROOT=${RUN_ROOT}
LOG_ROOT=${LOG_ROOT}
NUM_BATCHES=${NUM_BATCHES}
MAX_STEPS=${MAX_STEPS_OVERRIDE}
RESUME_CHECKPOINT_STEP=${RESUME_CHECKPOINT_STEP:-none}
GRADIENT_ACCUMULATION=${SUMMARY_GRADIENT_ACCUMULATION}
NUM_GENERATIONS=8
MAX_RESPONSE_LENGTH=${SUMMARY_MAX_RESPONSE_LENGTH}
BETA=0.001
DOT_THRESHOLD=0.0
LR_SCHEDULE=${SUMMARY_LR_SCHEDULE}
LR_INIT=${SUMMARY_LR_INIT}
LR_DECAY_STEPS=${SUMMARY_LR_DECAY_STEPS}
WARMUP=none
EXIT_CODE=${status}
EOF
exit "$status"
