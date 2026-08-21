#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/jason_chia925_gmail_com/Project/tunix}"
TPU_NAME="${TPU_NAME:-node-v5p-16-ziao1}"
ZONE="${ZONE:-us-central1-a}"
MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "transition" && "$MODE" != "formal" ]]; then
  echo "usage: $0 {smoke|transition|formal}" >&2
  exit 2
fi

METHOD=group_loo_policy
SEED=0
RESUME_STEP="${RESUME_STEP:-314}"
# Keep the original seed-0 data order (42 + experiment seed) unless an
# explicit continuation ablation asks for a reshuffle.
CONTINUATION_DATA_SEED="${CONTINUATION_DATA_SEED:-$((42 + SEED))}"
CONSTANT_LR="${CONSTANT_LR:-1e-6}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-2}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1024}"
RESUME_RUN_ROOT="${RESUME_RUN_ROOT:-${REPO}/runs_xuesong/runs/grpo_aime_dtv_selfinf_group_loo_policy_seed0_clean_20260805_095513}"
RESUME_ACTOR_CHECKPOINT_ROOT="${RESUME_RUN_ROOT}/checkpoints/actor"
RANKMAP_POINTER="${REPO}/runs_xuesong/logs/latest_teardown_diagnostic_root.txt"

# The TPU backend owns JAX process numbering; the VM used to invoke this
# launcher is not necessarily process 0. Reuse the rank order observed by the
# successful frozen distributed-rollout diagnostic, then reject it if its IP
# set no longer matches the TPU's current endpoints. This only supplies routing
# metadata to the existing launcher and never changes actor/rollout semantics.
[[ -r "$RANKMAP_POINTER" ]] || {
  echo "JAX rank-map pointer is missing: $RANKMAP_POINTER" >&2
  exit 1
}
RANKMAP_ROOT="$(<"$RANKMAP_POINTER")"
RANKMAP_LOG="${RANKMAP_ROOT}/launcher.nohup.log"
[[ -r "$RANKMAP_LOG" ]] || {
  echo "JAX rank-map log is missing: $RANKMAP_LOG" >&2
  exit 1
}

RANKMAP_LINE="$({
  tr -d '\000' < "$RANKMAP_LOG" |
    grep -E \
      'JAX_DISTRIBUTED_IDENTITY .*process_index=0 .*process_hosts=\[' || true
} | tail -n 1)"
[[ -n "$RANKMAP_LINE" ]] || {
  echo "No process-0 JAX identity found in: $RANKMAP_LOG" >&2
  exit 1
}

RANK0_HOST="$({
  sed -n \
    "s/.*process_hosts=\['\([^']*\)', '[^']*'\].*/\1/p" \
    <<< "$RANKMAP_LINE"
})"
RANK1_HOST="$({
  sed -n \
    "s/.*process_hosts=\['[^']*', '\([^']*\)'\].*/\1/p" \
    <<< "$RANKMAP_LINE"
})"

ipv4_regex='^([0-9]{1,3}\.){3}[0-9]{1,3}$'
[[ "$RANK0_HOST" =~ $ipv4_regex && "$RANK1_HOST" =~ $ipv4_regex ]] || {
  echo "Invalid JAX rank-map addresses: rank0=$RANK0_HOST rank1=$RANK1_HOST" >&2
  exit 1
}
[[ "$RANK0_HOST" != "$RANK1_HOST" ]] || {
  echo "JAX rank-map contains duplicate hosts: $RANK0_HOST" >&2
  exit 1
}

mapfile -t CURRENT_ENDPOINTS < <(
  gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$ZONE" \
    --format='value(networkEndpoints[].ipAddress)' |
    tr ';' '\n' |
    sed '/^[[:space:]]*$/d' |
    LC_ALL=C sort
)
mapfile -t RANKMAP_ENDPOINTS < <(
  printf '%s\n' "$RANK0_HOST" "$RANK1_HOST" |
    LC_ALL=C sort
)
[[ "${#CURRENT_ENDPOINTS[@]}" -eq 2 ]] || {
  echo "Expected two current TPU endpoints; found ${#CURRENT_ENDPOINTS[@]}" >&2
  exit 1
}
[[ "${CURRENT_ENDPOINTS[*]}" == "${RANKMAP_ENDPOINTS[*]}" ]] || {
  echo "Refusing stale JAX rank map." >&2
  echo "Current endpoints: ${CURRENT_ENDPOINTS[*]}" >&2
  echo "Rank-map endpoints: ${RANKMAP_ENDPOINTS[*]}" >&2
  exit 1
}

export PROCESS_HOSTS="${RANK0_HOST},${RANK1_HOST}"

[[ "$RESUME_STEP" =~ ^[1-9][0-9]*$ ]] || {
  echo "invalid RESUME_STEP=$RESUME_STEP" >&2
  exit 2
}
[[ "$CONTINUATION_DATA_SEED" =~ ^[0-9]+$ ]] || {
  echo "invalid CONTINUATION_DATA_SEED=$CONTINUATION_DATA_SEED" >&2
  exit 2
}
[[ -d "${RESUME_ACTOR_CHECKPOINT_ROOT}/${RESUME_STEP}" ]] || {
  echo "DTV-LOO resume checkpoint is missing: ${RESUME_ACTOR_CHECKPOINT_ROOT}/${RESUME_STEP}" >&2
  exit 1
}

case "$MODE" in
  smoke)
    # HBM depends on the per-device training microbatch and sequence length.
    # A smaller full batch reaches a real backward/update much sooner while
    # retaining DTV-LOO's eight generations per prompt and microbatch=2.
    ADDITIONAL_STEPS=1
    TARGET_STEP=$((RESUME_STEP + ADDITIONAL_STEPS))
    BATCH_SIZE="${SMOKE_BATCH_SIZE:-8}"
    MINI_BATCH_SIZE="$BATCH_SIZE"
    NUM_BATCHES=314
    NUM_TRAIN_EPOCHS=2
    CHECKPOINT_SAVE_INTERVAL=1000000
    export TUNIX_SKIP_FINAL_CHECKPOINT=1
    ;;
  transition)
    # The one-step smoke exits immediately after the first restored update and
    # cannot detect a post-update rollout freeze.  This gate deliberately runs
    # two full-size updates so it must enter the next rollout after step 315.
    ADDITIONAL_STEPS=2
    TARGET_STEP=$((RESUME_STEP + ADDITIONAL_STEPS))
    BATCH_SIZE="${TRANSITION_BATCH_SIZE:-128}"
    MINI_BATCH_SIZE="$BATCH_SIZE"
    NUM_BATCHES=314
    NUM_TRAIN_EPOCHS=2
    CHECKPOINT_SAVE_INTERVAL=1000000
    export TUNIX_SKIP_FINAL_CHECKPOINT=1
    ;;
  formal)
    ADDITIONAL_STEPS=64
    TARGET_STEP=$((RESUME_STEP + ADDITIONAL_STEPS))
    BATCH_SIZE=128
    MINI_BATCH_SIZE=128
    # The original one-pass dataset supplied 314 complete batches. Repeating
    # that deterministic shuffled dataset provides the continuation iterator;
    # the learner fast-forwards 314 batches restored from the checkpoint and
    # trains on the first 64 batches of epoch two.
    NUM_BATCHES=314
    NUM_TRAIN_EPOCHS=2
    # PeftTrainer evaluates periodic checkpoint intervals against the absolute
    # restored train step. Saving at the absolute target protects the formal
    # result from a later distributed teardown failure.
    CHECKPOINT_SAVE_INTERVAL="$TARGET_STEP"
    unset TUNIX_SKIP_FINAL_CHECKPOINT || true
    ;;
esac

(( BATCH_SIZE % 8 == 0 )) || {
  echo "BATCH_SIZE must be divisible by the eight generations per group" >&2
  exit 2
}
[[ "$TRAIN_MICRO_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "TRAIN_MICRO_BATCH_SIZE must be a positive integer" >&2
  exit 2
}
[[ "$MAX_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || {
  echo "MAX_CONCURRENCY must be a positive integer" >&2
  exit 2
}
(( MINI_BATCH_SIZE % TRAIN_MICRO_BATCH_SIZE == 0 )) || {
  echo "MINI_BATCH_SIZE must be divisible by TRAIN_MICRO_BATCH_SIZE=$TRAIN_MICRO_BATCH_SIZE" >&2
  exit 2
}
(( TARGET_STEP <= NUM_BATCHES * NUM_TRAIN_EPOCHS )) || {
  echo "dataset iterator cannot reach absolute target step $TARGET_STEP" >&2
  exit 2
}

RUN_STAMP="${TUNIX_RUN_TIMESTAMP:-extension2_16k_${MODE}_$(date -u +%Y%m%d_%H%M%S)}"
RUN_NAME="grpo_aime_dtv_selfinf_group_loo_policy_seed0_clean_${RUN_STAMP}"
RUN_ROOT="${REPO}/runs_xuesong/runs/${RUN_NAME}"
LOG_ROOT="${REPO}/runs_xuesong/logs/${RUN_NAME}"

export TUNIX_RUN_TIMESTAMP="$RUN_STAMP"
export TUNIX_RESUME_ACTOR_CHECKPOINT_ROOT="$RESUME_ACTOR_CHECKPOINT_ROOT"
export TUNIX_RESUME_CHECKPOINT_STEP="$RESUME_STEP"
export NUM_BATCHES
export MAX_STEPS_OVERRIDE="$TARGET_STEP"
export CHECKPOINT_INTERVAL="$CHECKPOINT_SAVE_INTERVAL"
export TUNIX_DISABLE_TRAJECTORY_LOGGING=1

mkdir -p "${REPO}/runs_xuesong/logs"
printf '%s\n' "$RUN_ROOT" > \
  "${REPO}/runs_xuesong/logs/latest_extension2_16k_${MODE}_root.txt"

echo "EXTENSION=2"
echo "MODE=$MODE"
echo "METHOD=dtv_loo"
echo "SEED=$SEED"
echo "RESUME_STEP=$RESUME_STEP"
echo "TARGET_STEP=$TARGET_STEP"
echo "ADDITIONAL_STEPS=$ADDITIONAL_STEPS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "MINI_BATCH_SIZE=$MINI_BATCH_SIZE"
echo "TRAIN_MICRO_BATCH_SIZE=$TRAIN_MICRO_BATCH_SIZE"
echo "MAX_CONCURRENCY=$MAX_CONCURRENCY"
echo "MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH"
echo "LR_SCHEDULE=constant_schedule"
echo "LR=$CONSTANT_LR"
echo "CONTINUATION_DATA_SEED=$CONTINUATION_DATA_SEED"
echo "CHECKPOINT_SAVE_INTERVAL=$CHECKPOINT_SAVE_INTERVAL"
echo "PROCESS_HOSTS=$PROCESS_HOSTS"
echo "ACTOR_PROCESS0_HOST=$RANK0_HOST"
echo "ROLLOUT_PROCESS1_HOST=$RANK1_HOST"
echo "RANKMAP_LOG=$RANKMAP_LOG"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"

if [[ "${TUNIX_EXTENSION2_DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; launcher configuration validated without starting JAX."
  exit 0
fi

exec "${REPO}/runs_xuesong/scripts/run_aime_seeded_full.sh" \
  "$METHOD" "$SEED" \
  "batch_size=${BATCH_SIZE}" \
  "num_train_epochs=${NUM_TRAIN_EPOCHS}" \
  train_fraction=1.0 \
  "data_config.seed=${CONTINUATION_DATA_SEED}" \
  rollout_config.max_prompt_length=1024 \
  "rollout_config.total_generation_steps=${MAX_RESPONSE_LENGTH}" \
  "rollout_config.max_tokens_to_generate=${MAX_RESPONSE_LENGTH}" \
  "agentic_grpo_config.max_response_length=${MAX_RESPONSE_LENGTH}" \
  "agentic_grpo_config.max_concurrency=${MAX_CONCURRENCY}" \
  "rl_training_config.mini_batch_size=${MINI_BATCH_SIZE}" \
  "rl_training_config.train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE}" \
  rl_training_config.actor_optimizer_config.learning_rate="$CONSTANT_LR" \
  rl_training_config.actor_optimizer_config.schedule_type=constant_schedule \
  rl_training_config.actor_optimizer_config.value="$CONSTANT_LR" \
  rl_training_config.actor_optimizer_config.init_value="$CONSTANT_LR" \
  "rl_training_config.actor_optimizer_config.decay_steps=${TARGET_STEP}" \
  "rl_training_config.checkpointing_options.save_interval_steps=${CHECKPOINT_SAVE_INTERVAL}" \
  rl_training_config.checkpointing_options.max_to_keep=2
