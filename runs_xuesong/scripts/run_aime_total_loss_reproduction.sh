#!/usr/bin/env bash
# Launch the matched AIME baseline or the original total-loss DTV variants.

set -euo pipefail

METHOD="${METHOD:-}"
NUM_BATCHES="${NUM_BATCHES:-}"
SEED="${SEED:-42}"
SELF_INFLUENCE_DOT_THRESHOLD="${SELF_INFLUENCE_DOT_THRESHOLD:-0.0}"

if [[ -z "${METHOD}" ]]; then
  echo "Set METHOD to baseline, dtv_batch_total_loss, or dtv_group_total_loss." >&2
  exit 1
fi
if [[ -z "${NUM_BATCHES}" || ! "${NUM_BATCHES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Set NUM_BATCHES to a positive integer; use 1 for the first smoke." >&2
  exit 1
fi

case "${METHOD}" in
  baseline)
    run_script="examples/deepscaler/run_deepscaler_disagg_v5p16_1epoch.sh"
    method_args=()
    ;;
  dtv_batch_total_loss)
    run_script="examples/deepscaler/run_deepscaler_disagg_v5p16_selfinf_batch_1epoch.sh"
    method_args=(
      "rl_training_config.self_influence_dot_threshold=${SELF_INFLUENCE_DOT_THRESHOLD}"
    )
    ;;
  dtv_group_total_loss)
    run_script="examples/deepscaler/run_deepscaler_disagg_v5p16_selfinf_group_1epoch.sh"
    method_args=(
      "rl_training_config.self_influence_dot_threshold=${SELF_INFLUENCE_DOT_THRESHOLD}"
    )
    ;;
  *)
    echo "Unsupported METHOD=${METHOD}." >&2
    exit 1
    ;;
esac

timestamp="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_NAME="${RUN_NAME:-grpo_aime_${METHOD}_seed${SEED}_${NUM_BATCHES}batch_${timestamp}}"
REPO="${REPO:-/home/jason_chia925_gmail_com/Project/tunix}"
RUN_ROOT="${RUN_ROOT:-${REPO}/runs_xuesong/runs/${RUN_NAME}}"
MAX_STEPS_OVERRIDE="${MAX_STEPS_OVERRIDE:-${NUM_BATCHES}}"
if [[ "${NUM_BATCHES}" -le 64 ]]; then
  default_checkpoint_interval="${NUM_BATCHES}"
else
  default_checkpoint_interval=500
fi
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-${default_checkpoint_interval}}"

export REPO RUN_NAME RUN_ROOT RUN_SCRIPT="${run_script}"
export NUM_BATCHES MAX_STEPS_OVERRIDE
export TUNIX_DISABLE_TRAJECTORY_LOGGING="${TUNIX_DISABLE_TRAJECTORY_LOGGING:-1}"

exec bash "${REPO}/runs_xuesong/scripts/run_official_like_dual_worker.sh" \
  model_config.rng_seed="${SEED}" \
  data_config.seed="${SEED}" \
  rollout_config.max_prompt_length=1024 \
  rollout_config.total_generation_steps=8192 \
  rollout_config.max_tokens_to_generate=8192 \
  agentic_grpo_config.num_generations=8 \
  agentic_grpo_config.max_response_length=8192 \
  agentic_grpo_config.beta=0.001 \
  agentic_grpo_config.loss_agg_mode=sequence-mean-token-mean \
  agentic_grpo_config.degenerate_group_masking=true \
  rl_training_config.checkpointing_options.save_interval_steps="${CHECKPOINT_INTERVAL}" \
  rl_training_config.checkpointing_options.max_to_keep=1 \
  "${method_args[@]}" \
  "$@"
