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
export TUNIX_EXPERIMENT_SEED="$SEED"
export TUNIX_REWARD_RANK_NOISE_SEED="${TUNIX_REWARD_RANK_NOISE_SEED:-$SEED}"
export TUNIX_GRPO_NUM_GENERATIONS=8
export TUNIX_DBC_DECISIONS_PATH="${LOG_ROOT}/selection.jsonl"
mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$CACHE_ROOT"
exec > >(tee "${LOG_ROOT}/nohup.log") 2>&1

ARGS=(
  "model_config.rng_seed=${SEED}"
  "data_config.seed=$((42 + SEED))"
  rollout_config.max_prompt_length=1024
  rollout_config.total_generation_steps=8192
  rollout_config.max_tokens_to_generate=8192
  agentic_grpo_config.num_generations=8
  agentic_grpo_config.max_response_length=8192
  agentic_grpo_config.beta=0.001
  agentic_grpo_config.loss_agg_mode=sequence-mean-token-mean
  agentic_grpo_config.degenerate_group_masking=true
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
exit "$status"
