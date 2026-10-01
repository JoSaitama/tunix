#!/usr/bin/env bash
# Full-parameter 7B preset sharing the existing seeded AIME training pipeline.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 2 ]]; then
  echo "usage: $0 {baseline|group_policy|group_loo_policy} SEED [config overrides...]" >&2
  exit 2
fi
case "$1" in
  baseline|group_policy|group_loo_policy) ;;
  *) echo "Unsupported 7B method: $1" >&2; exit 2 ;;
esac
method="$1"; seed="$2"; shift 2

# Require an explicit 7B cache rather than inheriting the old 1.5B path.
: "${MODEL_PATH:?Set MODEL_PATH to the DeepSeek-R1-Distill-Qwen-7B directory on both workers}"
export MODEL_PATH
export MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
export AIME_RUN_NAME_PREFIX=grpo_aime_ds7b_full
# Start with a small real-update gate, not a full training run.
export NUM_BATCHES="${NUM_BATCHES:-2}"

exec bash "${script_dir}/run_aime_seeded_full.sh" "$method" "$seed" \
  model_config.model_name=deepseek_r1_distill_qwen_7b \
  model_config.model_id="$MODEL_ID" \
  model_config.lora_config={} \
  actor_model_config.lora_config={} \
  reference_model_config.lora_config={} \
  rollout_model_config.lora_config={} \
  batch_size=16 \
  rl_training_config.mini_batch_size=16 \
  rl_training_config.train_micro_batch_size=1 \
  rl_training_config.actor_optimizer_config.decay_steps=314 \
  vllm_config.max_num_seqs=16 \
  agentic_grpo_config.max_concurrency=16 \
  "$@"
