#!/usr/bin/env bash
set -euo pipefail

# Frozen DeepSeek-R1-Distill-Qwen-1.5B diagnostic through the exact Agentic
# GRPO distributed-vLLM rollout stack. It collects AIME G=8 trajectories and
# exits before optimizer or checkpoint code runs.

REPO="${REPO:-/home/jason_chia925_gmail_com/Project/tunix}"
TPU_NAME="${TPU_NAME:-node-v5p-16-ziao1}"
ZONE="${ZONE:-us-central1-a}"
MODEL_PATH="${MODEL_PATH:-/home/lhf_hongfu_gmail_com/models/deepseek-r1-distill-qwen-1.5b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-/home/lhf_hongfu_gmail_com/tunix-hf-data/aime_eval.parquet}"
DIAGNOSTIC_GROUPS="${DIAGNOSTIC_GROUPS:-30}"
DIAGNOSTIC_TAG="${DIAGNOSTIC_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-frozen_distributed_rollout_aime_${DIAGNOSTIC_TAG}}"

export REPO TPU_NAME ZONE MODEL_PATH TOKENIZER_PATH EVAL_DATA_PATH RUN_NAME
export RUN_ROOT="${RUN_ROOT:-${REPO}/runs_xuesong/evals/${RUN_NAME}}"
export LOG_ROOT="${LOG_ROOT:-${REPO}/runs_xuesong/logs/${RUN_NAME}}"
export CACHE_ROOT="${CACHE_ROOT:-${REPO}/runs_xuesong/cache/${RUN_NAME}}"
export RUN_SCRIPT=examples/deepscaler/run_deepscaler_disagg_v5p16_1epoch.sh
export NUM_BATCHES=1
export MAX_STEPS_OVERRIDE=1
export TUNIX_EXPERIMENT_SEED=0
export TUNIX_GRPO_NUM_GENERATIONS=8
export TUNIX_DISABLE_TRAJECTORY_LOGGING=1
export TUNIX_SKIP_FINAL_CHECKPOINT=1
export TUNIX_ROLLOUT_DIAGNOSTIC_GROUPS="${DIAGNOSTIC_GROUPS}"
export TUNIX_ROLLOUT_DIAGNOSTIC_PATH="${RUN_ROOT}/rollouts.jsonl"
export TUNIX_DATA_MANIFEST_PATH="${RUN_ROOT}/data_manifest.json"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${CACHE_ROOT}"

echo "Frozen distributed training-rollout diagnostic"
echo "  TPU=${TPU_NAME} (${ZONE})"
echo "  GROUPS=${DIAGNOSTIC_GROUPS}"
echo "  SAMPLES_PER_GROUP=8"
echo "  TOPOLOGY=TP1/DP4"
echo "  VLLM=server,async,prefix-cache"
echo "  SAMPLING=temperature0.6,top_p=model-default(null override),max8192"
echo "  OUTPUT=${TUNIX_ROLLOUT_DIAGNOSTIC_PATH}"
echo "  OPTIMIZER_UPDATE=disabled"
echo "  CHECKPOINT_WRITE=disabled"

bash "${REPO}/runs_xuesong/scripts/run_official_like_dual_worker.sh" \
  data_module=tunix.cli.recipes.aime_diagnostic_data \
  data_config.shuffle=false \
  data_config.seed=42 \
  require_complete_num_batches=true \
  batch_size="${DIAGNOSTIC_GROUPS}" \
  rollout_config.max_prompt_length=1024 \
  rollout_config.total_generation_steps=8192 \
  rollout_config.max_tokens_to_generate=8192 \
  rollout_config.temperature=0.6 \
  rollout_config.top_p=null \
  rollout_config.top_k=null \
  agentic_grpo_config.num_generations=8 \
  agentic_grpo_config.max_response_length=8192 \
  agentic_grpo_config.max_concurrency=1024 \
  agentic_grpo_config.beta=0.001 \
  rl_training_config.mini_batch_size="${DIAGNOSTIC_GROUPS}" \
  rl_training_config.train_micro_batch_size=1 \
  rl_training_config.checkpointing_options.save_interval_steps=999 \
  rl_training_config.checkpointing_options.max_to_keep=1

summary="${RUN_ROOT}/rollouts.summary.json"
echo "DIAGNOSTIC_SUMMARY=${summary}"
if [[ -f "${summary}" ]]; then
  "${REPO}/.venv/bin/python" -m json.tool "${summary}"
fi
