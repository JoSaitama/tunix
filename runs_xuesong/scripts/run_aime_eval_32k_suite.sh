#!/usr/bin/env bash
set -euo pipefail

# Formal 32K comparison preset. Every protocol-defining value is exported
# explicitly; callers may still override model selection, paths, and resources.
REPO="${REPO:-/home/jason_chia925_gmail_com/Project/tunix}"

export EVAL_PROTOCOL_NAME="${EVAL_PROTOCOL_NAME:-aime2024_32k_nnx_offline}"
export EVAL_BASE_MODEL_LOAD_MODE="${EVAL_BASE_MODEL_LOAD_MODE:-nnx_sync}"
export EVAL_CHECKPOINT_STEP="${EVAL_CHECKPOINT_STEP:-314}"
export EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-16}"
export EVAL_MAX_PROMPT_LENGTH="${EVAL_MAX_PROMPT_LENGTH:-2048}"
export EVAL_MAX_GENERATION_STEPS="${EVAL_MAX_GENERATION_STEPS:-32768}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.6}"
export EVAL_TOP_P="${EVAL_TOP_P:-0.95}"
export EVAL_AIME_PROMPT_STYLE="${EVAL_AIME_PROMPT_STYLE:-legacy}"
export EVAL_PROBLEM_BATCH_SIZE="${EVAL_PROBLEM_BATCH_SIZE:-4}"
export EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-4}"
export EVAL_MAX_NUM_BATCHED_TOKENS="${EVAL_MAX_NUM_BATCHED_TOKENS:-38400}"
export EVAL_TENSOR_PARALLEL_SIZE="${EVAL_TENSOR_PARALLEL_SIZE:-4}"
export EVAL_DATA_PARALLEL_SIZE="${EVAL_DATA_PARALLEL_SIZE:-1}"
export EVAL_VLLM_HBM_UTILIZATION="${EVAL_VLLM_HBM_UTILIZATION:-0.8}"
export EVAL_VLLM_SERVER_MODE="${EVAL_VLLM_SERVER_MODE:-false}"
export EVAL_VLLM_ASYNC_SCHEDULING="${EVAL_VLLM_ASYNC_SCHEDULING:-false}"
export EVAL_VLLM_ENABLE_PREFIX_CACHING="${EVAL_VLLM_ENABLE_PREFIX_CACHING:-false}"
export EVAL_MODELS="${EVAL_MODELS:-pretrain baseline dtv dtv_loo}"
export SUITE_ROOT="${SUITE_ROOT:-${REPO}/runs_xuesong/evals/aime2024_32k_k16_seed0_nnx_offline}"

if [[ $# -eq 0 ]]; then
  set -- 0
fi

exec "${REPO}/runs_xuesong/scripts/run_aime_eval_seed_suite.sh" "$@"
