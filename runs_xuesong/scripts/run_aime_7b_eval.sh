#!/usr/bin/env bash
# Use the shared evaluator with the explicit DeepSeek 7B architecture.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to the DeepSeek-R1-Distill-Qwen-7B directory on both workers}"
export MODEL_PATH
export TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
exec bash "${script_dir}/run_aime_final_eval.sh" \
  --model-config deepseek_r1_distill_qwen_7b "$@"
