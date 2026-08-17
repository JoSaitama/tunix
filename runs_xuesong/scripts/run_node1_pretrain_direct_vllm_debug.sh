#!/usr/bin/env bash
set -euo pipefail

# Highest-information frozen-base diagnostic for the AIME regression:
# vLLM loads the local Hugging Face safetensors directly, bypassing the
# NNX/JAX -> vLLM update_params mapping used by the existing evaluator.

REPO="${REPO:-/home/jason_chia925_gmail_com/Project/tunix}"
TPU_NAME="${TPU_NAME:-node-v5p-16-ziao1}"
ZONE="${ZONE:-us-central1-a}"
MODEL_PATH="${MODEL_PATH:-/home/lhf_hongfu_gmail_com/models/deepseek-r1-distill-qwen-1.5b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-/home/lhf_hongfu_gmail_com/tunix-hf-data/aime_eval.parquet}"
DEBUG_TAG="${DEBUG_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
DEBUG_ROOT="${DEBUG_ROOT:-${REPO}/runs_xuesong/evals/pretrain_direct_vllm_debug_${DEBUG_TAG}}"
LIMIT="${LIMIT:-30}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"

export REPO TPU_NAME ZONE MODEL_PATH TOKENIZER_PATH EVAL_DATA_PATH

mkdir -p "${DEBUG_ROOT}"

echo "Frozen-base direct-vLLM diagnostic"
echo "  TPU_NAME=${TPU_NAME}"
echo "  ZONE=${ZONE}"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "  EVAL_DATA_PATH=${EVAL_DATA_PATH}"
echo "  LIMIT=${LIMIT}"
echo "  NUM_SAMPLES=${NUM_SAMPLES}"
echo "  OUTPUT=${DEBUG_ROOT}"

bash "${REPO}/runs_xuesong/scripts/run_aime_final_eval.sh" \
  --run-root "${DEBUG_ROOT}" \
  --output-dir "${DEBUG_ROOT}/direct_vllm_k${NUM_SAMPLES}" \
  --checkpoint-source base_model \
  --base-model-load-mode direct_vllm \
  --checkpoint-step 0 \
  --eval-seed 2026 \
  --limit "${LIMIT}" \
  --num-samples "${NUM_SAMPLES}" \
  --max-generation-steps 8192 \
  --problem-batch-size 16 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 38400

summary="${DEBUG_ROOT}/direct_vllm_k${NUM_SAMPLES}/summary.json"
echo "DIRECT_VLLM_SUMMARY=${summary}"
if [[ -f "${summary}" ]]; then
  "${REPO}/.venv/bin/python" - "${summary}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
  value = json.load(handle)

metrics = value.get("metrics", {})
for key in (
    f"avg@{metrics.get('k', 4)}",
    f"pass@{metrics.get('k', 4)}",
    f"maj@{metrics.get('k', 4)}",
    "truncation_rate",
    "extractable_answer_rate",
    "boxed_answer_rate",
    "avg_tokens",
    "correct_count",
):
  if key in metrics:
    print(f"{key}={metrics[key]}")
print(json.dumps(value, indent=2, sort_keys=True))
PY
fi
