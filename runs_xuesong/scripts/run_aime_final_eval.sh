#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/jason_chia925_gmail_com/Project/tunix}"
VENV="${VENV:-${REPO}/.venv}"
TPU_NAME="${TPU_NAME:-node-v5p-16-ziao1}"
ZONE="${ZONE:-us-central1-a}"
DATASET="${EVAL_DATA_PATH:-/home/lhf_hongfu_gmail_com/tunix-hf-data/aime_eval.parquet}"
MODEL_PATH="${MODEL_PATH:-/home/lhf_hongfu_gmail_com/models/deepseek-r1-distill-qwen-1.5b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"

RUN_ROOT=""
OUTPUT_DIR=""
CHECKPOINT_STEP="314"
CHECKPOINT_SOURCE="checkpoint"
EVAL_SEED="2026"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --checkpoint-step) CHECKPOINT_STEP="$2"; shift 2 ;;
    --checkpoint-source) CHECKPOINT_SOURCE="$2"; shift 2 ;;
    --eval-seed) EVAL_SEED="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$RUN_ROOT" ]]; then
  echo "--run-root is required." >&2
  exit 2
fi
if [[ "$CHECKPOINT_SOURCE" != "checkpoint" && "$CHECKPOINT_SOURCE" != "base_model" ]]; then
  echo "--checkpoint-source must be checkpoint or base_model." >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${RUN_ROOT}/eval/final_actor_${CHECKPOINT_STEP}_k16_seed${EVAL_SEED}_aime_full"
fi

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

cat > "${OUTPUT_DIR}/eval_config.json" <<EOF
{
  "run_root": "${RUN_ROOT}",
  "checkpoint_source": "${CHECKPOINT_SOURCE}",
  "checkpoint_step": "${CHECKPOINT_STEP}",
  "eval_seed": ${EVAL_SEED},
  "dataset": "${DATASET}",
  "model_path": "${MODEL_PATH}",
  "tokenizer_path": "${TOKENIZER_PATH}",
  "num_problems": 30,
  "num_samples": 16,
  "max_generation_steps": 8192,
  "temperature": 0.6,
  "top_p": 0.95,
  "mesh_shape": "4,1",
  "mesh_axes": "fsdp,sp",
  "problem_batch_size": 16
}
EOF

EVAL_ARGS=(
  --run_root "$RUN_ROOT"
  --checkpoint_source "$CHECKPOINT_SOURCE"
  --checkpoint_step "$CHECKPOINT_STEP"
  --dataset "$DATASET"
  --dataset_type aime
  --output_dir "$OUTPUT_DIR"
  --model_version "$MODEL_PATH"
  --tokenizer_path "$TOKENIZER_PATH"
  --model_config deepseek_r1_distill_qwen_1p5b
  --mesh_shape 4,1
  --mesh_axes fsdp,sp
  --sampler_type vllm
  --num_samples 16
  --batch_size 1
  --problem_batch_size 16
  --eval_seed "$EVAL_SEED"
  --max_prompt_length 2048
  --max_generation_steps 8192
  --temperature 0.6
  --top_p 0.95
  --vllm_hbm_utilization 0.8
  --tpu_backend_type jax
  --max_num_seqs 16
  --max_num_batched_tokens 38400
)

printf -v EVAL_COMMAND '%q ' "${VENV}/bin/python" \
  "${REPO}/examples/deepscaler/eval_final_checkpoint_metrics.py" \
  "${EVAL_ARGS[@]}"
REMOTE_COMMAND="cd $(printf '%q' "$REPO") && export TUNIX_INIT_JAX_DISTRIBUTED=1 PYTHONUNBUFFERED=1; ${EVAL_COMMAND}"

set +e
timeout --signal=TERM --kill-after=60s 86400s \
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --worker=all \
    --internal-ip \
    --command="bash -lc $(printf '%q' "$REMOTE_COMMAND")" \
    2>&1 | tee "${LOG_DIR}/launcher.log"
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -eq 0 ]]; then
  if [[ ! -f "${OUTPUT_DIR}/summary.json" || ! -f "${OUTPUT_DIR}/.primary_done" ]]; then
    status=1
  elif [[ "$(wc -l < "${OUTPUT_DIR}/samples.jsonl")" -ne 480 ]]; then
    status=1
  fi
fi

cat > "${LOG_DIR}/launcher.status" <<EOF
EXIT_CODE=${status}
FINISHED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
SUMMARY_PRESENT=$([[ -f "${OUTPUT_DIR}/summary.json" ]] && echo 1 || echo 0)
SAMPLES_PRESENT=$([[ -f "${OUTPUT_DIR}/samples.jsonl" ]] && echo 1 || echo 0)
SAMPLE_COUNT=$([[ -f "${OUTPUT_DIR}/samples.jsonl" ]] && wc -l < "${OUTPUT_DIR}/samples.jsonl" || echo 0)
EOF

echo "EVAL_OUTPUT_DIR=${OUTPUT_DIR}"
echo "EVAL_EXIT_CODE=${status}"
exit "$status"
