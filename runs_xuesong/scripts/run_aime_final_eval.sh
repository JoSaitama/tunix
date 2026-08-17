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
BASE_MODEL_LOAD_MODE="nnx_sync"
EVAL_SEED="2026"
LIMIT=""
NUM_SAMPLES="16"
MAX_GENERATION_STEPS="8192"
PROBLEM_BATCH_SIZE="16"
MAX_NUM_SEQS="16"
MAX_NUM_BATCHED_TOKENS="38400"
TENSOR_PARALLEL_SIZE="-1"
DATA_PARALLEL_SIZE="-1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --checkpoint-step) CHECKPOINT_STEP="$2"; shift 2 ;;
    --checkpoint-source) CHECKPOINT_SOURCE="$2"; shift 2 ;;
    --base-model-load-mode) BASE_MODEL_LOAD_MODE="$2"; shift 2 ;;
    --eval-seed) EVAL_SEED="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
    --max-generation-steps) MAX_GENERATION_STEPS="$2"; shift 2 ;;
    --problem-batch-size) PROBLEM_BATCH_SIZE="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
    --tensor-parallel-size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --data-parallel-size) DATA_PARALLEL_SIZE="$2"; shift 2 ;;
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
if [[ "$BASE_MODEL_LOAD_MODE" != "nnx_sync" && "$BASE_MODEL_LOAD_MODE" != "direct_vllm" ]]; then
  echo "--base-model-load-mode must be nnx_sync or direct_vllm" >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${RUN_ROOT}/eval/final_actor_${CHECKPOINT_STEP}_k${NUM_SAMPLES}_seed${EVAL_SEED}_aime_full"
fi

EXPECTED_PROBLEMS="${LIMIT:-30}"
EXPECTED_SAMPLES=$((EXPECTED_PROBLEMS * NUM_SAMPLES))

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

cat > "${OUTPUT_DIR}/eval_config.json" <<EOF
{
  "run_root": "${RUN_ROOT}",
  "checkpoint_source": "${CHECKPOINT_SOURCE}",
  "base_model_load_mode": "${BASE_MODEL_LOAD_MODE}",
  "checkpoint_step": "${CHECKPOINT_STEP}",
  "eval_seed": ${EVAL_SEED},
  "dataset": "${DATASET}",
  "model_path": "${MODEL_PATH}",
  "tokenizer_path": "${TOKENIZER_PATH}",
  "num_problems": ${EXPECTED_PROBLEMS},
  "num_samples": ${NUM_SAMPLES},
  "max_generation_steps": ${MAX_GENERATION_STEPS},
  "temperature": 0.6,
  "top_p": 0.95,
  "mesh_shape": "4,1",
  "mesh_axes": "fsdp,tp",
  "problem_batch_size": ${PROBLEM_BATCH_SIZE},
  "max_num_seqs": ${MAX_NUM_SEQS},
  "max_num_batched_tokens": ${MAX_NUM_BATCHED_TOKENS},
  "tensor_parallel_size": ${TENSOR_PARALLEL_SIZE},
  "data_parallel_size": ${DATA_PARALLEL_SIZE}
}
EOF

EVAL_ARGS=(
  --run_root "$RUN_ROOT"
  --checkpoint_source "$CHECKPOINT_SOURCE"
  --base_model_load_mode "$BASE_MODEL_LOAD_MODE"
  --checkpoint_step "$CHECKPOINT_STEP"
  --dataset "$DATASET"
  --dataset_type aime
  --output_dir "$OUTPUT_DIR"
  --model_version "$MODEL_PATH"
  --tokenizer_path "$TOKENIZER_PATH"
  --model_config deepseek_r1_distill_qwen_1p5b
  --mesh_shape 4,1
  --mesh_axes fsdp,tp
  --sampler_type vllm
  --num_samples "$NUM_SAMPLES"
  --batch_size 1
  --problem_batch_size "$PROBLEM_BATCH_SIZE"
  --eval_seed "$EVAL_SEED"
  --max_prompt_length 2048
  --max_generation_steps "$MAX_GENERATION_STEPS"
  --temperature 0.6
  --top_p 0.95
  --vllm_hbm_utilization 0.8
  --tpu_backend_type jax
  --max_num_seqs "$MAX_NUM_SEQS"
  --max_num_batched_tokens "$MAX_NUM_BATCHED_TOKENS"
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
  --data_parallel_size "$DATA_PARALLEL_SIZE"
)

if [[ -n "$LIMIT" ]]; then
  EVAL_ARGS+=(--limit "$LIMIT")
fi

printf -v EVAL_COMMAND '%q ' "${VENV}/bin/python" \
  "${REPO}/examples/deepscaler/eval_final_checkpoint_metrics.py" \
  "${EVAL_ARGS[@]}"
WORKER_COMMAND="cd $(printf '%q' "$REPO") && export TUNIX_INIT_JAX_DISTRIBUTED=1 PYTHONUNBUFFERED=1; timeout --signal=TERM --kill-after=60s 86400s ${EVAL_COMMAND}"
WORKER_LOG_DIR="${LOG_DIR}/workers"
mkdir -p "$WORKER_LOG_DIR"

# Worker 0 is the directly connected host. Run it locally so output files are
# owned by the invoking user. Only Worker 1 is launched through gcloud, whose
# SSH identity may be a service account with no write access to OUTPUT_DIR.
REMOTE_SCRIPT="${WORKER_COMMAND}; worker_status=\$?; echo HOST=\$(hostname) STATUS=\${worker_status}; exit \${worker_status}"

set +e
gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" \
  --worker=1 \
  --internal-ip \
  --command="bash -lc $(printf '%q' "$REMOTE_SCRIPT")" \
  > "${WORKER_LOG_DIR}/remote.log" 2>&1 &
remote_launcher_pid=$!

sleep 2
bash -lc "$WORKER_COMMAND" \
  2>&1 | tee "${WORKER_LOG_DIR}/local.log"
local_status=${PIPESTATUS[0]}

wait "$remote_launcher_pid"
remote_ssh_status=$?
set -e

remote_status="$(
  sed -n 's/^HOST=.* STATUS=\([0-9][0-9]*\)$/\1/p' \
    "${WORKER_LOG_DIR}/remote.log" | tail -1
)"
if [[ -z "$remote_status" ]]; then
  remote_status="$remote_ssh_status"
fi

status=0
if [[ "$local_status" -ne 0 || "$remote_status" -ne 0 ]]; then
  status=1
fi

cat "${WORKER_LOG_DIR}/local.log" \
  "${WORKER_LOG_DIR}/remote.log" > "${LOG_DIR}/launcher.log"

if [[ "$status" -eq 0 ]]; then
  if [[ ! -f "${OUTPUT_DIR}/summary.json" || ! -f "${OUTPUT_DIR}/.primary_done" ]]; then
    status=1
  elif [[ "$(wc -l < "${OUTPUT_DIR}/samples.jsonl")" -ne "$EXPECTED_SAMPLES" ]]; then
    status=1
  fi
fi

cat > "${LOG_DIR}/launcher.status" <<EOF
EXIT_CODE=${status}
LOCAL_STATUS=${local_status}
REMOTE_SSH_STATUS=${remote_ssh_status}
REMOTE_STATUS=${remote_status}
FINISHED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
SUMMARY_PRESENT=$([[ -f "${OUTPUT_DIR}/summary.json" ]] && echo 1 || echo 0)
SAMPLES_PRESENT=$([[ -f "${OUTPUT_DIR}/samples.jsonl" ]] && echo 1 || echo 0)
SAMPLE_COUNT=$([[ -f "${OUTPUT_DIR}/samples.jsonl" ]] && wc -l < "${OUTPUT_DIR}/samples.jsonl" || echo 0)
EOF

echo "EVAL_OUTPUT_DIR=${OUTPUT_DIR}"
echo "EVAL_EXIT_CODE=${status}"
exit "$status"
