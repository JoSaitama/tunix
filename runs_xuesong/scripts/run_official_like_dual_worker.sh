#!/usr/bin/env bash
set -euo pipefail

launcher_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=dual_worker_status.sh
source "${launcher_dir}/dual_worker_status.sh"

REPO="${REPO:-/home/jason_chia925_gmail_com/Project/tunix}"
VENV="${VENV:-$REPO/.venv}"
TPU_NAME="${TPU_NAME:-ziao-v5p16-flex7d-1-node}"
ZONE="${ZONE:-us-east5-a}"
REMOTE_WORKER_INDEX="${REMOTE_WORKER_INDEX:-1}"

RUN_NAME="${RUN_NAME:-official-like}"
RUN_SCRIPT="${RUN_SCRIPT:-examples/deepscaler/run_deepscaler_disagg_v5p16.sh}"
ROLLOUT_ENGINE="${ROLLOUT_ENGINE:-vllm}"
MESH_MODE="${MESH_MODE:-disagg}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/home/lhf_hongfu_gmail_com/tunix-hf-data/deepscaler_train.json}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-/home/lhf_hongfu_gmail_com/tunix-hf-data/aime_eval.parquet}"
MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
MODEL_PATH="${MODEL_PATH:-/home/lhf_hongfu_gmail_com/models/deepseek-r1-distill-qwen-1.5b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
HF_TOKEN_VALUE="${HF_TOKEN_VALUE:-dummy}"
MAX_STEPS_OVERRIDE="${MAX_STEPS_OVERRIDE:-}"
NUM_BATCHES="${NUM_BATCHES:-}"
ROLLOUT_VLLM_INIT_WITH_RANDOM_WEIGHTS="${ROLLOUT_VLLM_INIT_WITH_RANDOM_WEIGHTS:-}"
TUNIX_SKIP_FINAL_CHECKPOINT="${TUNIX_SKIP_FINAL_CHECKPOINT:-}"
TUNIX_DISABLE_TRAJECTORY_LOGGING="${TUNIX_DISABLE_TRAJECTORY_LOGGING:-}"
TUNIX_EXPERIMENT_SEED="${TUNIX_EXPERIMENT_SEED:-0}"
TUNIX_REWARD_RANK_NOISE_FRACTION="${TUNIX_REWARD_RANK_NOISE_FRACTION:-0}"
TUNIX_REWARD_RANK_NOISE_SEED="${TUNIX_REWARD_RANK_NOISE_SEED:-0}"
TUNIX_FIXED_FILTER_METHOD="${TUNIX_FIXED_FILTER_METHOD:-}"
TUNIX_FIXED_FILTER_SCOPE="${TUNIX_FIXED_FILTER_SCOPE:-}"
TUNIX_FIXED_FILTER_RATIO="${TUNIX_FIXED_FILTER_RATIO:-}"
TUNIX_GRPO_NUM_GENERATIONS="${TUNIX_GRPO_NUM_GENERATIONS:-8}"
TUNIX_DBC_SELF_INF_MIN_KEEP_FRACTION="${TUNIX_DBC_SELF_INF_MIN_KEEP_FRACTION:-0.25}"
TUNIX_DBC_DECISIONS_PATH="${TUNIX_DBC_DECISIONS_PATH:-}"
TUNIX_POLICY_DTV_SCORE_BACKEND="${TUNIX_POLICY_DTV_SCORE_BACKEND:-vmap}"

RUN_ROOT="${RUN_ROOT:-${REPO}/runs_xuesong/runs/${RUN_NAME}}"
CACHE_ROOT="${CACHE_ROOT:-${REPO}/runs_xuesong/cache/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${REPO}/runs_xuesong/logs/${RUN_NAME}}"
LOG_DIR="${LOG_ROOT}/workers"
TB_DIR="${LOG_ROOT}/tensorboard"
CKPT_DIR="${RUN_ROOT}/checkpoints"
TMP_DIR="${CACHE_ROOT}/tmp"

for required_file in \
  "${TRAIN_DATA_PATH}" \
  "${EVAL_DATA_PATH}" \
  "${MODEL_PATH}/config.json" \
  "${TOKENIZER_PATH}/tokenizer.json"; do
  if [[ ! -r "${required_file}" ]]; then
    echo "Required read-only input is not readable: ${required_file}" >&2
    exit 1
  fi
done
if ! compgen -G "${MODEL_PATH}/*.safetensors" >/dev/null && \
   ! compgen -G "${MODEL_PATH}/*.safetensors.index.json" >/dev/null; then
  echo "No safetensors weights found under MODEL_PATH=${MODEL_PATH}." >&2
  exit 1
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Python environment is missing: ${VENV}/bin/python" >&2
  exit 1
fi

mapfile -t endpoint_ips < <(
  gcloud alpha compute tpus tpu-vm describe "${TPU_NAME}" \
    --zone="${ZONE}" \
    --format='value(networkEndpoints[].ipAddress)' | tr ';' '\n'
)
if [[ "${#endpoint_ips[@]}" -ne 2 ]]; then
  echo "Expected exactly two TPU worker IPs for ${TPU_NAME}; found ${#endpoint_ips[@]}." >&2
  exit 1
fi

LOCAL_HOST="${LOCAL_HOST:-${endpoint_ips[0]}}"
REMOTE_HOST="${REMOTE_HOST:-${endpoint_ips[${REMOTE_WORKER_INDEX}]}}"
PROCESS_HOSTS="${PROCESS_HOSTS:-$(IFS=,; echo "${endpoint_ips[*]}")}"

mkdir -p "${LOG_DIR}" "${TB_DIR}" "${CKPT_DIR}" "${TMP_DIR}"

base_cmd=(
  bash "${RUN_SCRIPT}"
  "data_config.train_data_path=${TRAIN_DATA_PATH}"
  "data_config.eval_data_path=${EVAL_DATA_PATH}"
  "model_config.model_id=${MODEL_ID}"
  "model_config.model_download_path=${MODEL_PATH}"
  "actor_model_config.model_download_path=${MODEL_PATH}"
  "reference_model_config.model_download_path=${MODEL_PATH}"
  "rollout_model_config.model_download_path=${MODEL_PATH}"
  "tokenizer_config.tokenizer_path=${TOKENIZER_PATH}"
  "vllm_config.model_version=${MODEL_PATH}"
  "rollout_engine=${ROLLOUT_ENGINE}"
  "rl_training_config.metrics_logging_options.log_dir=${TB_DIR}"
  "rl_training_config.checkpoint_root_directory=${CKPT_DIR}"
)

if [[ -n "${MAX_STEPS_OVERRIDE}" ]]; then
  base_cmd+=("rl_training_config.max_steps=${MAX_STEPS_OVERRIDE}")
fi

if [[ -n "${ROLLOUT_VLLM_INIT_WITH_RANDOM_WEIGHTS}" ]]; then
  base_cmd+=(
    "rollout_config.rollout_vllm_init_with_random_weights=${ROLLOUT_VLLM_INIT_WITH_RANDOM_WEIGHTS}"
  )
fi

case "${MESH_MODE}" in
  disagg)
    ;;
  same-mesh)
    base_cmd+=(
      "actor_model_config.mesh.shape=(8,1)"
      "actor_model_config.mesh.axis_names=('fsdp','tp')"
      "reference_model_config.mesh=null"
      "reference_model_config.same_mesh_as=actor"
      "rollout_model_config.mesh=null"
      "rollout_model_config.same_mesh_as=actor"
    )
    ;;
  *)
    echo "Unsupported MESH_MODE=${MESH_MODE}" >&2
    exit 1
    ;;
esac

for arg in "$@"; do
  base_cmd+=("${arg}")
done

cmd_string="$(printf '%q ' "${base_cmd[@]}")"

echo "Resolved dual-worker launch:"
echo "  REPO=${REPO}"
echo "  VENV=${VENV}"
echo "  TPU=${TPU_NAME} (${ZONE})"
echo "  RUN_SCRIPT=${RUN_SCRIPT}"
echo "  RUN_ROOT=${RUN_ROOT}"
echo "  LOG_ROOT=${LOG_ROOT}"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  TRAIN_DATA_PATH=${TRAIN_DATA_PATH}"
echo "  EVAL_DATA_PATH=${EVAL_DATA_PATH}"
echo "  NUM_BATCHES=${NUM_BATCHES:-<launcher-default>}"
echo "  COMMAND=${cmd_string}"

run_worker() (
  local logfile="$1"
  echo "=== WORKER 0 LOG: $(hostname) ==="
  cd "${REPO}"
  source "${VENV}/bin/activate"
  mkdir -p "${LOG_DIR}" "${TB_DIR}" "${CKPT_DIR}" "${TMP_DIR}"
  export HF_TOKEN="${HF_TOKEN_VALUE}"
  export HF_HOME="${CACHE_ROOT}/huggingface"
  export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
  export TMPDIR="${TMP_DIR}"
  export PYTHONDONTWRITEBYTECODE=1
  export TUNIX_INIT_JAX_DISTRIBUTED=1
  export TUNIX_PROCESS_HOSTS="${PROCESS_HOSTS}"
  export SKIP_JAX_PRECOMPILE=true
  export TUNIX_EXPERIMENT_SEED TUNIX_REWARD_RANK_NOISE_FRACTION
  export TUNIX_REWARD_RANK_NOISE_SEED TUNIX_GRPO_NUM_GENERATIONS
  export TUNIX_DBC_SELF_INF_MIN_KEEP_FRACTION TUNIX_DBC_DECISIONS_PATH
  export TUNIX_POLICY_DTV_SCORE_BACKEND
  export TUNIX_FIXED_FILTER_METHOD TUNIX_FIXED_FILTER_SCOPE
  export TUNIX_FIXED_FILTER_RATIO
  if [[ -n "${NUM_BATCHES}" ]]; then
    export num_batches="${NUM_BATCHES}"
  fi
  if [[ -n "${TUNIX_SKIP_FINAL_CHECKPOINT}" ]]; then
    export TUNIX_SKIP_FINAL_CHECKPOINT
  fi
  if [[ -n "${TUNIX_DISABLE_TRAJECTORY_LOGGING}" ]]; then
    export TUNIX_DISABLE_TRAJECTORY_LOGGING
  fi
  status=0
  set +e
  bash -lc "${cmd_string}" 2>&1 | tee "${logfile}"
  status=${PIPESTATUS[0]}
  set -e
  echo "HOST=$(hostname) STATUS=${status}"
  return "${status}"
)

run_remote() {
  local logfile="$1"
  local remote_script
  remote_script=$(cat <<EOF
set -euo pipefail
echo "=== WORKER 1 LOG: \$(hostname) ==="
cd ${REPO}
source ${VENV}/bin/activate
mkdir -p ${LOG_DIR} ${TB_DIR} ${CKPT_DIR} ${TMP_DIR}
export HF_TOKEN=${HF_TOKEN_VALUE}
export HF_HOME=${CACHE_ROOT}/huggingface
export XDG_CACHE_HOME=${CACHE_ROOT}/xdg
export TMPDIR=${TMP_DIR}
export PYTHONDONTWRITEBYTECODE=1
export TUNIX_INIT_JAX_DISTRIBUTED=1
export TUNIX_PROCESS_HOSTS=${PROCESS_HOSTS}
export SKIP_JAX_PRECOMPILE=true
export TUNIX_EXPERIMENT_SEED=${TUNIX_EXPERIMENT_SEED@Q}
export TUNIX_REWARD_RANK_NOISE_FRACTION=${TUNIX_REWARD_RANK_NOISE_FRACTION@Q}
export TUNIX_REWARD_RANK_NOISE_SEED=${TUNIX_REWARD_RANK_NOISE_SEED@Q}
export TUNIX_FIXED_FILTER_METHOD=${TUNIX_FIXED_FILTER_METHOD@Q}
export TUNIX_FIXED_FILTER_SCOPE=${TUNIX_FIXED_FILTER_SCOPE@Q}
export TUNIX_FIXED_FILTER_RATIO=${TUNIX_FIXED_FILTER_RATIO@Q}
export TUNIX_GRPO_NUM_GENERATIONS=${TUNIX_GRPO_NUM_GENERATIONS@Q}
export TUNIX_DBC_SELF_INF_MIN_KEEP_FRACTION=${TUNIX_DBC_SELF_INF_MIN_KEEP_FRACTION@Q}
export TUNIX_DBC_DECISIONS_PATH=${TUNIX_DBC_DECISIONS_PATH@Q}
export TUNIX_POLICY_DTV_SCORE_BACKEND=${TUNIX_POLICY_DTV_SCORE_BACKEND@Q}
if [[ -n ${NUM_BATCHES@Q} ]]; then
  export num_batches=${NUM_BATCHES@Q}
fi
if [[ -n ${TUNIX_SKIP_FINAL_CHECKPOINT@Q} ]]; then
  export TUNIX_SKIP_FINAL_CHECKPOINT=${TUNIX_SKIP_FINAL_CHECKPOINT@Q}
fi
if [[ -n ${TUNIX_DISABLE_TRAJECTORY_LOGGING@Q} ]]; then
  export TUNIX_DISABLE_TRAJECTORY_LOGGING=${TUNIX_DISABLE_TRAJECTORY_LOGGING@Q}
fi
status=0
set +e
bash -lc ${cmd_string@Q} 2>&1 | tee ${logfile@Q}
status=\${PIPESTATUS[0]}
set -e
echo HOST=\$(hostname) STATUS=\${status}
exit \${status}
EOF
)
  gcloud alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone="${ZONE}" \
    --internal-ip \
    --worker="${REMOTE_WORKER_INDEX}" \
    --command="bash -lc $(printf '%q' "${remote_script}")"
}

remote_log="${LOG_DIR}/remote.log"
local_log="${LOG_DIR}/local.log"
remote_status_log="${LOG_DIR}/remote.status"

rm -f "${remote_status_log}"

run_remote "${remote_log}" 2>&1 |
  tee /dev/stderr |
  sed -n '/HOST=.*STATUS=/p' > "${remote_status_log}" &
remote_pid=$!

sleep 2

if run_worker "${local_log}"; then
  local_status=0
else
  local_status=$?
fi
tunix_wait_and_capture_status remote_ssh_status "${remote_pid}"
tunix_resolve_remote_status \
  remote_status "${remote_status_log}" "${remote_ssh_status}"

echo "LOCAL_STATUS=${local_status}"
echo "REMOTE_SSH_STATUS=${remote_ssh_status}"
echo "REMOTE_STATUS=${remote_status}"

if [[ "${local_status}" -ne 0 || "${remote_status}" -ne 0 ]]; then
  exit 1
fi
