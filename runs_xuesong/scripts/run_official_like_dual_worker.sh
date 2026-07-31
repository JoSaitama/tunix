#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/eve/tunix}"
VENV="${VENV:-$REPO/.venv}"
TPU_NAME="${TPU_NAME:-ziao-v5p16-flex7d-1-node}"
ZONE="${ZONE:-us-east5-a}"
REMOTE_WORKER_INDEX="${REMOTE_WORKER_INDEX:-1}"

RUN_NAME="${RUN_NAME:-official-like}"
RUN_SCRIPT="${RUN_SCRIPT:-examples/deepscaler/run_deepscaler_disagg_v5p16.sh}"
ROLLOUT_ENGINE="${ROLLOUT_ENGINE:-vllm}"
MESH_MODE="${MESH_MODE:-disagg}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/home/eve/tunix-hf-data/deepscaler_train.json}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-/home/eve/tunix-hf-data/aime_eval.parquet}"
MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
HF_TOKEN_VALUE="${HF_TOKEN_VALUE:-dummy}"
MAX_STEPS_OVERRIDE="${MAX_STEPS_OVERRIDE:-}"
ROLLOUT_VLLM_INIT_WITH_RANDOM_WEIGHTS="${ROLLOUT_VLLM_INIT_WITH_RANDOM_WEIGHTS:-}"
TUNIX_SKIP_FINAL_CHECKPOINT="${TUNIX_SKIP_FINAL_CHECKPOINT:-}"
TUNIX_DISABLE_TRAJECTORY_LOGGING="${TUNIX_DISABLE_TRAJECTORY_LOGGING:-}"

mapfile -t endpoint_ips < <(
  gcloud alpha compute tpus tpu-vm describe "${TPU_NAME}" \
    --zone="${ZONE}" \
    --format='value(networkEndpoints[].ipAddress)' | tr ';' '\n'
)
if [[ "${#endpoint_ips[@]}" -lt 2 ]]; then
  echo "Expected at least two TPU worker IPs for ${TPU_NAME}." >&2
  exit 1
fi

LOCAL_HOST="${LOCAL_HOST:-${endpoint_ips[0]}}"
REMOTE_HOST="${REMOTE_HOST:-${endpoint_ips[${REMOTE_WORKER_INDEX}]}}"
PROCESS_HOSTS="${PROCESS_HOSTS:-$(IFS=,; echo "${endpoint_ips[*]}")}"

RUN_ROOT="/tmp/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/logs"
TB_DIR="${RUN_ROOT}/tensorboard"
CKPT_DIR="${RUN_ROOT}/checkpoints"

mkdir -p "${LOG_DIR}" "${TB_DIR}" "${CKPT_DIR}"

base_cmd=(
  bash "${RUN_SCRIPT}"
  "data_config.train_data_path=${TRAIN_DATA_PATH}"
  "data_config.eval_data_path=${EVAL_DATA_PATH}"
  "model_config.model_id=${MODEL_ID}"
  "tokenizer_config.tokenizer_path=${MODEL_ID}"
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

run_worker() {
  local logfile="$1"
  cd "${REPO}"
  source "${VENV}/bin/activate"
  mkdir -p "${LOG_DIR}" "${TB_DIR}" "${CKPT_DIR}"
  export HF_TOKEN="${HF_TOKEN_VALUE}"
  export TUNIX_INIT_JAX_DISTRIBUTED=1
  export TUNIX_PROCESS_HOSTS="${PROCESS_HOSTS}"
  export SKIP_JAX_PRECOMPILE=true
  if [[ -n "${TUNIX_SKIP_FINAL_CHECKPOINT}" ]]; then
    export TUNIX_SKIP_FINAL_CHECKPOINT
  fi
  if [[ -n "${TUNIX_DISABLE_TRAJECTORY_LOGGING}" ]]; then
    export TUNIX_DISABLE_TRAJECTORY_LOGGING
  fi
  status=0
  bash -lc "${cmd_string}" > "${logfile}" 2>&1 || status=$?
  echo "HOST=$(hostname) STATUS=${status}"
  return "${status}"
}

run_remote() {
  local logfile="$1"
  local remote_script
  remote_script=$(cat <<EOF
set -euo pipefail
cd ${REPO}
source ${VENV}/bin/activate
mkdir -p ${LOG_DIR} ${TB_DIR} ${CKPT_DIR}
export HF_TOKEN=${HF_TOKEN_VALUE}
export TUNIX_INIT_JAX_DISTRIBUTED=1
export TUNIX_PROCESS_HOSTS=${PROCESS_HOSTS}
export SKIP_JAX_PRECOMPILE=true
if [[ -n ${TUNIX_SKIP_FINAL_CHECKPOINT@Q} ]]; then
  export TUNIX_SKIP_FINAL_CHECKPOINT=${TUNIX_SKIP_FINAL_CHECKPOINT@Q}
fi
if [[ -n ${TUNIX_DISABLE_TRAJECTORY_LOGGING@Q} ]]; then
  export TUNIX_DISABLE_TRAJECTORY_LOGGING=${TUNIX_DISABLE_TRAJECTORY_LOGGING@Q}
fi
status=0
bash -lc ${cmd_string@Q} > ${logfile@Q} 2>&1 || status=\$?
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

run_remote "${remote_log}" > "${remote_status_log}" 2>&1 &
remote_pid=$!

sleep 2

set +e
run_worker "${local_log}"
local_status=$?
wait "${remote_pid}"
remote_ssh_status=$?
set -e

remote_status=$(sed -n 's/.*STATUS=\([0-9][0-9]*\).*/\1/p' "${remote_status_log}" | tail -n 1)
if [[ -z "${remote_status}" ]]; then
  remote_status="${remote_ssh_status}"
fi

echo "LOCAL_STATUS=${local_status}"
echo "REMOTE_SSH_STATUS=${remote_ssh_status}"
echo "REMOTE_STATUS=${remote_status}"

if [[ "${local_status}" -ne 0 || "${remote_status}" -ne 0 ]]; then
  exit 1
fi
