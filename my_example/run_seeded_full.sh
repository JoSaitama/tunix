#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$#" -lt 2 ]; then
  echo "usage: $0 METHOD SEED [extra run_grpo_gemma.sh arguments...]" >&2
  echo "methods: baseline batch group batch_policy group_policy l2 batch_loo group_loo batch_loo_keep75 group_loo_keep75 batch_loo_policy group_loo_policy batch_loo_policy_keep75 group_loo_policy_keep75 batch_loo_policy_only group_loo_policy_only random_batch random_group reward_batch reward_group" >&2
  exit 2
fi

METHOD="$1"
SEED="$2"
shift 2

case "${SEED}" in
  ''|*[!0-9]*)
    echo "error: SEED must be a nonnegative integer; got '${SEED}'" >&2
    exit 2
    ;;
esac

case "${METHOD}" in
  baseline)
    METHOD_SLUG="baseline"
    METHOD_SCRIPT="./my_example/run_baseline.sh"
    ;;
  batch)
    METHOD_SLUG="dtv_selfinf_batch"
    METHOD_SCRIPT="./my_example/run_dbc_self_inf_batch.sh"
    ;;
  group)
    METHOD_SLUG="dtv_selfinf_group"
    METHOD_SCRIPT="./my_example/run_dbc_self_inf_group.sh"
    ;;
  batch_policy)
    METHOD_SLUG="dtv_selfinf_batch_policy"
    METHOD_SCRIPT="./my_example/run_dbc_self_inf_batch_policy.sh"
    ;;
  group_policy)
    METHOD_SLUG="dtv_selfinf_group_policy"
    METHOD_SCRIPT="./my_example/run_dbc_self_inf_group_policy.sh"
    ;;
  l2)
    METHOD_SLUG="dtv_outlier_l2"
    METHOD_SCRIPT="./my_example/run_dbc_outlier_l2.sh"
    ;;
  batch_loo)
    METHOD_SLUG="dtv_selfinf_batch_loo"
    METHOD_SCRIPT="./my_example/run_dbc_self_inf_batch_loo.sh"
    ;;
  group_loo)
    METHOD_SLUG="dtv_selfinf_group_loo"
    METHOD_SCRIPT="./my_example/run_dbc_self_inf_group_loo.sh"
    ;;
  batch_loo_keep75)
    METHOD_SLUG="dtv_selfinf_batch_loo_keep75"
    METHOD_SCRIPT="./my_example/run_dbc_self_inf_batch_loo.sh"
    ;;
  group_loo_keep75)
    METHOD_SLUG="dtv_selfinf_group_loo_keep75"
    METHOD_SCRIPT="./my_example/run_dbc_self_inf_group_loo.sh"
    ;;
  batch_loo_policy)
    METHOD_SLUG="dtv_selfinf_batch_loo_policy"
    METHOD_SCRIPT="./my_example/run_dbc_self_inf_batch_loo_policy.sh"
    ;;
  group_loo_policy)
    METHOD_SLUG="dtv_selfinf_group_loo_policy"
    METHOD_SCRIPT="./my_example/run_dbc_self_inf_group_loo_policy.sh"
    ;;
  batch_loo_policy_keep75)
    METHOD_SLUG="dtv_selfinf_batch_loo_policy_keep75"
    METHOD_SCRIPT="./my_example/run_dbc_self_inf_batch_loo_policy.sh"
    ;;
  group_loo_policy_keep75)
    METHOD_SLUG="dtv_selfinf_group_loo_policy_keep75"
    METHOD_SCRIPT="./my_example/run_dbc_self_inf_group_loo_policy.sh"
    ;;
  batch_loo_policy_only)
    METHOD_SLUG="dtv_selfinf_batch_loo_policy_only"
    METHOD_SCRIPT="./my_example/run_dbc_self_inf_batch_loo_policy_only.sh"
    ;;
  group_loo_policy_only)
    METHOD_SLUG="dtv_selfinf_group_loo_policy_only"
    METHOD_SCRIPT="./my_example/run_dbc_self_inf_group_loo_policy_only.sh"
    ;;
  random_batch)
    METHOD_SLUG="random_batch"
    METHOD_SCRIPT="./my_example/run_random_filter_batch.sh"
    ;;
  random_group)
    METHOD_SLUG="random_group"
    METHOD_SCRIPT="./my_example/run_random_filter_group.sh"
    ;;
  reward_batch)
    METHOD_SLUG="reward_batch"
    METHOD_SCRIPT="./my_example/run_reward_filter_batch.sh"
    ;;
  reward_group)
    METHOD_SLUG="reward_group"
    METHOD_SCRIPT="./my_example/run_reward_filter_group.sh"
    ;;
  *)
    echo "error: unknown METHOD '${METHOD}'" >&2
    exit 2
    ;;
esac

if pgrep -af '[p]ython.*my_example' >/dev/null 2>&1; then
  echo "error: another my_example Python task is already running:" >&2
  pgrep -af '[p]ython.*my_example' >&2
  exit 3
fi

RUN_TS="${TUNIX_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
DATASET_NAME="${TUNIX_DATASET_NAME:-gsm8k}"
case "${DATASET_NAME}" in
  ''|*[!A-Za-z0-9_-]*)
    echo "error: TUNIX_DATASET_NAME may contain only letters, digits, '_' and '-'" >&2
    exit 2
    ;;
esac
NOISE_FRACTION="${TUNIX_REWARD_RANK_NOISE_FRACTION:-}"
DATA_MODE="clean"
if [ -n "${NOISE_FRACTION}" ] && [ "${NOISE_FRACTION}" != "0" ]; then
  NOISE_FRACTION_CANONICAL="$(
    awk -v value="${NOISE_FRACTION}" 'BEGIN { printf "%g", value }'
  )"
  NOISE_FRACTION_SLUG="${NOISE_FRACTION_CANONICAL//./p}"
  DATA_MODE="mismatch${NOISE_FRACTION_SLUG}"
fi
FILTER_RATIO="${TUNIX_FILTER_RATIO:-}"
FILTER_SUFFIX=""
case "${METHOD}" in
  random_batch|random_group|reward_batch|reward_group)
    if [ -z "${FILTER_RATIO}" ]; then
      echo "error: TUNIX_FILTER_RATIO is required for ${METHOD}" >&2
      exit 2
    fi
    FILTER_PERCENT="$(awk -v value="${FILTER_RATIO}" 'BEGIN { printf "%02d", value * 100 }')"
    FILTER_SUFFIX="_filter0p${FILTER_PERCENT}"
    ;;
esac
RUN_NAME="grpo_${DATASET_NAME}_${METHOD_SLUG}${FILTER_SUFFIX}_seed${SEED}_${DATA_MODE}_${RUN_TS}"
RUN_ROOT="${ROOT_DIR}/runs/${RUN_NAME}"
LOG_ROOT="${ROOT_DIR}/logs/${RUN_NAME}"

export RUN_ROOT LOG_ROOT
export TUNIX_EXPERIMENT_SEED="${SEED}"
export TUNIX_REWARD_MODE="accuracy"
export TUNIX_MY_RESULT_DIR="${LOG_ROOT}/results"

if [[ "${METHOD}" == *_keep75 ]]; then
  export TUNIX_DBC_SELF_INF_MIN_KEEP_FRACTION="0.75"
else
  unset TUNIX_DBC_SELF_INF_MIN_KEEP_FRACTION || true
fi

mkdir -p \
  "${RUN_ROOT}" \
  "${LOG_ROOT}/results" \
  "${LOG_ROOT}/tensorboard"

exec > >(tee "${LOG_ROOT}/nohup.log") 2>&1

printf '%s\n' "$$" > "${LOG_ROOT}/pid"

echo "Method:          ${METHOD}"
echo "Dataset:         ${DATASET_NAME}"
echo "Seed:            ${SEED}"
echo "Dataset seed:    $((42 + SEED))"
echo "Rollout seed:    ${SEED}"
echo "Min keep:        ${TUNIX_DBC_SELF_INF_MIN_KEEP_FRACTION:-method-default}"
echo "Data mode:       ${DATA_MODE}"
echo "Reward noise:    ${NOISE_FRACTION:-0}"
echo "Filter ratio:    ${FILTER_RATIO:-n/a}"
echo "Noise seed:      ${TUNIX_REWARD_RANK_NOISE_SEED:-n/a}"
echo "Run:             ${RUN_ROOT}"
echo "Logs:            ${LOG_ROOT}"
echo "Method launcher: ${METHOD_SCRIPT}"

cd "${ROOT_DIR}"

"${METHOD_SCRIPT}" \
  --metrics-log-dir "${LOG_ROOT}/tensorboard" \
  --checkpoint-root "${RUN_ROOT}/checkpoints" \
  --output-dir "${RUN_ROOT}/model" \
  "$@"

status=$?
printf '%s\n' "${status}" > "${LOG_ROOT}/exit_code"

echo "Exit code: ${status}"
exit "${status}"
