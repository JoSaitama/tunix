#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="/home/lhf_hongfu_gmail_com/.venvs/DPO"
ROOT_ENV_FILE="${REPO_ROOT}/.env"
EXAMPLE_ENV_FILE="${REPO_ROOT}/my_example/.env"
PROFILE="${1:-full}"

if [[ "${PROFILE}" == "full" || "${PROFILE}" == "smoke" ]]; then
  shift
else
  PROFILE="full"
fi

if [[ -n "${1:-}" && "$1" != --* ]]; then
  VARIANT="$1"
  shift
else
  VARIANT="vanilla_dpo"
fi
CORRUPTION_CONFIG="${DPO_CORRUPTION_CONFIG:-clean}"
FT_MODE="full"
SFT_MODEL_PATH=""
SEED="${DPO_SEED:-}"
RUN_SEED="${DPO_RUN_SEED:-}"
TRAIN_SHUFFLE_SEED="${DPO_TRAIN_SHUFFLE_SEED:-}"
CURATION_SEED="${CURATION_SEED:-}"
declare -a USER_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --corruption-config)
      CORRUPTION_CONFIG="${2:-}"
      shift 2
      ;;
    --corruption-config=*)
      CORRUPTION_CONFIG="${1#*=}"
      shift
      ;;
    --ft-mode)
      FT_MODE="${2:-}"
      shift 2
      ;;
    --ft-mode=*)
      FT_MODE="${1#*=}"
      shift
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --seed=*)
      SEED="${1#*=}"
      shift
      ;;
    --run-seed)
      RUN_SEED="${2:-}"
      shift 2
      ;;
    --run-seed=*)
      RUN_SEED="${1#*=}"
      shift
      ;;
    --train-shuffle-seed)
      TRAIN_SHUFFLE_SEED="${2:-}"
      shift 2
      ;;
    --train-shuffle-seed=*)
      TRAIN_SHUFFLE_SEED="${1#*=}"
      shift
      ;;
    --curation-seed)
      CURATION_SEED="${2:-}"
      shift 2
      ;;
    --curation-seed=*)
      CURATION_SEED="${1#*=}"
      shift
      ;;
    *)
      if [[ -z "${SFT_MODEL_PATH}" && "$1" != *=* ]]; then
        SFT_MODEL_PATH="$1"
      else
        USER_OVERRIDES+=("$1")
      fi
      shift
      ;;
  esac
done

case "${FT_MODE}" in
  full|lora)
    ;;
  *)
    echo "Unsupported --ft-mode: ${FT_MODE}" >&2
    exit 1
    ;;
esac

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
fi

if [[ -f "${ROOT_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ROOT_ENV_FILE}"
  set +a
elif [[ -f "${EXAMPLE_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${EXAMPLE_ENV_FILE}"
  set +a
fi

cd "${REPO_ROOT}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required to load the Qwen tokenizer." >&2
  exit 1
fi

if [[ -z "${SFT_MODEL_PATH}" || ! -d "${SFT_MODEL_PATH}" ]]; then
  echo "Usage: $0 [full|smoke] [vanilla_dpo|random_pair_filtering|reward_based_filtering|self_inf|self_inf_loo|self_inf_loo_cos|outlier_l2] /path/to/sft_exported_model [--corruption-config clean|tail50_flip20|tail50_flip40|global_flip10|global_flip20|global_flip30|global_flip40] [--ft-mode full|lora] [--seed INT] [--run-seed INT] [--train-shuffle-seed INT] [--curation-seed INT] [overrides...]" >&2
  exit 1
fi

if [[ "${FT_MODE}" == "lora" ]]; then
  CONFIG_PATH="${REPO_ROOT}/examples/dpo/qwen2p5_1p5b_hh_rlhf_from_sft_lora.yaml"
else
  CONFIG_PATH="${REPO_ROOT}/examples/dpo/qwen2p5_1p5b_hh_rlhf_from_sft.yaml"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
export DPO_DATASET_MODULE="tunix/examples/data/hh_rlhf_dpo.py:create_dataset"
export DPO_CLEAN_TEST_DATA_MODULE="examples/data/hh_rlhf_dpo.py:create_dataset(split='test_prefs', partition='all', subset='all', seed=42)"
export DPO_RUN_DIR_PREFIX="dpo_qwen2p5_1p5b_hh_rlhf_from_sft"
export DPO_RUN_DATASET_TAG="qwen2p5-1p5b-hh-rlhf-from-sft"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
if [[ -z "${RUN_SEED}" && -n "${SEED}" ]]; then
  RUN_SEED="${SEED}"
fi
DEFAULT_RUN_ROOT="$("${PYTHON_BIN}" - <<'PY' "${REPO_ROOT}" "${VARIANT}" "${CORRUPTION_CONFIG}" "${FT_MODE}" "${PROFILE}" "${RUN_TS}"
import importlib.util
from pathlib import Path
import sys

module_path = Path(sys.argv[1]) / "examples" / "dpo" / "qwen2p5_dpo_experiments.py"
spec = importlib.util.spec_from_file_location("qwen2p5_dpo_experiments", module_path)
exp_lib = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(exp_lib)
print(
    exp_lib.build_run_root(
        repo_root=sys.argv[1],
        variant=sys.argv[2],
        corruption_config=sys.argv[3],
        ft_mode=sys.argv[4],
        profile=sys.argv[5],
        run_ts=sys.argv[6],
        seed=None,
    )
)
PY
)"
if [[ -n "${RUN_SEED}" ]]; then
  DEFAULT_RUN_ROOT="$("${PYTHON_BIN}" - <<'PY' "${REPO_ROOT}" "${VARIANT}" "${CORRUPTION_CONFIG}" "${FT_MODE}" "${PROFILE}" "${RUN_TS}" "${RUN_SEED}"
import importlib.util
from pathlib import Path
import sys

module_path = Path(sys.argv[1]) / "examples" / "dpo" / "qwen2p5_dpo_experiments.py"
spec = importlib.util.spec_from_file_location("qwen2p5_dpo_experiments", module_path)
exp_lib = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(exp_lib)
print(
    exp_lib.build_run_root(
        repo_root=sys.argv[1],
        variant=sys.argv[2],
        corruption_config=sys.argv[3],
        ft_mode=sys.argv[4],
        profile=sys.argv[5],
        run_ts=sys.argv[6],
        seed=int(sys.argv[7]),
    )
)
PY
)"
fi
RUN_ROOT="${RUN_ROOT:-${DEFAULT_RUN_ROOT}}"
RUN_NAME="$("${PYTHON_BIN}" - <<'PY' "${REPO_ROOT}" "${VARIANT}" "${CORRUPTION_CONFIG}" "${FT_MODE}" "${PROFILE}" "${RUN_TS}"
import importlib.util
from pathlib import Path
import sys

module_path = Path(sys.argv[1]) / "examples" / "dpo" / "qwen2p5_dpo_experiments.py"
spec = importlib.util.spec_from_file_location("qwen2p5_dpo_experiments", module_path)
exp_lib = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(exp_lib)
print(
    exp_lib.build_run_name(
        variant=sys.argv[2],
        corruption_config=sys.argv[3],
        ft_mode=sys.argv[4],
        profile=sys.argv[5],
        run_ts=sys.argv[6],
        seed=None,
    )
)
PY
)"
if [[ -n "${RUN_SEED}" ]]; then
  RUN_NAME="$("${PYTHON_BIN}" - <<'PY' "${REPO_ROOT}" "${VARIANT}" "${CORRUPTION_CONFIG}" "${FT_MODE}" "${PROFILE}" "${RUN_TS}" "${RUN_SEED}"
import importlib.util
from pathlib import Path
import sys

module_path = Path(sys.argv[1]) / "examples" / "dpo" / "qwen2p5_dpo_experiments.py"
spec = importlib.util.spec_from_file_location("qwen2p5_dpo_experiments", module_path)
exp_lib = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(exp_lib)
print(
    exp_lib.build_run_name(
        variant=sys.argv[2],
        corruption_config=sys.argv[3],
        ft_mode=sys.argv[4],
        profile=sys.argv[5],
        run_ts=sys.argv[6],
        seed=int(sys.argv[7]),
    )
)
PY
)"
fi
TMP_CONFIG="$(mktemp "/tmp/qwen2p5_1p5b_ultrafeedback_dpo_${VARIANT}_${CORRUPTION_CONFIG}_${FT_MODE}_${PROFILE}_XXXX.yaml")"
trap 'rm -f "${TMP_CONFIG}"' EXIT

"${PYTHON_BIN}" - <<'PY' \
  "${CONFIG_PATH}" \
  "${TMP_CONFIG}" \
  "${REPO_ROOT}" \
  "${RUN_ROOT}" \
  "${RUN_NAME}" \
  "${PROFILE}" \
  "${VARIANT}" \
  "${CORRUPTION_CONFIG}" \
  "${SFT_MODEL_PATH}" \
  "${CURATION_THRESHOLD:-}" \
  "${CURATION_KEEP_RATIO:-}" \
  "${SELF_INFLUENCE_DOT_THRESHOLD:-}" \
  "${SEED}" \
  "${CURATION_SEED}" \
  "${TRAIN_SHUFFLE_SEED}"
import importlib.util
from pathlib import Path
import sys

(
    config_path,
    output_path,
    repo_root,
    run_root,
    run_name,
    profile,
    variant,
    corruption_config,
    sft_model_path,
    curation_threshold_env,
    curation_keep_ratio_env,
    self_influence_dot_threshold_env,
    dpo_seed_text,
    curation_seed_text,
    train_shuffle_seed_text,
) = sys.argv[1:]

module_path = Path(repo_root) / "examples" / "dpo" / "qwen2p5_dpo_experiments.py"
spec = importlib.util.spec_from_file_location("qwen2p5_dpo_experiments", module_path)
exp_lib = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(exp_lib)

exp_lib.prepare_launch_config(
    config_path=config_path,
    output_path=output_path,
    run_root=run_root,
    run_name=run_name,
    profile=profile,
    variant=variant,
    corruption_config=corruption_config,
    sft_model_path=sft_model_path,
    curation_threshold=curation_threshold_env,
    curation_keep_ratio=curation_keep_ratio_env,
    self_influence_dot_threshold=self_influence_dot_threshold_env,
    dpo_seed=int(dpo_seed_text) if dpo_seed_text else None,
    curation_seed=int(curation_seed_text) if curation_seed_text else (
        int(dpo_seed_text) if dpo_seed_text else None
    ),
    train_shuffle_seed=(
        int(train_shuffle_seed_text) if train_shuffle_seed_text else None
    ),
)
PY

"${PYTHON_BIN}" -m tunix.cli.dpo_main \
  "${TMP_CONFIG}" \
  "${USER_OVERRIDES[@]}"
