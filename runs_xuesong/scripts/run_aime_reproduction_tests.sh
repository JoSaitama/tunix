#!/usr/bin/env bash
# Run AIME reproduction unit tests on CPU without acquiring the TPU runtime.

set -euo pipefail

REPO="${REPO:-/home/jason_chia925_gmail_com/Project/tunix}"
VENV="${VENV:-${REPO}/.venv}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Python environment is missing: ${VENV}/bin/python" >&2
  exit 1
fi

cd "${REPO}"
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1

exec "${VENV}/bin/python" -m pytest -q \
  tests/oss_utils_test.py \
  tests/rl/self_inf_trainer_test.py \
  tests/rl/rl_cluster_test.py \
  tests/rl/agentic/agentic_grpo_learner_test.py \
  tests/cli/grpo_main_test.py
