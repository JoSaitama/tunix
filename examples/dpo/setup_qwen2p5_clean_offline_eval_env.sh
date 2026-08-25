#!/usr/bin/env bash

set -euo pipefail

OFFLINE_EVAL_VENV="${OFFLINE_EVAL_VENV:-/home/lhf_hongfu_gmail_com/.venvs/DPO-EVAL-OFFLINE}"
REWARDBENCH_ROOT="${REWARDBENCH_ROOT:-/home/lhf_hongfu_gmail_com/.cache/reward-bench}"
IFEVAL_ROOT="${IFEVAL_ROOT:-/home/lhf_hongfu_gmail_com/.cache/google-research-ifeval}"
IFBENCH_ROOT="${IFBENCH_ROOT:-/home/lhf_hongfu_gmail_com/.cache/IFBench}"
SAFETY_EVAL_ROOT="${SAFETY_EVAL_ROOT:-/home/lhf_hongfu_gmail_com/.cache/safety-eval}"
LIVEBENCH_ROOT="${LIVEBENCH_ROOT:-/home/lhf_hongfu_gmail_com/.cache/LiveBench}"
WILDBENCH_ROOT="${WILDBENCH_ROOT:-/home/lhf_hongfu_gmail_com/.cache/WildBench}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"
export IFEVAL_ROOT
export IFBENCH_ROOT
export SAFETY_EVAL_ROOT
export LIVEBENCH_ROOT
export WILDBENCH_ROOT

create_venv() {
  local venv_path="$1"
  if python3 -c 'import virtualenv' >/dev/null 2>&1; then
    python3 -m virtualenv "${venv_path}"
  else
    python3 -m venv "${venv_path}"
  fi
}

if [[ "${FORCE_RECREATE}" == "1" ]]; then
  rm -rf "${OFFLINE_EVAL_VENV}"
fi

if [[ ! -x "${OFFLINE_EVAL_VENV}/bin/python" ]]; then
  create_venv "${OFFLINE_EVAL_VENV}"
fi

# shellcheck disable=SC1090
source "${OFFLINE_EVAL_VENV}/bin/activate"
export PIP_NO_CACHE_DIR=1
python -m pip install --upgrade pip
python -m pip install "setuptools<81"
python -m pip install \
  rewardbench \
  datasets \
  huggingface_hub \
  tensorboard \
  absl-py \
  omegaconf \
  langdetect \
  immutabledict \
  nltk \
  pandas \
  shortuuid \
  tabulate \
  pyyaml \
  openai \
  jsonlines \
  tenacity \
  tiktoken \
  fire \
  google-generativeai \
  cohere \
  mistralai \
  anthropic \
  reka-api \
  together \
  emoji \
  syllapy \
  unicodedata2

mkdir -p "$(dirname "${REWARDBENCH_ROOT}")"
if [[ ! -d "${REWARDBENCH_ROOT}/.git" ]]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/allenai/reward-bench.git "${REWARDBENCH_ROOT}"
  git -C "${REWARDBENCH_ROOT}" sparse-checkout set rewardbench scripts tests
else
  git -C "${REWARDBENCH_ROOT}" pull --ff-only
  git -C "${REWARDBENCH_ROOT}" sparse-checkout set rewardbench scripts tests
fi

mkdir -p "$(dirname "${IFEVAL_ROOT}")"
if [[ ! -d "${IFEVAL_ROOT}/.git" ]]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/google-research/google-research.git "${IFEVAL_ROOT}"
  git -C "${IFEVAL_ROOT}" sparse-checkout set instruction_following_eval
else
  git -C "${IFEVAL_ROOT}" pull --ff-only
  git -C "${IFEVAL_ROOT}" sparse-checkout set instruction_following_eval
fi

mkdir -p "$(dirname "${IFBENCH_ROOT}")"
if [[ ! -d "${IFBENCH_ROOT}/.git" ]]; then
  git clone --depth 1 https://github.com/allenai/IFBench.git "${IFBENCH_ROOT}"
else
  git -C "${IFBENCH_ROOT}" pull --ff-only
fi

mkdir -p "$(dirname "${SAFETY_EVAL_ROOT}")"
if [[ ! -d "${SAFETY_EVAL_ROOT}/.git" ]]; then
  git clone --depth 1 https://github.com/allenai/safety-eval.git "${SAFETY_EVAL_ROOT}"
else
  git -C "${SAFETY_EVAL_ROOT}" pull --ff-only
fi

mkdir -p "$(dirname "${LIVEBENCH_ROOT}")"
if [[ ! -d "${LIVEBENCH_ROOT}/.git" ]]; then
  git clone --depth 1 https://github.com/livebench/livebench.git "${LIVEBENCH_ROOT}"
else
  git -C "${LIVEBENCH_ROOT}" pull --ff-only
fi

mkdir -p "$(dirname "${WILDBENCH_ROOT}")"
if [[ ! -d "${WILDBENCH_ROOT}/.git" ]]; then
  git clone --depth 1 https://github.com/allenai/WildBench.git "${WILDBENCH_ROOT}"
else
  git -C "${WILDBENCH_ROOT}" pull --ff-only
fi
python -m pip install absl-py langdetect immutabledict nltk
python - <<'PY'
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
print("nltk resources: punkt, punkt_tab")
PY

python - <<'PY'
import os
import sys

import rewardbench  # pylint: disable=unused-import

repo_root = os.environ["IFEVAL_ROOT"]
sys.path.insert(0, repo_root)
import instruction_following_eval.evaluation_main  # pylint: disable=unused-import

ifbench_root = os.environ.get("IFBENCH_ROOT")
if ifbench_root:
  sys.path.insert(0, ifbench_root)
  import evaluation_lib  # pylint: disable=unused-import

print("rewardbench import: ok")
print("instruction_following_eval import: ok")
print("ifbench import: ok")
PY
deactivate

cat <<EOF
Offline benchmark eval environment is ready.

  offline_eval_venv: ${OFFLINE_EVAL_VENV}
  rewardbench_root:  ${REWARDBENCH_ROOT}
  ifeval_root:       ${IFEVAL_ROOT}
  ifbench_root:      ${IFBENCH_ROOT}
  safety_eval_root:  ${SAFETY_EVAL_ROOT}
  livebench_root:    ${LIVEBENCH_ROOT}
  wildbench_root:    ${WILDBENCH_ROOT}

Notes:
  - RewardBench v1 scoring still runs through the Tunix/JAX TPU stack.
  - This env hosts offline benchmark tooling and reference repos.
EOF
