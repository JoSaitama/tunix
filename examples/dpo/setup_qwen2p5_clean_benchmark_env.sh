#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/lhf_hongfu_gmail_com/tunix}"
MT_BENCH_JUDGE_VENV="${MT_BENCH_JUDGE_VENV:-/home/lhf_hongfu_gmail_com/.venvs/DPO-EVAL-MTB}"
OPENAI_JUDGE_VENV="${OPENAI_JUDGE_VENV:-/home/lhf_hongfu_gmail_com/.venvs/DPO-EVAL-OAI}"
ARENA_HARD_ROOT="${ARENA_HARD_ROOT:-/home/lhf_hongfu_gmail_com/.cache/arena-hard-auto}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

create_venv() {
  local venv_path="$1"
  if python3 -c 'import virtualenv' >/dev/null 2>&1; then
    python3 -m virtualenv "${venv_path}"
  else
    python3 -m venv "${venv_path}"
  fi
}

create_venv "${MT_BENCH_JUDGE_VENV}"
# shellcheck disable=SC1090
source "${MT_BENCH_JUDGE_VENV}/bin/activate"
python -m pip install --upgrade pip
python -m pip install "setuptools<81"
python -m pip install "fschat[llm_judge]" psutil transformers accelerate sentencepiece
if [[ "${INSTALL_TORCH}" == "1" ]]; then
  python -m pip install torch --index-url "${TORCH_INDEX_URL}"
fi
deactivate

create_venv "${OPENAI_JUDGE_VENV}"
# shellcheck disable=SC1090
source "${OPENAI_JUDGE_VENV}/bin/activate"
python -m pip install --upgrade pip
python -m pip install "setuptools<81"
python -m pip install "openai>=1,<2" alpaca-eval
if [[ "${INSTALL_TORCH}" == "1" ]]; then
  python -m pip install torch --index-url "${TORCH_INDEX_URL}"
fi
mkdir -p "$(dirname "${ARENA_HARD_ROOT}")"
if [[ ! -d "${ARENA_HARD_ROOT}/.git" ]]; then
  git clone https://github.com/lmarena/arena-hard-auto.git "${ARENA_HARD_ROOT}"
else
  git -C "${ARENA_HARD_ROOT}" pull --ff-only
fi
python -m pip install -r "${ARENA_HARD_ROOT}/requirements.txt"
deactivate

cat <<EOF
Benchmark eval environment is ready.

  repo_root:        ${REPO_ROOT}
  mt_bench_venv:    ${MT_BENCH_JUDGE_VENV}
  openai_judge_venv:${OPENAI_JUDGE_VENV}
  arena_hard_root:  ${ARENA_HARD_ROOT}
  install_torch:    ${INSTALL_TORCH}
  torch_index_url:  ${TORCH_INDEX_URL}

Notes:
  - Stage A generation uses the existing Tunix/JAX TPU stack from the DPO env.
  - MT-Bench uses the MT judge venv because FastChat still wants the legacy
    OpenAI SDK plus transformers imports.
  - AlpacaEval 2 and Arena-Hard use the OpenAI judge venv because they require
    the modern `openai>=1` client.
  - Stage B judge commands are prepared separately by
    examples/dpo/eval_qwen2p5_clean_benchmarks.py.
EOF
