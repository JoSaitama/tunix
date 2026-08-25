#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="/home/lhf_hongfu_gmail_com/.venvs/DPO"
LAUNCHER="${REPO_ROOT}/examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh"
EVAL_SCRIPT="${REPO_ROOT}/examples/dpo/eval_qwen2p5_gf_benchmarks.py"
SFT_MODEL_PATH="${REPO_ROOT}/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/exported_model"
RUN_TS="${RUN_TS:-20260417_013847}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs/results/qwen2p5_gf_benchmarks_${RUN_TS}}"

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
fi

mkdir -p "${OUTPUT_ROOT}"

run_root_for() {
  local variant="$1"
  local dataset="$2"
  python - "${REPO_ROOT}" "${variant}" "${dataset}" "${RUN_TS}" <<'PY'
import importlib.util
from pathlib import Path
import sys

repo_root, variant, dataset, run_ts = sys.argv[1:]
module_path = Path(repo_root) / "examples" / "dpo" / "qwen2p5_dpo_experiments.py"
spec = importlib.util.spec_from_file_location("qwen2p5_dpo_experiments", module_path)
exp_lib = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(exp_lib)
print(
    exp_lib.build_run_root(
        repo_root=repo_root,
        variant=variant,
        corruption_config=dataset,
        ft_mode="lora",
        profile="full",
        run_ts=run_ts,
        seed=None,
    )
)
PY
}

for dataset in global_flip20 global_flip40; do
  for variant in random_pair_filtering_filt5 reward_based_filtering_filt5; do
    run_root="$(run_root_for "${variant}" "${dataset}")"
    run_key="${dataset}_${variant}"
    per_run_json="${OUTPUT_ROOT}/per_run/${run_key}.json"
    echo "== ${dataset} / ${variant} =="
    if [[ -f "${per_run_json}" ]]; then
      echo "already evaluated: ${per_run_json}"
      continue
    fi
    if [[ ! -d "${run_root}/exported_model" ]]; then
      echo "training: ${run_root}"
      RUN_TS="${RUN_TS}" "${LAUNCHER}" full "${variant}" "${SFT_MODEL_PATH}" \
        --corruption-config "${dataset}" \
        --ft-mode lora
    else
      echo "using existing exported_model: ${run_root}/exported_model"
    fi
    echo "evaluating: ${run_key}"
    python "${EVAL_SCRIPT}" \
      --run-ts "${RUN_TS}" \
      --datasets "${dataset}" \
      --methods "${variant}" \
      --output-root "${OUTPUT_ROOT}"
    if [[ -d "${run_root}/exported_model" ]]; then
      rm -rf "${run_root}/exported_model"
      echo "cleaned exported_model: ${run_root}/exported_model"
    fi
  done
done

echo "GF filt5 benchmark pipeline complete."
echo "Results: ${OUTPUT_ROOT}"
