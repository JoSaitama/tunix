#!/usr/bin/env bash
set -euo pipefail

# Parameterized UltraFeedback prompt-response mismatch runner.
#
# Usage:
#   MISMATCH_PCT=20 EXP_SEED=0 bash run_mismatch_seed_six_methods_train_eval.sh
#   MISMATCH_PCT=40 EXP_SEED=1 bash run_mismatch_seed_six_methods_train_eval.sh
#
# This script generates a temporary six-method train+eval script from the
# existing no_pref20 seed2 template, then executes it.
#
# Actual corruption config passed to Python:
#   global_mismatch${MISMATCH_PCT}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MISMATCH_PCT="${MISMATCH_PCT:-20}"
EXP_SEED="${EXP_SEED:-0}"
TEMPLATE="${TEMPLATE:-${SCRIPT_DIR}/run_no_pref20_seed2_six_methods_train_eval.sh}"

if [[ ! "${MISMATCH_PCT}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: MISMATCH_PCT must be an integer percentage, got ${MISMATCH_PCT}"
  exit 1
fi

if [ "${MISMATCH_PCT}" -lt 1 ] || [ "${MISMATCH_PCT}" -gt 100 ]; then
  echo "ERROR: MISMATCH_PCT must be in [1, 100], got ${MISMATCH_PCT}"
  exit 1
fi

if [[ ! "${EXP_SEED}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: EXP_SEED must be an integer, got ${EXP_SEED}"
  exit 1
fi

if [ ! -f "${TEMPLATE}" ]; then
  echo "ERROR: template script not found: ${TEMPLATE}"
  exit 1
fi

GEN_SCRIPT="/tmp/run_mismatch${MISMATCH_PCT}_seed${EXP_SEED}_six_methods_train_eval_${USER}_$$.sh"

python - "${TEMPLATE}" "${GEN_SCRIPT}" "${MISMATCH_PCT}" "${EXP_SEED}" <<'PY'
from pathlib import Path
import sys

template = Path(sys.argv[1])
out = Path(sys.argv[2])
pct = sys.argv[3]
seed = sys.argv[4]

s = template.read_text()

replacements = {
    "CORRUPTION=global_no_pref20": f"CORRUPTION=global_mismatch{pct}",
    "EXP_SEED=2": f"EXP_SEED={seed}",
    "RUN_TAG=seed2": f"RUN_TAG=seed{seed}",
    "no_pref20_seed2_six_methods_5metrics": f"mismatch{pct}_seed{seed}_six_methods_5metrics",
    "no_pref20 seed2": f"mismatch{pct} seed{seed}",
    "no_pref20_seed2": f"mismatch{pct}_seed{seed}",
    "global_no_pref20": f"global_mismatch{pct}",
}

for old, new in replacements.items():
  s = s.replace(old, new)

# Slightly less strict disk threshold, to avoid stopping after model export.
s = s.replace('if [ "${avail_gb}" -lt 4 ]; then', 'if [ "${avail_gb}" -lt 2 ]; then')
s = s.replace("less than 4GB", "less than 2GB")

out.write_text(s)
out.chmod(0o755)
print(out)
PY

echo "Generated script: ${GEN_SCRIPT}"
bash -n "${GEN_SCRIPT}"

echo
echo "============================================================"
echo "Starting UltraFeedback mismatch experiment"
echo "MISMATCH_PCT=${MISMATCH_PCT}"
echo "EXP_SEED=${EXP_SEED}"
echo "CORRUPTION=global_mismatch${MISMATCH_PCT}"
echo "Template=${TEMPLATE}"
echo "Generated=${GEN_SCRIPT}"
echo "============================================================"

bash "${GEN_SCRIPT}"
