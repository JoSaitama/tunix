#!/usr/bin/env bash
# Independent frozen-checkpoint audit. No optimizer updates or source-run writes.
set -euo pipefail
AUDIT_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${AUDIT_REPO_DIR}"
if [ -f "${AUDIT_REPO_DIR}/my_example/.env" ]; then
  set -a
  source "${AUDIT_REPO_DIR}/my_example/.env"
  set +a
fi
export TPU_PROCESS_BOUNDS="${TPU_PROCESS_BOUNDS:-1,1,1}"
export TPU_CHIPS_PER_PROCESS_BOUNDS="${TPU_CHIPS_PER_PROCESS_BOUNDS:-2,2,1}"
export TPU_VISIBLE_CHIPS="${TPU_VISIBLE_CHIPS:-0,1,2,3}"
export CLOUD_TPU_TASK_ID="${CLOUD_TPU_TASK_ID:-0}"
export WANDB_DISABLED=true WANDB_MODE=disabled
AUDIT_PYTHON="${PYTHON_BIN:-python}"
if [ "${1:-}" = "--self-test" ]; then
  exec "${AUDIT_PYTHON}" -m my_example.audit_alignment_projection --self-test
fi
if [ "$#" -lt 1 ]; then
  echo "usage: bash my_example/audit_alignment_projection.sh RUN_DIR [audit/frozen config flags...]" >&2
  echo "example: ... logs/learnalign_seed42_mismatch0.2_TIMESTAMP --candidate-count 64" >&2
  exit 2
fi
AUDIT_SOURCE_RUN="$1"
shift
mkdir -p "${AUDIT_REPO_DIR}/logs/projection_audits"
AUDIT_OUTPUT="$(mktemp -d "${AUDIT_REPO_DIR}/logs/projection_audits/audit_$(date +%Y%m%d_%H%M%S).XXXXXX")"
# Python requires a fresh output directory; the shell-created container holds it.
echo "Audit artifacts: ${AUDIT_OUTPUT}/report"
"${AUDIT_PYTHON}" -m my_example.audit_alignment_projection \
  --run-dir "${AUDIT_SOURCE_RUN}" --audit-output-dir "${AUDIT_OUTPUT}/report" \
  "$@" 2>&1 | tee "${AUDIT_OUTPUT}/stdout.log"
