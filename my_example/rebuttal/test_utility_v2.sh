#!/usr/bin/env bash
# Server only. No training and no automatic large scientific audit.
set -euo pipefail
AUDIT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$AUDIT_REPO"
if pgrep -af '[p]ython.*( -m my_example\.|/my_example/)'; then
  echo 'STOP: another model/test task is active on this worker; do not interrupt it.' >&2
  exit 3
fi
echo '[1/3] CPU regression (JAX_PLATFORMS=cpu)'
bash my_example/rebuttal/test_utility_cpu.sh
echo '[2/3] Read-only metadata/replay preflight'
bash my_example/rebuttal/run_utility_audit.sh preflight --reward-mode accuracy "$@"
echo '[3/3] Current-worker numerical controls; no candidate gradient scan'
set +e
bash my_example/rebuttal/run_utility_audit.sh numerics --reward-mode accuracy "$@"
AUDIT_STATUS=$?
set -e
if [ "$AUDIT_STATUS" -eq 2 ]; then
  echo 'STOP: numerical controls did not validate finite-step utility (or CLI rejected input).'
  echo 'Inspect the log and numerical_report.json. Do NOT repeat large probes or tune for a positive result.'
fi
exit "$AUDIT_STATUS"
