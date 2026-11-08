#!/usr/bin/env bash
set -euo pipefail

AUDIT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIT_MODE="${1:-metadata}"
if [ "$#" -gt 0 ]; then shift; fi
case "$AUDIT_MODE" in
  metadata|restore) ;;
  *) echo "usage: bash $0 metadata|restore [audit arguments...]" >&2; exit 2 ;;
esac
cd "$AUDIT_REPO"
AUDIT_PYTHON="${AUDIT_PYTHON:-$AUDIT_REPO/.venv_jax081/bin/python}"
if [ ! -x "$AUDIT_PYTHON" ]; then
  echo "Set AUDIT_PYTHON to the server environment's python executable." >&2
  exit 2
fi
if [ "$AUDIT_MODE" = restore ]; then
  if pgrep -af '[p]ython.*my_example\.(main|alignment_main|pods_main|ds_main|test_.*hbm|audit_grpo_checkpoint_restore)' >/dev/null; then
    echo "Another model task may be running. Check this worker before restore:" >&2
    pgrep -af '[p]ython.*my_example\.' >&2 || true
    exit 3
  fi
  if [ -f my_example/.env ]; then
    set -a
    source my_example/.env
    set +a
  fi
fi
AUDIT_TS="$(date +%Y%m%d_%H%M%S)"
AUDIT_INPUT="$AUDIT_REPO/logs/utility_audit_inputs/vanilla_seed0_clean_20260717_144514"
AUDIT_REPORT="$AUDIT_REPO/logs/utility_checkpoint_checks/${AUDIT_MODE}_seed0_step500_${AUDIT_TS}"
env -u JAX_PLATFORMS -u JAX_PLATFORM_NAME \
  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_VISIBLE_CHIPS=0,1,2,3 CLOUD_TPU_TASK_ID=0 \
  WANDB_DISABLED=true WANDB_MODE=disabled \
  "$AUDIT_PYTHON" -m my_example.audit_grpo_checkpoint_restore \
  --input-dir "$AUDIT_INPUT" --step 500 \
  --expected-commit 6fba11c8d2e4cb00ee19e8ed45e196521d3f93d0 \
  --report-dir "$AUDIT_REPORT" "$@" --mode "$AUDIT_MODE"
