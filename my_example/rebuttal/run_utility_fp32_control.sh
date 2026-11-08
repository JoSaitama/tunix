#!/usr/bin/env bash
# One server-only control. Does not queue a scientific scan or any training.
set -euo pipefail
CONTROL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CONTROL_REPO"
if pgrep -af '[p]ython.*( -m my_example\.|/my_example/)'; then
  echo 'STOP: a model/test is active on this worker; do not interrupt it.' >&2
  exit 3
fi
CONTROL_PYTHON="${AUDIT_PYTHON:-$CONTROL_REPO/.venv_jax081/bin/python}"
if [ ! -x "$CONTROL_PYTHON" ]; then
  echo 'Set AUDIT_PYTHON to the SERVER training Python.' >&2
  exit 2
fi
if ! command -v timeout >/dev/null; then
  echo 'This server launcher requires GNU timeout for its 45-minute wall-clock budget.' >&2
  exit 2
fi
if [ -f my_example/.env ]; then
  set -a
  source my_example/.env
  set +a
fi
echo '[1/2] Server CPU regression only; no TPU initialization'
AUDIT_PYTHON="$CONTROL_PYTHON" bash my_example/rebuttal/test_utility_cpu.sh
echo '[2/2] Single-worker FP32 measurement/HBM control; eight heldout prompts ONLY'
CONTROL_TS="$(date +%Y%m%d_%H%M%S)"
set +e
timeout --signal=TERM --kill-after=60s 45m env -u JAX_PLATFORMS -u JAX_PLATFORM_NAME \
  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_VISIBLE_CHIPS=0,1,2,3 CLOUD_TPU_TASK_ID=0 \
  WANDB_DISABLED=true WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
  "$CONTROL_PYTHON" -m my_example.rebuttal.utility_fp32_control \
  --input-dir "$CONTROL_REPO/logs/utility_audit_inputs/vanilla_seed0_clean_20260717_144514" \
  --output-dir "$CONTROL_REPO/logs/rebuttal_grpo_gsm8k_utility_fp32control_seed0_mismatch0_step500_${CONTROL_TS}" \
  --reward-mode accuracy "$@"
CONTROL_STATUS=$?
set -e
case "$CONTROL_STATUS" in
  0) echo 'FP32 control passed. This does NOT validate BF16 production utility or any method benefit.' ;;
  2) echo 'STOP: control not validated (or invalid CLI); inspect control_summary/numerical_report. No automatic retry.' ;;
  124|137) echo 'STOP: bounded control timed out. Do not queue larger probes; prioritize norm-top50.' ;;
  *) echo 'STOP: control execution failed; inspect log/HBM. No automatic retry or large scan.' ;;
esac
exit "$CONTROL_STATUS"
