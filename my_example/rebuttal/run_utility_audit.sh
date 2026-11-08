#!/usr/bin/env bash
set -euo pipefail

AUDIT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUDIT_MODE="${1:-preflight}"
if [ "$#" -gt 0 ]; then shift; fi
case "$AUDIT_MODE" in
  preflight|roundtrip|smoke|hbm|score|probe) ;;
  *) echo "usage: bash $0 preflight|roundtrip|smoke|hbm|score|probe --reward-mode accuracy|legacy [args...]" >&2; exit 2 ;;
esac
cd "$AUDIT_REPO"
AUDIT_PYTHON="${AUDIT_PYTHON:-$AUDIT_REPO/.venv_jax081/bin/python}"
if [ ! -x "$AUDIT_PYTHON" ]; then
  echo "Set AUDIT_PYTHON to the server training environment's Python." >&2
  exit 2
fi
if [ "$AUDIT_MODE" != preflight ]; then
  if pgrep -af '[p]ython.*( -m my_example\.|/my_example/)' >/dev/null; then
    echo "Another model/test task may be using this worker; inspect before running:" >&2
    pgrep -af '[p]ython.*( -m my_example\.|/my_example/)' >&2 || true
    exit 3
  fi
  if [ -f my_example/.env ]; then
    set -a
    source my_example/.env
    set +a
  fi
fi
AUDIT_SEED=0
AUDIT_NOISE=0
AUDIT_STEP=500
AUDIT_ARGS=("$@")
for ((AUDIT_I=0; AUDIT_I<${#AUDIT_ARGS[@]}; AUDIT_I++)); do
  case "${AUDIT_ARGS[$AUDIT_I]}" in
    --rollout-seed) AUDIT_SEED="${AUDIT_ARGS[$((AUDIT_I+1))]:?missing seed}" ;;
    --rollout-seed=*) AUDIT_SEED="${AUDIT_ARGS[$AUDIT_I]#*=}" ;;
    --candidate-noise) AUDIT_NOISE="${AUDIT_ARGS[$((AUDIT_I+1))]:?missing noise}" ;;
    --candidate-noise=*) AUDIT_NOISE="${AUDIT_ARGS[$AUDIT_I]#*=}" ;;
    --step) AUDIT_STEP="${AUDIT_ARGS[$((AUDIT_I+1))]:?missing step}" ;;
    --step=*) AUDIT_STEP="${AUDIT_ARGS[$AUDIT_I]#*=}" ;;
  esac
done
if [[ ! "$AUDIT_SEED" =~ ^[0-9]+$ || ! "$AUDIT_STEP" =~ ^[0-9]+$ || ! "$AUDIT_NOISE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "invalid seed/noise/step for output directory" >&2
  exit 2
fi
AUDIT_TS="$(date +%Y%m%d_%H%M%S)"
AUDIT_INPUT="$AUDIT_REPO/logs/utility_audit_inputs/vanilla_seed0_clean_20260717_144514"
AUDIT_OUTPUT="$AUDIT_REPO/logs/rebuttal_grpo_gsm8k_utility_${AUDIT_MODE}_seed${AUDIT_SEED}_mismatch${AUDIT_NOISE}_step${AUDIT_STEP}_${AUDIT_TS}"
env -u JAX_PLATFORMS -u JAX_PLATFORM_NAME \
  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_VISIBLE_CHIPS=0,1,2,3 CLOUD_TPU_TASK_ID=0 \
  WANDB_DISABLED=true WANDB_MODE=disabled \
  "$AUDIT_PYTHON" -m my_example.rebuttal.audit_grpo_utility \
  --input-dir "$AUDIT_INPUT" --output-dir "$AUDIT_OUTPUT" \
  "${AUDIT_ARGS[@]}" --mode "$AUDIT_MODE"
