#!/usr/bin/env bash
# SERVER ONLY. CPU -> tiny real clean/mismatch runs -> current-worker HBM.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
MODE="${1:-all}"
case "$MODE" in all|cpu|smoke|hbm) ;; *) echo 'usage: bash test_gradient_norm.sh [all|cpu|smoke|hbm]' >&2; exit 2 ;; esac
if [ "$MODE" != cpu ] && pgrep -af '[p]ython.*( -m my_example\.|/my_example/)'; then
  echo 'STOP: another model/test process is running on this worker.' >&2; exit 3
fi
if [ -f my_example/.env ]; then set -a; source my_example/.env; set +a; fi
NORM_PYTHON="${AUDIT_PYTHON:-$ROOT_DIR/.venv_jax081/bin/python}"
export PATH="$(dirname "$NORM_PYTHON"):$PATH"
if [ "$MODE" = all ] || [ "$MODE" = cpu ]; then
  env -u JAX_PLATFORM_NAME JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    "$NORM_PYTHON" -m unittest discover \
    -s tests/my_example -p 'gradient_norm*test.py'
fi
if [ "$MODE" = cpu ]; then exit 0; fi
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
export TPU_VISIBLE_CHIPS=0,1,2,3
export CLOUD_TPU_TASK_ID=0
mkdir -p logs
TEST_ROOT="$ROOT_DIR/logs/rebuttal_grpo_gsm8k_gradient_norm_top50_test_$(date +%Y%m%d_%H%M%S)"
mkdir "$TEST_ROOT"
echo "Disposable norm test artifacts: $TEST_ROOT"
if [ "$MODE" = all ] || [ "$MODE" = smoke ]; then
  for NOISE in 0 0.2; do
    RUN_DIR="$TEST_ROOT/mismatch${NOISE}"
    env TUNIX_NORM_RUN_ROOT="$RUN_DIR" TUNIX_MY_RESULT_DIR="$RUN_DIR/results" \
      TUNIX_EXPERIMENT_SEED=5 TUNIX_REWARD_RANK_NOISE_SEED=5 \
      TUNIX_REWARD_RANK_NOISE_FRACTION="$NOISE" \
      bash my_example/rebuttal/run_gradient_norm_baseline.sh \
        --max-train-examples 16 --train-fraction .5 --max-eval-examples 1 --save-interval-steps 2
    "$NORM_PYTHON" - "$RUN_DIR" <<'PY'
import json
from pathlib import Path
import sys
import numpy as np
root = Path(sys.argv[1])
summary = json.loads((root / 'selection/norm_summary.json').read_text())
assert summary['updates_logged'] == 2, summary
assert summary['observed_completions'] == 32 and summary['retained_completions'] == 16, summary
assert summary['retained_fraction'] == .5, summary
assert (root / 'exit_code').read_text().strip() == '0'
records = [json.loads(x) for x in (root / 'selection/norm_selection.jsonl').read_text().splitlines()]
assert len(records) == 2
assert len({r['train_step'] for r in records}) == 2
for record in records:
    assert not record['advantages_recomputed']
    assert len(record['groups']) == 4
    for group in record['groups']:
        expected = sorted(np.argsort(-np.asarray(group['policy_gradient_squared_l2_norms']), kind='stable')[:2].tolist())
        assert group['selected_local_indices'] == expected, group
        assert len(group['advantages']) == 4
files = list((root / 'results').glob('*eval_accuracy*meta.json'))
assert len(files) == 1, files
metrics = json.loads(files[0].read_text())
for stage in ('pre-train', 'post-train'):
    for key in ('accuracy', 'partial_accuracy', 'format_accuracy', 'num_correct', 'total'):
        assert key in metrics[stage], (stage, key)
    assert metrics[stage]['total'] == 1
print('PASS: real norm smoke, unchanged4-answer advantages, 50% retention, pre/post schema:', root)
PY
  done
fi
if [ "$MODE" = all ] || [ "$MODE" = hbm ]; then
  HBM_ARGS=()
  if [ -n "${TUNIX_NORM_TEST_BACKBONE_PATH:-}" ]; then HBM_ARGS+=(--backbone-path "$TUNIX_NORM_TEST_BACKBONE_PATH"); fi
  env -u JAX_PLATFORMS -u JAX_PLATFORM_NAME TUNIX_EXPERIMENT_SEED=5 \
    TUNIX_REWARD_RANK_NOISE_SEED=5 TUNIX_REWARD_RANK_NOISE_FRACTION=.2 \
    "$NORM_PYTHON" -m my_example.rebuttal.test_gradient_norm_hbm \
    --report-dir "$TEST_ROOT/hbm" ${HBM_ARGS[@]+"${HBM_ARGS[@]}"}
fi
echo "NORM TEST STAGE PASSED: $TEST_ROOT"
