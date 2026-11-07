#!/usr/bin/env bash
# Run ONLY on an idle worker. Never run this concurrently with its training.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
if [ -f my_example/.env ]; then set -a; source my_example/.env; set +a; fi
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
export TPU_VISIBLE_CHIPS=0,1,2,3
export CLOUD_TPU_TASK_ID=0
unset TUNIX_PODS_KEEP_COUNT
env JAX_PLATFORMS=cpu python -m unittest discover -s tests/my_example -p 'dynamic_sampling*test.py'
mkdir -p logs
TEST_ROOT="$(mktemp -d "$ROOT_DIR/logs/ds_smoke.XXXXXX")"
echo "Disposable test artifacts: $TEST_ROOT (formal runs use named timestamp directories)"
for NOISE in 0 0.2; do
  RUN_DIR="$TEST_ROOT/mismatch${NOISE}"
  env -u JAX_PLATFORMS -u JAX_PLATFORM_NAME \
    TUNIX_DS_RUN_ROOT="$RUN_DIR" TUNIX_MY_RESULT_DIR="$RUN_DIR/results" \
    TUNIX_EXPERIMENT_SEED=5 TUNIX_REWARD_RANK_NOISE_SEED=5 TUNIX_REWARD_RANK_NOISE_FRACTION="$NOISE" \
    bash my_example/run_ds_baseline.sh --ds-max-generation-batches 20 \
      --max-train-examples 16 --train-fraction .5 --max-eval-examples 1 --save-interval-steps 2
  python - "$RUN_DIR" <<'PY'
import json
from pathlib import Path
import sys
root=Path(sys.argv[1])
summary=json.loads((root/'selection/ds_summary.json').read_text())
assert summary['prepared_updates']==2, summary
assert summary['selected_groups']==8, summary
assert summary['generated_groups']>=8, summary
assert (root/'exit_code').read_text().strip()=='0'
records=[json.loads(line) for line in (root/'selection/ds_selection.jsonl').read_text().splitlines()]
assert sum(row['selected'] for row in records)==8
assert all(row['eligible'] and row['observed_reward_std']>0 for row in records if row['selected'])
files=list((root/'results').glob('*eval_accuracy*meta.json'))
assert len(files)==1, files
metrics=json.loads(files[0].read_text())
for stage in ['pre-train','post-train']:
    for key in ['accuracy','partial_accuracy','format_accuracy','num_correct','total']:
        assert key in metrics[stage], (stage,key)
    assert metrics[stage]['total']==1
print('PASS: real DS smoke and pre/post result schema:',root)
PY
done
env -u JAX_PLATFORMS -u JAX_PLATFORM_NAME TUNIX_EXPERIMENT_SEED=5 \
  TUNIX_REWARD_RANK_NOISE_SEED=5 TUNIX_REWARD_RANK_NOISE_FRACTION=.2 \
  python -m my_example.test_ds_hbm --report-dir "$TEST_ROOT/hbm"
echo "ALL DS TESTS PASSED: $TEST_ROOT"
