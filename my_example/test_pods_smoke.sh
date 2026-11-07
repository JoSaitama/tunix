#!/usr/bin/env bash
# Run only on an idle worker. Topology is isolated below before any TPU process.
# Frozen sequence limits (256/768), four prompts, four generated / two retained.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
if [ -f my_example/.env ]; then
  set -a
  source my_example/.env
  set +a
fi
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
export TPU_VISIBLE_CHIPS=0,1,2,3
export CLOUD_TPU_TASK_ID=0
env JAX_PLATFORMS=cpu python -m unittest discover -s tests/my_example -p 'pods*test.py'
mkdir -p logs
TEST_ROOT="$(mktemp -d "${ROOT_DIR}/logs/pods_smoke.XXXXXX")"
echo "Smoke artifacts: $TEST_ROOT (retained for inspection)"
for NOISE in 0 0.2; do
  RUN_DIR="${TEST_ROOT}/mismatch${NOISE}"
  env -u JAX_PLATFORMS -u JAX_PLATFORM_NAME \
    TUNIX_PODS_RUN_ROOT="$RUN_DIR" TUNIX_MY_RESULT_DIR="$RUN_DIR/results" TUNIX_EXPERIMENT_SEED=5 \
    TUNIX_REWARD_RANK_NOISE_SEED=5 TUNIX_REWARD_RANK_NOISE_FRACTION="$NOISE" \
    bash my_example/run_pods_baseline.sh --pods-keep-count 2 \
      --max-train-examples 16 --train-fraction 0.5 \
      --max-prompt-length 256 --total-generation-steps 768 \
      --max-eval-examples 1 --save-interval-steps 2
  python - "$RUN_DIR" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
summary = json.loads((root / "selection/pods_summary.json").read_text())
assert summary["generated_completions"] == 32, summary
assert summary["selected_completions"] == 16, summary
assert summary["chunks"] == 2, summary
assert (root / "exit_code").read_text().strip() == "0"
records = [json.loads(x) for x in (root / "selection/pods_selection.jsonl").read_text().splitlines()]
assert len(records) == 8
assert all(r["n"] == 4 and r["m"] == 2 for r in records)
assert len({r["update_step"] for r in records}) == 2, records
evaluation_files = list((root / "results").glob("*eval_accuracy*meta.json"))
assert len(evaluation_files) == 1, evaluation_files
evaluation = json.loads(evaluation_files[0].read_text())
for stage in ("pre-train", "post-train"):
    for key in ("accuracy", "partial_accuracy", "format_accuracy", "num_correct", "total"):
        assert key in evaluation[stage], (stage, key)
    assert evaluation[stage]["total"] == 1
print("PASS: 2 update steps, 32 generated / 16 retained;", root)
PY
done
env -u JAX_PLATFORMS -u JAX_PLATFORM_NAME \
  TUNIX_EXPERIMENT_SEED=5 TUNIX_REWARD_RANK_NOISE_SEED=5 \
  TUNIX_REWARD_RANK_NOISE_FRACTION=0.2 \
  python -m my_example.test_pods_hbm --report-dir "$TEST_ROOT/hbm" --keep-count 2
echo "PODS clean/mismatch smokes and full-length stress passed. Reports: $TEST_ROOT"
