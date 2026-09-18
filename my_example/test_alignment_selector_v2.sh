#!/usr/bin/env bash
# Runs host tests, four end-to-end smokes, then isolated full-length HBM tests.
set -euo pipefail
ALIGN_TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ALIGN_TEST_ROOT"
if [ -f my_example/.env ]; then
  set -a
  source my_example/.env
  set +a
fi
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
export TPU_VISIBLE_CHIPS=0,1,2,3
export CLOUD_TPU_TASK_ID=0
export WANDB_DISABLED=true
export WANDB_MODE=disabled
unset TUNIX_LEARNALIGN_EQUIVALENCE_REPORT
export TUNIX_EXPERIMENT_SEED=5
export TUNIX_REWARD_RANK_NOISE_SEED=5
mkdir -p "$ALIGN_TEST_ROOT/logs"
ALIGN_TEST_DIR="$(mktemp -d "$ALIGN_TEST_ROOT/logs/selector_v2_test.XXXXXX")"
echo "Test artifacts: $ALIGN_TEST_DIR"
python -m unittest discover -s tests/my_example -p 'alignment_*test.py'
for ALIGN_TEST_NOISE in 0 0.2; do
  export TUNIX_REWARD_RANK_NOISE_FRACTION="$ALIGN_TEST_NOISE"
  for ALIGN_TEST_METHOD in learnalign gradalign; do
    ALIGN_TEST_RUN="$ALIGN_TEST_DIR/${ALIGN_TEST_METHOD}_mismatch${ALIGN_TEST_NOISE}"
    mkdir -p "$ALIGN_TEST_RUN"
    # Only shorten the dataset/warmup/evaluation for smoke tests. Keep selector
    # rollouts=8/4, q=4, D=4096, Grad ref=30/interval=10 and real update=4x4.
    TUNIX_MY_RESULT_DIR="$ALIGN_TEST_RUN/results" \
    bash my_example/run_alignment_baseline.sh "$ALIGN_TEST_METHOD" \
      --max-train-examples 80 --train-fraction 0.5 \
      --learnalign-warmup-prompts 4 \
      --max-eval-examples 1 --save-interval-steps 10 \
      --alignment-artifact-dir "$ALIGN_TEST_RUN/selection" \
      --metrics-log-dir "$ALIGN_TEST_RUN/tensorboard" \
      --checkpoint-root "$ALIGN_TEST_RUN/checkpoints" \
      --output-dir "$ALIGN_TEST_RUN/model" \
      2>&1 | tee "$ALIGN_TEST_RUN/smoke.log"
    python - "$ALIGN_TEST_RUN/selection" "$ALIGN_TEST_METHOD" <<'PY'
import json
from pathlib import Path
import sys
directory, method = Path(sys.argv[1]), sys.argv[2]
metadata = json.loads((directory / "run_metadata.json").read_text())
summary = json.loads((directory / f"{method}_summary.json").read_text())
definition = metadata["selector_definition"]
assert definition["selector_version"] == "alignment_selector_v2"
assert definition["projection_version"] == "signed_feature_hash_v2"
assert definition["selector_token_reduction"] == ("mean" if method == "learnalign" else "sum")
assert definition["selector_beta"] == (metadata["frozen_update_configuration"]["beta"] if method == "learnalign" else 0)
assert summary["selector_version"] == definition["selector_version"]
assert summary["selector_beta"] == definition["selector_beta"]
assert metadata["effective_max_steps"] == 10
assert (directory / f"{method}_selection.jsonl").stat().st_size > 0
eval_files = list((directory.parent / "results").glob("*eval_accuracy*meta.json"))
assert len(eval_files) == 1, eval_files
evaluation = json.loads(eval_files[0].read_text())
for stage in ("pre-train", "post-train"):
    for key in ("accuracy", "partial_accuracy", "format_accuracy", "num_correct", "total"):
        assert key in evaluation[stage], (stage, key)
    assert evaluation[stage]["total"] == 1
print("PASS: selector v2 metadata/summary/selection")
PY
  done
done
# Noise changes reward values, not static tensor sizes. Stress both methods
# once under 20% mismatch, after the clean/noisy end-to-end tests above.
export TUNIX_REWARD_RANK_NOISE_FRACTION=0.2
for ALIGN_TEST_METHOD in learnalign gradalign; do
  python -m my_example.test_alignment_selector_hbm \
    --method "$ALIGN_TEST_METHOD" --report-dir "$ALIGN_TEST_DIR/${ALIGN_TEST_METHOD}_hbm" \
    2>&1 | tee "$ALIGN_TEST_DIR/${ALIGN_TEST_METHOD}_hbm.log"
done
printf 'ALL TESTS PASSED. Artifacts: %s\n' "$ALIGN_TEST_DIR"
