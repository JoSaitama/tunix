#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m unittest \
  tests.my_example.alignment_equivalence_test \
  tests.my_example.alignment_config_test \
  tests.my_example.alignment_curriculum_test \
  tests.my_example.alignment_mismatch_flow_test

SMOKE_ROOT="$(mktemp -d /tmp/learnalign_memory_smoke.XXXXXX)"
KEEP_SMOKE_ARTIFACTS="${KEEP_SMOKE_ARTIFACTS:-0}"

cleanup() {
  exit_code=$?
  if [ "${KEEP_SMOKE_ARTIFACTS}" = "1" ] || [ "${exit_code}" -ne 0 ]; then
    echo "Keeping smoke artifacts: ${SMOKE_ROOT}"
  else
    rm -rf -- "${SMOKE_ROOT}"
  fi
  return "${exit_code}"
}
trap cleanup EXIT

echo "Running seed-5 Mismatch-20% LearnAlign selector smoke."
echo "Temporary artifacts: ${SMOKE_ROOT}"

TUNIX_EXPERIMENT_SEED=5 \
TUNIX_REWARD_RANK_NOISE_SEED=5 \
TUNIX_REWARD_RANK_NOISE_FRACTION=0.2 \
TUNIX_MY_RESULT_DIR="${SMOKE_ROOT}/results" \
"${ROOT_DIR}/my_example/run_alignment_baseline.sh" learnalign \
  --max-train-examples 16 \
  --train-fraction 0.5 \
  --learnalign-warmup-prompts 4 \
  --learnalign-estimation-rollouts 8 \
  --selection-ratio 2 \
  --alignment-artifact-dir "${SMOKE_ROOT}/selection" \
  --metrics-log-dir "${SMOKE_ROOT}/tensorboard" \
  --checkpoint-root "${SMOKE_ROOT}/checkpoints" \
  --output-dir "${SMOKE_ROOT}/model" \
  --max-eval-examples 1 \
  --eval-every-n-steps 500 \
  --save-interval-steps 500 \
  --max-to-keep 1 \
  --skip-eval-before \
  --skip-eval-after

"${PYTHON_BIN}" - "${SMOKE_ROOT}/selection/learnalign_summary.json" \
  "${SMOKE_ROOT}/selection/learnalign_selection.jsonl" <<'PY'
import json
from pathlib import Path
import sys

summary_path = Path(sys.argv[1])
records_path = Path(sys.argv[2])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
records = [
    json.loads(line)
    for line in records_path.read_text(encoding="utf-8").splitlines()
]

assert summary["method"] == "learnalign"
assert summary["estimation_rollouts"] == 8
assert summary["gradient_feature_prompt_batch_size"] == 2
assert summary["gradient_feature_completion_batch_size"] == 16
assert summary["gradient_feature_calls_per_rollout_chunk"] == 2
assert summary["gradient_aggregation"] == "rollout_mean_loss_before_gradient"
assert len(records) == summary["candidate_prompts"]
selected_count = sum(bool(record["selected"]) for record in records)
assert selected_count == summary["selected_prompts"]
assert all(len(record["selector_rewards"]) == 8 for record in records)

print(
    "PASS: 2x8 grouped-gradient LearnAlign feature path completed; "
    f"candidates={len(records)}, selected={summary['selected_prompts']}"
)
PY
