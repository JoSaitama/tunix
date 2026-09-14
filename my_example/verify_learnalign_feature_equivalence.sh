#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OLD_SELECTION_JSONL="${1:-}"
VERIFY_GENERATION_STEPS="${VERIFY_GENERATION_STEPS:-128}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m unittest \
  tests.my_example.alignment_equivalence_test

VERIFY_ROOT="$(mktemp -d /tmp/learnalign_equivalence.XXXXXX)"
KEEP_VERIFY_ARTIFACTS="${KEEP_VERIFY_ARTIFACTS:-0}"

cleanup() {
  exit_code=$?
  if [ "${KEEP_VERIFY_ARTIFACTS}" = "1" ] || [ "${exit_code}" -ne 0 ]; then
    echo "Keeping verification artifacts: ${VERIFY_ROOT}"
  else
    rm -rf -- "${VERIFY_ROOT}"
  fi
  return "${exit_code}"
}
trap cleanup EXIT

echo "Running a 32-prompt LearnAlign selected-set A/B check."
echo "Temporary artifacts: ${VERIFY_ROOT}"

TUNIX_EXPERIMENT_SEED=0 \
TUNIX_REWARD_RANK_NOISE_SEED=0 \
TUNIX_REWARD_RANK_NOISE_FRACTION=0.2 \
TUNIX_LEARNALIGN_EQUIVALENCE_REPORT="${VERIFY_ROOT}/equivalence.json" \
TUNIX_MY_RESULT_DIR="${VERIFY_ROOT}/results" \
"${ROOT_DIR}/my_example/run_alignment_baseline.sh" learnalign \
  --max-train-examples 64 \
  --train-fraction 0.5 \
  --learnalign-warmup-prompts 4 \
  --learnalign-estimation-rollouts 8 \
  --selection-ratio 4 \
  --projection-dim 4096 \
  --total-generation-steps "${VERIFY_GENERATION_STEPS}" \
  --alignment-artifact-dir "${VERIFY_ROOT}/selection" \
  --metrics-log-dir "${VERIFY_ROOT}/tensorboard" \
  --checkpoint-root "${VERIFY_ROOT}/checkpoints" \
  --output-dir "${VERIFY_ROOT}/model" \
  --max-eval-examples 1 \
  --eval-every-n-steps 500 \
  --save-interval-steps 500 \
  --max-to-keep 1 \
  --skip-eval-before \
  --skip-eval-after

"${PYTHON_BIN}" - "${VERIFY_ROOT}/equivalence.json" \
  "${OLD_SELECTION_JSONL}" <<'PY'
import json
from pathlib import Path
import sys

report_path = Path(sys.argv[1])
report = json.loads(report_path.read_text(encoding="utf-8"))
print(json.dumps(report, indent=2, sort_keys=True))
if not report["passed"]:
    raise SystemExit("FAIL: legacy and grouped selected prompt sets differ")
if report["prompt_count"] != 32 or report["selected_count"] != 8:
    raise SystemExit(
        "FAIL: expected 32 compared prompts and 8 selected prompts; got "
        f"{report['prompt_count']} and {report['selected_count']}"
    )
if report["acceptance_mode"] != "selected-set":
    raise SystemExit("FAIL: verification did not use selected-set acceptance")
if report["nonzero_learnability_prompts"] == 0:
    raise SystemExit(
        "FAIL: all 32 prompt groups had zero learnability; increase "
        "--total-generation-steps and rerun to avoid a degenerate tie test"
    )
print("PASS: both paths selected the same 8 of 32 prompt source indices")

old_path_text = sys.argv[2]
if not old_path_text:
    print("Old seed-0 selection JSONL was not supplied; boundary check skipped.")
    raise SystemExit(0)
old_path = Path(old_path_text)
records = [
    json.loads(line)
    for line in old_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
selected = [float(record["score"]) for record in records if record["selected"]]
dropped = [float(record["score"]) for record in records if not record["selected"]]
if not selected or not dropped:
    raise SystemExit("Old selection JSONL needs both selected and dropped records")
selected_min = min(selected)
dropped_max = max(dropped)
print("Old seed-0 boundary diagnostics:")
print("  records      =", len(records))
print("  selected     =", len(selected))
print("  selected min =", selected_min)
print("  dropped max  =", dropped_max)
print("  boundary gap =", selected_min - dropped_max)
PY
