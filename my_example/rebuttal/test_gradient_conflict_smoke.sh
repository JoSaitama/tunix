#!/usr/bin/env bash
# Run both default gradient conflict baselines sequentially on one idle worker.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

check_run() {
  python - "$1" "$2" <<'PY'
import json
import math
from pathlib import Path
import shlex
import sys

from my_example.save_results_to_my_result import _read_scalar_points


def require(condition, message):
    if not condition:
        raise SystemExit(f"FAIL {method}: {message}")


root, method = Path(sys.argv[1]), sys.argv[2]
require((root / "exit_code").read_text().strip() == "0", "launcher failed")
command = shlex.split((root / "launch_command.txt").read_text())
require(
    {
        "TUNIX_DBC_VARIANT=gradient_conflict",
        f"TUNIX_GRADIENT_CONFLICT_METHOD={method}",
        "TUNIX_GRADIENT_CONFLICT_INCLUDE_KL=0",
        "TUNIX_CAGRAD_C=0.5",
        "TUNIX_EXPERIMENT_SEED=5",
        "TUNIX_REWARD_RANK_NOISE_FRACTION=0",
        "JAX_PLATFORMS=tpu",
        "TPU_PROCESS_BOUNDS=1,1,1",
        "TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1",
        "TPU_VISIBLE_CHIPS=0,1,2,3",
        "CLOUD_TPU_TASK_ID=0",
    }.issubset(command),
    "unexpected baseline or TPU configuration",
)

metrics = [
    "loss",
    "skipped_samples",
    "gradient_conflict_negative_pairs",
    "gradient_conflict_shift_norm",
]
if method == "pcgrad":
    metrics.append("pcgrad_projections")
series = {}
for metric in metrics:
    points = _read_scalar_points(root / "tensorboard", f"actor/train/{metric}")
    require({point.step for point in points} == {1, 2}, f"unexpected steps for {metric}")
    require(all(math.isfinite(point.value) for point in points), f"nonfinite {metric}")
    series[metric] = {point.step: point.value for point in points}
require(all(value == 0 for value in series["skipped_samples"].values()), "sample dropped")

checkpoint = root / "checkpoints/actor/2/model_params"
require(checkpoint.is_dir() and any(checkpoint.iterdir()), "step 2 checkpoint missing")
log = (root / "nohup.log").read_text(errors="replace")
require("Checkpoint not found" not in log, "checkpoint restore was skipped")
require("Model saved successfully!" in log, "model export did not finish")
require(
    any(path.is_file() and path.stat().st_size for path in (root / "model").glob("*.safetensors")),
    "exported model weights missing",
)

eval_files = list((root / "results").glob("*__eval_accuracy__meta.json"))
require(len(eval_files) == 1, "missing or ambiguous evaluation export")
evaluation = json.loads(eval_files[0].read_text())
for phase in ("pre-train", "post-train"):
    report = evaluation[phase]
    require(
        report["total"] == 4 and 0 <= report["num_correct"] <= 4,
        f"invalid {phase} count",
    )
    for key in ("accuracy", "partial_accuracy", "format_accuracy"):
        require(0 <= report[key] <= 100, f"invalid {phase} {key}")

print(
    f"PASS {method}: steps=1,2; pre={evaluation['pre-train']['accuracy']}%; "
    f"post={evaluation['post-train']['accuracy']}%; checkpoint and model present"
)
for metric, values in series.items():
    print(f"  {metric}: {values[1]:.6g}, {values[2]:.6g}")
PY
}

CHECK_ONLY=0
if [ "${1:-}" = --check-only ]; then CHECK_ONLY=1; shift; fi
if [ "$#" -ne 1 ] || [ ! -d "$1" ]; then
  echo "usage: $0 [--check-only] EXISTING_SMOKE_DIRECTORY" >&2
  exit 2
fi
SMOKE_ROOT="$(cd "$1" && pwd)"
if [ "$CHECK_ONLY" -eq 1 ]; then
  if [ ! -f "$SMOKE_ROOT/exit_code" ] || [ "$(cat "$SMOKE_ROOT/exit_code")" != 0 ]; then
    echo "Smoke queue has not passed. See $SMOKE_ROOT/queue.log" >&2
    exit 1
  fi
else
  for path in exit_code pcgrad cagrad; do
    if [ -e "$SMOKE_ROOT/$path" ]; then
      echo "Use a fresh smoke directory. Found $SMOKE_ROOT/$path" >&2
      exit 2
    fi
  done
  trap 'code=$?; printf "%s\n" "$code" > "$SMOKE_ROOT/exit_code"' EXIT
  export PYTHONUNBUFFERED=1
  if pgrep -af '[p]ython.*( -m my_example\.|/my_example/)'; then
    echo 'Another my_example task is active on this worker.' >&2
    exit 3
  fi
  test -d data/train && test -d data/test || { echo 'GSM8K data/train or data/test missing.' >&2; exit 2; }
  python -c 'from my_example.save_results_to_my_result import _read_scalar_points'
  df -h "$SMOKE_ROOT"
  env -u JAX_PLATFORM_NAME JAX_PLATFORMS=tpu \
    TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
    TPU_VISIBLE_CHIPS=0,1,2,3 CLOUD_TPU_TASK_ID=0 python - <<'PY'
import jax

devices = jax.local_devices()
if jax.process_count() != 1 or len(devices) != 4 or any(d.platform != "tpu" for d in devices):
    raise SystemExit(f"Expected 1 JAX process and 4 local TPU devices, found {jax.process_count()}: {devices}")
print(f"Single-worker TPU preflight passed: {devices}")
PY
fi

for method in pcgrad cagrad; do
  if [ "$CHECK_ONLY" -eq 0 ]; then
    echo "START $method $(date -Is)"
    env -u TUNIX_DYNAMIC_SAMPLING -u TUNIX_PODS_KEEP_COUNT \
      TUNIX_EXPERIMENT_SEED=5 TUNIX_REWARD_RANK_NOISE_SEED=5 TUNIX_REWARD_RANK_NOISE_FRACTION=0 \
      TUNIX_GRADIENT_CONFLICT_RUN_ROOT="$SMOKE_ROOT/$method" \
      bash my_example/rebuttal/run_gradient_conflict_baseline.sh "$method" \
        --max-train-examples 16 --train-fraction 0.5 --max-eval-examples 4 \
        --save-interval-steps 2 --metrics-flush-every 1
  fi
  check_run "$SMOKE_ROOT/$method" "$method"
done
echo "ALL PASSED $SMOKE_ROOT"
