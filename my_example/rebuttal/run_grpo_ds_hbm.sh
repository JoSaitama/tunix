#!/usr/bin/env bash
# Opt-in DS HBM diagnostic. The benchmark launcher controls single-worker TPU
# isolation, the training configuration, and the full iteration boundaries.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ $# != 2 ]]; then
  echo "usage: bash $0 NEW_JOB_DIRECTORY ABSOLUTE_XPROF_PYTHON" >&2
  exit 2
fi
JOB="$1"
XPROF_PY="$2"
[[ -n "$JOB" && "$XPROF_PY" == /* && -f "$XPROF_PY" && -x "$XPROF_PY" ]] || {
  echo 'Provide a nonempty job directory and an executable absolute XProf Python path.' >&2
  exit 2
}
[[ "$JOB" == /* ]] || JOB="$ROOT/$JOB"
# Import check is CPU-only and fails before any training or TPU initialization.
env JAX_PLATFORMS=cpu "$XPROF_PY" -c 'from xprof.convert import raw_to_tool_data'
python -m my_example.rebuttal.grpo_benchmark disk-check "$(dirname "$JOB")" \
  --min-free-gib "${TUNIX_BENCH_MIN_FREE_GIB:-6}"
mkdir -- "$JOB"
trap 'rc=$?; printf "%s\n" "$rc" >"$JOB/exit_code"' EXIT

export TUNIX_BENCH_DS_XPROF_PYTHON="$XPROF_PY"
RUN="$JOB/run"
bash "$ROOT/my_example/rebuttal/run_grpo_benchmark.sh" grpo-ds memory "$RUN" 1 2 off
python -m my_example.rebuttal.grpo_benchmark check --expect-warmup 1 --expect-steps 2 "$RUN"
env JAX_PLATFORMS=cpu "$XPROF_PY" -m my_example.rebuttal.grpo_benchmark \
  xprof-observed "$RUN" >"$JOB/xprof_observed.pending.json"
mv -- "$JOB/xprof_observed.pending.json" "$JOB/xprof_observed.json"
echo "DS HBM EXTRACTION PASS: $JOB/xprof_observed.json"
