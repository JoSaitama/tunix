#!/usr/bin/env bash
# Only the requested phase runs. Postselect collects selector chunks and actor
# updates from one full-pool LearnAlign run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ $# != 3 && $# != 4 ]] || [[ $# == 4 && "$4" != --segmented ]]; then
  echo "usage: bash $0 {grpo-ds|gradalign|learnalign-prelearn|learnalign-postselect} NEW_JOB_DIRECTORY ABSOLUTE_XPROF_PYTHON [--segmented]" >&2
  exit 2
fi
PHASE="$1"
JOB="$2"
XPROF_PY="$3"
[[ -n "$JOB" && "$XPROF_PY" == /* && -f "$XPROF_PY" && -x "$XPROF_PY" ]] || {
  echo 'Provide a nonempty job directory and an executable absolute XProf Python path.' >&2
  exit 2
}
[[ "$JOB" == /* ]] || JOB="$ROOT/$JOB"
unset TUNIX_BENCH_DS_XPROF_PYTHON TUNIX_BENCH_LEARNALIGN_XPROF_PYTHON TUNIX_BENCH_SEGMENTED_XPROF_PYTHON
EXTRACT=xprof-observed
if [[ "${4:-}" == --segmented ]]; then
  export TUNIX_BENCH_SEGMENTED_XPROF_PYTHON="$XPROF_PY"
  EXTRACT=xprof-segmented
fi
case "$PHASE" in
  grpo-ds)
    [[ "$EXTRACT" == xprof-segmented ]] || { echo 'GRPO+DS requires --segmented here' >&2; exit 2; }
    METHOD=grpo-ds; WARMUP=1; STEPS=1
    ;;
  gradalign) METHOD=gradalign; WARMUP=0; STEPS=1 ;;
  learnalign-prelearn) METHOD=learnalign; WARMUP=1; STEPS=2 ;;
  learnalign-postselect)
    METHOD=learnalign; WARMUP=10; STEPS=2
    if [[ "$EXTRACT" == xprof-observed ]]; then
      export TUNIX_BENCH_LEARNALIGN_XPROF_PYTHON="$XPROF_PY"
    fi
    ;;
  *) echo "Unknown alignment HBM phase: $PHASE" >&2; exit 2 ;;
esac

env JAX_PLATFORMS=cpu "$XPROF_PY" -c 'from xprof.convert import raw_to_tool_data'
python -m my_example.rebuttal.grpo_benchmark disk-check "$(dirname "$JOB")" \
  --min-free-gib "${TUNIX_BENCH_MIN_FREE_GIB:-6}"
mkdir -- "$JOB"
trap 'rc=$?; printf "%s\n" "$rc" >"$JOB/exit_code"' EXIT

RUN="$JOB/run"
bash "$ROOT/my_example/rebuttal/run_grpo_benchmark.sh" "$METHOD" memory "$RUN" "$WARMUP" "$STEPS" off
if [[ "$PHASE" == learnalign-postselect ]]; then
  test -s "$RUN/selector_hbm.json"
  echo "LearnAlign selector sampled HBM: $RUN/selector_hbm.json"
fi
python -m my_example.rebuttal.grpo_benchmark check --expect-warmup "$WARMUP" --expect-steps "$STEPS" "$RUN"
env JAX_PLATFORMS=cpu "$XPROF_PY" -m my_example.rebuttal.grpo_benchmark \
  "$EXTRACT" "$RUN" >"$JOB/xprof_observed.pending.json"
mv -- "$JOB/xprof_observed.pending.json" "$JOB/xprof_observed.json"
echo "Alignment HBM EXTRACTION PASS ($PHASE): $JOB/xprof_observed.json"
