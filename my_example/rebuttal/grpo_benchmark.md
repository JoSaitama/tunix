# Clean GSM8K: wall-clock and HBM benchmark

Use a single idle worker on a v5p-16 node. Run all three methods on the
**same worker**, sequentially, in fresh processes, and use the same environment,
checkout, and data. No benchmark is enabled in an ordinary launch.

The new launcher calls `my_example/run_grpo_gemma.sh`; it mirrors the exact
`baseline`, `group_policy`, and `group_loo_policy` route flags from
`run_seeded_full.sh`, without invoking its post-training exporters. Python
asserts the actual route. Vanilla uses the ordinary actor trainer and no
per-completion valuation gradients. DTV/LOO retain their policy-only scoring
and full-gradient masked update, including the production LOO retention floor.

The benchmark keeps the clean accuracy reward, formal Gemma 3 1B model and
initialization, seed 13 (data shuffle 55), four prompts with four completions
each, and the 256/768 generation shape. It stops at the outer learner boundary,
without changing the 691-update LR schedule or its 69-step optimizer warmup.
The 15-iteration measurement warmup is unrelated to the optimizer warmup.
No pre/post or periodic evaluation, checkpoint/restore (including final save),
merged model, or legacy CSV/plot export. Unless diagnostics are explicitly
enabled, no legacy TensorBoard, progress output, or selection JSONL; purely
diagnostic JIT outputs and their host processing are skipped at the source.
Persistent JAX compilation cache reuse and the optional static TFLOPs probe
are disabled for all arms. Per-iteration device synchronization is not added.

## Smoke and full runs

On the worker, after pulling and activating its existing TPU environment:

```bash
set -e
python -m pip check
env JAX_PLATFORMS=cpu python -m unittest discover -s tests/my_example -p grpo_benchmark_test.py
env JAX_PLATFORMS=cpu python -m unittest discover -s tests/rl -p 'self_inf*trainer_test.py'
bash -n my_example/rebuttal/run_grpo_benchmark.sh

SMOKE="logs/rebuttal_grpo_gsm8k_cost_smoke_seed13_clean_$(date +%Y%m%d_%H%M%S)"
for mode in timing memory; do
  for method in vanilla dtv dtv-loo; do
    bash my_example/rebuttal/run_grpo_benchmark.sh "$method" "$mode" "$SMOKE/$mode/$method" 1 2 off
  done
done
python -m my_example.rebuttal.grpo_benchmark check --expect-warmup 1 --expect-steps 2 \
  "$SMOKE"/timing/{vanilla,dtv,dtv-loo} "$SMOKE"/memory/{vanilla,dtv,dtv-loo}
```

The smoke validates the full flow and that a trace is saved. It does **not**
validate TPU memory coverage; inspect the smoke trace first. Replacing the
window arguments with `15 50 off` in a **fresh** output root produces the
reviewer runs. `timing/off` alone supplies the timing cells; `memory/off` is
the independent phase trace and memory collection, not a timing substitute.
Use the `check` command with `--expect-warmup 15 --expect-steps 50` on the six
run directories. It rejects mismatched source/configuration, initial LoRA,
topology, or prompt order. First compare the actual model revision and source
environment to the original experiment records, especially if a historical run
used overrides.

Optional I/O comparison: add fresh `timing` runs with `15 50 on` for `dtv`
and `dtv-loo`, and compare each to its own `timing/off` run (or repeat both
arms on the same idle worker if conditions changed). DTV restores its original
scalar logs, whereas LOO restores scalar logs plus its per-step selection JSONL.
This includes diagnostic computation, transfers, serialization and output;
it is not a pure storage microbenchmark. The final flush/cleanup is outside the
measured window and recorded separately in `post_window_teardown_seconds`.

## Reports and interpretation

- `summary.json`: startup/loading details are in `metadata.json`; warmup,
  measured wall time, iteration mean/median, optimizer update count, and
  generated tokens/s are in the summary. Median uses adjacent outer iteration
  boundaries, not per-kernel latency. End-of-window synchronization is included
  in the last interval. No new barriers inside the timed window.
- `iterations.csv`: host timestamps reduced to per-iteration intervals,
  non-padding sampled completion tokens (stop EOS excluded by the sampler),
  prompt-order hashes, warmup/measurement membership, and compile flags.
  Tokens/s is generated output tokens divided by **end-to-end** training wall
  time, not pure decode speed. `compiler_events.json` keeps raw JAX log events
  and any unassigned events. The primary summary includes all measured steps;
  the secondary summary excludes steps with a logged completed XLA compile.
- `metadata.json` and `stdout.log`: resolved configuration, model snapshot,
  initial adapter SHA256, runtime versions, topology and commit. The runner
  refuses to overwrite existing run directories.
- `trace/`: use XProf Trace Viewer to inspect `benchmark/iteration`,
  `rollout`, `reference_logps`, `old_policy_logps`, `reward_advantage`,
  `actor_update`, and `weight_sync` (when required). Annotations label host
  dispatch regions; use device lanes/dependencies for device timings. Do not
  sum overlapping lanes. For DTV/LOO, `actor_update` is fused valuation +
  update. The old-policy prepass is an additional DTV/LOO cost, separate
  from that fused phase.

### HBM qualification

`memory.json` records allocator counters per local device (where supported)
after initialization and at both window boundaries, plus paths to raw XPlane.
`peak_bytes_in_use` is reported as an allocator lifetime high-water mark. If
it rises within the window, the new allocator high-water value is shown as
such; it is **not** automatically treated as a total physical HBM peak. A
static Memory Viewer estimate, sampled value, missing metric, or flat trace
that does not account for resident buffers is not a verified window peak.

Open the trace in [**XProf Memory Profile**](https://openxla.org/xprof/memory_profile),
not Memory Viewer. Match every
device ID, the measurement window, and resident allocations present at its
start (model, optimizer, and other live allocations). The post-initialization
and window-start allocator snapshots are useful sanity checks but are not
proof of total physical coverage. Use a phase label only when the device
timeline identifies one. When the runtime peak is qualified, transcribe the
following fields to a JSON file (one key per device; no numeric placeholders
should be used for actual evidence):

- `collector`: `"XProf Memory Profile"`; `peak_scope`: `"measurement_window"`.
- `resident_allocations_verified`: `true`; `verification_notes`: how the
  device IDs, resident HBM, window and units were verified.
- `iterations_inclusive`: inclusive bounds from the run (e.g. `[16, 65]`
  for the default window).
- `devices`: object keyed by device ID, with `peak_hbm_bytes` (integer
  bytes) and optional `peak_phase` (or `null`).

```bash
python -m my_example.rebuttal.grpo_benchmark xprof-peaks \
  "$BENCH/memory/dtv-loo" verified_dtv_loo_xprof.json
```

That command verifies the evidence schema, device coverage and interval, then
prints per-device values and the max across devices. It does not independently
verify the analyst's reading of the trace. If the installed TPU stack cannot
provide this coverage, report the supported allocator metric separately or
leave runtime peak unreported. No missing data are set to zero.

Historical 691-step logs are separate: report complete-job wall time only from
verified launch/finish timestamps for each run and state which phases are
included. Tqdm or TensorBoard actor-only metrics are not replacements for full
wall time. Do not multiply a short-window average by 691 to populate that table.
