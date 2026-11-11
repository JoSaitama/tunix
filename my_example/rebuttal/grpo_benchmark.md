# Clean GSM8K: wall-clock and HBM benchmark

Use a single idle worker on a v5p-16 node. Run methods on the
**same worker**, sequentially, in fresh processes, and use the same environment,
checkout, and data. No benchmark is enabled in an ordinary launch.

The launcher calls the same Python entrypoint as `my_example/run_grpo_gemma.sh`
(or the standalone alignment entrypoint). It mirrors the exact
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

### Additional baseline cost measurements

The same opt-in launcher also accepts `pods`, `grpo-ds`, `norm-top50`, `pcgrad`,
`cagrad`, `gradalign`, `random10`, `reward10`, and `learnalign`. They use clean seed 13, the same accuracy reward
mode and 256/768 limits as the existing benchmark, not necessarily the reward
environment or overrides of a historical full training run. Ordinary training
launchers are unchanged. These methods require `diagnostics=off`; neither checkpoint
nor model is saved. GradAlign still uses its held-out **selector reference**;
this is not periodic training evaluation.

| Method | Default timing (warmup + measured updates) | Default memory |
| --- | --- | --- |
| PODS, GRPO+DS, Norm Top-50%, PCGrad, CAGrad | 15 + 50 | 1 + 2 |
| GradAlign | 20 + 50 (five complete selector/update cycles) | 0 + 1 (first selector and first update) |
| Random 10%, Reward 10% | 15 + 50 | 1 + 2 |
| LearnAlign cost probe | 10 + 50, plus one separately timed selector | 1 + 2 (pre-learning) or 10 + 2 (post-selection) |

The GradAlign round uses q=4, 160 candidates and 30 held-out reference
prompts for a full round, each with four selector rollouts. The first selector
is not executed by the learner's batch-size probe. `selector_seconds` measures
reference/candidate estimation and ranking together, with audit output disabled;
`window_seconds_outside_selector` is a residual, not a kernel-only update time.
The short GradAlign HBM run measures the combined early window and includes
initial compilation; it is not the same iteration range as 1+2 or a whole-run
peak. Trace annotations distinguish selector and actor-update dispatch.

`iterations.csv` counts training rollout tokens in `generated_tokens`, and
GradAlign selector tokens separately. `summary.json` counts both in
`generated_tokens` and `generated_tokens_per_wall_second`, and reports the
training-only count/rate separately. DS training rollout counts include all
candidate batches, even rejected or excess groups; its actor still gets 16
rows. PODS generates 16 rows and the actor receives 8, verified at runtime.
Cross-method prompt hashes are checked only for methods with a shared input
stream. DS and both alignment methods change that stream; their repeated runs are checked
within their method, rather than against the fixed stream. The checker still
rejects different commits and runtime configurations; rerun a Vanilla anchor
on the new checkout if a same-commit comparison is needed.

### Random/Reward 10% and LearnAlign phase windows

Random and Reward use the production group-scoped fixed filter at 0.10.
Fractional group quotas use the production stochastic rounding, so the observed
fraction over the measurement window need not be exactly 10%. The actor still
receives all 16 completions for per-completion gradients and masks the update.
`iterations.csv` records `filtered_completions`; `summary.json` also gives
`actual_filtered_fraction`. Only scalar counts are retained on device and
copied to the host after timing; decision logs remain disabled.

LearnAlign runs ten actual GRPO updates before selection in this **cost probe
only**. The normal pre-learning budget (75 updates) and the 691-step optimizer
schedule are unchanged. Report these as three separate measurements:

- `learnalign_prelearn`: updates 6–10; updates 1–5 are timing warmup.
- `learnalign_selector`: one complete scoring/selection pass after update 10
  (2,764 prompts × 8 completions); time and tokens are separate.
- `measurement`: post-selection updates 11–60; time and tokens exclude the
  selector. This uses the selected set from the step-10 model, not step 75.

The two independent HBM launches record updates 2–3 (`prelearn`) and 11–12
(`postselect`). The post-selection run performs the ten updates and full
selector again, then starts tracing. Neither trace covers the selector peak.
Do not treat the maximum of the two as a whole-run LearnAlign peak.

```bash
bash my_example/rebuttal/run_grpo_benchmark.sh learnalign memory "$BENCH/memory/learnalign-prelearn" 1 2 off
bash my_example/rebuttal/run_grpo_benchmark.sh learnalign memory "$BENCH/memory/learnalign-postselect" 10 2 off
```

For limited space, queue `timing` first and capture `memory` one method at a
time or use a larger output filesystem. The launcher and trace start require
6 GiB free by default (`TUNIX_BENCH_MIN_FREE_GIB`); this is a preflight check,
not an upper bound on XPlane size. Raw traces are never removed automatically.
Extract a small JSON with an isolated XProf Python from the repo root:

```bash
"$XPROF_PY" -m my_example.rebuttal.grpo_benchmark xprof-observed "$RUN" \
  > "$RUN/xprof_observed.json"
```

An empty/incomplete device map fails extraction rather than reporting zero.
These are observed per-chip stack-plus-heap peaks, with their stack/heap
breakdowns and timestamps, not an automatic resident-buffer qualification.
Preserve or transfer raw traces for provenance. The extraction does not
initialize TPU and may be run on another compatible machine after transfer.

Historical 691-step logs are separate: report complete-job wall time only from
verified launch/finish timestamps for each run and state which phases are
included. Tqdm or TensorBoard actor-only metrics are not replacements for full
wall time. Do not multiply a short-window average by 691 to populate that table.
