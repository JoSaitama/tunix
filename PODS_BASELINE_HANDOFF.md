# PODS baseline for frozen GSM8K

Implementation reference: Xu, Savani, Fang and Kolter, *Not All Rollouts are
Useful: Down-Sampling Rollouts in LLM Reinforcement Learning*, arXiv:2504.13818.
Author code (Apache-2.0), fixed revision:
https://github.com/YixuanEvenXu/pods/tree/187f43b648f69385103029c2dd0934521e17bb71

## Implementation

`my_example/pods.py` contains the max-variance selector, subset advantages,
PODS learner mixin, micro-batch splitting, metadata and per-prompt decisions.
`my_example/train.py` has one opt-in factory branch, enabled only when
`TUNIX_PODS_KEEP_COUNT` is set. The existing main, CLI, training configuration,
reward functions, optimizer, checkpoint/evaluation code and other filters are
unchanged. Do not combine PODS with dynamic batch curation / DTV / fixed filters.

The default PODS launcher keeps the frozen four prompts/update and four
generated completions/prompt, and retains two completions/prompt. It retains
691 updates, 1e-6 learning rate, beta=0.08, 256/768 prompt/completion limits,
the same Gemma-3-1B LoRA configuration, dataset split/seeding and greedy 1319-item
pre/post evaluation. The actor trains on eight selected completions/update;
this smaller update batch is the intentional PODS intervention.

Selection uses observed dense rewards **after** deterministic prompt-level
rank reversal. For each prompt enumerate top-k + bottom-(m-k) candidates,
choose the greatest sample variance, then recompute advantages on that subset
using ddof=1 and epsilon=1e-4. Keep all-zero-advantage groups; no resampling or
DAPO dynamic sampling is added. Stable reward ties use source order; equal
subset variances retain the first candidate, while m=n is an identity path.
The author's single-GPU flattened sort is replaced by grouped selection as in
their Open-R1 implementation, because our batch contains four different prompts.

Generation and reference/old-policy logps are prepared by the original learner
on all n completions. Tokens, masks, logps and auxiliary per-row values are
gathered together; only the actor update is reduced to m completions/prompt.
No reference-inference speedup is claimed. Generation count stays n in the
shared config; the PODS subclass rescales micro-batch split sizes to prompts*m.
PODS is disabled during evaluation. The selected rows retain the frozen KL
penalty; unselected rows do not contribute policy or KL gradients.

## Artifacts

`logs/pods_m2_seedX_mismatchY_TIMESTAMP/` contains:

- `selection/run_metadata.json`: n/m, beta, seeds, CLI and upstream revision.
- `selection/pods_selection.jsonl`: prompt hash, preparation chunk/update step,
  mismatch membership, observed n rewards, m local indices/rewards/advantages.
- `selection/pods_summary.json`: generated/selected counts and retention rate.
- `results/*__eval_accuracy__meta.json`: unchanged pre/post accuracy,
  partial_accuracy, format_accuracy, num_correct and total schema.
- Original stdout, TensorBoard, checkpoints, final model and exit code.

Summary counts describe the current process session. Selection JSONL appends
when explicitly resuming into the same artifact directory. A default launcher
invocation makes a new run directory; it does not silently resume an old run.

## Validation

On a host without TPU:

```bash
env JAX_PLATFORMS=cpu python -m unittest discover -s tests/my_example -p 'pods*test.py'
env JAX_PLATFORMS=cpu python -m unittest discover -s tests/my_example -p 'alignment_*test.py'
bash my_example/run_pods_baseline_suite.sh --seeds 5 0 13 21 42 --mismatch 0.2 --pods-keep-count 2 --dry-run
```

On an **idle** four-device TPU worker, with the same environment as the frozen
GSM8K runs:

```bash
env TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_VISIBLE_CHIPS=0,1,2,3 CLOUD_TPU_TASK_ID=0 \
  bash my_example/test_pods_smoke.sh
```

This runs host tests, clean/mismatch two-update real smokes with n=4/m=2 and
one-item pre/post evaluations, checks artifact counts and evaluation schema,
then launches an independent disposable HBM stress process. The stress first
performs a real rollout/update to initialize generation caches and optimizer
state, then performs two full-length synthetic updates with nonzero advantages.
It never loads old training checkpoints and does not export a full final model.
Synthetic generation/reward patches exist only in the test process.

The HBM report records each local device's memory counters when exposed. If
peak/limit counters exist it requires at least 5% headroom; otherwise execution
may pass with `headroom_verified=false`. Passing is evidence of this workload's
execution, not a guarantee of every future allocator peak. All test artifacts
are retained under `logs/pods_smoke.*/`; no historical run is deleted.

After server tests pass:

```bash
bash my_example/run_pods_baseline_suite.sh --seeds 5 0 13 21 42 --mismatch 0.2 --pods-keep-count 2
# Run clean on another idle worker, or after the first suite finishes:
bash my_example/run_pods_baseline_suite.sh --seeds 5 0 13 21 42 --mismatch 0 --pods-keep-count 2
```

Keep single-worker topology variables in the launch environment. The suite
stops at the first failed run and writes status.tsv. m=3 is a 75%-retention
alternative and m=4 is a no-selection control; choose the main configuration
before inspecting test accuracy. Do not change n=4 just to match the author's
larger rollout experiments. m=2 is a frozen-budget adaptation, not a claim that
the author's paper established it as optimal for Gemma/GSM8K.
