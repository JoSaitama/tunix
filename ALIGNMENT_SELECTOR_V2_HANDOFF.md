# Alignment selector v2: implementation and TPU preflight

Date: 2026-09-18. Scope: the two independent alignment baselines only.

## Repaired objective contracts

For each response, let `l_t = log pi_theta(token_t)` and `m_t` be the completion
mask. Only actor LoRA parameters are differentiated. We store **loss gradients**,
the negative of the papers' ascent gradients; flipping candidate and reference
gradients together does not change their cosine or LearnAlign pairwise scores.

- LearnAlign: `L = mean_valid_t[-A*l_t + beta*(exp(l_ref_t-l_t) - (l_ref_t-l_t) - 1)]`.
  Its negative gradient has the Eq.7 coefficient
  `A + beta*(pi_ref/pi_theta - 1)`. `beta` comes from the passed frozen GRPO config,
  not a selector-local literal. Reference logps are stop-gradient values from the
  same existing reference model. Eq.8 row-mean includes the self pair. `p(1-p)`,
  static warmup-selected subset, stable top-k, and cyclic fresh training rollouts
  are unchanged. The explicit surrogate matches the on-policy gradient; it does
  not perform clipped/off-policy selector optimization.
- GradAlign: `L = -A * sum_valid_t(l_t)`, not token mean, and no selector KL.
  Mean the unnormalized per-problem reference features before cosine. Reference
  problem identities remain fixed and responses/gradients are recomputed each
  selection round. True updates still use the frozen dense reward and KL.

## Minimum engineering adaptations retained

1. Gemma/LoRA, update batch 4 prompts x 4 completions, learning rate, update KL,
   update budget, generation settings, evaluation schema and mismatch chain are
   untouched. Original vanilla/DTV/DTV-Loo trainer/config/CLI files are untouched.
2. Both selectors retain 4096-dimensional signed feature hashing. Bucket hashing
   is unchanged; the sign hash uses a separate seeded mixing stream. Same seed
   fixes the map across every selection call. This removes the old bucket-parity
   sign defect, but does NOT prove independent hash families or a JL guarantee.
3. GradAlign compressed cosine is still an approximation to original raw-gradient
   cosine. LearnAlign's hash is still an engineering choice of gradient compression,
   not a claim to reproduce the paper's exact random matrix. Neither method uses
   full-model gradients; they use the frozen experiment's trainable LoRA space.
4. LearnAlign uses all 8 responses for binary rewards and advantages before two
   4-response backward subbatches. Each default backward handles at most 16
   completions. Per-completion gradients are differentiated before prompt averaging.
   Only compact features are transferred to CPU. Reference forward uses at most
   four completions per microbatch and finishes before actor backward.
5. Selection ratios are now method-specific: LearnAlign q=2 (approximately 50%
   of the full training pool, rounded down to a multiple of the batch size),
   GradAlign q=4 (25% of each round's candidate pool). LearnAlign warmup=300,
   selector rollouts=8/4, GradAlign validation=30 and interval=10 are unchanged.
   Binary selector vs dense true-update
   rewards, clean GradAlign reference and deterministic prompt-scoped mismatch
   remain disclosed adaptations. No extra label access was added by this repair.

### Independent selection controls

Both the Python entry point and the suite accept `--learnalign-selection-ratio 2`
and `--gradalign-selection-ratio 4`. The suite requires these before its `--`
extra-argument separator, and rejects a shared `--selection-ratio` override.
The single-method entry point retains that legacy override for old smoke/verification
commands; it affects only the method being launched. Resolved q is recorded in
`run_metadata.json` under `alignment.selection_ratio` and in the method summary.

Preview the exact seed/method/q routing without training or creating run directories:

```bash
bash my_example/run_alignment_baseline_suite.sh --seeds 5 0 13 21 42 \
  --mismatch 0.2 --learnalign-selection-ratio 2 --gradalign-selection-ratio 4 --dry-run
```

Then remove `--dry-run` to train. Use `--mismatch 0` on the separate clean node.
For 2764 training problems LearnAlign selects 1380; GradAlign's ordinary round
selects 40 of 160 candidates for 10 updates. The final one-update round selects
4 of 16. LearnAlign's 50% setting approximates the original paper's GSM8K
best-reported 4000/7473 (~53.5%) data-size point, not a proven optimum here.

## Implementation map

- `my_example/alignment_baselines/selector_objectives.py`: method objectives and
  selector/projection version definitions.
- `my_example/alignment_baselines/projection_hash.py`: separate bucket/sign streams.
- `my_example/alignment_baselines/gradient_features.py`: LoRA differentiation,
  reference forward and bounded rollout schedule. Legacy 32-completion A/B is
  disabled for v2. No global raw-gradient pool or dense P x D matrix allocation.
- `my_example/alignment_main.py` and `alignment_baselines/curriculum.py`: explicit
  method routing and matching run/summary metadata. Existing per-response selection
  records and pre/post evaluation outputs remain intact.
- `tests/my_example/alignment_selector_objectives_test.py`: mathematical gradient,
  masking, sign/bucket, deterministic map, linear aggregation and reference schedule
  tests; two actual-JAX tests also run where JAX/Flax training dependencies exist.
- `my_example/test_alignment_selector_v2.sh`: CPU/code tests, both methods x
  clean/20% mismatch end-to-end smoke, metadata/evaluation export assertions, then
  both full-length HBM tests. Smoke ONLY shortens data/warmup/eval; it does not
  lower projection dimension, selector rollouts or actual update batch.
- `my_example/test_alignment_selector_hbm.py`: isolated disposable model, two
  real updates retaining train executables/optimizer state, real generation then
  synthetic full configured lengths, candidate and clean reference modes, and
  two-prompt tails. No existing checkpoint is loaded or overwritten, no merged
  model is exported, and no raw-gradient audit files are saved.

## Running and interpreting tests

Activate the existing server `.venv_jax081`, then run:

```bash
bash my_example/test_alignment_selector_v2.sh
```

It loads `my_example/.env` without printing secrets, sets the known single-worker
four-chip bounds, clears the obsolete legacy A/B flag, and uses seed 5. Tests are
serial; do not run another training job on the same worker. Failures stop the
suite. Artifacts are under `logs/selector_v2_test.XXXXXX`. Four smoke model exports
can consume approximately 8 GB; HBM tests do not need tens of GB of raw gradients.

HBM reports contain `passed`, compiler memory estimates where this NNX/JAX version
exposes them, and device memory snapshots. If both `peak_bytes_in_use` and
`bytes_limit` exist, the default gate requires 10% measured headroom. Optional
`--max-program-gib` sets an explicit compiled-program budget; it fails if the
runtime cannot provide its analysis. A compiler estimate is NOT total live-device
memory. `headroom_verified=false` means execution succeeded but the runtime did
not expose counters needed to certify headroom. Do not describe that as a measured
10% margin. Exceptions/OOM write `passed=false` and propagate nonzero exit codes.

The stress input intentionally has nonzero advantages and full effective sequence
lengths. Synthetic text/reward assignments are a memory workload, not evidence of
selection quality. Two disposable updates do not reproduce every post-warmup or
later allocator state. Successful smoke/stress is a preflight, not a mathematical
promise of no OOM during all 691 updates. Selector KL adds real reference-forward
cost; actual v2 wall time has not been measured on TPU locally.

## Result/version policy

Objective/sign repairs can change rankings and selected prompts. Run both methods
again for all five seeds; do not mix v1 and v2 results in a v2 reproduction table.
Existing runs are not erased. Metadata records `alignment_selector_v2` and
`signed_feature_hash_v2`. The historical projection audit is explicitly blocked
under this revision because its objective/mirror is v1; use its original git
revision for historical work. No new advantage-zero or mirror-consistency audit
is introduced.

Local validation: 29 unittest cases, 27 passed and 2 actual-JAX cases skipped
because this Mac runtime has no JAX/Flax. Python syntax, shell syntax and
`git diff --check` pass. TPU/end-to-end/HBM validation remains a server requirement.
