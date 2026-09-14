# LearnAlign and GradAlign GSM8K baselines

These baselines use independent entrypoints and do not modify the existing
GRPO, DTV, or DTV-Loo launch paths.

## Frozen update configuration

`run_alignment_baseline.sh` preserves the GSM8K comparison setup: Gemma-3-1B,
four prompts per update, four training completions per prompt, learning rate
`1e-6`, KL coefficient `0.08`, and the existing dense reward.  The optional
20% mismatch remains the repository's deterministic within-group reward-rank
reversal.

Selector-only rollouts first use binary exact correctness. Under mismatch, the
same deterministic prompt hash and within-group reward-rank reversal used by the
existing experiment are applied to every LearnAlign training candidate and to
every GradAlign training candidate. GradAlign's held-out validation prompts stay
clean because their gradient defines the method's reference direction. Actual
GRPO updates independently regenerate four completions and continue to use the
existing dense reward with the same prompt-level mismatch assignment.

## LearnAlign adaptation

- Warm up on 300 prompts.
- Estimate success probability with 8 rollouts and `V=p(1-p)`.
- Compute policy-only prompt gradients in LoRA space and project them to 4096
  dimensions with deterministic sparse feature hashing.
- Keep each four-prompt rollout-generation chunk unchanged, but evaluate the
  eight-rollout gradient feature for one prompt at a time. This preserves
  rollout order, rewards, advantages, projection, and ranking while avoiding
  the 32-completion gradient-tree HBM peak.
- Compute the exact Eq. 8 row mean as `z_i dot mean(z)`, avoiding the nominal
  quadratic score matrix.
- Select the top 25%.  Warmup updates count toward the frozen total update
  budget by default.

## GradAlign adaptation

- Reserve 30 clean prompts from the existing held-out training split.
- Every 10 updates, recompute the validation direction and candidate scores.
- Use 4 candidate and 4 validation rollouts per prompt and retain the top 25%.
- Compute `cos(g_candidate, mean(g_validation))` using the same policy-only,
  projected LoRA gradients.

The paper uses far more rollouts per actual training problem and more validation
rollouts.  Those values are intentionally not copied because the frozen GSM8K
training setup uses four completions per prompt.  All deviations and run-time
values are saved in `run_metadata.json` and the method summary JSON.

The selection JSONL is audit-oriented. LearnAlign records every candidate's
clean binary outcomes, actual selector rewards, selector GRPO advantages,
clean/selector success rates, mismatch-selected/effective flags, selection
score, and final decision. GradAlign records the same fields for validation and
candidate prompts in every round; validation mismatch flags are always false.
These are selector-only values. Ordinary training metrics, checkpoints, and the
merged model continue to use the existing framework outputs.

Each run is self-contained under
`logs/<method>_seed<seed>_mismatch<ratio>_<timestamp>/`. In addition to the
method-specific `selection/` artifacts, `tensorboard/`, `checkpoints/`, and
`model/`, the launcher writes the same common `results/` artifacts as the
existing GSM8K methods:

- `<method>__grpo_<timestamp>__stdout.log`;
- `<method>__grpo_<timestamp>__eval_accuracy__meta.json`, containing the exact
  `pre-train`/`post-train` accuracy, partial accuracy, format accuracy,
  `num_correct`, and `total` structure;
- CSV and metadata JSON for every available standard exported TensorBoard tag
  (`global/eval/rewards/sum` and, when present,
  `actor/train/skipped_samples`);
- `global_eval_rewards_sum__overlay.png` when the reward scalar is available.

Passing `--skip-eval-before` or `--skip-eval-after` intentionally omits that
phase from the accuracy JSON. The full-matrix defaults run and retain both
phases; the reduced smoke commands below skip them to save time.

## Launch examples

```bash
./my_example/run_alignment_baseline.sh learnalign
./my_example/run_alignment_baseline.sh gradalign
```

For the full five-seed matrix, run clean and mismatch separately:

```bash
./my_example/run_alignment_baseline_suite.sh \
  --seeds 0 5 13 21 42 --mismatch 0

./my_example/run_alignment_baseline_suite.sh \
  --seeds 0 5 13 21 42 --mismatch 0.2
```

On the isolated worker of the current v5p-16 node, export the already validated
single-worker TPU variables in the same shell before either command.

## TPU smoke tests

After changing the LearnAlign gradient path, run the combined host/TPU memory
regression smoke on an idle single-worker TPU:

```bash
./my_example/smoke_learnalign_memory_fix.sh
```

It first runs the fast batching, configuration, curriculum, and mismatch tests,
then runs a two-update seed-5 Mismatch-20% LearnAlign job using the production
eight-rollout selector. Large smoke artifacts are written under a temporary
directory and removed after success. Failed-run artifacts are retained for
diagnosis; set `KEEP_SMOKE_ARTIFACTS=1` to retain successful artifacts too.

Before loading the model, run the fast host-side mismatch preflight. It uses the
production prompt hash/rank-reversal implementation and the real curricula to
verify an exact 20% noisy/80% clean candidate mixture, clean GradAlign
validation, selector audit records, selected batches, and residual noisy prompts
reaching the dense-reward update corruption path.

```bash
python -m unittest -v tests.my_example.alignment_mismatch_flow_test
```

Run these two reduced jobs before the full matrix. They exercise selector
rollout, projected per-completion gradient compilation, selection, a real GRPO
update, checkpoint restore, and merged-model save.

```bash
./my_example/run_alignment_baseline.sh learnalign \
  --max-train-examples 16 --train-fraction 0.5 \
  --learnalign-warmup-prompts 4 \
  --learnalign-estimation-rollouts 2 \
  --selection-ratio 2 --projection-dim 128 \
  --max-eval-examples 1 --skip-eval-before --skip-eval-after \
  --save-interval-steps 2

./my_example/run_alignment_baseline.sh gradalign \
  --max-train-examples 32 --train-fraction 0.5 \
  --gradalign-validation-prompts 4 \
  --gradalign-candidate-rollouts 2 \
  --gradalign-validation-rollouts 2 \
  --gradalign-selection-interval 2 \
  --selection-ratio 2 --projection-dim 128 \
  --max-eval-examples 1 --skip-eval-before --skip-eval-after \
  --save-interval-steps 4
```
