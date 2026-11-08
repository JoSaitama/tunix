# Gradient-norm top50% (rebuttal baseline)

## Scientific question and exact rule

This direction-free baseline addresses R1's request for filtering by gradient
norm: can magnitude-based selection substitute for gradient agreement?
It does not itself test Assumption1 or guarantee preservation of minority
solutions. It is not a substitute for the independent clean-objective audit.

For each prompt's ORIGINAL four completions, use the same KL-free policy loss
and trainable parameter space as policy-score DTV:

\[
g_{jr}^{\rm pol}=\nabla_{\theta_{\rm trainable}}
\ell_{jr}^{\rm policy},\qquad q_{jr}=\|g_{jr}^{\rm pol}\|_2^2.
\]

Keep the two largest q within EACH prompt, not the largest eight across the
entire batch. Ties use original completion index (four zeros keep0/1).
Thus the precise name is **norm top50%**, not a strict `norm>median` rule.
Squaring preserves ranking; every gradient-tree leaf participates in an FP32
squared-norm reduction. There is no hash projection or full-model FP32 cast.

The actual update is

\[
\bar g=\frac{\sum_{j,r}m_{jr}\nabla\ell_{jr}^{\rm full}}
                   {\sum_{j,r}m_{jr}},
\qquad \ell^{\rm full}=\ell^{\rm policy}+\beta\ell^{\rm KL}.
\]

Beta is passed by the ORIGINAL GRPO configuration (default0.08), not added to
the norm score. Advantages stay computed on all four observed rewards, after
the original clean/mismatch pipeline. No re-normalization on the selected two,
unlike PODS. Generation/old/reference forwards and backwards keep their original
16-answer shape, then eight full gradients contribute to the mean update.
No extra rollouts, heldout set, candidate pool or historical checkpoint is needed.
Nonfinite policy/full gradients or loss abort; they are not silently filtered.

LoRA runs differentiate only LoRAParam, as DTV policy-score does. The fallback
Param space is recorded when there is no LoRA. Model dtype and AD order remain
the production ones, NOT the earlier FP32 utility control.

## Isolation and frozen settings

The ONLY shared integration is a new opt-in branch in rl_cluster.py when
`use_dynamic_batch_curation=True` AND `TUNIX_DBC_VARIANT=gradient_norm_top50`.
No original config/CLI structure, RobustTrainer, DTV/LOO, PODS/DS, reward, model,
optimizer, checkpoint format or evaluator is changed.

`run_gradient_norm_baseline.sh` uses the same current defaults/flags as the
PODS/DS frozen launchers: Gemma3-1B-IT, four prompts x four answers,3072 examples,
train_fraction0.9, one epoch (2764 training prompts/691 updates), peakLR1e-6,
original warmup/cosine/AdamW/clipping/KL/LoRA and pre/post evaluation1319.
The same experiment and mismatch seeds apply. Clean and mismatch use the same
learner factory as existing methods, so selected corrupted groups remain
corrupted in the actual update; no clean-label selector is introduced.

Important provenance: formal launch inherits the current parser's generation
length exactly as PODS/DS do (currently512). It does NOT silently substitute
the utility/HBM stress length768. If the actual comparison group's launch had
an explicit generation-length override, pass that SAME override here. The new
`runtime_config.json` captures all actual parser defaults/CLI overrides;
`launch_command.txt` captures the invocation. This does not independently verify
historical launches. HBM deliberately stresses max256+768 regardless of this.

All launch/test scripts restrict to the current physical worker, logical
process0 and four local TPU devices. They do not start distributed training
across worker0/1 and refuse to launch a second model task on the same worker.

## Outputs

Formal directories (no random suffix):

```
logs/rebuttal_grpo_gsm8k_gradient_norm_top50_seed5_mismatch0.2_TIMESTAMP/
  runtime_config.json
  launch_command.txt
  selection/run_metadata.json
  selection/norm_selection.jsonl
  selection/norm_summary.json
  tensorboard/
  checkpoints/
  model/
  results/*__stdout.log
  results/*__eval_accuracy__meta.json
  results/*__global_eval_rewards_sum.csv
  results/*__actor_train_skipped_samples.csv
  exit_code
```

The original exporter supplies the same pre-train/post-train accuracy,
partial_accuracy, format_accuracy, num_correct and total structure. Explicit
skip-eval flags still omit their corresponding stages, as in existing methods.
Selection logs include original four advantages, L2 norms AND squared norms,
selected local indices, train step, and token-based prompt hash. Token hashes
are not identical to other tools' raw-prompt-string hashes. No correctness or
clean reward labels are fabricated from gradients. Summary counts refer only
to the current process session if resumed; the launcher requires a fresh run
root by default. Usual checkpoints/model exports have the same disk cost as
the frozen framework; this is not a disk-usage optimization.

## Server validation, then training

No test is claimed passed until executed on the SERVER. No Mac model tests.

```bash
bash my_example/rebuttal/test_gradient_norm.sh cpu
bash my_example/rebuttal/test_gradient_norm.sh all
```

CPU tests use four virtual CPU devices only in their subprocess. Tiny real
JAX/NNX/Optax tests cover exact ranking, ties/zeros, masks, full/KL update,
LoRA space, original advantages, analytic production GRPO gradients, nonfinite
guards, real four-device sharded JIT, opt-in routing, serialization and launchers.

`all` also runs two real 2-update smokes (clean and20%mismatch; pre/post1example)
and one HBM test. HBM executes sequential real actor-training calls: one real
generation, two synthetic all-valid256+768 updates,16 completions per backward,
configured beta0.08 and the original691-step LR schedule. Controlled rewards
in ALL HBM phases guarantee nonzero synthetic gradient exercise; they are NOT
production correctness rewards or scientific results. It checks real norms,
two selections per group, actual optimizer update counts, all four device peak
counters and at least5% observed headroom. It does not certify arbitrary future
allocator peaks or measure training performance. `smoke`/`hbm` stages can be run
separately; neither changes production training.

After these gates pass, inspect the dry run and start a first formal seed:

```bash
bash my_example/rebuttal/run_gradient_norm_suite.sh \
  --seeds 5 0 13 21 42 --mismatch 0.2 --dry-run

# First seed only; queue the other four after reviewing its outputs.
bash my_example/rebuttal/run_gradient_norm_suite.sh --seeds 5 --mismatch 0.2
```

Use nohup for long jobs. Formal training has NO 45-minute utility-control limit.
Do not run tests concurrently with training on the same worker. Existing
norm/magnitude plots or historical gradients alone do not constitute this
TRAINING baseline. Gradients are computed online at each actual update, not
reused from a different method/model trajectory.

## Rebuttal interpretation boundary

This supplies a clearly defined magnitude-only comparator at50% retention.
DTV/LOO's adaptive retention can differ; its comparison does not isolate
direction at matched drop rate. Report that difference rather than claiming a
perfectly controlled direction-only ablation. Global sign reversal leaves
norm ranking unchanged, so this is not a guaranteed corruption detector.
Results, including negative ones, require the same seed protocol/statistics.

The FP32 control report20261108_122043 is archived separately: zero/restoration
passed; +/- directions have the expected signs. Eta1e-6 remains below the
declared resolution requirement, while eta1e-4 shows roughly55--57% magnitude
disagreement with the first-order prediction (beyond the10%+absolute tolerance).
Thus finite-step utility remains unvalidated. This baseline does not repair it
or prove Assumption1; no further utility retries/threshold searches are queued.
