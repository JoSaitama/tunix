# GRPO fixed-checkpoint utility audit (rebuttal only)

This directory is independent of all training entry points. It never resumes a
trainer, writes checkpoint weights, changes the benchmark table, or supplies the
heldout gradient to the actual DTV selector. All model execution and tests must
be run on the server, not the Mac. The first supported input is the restored
Vanilla clean seed0 step500 checkpoint already downloaded to
`logs/utility_audit_inputs/vanilla_seed0_clean_20260717_144514`.

## Questions and evidence

| Reviewer question | Output / interpretation |
| --- | --- |
| R1: is the endogenous reference locally informative? | `reference_alignment.csv`: cosine with an independent clean gradient; zero/uninformative references are counted, not silently removed. Auxiliary evidence only. |
| R2: does agreement predict useful signal? | `unit_utility.csv`: actual DTV/LOO scores versus independent first-order utility; full/non-near-zero Spearman, prompt-cluster bootstrap interval, and norm/utility correlation in `audit_summary.json`. |
| R2 targeted check: do filtering decisions improve a separate clean objective? | `batch_utility.csv`: paired finite-step clean-loss changes from the same theta0 and the same rollouts; exact restoration after every branch. |
| Does merely using fewer units explain the result? | Matched-random masks with exactly the same per-group (or per-batch) retained counts; first-order and finite-step comparisons. |
| R1/R2: minority, correctness, negative advantages, corruption? | Kept/dropped statistics by selected/effectively corrupted groups, advantage sign, correctness, observed 0..4 group successes, unique-correct units and useful negative-advantage units. |
| When do DTV/LOO differ? | Threshold and post-cap masks separately, cap-rescue flags, actual DTV-keep/LOO-drop conflict utility and paired finite-step differences. Not a universal method-selection theorem. |

## Mathematical contract

All `g` are LOSS gradients; gradient descent steps are negative gradients.
Candidate score gradients use the KL-free repository GRPO loss on the original
observed dense reward (including optional prompt-hash rank reversal). DTV uses
the original mean reference; LOO removes the current unit. Production-style
`vmap(value_and_grad)` differentiates each completion at the fixed 4 prompts x
4 completions shape. The ordinary DTV reduction is reproduced and LOO statistics
reuse the production helper. No hash projection or approximate gradient cosine
is used for selection.

Let `F_H` be the separate CLEAN fixed-rollout clipped-GRPO policy surrogate, with
advantages and old policy log probabilities frozen at theta0, beta=0. It uses
only heldout prompts, not final test prompts. By default its dense reward branch
matches the source branch; `--heldout-reward-mode binary` is a separately declared
alternative based on the repository exact-correctness checker. Never select the
more favorable objective after seeing results.

`h = grad F_H(theta0)`, `u_i_policy = <h,g_i_policy>`, and
`u_i_full = <h,g_i_full>`, with `g_i_full` INCLUDING the configured KL beta.
`--mask-application full` matches policy-score/full-update filtering; explicit
`policy-only` retains all-unit KL while masking policy gradients. Actual
directions use the production masked-mean normalization. Scalar scores may
differ from a Gram reconstruction by floating-point reduction; decisions use
the production-style score reductions, not the reconstruction.

Version2 also accumulates alternating whole heldout four-prompt batches into
`h_A`/`h_B`, without extra backward calls, and repeats alignment against both.
A nonzero `h` or positive split-half cosine is not a true-direction guarantee.
Intervals condition on estimated h, not heldout/independent training seeds.
The mean `<r,h>` INCLUDING zero references and its candidate-cluster interval
estimate local reference informativeness, not a universal majority-noise theorem.

For each method separately, construct `d_M` at theta0 and apply
`theta_M = theta0 - eta * clip_by_global_norm(d_M)`; measure
`U_M = F_H(theta0) - F_H(theta_M)`. This is an **SGD-direction probe**, NOT an
AdamW replay. Step sizes are declared before running (full default 1e-6 and
1e-4; smoke/HBM 1e-4); they are diagnostic parameters, not changes to frozen
training hyperparameters. No temporary update is persisted. Report update norms,
changed parameter counts, repeat-evaluation difference and the numerical floor;
an unresolvable tiny positive change is not evidence of a benefit.

Version2 gates method finite steps with bounded +/- heldout SELF-gradient
controls, zero-step writeback and restoration checks. Only declared step sizes
with sign-correct, resolvable changes within `atol=1e-7 + rtol=.1 * |prediction|`
are used. Failed/unresolved sizes are recorded, not silently passed. Each method
probe has its own consistency flag; raw means are not verified benefits if a
constituent method probe is inconsistent. Repeated unchanged evaluation alone
does NOT validate precision after parameter perturbations.

This is a local surrogate-objective test. It is NOT greedy accuracy, actual
expected reward, long-term generalization, or a formal validation of Assumption1.
All clean labels and gradients beyond the training-visible reward are used only
for measurement; they never enter the candidate selector masks.

## Source configuration must be explicit

The archived commit is `6fba11c8d2e4cb00ee19e8ed45e196521d3f93d0` (`for_GRPO`).
Its model implementation matches the current restoration path. Its parser
default completion length was 768, whereas the current parser default is 512.
Its data shuffle was 42; the original source launch overrides have not all
been recovered, and no optimizer state exists. Therefore this audit does NOT
silently call `config_from_args()` or assume the current training defaults.

`--reward-mode accuracy|legacy` is required. The two branches call exactly the
corresponding reward functions in `my_example/rewards.py`. Confirm the original
launch reward mode/overrides before using numbers in rebuttal. This is NOT
recoverable from the weights. `--historical-source-config-confirmed` records
your confirmation, not an automatic verification.

Explicit initial profile: Gemma-3-1B-IT, LoRA64/alpha64, four prompts x four
rollouts, prompt256/completion768, beta0.08, clip0.2, sampling temperature1,
top-p0.95/top-k64, first3072 shuffled training prompts, first691 four-prompt
batches as train (2764 prompts), remaining308 as heldout. The actual selected
IDs and all arguments are saved. Duplicate prompts across train/heldout fail.
The audit RNG is independent and saved; it does not claim bitwise historical
rollout replay. The historical HF backbone revision is not certified simply by
restoring LoRA; optionally pin the known cache snapshot via `--backbone-path`.

`--checkpoint-training-condition` is provenance (initially clean).
`--candidate-noise 0.2` on a CLEAN checkpoint only measures local sensitivity at
that state. It is not evidence about a model trained on mismatch20. To examine
the latter, supply its own checkpoint and label the provenance correctly.

## Files and modes

- `audit_grpo_utility.py`: CLI, explicit protocol, disjoint dataset sampling,
  matched masks, statistics, outputs and failure reports.
- `utility_runtime.py`: read-only restoration, shared real rollouts, original
  GRPO derivatives, exact TPU tree dot products, sequential probes and HBM data.
- `utility_math.py`: NumPy controls, score oracle, stratified/cluster statistics.
- `utility_data.py`: independent prefix-stable partitions, fixed IDs and strict
  no-pickle rollout replay with production/config/parameter fingerprints.
- `parameter_snapshot.py`: detached copied array values; fresh NNX wrappers for
  every restoration, strict parameter path/shape/dtype/bitwise checks.
- `run_utility_audit.sh`: safe single-worker launcher, no training entry point.
- `test_utility_cpu.sh` and `tests/`: tiny server CPU tests; no HF downloads.
- `test_utility_v2.sh`: CPU -> metadata -> small numerical checks; no automatic
  large scientific audit or training.

Modes:

- `preflight`: no JAX import; verify checkpoint metadata, payload and source hash.
- `roundtrip`: load the actual checkpoint, perturb/restore twice in memory and
  verify the backup stays original. No generation, dataset loading or backward;
  this mode does NOT validate utility or gradient HBM.
- `numerics`: default4 candidate/8 heldout, declared sizes1e-6/1e-4; +/-SELF
  gradient, zero-step, post-restore loss and full16-vs-unit old-logps checks.
  NO candidate-gradient scan. Exit2 if execution completes but no size passes.
- `smoke`: four candidate/eight heldout prompts, one matched-random repetition,
  real/replayed rollouts, exact scores; five finite-step branches ONLY if the
  numerical gate passes. Geometry can finish with finite steps disabled.
- `hbm`: smoke PLUS full configured prompt/completion lengths with all-valid
  synthetic masks and nonzero advantages; total/policy gradients, exact Gram
  and a finite-step probe. Synthetic examples never enter scientific statistics.
  Missing peak counters mean headroom cannot be certified and this mode fails.
- `score`: default32 candidate/64 heldout prompts; first-order utility only.
- `probe`: same as score plus independent finite steps. Three random masks per
  target by default; only repetition0 and numerically validated sizes are
  finite-stepped. Check individual rows AND `finite_step_validated`.

TPU topology is always one logical process with four local devices, even on
physical worker1. Source checkpoint files are fingerprinted before/after. No
full candidate gradient pool is written to disk; only the current batch's raw
gradients are resident. Calls synchronize, so total/policy AD programs do not
overlap. Full16-response per-sample AD can still be expensive and can OOM; no
silent batch-size/projection fallback is allowed. A passing restore-only check
is not a gradient HBM check. Device counters are observed peaks, not a guarantee
for all future inputs. Run only on an idle worker.

## Server commands

Run the first two commands even if the TPU worker is busy; they do not use TPU.
The example **explicitly chooses accuracy reward**; verify this matches your
intended protocol rather than treating it as inferred historical provenance.

```bash
cd /home/jason_chia925_gmail_com/Project_8k/tunix
bash my_example/rebuttal/test_utility_cpu.sh
bash my_example/rebuttal/run_utility_audit.sh preflight --reward-mode accuracy
```

Only after the worker is idle and CPU/preflight pass:

```bash
bash my_example/rebuttal/run_utility_audit.sh smoke --reward-mode accuracy
bash my_example/rebuttal/run_utility_audit.sh hbm --reward-mode accuracy
```

After reviewing smoke/HBM results, run the declared scientific audit:

```bash
bash my_example/rebuttal/run_utility_audit.sh probe --reward-mode accuracy --candidate-noise 0
```

Do not automatically launch clean and noisy audits together. To audit a second
checkpoint, pass `--input-dir`, `--step`, `--expected-commit` and correct provenance.
Default output names are
`logs/rebuttal_grpo_gsm8k_utility_MODE_seedR_mismatchE_stepT_YYYYMMDD_HHMMSS/`.
R is the audit rollout seed; E is candidate corruption, not checkpoint provenance.
Explicit `--output-dir` overrides the name; existing output directories fail
rather than overwrite or resume. There are no random suffixes.

## Output and interpretation

`audit_manifest.json`: source fingerprints, code revision, explicit arguments,
historical configuration caveats, topology, backbone and LoRA details, split IDs.
The git executable/repository is optional on synced worker1; executed source
file hashes are recorded even when a git revision cannot be obtained.
`fixed_rollouts/*.npz`: same tokens, masks, observed/clean rewards, advantages,
old/ref log probabilities, correctness and audit IDs used by every method.
Saved log probabilities use portable float32 with their original dtype recorded;
the running audit retains the original dtype rather than changing the loss.
`unit_utility.csv`: unit scores, utility, actual/threshold masks and strata.
`direction_utility.csv`: first-order aggregated utility, norm, retained counts,
including every matched-random control.
`reference_alignment.csv`: auxiliary cosine, all zero/uninformative counts.
`random_controls.csv`: exact selected mask for every random control.
`batch_utility.csv`: actual finite surrogate changes, update norm, clipping,
roundoff/zero-delta flags and step sizes (absent in score/preflight mode).
`hbm_report.json`: device counters after restore, rollout, AD and probes.
`audit_summary.json`: execution status, results, correlations, paired contrasts,
bootstrap intervals, warnings; scientific acceptance is never auto-declared.

Bootstrap resamples candidate PROMPTS for unit correlations and candidate BATCHES
for finite-step differences. Intervals are conditional on this checkpoint and
this heldout-gradient estimate. They do NOT capture independent training-seed
variation or all heldout sampling noise. Report full and non-near-zero counts;
zeros/ties can inflate an all-unit correlation. Observed group correctness is a
four-rollout difficulty proxy, not true prompt difficulty. An observed unique
correct answer is not a claim about all rare reasoning modes.

Do not conflate `execution_passed=true` with reviewer evidence. Preflight has
`utility_audit_completed=false`. A near-zero heldout direction is flagged as
uninformative, not rescued by selecting a favorable subset. Single checkpoint
results cannot establish robustness over training or predict every final-test
winner. In particular, whole-group gradient sign reversal can preserve internal
scores but reverse external utility; the code/tests explicitly acknowledge this
counterexample instead of presupposing that Assumption1 must be confirmed.

## Validation status

Local work is restricted to syntax/shell/static checks per user instruction.
The CPU suite, real TPU smoke and full-length HBM tests must be run on the
server. Until those results are received, neither utility execution nor HBM
headroom is claimed to pass. Existing restore-only code/input remains unchanged.

### Snapshot alias fix and fast regression (2026-11-08)

The original implementation incorrectly retained `nnx.state(actor, LoRAParam)`
as an immutable backup. Flax0.12.6 state mappings can contain live Variable
references, so actor updates could also change that alleged backup. Comparing
actor against the same live backup after each probe was insufficient; the final
initial SHA256 check correctly rejected the run. Do not use that failed run's
finite-step comparisons. No source checkpoint or formal training model was saved.

The authoritative backup is now a tuple of copied immutable parameter arrays,
with no live NNX Variable wrappers. `materialize()` reconstructs disposable
wrappers each time. Every probe checks theta0 before updating and restores/checks
exact parameter bits afterwards, including when evaluation raises. Extra guards
surround rollout/gradient boundaries and the audit end. The original final
SHA256 check is retained. No allclose tolerance is substituted. A failure writes
paths/changed element counts/max errors to `parameter_restore_failure.json`.
The extra storage is one trainable LoRA parameter copy, not a full model copy;
its byte count is in the manifest/roundtrip report.

Server fast regression (no TPU):

```bash
env -u JAX_PLATFORM_NAME JAX_PLATFORMS=cpu .venv_jax081/bin/python -m unittest discover -s my_example/rebuttal/tests -p 'parameter_snapshot_test.py'
```

Then on an idle CURRENT worker:

```bash
bash my_example/rebuttal/run_utility_audit.sh roundtrip --reward-mode accuracy
```

If that passes, rerun smoke under nohup, followed by HBM only after inspection.
`parameter_roundtrip.passed` is a restoration check, not utility evidence.
Do not overwrite the failed report or reuse its partial finite-step results.

## Version2 acceptance and reviewer evidence

All actual tests remain SERVER-only. On the currently idle worker0, after
pulling code and activating the original environment, run under nohup:

```bash
bash my_example/rebuttal/test_utility_v2.sh \
  --replay-rollouts logs/rebuttal_grpo_gsm8k_utility_smoke_seed0_mismatch0_step500_20261108_061613 \
  --candidate-noise 0 --noise-seed 0
```

This reuses the successful smoke's answers, no generation/TFDS loading. Replay
requires the same theta0 SHA256, backbone snapshot identity, production loss/
reward sources and rollout protocol. V1 archives lack exact prompt strings and
allow only SAME-condition corruption replay. V2 archives save prompt strings
and can reapply a predeclared candidate-noise condition with the original hash
rule. Heldout always stays clean. Source files are read-only and hashed before/
after, and replay output/input must not overlap. Pin `--backbone-path` to the
recorded existing snapshot to avoid resolving a newer HF revision.

Exit0 means at least one declared +/-SELF control pair passed (read per-size
results), NOT Assumption1 passed. Exit2 with completed execution means the gate
did not pass (or input was rejected: distinguish by log). Do not repeat large
probes or loosen tolerances to obtain favorable method values. Gradient-only
`score` mode is independent of the finite-step gate, but its results must be
called gradient-alignment evidence, not verified loss improvement.

After CPU, small checks and full-length HBM review, the predeclared main
geometry audit can use e.g.128 candidate/128 independent heldout prompts:

```bash
bash my_example/rebuttal/run_utility_audit.sh score --reward-mode accuracy \
  --candidate-prompts 128 --heldout-prompts 128 --sample-seed 0 --rollout-seed 0 \
  --candidate-noise 0.2 --noise-seed 0
```

The default is still the CLEAN-trained model. This example is fixed-state
candidate-corruption sensitivity, NOT a mismatch-trained trajectory. Supply
the actual noisy checkpoint/input-dir/provenance to study that setting.
`--partition-file <data_partition.json>` reuses IDs; `--replay-rollouts <run>`
reuses exact answers. New sampling uses independent permutation streams: changing
candidate count does not move heldout IDs, and increasing either count preserves
its own ID prefix. Saved prompt hashes guard against changed dataset/format when
reusing IDs. Generation reproducibility is not assumed instead of replay hashes.

Reviewer-facing outputs:

- `assumption_report.json`: local mean-reference-dot and score/utility interval
  readouts, heldout-half sensitivity, kept/drop utility and numerical status.
  `assumption_proven=false` ALWAYS means no universal theorem; not a failed test.
  Negative/inconclusive results stay visible. No automatic DTV/LOO choice rule or
  original influence-function ranking guarantee is claimed.
- `reference_alignment.csv`: dot/cosine against h and both halves, selected/
  effective group-corruption labels. Summary counts zero references and reports
  candidate-cluster intervals, including zeros for raw mean dot.
- `unit_utility.csv` and summary: full Spearman, norm-only nonzero-gradient
  subset, split-half sensitivity, zero/tied scores and minority risks. The
  legacy nonzero-score AND nonzero-utility subset is CONDITIONAL sensitivity
  only, not the primary result.
- `filter_strata.csv`: method x selected/effective corruption x advantage sign,
  with unit/prompt denominators, dropped counts and correctness. These are
  COMPLETION drop rates within groups, not whole-group deletion rates.
- `direction_utility.csv` and summary: paired first-order differences vs
  Vanilla/LOO/count-matched random, with batch-cluster intervals. Random repeats
  are averaged WITHIN each batch before the paired bootstrap.
- `numerical_report.json`: untouched old-logps forward-shape differences,
  raw +/-SELF/zero-step controls, actual/predicted changes and per-size status.
  Precision, old logps and batch shapes are not rewritten merely to pass.

Limits: whole-group synchronous policy-gradient reversal preserves internal
scores but reverses external utility; actual dense rank reversal is not always
exact gradient sign reversal. No guarantee follows from corruption<50% by count.
Observed0..4 correct answers and unique-correct are finite-rollout difficulty/
minority proxies, not intrinsic difficulty or every rare reasoning mode.
Norm/utility correlation does not replace norm-top50 TRAINING baselines. This
diagnostic does not answer scaling/cost or figure/table evaluation discrepancies.

Memory: h_A/h_B need extra aggregate gradient trees, but no extra backward calls,
full candidate gradient pool or model exports. Keep original16-response AD;
do not infer new HBM safety from an older report. Run HBM before larger scans.

## Bounded FP32 measurement control (2026-11-08)

The archived BF16 numerical gate failed. Do not repeat candidate scans hoping
for a positive utility. `utility_fp32_control.py` is a separate measurement
control, not an alteration to v2 or any formal training method.

- Replays ONLY the SAME eight heldout prompts / 32 answers, masks, clean dense
  rewards and advantages. No generation, candidate gradients, selection or
  optimizer; no new training run or model export.
- Checks checkpoint LoRA SHA, pinned backbone snapshot, replay contract/source
  hashes and exact parameter restoration. The original BF16 files remain intact.
- Widens the diagnostic actor's floating state to FP32, uses `highest` matmul
  precision, checks FP32 logps/loss, and inspects the nested actual forward JAXPR
  for non-FP32 floating intermediates. It fails rather than silently casting
  only the final scalar or claiming unverified FP32 execution.
- Defines **NEW** FP32 theta0 old-policy logps via the per-unit/vmap forward,
  saving them under `fp32_control_rollouts/`. It does NOT replace archived BF16
  logps, re-score candidates, or claim that this objective is the production one.
- Computes h by differentiating the SAME mean(vmap(per-unit loss)) scalar
  evaluated by the probe. This avoids 16 per-unit gradient trees for a scalar
  measurement control. Four prompts x four answers and original lengths remain
  fixed. This is not an equivalence certificate for production gradient ranking.
- Zero/+/-SELF controls retain declared steps `[1e-6, 1e-4]`, clip `.1`, rtol `.1`,
  atol `1e-7`, and the 10x resolution floor. No automatic step search/relaxation.
- `hbm_report.json` records actual small-control peaks with a 5% minimum observed
  headroom guard. This is NOT a worst-case synthetic full-length HBM stress test,
  nor proof that larger FP32 scans fit.

Run on an idle SERVER worker (launcher runs CPU regression first):

```bash
bash my_example/rebuttal/run_utility_fp32_control.sh \
  --replay-rollouts logs/rebuttal_grpo_gsm8k_utility_smoke_seed0_mismatch0_step500_20261108_061613 \
  --backbone-path /home/jason_chia925_gmail_com/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752
```

The launcher enforces current-worker logical process0 / four local TPU devices,
refuses another model task, and gives the TPU subprocess a 45-minute timeout
(60-second TERM grace). It does not touch another worker or existing jobs.
Output is `logs/rebuttal_grpo_gsm8k_utility_fp32control_seed0_mismatch0_step500_TIMESTAMP/`.

Inspect `control_summary.json`, `numerical_report.json`, `control_manifest.json`,
`logp_comparison.json`, and `hbm_report.json`. Exit0 requires the numerical gate
AND verified headroom; exit2 is an unvalidated control or invalid CLI; timeout
usually exits124 (137 if forced kill); other nonzero is an execution failure.
`execution_passed` alone is NOT numerical acceptance. Raw +/- results are kept.

Even a pass validates only THIS FP32 measurement. Dtype, old-policy definition
and scalar-gradient execution change together, so it does not isolate a sole
BF16 cause, validate prior BF16 finite-step results, establish Assumption1, or
answer which method is better. No follow-on128 scan is launched. If it fails or
exceeds budget, pause finite-step work and prioritize the reviewer-requested
gradient-norm top50% TRAINING baseline; disclose the finite-step limitation.
