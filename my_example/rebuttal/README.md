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

For each method separately, construct `d_M` at theta0 and apply
`theta_M = theta0 - eta * clip_by_global_norm(d_M)`; measure
`U_M = F_H(theta0) - F_H(theta_M)`. This is an **SGD-direction probe**, NOT an
AdamW replay. Step sizes are declared before running (full default 1e-6 and
1e-4; smoke/HBM 1e-4); they are diagnostic parameters, not changes to frozen
training hyperparameters. No temporary update is persisted. Report update norms,
changed parameter counts, repeat-evaluation difference and the numerical floor;
an unresolvable tiny positive change is not evidence of a benefit.

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
- `parameter_snapshot.py`: detached copied array values; fresh NNX wrappers for
  every restoration, strict parameter path/shape/dtype/bitwise checks.
- `run_utility_audit.sh`: safe single-worker launcher, no training entry point.
- `test_utility_cpu.sh` and `tests/`: tiny server CPU tests; no HF downloads.

Modes:

- `preflight`: no JAX import; verify checkpoint metadata, payload and source hash.
- `roundtrip`: load the actual checkpoint, perturb/restore twice in memory and
  verify the backup stays original. No generation, dataset loading or backward;
  this mode does NOT validate utility or gradient HBM.
- `smoke`: four candidate/eight heldout prompts, one matched-random repetition,
  real rollouts, gradients, exact scores and five finite-step branches.
- `hbm`: smoke PLUS full configured prompt/completion lengths with all-valid
  synthetic masks and nonzero advantages; total/policy gradients, exact Gram
  and a finite-step probe. Synthetic examples never enter scientific statistics.
  Missing peak counters mean headroom cannot be certified and this mode fails.
- `score`: default32 candidate/64 heldout prompts; first-order utility only.
- `probe`: same as score plus independent finite steps. Three random masks per
  target by default; only repetition0 is finite-stepped to bound cost.

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
