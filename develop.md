# AIME GRPO vLLM Development Log

## 2026-08-01 — GSM8K policy-method port for Agentic AIME/vLLM

- Added policy-score/full-update ordinary DTV and leave-one-out DTV trainers while preserving the Agentic singleton-batch restoration, static input axes, gradient norm return, and full-loss update path.
- Added the GSM8K fixed-filter implementation for `random_batch`, `random_group`, `reward_batch`, and `reward_group`. The reward method ranks the already computed advantages and does not recompute rewards or advantages after filtering.
- Added deterministic TRAIN-only reward-rank mismatch. It reverses reward assignments inside selected prompt groups before advantage computation, while evaluation rewards remain clean.
- Extended Agentic `TrainExample` with fixed-filter random values and rewards, and derived stable random values from experiment seed, expected step, and prompt-group identity so asynchronous completion order does not control selection.
- Added RL trainer routing for policy DTV, policy LOO DTV, and fixed filtering. The existing `self_inf_batch` and `self_inf_group` total-loss variants remain unchanged and distinct.
- Added `runs_xuesong/scripts/run_aime_seeded_full.sh` and `run_aime_reward_rank_noise_suite.sh`. Run checkpoints are stored under `runs_xuesong/runs/<RUN_NAME>` and logs under `runs_xuesong/logs/<RUN_NAME>` using the same run name.
- Extended the dual-worker launcher to pass the experiment, mismatch, fixed-filter, LOO, and decision-log environment variables to worker 1 explicitly.
- Added focused reward-rank mismatch and fixed-filter selection tests.
- Validation performed locally without TPU execution: Python byte-compilation, Bash syntax checking, and `git diff --check` passed. Pytest was not run because the Mac system Python does not have pytest installed; the tests must be run in the server environment.
- Confirmed the port boundary: AIME retains its original distributed vLLM rollout, two-worker orchestration, online reward computation, Agentic queue/coalescing, resharding, checkpoint, and dataset-loading paths. GSM8K is used only as the semantic reference for score gradients, LOO/fixed-filter selection, deterministic seeds, reward-rank mismatch, method names, and output naming.
- The stable reproduction branch `for_GRPO_vLLM` must remain unchanged. These uncommitted changes should be committed on a new branch named `for_GRPO_vLLM_aime`.
- Confirmed the staged experiment policy from `short_sweep_queue_20260707.md`: validate one-batch checkpoint-producing smoke runs first; start the short formal sweep at threshold `0.0`; compare `-0.05`, `0.0`, and `0.05` over 64 batches before selecting a threshold; then sweep GRPO beta over `0.0003`, `0.001`, and `0.003`; only sweep response length (`4096`, `8192`) if needed. Threshold applies to both ordinary and leave-one-out policy DTV, because LOO changes the score reference but retains the score-versus-threshold filter rule. Fixed random/reward filters do not use a DTV score threshold.
- Corrected the Agentic LOO trainer so `rl_training_config.self_influence_dot_threshold` controls its score boundary instead of hard-coding zero. Its default remains `0.0`, matching the paper and the PPO, DPO, and GSM8K GRPO implementations.
- Because worker 0 has only 18 GB free, checkpoint-producing smoke runs must be launched one method at a time. Verify and delete each run's exact run directory before starting the next method; do not queue all checkpoint-producing smoke methods together.
- The first `group_policy` smoke (`grpo_aime_dtv_selfinf_group_policy_seed0_clean_20260801_104749`) did not reach trainer construction, rollout, or score computation. Worker 1 failed during JAX TPU backend initialization because `/dev/vfio/0` returned `Device or resource busy`; worker 0 then timed out at the two-task shutdown barrier and both processes ended with status 134 (`SIGABRT`). Disk space was sufficient and no kernel OOM event was reported. This is a pre-existing TPU device-ownership/environment failure, not evidence of a policy DTV implementation failure.
- The worker-1 VFIO owners were identified as orphaned PIDs `1706364`, `1706365`, `1706367`, and `1706368` from the earlier `grpo_aime_dtv_group_total_loss_smoke_20260801T071159Z` run. Their command lines use `self_inf_group`, their parent PID is 1, and they predate the failed policy smoke. Deleting run output does not release TPU devices; these exact stale processes must be terminated and `/dev/vfio/0` rechecked before retrying.
- SIGTERM did not stop the four orphaned worker-1 processes, but an exact-PID SIGKILL released `/dev/vfio/0`; the subsequent GRPO process query and VFIO-holder query were empty. Worker 0 remained at 18 GB free while worker 1 had 60 GB free because TPU VM workers have independent root disks. The failed policy run directory was only 8 KB, so deleting it correctly produced no measurable disk-space change.
- The second `group_policy` smoke (`grpo_aime_dtv_selfinf_group_policy_seed0_clean_20260801_111403`) exposed source-version skew between workers. Worker 0 accepted `self_inf_group_policy` and proceeded to distributed vLLM initialization, while worker 1 ran an older snapshot whose `RLTrainingConfig` only accepted `self_inf_batch` and `self_inf_group`. Worker 1 exited with `ValueError`, and worker 0 then received `Connection refused` when contacting the worker-1 rollout owner. Worker 1 is intentionally a deployed source snapshot rather than a Git checkout, so every committed worker-0 update must be redeployed with `git archive` and verified by checksums before launch.
- Commit `5e1859153e0bf7d3001ba9a48c8259edf54f82af` was deployed from worker 0 to worker 1 with `git archive`. Checksums for the core trainer, learner, and launcher files matched, worker 1 recognized all new variants, byte-compilation passed, and no GRPO process or VFIO owner remained. The smoke sequence may resume one method at a time.
- The synchronized `group_policy` run `grpo_aime_dtv_selfinf_group_policy_seed0_clean_20260801_121422` completed rollout and reached the first policy-DTV train-step compilation. XLA then raised `CompileTimeHbmOom`: the compiled program required 320.13 GB HBM on a 95.74 GB device. The largest allocations came from vmapped per-sample gradient attention at the 8192/9216-token shape, and the current policy trainer constructs both score-gradient and full-update-gradient paths inside one JIT. The later status 134 was distributed shutdown propagation. Repeating the same smoke cannot succeed; a short-length logic smoke and a memory-bounded gradient implementation are required before an 8192-token formal run.

## Execution boundary

- Source changes are made in the Mac checkout, then committed and pushed.
- TPU environment setup, tests, smoke runs, and experiments are executed only
  after a manual `git pull --ff-only` in
  `/home/jason_chia925_gmail_com/Project/tunix`.
- No Mac process controls the Google TPU workers.
- Do not poll experiments in a loop.

## Verified historical inputs

Read-only base model:

```text
/home/lhf_hongfu_gmail_com/models/deepseek-r1-distill-qwen-1.5b
```

Read-only datasets:

```text
/home/lhf_hongfu_gmail_com/tunix-hf-data/deepscaler_train.json
/home/lhf_hongfu_gmail_com/tunix-hf-data/aime_eval.parquet
```

Historical environment evidence:

```text
Python 3.11.13
jax 0.9.2
jaxlib 0.9.2
libtpu 0.0.39.dev20260403+nightly
vLLM f6983f01de2bf2e92ab468fa735ebac39cddd670 (TPU build)
Flax 0.12.4
Orbax Checkpoint 0.11.36
Transformers 4.57.1
Datasets 4.8.4
```

The LHF environment has no installed `tunix` distribution under that name;
historical runs imported source from the checkout. Jason's environment must
instead install the Jason checkout in editable mode.

## 2026-08-01 implementation

Implemented locally without running a TPU experiment:

1. Added a local Hugging Face model fast path. A directory containing
   `config.json` and safetensors weights is loaded without repository listing,
   download, or writes into the LHF model directory.
2. Added a focused test proving Hugging Face APIs are skipped for a complete
   local model.
3. Updated the dual-worker wrapper with Jason-owned code, environment, output,
   cache, and temporary paths.
4. Added explicit local model/tokenizer/vLLM paths and verified LHF dataset
   defaults.
5. Added identical `NUM_BATCHES` propagation to TPU worker 0 and worker 1.
6. Added exact two-endpoint validation and a resolved launch summary.
7. Added a Jason environment bootstrap that works without system `ensurepip`.
8. Added one shared reproduction entry point for baseline, original total-loss
   DTV batch, and original total-loss DTV group.

The original `SelfInfTrainer` computes each selection score from per-sample
gradients of its configured `self.loss_fn`. With the matched
`agentic_grpo_config.beta=0.001`, this is the complete historical Policy+KL
loss. Both batch and group variants therefore preserve the original total-loss
score semantics. No Policy-DTV score-loss split was introduced in this phase.

Local validation performed:

```text
bash syntax: passed for all three operational scripts
Python bytecode compilation: passed for modified Python and its test
git diff --check: passed
pytest: not run on Mac because the Mac Python has no pytest installation
TPU execution: not run by design
```

## Server environment bootstrap

The earlier failed `python3 -m venv .venv` may have left a partial directory.
Inspect it first. If it is the failed empty environment, move it to a recoverable
name rather than deleting it:

```bash
cd /home/jason_chia925_gmail_com/Project/tunix
test -e .venv && mv .venv .venv.failed-20260801
```

Confirm the historical freeze exists and then bootstrap:

```bash
test -r pip-freeze-lhf-reference.txt
bash runs_xuesong/scripts/bootstrap_jason_env.sh
```

The first bootstrap implementation attempted to install the historical freeze.
The Git-pinned TPU vLLM build created an isolated build environment that pulled
CUDA PyTorch and NVIDIA wheels, exhausting server storage. Do not rebuild the
historical vLLM stack.

The corrected bootstrap creates a lightweight Jason-owned shim `.venv` without
pip and writes one `.pth` file containing Jason's checkout followed by the
historical LHF site-packages directory. JAX, libtpu, vLLM, and the remaining
dependencies are reused read-only; Tunix always imports from Jason's checkout.
No package download, dependency copy, or LHF environment modification occurs.

Before rerunning the corrected bootstrap, remove only the failed generated
Jason environment and purge Jason's pip download cache after verifying sizes:

```bash
cd /home/jason_chia925_gmail_com/Project/tunix
du -sh .venv /home/jason_chia925_gmail_com/.cache/pip 2>/dev/null || true
rm -rf -- /home/jason_chia925_gmail_com/Project/tunix/.venv
/home/lhf_hongfu_gmail_com/tunix/.venv/bin/python -m pip cache purge
bash runs_xuesong/scripts/bootstrap_jason_env.sh
```

The first `rm` target is the failed generated environment only. The pip command
purges cache files owned by Jason; it does not uninstall LHF packages.

After installation, verify imports on worker 0 without initializing the TPU:

```bash
source .venv/bin/activate
python -c 'import jax, tunix; print(jax.__version__); print(tunix.__file__)'
bash runs_xuesong/scripts/run_aime_reproduction_tests.sh
```

The test wrapper forces `JAX_PLATFORMS=cpu`; unit tests must not acquire TPU
devices. Repeat only the import-origin check on worker 1.

### TPU-in-use test failure diagnosis

An initial test invocation did not force CPU. JAX discovered the TPU backend,
which was already owned by PID `3528798`. The result was 17 passing tests and
69 repeated failures/errors with the same root exception:

```text
ABORTED: The TPU is already in use by process with pid 3528798
```

This does not indicate 69 code or environment failures. It confirms that JAX,
pytest, and the repository imported successfully before device acquisition.
Inspect the process once with `ps -fp 3528798`; do not terminate it without
confirming ownership and purpose. CPU unit tests can run while the TPU is busy.
A distributed smoke must wait until the allocated TPU slice is genuinely free.

## Reproduction commands

Run interactively on TPU worker 0. Replace TPU name and zone with the allocated
slice. The wrapper starts worker 1.

Baseline one-step smoke:

```bash
cd /home/jason_chia925_gmail_com/Project/tunix
METHOD=baseline \
NUM_BATCHES=1 \
TPU_NAME=<tpu-name> \
ZONE=<zone> \
bash runs_xuesong/scripts/run_aime_total_loss_reproduction.sh
```

Original total-loss DTV batch one-step smoke:

```bash
METHOD=dtv_batch_total_loss \
NUM_BATCHES=1 \
TPU_NAME=<tpu-name> \
ZONE=<zone> \
bash runs_xuesong/scripts/run_aime_total_loss_reproduction.sh
```

Original total-loss DTV group one-step smoke:

```bash
METHOD=dtv_group_total_loss \
NUM_BATCHES=1 \
TPU_NAME=<tpu-name> \
ZONE=<zone> \
bash runs_xuesong/scripts/run_aime_total_loss_reproduction.sh
```

The shared entry point fixes the matched settings:

```text
num_generations=8
max_prompt_length=1024
max_response_length=8192
beta=0.001
loss_agg_mode=sequence-mean-token-mean
degenerate_group_masking=true
trajectory logging disabled
```

It requires an explicit positive `NUM_BATCHES`. The wrapper exports the same
value to both workers before the inner launcher computes maximum, warmup, and
decay steps.

After each command completes, inspect local and remote logs once. A smoke is
accepted only when both statuses are zero, one rollout and optimizer step
complete, and a checkpoint exists under Jason's run root.

After all three one-step smokes pass, use `NUM_BATCHES=64` and set an explicit
clean output directory, for example:

```bash
RUN_NAME=grpo_aime_baseline_seed42_clean_<timestamp>
METHOD=baseline \
NUM_BATCHES=64 \
RUN_NAME="$RUN_NAME" \
RUN_ROOT="/home/jason_chia925_gmail_com/tunix-runs/saved_clean/$RUN_NAME" \
TPU_NAME=<tpu-name> \
ZONE=<zone> \
bash runs_xuesong/scripts/run_aime_total_loss_reproduction.sh
```

Use equivalent unique names for `dtv_batch_total_loss` and
`dtv_group_total_loss`. Do not overwrite historical LHF runs.

## Future method work

Do not implement Policy-DTV or fixed filters until baseline plus both original
total-loss variants pass smoke, checkpoint restore, and AIME evaluation.

Future work should proceed in reviewable stages:

1. Group Policy-DTV with a KL-free score loss and the unchanged full Policy+KL
   update loss.
2. Group Policy-DTV-LOO with group-local retention.
3. One deterministic fixed-filter implementation for Random and Reward at 5%
   and 10%.

Each stage requires focused tests, baseline regression, a one-step smoke, an
effective configuration record, and a separate Git commit.

## 2026-08-01 TPU ownership diagnosis

The server test failure reports that TPU worker 0 is already owned by PID
`3528798`. The subsequent JAX test failures are cascading backend
initialization failures, not evidence that the trainer assertions failed.

Unit tests must run with `JAX_PLATFORMS=cpu`; otherwise importing or creating a
JAX array on a TPU VM can acquire the TPU and block the later training process.

Use finite read-only inspection commands before terminating anything:

```bash
TPU_PID=3528798
ps -o user,pid,ppid,lstart,etime,stat,%cpu,%mem,args -p "$TPU_PID"
printf 'cwd: '
readlink -f "/proc/$TPU_PID/cwd"
printf 'cmdline: '
tr '\0' ' ' < "/proc/$TPU_PID/cmdline"
printf '\n'
PARENT_PID="$(ps -o ppid= -p "$TPU_PID" | tr -d ' ')"
ps -o user,pid,ppid,lstart,etime,stat,args -p "$PARENT_PID"
command -v pstree >/dev/null && pstree -aps "$TPU_PID" || true
pgrep -a -f 'pytest|grpo|deepscaler|vllm|python' || true
```

Run the same process listing on TPU worker 1 because TPU ownership is local to
each worker process host.

If and only if the command line confirms that PID `3528798` is the abandoned
pytest command, stop it gracefully:

```bash
kill -TERM 3528798
```

After a single manual recheck, use `kill -KILL 3528798` only if the confirmed
pytest process ignored SIGTERM. Do not kill an active training or vLLM process.

Future targeted tests must be invoked as:

```bash
JAX_PLATFORMS=cpu python -m pytest -q \
  tests/oss_utils_test.py \
  tests/rl/self_inf_trainer_test.py \
  tests/rl/rl_cluster_test.py \
  tests/rl/agentic/agentic_grpo_learner_test.py \
  tests/cli/grpo_main_test.py
```

After CPU tests exit, use one fresh process to verify TPU availability:

```bash
JAX_PLATFORMS=tpu python -c \
  'import jax; print(jax.devices()); print("TPU_AVAILABLE")'
```

This availability check itself briefly acquires the TPU and must exit before a
training launch.

## 2026-08-01 dual-worker smoke deployment and output layout

Worker 0 is the only Git working tree and launch host. Worker 1 receives a
committed source snapshot and runs it as the TPU VM service account. Worker 1
reuses `/home/eve/tunix/.venv`, `/home/eve/models`, and
`/home/eve/tunix-hf-data` through compatibility symlinks. Model and dataset
files are not copied between workers.

New runs use repository-local output roots by default:

```text
runs_xuesong/runs/<run-name>/checkpoints
runs_xuesong/logs/<run-name>/workers
runs_xuesong/logs/<run-name>/tensorboard
runs_xuesong/cache/<run-name>
```

These generated directories are ignored by Git. A detached launcher should
write its combined output to
`runs_xuesong/logs/<run-name>/launcher.out` and its PID to the same directory.

The first baseline smoke reached model loading, distributed JAX setup, GRPO
training initialization, and rollout dispatch. It then reported repeated
`lost its connection to the rollout owner` warnings. This is not a launch
summary error: it means the rollout owner on worker 1 restarted or exited.
Diagnose the finite worker-1 `remote.log` tail and both process states before
changing model or training settings. Do not relaunch while either worker still
has a Python training process.

## 2026-08-01 operational handoff

Added `start.md` as a self-contained handoff for preparing another two-worker
TPU VM. It documents repository cloning, committed snapshot deployment,
per-worker environment/model/dataset compatibility paths, finite disk and
process checks, healthAgent OOM recovery, journal cleanup, repository-local
experiment outputs, and detached smoke launches. The ziao2 procedure assumes
the operator logs into worker 1 and launches worker 0 remotely with
`REMOTE_WORKER_INDEX=1`; the canonical JAX process-host ordering remains
worker 0 followed by worker 1. This matches the established ziao1 launch
direction after the ziao2 login configuration was updated.

## 2026-08-01 memory-bounded policy DTV implementation

The 8192-token `group_policy` smoke compile estimated approximately 320 GiB of
HBM because the actor train step stacked a full parameter-gradient tree for
every trajectory with `vmap`. Policy DTV stacked both policy-only score
gradients and full Policy+KL update gradients in the same compiled step.

Added `tunix/rl/memory_bounded_curation.py` and changed only the new policy DTV,
policy DTV-LOO, and fixed-filter trainers. The original reproduced baseline and
total-loss DTV trainers remain unchanged.

The bounded implementation preserves the original mathematics. For each batch
or prompt group it computes the exact gradient sum in a sequential JAX loop,
then reevaluates one score gradient at a time to calculate the ordinary score
`dot(g_i, sum(g) / G)` and the strict LOO score
`dot(g_i, (sum(g) - g_i) / (G - 1))`. It does not retain a
`batch_size x parameter_tree` gradient object. After threshold and retention-cap
selection, it sequentially accumulates the masked full-loss gradients and
per-sample auxiliary values before the single optimizer update. Fixed random
and advantage-ranked filters use the same bounded masked-gradient accumulator
after constructing their unchanged deterministic masks.

This is an exact compute-for-memory trade: score gradients are evaluated twice
and update gradients once, so wall time can increase. Floating-point reduction
order changes from a vectorized tree reduction to sequential accumulation, so
minor roundoff differences are possible near an exact threshold boundary, but
the loss functions, score formulas, masks, denominator, and optimizer update
definition are unchanged. Added synthetic numerical tests for batch and group
ordinary/LOO statistics. Local syntax checks passed. The Mac environment lacks
JAX, Flax, and pytest, so CPU numerical tests and the 8192-token TPU compile
smoke must run in the server environment before treating the change as
validated.

The first bounded `group_policy` TPU smoke reached the actor train step but
aborted with `Cannot extract graph node from different trace level`. The cause
was calling an NNX gradient transform from a `jax.lax.fori_loop` that closed
over the model from the outer `nnx.jit` trace. Replaced those lax loops with
static Python loops over the known actor microbatch/group sizes. All gradient
calls now remain at the train step's NNX trace level; accumulation is still
sequential and does not stack per-sample gradient trees. Added an outer
`nnx.jit` scalar-model regression test specifically for this failure mode.

The dual-worker launcher now streams both worker commands through `tee`.
Worker-local copies remain in `workers/local.log` and `workers/remote.log`,
while the suite's per-run `nohup.log` receives the complete combined output
with worker headers. `remote.status` retains only the final remote host status,
avoiding a third full log copy on worker 0.

## 2026-08-01 staged policy DTV compilation and dataset validation

The static-loop 8192-token smoke remained in first-step tracing/compilation for
approximately one hour while both process hosts stayed alive. Worker 0 consumed
nearly one CPU core continuously and held the TPU, but the run produced no
post-rollout log or exit code. Static unrolling is therefore not an acceptable
final compilation strategy even though it avoids the NNX nested-trace error.

Policy DTV and Policy DTV-LOO now override trainer JIT construction with three
separate reusable programs: a one-sample policy-only score gradient, a
one-sample full Policy+KL update gradient, and optimizer apply. Python-level
orchestration invokes the two gradient programs sequentially, accumulates exact
batch/group ordinary or LOO statistics, applies the unchanged mask/cap rules,
and calls optimizer apply once. Each gradient program compiles for one 8192
trajectory rather than tracing all score and update reverse passes into one
large HLO module. The base trainer's JIT cache logging now tolerates this
composite staged train step.

The DeepScaleR loader now deterministically removes rows with empty `problem`
or `answer` fields before shuffling. It logs input, retained, empty-problem, and
empty-answer counts, raises on missing required columns or an empty retained
dataset, and does not synthesize answers from `solution`. Added recipe tests for
empty fields, no solution recovery, missing columns, and an entirely invalid
dataset. The original JSON remains read-only and unchanged.

The first staged-JIT CPU regression test called the returned bound train step
with the old unbound `_train_step(model, optimizer, inputs)` signature. The
production training loop correctly calls the bound staged function with only
`train_example`. Updated the regression test to use that production signature.
CPU tests are a fast gate for Python, NNX/JIT API, filtering, and small-model
numerical behavior; they do not validate TPU SPMD sharding, 8192-token compile
memory, or dual-host distributed execution, which remain TPU smoke-test gates.

The next TPU smoke exposed a production-only staged-input mismatch. Agentic
`gen_model_input_fn` includes a `GRPOConfig` object under `algo_config`; the
single-sample JIT attempted to abstract it as an array and failed before
compilation. Both full and policy-only loss lambdas already capture the intended
configs and ignore the passed object. Their compatibility parameter is now
optional, and staged policy trainers remove `algo_config` before invoking the
single-sample JIT. The CPU regression input now includes a non-array config
sentinel so this production shape is covered. The vLLM restart messages after
the exception were distributed shutdown recovery and did not indicate resumed
training.

The broader Agentic CPU suite exposed two stale test assumptions unrelated to
staged DTV. The checkpoint test inherited the server's optional
`TUNIX_SKIP_FINAL_CHECKPOINT` environment setting, so it now explicitly enables
final checkpointing for isolation. The shared mock generator now accepts and
ignores the production `internal_request_tags` argument used for deterministic
distributed rollout request IDs. Production training behavior is unchanged.

The isolated checkpoint test still restored RL `global_steps=0`. This exposed a
real final-checkpoint metadata regression: close-time saving forced model and
optimizer state but omitted custom metadata, and the RL metadata callback still
used the obsolete pre-increment `global_steps + 1` convention. Final checkpoint
saves now include custom metadata, and actor/critic callbacks record the current
completed global step because checkpoints are emitted on close after the RL
loop increments it. Model and optimizer checkpoint contents are unchanged.
