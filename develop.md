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

## 2026-08-01 reproducible CPU gate before a dual-worker TPU smoke

The server CPU gate must use two independent pytest processes. The general
suite initializes the JAX CPU backend with four devices, while
`AgenticGrpoLearnerTest.setUpClass()` must call `chex.set_n_cpu_devices(2)`
before any backend initialization. Combining them in one pytest process causes
a test-setup error even when the implementation is correct. Optional
SGLang-JAX tests are excluded because the AIME experiments use vLLM and the
server environment intentionally does not install `sgl_jax`.

Run the general gate on worker 0:

```bash
timeout --signal=TERM --kill-after=10s 600s \
  env JAX_PLATFORMS=cpu \
  ./.venv/bin/python -m pytest -q -x \
    -k 'not sglang_jax' \
    tests/cli/recipes/deepscaler_data_test.py \
    tests/rl/memory_bounded_curation_test.py \
    tests/rl/fixed_filter_trainer_test.py \
    tests/rl/self_inf_loo_trainer_test.py \
    tests/rl/self_inf_trainer_test.py \
    tests/rl/rl_cluster_test.py \
    tests/cli/grpo_main_test.py
```

Then run the Agentic gate in a fresh process on worker 0:

```bash
timeout --signal=TERM --kill-after=10s 300s \
  env JAX_PLATFORMS=cpu \
  ./.venv/bin/python -m pytest -q -x \
    tests/rl/agentic/agentic_grpo_learner_test.py::AgenticGrpoLearnerTest::test_checkpointing \
    tests/rl/agentic/agentic_grpo_learner_test.py::AgenticGrpoLearnerTest::test_customized_agent_env
```

Both commands must return zero before source deployment to worker 1. This CPU
gate validates routing, dataset filtering, staged trainer APIs, small-model
numerics, vLLM cluster construction, Agentic integration, and checkpoint
metadata. It does not replace the 8192-token dual-worker TPU smoke, which is
still required to validate SPMD sharding, TPU compilation/HBM, distributed
rollout, and shutdown behavior.

## 2026-08-02 successful group-policy smoke and launcher status correction

The `group_policy`, seed-0, clean smoke named
`grpo_aime_dtv_selfinf_group_policy_seed0_clean_20260801_151731` successfully
completed one global update. Both worker programs emitted status zero, train
step 1 reported loss `-0.044901`, and the actor checkpoint at step 1 was fully
committed with custom metadata `global_step=1`. The suite nevertheless recorded
exit code 1. This was a launcher false failure, not a training failure.

`run_worker` re-enabled Bash `errexit` inside a normal shell function. Shell
options set in a function affect its caller, so a later nonzero `wait` for the
background gcloud/SSH logging pipeline terminated the launcher before it could
capture the transport status and prefer the explicit remote program status.
The local worker function now runs in a subshell. Background wait status is
captured through an `if` condition that is safe under `errexit`, and the
explicit `HOST=... STATUS=...` value remains authoritative when available.
The SSH status is only a fallback if the remote program never emitted a status.
Added a shell-backed Python regression test for both the false-failure case and
the missing-program-status fallback. Bash syntax checks, `git diff --check`,
and both regression cases passed locally.

The smoke log also identifies the performance bottleneck. The full global step
took 6531.50 seconds. Its 64 actor microbatch calls reported a summed 6349.02
seconds and averaged 99.20 seconds each, accounting for approximately 97.2% of
the global-step time. Rollout began near 15:19 and drained near 15:31 while the
first training microbatches were already running; it was therefore mostly
overlapped and is not the dominant end-to-end limit. The final synchronous
checkpoint wrote about 3.3 GiB of model state and 19.9 GiB of optimizer state in
87.91 seconds, approximately 1.3% of the global-step time.

The dominant cost is the exact memory-bounded Policy-DTV implementation. Each
training microbatch contains two prompt groups, or sixteen trajectories. To
avoid the previous 320 GiB compile-time HBM requirement, it sequentially
computes per-trajectory policy-only gradients for the group sum, reevaluates
score information, and computes retained full Policy+KL update gradients. This
preserves the DTV formulas but trades substantial repeated model computation
and host dispatch for bounded HBM. The stable approximately 96.5-second time of
later microbatches, after compilation and after rollout drained, confirms that
this staged gradient work rather than logging or vLLM is the primary formal-run
bottleneck.

For strict reproduction, retain batch size 128, eight generations, 8192-token
limits, train microbatch size 2, off-policy steps 0, concurrency 1024, vLLM
maximum sequences 768, HBM utilization 0.4, and the original checkpoint
schedule. Quiet reward logging changes no reward, advantage, mask, or optimizer
math. vLLM scheduler/HBM tuning retains the intended sample count and sampling
distribution but can change request batching and is not guaranteed bitwise
identical on TPU. Raising train microbatch size from 2 to 4 retains the effective
batch and mathematical aggregate update but changes floating-point reduction
order and HBM demand. Enabling off-policy overlap changes which policy generated
a trajectory and therefore changes experiment semantics; it is not suitable
for the strict comparison. The highest-value future optimization is reducing
the repeated staged Policy-DTV gradient cost while preserving its exact score
and update definitions, rather than prioritizing rollout tuning.

## 2026-08-02 masked aggregate Policy-DTV update

Policy-DTV and Policy-DTV-LOO now retain the existing exact policy-only score
stage, threshold, retention-cap behavior, and batch/group boundaries, but no
longer compute a separate full Policy+KL update gradient for every trajectory.
After the DTV mask is finalized, rejected rows are zeroed in the Agentic
`TrainExample.completion_mask`. The original batched GRPO loss then performs
one aggregate forward/backward pass. This mask applies consistently to policy
loss, KL loss, entropy, and their native reductions without changing retained
trajectory tensors.

For the formal AIME configuration, which uses
`sequence-mean-token-mean`, the new update gradient is mathematically equal to
the mean of the retained non-empty per-trajectory full-loss gradients. Existing
all-zero completion rows remain excluded by the native GRPO denominator. The
old staged implementation could count such zero-loss rows in its external mask
denominator; the new path therefore restores baseline GRPO semantics for those
already-degenerate rows. Other configured aggregation modes deliberately use
their native batched GRPO normalization rather than imposing an external
trajectory mean.

The score phase is intentionally unchanged and still evaluates policy-only
per-trajectory gradients twice: once to construct each batch/group reference
gradient sum and once to compute self and cross dot products. The optimization
therefore reduces the update phase from up to one full-loss backward pass per
trajectory to one full-batch backward pass. It does not change the DTV or LOO
formula, policy-only score objective, threshold, selected mask, optimizer, vLLM
rollout, two-worker topology, sequence limits, or experiment naming.

Added CPU regression coverage for the staged policy trainer, equality between
the masked aggregate gradient and the explicit retained mean on a synthetic
example, and preservation of pre-existing zero completion rows. Local Python
syntax compilation and `git diff --check` passed. Full JAX tests must run in
the server environment because the Mac repository does not contain the TPU
project's Python dependency environment. A dual-worker 8192-token TPU smoke is
still required after the CPU gate to validate compilation and HBM behavior.

The first broad CPU gate after deployment collected the optional
`sglang_jax` cases in `rl_cluster_test.py` and failed because this vLLM
environment does not install the `sgl_jax` package. This is unrelated to the
masked aggregate implementation: its targeted gate completed with 13 passing
tests. The reproducible AIME/vLLM gate excludes only tests whose node ID
contains `sglang_jax`; vLLM cluster tests remain enabled. Installing a second
rollout engine solely to run irrelevant tests would change the established
environment and is not required for this experiment.

The worker-0 disk audit after the successful pre-optimization group-policy
smoke found 3.8 GiB free. The completed run occupied 14 GiB: 2.6 GiB of actor
model parameters and 12 GiB of optimizer state. Its logs occupied less than
1 MiB and its run cache only 12 KiB, so checkpoint state, especially the
optimizer, explains the observed disk decrease. `/tmp/models` contains a
separate 3.4 GiB model copy but is retained for the next reproduction smoke to
avoid changing the established vLLM launch environment. The apparent 474 GB
`/var/log/lastlog` value is a sparse logical file size; `/var/log` consumes only
1.7 GiB of allocated blocks and `lastlog` must not be deleted based on the
logical size. No material deleted-open experiment files were found.

The first masked-aggregate `group_policy` smoke began at 2026-08-02 01:29 UTC.
After its first compile-heavy actor call took 194.0 seconds, steady calls first
settled near 68.2 seconds, with several temporary 78-85 second calls while
rollout activity was still overlapping. The previous exact staged update had
settled near 96.5 seconds per actor call, so replacing per-trajectory full-loss
update gradients with one aggregate backward reduced steady actor-call time by
approximately 29 percent. It does not make Policy-DTV comparable to baseline or
original total-loss DTV because the exact policy-score phase still performs two
per-trajectory gradient passes.

`NUM_BATCHES=1` is one full global AIME batch, not one prompt or one actor
microbatch. With `batch_size=128`, eight generations per prompt, and the
established actor microbatching, it produces 1,024 trajectories and 64 actor
train calls. At 68.2 seconds per steady call, the actor portion alone is about
73 minutes, before startup, remaining rollout overlap, and checkpoint close.
The quoted historical total-loss smoke command also used checkpoint interval
999 and `TUNIX_SKIP_FINAL_CHECKPOINT=1`, whereas the current smoke saves a
step-1 checkpoint; this removes roughly the observed 88-second checkpoint cost
from the historical wall time but cannot explain a large compute difference.
Historical wall time must be established from its complete worker log rather
than inferred from the detached launcher command.

Because the 64-call full-global-batch smoke is too slow for iteration, the
next optimization replaces the two sequential singleton policy-gradient score
passes with one isolated vmapped policy-only score JIT. The JIT constructs the
same per-trajectory gradient tree, computes the exact batch/group standard and
LOO self/cross statistics inside the compiled program, and returns only the
small statistics arrays. The full Policy+KL update remains the separate masked
aggregate backward introduced above. No score, threshold, cap, mask, loss, or
optimizer formula changes.

The required fast TPU gate uses `batch_size=2`, eight generations, the formal
8192-token limits, one global step, and no checkpoint. Two prompts produce one
complete 16-trajectory actor microbatch, so this gate exercises the same score
and update compilation shape used by each formal actor call while avoiding the
other 63 repeated calls and the 14-GiB checkpoint. If this representative JIT
does not fit HBM, formal training must not be launched; if it fits, its measured
post-compile actor time provides the basis for the formal runtime estimate.

The first reduced gate exited before rollout or compilation because overriding
only `batch_size=2` left `rl_training_config.mini_batch_size=128`, violating the
learner's full-batch divisibility check. This is a gate-configuration error, not
a vmapped-score, HBM, or distributed-worker failure. The corrected gate sets
both values to 2 while retaining `train_micro_batch_size=2`; it therefore still
executes exactly one 2-prompt, 16-trajectory actor call.

The corrected reduced gate then exited before XLA compilation with
`axis 1 is out of bounds for array of dimension 1`; it did not report HBM OOM.
The new score-only vmap had sliced each trajectory to rank-one token arrays but
had not restored the singleton batch dimension required by the Agentic GRPO
loss. Original total-loss DTV explicitly performs this restoration before its
per-sample loss. The policy and LOO score vmaps now apply the same restoration,
and the synthetic score loss regression rejects inputs without the required
two-dimensional completion mask.

The batch-axis-corrected reduced gate
`grpo_aime_dtv_selfinf_group_policy_seed0_clean_20260802_021914` completed
successfully at the formal 8192-token limits. The isolated vmapped policy-score
program compiled and ran without HBM OOM, the masked aggregate full-loss update
completed with train loss `0.003309`, and global step 0 finished in 287.36
seconds. The single actor call, including first compilation, took 154.68
seconds. Final checkpointing was intentionally skipped. Both local and remote
program statuses were zero. The SSH transport returned one after the remote
program had emitted its explicit zero status; the corrected launcher properly
treated the remote program status as authoritative.

This one-call gate proves correctness and HBM feasibility but cannot separate
compile time from steady execution time. The final runtime gate should use two
global batches with `batch_size=2`, `mini_batch_size=2`, and no checkpoint. Its
second actor-call duration is the steady per-microbatch estimate needed to
project the 64 actor calls in each formal `batch_size=128` global step. No
additional full-size smoke is required before that estimate.

The successful implementation currently uses an isolated vmapped tree of exact
per-trajectory policy gradients; it does not yet use the proposed aggregate
gradient plus JVP identity. Because the same vmapped tree already computes both
`raw_self = ||g_i||^2` and cross terms, ordinary and LOO policy DTV have the
same dominant score-stage HBM shape and nearly the same compute cost in this
implementation. LOO adds only small score normalization and retention-cap mask
operations. The JVP optimization would make ordinary DTV cheaper, but exact LOO
would still need its self-gradient norm term.

The six-minute gate used a deliberately reduced global `batch_size=2`, not the
formal `batch_size=128`. It generated 16 trajectories and executed one actor
call; a formal global batch generates 1,024 trajectories and executes 64 actor
calls of the same compiled shape. The observed 154.68-second actor call includes
first compilation and must not be multiplied by 64. A second call in the same
process is required to measure steady execution time `X`; formal actor time per
global batch is approximately the one-time compile plus `63 * X`, with rollout
and synchronization added separately.

## 2026-08-02 policy-only versus total-loss DTV decision

For the paper's literal question of whether a trajectory opposes the model's
actual GRPO update, total-loss DTV is the canonical formulation. If the update
gradient is `g_total = g_policy + beta * g_KL`, then scoring alignment with
`g_total` directly measures alignment with the optimizer direction and allows
the same per-trajectory gradients to be reused for the masked update. This is
both conceptually direct and computationally efficient.

Policy-only DTV instead asks whether a trajectory opposes the reward-driven
policy component while deliberately excluding the KL regularizer from data
valuation. Applying that mask to a full Policy+KL update is a valid two-objective
design and can prevent the regularizer from being interpreted as trajectory
quality, but it is an ablation/hypothesis rather than an inherently more correct
DTV definition. Because scoring and updating use different objectives, exact
implementation requires additional gradient work and cannot reuse the original
total-loss DTV gradient tree. With formal beta `0.001`, total-loss and
policy-only rankings may be close, but this must be measured rather than
assumed.

The recommended experiment hierarchy is therefore baseline and original
total-loss DTV as primary reproduction methods; random/reward filters as fixed
controls; and policy-only DTV/LOO as optional ablations only if their additional
cost is scientifically justified. Engineering should not delay the primary
formal experiment suite solely to make policy-only DTV fast.

Repeated incomplete exits are a separate distributed lifecycle defect. A local
exception terminates the driver, but vLLM multiprocessing children, the remote
rollout service, gcloud/SSH transport, and shell logging pipelines do not share
one reliably terminated process group. Graceful shutdown can therefore report
a status while leaving a VFIO owner behind. The launcher needs bounded cleanup
traps on both hosts: signal known process groups with TERM, wait for a fixed
grace period, escalate only remaining members with KILL, and verify
`/dev/vfio/0` before returning. This does not indicate a DTV mathematical
problem.

The prior PPO and GSM8K results establish policy-only DTV as the primary
research hypothesis rather than merely an optional ablation. Its selection
objective intentionally measures alignment with reward-driven policy
improvement, while the retained trajectories are updated with the full
Policy+KL objective. KL gradients generally change direction as well as
magnitude, but they represent reference-policy regularization rather than
trajectory reward quality; including them in valuation can favor samples that
preserve the reference policy instead of samples that improve the policy
objective. Total-loss DTV remains the reproduction baseline and control.

For ordinary policy DTV, the exact score does not require materializing every
per-trajectory gradient. For a group mean policy gradient `g_bar`,
`score_i = <grad(loss_i), g_bar>` is the directional derivative of `loss_i`
along `g_bar`. It can be computed with one aggregate policy-gradient pass per
reference batch/group plus a JVP that returns per-trajectory directional
derivatives, followed by the existing single masked aggregate full-loss
update. This preserves the exact standard DTV formula while avoiding both the
slow two-pass singleton gradients and the HBM-heavy vmapped gradient tree.
Exact LOO additionally requires each `||grad(loss_i)||^2` self term, so it does
not receive the same simplification without retaining/sequentially recomputing
per-sample gradients or introducing an explicitly approximate estimator.

Directly piping `git archive` into `gcloud ... ssh` is unsafe when gcloud
retries a broken SSH connection. A retry resumes consumption of the local pipe
instead of replaying the tar stream from byte zero, so the remote extractor
receives a truncated non-tar suffix and can leave a partially updated source
tree. Dual-worker deployment must therefore create a persistent compressed
archive on worker 0, copy that file to worker 1, verify its SHA-256 checksum,
and only then extract it. This makes both transport retries and source
verification deterministic.

The two-step vmapped-score timing gate completed successfully. The first actor
call took 154.40 seconds including compilation; the second steady actor call
took 21.82 seconds. Global steps took 293.86 and 136.29 seconds respectively,
so the latter global duration must not be confused with actor compute. At 64
actor calls per formal 128-prompt global batch, the steady actor component is
approximately 23.3 minutes, and a 64-batch short run is roughly a one-day job
before final checkpoint overhead. The vmapped score is about 4.4 times faster
than the prior 96.5-second steady staged implementation, but exact JVP remains
worth implementing for ordinary policy DTV.

An exact JVP implementation does not change ordinary DTV scores: it computes
the same directional derivative `<grad(loss_i), g_bar>`. The reference
direction must be stop-gradient/frozen so no Hessian term enters, and group
scope must use each prompt group's own mean direction. Expected differences
from the vmapped reference are limited to floating-point reduction order. JVP
does not provide the exact LOO self norm, so LOO retains the validated vmapped
path. The working vmapped implementation is now the numerical oracle for CPU
tests comparing JVP scores, masks, and masked updates before a TPU gate.

The approximately 25-hour projection refers to one method, one seed, and the
64-global-batch short run. It is not a full pass over the 40,309-example
training set. If the final run uses approximately 312 global batches, linear
scaling from the current 21.82-second vmapped actor call gives roughly 121
hours, or five days, per method and seed before uncertainty from rollout
overlap, checkpoints, and system variation.

Group methods are the immediate priority. Ordinary group Policy-DTV will use
the exact JVP path. Exact group Policy-DTV-LOO retains the validated isolated
vmap because its `||g_i||^2` diagonal Gram term cannot be obtained from the
single mean-gradient directional derivative. The vmap already computes all
group self and cross terms in one compiled reverse-mode program and is likely
the strongest conventional exact baseline. Possible additional exact work is
limited to compiler/rematerialization/sharding tuning or carefully chunked
gradient-tree reductions, which trade HBM against dispatch and are not
guaranteed faster. Random-projection/Hutchinson self-norm estimates, last-layer
gradients, or gradient sketches may be faster but change LOO into explicitly
approximate methods and require separate names and validation.

Batch methods remain implemented and testable but are deferred, not removed.
Ordinary batch Policy-DTV can later use the same exact JVP identity with one
batch reference direction. Exact batch LOO retains the same self-norm obstacle
as group LOO and can reuse the isolated vmap while its runtime and semantics are
evaluated separately.
## 2026-08-02: Group Policy-DTV score optimization

- Kept the original AIME distributed vLLM rollout, dual-worker execution,
  Agentic queue, 8192-token configuration, and masked aggregate Policy+KL
  update unchanged.
- Added a lean GRPO policy-score loss. It preserves the original PPO/GSPO
  importance-ratio clipping, dual clipping, advantage handling, old-policy
  log-probability handling, completion mask, and configured loss aggregation.
  It omits reference KL, entropy, logits, and logging-only metrics from the
  score graph. The update loss remains the full Policy+KL objective.
- Changed only `group_policy` scoring to the exact directional-derivative
  formulation. For each prompt group, the implementation computes the mean
  policy gradient with a fixed `num_generations` denominator and evaluates all
  trajectory scores with an NNX JVP along that stop-gradient direction. This
  is mathematically identical to `dot(grad(L_i), mean_j grad(L_j))`; it is not
  an approximation and does not change the threshold or mask semantics.
- Kept `group_loo_policy` on the exact isolated-gradient vmap because exact LOO
  additionally requires each trajectory's squared gradient norm. It now uses
  the same lean score loss, while preserving the original LOO score, retention
  cap, threshold, diagnostics, and masked aggregate update.
- Left total-loss DTV and batch method routing unchanged. Batch Policy-DTV can
  be optimized separately after the group methods are validated.
- Added a regression comparison between the lean policy-score loss and the
  policy component of the original GRPO loss with `beta=0` for both GRPO and
  GSPO-token modes. The existing staged trainer test exercises the new group
  JVP path and masked update.
- The server Flax version does not expose `nnx.jvp`. Replaced that API call
  with a version-compatible functional path: `nnx.split(model, wrt, ...)`
  isolates the selected trainable State, `jax.grad` computes the exact group
  mean direction, `jax.jvp` computes the directional derivatives, and
  `nnx.merge` reconstructs the temporary model. No dependency upgrade is
  required, and the mathematical score is unchanged.

## 2026-08-02: Restore exact vmap as the default group Policy-DTV backend

- The matched 8192-token, 16-prompt gates completed successfully for both
  ordinary group Policy-DTV and group Policy-DTV-LOO. The gate represents one
  reduced global update: 16 prompts, eight generations per prompt, 128 total
  trajectories, and eight actor calls at train microbatch size two.
- Ordinary JVP scoring took 295.39 seconds for the first actor call and about
  45.35 seconds per steady actor call. Exact LOO vmap scoring took 173.52
  seconds for the first call and about 21.8 seconds per steady call. The JVP
  path reduced gradient-tree memory pressure but lost TPU vectorization and
  was approximately 2.1 times slower in steady state.
- Restored lean-loss exact vmap as the default ordinary Policy-DTV score
  backend. This uses the same vectorized per-trajectory gradients already
  validated by the exact LOO gate and preserves the standard DTV equation.
- Retained the exact functional JVP implementation as an explicit HBM fallback
  selected with `TUNIX_POLICY_DTV_SCORE_BACKEND=jvp`. The default is `vmap`.
  The dual-worker launcher forwards this setting to both workers. Added CPU
  regression coverage for both backends.
- Total-loss DTV remains unchanged and will be measured with the same
  16-prompt, 8192-token, no-checkpoint gate for a directly comparable timing
  baseline.
