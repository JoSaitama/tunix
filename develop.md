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

## 2026-08-02: Train microbatch HBM limit and full-smoke decision

- The original AIME reproduction path already sets
  `rl_training_config.train_micro_batch_size=2`; this was not introduced by
  the Policy-DTV port. With eight generations, each actor call processes two
  prompt groups, or sixteen trajectories.
- Exact-vmap `group_policy` gates at train microbatch sizes eight and four both
  failed with XLA HBM exhaustion. The microbatch-four failure occurred while
  compiling `score_batch` for 32 trajectories and included multi-GiB HLO
  temporaries, so it was not a stale-process or host-memory failure.
- Train microbatch size two is therefore the validated maximum for the exact
  8192-token, eight-generation Policy-DTV configuration on this v5p-16 setup.
  Increasing this parameter is not a viable speed optimization without
  changing the score implementation or experimental shape.
- The next validation is one full 128-prompt global update for each of
  `group_policy` and `group_loo_policy`, run sequentially with train
  microbatch size two and a step-1 checkpoint. Checkpoints must be removed
  between methods because worker 0 has limited disk capacity.

## 2026-08-02: Full group Policy-DTV smoke result

- The full `group_policy` smoke completed one 128-prompt global update with
  eight generations, 1024 trajectories, train microbatch size two, and 64
  accumulated actor calls. Global step zero took 2000.74 seconds, the logged
  step-1 loss was -0.042262, and the step-1 Orbax checkpoint finalized
  successfully.
- Both worker processes reported status zero (`HOST ... w-0 STATUS=0` and
  `HOST ... w-1 STATUS=0`), with `LOCAL_STATUS=0` and `REMOTE_STATUS=0`.
  The separately reported `REMOTE_SSH_STATUS=1` is a launcher/gcloud transport
  status artifact and does not override the explicit remote process status.
- Orbax reported 3.3 GiB of logical model parameters and 9.9 GiB of logical
  optimizer state. The on-disk checkpoint occupied less than 7 GiB because
  OCDBT stores chunked/compressed physical data; logical serialization volume
  is not equal to `du` usage. Earlier large disk growth included accumulated
  run directories, failed-run artifacts, logs/caches, or multiple checkpoints,
  rather than a larger model produced by Policy-DTV.
- The full step took 2000.74 seconds including the first-step compilation
  overhead. Removing roughly 131 seconds of first-call compilation gives a
  provisional steady-step estimate near 31.2 minutes. A 64-step short-sweep
  run is therefore expected to require about 34--40 hours for one Policy-DTV
  method and seed, allowing for rollout-length variability and final
  checkpointing. This estimate must be updated with the full LOO smoke and a
  multi-step stability run; it is not a one-epoch estimate.

## 2026-08-02: Historical node-2 baseline audit

- The historical run named
  `baseline-630-prompt1024-beta001-seqmean-20260515-finalonly` completed
  successfully on node 2 and contains a finalized 7.6-GiB actor checkpoint at
  step 314 plus a full AIME k=16 evaluation directory. Despite `630` in the
  directory name, TensorBoard contains 314 global training steps and the final
  checkpoint is step 314; the name must not be interpreted as 630 completed
  optimizer updates.
- The run used the intended beta-0.001/sequence-mean configuration. Its
  reference-policy KL remained finite and small, from approximately 8.28e-6
  at step 1 to 6.61e-5 at step 314. Gradient norm remained finite (about 0.194
  to 0.211), and the launcher and both actual worker processes reported status
  zero. This is sufficient evidence that beta 0.001 is stable for the matched
  AIME baseline and removes the need for an expensive three-value beta sweep.
- The historical final checkpoint is a matched one-epoch baseline for a
  one-epoch DTV result. It is not a matched final evaluation for a 64-step DTV
  short run because only the step-314 checkpoint was retained. The 64-step
  phase should be treated as method selection; final one-epoch comparison can
  reuse the historical baseline only after its saved evaluation summary and
  exact seed/config provenance are confirmed.
- The archived distributed AIME evaluation is complete despite an earlier
  failed single-worker attempt: it has `SUMMARY_OK=1`, 480 samples for 30
  problems at k=16, and checkpoint step 314. Reported metrics are avg@16
  0.04167, maj@16 0.20, pass@16 0.23333, and truncation rate 0.87292.
- The short-sweep document does not define 64 steps as the paper's final
  horizon. It explicitly defines 64-step rescue/screening runs. Because the
  handoff requires matched training duration when reusing the historical
  baseline, the formal one-epoch DTV horizon is 314 updates. A separate
  64-step run is optional when threshold and beta are already fixed. The
  efficient risk-control plan is one 314-step run with checkpoints every 64
  steps and `max_to_keep=1`, allowing a step-64 decision without restarting
  training or duplicating its first 64 updates.

## 2026-08-02: Full group Policy-DTV-LOO smoke result

- The full `group_loo_policy` smoke completed the matched 128-prompt,
  eight-generation global update with train microbatch size two and 64 actor
  calls. Global step zero took 1953.06 seconds (32.55 minutes), about 2.4%
  faster than the ordinary group Policy-DTV full smoke. The step-1 loss was
  0.012885.
- Its step-1 Orbax checkpoint finalized successfully in about 50.24 seconds.
  Orbax again reported 3.3 GiB of logical model parameters and 9.9 GiB of
  logical optimizer state. Both worker processes, local execution, and the
  parsed remote process status were zero.
- Periodic checkpoints with `max_to_keep=1` still require enough transient
  disk for the existing checkpoint and the new temporary checkpoint before
  retention cleanup. With a physical checkpoint near 7--8 GiB and only about
  18 GiB free on worker 0, saving every 64 steps may have unsafe peak-space
  headroom. The historical node-1 checkpoint layout and save cadence must be
  audited before changing the formal launcher.

## 2026-08-02: Final checkpoint and formal-run decision

- The node-1 audit confirmed that the three historical DTV runs retained only
  their final actor checkpoint: steps 311, 314, and 63. Their recipe used
  `save_interval_steps=500`; consequently, all three checkpoints were forced
  final saves rather than periodic saves. This establishes final-only saving
  as the reproduced historical behavior for runs shorter than 500 updates.
- Historical checkpoints occupy approximately 14 GiB each, whereas the
  current group Policy-DTV-LOO checkpoint occupies 7.5 GiB. The difference is
  a physical serialization/layout difference and must not be assumed to
  reduce the logical model or optimizer state. The current checkpoint is the
  relevant storage estimate for the current branch.
- Worker 0 had only 10 GiB free while the 7.5-GiB LOO smoke checkpoint was
  present. Deleting that completed smoke output should restore approximately
  17.5 GiB. This is sufficient for one final 7.5-GiB checkpoint, but it is not
  sufficient for a safe two-checkpoint overlap during retention cleanup.
- The earlier proposal to checkpoint every 64 updates is therefore withdrawn
  for the present local disk. `max_to_keep=1` does not prevent transient
  coexistence of the old checkpoint, the new checkpoint, and Orbax temporary
  files. Formal runs should use `save_interval_steps=500`, retain the forced
  final checkpoint, and start only after prior smoke checkpoints are removed.
- The formal matched configuration is 314 optimizer updates, beta 0.001,
  score threshold 0.0, clean rewards, eight generations, 8192 generation
  tokens, batch and mini-batch size 128, and train microbatch size two. Seeds
  are launched one at a time. At approximately 31--33 minutes per update, one
  method/seed is expected to require roughly 6.8--7.5 days plus final
  checkpointing; the two group Policy-DTV methods require roughly two weeks
  per seed when run sequentially.

## 2026-08-02: Historical step numbering and smoke-artifact audit

- Historical checkpoint directory names are not directly comparable across
  launcher/code versions. The older loop stored the final zero-based loop
  index: 312 updates ended at checkpoint directory 311, while the
  `officialish8k` configuration used 630 batches with a 0.5 train fraction,
  producing 315 updates and final directory 314. The current branch stores a
  one-update smoke as checkpoint directory 1. Formal duration must therefore
  be defined by configured `max_steps` and verified update metrics, not by
  assuming every checkpoint directory uses the same indexing convention.
- Directories named `base_model_k16_aime_smoke*` are evaluation outputs, not
  necessarily model checkpoints. Historical cleanup must first inventory all
  directories named `smoke`, their physical sizes, and any nested checkpoint
  or model-parameter data. Cleanup should remove only confirmed disposable
  model/checkpoint payloads while retaining launcher logs, status files,
  TensorBoard data, evaluation summaries, and sample outputs.

## 2026-08-02: Historical cleanup decision

- The historical base-model and checkpoint evaluation smoke directories are
  tiny (8 KiB to 660 KiB) and contain no model/checkpoint payloads. They should
  be retained because deleting them provides no meaningful disk recovery.
- Disk consumption is dominated by three historical actor checkpoints of
  approximately 14 GiB each. Each contains about 2.6 GiB of model parameters
  and 12 GiB of optimizer state.
- The unambiguous cleanup target is
  `selfinf-thr-m005-64step-20260707/checkpoints`. This was a 64-step screening
  run, has no archived evaluation directory, and keeps its launcher files,
  manifest, TensorBoard events, and status logs outside the checkpoint tree.
  Removing only this checkpoint should recover about 14 GiB while preserving
  the experimental record.
- The one-epoch checkpoints at steps 311 and 314 remain protected because they
  represent historical trained models with evaluation provenance. They must
  not be removed as part of smoke cleanup.

## 2026-08-02: Checkpoint-size interpretation and worker-1 archival

- Cleanup of the 64-step checkpoint is deferred. Its 14-GiB size is not caused
  by retaining 64 steps of model history: a checkpoint stores one model and
  optimizer state, whose tensor shapes do not grow with the number of completed
  updates.
- The old and current checkpoints both contain approximately 2.6 GiB of
  physical model parameters. The observed size difference is concentrated in
  optimizer state: approximately 12 GiB in the historical checkpoint versus
  about 5 GiB in the current 7.5-GiB checkpoint. This is consistent with a
  change in optimizer/Orbax physical representation, dtype, compression,
  deduplication, or shard layout. It is not evidence that the current run lost
  trained model parameters.
- Worker 1 has enough local disk to archive the historical 64-step checkpoint.
  Because earlier `gcloud ... scp` transfers failed even for a 28-MiB archive,
  a single non-resumable 14-GiB SCP is not acceptable. The archive should be
  copied with resumable `rsync` over the exact SSH command generated by gcloud,
  then verified with a relative-path SHA-256 manifest before any source
  checkpoint is removed from worker 0.
- The verified worker-1 transport endpoint is
  `sa_109738781960981123169@10.128.0.24`, using worker 0's
  `~/.ssh/google_compute_engine` key and gcloud's TPU host-key options. Both
  workers provide rsync 3.2.7. The archive transfer uses `rsync -rlptH` with
  `--partial --append-verify`, without compression or owner/group
  preservation. Source deletion is authorized only after source and
  destination relative-path SHA-256 manifest hashes match exactly.

## 2026-08-02: Formal group-policy launch and two-checkpoint policy

- The 64-step historical checkpoint was copied to worker 1 and verified with
  29 files and matching manifest hash
  `c90891774fa50bcde0a2aabeee6bae10b805097eb92212779de0bc6450a6ce9c`.
  Worker 0 then removed only the checkpoint tree while retaining logs,
  TensorBoard, launcher files, and the manifest. Free space increased to
  32 GiB; worker 1 retained 46 GiB free after the archive.
- The formal `group_policy` run uses 314 optimizer updates, seed 0, clean
  rewards, beta 0.001, DTV threshold 0.0, exact lean-vmap scoring, batch and
  mini-batch size 128, train microbatch size two, eight generations, and 8192
  generation tokens.
- Checkpoints are written every 64 updates with `max_to_keep=1`. The step-64
  checkpoint must be archived manually to worker 1 before step 128 is saved;
  the current launcher does not perform automatic cross-worker archival.
  Worker 0
  then continues normal retention and keeps only the latest checkpoint,
  ending with the final step-314 checkpoint. This yields the requested two
  useful checkpoints without retaining multiple 7.5-GiB checkpoints on the
  constrained worker-0 filesystem.
- Steps 128, 192, and 256 do not need to be copied to worker 1 unless they are
  independently required as permanent evaluation milestones. They remain
  temporary local recovery checkpoints: each replaces the preceding local
  checkpoint after a successful save. Missing the manual step-64 archive
  window means that step 64 will be deleted when step 128 is committed.
- The user selected this two-checkpoint plan for the seed-0 formal
  `group_policy` run and will perform occasional manual checks. No polling
  process or automatic synchronization is installed. The formal launch uses
  `NUM_BATCHES=314`, `MAX_STEPS_OVERRIDE=314`, checkpoint interval 64,
  `max_to_keep=1`, exact-vmap scoring, and an enabled forced final checkpoint.
- The resolved seed-0 `group_policy` launch was verified from both worker
  configuration dumps. It uses 314 batches/updates, data seed 42, model seed
  0, batch and mini-batch size 128, train microbatch size two, eight
  generations, 1024 prompt tokens, 8192 generation tokens, Agentic beta
  0.001, sequence-mean-token-mean loss, clean rewards, group Policy-DTV,
  threshold 0.0, checkpoint interval 64, and `max_to_keep=1`. The separately
  printed legacy `grpo_config.beta=0.08` is inactive because the run uses
  `training_mode=agentic_grpo`; the effective beta is the verified
  `agentic_grpo_config.beta=0.001`.

## 2026-08-04: Formal-run progress diagnostics

- The earlier roughly 35--37 hour estimate for reaching update 64 was a
  smoke-derived optimistic projection, not a deadline. Formal rollout latency
  varies with generated sequence lengths, request scheduling, and the number
  of usable groups. A missing step-64 checkpoint after 45 hours is not by
  itself evidence of a stalled process.
- Progress should be checked once from three independent signals: recent
  training-step lines in the per-run `nohup.log`, the maximum scalar step in
  TensorBoard event files, and live GRPO/VFIO ownership on both workers.
  TensorBoard uses `flush_every_n_steps=20`, so its visible maximum may lag the
  live trainer by up to approximately twenty updates. Checkpoint 64 appears
  only after update 64 completes and Orbax finishes committing the checkpoint.

## 2026-08-04: Periodic-checkpoint configuration is currently inactive

- The formal node-1 run reached global step 73 while its actor checkpoint
  directory remained empty. Source inspection identified the exact reason:
  `PeftTrainer.train()` intentionally contains no periodic save call and states
  that checkpoints are saved on close. Consequently,
  `save_interval_steps=64` is parsed into the configuration but is never
  applied during this Agentic training loop.
- The current run will force-save only its final checkpoint when
  `RLCluster.close()` calls `actor_trainer.close()`. The earlier plan describing
  automatic local checkpoints at 64, 128, 192, and 256 is invalid for the
  current commit and is superseded by this finding.
- A step-64 checkpoint cannot be reconstructed from the in-memory step-73
  process. Stopping the node-1 run would discard all completed progress, so it
  should continue as a final-only run. Periodic checkpointing requires a source
  fix and a newly started process; hot-patching files cannot change the already
  loaded trainer code safely.

## 2026-08-04: Meeting explanation of the Policy-DTV HBM bottleneck

- HBM is the TPU accelerator memory used for model parameters, optimizer
  state, activations, gradients, and XLA temporary buffers. An HBM OOM is
  independent of host RAM and filesystem capacity; free disk cannot prevent
  it.
- Standard GRPO computes one aggregated training loss and one aggregated
  backward pass for a microbatch. Policy-DTV must additionally obtain an exact
  policy-loss gradient for each trajectory in order to evaluate gradient
  alignment. For ordinary group DTV, the exact score is
  `s_i = <g_i, mean_j(g_j)>`, where `g_i` is the policy-only trajectory
  gradient. LOO-DTV uses the same per-trajectory gradients but excludes
  `g_i` from its reference mean. These score passes retain substantially more
  per-trajectory activations, gradient values, and XLA intermediates than an
  ordinary aggregate GRPO backward pass.
- The implementation already separates scoring from updating. Scoring uses
  the lean policy-only loss with beta zero and exact vmapped gradients. After
  the score mask is known, the update uses one masked aggregate Policy+KL
  backward pass with beta 0.001. This removed the previous expensive
  per-trajectory update gradients, but the exact per-trajectory score pass
  remains the peak-HBM operation.
- A larger train microbatch can improve speed by processing more trajectories
  in parallel and reducing the number of actor calls. With global batch 128,
  microbatch two requires 64 actor calls; four would require 32, and eight
  would require 16. This changes execution partitioning, not the global batch,
  number of generations, token length, DTV formula, or intended update.
- HBM usage grows sharply with microbatch size because multiple 8192-token
  activation/gradient graphs coexist. Exact-vmap experiments with microbatch
  eight and four both failed with HBM OOM. Microbatch two completed full
  128-prompt, eight-generation, 8192-token smoke tests for both group
  Policy-DTV and group Policy-DTV-LOO. It is therefore the largest empirically
  validated setting for this hardware and implementation, not an arbitrary
  conservative choice.

## 2026-08-05: Scaling to additional TPU workers

- A larger TPU slice increases aggregate HBM, but it does not increase HBM per
  accelerator. It reduces per-device memory only when model parameters,
  optimizer state, activations, and Policy-DTV gradient intermediates are
  correctly sharded over the additional devices.
- Adding worker processes without changing the distributed process count,
  process-host list, actor mesh, rollout topology, and vLLM configuration does
  not provide usable memory or speed. The current launcher and recipe are
  designed around a dual-worker v5p-16 slice; a larger slice requires an
  explicit multi-worker port and cannot be attached dynamically to an already
  running experiment.
- Data parallelism may replicate the model and optimizer on each replica, so
  aggregate HBM can increase without relieving the per-replica DTV score OOM.
  FSDP/tensor/sequence sharding can reduce per-device state, but exact
  per-trajectory gradient activations must also follow the intended sharding
  for microbatch four to become feasible.
- More workers can improve throughput by increasing parallel rollout or actor
  compute, but collective communication, resharding, compilation, synchronization,
  and a fixed global batch limit the gain. The speedup is therefore uncertain
  and non-linear. A larger-slice experiment should first reproduce the exact
  microbatch-two gate, then test microbatch four and compare peak HBM and
  steady global-step time under identical algorithmic settings.

## 2026-08-05: Reduced generation-length experiments

- The AIME/DeepScaler training pipeline can run with generation limits below
  8192 tokens. A 4096- or 2048-token cap reduces rollout work, activation
  storage, gradient intermediates, and XLA buffer pressure, and may make a
  larger train microbatch feasible.
- This is not a strict reproduction of the historical 8192-token experiment.
  Shorter caps truncate more reasoning trajectories before a final answer is
  emitted, which increases zero rewards and degenerate all-zero groups and
  changes the trajectory population scored by Policy-DTV. It therefore changes
  both training efficiency and the effective learning problem.
- The archived 8192-token baseline evaluation already reported an unusually
  high truncation rate of approximately 87.3 percent. This makes an aggressive
  reduction especially risky: the nonzero-reward rate may collapse even if the
  code continues to train normally.
- Reduced-length runs are appropriate as pilots or explicitly labeled compute
  ablations. A fair method comparison at 4096 tokens requires a matched
  4096-token baseline and matched random/reward/DTV methods. The existing
  8192-token baseline must not be used as the sole direct comparator.

## 2026-08-05: Read-only audit of `AIME GRPO_issues.pdf`

Scope: this audit traced the current AIME launcher, data pipeline, Agentic
rollout, reward and advantage construction, Policy-DTV trainers, gradient
accumulation, checkpointing, and offline AIME evaluator. No training code was
changed. The two active seed-0 jobs should not be interrupted on the basis of
this audit.

### Effective launch and update structure

- The formal path is
  `run_aime_reward_rank_noise_suite.sh -> run_aime_seeded_full.sh ->
  run_official_like_dual_worker.sh ->
  run_deepscaler_disagg_v5p16_1epoch.sh -> grpo_main_distributed.py ->
  grpo_main.py -> GRPOLearner -> RLCluster -> actor trainer`.
- The current full training batch is 128 prompts. Each prompt produces eight
  trajectories. `train_micro_batch_size=2` therefore means two prompt groups,
  or 16 trajectories, per actor call.
- RL configuration intentionally rejects an explicitly supplied
  `gradient_accumulation_steps`. It derives the effective value as
  `mini_batch_size // train_micro_batch_size = 128 // 2 = 64`. The CLI also
  removes the raw YAML field before constructing the RL configuration.
- `optax.MultiSteps` performs one optimizer update after those 64 actor
  microbatches. The Agentic learner also advances one global step after the
  same 64 microbatches. Consequently, 314 global steps are 314 optimizer
  updates, not 314 microbatch calls.
- The PDF is correct that early/raw configuration output showing accumulation
  one is misleading. Its proposed launcher fix of explicitly passing 64 is
  incorrect for this repository and would be rejected. The appropriate future
  fix is to log and assert the constructed/effective value after RL config
  initialization.

### Training-data cardinality and the 314-step contract

- The DeepScaleR JSON contains 40,315 raw rows. The current loader
  deterministically removes six rows with empty questions or answers, leaving
  40,309 rows before prompt-token-length filtering.
- A nominal 314-step run requests `314 * 128 = 40,192` prompts. The pre-length
  margin is therefore 117 rows, not 123 as stated in the PDF.
- Prompt-length filtering occurs before the final slice. If more than 117 of
  the 40,309 retained rows exceed the 1,024-token prompt limit, fewer than
  40,192 examples remain. The Agentic loader rejects/skips a final incomplete
  batch, so the job can finish before update 314. This exact post-tokenization
  count is not asserted by the launcher and must be checked once on the server.
- Dataset shuffling is deterministic for a fixed data seed. The selected
  40,192-example subset is therefore reproducible once the tokenizer, model
  files, loader revision, and seed are fixed.

### AIME 2024 evaluation semantics

- The AIME parquet path is opened by the training recipe, but its rows are not
  returned to `GRPOLearner` and no online AIME evaluation is performed during
  training. `eval_every_n_steps` therefore does not evaluate AIME in this path.
- Final AIME evaluation is a separate script. It iterates all parquet rows and
  supports a final partial inference batch, so a 30-question dataset does not
  have to be divisible by the training batch size of 128.
- AIME 2024 consists of 15 AIME I and 15 AIME II questions. With `k=16`, the
  evaluator produces 30 * 16 = 480 samples. `avg@16` is sample accuracy over
  480 generations; `pass@16` and `maj@16` are averaged over the 30 problems.
  The archived summary reporting 30 problems and 480 samples demonstrates
  that all 30 rows were evaluated.
- The evaluator does not independently prove that an arbitrary parquet is
  AIME 2024. It trusts the configured file. Dataset provenance, row count,
  unique-question count, missing fields, and any available year/source columns
  should be checked and recorded once; a file hash is appropriate when the
  parquet lacks provenance metadata.

### Policy-DTV and Policy-DTV-LOO mathematics

- The score loss is policy-only GRPO loss with beta zero. The masked update
  uses the normal Policy+KL loss with beta 0.001. This matches the intended
  experiment: KL does not alter trajectory ranking, while it remains active in
  the model update.
- For ordinary group DTV, the implementation computes exactly
  `s_i = <g_i, mean_j(g_j)>`. For LOO it computes exactly
  `s_i = <g_i, mean_{j != i}(g_j)>`. The group size is eight and the Agentic
  queue preserves the contiguous eight-trajectory group layout required by
  the reshape.
- The exact-vmap implementation changes execution and memory behavior but not
  those formulas. The PDF does not identify a mathematical error in the DTV
  dot-product or LOO mean.

### Confirmed high-priority normalization issue

- After selecting a mask, `make_masked_aggregate_grad_fn` zeroes rejected rows
  and applies `sequence-mean-token-mean`. This divides by the number retained
  inside each 16-trajectory actor microbatch. `optax.MultiSteps` then combines
  64 equally weighted microbatch-mean gradients.
- If retained counts differ across microbatches, this is not the same objective
  as one mean over every retained trajectory in the full 128-prompt batch.
  Microbatches with fewer retained trajectories receive disproportionate
  weight. This is partition-dependent and is a real experimental issue, not
  only a logging issue.
- A correct full-batch retained-mean implementation must accumulate gradient
  numerators and effective retained counts across all 64 microbatches, then
  divide once. The count must include only rows that pass the method mask and
  have a nonempty `completion_mask`, because degenerate groups are already
  disabled by the learner.
- The PDF's conceptual numerator/count solution is correct, but simply changing
  a denominator or passing an accumulation flag is insufficient. It requires a
  coordinated accumulator/optimizer change while preserving one scheduler and
  optimizer update per global step, plus an integration test using unequal
  retained counts.
- The two active runs use the legacy microbatch-local retained-mean objective.
  They can finish and provide useful preliminary results, but should not be
  labeled final results for a paper that defines a uniform mean over all
  retained trajectories.

### Ordinary versus LOO selection policy

- Ordinary group Policy-DTV currently applies only `score >= threshold`.
  Group Policy-DTV-LOO additionally enforces a default 25-percent minimum keep,
  which retains at least two of eight trajectories per group.
- This difference is real, but it is faithful to the GSM8K reference rather
  than an accidental AIME porting bug. It becomes an experimental confound if
  the paper describes ordinary DTV and LOO as differing only in whether the
  self term is included.
- Before new final seeds, the experiment contract must explicitly choose one
  of: a shared minimum-keep rule, threshold-only for both, or documenting the
  LOO cap as part of the method. Existing group-policy versus group-LOO-policy
  results otherwise compare both score definition and selection policy.
- Ordinary DTV also lacks the LOO path's explicit finite-value guard and
  fallback. `NaN >= threshold` is false, positive infinity is retained, and an
  all-negative group can retain zero. Training remains numerically executable
  because the masked denominator is clipped, but diagnostics and the method
  contract are incomplete.

### Fixed random/reward filters

- The method called reward filtering ranks `advantages`, not raw reward. This
  matches the GSM8K implementation. For a nondegenerate GRPO group,
  standardized advantage preserves reward ordering, so the distinction mainly
  matters for ties and degenerate groups.
- Fractional group filtering uses deterministic stochastic rounding. At a
  0.10 ratio with eight generations, a group filters one trajectory with 0.8
  probability and none with 0.2 probability; ten percent is an expectation,
  not an exact per-group count.
- These trainers share the same microbatch-local retained-mean normalization.
  At ratios that produce variable retained counts, they are affected by the
  normalization issue as well.

### Logging and checkpoint findings

- Ordinary Policy-DTV records aggregate score statistics and kept fraction but
  does not write the per-trajectory decision record supported by the LOO and
  fixed-filter paths. This does not change gradients, but it prevents complete
  post-hoc auditing of score, mask, group, and effective retained count.
- Periodic checkpoint saving is currently disabled in the trainer; the normal
  close path force-saves the final checkpoint. The active jobs therefore must
  continue to completion to produce their checkpoints.
- Adding a save directly inside the inner trainer loop, as proposed in the PDF,
  risks an off-by-one Agentic metadata state because the outer learner advances
  `global_steps` only after `update_actor` returns. A future periodic-save fix
  should occur at the completed full-batch boundary (or explicitly pass the
  completed step) and must have a real resume integration test.

### Disposition

- Do not restart the active seed-0 runs. Finish and evaluate them as preliminary
  runs with their exact code revision and the two disclosed semantics:
  microbatch-local retained means and different ordinary/LOO keep policies.
- Before launching additional final seeds, address the full-batch retained-mean
  normalization, decide and document the shared selection contract, add
  effective-config/data-cardinality assertions, and add common decision
  diagnostics. Periodic checkpoint/resume support is important operationally
  but can be implemented independently of the method mathematics.
- The PDF is strongest on the normalization, observability, and checkpoint
  risks. It is incorrect about explicitly passing gradient accumulation, stale
  by six rows in its dataset arithmetic, and treats the ordinary/LOO selection
  difference as a bug even though that difference is inherited from the GSM8K
  reference.

## 2026-08-05: Minimal final-experiment repair scope

The intended paper matrix is now limited to baseline, group Policy-DTV, and
group Policy-DTV-LOO, with one seed for each method. This reduced matrix changes
the repair priorities but not the mathematical findings above.

- `8192` is the maximum completion length per trajectory. It is unrelated to
  the coincidental arithmetic `128 * 64 = 8192`. One optimizer update uses 128
  prompts, eight trajectories per prompt (1,024 trajectories total), split
  into 64 actor calls of two prompts / 16 trajectories each.
- The 314-update horizon is an explicit historical-baseline alignment. The
  archived `officialish8k` baseline used 630 requested batches with a 0.5 train
  fraction, yielding 315 update slots and a final zero-based checkpoint
  directory numbered 314. The new launcher directly requests 314 updates.
  This is an experiment convention, not a value derived from the 8,192-token
  limit or gradient accumulation.
- Full-batch effective-trajectory normalization should be shared by all three
  final methods. Baseline has no DTV rejection, but degenerate all-zero groups
  have zero completion masks and can still make effective counts vary between
  microbatches. Correcting only DTV would therefore create a new infrastructure
  mismatch against baseline.
- For a strict paper ablation, ordinary and LOO DTV should use threshold zero,
  the same finite-value handling, and the same zero-retention behavior. The LOO
  25-percent minimum-keep cap should not remain enabled only for LOO if the
  claimed independent variable is the score formula. Removing it is closer to
  the paper rule that trajectories with score below zero are filtered. If a
  shared cap is retained for operational reasons, it must be enabled for both
  DTV variants and disclosed as part of the method.
- Degenerate groups should contribute neither a gradient numerator nor an
  effective retained count in all three methods. This preserves their existing
  absence of learning signal while preventing denominator-dependent weighting.
- Random/reward filters, mismatch data, renaming reward to advantage, and their
  ratio semantics are outside the final three-method matrix and do not block
  the final experiments. Their code can remain unchanged.
- Offline AIME evaluation is the correct design for this slow pipeline. Wiring
  AIME into the trainer or lowering `eval_every_n_steps` is unnecessary and
  would add expensive, potentially disruptive work. Dataset provenance and the
  30-problem/480-sample contract should instead be asserted in the offline
  evaluator or launch manifest.
- Periodic checkpoint and tested full-batch-boundary resume are operationally
  important for multi-day final runs. They should be repaired before restarting
  final runs, but separately from DTV score mathematics. The currently active
  processes cannot acquire this behavior through a source hot patch.
- Learning-rate schedule advancement, group contiguity, checkpoint/global-step
  agreement, and exact dataset cardinality should be implemented as assertions
  or small integration gates. They require verification, not speculative
  algorithm changes.

Minimal blocking changes before the final three runs:

1. Implement one shared full-batch numerator/count accumulation contract for
   baseline, group Policy-DTV, and group Policy-DTV-LOO.
2. Unify ordinary/LOO selection semantics and nonfinite handling.
3. Assert at launch that the token-filtered dataset supplies at least 314 full
   batches and log the actual dropped/remainder counts.
4. Log the effective constructed configuration, including accumulation 64,
   group size eight, beta, threshold, full batch, microbatch, and update count.
5. Add full-batch-boundary periodic checkpoint/resume support or explicitly
   accept final-only checkpoint risk.

Useful but nonblocking for the three-run matrix: detailed per-trajectory DTV
decision logs and an end-of-run consistency summary. Out of scope: online AIME
evaluation and all random/reward/mismatch-specific cleanup.

## 2026-08-05: Degenerate, nonfinite, zero-retention, and checkpoint semantics

No code was changed in this analysis.

### Three distinct cases that must not share an implicit fallback

1. A degenerate GRPO group has identical rewards for all eight completions.
   Group-standardized advantages are consequently all zero. With
   `degenerate_group_masking=true`, the learner zeros the completion masks, so
   the group intentionally supplies no policy-learning signal. Such rows must
   not contribute to either the gradient numerator or the effective retained
   denominator, regardless of whether a DTV threshold mask happens to label
   their zero scores as kept.
2. A nonfinite DTV score (`NaN`, positive infinity, or negative infinity) is a
   numerical failure, not an ordinary filtering decision. Current threshold
   comparison rejects NaN and negative infinity but accepts positive infinity.
   The robust contract is to require `isfinite(score)` before selection, count
   and log nonfinite values, and exclude them from both numerator and count.
   A nonfinite event should optionally fail a strict validation run because it
   can indicate unstable gradients.
3. A finite all-negative score group is a legitimate threshold outcome. It is
   essentially impossible for exact ordinary DTV in exact arithmetic because
   the sum of ordinary scores is `G * ||mean(g)||^2 >= 0`, but it can occur for
   LOO scores because the self-norm term is removed. Under the paper's strict
   `score < 0` rule, that LOO group retains zero trajectories and contributes
   neither numerator nor count. This should not silently trigger an LOO-only
   minimum-keep fallback if ordinary and LOO are intended to differ only in
   score definition.

The effective row predicate for normalization should therefore be conceptually
`method_selected & isfinite(score) & completion_mask_has_any_token`. Baseline
has no method score, so its predicate is simply
`completion_mask_has_any_token`. If a complete full batch has zero effective
rows, the optimizer update should be explicitly skipped or applied as a
well-defined zero update and logged; division by a clipped denominator alone
is not sufficient observability.

### Minimal checkpoint policy under limited disk

- Saving every step is unnecessary. Checkpoint frequency determines the amount
  of recomputation after failure; `max_to_keep` and checkpoint contents
  determine steady disk use.
- A full resumable actor checkpoint includes model parameters and optimizer
  state and is approximately 7.5 GB in the current implementation. With
  `max_to_keep=1`, only the newest completed checkpoint remains, although an
  atomic write/rotation may transiently require roughly two checkpoints
  (approximately 15 GB).
- The minimal robust policy for multi-day runs is one rolling resumable
  checkpoint plus the final checkpoint, saved only at full optimizer/global
  step boundaries. A 64-step interval limits lost work to at most 64 updates;
  a 128-step interval halves checkpoint I/O frequency but can lose several days
  of work. Neither interval accumulates all historical checkpoints when
  `max_to_keep=1` works correctly.
- Model-parameters-only checkpoints are smaller and sufficient for final AIME
  evaluation, but they cannot faithfully resume Adam/optimizer state and the
  learning-rate schedule. They are therefore suitable for an archived
  milestone, not as the sole crash-recovery checkpoint.
- Under tight Worker 0 capacity, use one rolling full checkpoint locally. If a
  scientifically useful milestone such as step 64 must be preserved, copy it
  once to Worker 1 after checksum verification, then allow Worker 0 rotation to
  replace it. The final checkpoint remains on Worker 0 and may also be archived
  after completion.
- Periodic saving should not be implemented until a full-batch-boundary save
  and resume integration test proves that checkpoint step, Agentic global
  step, optimizer state, and dataset fast-forward position agree.

## 2026-08-05: Final recommendations for LR, selection, and step 314

No source code was changed in this analysis.

### Learning-rate step units

- The current construction is structurally correct: the scheduled base
  optimizer is wrapped in `optax.MultiSteps` with an automatically derived
  accumulation interval of 64. The inner optimizer and its schedule advance
  only when the accumulated update is emitted, not on every actor microbatch.
- Therefore warmup, decay, and max-step values are intended to be measured in
  global optimizer updates. A 314-step run should traverse 314 schedule update
  points rather than 20,096 (`314 * 64`). Historical TensorBoard behavior is
  consistent with this interpretation.
- This remains an item to assert, not a reason to redesign the optimizer. A
  minimal gate should log optimizer count and LR for the first 65 microbatches:
  the first 64 calls form one update, and the schedule must advance exactly
  once. The end summary should compare optimizer, actor, Agentic global, and
  checkpoint steps.

### Recommended final DTV selection contract

- Both ordinary and LOO use their exact policy-only scores and the shared
  threshold zero.
- Both require finite scores. Nonfinite values are excluded and separately
  counted; a strict run should fail if persistent nonfinite values appear.
- Neither method uses a minimum-keep cap. This implements the paper rule
  `score < 0 -> drop` and isolates the ordinary-versus-LOO score definition.
- Degenerate zero-advantage groups are excluded through the effective
  completion mask and do not enter the numerator or denominator.
- A finite all-negative LOO group legitimately contributes zero rows. If the
  entire 128-prompt full batch has zero effective trajectories, emit a clearly
  logged zero/skip update and do not advance the optimizer or LR schedule.
  This last event should be extremely visible because advancing a scheduler
  without a learning update changes the experiment.

### Exact dataset construction for the final update

- Do not rely on the downstream learner to encounter and discard a partial
  batch. Before batching, evaluate the prompt length with the exact production
  tokenizer, after empty-field filtering and deterministic shuffling.
- Build the ordered valid-example index set, assert it contains at least
  `314 * 128 = 40,192` rows, select exactly the first 40,192 valid rows, and
  then batch with an explicit complete-batch requirement.
- This guarantees 314 batches. The final batch consists of valid selected rows
  40,064 through 40,191 and contains exactly 128 prompts. No duplication,
  padding, or partial-batch update is required.
- Record raw, empty-filtered, overlength-filtered, selected, unused, full-batch,
  and remainder counts in the run manifest. Also record a deterministic hash
  of the selected source indices or problem identifiers.
- If fewer than 40,192 valid rows exist, abort before allocating TPU resources.
  Do not silently repeat examples or pad the last batch. The clean alternatives
  are to use the common horizon `floor(valid_count / 128)` for all three methods
  (which requires a matched baseline) or deliberately revise the prompt-length
  contract and rerun all three. Preserving the historical 314 comparison makes
  the preflight abort the preferred behavior.

## 2026-08-05: Cost of cardinality and optimizer-step validation

No source code was changed in this analysis.

- The exact-cardinality change is localized to dataset preparation and a run
  manifest. It tokenizes approximately 40,000 prompts once, constructs the
  deterministic valid-index list, selects 40,192 rows, and then uses the same
  training iterator as before. This adds seconds to a few minutes of CPU startup
  work and should not change steady TPU training time, rollout volume, sequence
  length, compilation shape, or gradient computation.
- The valid-index result can be cached by dataset hash, tokenizer/model hash,
  prompt-template version, maximum prompt length, and shuffle seed. Both workers
  must validate the same manifest/hash; they do not need to recompute it for
  every restart when those inputs are unchanged.
- Verifying accumulation 64 does not require a 64-global-step AIME smoke. One
  global optimizer update contains 64 microbatch calls. Existing TensorBoard
  data can confirm that LR and optimizer-indexed metrics use global steps rather
  than thousands of microsteps, but global-only logging cannot independently
  prove what happened inside an accumulation window.
- The decisive cheap test is a synthetic CPU unit/integration test with the
  real scheduled optimizer wrapped by `optax.MultiSteps`: feed 63 small
  gradients and assert parameters and inner schedule count do not advance;
  feed the 64th and assert exactly one update; feed the 65th and assert no
  second update. This takes seconds and does not perform rollout or compile the
  1.5B model.
- Existing completed smoke/formal logs should be inspected first for LR scalar
  step indices and values. A one-global-step TPU smoke remains appropriate only
  as the final end-to-end gate after the normalization/selection/data changes,
  not as the primary test of optimizer accumulation.

## 2026-08-05: Learning-rate log audit from the active group-policy run

No source code was changed in this analysis.

- The active TensorBoard file contains 125 `actor/train/learning_rate` points at
  steps 1 through 125. This confirms that metric step indices are global actor
  updates rather than the 64-times-larger microbatch count.
- Every recorded value is approximately `9.983778e-7`. This scalar cannot be a
  faithful reading of the configured 314-step cosine schedule: a standard
  cosine decay from `1e-6` reaches approximately `6.57e-7` at schedule step 125
  and zero at step 314.
- The recipe selects `cosine_decay_schedule`, not
  `warmup_cosine_decay_schedule`. `_extract_kwargs` passes only parameters
  accepted by the selected Optax schedule, so `warmup_steps` and `warmup_ratio`
  are not used by this schedule. The effective intended schedule is cosine
  decay from `init_value=1e-6` over `decay_steps=314`, without warmup.
- `PeftTrainer._try_get_learning_rate` was written for a direct or simply
  chained injected-hyperparameter optimizer. The RL optimizer is additionally
  wrapped in `optax.MultiSteps`; the constant scalar strongly suggests that the
  logger is reading a stale or incorrect nested hyperparameter state. It does
  not by itself prove that the optimizer is using a constant LR.
- The running job should not be stopped based only on this metric. Before final
  runs, a synthetic 65-call MultiSteps test must inspect both parameter updates
  and the nested injected schedule count/value. It should establish whether the
  actual optimizer schedule advances once per 64 calls. The logger must then be
  updated to read that same authoritative nested state.
- The final experiment must explicitly choose either no-warmup cosine (matching
  the currently selected schedule type) or a real warmup-cosine schedule. Merely
  passing `warmup_steps` alongside `cosine_decay_schedule` does not enable
  warmup. The chosen schedule must be shared by baseline, DTV, and DTV-LOO.

## 2026-08-05: Chosen LR contract and consolidated pre-implementation list

No training source code was changed in this analysis.

### Chosen learning-rate contract

- Final baseline, group Policy-DTV, and group Policy-DTV-LOO will all use
  `cosine_decay_schedule`, `init_value=1e-6`, `decay_steps=314`, and no warmup.
- The unused `warmup_ratio`, `warmup_steps`, and `end_value` fields should not
  be presented as effective schedule parameters. Standard Optax cosine decay
  reaches zero at the decay horizon under its default alpha.
- Optimizer construction creates the dynamic injected-hyperparameter AdamW
  transform first and then wraps it in `optax.MultiSteps(64)`. MultiSteps calls
  the inner transform only when it emits an accumulated update, so the expected
  actual schedule unit is one global optimizer update.
- Existing gradient-accumulation tests use a scalar or constant schedule and
  therefore cannot detect a stuck or incorrectly logged dynamic schedule.
- `PeftTrainer._try_get_learning_rate` checks a direct `hyperparams` field or a
  shallow chain. It does not explicitly traverse MultiSteps'
  `inner_opt_state`. The constant TensorBoard value is therefore a logger defect
  with strong code-level support, while a 65-call test against the exact server
  Optax version remains the authoritative proof that the real schedule state
  advances once at call 64.

### Blocking changes before the three final one-seed runs

1. Add exact prompt-cardinality preparation: production tokenizer and template,
   deterministic valid indices, at least 40,192 valid rows, exact selection of
   40,192, complete 128-prompt batches, and a selection hash/manifest.
2. Replace microbatch-local means with one RL full-batch gradient numerator and
   effective-row-count accumulation shared by baseline, group DTV, and group
   DTV-LOO. Apply the optimizer once after 64 microbatches.
3. Unify group DTV selection: threshold zero, no method-specific minimum keep,
   finite-score requirement, degenerate rows excluded by effective completion
   mask, and explicit zero-retention/full-batch-zero behavior.
4. Lock the LR contract to no-warmup cosine, remove misleading effective-config
   claims for ignored warmup fields, make the logger recursively read the
   authoritative MultiSteps inner schedule state, and add the 65-call dynamic
   schedule test.
5. Log/assert the constructed runtime configuration and end-state agreement:
   batch 128, microbatch two, accumulation 64, generations eight, response
   length 8,192, beta 0.001, threshold zero, 314 updates, optimizer/actor/global
   step agreement, and LR schedule count.
6. Implement a rolling full resumable checkpoint at completed global-step
   boundaries, preferably interval 64 and `max_to_keep=1`, with an
   interrupted-versus-continuous resume integration test. Account for the
   approximately 15-GB transient two-checkpoint rotation requirement.

### Required tests/gates

- Dataset cardinality and deterministic selection-hash test.
- Unequal-kept-count full-batch gradient equivalence test covering baseline,
  DTV, LOO, degenerate groups, and an entire zero-effective microbatch.
- Ordinary/LOO score and shared-selection tests for finite, nonfinite,
  all-negative, and zero-retention cases.
- Dynamic cosine plus MultiSteps 65-call CPU test, including parameters,
  gradient step, nested schedule count/value, and logged LR.
- Checkpoint/resume equivalence at a completed accumulation boundary.
- One full 8,192-token, dual-worker, one-global-step TPU gate for each of the
  three final methods after all CPU gates pass.

### Nonblocking but recommended

- Common per-trajectory decision records for DTV and LOO.
- Offline AIME input provenance/hash and assertions for 30 unique problems and
  480 outputs at k=16.
- A concise final run summary including filtered data counts, zero-retention and
  nonfinite counts, selected fraction, optimizer/global/checkpoint steps, and
  actual LR endpoints.

### Explicitly deferred

- Online AIME evaluation.
- Random/reward fixed-filter changes, method renaming, ratio semantics, and
  mismatch-data changes.
- Batch-scope DTV methods and additional seeds.

## 2026-08-05: Estimated implementation footprint for items 1-10

This is a planning estimate only; no training source was changed.

- Expected unique footprint: approximately 16-22 files, including 9-12
  production Python/shell files and 7-10 test files. A clean shared
  implementation is approximately 450-750 production lines plus 350-650 test
  lines, for roughly 800-1,400 added/changed lines in total. The range includes
  deletions and refactoring, not only net additions.
- Dataset cardinality and manifest: `deepscaler_data.py`, possibly shared data
  utilities, the seeded launcher, and data tests; approximately 60-110
  production plus 60-100 test lines.
- Shared full-batch numerator/count accumulation is the largest mathematical
  change. It affects `peft_trainer.py` or a new RL-specific accumulation helper,
  `memory_bounded_curation.py`, baseline trainer routing in `rl_cluster.py`, both
  policy trainers, and equivalence tests; approximately 160-280 production plus
  140-240 test lines.
- Unified selection/nonfinite/degenerate behavior affects the shared curation
  helper and both policy trainers/tests; approximately 50-100 production plus
  80-140 test lines.
- No-warmup cosine cleanup and effective configuration output affect the recipe,
  seeded launcher, config/runtime logging, and tests; approximately 35-75
  production plus 30-70 test lines.
- MultiSteps-aware LR introspection affects `peft_trainer.py` and its tests;
  approximately 25-55 production plus 50-90 test lines.
- Runtime/end summary affects `grpo_main.py`, `rl_cluster.py` or the Agentic
  learner, and launcher status output; approximately 50-100 production plus
  40-80 test lines.
- Full-batch-boundary rolling checkpoint/resume affects `agentic_rl_learner.py`,
  `peft_trainer.py`, possibly `checkpoint_manager.py`, and checkpoint/Agentic
  integration tests; approximately 100-190 production plus 100-180 test lines.
- Common DTV decision logging affects both policy trainers and potentially a
  shared logging helper; approximately 40-80 production plus 40-80 test lines.
- Offline AIME provenance/cardinality validation affects
  `eval_final_checkpoint_metrics.py` and evaluator tests; approximately 25-50
  production plus 30-60 test lines.
- Final run summary overlaps the effective-config, selection, data-manifest, and
  checkpoint work; incremental cost approximately 25-60 production plus 20-50
  test lines.

The implementation should be split into independently reviewable commits:
data/config preflight; selection and full-batch normalization; LR logging/tests;
checkpoint/resume; diagnostics/offline evaluation. Combining all work into one
patch would make mathematical regression review and TPU failure isolation much
harder.

## 2026-08-05: CPU-first validation strategy for the five implementation stages

No source code was changed in this planning analysis.

- Every implementation stage can have a blocking CPU gate. No TPU run is
  required between stages. CPU tests can validate deterministic data selection,
  mathematical gradient equivalence, optimizer/schedule state transitions,
  checkpoint/resume metadata, and diagnostics using small NNX models and
  synthetic Agentic batches.
- Stage 1 (data/config): use the production tokenizer on CPU, verify exactly
  40,192 selected examples and 314 complete batches, deterministic hashes, and
  final effective configuration. Tokenizer work requires no accelerator.
- Stage 2 (selection/normalization): use small scalar/tree models and synthetic
  groups to compare one monolithic retained mean against 64 split
  numerator/count microbatches, including unequal counts, degenerate groups,
  nonfinite scores, zero-retention LOO groups, and an all-zero full batch.
- Stage 3 (LR): run 65 and 128 synthetic MultiSteps calls using the exact server
  Optax version; assert parameter-update count, nested gradient/schedule count,
  cosine values, and logger equality. No model rollout is needed.
- Stage 4 (checkpoint/resume): in a temporary CPU directory, compare a small
  continuous run with a run interrupted at a completed accumulation boundary,
  restored, and continued. Compare model parameters, optimizer state, schedule
  count, Agentic global step, dataset position, and rolling retention behavior.
- Stage 5 (diagnostics/offline evaluation): use synthetic selection records,
  small parquet fixtures, fake 30-problem/480-sample output, and mocked launcher
  statuses. Verify JSON schemas, hashes, counts, and exit-status propagation.
- CPU tests cannot validate TPU HBM allocation, XLA SPMD compilation, real
  two-worker sharding/collectives, distributed vLLM rollout, or multi-host
  shutdown. Those risks require final TPU integration gates after all five CPU
  stages pass.
- The minimum defensible TPU plan is three sequential one-global-step, full
  8,192-token dual-worker gates: baseline, group Policy-DTV, and group
  Policy-DTV-LOO. They may be launched by one suite command, but they remain
  three method-specific jobs. There is no need for five rounds of TPU debugging.
- A one-step gate with checkpoint interval one can validate real sharded
  checkpoint writing. Full distributed resume can be optional if TPU time is
  exceptionally constrained, but the CPU resume test must pass and the first
  formal step-64 checkpoint should then be treated as a monitored operational
  milestone rather than assumed proven behavior.

## 2026-08-05: Five-stage implementation completed locally, CPU gates pending

- The two pre-fix TPU jobs should be terminated and retained only as preliminary
  logs. The new objective changes full-batch trajectory weighting for both jobs,
  and removes the LOO-only 25-percent minimum-keep policy, so continuing them
  cannot produce the final comparable results.
- Stage 1 adds opt-in strict dataset preparation, exact token-length validation,
  an exact `num_batches * batch_size` selection, and a deterministic selection
  manifest before cluster construction.
- Stage 2 adds an effective-count weighted Optax accumulation transform. It
  multiplies each local mean gradient by its active trajectory count, sums 64
  numerators, divides once by the total count, and calls the inner optimizer
  once. Baseline and policy DTV variants use it; legacy total-loss and fixed
  filters remain on their previous path. Ordinary and LOO policy selection now
  share finite threshold-zero semantics with no minimum-keep cap.
- Stage 3 locks the recipe to no-warmup cosine and adds nested optimizer-state
  LR discovery plus dynamic-schedule tests.
- Stage 4 restores periodic checkpoint attempts only after completed
  accumulation boundaries. Actor metadata records completed actor/global and
  iterator steps, and the seeded formal launcher defaults to interval 64 with
  `max_to_keep=1` supplied by its existing config.
- Stage 5 adds compact ordinary-DTV score/mask decisions, strict offline AIME
  row/uniqueness/hash validation, a data manifest, an effective runtime config
  log, and launcher/training summaries.
- `run_aime_cpu_gates.sh` runs each JAX-sensitive group in a separate process
  with a ten-minute hard timeout. The Mac workspace lacks the pinned JAX/Flax/
  Optax environment, so only syntax, shell parsing, and diff checks were run
  locally. Server CPU gates are mandatory before any TPU gate.

## 2026-08-05: CPU gate optional-backend correction

- The first server CPU-gate run passed data/evaluation, weighted accumulation
  and learning-rate, policy selection/gradient, and trainer-checkpointing
  groups. The routing group then failed only because the pinned vLLM training
  environment does not install the optional `sgl_jax` package.
- This failure does not exercise the AIME distributed-vLLM path and does not
  indicate a baseline, Policy-DTV, Policy-DTV-LOO, optimizer, or checkpoint bug.
  Installing or upgrading accelerator dependencies would risk the already
  reproduced vLLM environment and is not required.
- The routing CPU gate now excludes tests whose names contain `sglang_jax`.
  All vLLM routing and CLI tests remain included.
- Nested learning-rate lookup now uses the supported NNX `get_value()` API when
  available, avoiding the deprecation warning introduced by direct `.value`
  access. Warnings emitted inside the pinned third-party `qwix` package are
  unchanged and non-blocking.

## 2026-08-05: Final CPU-gate warning audit

- The three remaining isolated server gates completed successfully: weighted
  accumulation/LR `3 passed`, vLLM routing/CLI `45 passed` with three optional
  sglang-jax tests deselected, and Agentic loss/checkpointing `11 passed` with
  25 unrelated tests deselected. All three process statuses were zero.
- `SwigPyPacked`, `SwigPyObject`, and `swigvarlink` warnings originate during
  imports of pinned compiled dependencies and are non-blocking.
- The `tpu_info` Python-3.13 enum warning is forward-looking; the server uses
  Python 3.11 and no behavior changes in the present environment.
- The `pxla.thread_resources` messages are JAX API deprecations in existing
  sharding or test code. They do not indicate incorrect current mesh behavior.
- The `os.fork()` warning is emitted by a CPU unit test that initializes the
  vLLM routing object after JAX threads exist. That test completed, and the
  production launcher uses separate distributed processes rather than this
  in-process pytest pattern.
- The sampler `_init_cache` warnings exercise the legacy test sampler, not the
  formal distributed-vLLM rollout path.
- The integer-to-boolean scatter warning is a future JAX compatibility issue in
  a mask-focused test. It is accepted by the pinned JAX version and the relevant
  mask assertion passed; it should be repaired before a future JAX upgrade but
  is not a blocker for the pinned experiment.
- The `Mean of empty slice` warnings arise in synthetic on/off-policy tests whose
  fixtures intentionally leave some metric series empty. They do not represent
  nonfinite training loss or gradients, and all affected assertions passed.
- No warning reports TPU HBM exhaustion, a failed collective, nonfinite DTV
  scores, optimizer failure, checkpoint corruption, or a distributed-vLLM
  error. The CPU validation phase is therefore complete; the next validation
  boundary is the deliberately small TPU integration gate.

## 2026-08-05: Two-host TPU integration validation procedure

- A v5p-16 TPU VM consists of two workers that jointly execute one distributed
  method. Worker 0 launches and coordinates Worker 1; the two workers cannot be
  used as independent eight-chip experiments.
- The first integration job should run `group_policy` alone on node-ziao1. It is
  the representative exact policy-gradient selection path. After it succeeds,
  a second v5p-16 node may run one remaining method concurrently while node-ziao1
  runs the other, completing baseline and `group_loo_policy` validation in
  parallel.
- The compact gate overrides the formal `128/128/2` batch, mini-batch, and train
  microbatch sizes with `16/16/2`. It therefore executes eight microbatches and
  one real optimizer update. It preserves dual-worker distributed vLLM, eight
  generations per prompt, 8,192-token response limits, beta 0.001, clean data,
  threshold zero, the exact-vmap DTV implementation, and strict complete-batch
  dataset selection.
- The gate disables final checkpoint output and trajectory logging to minimize
  disk use. Checkpoint serialization and resume semantics are covered by the
  completed CPU tests; formal jobs retain periodic checkpoints.
- Worker 1 is a deployment tree rather than a Git checkout. Synchronization
  therefore uses a compressed `git archive` transferred with direct rsync and
  extracted in place, preserving the worker-local virtual-environment symlink
  and model/data layout. The deployed commit ID is recorded in
  `.deployed_git_head` and representative tracked-file checksums are compared
  before launch.

## 2026-08-05: node-ziao1 preflight result

- Worker 0 and Worker 1 produced identical SHA-256 hashes for all seven sampled
  implementation and launcher files, including weighted accumulation, trainer,
  ordinary/LOO policy trainers, Agentic learner, seeded wrapper, and dual-worker
  launcher. The deployment is consistent.
- Worker 0 had 33 GiB free and Worker 1 had 46 GiB free before the compact gate.
- Neither worker reported an owner for `/dev/vfio/0`, and Worker 1 reported no
  matching training process. The two Worker-0 matches were an old inactive tmux
  command line and a `tail -F` log follower; neither process owns the TPU device.
- node-ziao1 is therefore available for the compact dual-worker `group_policy`
  TPU integration gate.

## 2026-08-05: strict complete-batch CLI schema integration fix

- The first compact `group_policy` TPU gate exited on both workers before JAX,
  TPU, vLLM, model, or dataset initialization. The complete error was
  `ValueError: Key require_complete_num_batches was passed at the command line
  but isn't in config.`
- Root cause: strict complete-batch handling was implemented in dataset
  preparation and wired through `grpo_main.py`, but its default schema entry was
  omitted from `tunix/cli/base_agentic_config.yaml`. Unit tests called the data
  helper directly and therefore did not cover the production CLI parser.
- Added `require_complete_num_batches: false` to the production base Agentic
  configuration. The seeded AIME wrapper continues to override it to `true`.
- Added a regression test that constructs `GrpoPipeline` from the actual
  production base Agentic YAML with the same CLI override used by the launcher.
- The failed attempt did not allocate TPU HBM or write a checkpoint. After the
  focused CPU CLI gate passes and Worker 1 is redeployed, the same compact TPU
  gate may be retried without changing experimental semantics.

## 2026-08-05: Worker-1 targeted deployment fallback

- A subsequent Worker-0-to-Worker-1 rsync of the full 28 MiB Git archive timed
  out after transferring 32 KiB. This is consistent with earlier broken-pipe
  failures for bulk SSH/SCP streams between these TPU VM workers.
- Repeated full-archive transfers are unnecessary for this commit. The previous
  deployment had verified identical hashes for all runtime implementation files;
  the strict-schema fix changes only
  `tunix/cli/base_agentic_config.yaml` at runtime. Its regression test and
  `develop.md` do not participate in training.
- The reliable fallback is for Worker 1 to fetch that one file from the public
  GitHub repository at the exact Worker-0 commit SHA, install it atomically, and
  compare its SHA-256 with Worker 0. This preserves exact-version deployment
  without relying on an unstable bulk inter-worker stream.

## 2026-08-05: BF16 weighted-optimizer state dtype integration fix

- The second compact `group_policy` gate passed strict dataset preparation on
  both workers. Each worker deterministically selected the same 16 rows from
  40,300 valid rows; nine rows exceeded the 1,024-token prompt limit. The
  manifest remainder of 12 describes the unused full valid pool modulo 16 and
  does not indicate a partial selected gate batch.
- Rollout completed, but the first optimizer update failed in
  `weighted_multisteps`. The BF16 model caused Optax to initialize Adam moments
  and injected hyperparameters as BF16, while the first real Adam update
  promoted those leaves to FP32. The apply and skip branches of a JAX conditional
  consequently returned different dtypes, which XLA rejects.
- The launcher remained visible because a peer distributed process/vLLM
  keepalive can remain waiting after the primary training process fails. This is
  a cleanup/status-propagation symptom, not continued successful training.
- Weighted accumulation now promotes FP16/BF16 floating optimizer-state leaves
  to FP32 during initialization. The inner optimizer is initialized from an FP32
  parameter template before a defensive state-tree promotion, so scalar AdamW
  hyperparameters are not first quantized through BF16. This matches the dtype
  produced by the first real Adam update, makes every conditional branch
  type-stable, and does not
  change the effective-count numerator/denominator, DTV score, selection mask,
  schedule count, or optimizer formula. Steady-state Adam storage was already
  FP32 after the first update, so this does not add steady-state HBM usage.
- Added a JIT regression test using BF16 parameters, injected scheduled AdamW,
  an accumulation branch, and an emitting branch. This specifically exercises
  the TPU failure mode that the previous FP32 eager unit tests missed.
- This failure is new to the five-stage semantics correction, specifically the
  custom effective-count weighted accumulator. Earlier reproduced jobs used
  Optax's standard `MultiSteps` or the untouched legacy total-loss path; they
  did not execute the new conditional that preserves Adam/LR state when an
  entire accumulation window has zero effective trajectories. The change was
  required to make unequal retained counts aggregate as one full-batch retained
  mean, but its first BF16 integration exposed the missing dtype normalization.
  It is unrelated to the policy-only DTV score formula, LOO subtraction, vLLM,
  or the 8,192-token rollout length.
- A post-failure process audit still found Worker-0 PID 4162255 holding
  `/dev/vfio/0`; the old tmux command and `tail -F` remained harmless. The failed
  gate must be force-terminated before any CPU or TPU retry.
- The first BF16 JIT regression run then exposed the corresponding outer
  conditional mismatch: the emitting AdamW branch returned FP32 parameter
  updates while the accumulating branch returned BF16 zero updates. The
  accumulating branch now returns FP32 zero optimizer updates, matching AdamW.
  The persistent gradient accumulator intentionally remains in the incoming
  gradient/model dtype (BF16 in the formal run), while its scalar effective
  count remains FP32. This avoids adding a persistent model-sized FP32 gradient
  buffer and therefore avoids an unnecessary HBM increase.
- Server commit `16f1665c76c0100a90d722520c5d86c4d57efffd` passed the complete
  focused weighted-accumulation CPU gate: four tests passed in 7.83 seconds and
  the process status was zero. This covers BF16 parameters, injected scheduled
  AdamW, JIT tracing, the non-emitting accumulation branch, the emitting update
  branch, FP32 optimizer updates/state, and a persistent BF16 gradient sum.
- The remaining SWIG import deprecation warnings are unchanged and non-blocking.
  Worker 1 may now receive the exact runtime files from this commit and the
  compact dual-worker `group_policy` TPU gate may be retried.

## 2026-08-05: successful group-policy TPU gate audit and remaining blockers

- The compact dual-worker `group_policy` gate completed successfully on both
  workers. Worker 0 and Worker 1 returned status zero; global step, actor train
  step, and checkpoint step were one, while actor iterator step was eight. This
  proves that the `16/16/2` configuration executed eight microbatches and emitted
  one real weighted optimizer update after distributed vLLM rollout at the
  8,192-token limit.
- Both workers selected the same 16 prompts from 40,300 tokenizer-valid rows.
  Across 16 prompts and eight generations there were 128 trajectories. The eight
  decision records retained 119 trajectories by finite threshold-zero DTV score
  and 63 trajectories in the final effective mask. Seven all-zero-advantage
  groups (56 trajectories) were removed by degenerate-group masking, and nine
  additional finite negative-score trajectories were removed by DTV. The final
  per-microbatch effective counts were `15, 15, 7, 15, 8, 3, 0, 0`, summing to
  63. No score was nonfinite.
- An all-true threshold mask means only that every finite score in that one
  16-trajectory microbatch was greater than or equal to zero. Zero scores pass
  the threshold. It does not imply that all trajectories contribute to the
  update: the final mask additionally removes trajectories belonging to
  degenerate all-zero-advantage groups.
- The final runtime learning-rate state was `9.999999974752427e-07`, the FP32
  representation of `1e-6`. The printed top-level `optimizer_config` value
  `1e-5` belongs to the unused generic/non-Agentic optimizer configuration. The
  formal actor is constructed from
  `rl_training_config.actor_optimizer_config`.
- A remaining critical issue was found during this audit: although the actor
  config says `schedule_type=cosine_decay_schedule`, `init_value=1e-6`, and
  `decay_steps=1`, it also contains `learning_rate=1e-6`. In
  `HyperParameters._extract_kwargs`, a config-provided `learning_rate` currently
  takes precedence over the schedule object passed by `_create_learning_rate`.
  Therefore the optimizer receives the scalar `1e-6`, not the constructed
  cosine schedule. A one-update gate cannot expose this because both have value
  `1e-6` at schedule index zero. This must be fixed and covered by a multi-update
  CPU optimizer-construction test before formal training or the remaining TPU
  gates.
- Despite `TUNIX_SKIP_FINAL_CHECKPOINT=1` and save interval 999, Orbax saved step
  one because the trainer calls `CheckpointManager.save` at every completed
  accumulation boundary and Orbax saves the first checkpoint when none exists.
  The environment flag skipped only the later forced final-save path. This gate
  consequently validated real distributed checkpoint serialization, but the
  behavior does not meet the intended formal policy of retaining only interval
  64 and final checkpoints. Exact interval gating must be corrected before the
  formal 314-step runs to avoid an unnecessary large step-one checkpoint.

## 2026-08-05: learning-rate, exact-checkpoint, and summary corrections

- Optimizer construction now gives the schedule object returned by
  `_create_learning_rate` precedence over a scalar `learning_rate` field in the
  same optimizer dictionary. Scalar learning rates remain unchanged when no
  schedule is configured. A regression test constructs AdamW from a config that
  deliberately contains both a conflicting scalar and cosine schedule, performs
  two updates, and asserts that the authoritative injected optimizer state equals
  cosine step one rather than the scalar.
- Periodic checkpoint calls are now made only when the completed optimizer step
  is exactly divisible by `save_interval_steps`. Configurations without an
  explicit step interval retain their previous manager-driven behavior. A CPU
  test asserts that interval two produces only periodic steps two and four, not
  the Orbax implicit first checkpoint. Formal interval 64 will therefore attempt
  steps 64, 128, 192, and 256; the independent final-save path writes step 314.
  With `max_to_keep=1`, only the latest local checkpoint remains after each
  finalized save unless step 64 is externally archived before step 128.
- The seeded launcher now derives `GRADIENT_ACCUMULATION` in `run_summary.env`
  from the effective mini-batch and train-microbatch overrides. Compact
  `16/2` gates record eight, while formal `128/2` jobs record 64.
- Server focused validation passed: the schedule-precedence test and exact
  checkpoint-interval test each passed with status zero, and the broader trainer
  selection passed nine tests with status zero. Pytest quiet mode intentionally
  does not print the conflicting test values `0.123` and `0.001`; the passing
  assertion proves that the injected optimizer state matched cosine step one
  derived from `0.001`, rather than the scalar `0.123`.
- The full config test then exposed an unrelated pre-existing test isolation
  issue. Its mesh cases mocked `jax.device_count()` as four or eight but called
  the real `jax.make_mesh()`, which sees the server's single CPU device in a
  `JAX_PLATFORMS=cpu` process. The test now mocks `jax.make_mesh()` as well and
  asserts the exact requested shape, axis names, and automatic axis types. This
  changes no production mesh construction.

## 2026-08-05: final compact TPU gate procedure after CPU validation

- After the complete CPU config gate passes, deploy the exact committed runtime
  files from Worker 0 to Worker 1 before starting TPU work. Worker 1 is a runtime
  copy rather than a Git checkout, so deployment uses small per-file downloads
  pinned to Worker 0's exact Git commit instead of an unreliable streamed Git
  archive.
- Remove only the explicitly named prior compact-gate run and log directories;
  do not delete shared model, dataset, cache, or historical experiment paths.
- The final compact `group_policy` TPU gate uses batch size 16, mini-batch size
  16, and train microbatch size 2. This produces eight microbatches followed by
  one optimizer update. It is the direct HBM test for microbatch two at the full
  eight generations and 8,192-token response limit.
- The gate succeeds only when the suite records exit code zero, both distributed
  worker statuses are zero, global and actor train step equal one, actor iterator
  step equals eight, the recorded gradient accumulation equals eight, and logs
  contain no HBM resource-exhausted or out-of-memory failure. Interval 999 plus
  final-checkpoint suppression should now leave no checkpoint after the exact
  checkpoint-gating correction.

## 2026-08-05: final microbatch-two group-policy TPU gate result

- Run `grpo_aime_dtv_selfinf_group_policy_seed0_clean_20260805_085831`
  completed successfully. Worker 0 and Worker 1 both returned status zero; the
  suite returned exit code zero. No HBM resource-exhausted, process abort,
  traceback, or optimizer dtype error occurred.
- The effective runtime configuration was batch size 16, mini-batch size 16,
  train microbatch size 2, eight gradient-accumulation calls, eight generations,
  8,192 response tokens, Policy-DTV group scope, threshold zero, beta 0.001,
  and sequence-mean-token-mean loss aggregation.
- The final summary reported one global step, one optimizer step, eight actor
  iterator steps, no checkpoint, and learning rate approximately `1e-6`. Both
  workers explicitly constructed `cosine_decay_schedule(init_value=1e-6,
  decay_steps=1)`. The gate's single update cannot display later decay values,
  but the multi-update CPU regression already verified schedule precedence over
  the conflicting scalar field.
- Exact checkpoint gating behaved as intended: interval 999 did not trigger a
  periodic save, final saving was disabled, `checkpoint_step` was null, and no
  model files were written.
- The dataset gate removed nine overlength prompts, found 40,300 valid rows,
  selected exactly 16 rows for one complete gate batch, and recorded the stable
  selection hash. All eight DTV decision records had zero nonfinite scores.
  Degenerate all-zero-advantage groups were excluded from the final update mask,
  while finite negative DTV scores were removed by the threshold-zero rule.
- This result validates the peak-HBM shape used by formal group Policy-DTV:
  formal `128/128/2` training repeats the same two-prompt/eight-generation
  microbatch program 64 times per optimizer step instead of eight. It does not
  mathematically prove that a hardware fault or memory leak cannot occur during
  a multi-day run, but there is no evidence of iteration-dependent HBM growth in
  the tested compiled update path. The group-policy method is ready for formal
  training. Baseline and group-policy-LOO still require their own final compact
  TPU integration gates before their respective formal runs.

## 2026-08-05: degenerate groups and sequential final TPU gates

- A degenerate GRPO group is a prompt whose eight sampled completions receive
  identical rewards. With binary math rewards this commonly means all eight are
  incorrect, but it can also mean all eight are correct. Group centering and
  normalization then produce eight zero advantages, so the group contains no
  within-prompt preference signal. It is not classified as degenerate merely
  because some answers are incorrect.
- `degenerate_group_masking=true` excludes such trajectories from the complete
  actor update, including both the policy term and the KL term. This avoids
  allowing signal-free groups to dilute effective-count normalization or apply
  KL-only updates. The tradeoff is that an all-incorrect prompt does not teach
  the model from that rollout; learning can occur later only if another rollout
  for that prompt produces reward variation. This behavior is shared by the
  baseline, group Policy-DTV, and group Policy-DTV-LOO comparisons.
- The remaining final integration validation is performed sequentially on the
  same v5p-16 node. First run `group_loo_policy` with `16/16/2`; verify both
  worker statuses and suite exit code are zero and TPU processes have exited.
  Then run `baseline` with the identical data, rollout, optimizer, and
  microbatch settings. Interval 999 and final-checkpoint suppression prevent
  either compact gate from writing model checkpoints.

## 2026-08-05: final baseline and group-LOO gate acceptance

- The final baseline and group Policy-DTV-LOO compact gates both completed with
  Worker 0 status zero, Worker 1 status zero, and no HBM OOM or Python failure.
  Each run completed eight actor iterator calls, one optimizer update, and one
  global step; checkpoint step was null and the authoritative learning rate was
  approximately `1e-6`.
- Both gates used the same `16/16/2` accumulation geometry, eight generations,
  8,192 response tokens, beta 0.001, sequence-mean-token-mean aggregation,
  deterministic seed/data selection, and degenerate-group masking as the final
  group-policy gate. All three formal methods have therefore passed their final
  method-specific distributed TPU integration gate.
- Baseline timing was a 121.57-second first compiled actor call followed by
  seven steady calls of approximately 3.93--4.05 seconds. Group LOO used a
  181.87-second first compiled call followed by seven steady calls of
  approximately 21.75--21.79 seconds. The baseline is expected to be much
  faster: it performs one aggregate Policy+KL backward per microbatch, while
  each DTV method additionally computes exact per-trajectory policy-gradient
  scores before its masked aggregate Policy+KL update. The earlier claim that
  baseline steady actor calls were approximately 18 seconds was not supported
  by these final logs.
- Group Policy-DTV uses the exact standard score
  `<g_i, sum_j g_j>/G`; group Policy-DTV-LOO uses the exact leave-one-out score
  `<g_i, sum_{j != i} g_j>/(G-1)`. Both use policy-only gradients for scoring,
  finite `score >= 0` selection, no minimum-retention cap, the same degenerate
  group mask, and an effective-count-weighted masked aggregate Policy+KL update.
- The compact gates prove method routing and peak microbatch HBM behavior. The
  formal jobs must explicitly set 314 batches and optimizer steps, batch and
  mini-batch size 128, train microbatch size 2, LR decay steps 314, checkpoint
  interval 64, and complete-batch enforcement. The validated 40,300-row dataset
  supplies the required 40,192 rows for exactly 314 complete batches.

## 2026-08-05: formal group Policy-DTV-LOO launch configuration

- The formal seed-zero clean group Policy-DTV-LOO run uses 314 complete batches
  and optimizer updates, batch and mini-batch size 128, train microbatch size 2,
  64 effective-count-weighted accumulation calls per optimizer update, eight
  generations, 8,192 response tokens, beta 0.001, threshold zero, and a
  no-warmup cosine schedule from `1e-6` over 314 optimizer updates.
- Periodic checkpoints are requested at completed optimizer steps 64, 128, 192,
  and 256. The independent final-save path writes step 314. With
  `max_to_keep=1`, a newly finalized checkpoint replaces the previous local
  checkpoint; this is rolling retention at a 64-step interval, not permanent
  retention of every 64-step model. Step 64 must be copied to external storage
  after its metadata is complete and before step 128 is saved if it is required
  for later analysis.
- Trajectory text logging remains disabled to limit disk consumption, while DTV
  decision diagnostics, TensorBoard metrics, run summary, data manifest,
  distributed worker logs, and checkpoint metadata remain enabled.

## 2026-08-05: formal group-LOO data-gate confirmation

- Formal run `grpo_aime_dtv_selfinf_group_loo_policy_seed0_clean_20260805_095513`
  created a valid strict data manifest: 40,309 input rows, nine tokenizer-
  overlength rows, 40,300 valid rows, exactly 40,192 selected rows, 314 complete
  batches of 128, and 108 unused valid rows. The deterministic formal selection
  hash is `adc37181ea707583d3a775a9032b1ebc3c617931e4180bda9267d4b65ecc0928`.
- `run_summary.env` is a suite completion artifact and is not expected while the
  multi-day job is still active. Runtime method, microbatch, learning-rate, and
  checkpoint settings must be confirmed from the effective configuration in the
  live worker log without polling or modifying the process.

## 2026-08-05: formal group-LOO runtime configuration accepted

- The live effective configuration for formal run
  `grpo_aime_dtv_selfinf_group_loo_policy_seed0_clean_20260805_095513` exactly
  matches the approved experiment: batch and mini-batch size 128, train
  microbatch size 2, 64 accumulation calls, 314 complete batches and optimizer
  steps, eight generations, 8,192 response tokens, beta 0.001, and threshold
  zero.
- Runtime routing explicitly enabled `self_inf_group_loo_policy` with group
  scope and eight generations. The actor optimizer explicitly constructed a
  no-warmup cosine schedule with initial value `1e-6` and decay length 314.
  Periodic checkpoint configuration is interval 64 with rolling retention one.
- Generic non-Agentic configuration values still printed elsewhere in the
  merged configuration are not selected by this training path. The effective
  Agentic summary and actor-specific configuration are authoritative. No restart
  or modification of the active formal process is required.

## 2026-08-05: interpretation of group-LOO retention diagnostics

- Formal group Policy-DTV-LOO uses `min_keep_fraction=0.0`; the legacy retention
  cap is disabled. Consequently `loo_retention_cap_triggered`,
  `loo_groups_with_cap_triggered`, `loo_retained_by_cap_mask`, and every element
  of `loo_group_cap_triggered` remain zero or false.
- `loo_pre_cap_kept_samples` counts the finite scores satisfying the threshold
  before the common effective trajectory mask. `loo_post_cap_kept_samples`
  currently counts the final effective mask after degenerate-group masking. Its
  historical name is therefore misleading when the cap is disabled; it is best
  interpreted as final effective retained samples.
- In the inspected record, group zero retained only generation one because its
  LOO score was the sole nonnegative finite score. Group one had eight zero LOO
  scores, so all eight passed `score >= 0` initially, but the group had all-zero
  advantages and all eight were removed from the final effective update. This
  explains pre-threshold retention nine and final retention one without any cap
  activation.

## 2026-08-07: formal step-64 checkpoint archive

- The formal group Policy-DTV-LOO step-64 checkpoint completed successfully:
  root metadata, model manifest, and optimizer manifest existed, and no Orbax
  temporary checkpoint directory remained.
- The completed checkpoint tree, approximately 14.3 GB, was copied from Worker
  0 to the Worker 1 archive with `rsync --append-verify`. Final archive
  acceptance requires a read-only comparison of relative-path SHA256 manifests,
  file counts, and required remote metadata. Verification is scoped only to the
  immutable `actor/64` directory and does not interact with the active training
  process or later rolling checkpoints.

## 2026-08-07: formal group-LOO runtime projection at step 68

- The formal run launched at approximately 2026-08-05 09:55 UTC and completed
  logged global step 68 at approximately 2026-08-07 02:25 UTC, about 40.5 hours
  of wall-clock time including initialization, compilation, rollout, training,
  synchronization, and the step-64 checkpoint.
- The latest complete optimizer step took 2,117.34 seconds, or approximately
  35.29 minutes. This agrees closely with the wall-clock average after allowing
  for zero-based versus one-based step logging.
- Approximately 245--246 optimizer updates remain out of 314. At the observed
  steady rate, pure remaining computation is about 144--145 hours, or 6.0 days.
  Allowing for checkpoints at 128, 192, 256, and final 314, rollout-length
  variance, and filesystem delays gives a practical estimate of 6.0--6.3 more
  days, assuming no interruption. The expected finish is around 2026-08-13
  02:30--09:30 UTC, equivalent to 2026-08-13 10:30--17:30 Asia/Shanghai.

## 2026-08-07: local retention decision after archiving step 64

- Do not manually delete the active run's local `actor/64` directory while the
  in-process Orbax CheckpointManager still records it as the latest managed
  checkpoint. External deletion can leave the manager's in-memory retention
  state inconsistent with the filesystem when step 128 is finalized.
- Worker 0 currently has approximately 18 GB free and the completed checkpoint
  is approximately 14.3 GB. This should fit one additional checkpoint before
  rolling retention removes step 64, but the temporary low-water mark may be
  only several gigabytes. Free space should first be recovered from unrelated
  historical gates, caches, or checkpoints after a read-only size audit.
- Once step 128 completes, `max_to_keep=1` should remove local step 64
  automatically. The separately verified Worker 1 archive remains the permanent
  step-64 copy.

## 2026-08-07: Worker 0 disk headroom before step 128

- Truncating oversized Docker container JSON logs increased Worker 0 free space
  from approximately 18 GB to 21 GB. This did not remove the experiment's repo
  logs, TensorBoard data, DTV decisions, or checkpoints.
- The active step-64 checkpoint occupies approximately 14 GB. A same-sized step
  128 checkpoint should leave approximately 6--7 GB of transient free space
  before Orbax rolling retention removes step 64. This is adequate but should be
  observed once after step 128 completes; the active step-64 directory should
  not be deleted manually.
- Other large space consumers are historical LHF checkpoints (approximately
  29 GB total) and a 3.4 GB `/tmp/models` copy. They may be valuable or active
  and must not be removed during the current run without separate verification.

## 2026-08-13: formal baseline launch parity

- The formal baseline run must differ from the two DTV runs only in method
  routing. It uses seed zero, clean data, 314 complete batches and optimizer
  updates, batch and mini-batch size 128, train microbatch size 2, eight
  generations, 8,192 response tokens, beta 0.001, sequence-mean-token-mean loss
  aggregation, and the common degenerate-group mask.
- The actor optimizer uses the same no-warmup cosine schedule from `1e-6` over
  314 optimizer updates. Checkpoints use interval 64, `max_to_keep=1`, and an
  enabled final step-314 save. Trajectory text logging remains disabled. The
  suite's required filter argument has no effect on the baseline method.

## 2026-08-14: final group-LOO checkpoint and AIME 2024 evaluation design

- A training checkpoint does not embed the DeepScaleR examples, rollout text,
  or reward decisions. Step 314 contains model parameters, optimizer state, and
  checkpoint metadata. Dataset provenance is established by the run's strict
  data manifest, selection hash, effective configuration, and retained logs.
- Final acceptance has three layers. First, verify the step-314 root metadata,
  model and optimizer manifests, absence of Orbax temporary directories, final
  suite/worker exit statuses, and a 314-step training summary. Second, audit
  TensorBoard and DTV decision logs for finite loss/LR, zero nonfinite scores,
  sensible retained fractions, and the expected cosine endpoint. Third, restore
  step 314 and run the tracked offline evaluator on the exact AIME 2024 parquet.
- The formal evaluation uses all 30 unique AIME 2024 problems, 16 stochastic
  samples per problem (480 records), vLLM, temperature 0.6, top-p 0.95, maximum
  generation length 8,192, maximum prompt length 2,048, batch size one,
  `max_num_seqs=16`, and `max_num_batched_tokens=38400`. Required outputs are
  `samples.jsonl`, `summary.json`, dataset SHA256 provenance, and metrics
  `avg@16`, `pass@16`, `maj@16`, average tokens, and truncation rate.
- Scientific comparison requires the base model, baseline step 314, group
  Policy-DTV step 314, and group Policy-DTV-LOO step 314 to use the same evaluator
  commit, dataset hash, tokenizer, decoding parameters, and sample count. A
  training loss alone is not evidence of improved AIME accuracy.
## 2026-08-15 — Formal group LOO run exit-status audit

- The formal `group_loo_policy` run reached global/actor train step 314 and produced a structurally complete final actor checkpoint at step 314.
- Its recorded `EXIT_CODE=1` is the return code of `run_official_like_dual_worker.sh`; that launcher returns 1 when either `LOCAL_STATUS` or the resolved `REMOTE_STATUS` is nonzero.
- The exit code alone therefore does not prove that optimizer updates or checkpoint persistence failed. The retained worker status records and the tails of `local.log` and `remote.log` must be inspected before classifying the run.
- Do not rerun or delete the final checkpoint until that post-training exit source is identified and the checkpoint is restore/evaluation tested.
- The Worker 0 log confirms all 314 updates, final weight synchronization, and successful synchronous finalization of checkpoint 314. The final checkpoint replaced checkpoint 256 according to `max_to_keep=1`.
- No fatal training, HBM OOM, or Orbax save error appears at the end of the Worker 0 log. The tolerant math-parser messages and Orbax "No metadata found" informational messages did not prevent successful saves.
- The Worker 0 filesystem does not contain the Worker 1 `remote.log`, because the same absolute path resolves to Worker 1's node-local disk inside the remote command. The remaining exit-code diagnosis therefore requires a one-time read of that file on Worker 1.
- Worker 1's remote log ends abruptly at 2026-08-12 06:32 while the rollout engine still reports roughly 965 running requests. It contains neither a clean shutdown nor the launcher's `HOST=... STATUS=0` record. Worker 0 continued until 2026-08-13 02:33.
- This missing remote program status explains the outer exit code: the launcher had to fall back to the failed/disconnected SSH transport status and returned 1.
- The final checkpoint is structurally complete, but scientific acceptance of the run additionally requires confirming that Worker 0 explicitly recovered/rerouted the lost rollout process and that no incomplete batch was consumed. A checkpoint restore and AIME evaluation remain required.
- Worker 0 explicitly logged lost distributed-rollout connections followed by one retry after remote restart. This recovery behavior occurred multiple times and training subsequently completed contiguous global steps 280 through 314 with normal per-step durations.
- The complete 314-step decision/selection cardinality and contiguous optimizer-step sequence provide no evidence that a partial rollout batch was consumed. Retried unfinished generations may change exact stochastic samples, but they do not change the method, objective, batch cardinality, or data-selection semantics.
- The reported kernel-journal command was executed at the Worker 0 shell prompt, so its empty result proves only that Worker 0 had no matching kernel OOM event. It does not audit Worker 1's kernel journal.
- Operational classification: training and final checkpoint succeeded; the outer exit code is a remote-session/status-record failure. Scientific acceptance still requires a successful checkpoint restore and the planned AIME 2024 evaluation.

### Restore validation plan

- Validate actor checkpoint 314 first; checkpoint 64 is an optional intermediate comparison and is not required for final-model restore validation.
- Run the restore on both TPU hosts so JAX distributed initialization is valid. Process 0 constructs the historical `(4, 1)` `fsdp,sp` mesh and calls the existing evaluator's `_load_final_actor_model`; process 1 participates only in initialization and a final global barrier.
- The gate must print `Restored actor checkpoint step 314.` and `RESTORE_GATE_OK step=314` and return zero on both workers.
- This gate reads model parameters only. It does not generate samples, update parameters, save a checkpoint, or require the step-64 archive to be copied back.

### Restore validation result

- The two-host restore gate passed with `process_count=2`, four local TPU devices per process, and eight global TPU devices.
- Process 0 restored exactly actor checkpoint step 314 from the formal `group_loo_policy` run.
- Both processes completed the final multihost barrier and the command returned `RESTORE_GATE_STATUS=0`.
- The final actor checkpoint is therefore readable and mesh-compatible. The earlier experiment `EXIT_CODE=1` is classified as a launcher/remote-session termination-status failure rather than a failed training or failed final checkpoint.
- The remaining scientific validation is the full AIME 2024 evaluation: 30 problems, 16 samples per problem, 480 total generations, using checkpoint 314.
- `RESTORE_GATE_DONE process=1 restored=None` is expected: the restore gate intentionally executes `_load_final_actor_model` only on JAX process 0. Process 1 initializes its four local TPU devices and joins the final global barrier, but does not read the checkpoint; its local `restored` variable therefore remains `None`.
- `short_sweep_queue_20260707.md` defines only the 64-step training sweep (threshold, beta, and optional response-length stages). It contains no evaluation protocol.
- Final evaluation should therefore match the recovered historical baseline evaluation rather than infer settings from the sweep note: AIME 2024, checkpoint 314, 30 problems, 16 samples per problem, 8192 generation steps, temperature 0.6, top-p 0.95, batch size 1, vLLM, mesh `(4, 1)` with `fsdp,sp`, `max_num_seqs=16`, `max_num_batched_tokens=38400`, and vLLM HBM utilization 0.8.

## Proposed detailed AIME evaluation implementation (discussion only)

- Re-evaluate the base model, the newly trained baseline step 314, Group Policy-DTV step 314, and Group Policy-DTV-LOO step 314 under one new matched protocol. Historical baseline numbers are protocol references, not substitutes for the retrained baseline.
- The current evaluator explicitly sets `seed=None` for vLLM. `VllmSampler` accepts one sampling seed but documents that the JAX vLLM backend does not support a distinct seed for each request inside one call.
- Merely replacing `None` with a scalar evaluation seed is unsafe because the current evaluator sends sixteen duplicate prompts for one problem in the same call; all duplicate requests could share an unsuitable seed configuration.
- Proposed deterministic schedule: iterate sample slots 0 through 15; for each slot, generate one completion for every AIME problem in fixed dataset order, chunked by `max_num_seqs`; pass `eval_seed + sample_slot` as the call seed. This keeps duplicate copies of the same problem out of the same seeded call and requires roughly the same number of calls as the historical problem-major batching.
- Before full evaluation, run the same two-problem deterministic gate twice and compare response/token hashes. A fixed seed provides matched, reproducible sampling intent, but bitwise determinism must be empirically verified for the exact distributed vLLM/JAX build and scheduling configuration.
- Store every response and primitive field once, then compute correctness, format, length, diversity, pass@k, and paired model comparisons offline without additional TPU generation.

## 2026-08-15 — Detailed AIME evaluation implementation

- Updated `examples/deepscaler/eval_final_checkpoint_metrics.py` with an explicit `--eval_seed`, sample-slot-major generation, a fixed auditable seed schedule, strict `(problem_id, sample_index)` cardinality validation, enriched per-sample facts, generation wall time, and a real multihost completion barrier.
- The seed schedule is `eval_seed + sample_index`. Each seeded call contains distinct problems only; the same problem is never duplicated inside one JAX-vLLM call that shares a seed. With 30 AIME problems and `problem_batch_size=16`, a full k=16 evaluation uses 32 sampler calls.
- Extended `tunix/utils/math_eval_metrics.py` while retaining all historical keys. New metrics cover standard estimated pass@1/2/4/8/16, absolute correct/solved/majority counts, boxed/extractable/AIME-valid format rates, conditional accuracy, token percentiles, truncated versus non-truncated accuracy, correct/incorrect lengths, answer diversity, vote concentration, and answer entropy.
- Added `examples/deepscaler/analyze_aime_eval.py` to generate `detailed_summary.json` and `per_problem.jsonl` from retained samples without TPU generation.
- Added `examples/deepscaler/compare_aime_evals.py` for problem-aligned model comparison and paired problem-level bootstrap confidence intervals. Bootstrap resampling treats the 30 problems, not the 480 correlated generations, as independent units.
- Added `runs_xuesong/scripts/run_aime_final_eval.sh`, which fixes the matched AIME-2024 k=16 protocol, launches both TPU workers, records configuration and logs, and requires a summary, completion sentinel, and exactly 480 sample records before returning zero.
- Added or extended CPU tests for detailed metrics, pass@k, AIME answer validation, explicit seed forwarding, deterministic scheduling, offline per-problem summaries, aligned comparisons, and reproducible bootstrap.
- Local static validation passed with `python3 -m py_compile`, `bash -n`, and `git diff --check`. The Mac checkout has no project `.venv` or pytest installation, so dependency-backed pytest must run in the server environment before deployment or TPU evaluation.
- The Worker 0 CPU gate passed all 21 focused tests in 4.43 seconds. The three emitted messages are existing SWIG and JAX `shard_map` deprecation warnings, not evaluation correctness failures.
- The next integration gate is two identical two-problem, two-sample, 512-token distributed-vLLM evaluations with `eval_seed=2026`. Acceptance requires both runs to return zero, each to contain four unique `(problem_id, sample_index)` records, and response/token hashes to match exactly. This gate tests protocol determinism cheaply before the 30-problem k=16 evaluation.
- Added `token_ids_sha256` to each generated record so the mini-gate can compare exact generated token sequences rather than relying on response text and token counts alone. Hashing occurs after generation on the host and does not affect sampling or model computation.

## 2026-08-15 — Mini-evaluation permission failure and launcher correction

- The first distributed mini-evaluation restored checkpoint 314, initialized both JAX hosts, initialized vLLM, and synchronized model weights successfully. It failed before generation with `PermissionError` while opening `run1/samples.jsonl`.
- Root cause: `gcloud --worker=all` connected to Worker 0 with a service-account identity, while the output directory had been created by `jason_chia925_gmail_com`. This was a host filesystem ownership mismatch, not a checkpoint, evaluator, TPU, vLLM, or HBM failure.
- Corrected `run_aime_final_eval.sh` to execute Worker 0 directly under the invoking local user and launch only Worker 1 through gcloud. Worker 1 participates in distributed initialization and barriers but does not write primary evaluation artifacts.
- Added bounded mini-gate overrides for dataset limit, sample count, generation length, problem batch size, vLLM sequence capacity, and batched-token capacity. Formal evaluation defaults remain unchanged at 30 problems, k=16, and 8,192 generation steps.
- The launcher now records local, remote-program, and remote-SSH statuses separately and validates the expected sample cardinality derived from the selected mini or formal protocol.
