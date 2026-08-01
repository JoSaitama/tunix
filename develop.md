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
- Confirmed the staged experiment policy from `short_sweep_queue_20260707.md`: validate one-batch checkpoint-producing smoke runs first; start the short formal sweep at threshold `0.0`; compare `-0.05`, `0.0`, and `0.05` over 64 batches before selecting a threshold; then sweep GRPO beta over `0.0003`, `0.001`, and `0.003`; only sweep response length (`4096`, `8192`) if needed. Threshold applies to ordinary policy DTV only, not LOO or fixed filters.
- Because worker 0 has only 18 GB free, checkpoint-producing smoke runs must be launched one method at a time. Verify and delete each run's exact run directory before starting the next method; do not queue all checkpoint-producing smoke methods together.

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
