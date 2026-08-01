# AIME GRPO vLLM Development Log

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

The bootstrap uses the LHF Python only to create and seed a new Jason-owned
environment. It filters out editable/file-URL Tunix installs, installs the
historical dependency snapshot, verifies that it contains pytest, then installs
the current Jason checkout without dependency resolution in editable mode.

Installation requires package and Git network access. It may take significant
time because the historical TPU vLLM entry is pinned to a Git commit. Do not
modify the LHF `.venv`.

After installation, run once on worker 0:

```bash
source .venv/bin/activate
python -c 'import jax, tunix; print(jax.__version__); print(tunix.__file__)'
python -c 'import jax; print(jax.devices())'
python -m pytest -q \
  tests/oss_utils_test.py \
  tests/rl/self_inf_trainer_test.py \
  tests/rl/rl_cluster_test.py \
  tests/rl/agentic/agentic_grpo_learner_test.py \
  tests/cli/grpo_main_test.py
```

Repeat only the import-origin and device checks on worker 1.

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
