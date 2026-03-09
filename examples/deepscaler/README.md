# DeepScaler Examples (CLI)

`examples/deepscaler/` now supports direct CLI execution for training and evaluation.

## Files

- `train_deepscaler_nb.py`: GRPO training entrypoint with `argparse`.
- `math_eval_nb.py`: math evaluation entrypoint with `argparse`.
- `run_train.sh`: wrapper for training (includes local default model/data paths on this machine).
- `run_eval.sh`: wrapper for evaluation (includes local default model/data paths on this machine).

## Quick start

Show available flags:

```bash
python examples/deepscaler/train_deepscaler_nb.py --help
python examples/deepscaler/math_eval_nb.py --help
```

Smoke-test (minimal run shape, useful to validate runtime wiring):

```bash
./examples/deepscaler/run_train.sh

./examples/deepscaler/run_eval.sh
```

Smoke-test training only:

```bash
./examples/deepscaler/run_train.sh --smoke-test
```

Override script defaults with environment variables:

```bash
MODEL_PATH=/path/to/model \
TRAIN_DATA_PATH=/path/to/deepscaler.json \
TEST_DATA_PATH=/path/to/train-00000-of-00001.parquet \
./examples/deepscaler/run_train.sh

MODEL_PATH=/path/to/model \
TEST_DATA_PATH=/path/to/train-00000-of-00001.parquet \
./examples/deepscaler/run_eval.sh
```

## Rollout backends

Training now supports `vanilla`, `vllm`, and `sglang_jax` rollout engines:

```bash
# vanilla (default)
./examples/deepscaler/run_train.sh --smoke-test --rollout-engine vanilla

# sglang-jax
./examples/deepscaler/run_train.sh --smoke-test --rollout-engine sglang_jax

# vllm
./examples/deepscaler/run_train.sh --smoke-test --rollout-engine vllm
```

Optional rollout model source for non-vanilla engines (local path, `gs://`, or
HF repo id):

```bash
./examples/deepscaler/run_train.sh \
  --smoke-test \
  --rollout-engine vllm \
  --rollout-model-source deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

Notes:

- `--rollout-engine sglang-jax` is accepted as an alias of `sglang_jax`.
- If rollout dependency packages are missing, preflight will fail early with install hints.
- `vllm` exposes tuning flags like `--rollout-vllm-hbm-utilization`, `--rollout-dp`, and `--rollout-tp`.
- `sglang_jax` exposes tuning flags like `--rollout-sglang-jax-context-length` and `--rollout-sglang-jax-mem-fraction-static`.

## Repro commands (this machine)

Use the tested Python 3.12 environment for `sglang_jax` runs:

```bash
source .venv_sglang312/bin/activate
# Optional override (run_train.sh default is 1):
# export GRPO_MAX_CONCURRENCY=2
```

Run baseline vanilla smoke regression:

```bash
RUN_TS=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_DIR=/tmp/deepscaler_ckpt_${RUN_TS} \
METRICS_LOG_DIR=/tmp/deepscaler_tb_${RUN_TS} \
./examples/deepscaler/run_train.sh --smoke-test --rollout-engine vanilla
```

Run `sglang_jax` smoke test with tensor parallel rollout (`--rollout-tp 2`):

```bash
RUN_TS=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_DIR=/tmp/deepscaler_ckpt_${RUN_TS} \
METRICS_LOG_DIR=/tmp/deepscaler_tb_${RUN_TS} \
./examples/deepscaler/run_train.sh --smoke-test --rollout-engine sglang_jax --rollout-tp 2
```

Run `sglang_jax` non-smoke experiment:

```bash
RUN_TS=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_DIR=/tmp/deepscaler_ckpt_${RUN_TS} \
METRICS_LOG_DIR=/tmp/deepscaler_tb_${RUN_TS} \
./examples/deepscaler/run_train.sh --rollout-engine sglang_jax --rollout-tp 2
```

Run `self-inf-group` DBC on the current default DeepScaler geometry
(`num_generations=2` by default):

```bash
RUN_TS=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_DIR=/tmp/deepscaler_ckpt_${RUN_TS} \
METRICS_LOG_DIR=/tmp/deepscaler_tb_${RUN_TS} \
./examples/deepscaler/run_train.sh \
  --rollout-engine sglang_jax \
  --rollout-tp 2 \
  --use-dynamic-batch-curation \
  --use-dbc-self-inf-group
```

Run `pass@1` averaged over 16 independent `sglang-jax` eval runs:

```bash
source .venv_sglang312/bin/activate
LOG_DIR=/tmp/deepscaler_pass1_avg16_$(date +%Y%m%d_%H%M%S) \
./examples/deepscaler/run_eval_pass1_avg16.sh
```

Observed local result on this machine (`2026-03-07`, AIME 2024, seeds `0..15`):

```text
===== FINAL SUMMARY =====
Runs: 16
Sampler: sglang-jax
Seeds: 0..15
Metric: Pass@1 averaged over 16 independent runs
Average Correct: 5.6875/30.0000
Average Accuracy: 18.9594%
Logs saved to: /tmp/deepscaler_pass1_avg16_20260307_171010
```

If non-smoke run fails with `RESOURCE_EXHAUSTED` at `jit__train_step`, start with
this lower-memory profile (verified in this environment):

```bash
RUN_TS=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_DIR=/tmp/deepscaler_ckpt_${RUN_TS} \
METRICS_LOG_DIR=/tmp/deepscaler_tb_${RUN_TS} \
./examples/deepscaler/run_train.sh \
  --rollout-engine sglang_jax \
  --rollout-tp 2 \
  --max-prompt-length 1024 \
  --total-generation-steps 512 \
  --batch-size 8 \
  --mini-batch-size 8 \
  --train-micro-batch-size 1
```

To probe only compile + first step before a long run:

```bash
RUN_TS=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_DIR=/tmp/deepscaler_ckpt_${RUN_TS} \
METRICS_LOG_DIR=/tmp/deepscaler_tb_${RUN_TS} \
./examples/deepscaler/run_train.sh \
  --rollout-engine sglang_jax \
  --rollout-tp 2 \
  --max-prompt-length 1024 \
  --total-generation-steps 512 \
  --batch-size 8 \
  --mini-batch-size 8 \
  --train-micro-batch-size 1 \
  --num-batches 1 \
  --num-epochs 1 \
  --max-steps 1
```

If you see checkpoint save failures like `No space left on device`, free old
temporary checkpoints first:

```bash
rm -rf /tmp/deepscaler_ckpt_*
```

## Authentication modes

### 1) Local files only (no login needed)

- Pass local filesystem paths for `--model-path`, `--train-dataset-path`, `--test-dataset-path`, or `--dataset-path`.
- This is the simplest way to run without any cloud credentials.

### 2) Hugging Face model/tokenizer

- Public repos may work anonymously.
- Gated/private repos need a token:

```bash
export HF_TOKEN=...
# optional: huggingface-cli login
```

- You can enforce token presence with `--require-hf-token`.

#### Backend-specific runtime deps

- `vllm` rollout requires `vllm`/`tpu-inference` runtime support.
- `sglang_jax` rollout requires `sglang-jax` (`sgl_jax` Python package).
- Keep these in an isolated environment from baseline vanilla runs.

### 3) `gs://` model/data

- Requires GCP ADC/service-account credentials:

```bash
gcloud auth application-default login
# or:
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

## Notes

- Preflight checks run by default and fail early on missing local paths/credentials.
- Add `--skip-preflight` to bypass checks.
- WandB is disabled by default in training; enable only when needed with `--enable-wandb`.
