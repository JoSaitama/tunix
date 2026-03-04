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
