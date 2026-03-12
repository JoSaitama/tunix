# Qwen3 UltraFeedback DPO Baseline

This directory contains the runnable recipe for the `Qwen/Qwen3-4B-Instruct-2507`
LoRA DPO baseline on `HuggingFaceH4/ultrafeedback_binarized`.

## Files

- `qwen3_4b_ultrafeedback.yaml`: baseline config
- `run_qwen3_4b_ultrafeedback.sh`: launcher for `full` and `smoke`

## Baseline

- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Dataset: `HuggingFaceH4/ultrafeedback_binarized`
- Train split: `train_prefs`
- Eval split: `test_prefs`
- Method: LoRA DPO
- LoRA: `rank=128`, `alpha=128`
- Max lengths: `prompt=512`, `response=512`
- `beta=0.01`
- Optimizer: `AdamW`
- LR schedule: warmup cosine decay
- Peak LR: `5e-6`
- Effective batch size: `8`
  - `batch_size=1`
  - `gradient_accumulation_steps=8`
- Train steps: `5464`
- Eval interval: `500`

## Run

Activate the prepared environment and start training:

```bash
source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate
cd /home/lhf_hongfu_gmail_com/tunix
./examples/dpo/run_qwen3_4b_ultrafeedback.sh full
```

Run the short smoke test:

```bash
source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate
cd /home/lhf_hongfu_gmail_com/tunix
./examples/dpo/run_qwen3_4b_ultrafeedback.sh smoke
```

The launcher loads `.env` from either:

- `/home/lhf_hongfu_gmail_com/tunix/.env`
- `/home/lhf_hongfu_gmail_com/tunix/my_example/.env`

`HF_TOKEN` is required.

## Environment

This baseline was prepared with a dedicated virtual environment:

- Environment name: `DPO`
- Path: `/home/lhf_hongfu_gmail_com/.venvs/DPO`
- Python: `3.11`

Create the environment:

```bash
python3.11 -m venv /home/lhf_hongfu_gmail_com/.venvs/DPO
source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate
```

Install Tunix and the required packages:

```bash
cd /home/lhf_hongfu_gmail_com/tunix
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m pip install "jax[tpu]==0.8.1"
```

Runtime assumptions:

- Single-host TPU VM
- `4` TPU devices visible to JAX
- Mesh: `(1,4)` with axis `('fsdp','tp')`

Sanity check:

```bash
source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate
python -c "import jax; print(jax.default_backend(), jax.device_count())"
```

Expected output:

```text
tpu 4
```

Credentials:

- `HF_TOKEN`: required for downloading `Qwen/Qwen3-4B-Instruct-2507`
- `WANDB_API_KEY`: optional

## Outputs

The default full run writes to:

- Checkpoints: `/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen3_4b_ultrafeedback/checkpoints`
- TensorBoard: `/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen3_4b_ultrafeedback/tensorboard`
- Merged model: `/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen3_4b_ultrafeedback/merged_lora`

## Implementation

This directory only contains the experiment recipe. The training
infrastructure lives under `tunix/`, mainly:

- `tunix/cli/dpo_main.py`
- `tunix/sft/dpo/dpo_trainer.py`
- `tunix/sft/peft_trainer.py`
- `tunix/cli/utils/model.py`
- `tunix/examples/data/ultrafeedback_dpo.py`
