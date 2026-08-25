# Qwen3 UltraFeedback DPO Baseline

This directory contains the runnable recipe for the `Qwen/Qwen3-4B-Instruct-2507`
LoRA DPO baseline on `HuggingFaceH4/ultrafeedback_binarized`.

It also contains a second recipe for `Qwen/Qwen2.5-1.5B` that starts from an
SFT-exported model checkpoint and now supports four primary methods:
`vanilla_dpo`, `random_pair_filtering`, `reward_based_filtering`, and
`self_inf`. A legacy `outlier_l2` option is still available for backward
comparisons. That workflow defaults to full DPO and accepts `--ft-mode lora`
when you want a LoRA actor. Its training-time eval is taken from a prompt-level
holdout inside `train_prefs`, leaving `test_prefs` for final reporting.

## Files

- `qwen3_4b_ultrafeedback.yaml`: baseline config
- `run_qwen3_4b_ultrafeedback.sh`: launcher for `full` and `smoke`
- `qwen2p5_1p5b_ultrafeedback_from_sft.yaml`: full DPO-from-SFT config
- `qwen2p5_1p5b_ultrafeedback_from_sft_lora.yaml`: LoRA DPO-from-SFT config
- `run_qwen2p5_1p5b_ultrafeedback_from_sft.sh`: launcher for
  `vanilla_dpo|random_pair_filtering|reward_based_filtering|self_inf`
- `run_qwen2p5_1p5b_ultrafeedback_from_sft_flip_matrix.sh`: batch launcher for
  the focused static flip-correlation matrix

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
  - `batch_size=2`
  - `gradient_accumulation_steps=4`
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

Run the same smoke recipe with cross-accumulation DBC enabled:

```bash
source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate
cd /home/lhf_hongfu_gmail_com/tunix
./examples/dpo/run_qwen3_4b_ultrafeedback.sh smoke \
  dpo_config.use_dynamic_batch_curation=true \
  dpo_config.curation_threshold=3.0
```

Run the full DPO recipe with `outlier_l2` curation:

```bash
source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate
cd /home/lhf_hongfu_gmail_com/tunix
./examples/dpo/run_qwen3_4b_ultrafeedback.sh full \
  dpo_config.use_dynamic_batch_curation=true \
  dpo_config.curation_variant=outlier_l2 \
  dpo_config.curation_threshold=3.0
```

Available DPO DBC variants:

- `dpo_config.curation_variant=outlier_l2` (default): filter large gradient-norm outliers using `mean + curation_threshold * std`
- `dpo_config.curation_variant=self_inf_batch`: filter samples whose gradient has negative or weak alignment with the full accumulation-window mean gradient using `dpo_config.self_influence_dot_threshold`

Run the `Qwen/Qwen2.5-1.5B` DPO-from-SFT smoke recipe:

```bash
source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate
cd /home/lhf_hongfu_gmail_com/tunix
./examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh \
  smoke \
  vanilla_dpo \
  /path/to/sft_exported_model
```

Swap `vanilla_dpo` for `random_pair_filtering`, `reward_based_filtering`, or
`self_inf` to run the other methods. `outlier_l2` is still accepted as a legacy
comparison baseline. The script writes each run into a variant-specific
directory so the artifacts do not overwrite one another.

Run the same workflow with a LoRA actor:

```bash
./examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh \
  smoke \
  reward_based_filtering \
  /path/to/sft_exported_model \
  --corruption-config clean \
  --ft-mode lora
```

Static train-set corruption is configured via `--corruption-config`:

- `clean`
- `tail50_flip20`
- `tail50_flip40`
- `global_flip10`
- `global_flip20`
- `global_flip30`
- `global_flip40`

These settings only affect the train split. The eval holdout stays clean.

For this matrix, the launcher also disables training-time `late_flip` so that
static dataset corruption is not mixed with dynamic chosen/rejected swapping.

Method semantics for the focused Qwen2.5 study:

- `vanilla_dpo`: standard DPO, keep all preference pairs
- `random_pair_filtering`: keep a fixed random fraction of pairs per
  accumulation window
- `reward_based_filtering`: keep the top fraction of pairs ranked by the DPO
  trainer's own per-sample `rewards/margin`
- `self_inf`: keep pairs whose gradients align well with the window mean

`reward_based_filtering` does not use an external reward model. It ranks pairs
by the existing per-sample DPO reward margin.

`random_pair_filtering` and `reward_based_filtering` share
`CURATION_KEEP_RATIO` and default to `0.9` when unset.

Run the focused `4 methods x 3 datasets` matrix and print every command before
execution:

```bash
./examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft_flip_matrix.sh \
  /path/to/sft_exported_model \
  --ft-mode lora
```

Preview commands without executing them:

```bash
./examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft_flip_matrix.sh \
  /path/to/sft_exported_model \
  --ft-mode lora \
  --print-only
```

Filter the matrix to a subset of methods or corruption settings:

```bash
./examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft_flip_matrix.sh \
  /path/to/sft_exported_model \
  --ft-mode lora \
  --methods vanilla_dpo,reward_based_filtering \
  --datasets clean,global_flip20
```

For the full SFT -> DPO workflow, including the prompt-disjoint dataset split,
see `examples/ultrafeedback/README.md`.

Run the DPO smoke recipe with `self_inf_batch` curation:

```bash
source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate
cd /home/lhf_hongfu_gmail_com/tunix
./examples/dpo/run_qwen3_4b_ultrafeedback.sh smoke \
  dpo_config.use_dynamic_batch_curation=true \
  dpo_config.curation_variant=self_inf_batch \
  dpo_config.self_influence_dot_threshold=0.0
```

Run the full DPO recipe with `self_inf_batch` curation:

```bash
source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate
cd /home/lhf_hongfu_gmail_com/tunix
./examples/dpo/run_qwen3_4b_ultrafeedback.sh full \
  dpo_config.use_dynamic_batch_curation=true \
  dpo_config.curation_variant=self_inf_batch \
  dpo_config.self_influence_dot_threshold=0.0
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
