# Qwen2.5-1.5B UltraFeedback End-to-End Workflow

This workflow uses the pretrain-only base `Qwen/Qwen2.5-1.5B` and splits
`HuggingFaceH4/ultrafeedback_binarized` into prompt-disjoint `sft` and `dpo`
partitions. The default ratio is `0.25 / 0.75`, and training-time evaluation
uses a second prompt-level holdout carved out of `train_prefs`.

## Data split

- SFT train: `train_prefs`, `partition='sft'`, `subset='train'`,
  `sft_fraction=0.25`, `eval_fraction=0.1`, `seed=42`
- SFT eval: `train_prefs`, `partition='sft'`, `subset='eval'`,
  `sft_fraction=0.25`, `eval_fraction=0.1`, `seed=42`
- DPO train: `train_prefs`, `partition='dpo'`, `subset='train'`,
  `sft_fraction=0.25`, `eval_fraction=0.1`, `seed=42`
- DPO eval: `train_prefs`, `partition='dpo'`, `subset='eval'`,
  `sft_fraction=0.25`, `eval_fraction=0.1`, `seed=42`
- Final test: reserve `test_prefs` for final reporting instead of early-stop
  selection

The split is deterministic and prompt-disjoint: the same prompt never appears
in both the SFT train set and the DPO train set, and the `train/eval` holdout
inside each stage is also deterministic at the prompt level.

## Step 1: Run SFT

Default mode is full fine-tuning:

```bash
source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate
cd /home/lhf_hongfu_gmail_com/tunix
./examples/sft/ultrafeedback/run_qwen2p5_1p5b_ultrafeedback.sh smoke
./examples/sft/ultrafeedback/run_qwen2p5_1p5b_ultrafeedback.sh full
```

Run LoRA SFT instead:

```bash
./examples/sft/ultrafeedback/run_qwen2p5_1p5b_ultrafeedback.sh smoke --ft-mode lora
./examples/sft/ultrafeedback/run_qwen2p5_1p5b_ultrafeedback.sh full --ft-mode lora
```

Each run writes a loadable safetensors model directory to:

```text
runs/sft_qwen2p5_1p5b_ultrafeedback_<ft-mode>_<profile>_<timestamp>/exported_model
```

Use that `exported_model` path as the next stage's input.

## Step 2: Run DPO + DBC

All DPO variants must reuse the same SFT `exported_model` path.

Default mode is full DPO:

```bash
SFT_MODEL=/path/to/sft_exported_model

./examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh smoke baseline "${SFT_MODEL}"
./examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh smoke outlier_l2 "${SFT_MODEL}"
./examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh smoke self_inf_batch "${SFT_MODEL}"
```

Full runs:

```bash
./examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh full baseline "${SFT_MODEL}"
./examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh full outlier_l2 "${SFT_MODEL}"
./examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh full self_inf_batch "${SFT_MODEL}"
```

Run LoRA DPO instead:

```bash
./examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh smoke baseline "${SFT_MODEL}" --ft-mode lora
./examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh smoke outlier_l2 "${SFT_MODEL}" --ft-mode lora
./examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh smoke self_inf_batch "${SFT_MODEL}" --ft-mode lora
```

The DPO launcher supports all stage-wise combinations:

- `full SFT -> full DPO`
- `full SFT -> lora DPO`
- `lora SFT -> full DPO`
- `lora SFT -> lora DPO`

## Notes for experiments

- `baseline`, `outlier_l2`, and `self_inf_batch` should share the exact same
  SFT `exported_model`.
- DPO outputs are isolated by variant, `ft-mode`, profile, and timestamp, so
  runs do not overwrite one another.
- `HF_TOKEN` is required.
