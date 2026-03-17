# Qwen2.5-1.5B UltraFeedback SFT

This recipe runs SFT on the prompt-disjoint `sft` partition of
`HuggingFaceH4/ultrafeedback_binarized` using the pretrain-only base
`Qwen/Qwen2.5-1.5B`. Default mode is full fine-tuning; pass `--ft-mode lora`
to switch to LoRA. The default recipe uses `sft_fraction=0.25` and evaluates
on a prompt-level holdout inside `train_prefs`, not on `test_prefs`.

## Files

- `qwen2p5_1p5b_ultrafeedback.yaml`: full SFT config
- `qwen2p5_1p5b_ultrafeedback_lora.yaml`: LoRA SFT config
- `run_qwen2p5_1p5b_ultrafeedback.sh`: launcher for `full` and `smoke`

## Run

```bash
source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate
cd /home/lhf_hongfu_gmail_com/tunix
./examples/sft/ultrafeedback/run_qwen2p5_1p5b_ultrafeedback.sh smoke
```

Run LoRA mode:

```bash
./examples/sft/ultrafeedback/run_qwen2p5_1p5b_ultrafeedback.sh smoke --ft-mode lora
```

The script writes a loadable safetensors model into a unique
`runs/.../exported_model` directory. That output is the expected input for the
matching DPO recipe under `examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh`.

For the full two-stage workflow, see `examples/ultrafeedback/README.md`.
