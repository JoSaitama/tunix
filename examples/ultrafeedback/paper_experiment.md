# Qwen2.5-1.5B UltraFeedback Full-SFT Experiment

This note records the exact `qwen2.5-1.5b` SFT experiment that was run in this
repo, how it is implemented, and the exact command line needed to reproduce it.

## Goal

Train the pretrain-only base `Qwen/Qwen2.5-1.5B` into a usable chat-style model
with full-weight SFT on the `sft` partition of
`HuggingFaceH4/ultrafeedback_binarized`.

This SFT model is intended to be the initialization for the later DPO stage.

## Implementation

The experiment is implemented by these files:

- `tunix/examples/data/ultrafeedback_sft.py`
  - loads `HuggingFaceH4/ultrafeedback_binarized`
  - keeps only `prompt + chosen`
  - converts each sample into:
    - `prompt=[{"role": "user", "content": ...}]`
    - `response=<chosen response text>`
- `tunix/cli/utils/data.py`
  - applies the Hugging Face chat template
  - tokenizes `prompt + response`
  - builds the SFT `TrainingInput`
  - filters examples longer than `max_target_length`
- `examples/sft/ultrafeedback/qwen2p5_1p5b_ultrafeedback.yaml`
  - full-weight SFT recipe
- `examples/sft/ultrafeedback/run_qwen2p5_1p5b_ultrafeedback.sh`
  - launcher used to run the experiment
  - writes a temporary resolved config, then launches `tunix.cli.peft_main`

## Data split

This experiment uses a deterministic prompt-level split:

- train split source: `train_prefs`
- SFT partition: `partition='sft'`
- SFT fraction: `sft_fraction=0.25`
- SFT eval holdout inside `train_prefs`: `eval_fraction=0.1`
- seed: `42`

Concrete modules used by the recipe:

- train:
  - `examples/data/ultrafeedback_sft.py:create_dataset(split='train_prefs', partition='sft', subset='train', sft_fraction=0.25, eval_fraction=0.1, seed=42)`
- eval:
  - `examples/data/ultrafeedback_sft.py:create_dataset(split='train_prefs', partition='sft', subset='eval', sft_fraction=0.25, eval_fraction=0.1, seed=42)`

This means:

- SFT and DPO training prompts are disjoint
- SFT training and SFT eval prompts are also disjoint
- `test_prefs` is not used for SFT early-stop

## Hyperparameters

The full-SFT recipe used these settings:

- model: `qwen2.5-1.5b`
- mode: `full`
- mesh: `(2,2)`
- batch size: `2`
- eval batch size: `2`
- gradient accumulation: `4`
- max target length: `768`
- optimizer: `adamw`
- peak lr: `1e-5`
- warmup steps: `100`
- decay steps: `1500`
- weight decay: `0.05`
- max grad norm: `1.0`
- eval every: `100` steps
- save every: `100` steps
- max steps: `1500`

The actual run ended at step `1493` because the dataset iterator finished
before reaching the hard `max_steps` cap.

## Exact reproduction command

Prerequisites:

- TPU worker is available
- `HF_TOKEN` is set in `my_example/.env` or `.env`
- repo root is `/home/lhf_hongfu_gmail_com/tunix`

If this worker has a stale TPU lockfile, clear it first:

```bash
python - <<'PY'
import os
path = "/tmp/libtpu_lockfile"
if os.path.exists(path):
    os.unlink(path)
    print(f"removed {path}")
else:
    print(f"not found: {path}")
PY
```

Then run the experiment:

```bash
source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate
cd /home/lhf_hongfu_gmail_com/tunix
set -a
source my_example/.env
set +a
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_ROOT=/home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_${RUN_TS}
export RUN_TS RUN_ROOT
./examples/sft/ultrafeedback/run_qwen2p5_1p5b_ultrafeedback.sh full --ft-mode full
```

One-line form:

```bash
source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && cd /home/lhf_hongfu_gmail_com/tunix && set -a && source my_example/.env && set +a && RUN_TS=$(date +%Y%m%d_%H%M%S) && RUN_ROOT=/home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_${RUN_TS} && export RUN_TS RUN_ROOT && ./examples/sft/ultrafeedback/run_qwen2p5_1p5b_ultrafeedback.sh full --ft-mode full
```

## Outputs

The run writes to:

- checkpoints:
  - `${RUN_ROOT}/checkpoints`
- tensorboard:
  - `${RUN_ROOT}/tensorboard`
- exported model:
  - `${RUN_ROOT}/exported_model`

The finished run produced:

- run root:
  - `/home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657`
- exported model:
  - `/home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/exported_model`
- retained checkpoint:
  - `/home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/checkpoints/1493`

## What to use later

For DPO, use the exported model directory, not the checkpoint directory:

```bash
SFT_MODEL=/home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/exported_model
```

Then pass `${SFT_MODEL}` into the DPO launcher.

## Sanity checks

Check TensorBoard metrics:

```bash
python - <<'PY'
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

run_dir = Path("/home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/tensorboard")
event_file = sorted(run_dir.glob("events.out.tfevents.*"))[-1]
ea = event_accumulator.EventAccumulator(str(event_file))
ea.Reload()
for tag in ["sft/eval/loss", "sft/eval/perplexity"]:
    print(tag)
    for point in ea.Scalars(tag):
        print(point.step, point.value)
PY
```

Check that the exported tokenizer formats chat prompts correctly:

```bash
source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate
python - <<'PY'
from transformers import AutoTokenizer

model_dir = "/home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/exported_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "请用一句话介绍你自己。"}],
    tokenize=False,
    add_generation_prompt=True,
)
print(text)
PY
```

## Summary of the completed run

- SFT completed successfully
- exported model was saved successfully
- final checkpoint retained: `1493`
- the model can answer normal chat prompts
- it is suitable to use as the SFT initialization for the next DPO stage
