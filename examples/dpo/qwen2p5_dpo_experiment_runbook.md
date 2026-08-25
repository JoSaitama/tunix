# Qwen2.5 DPO Experiment Runbook

This note explains how to rerun the DPO experiments used for the clean/GF20/GF40 table. It is meant as a handoff document for another person to reproduce or extend the runs. It describes commands and benchmark coverage only, not results.

## Current Status

- The experiment stack is implemented under `examples/dpo/`.
- This is a two-stage pipeline:
  - Stage 1: full-weight SFT from the pretrain-only base `Qwen/Qwen2.5-1.5B`.
  - Stage 2: LoRA DPO variants initialized from the shared SFT export.
- The shared SFT initialization used by the DPO table is:
  - `runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/exported_model`
- The main DPO datasets currently used in the table are:
  - `clean`
  - `global_flip20`
  - `global_flip40`
- The main method rows are:
  - `vanilla_dpo`
  - `random_pair_filtering_filt5`
  - `random_pair_filtering` / `random_pair_filtering_filt10`
  - `reward_based_filtering_filt5`
  - `reward_based_filtering` / `reward_based_filtering_filt10`
  - `self_inf`
- The current final table uses these metrics:
  - `Val Acc-AUC`
  - `Test Acc`
  - `LiveBench-IF`
  - `RewardBench 2 Precise IF`
  - `IFBench Prompt Strict`

## Environment

Run from the repository root:

```bash
cd /home/lhf_hongfu_gmail_com/tunix
source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate
```

The training launcher expects `HF_TOKEN` to be available through `.env`, `my_example/.env`, or the shell environment because the Qwen tokenizer is loaded from Hugging Face.

The offline benchmark tooling lives in a separate environment. If it is missing, create it once with:

```bash
bash examples/dpo/setup_qwen2p5_clean_offline_eval_env.sh
```

Recommended shared variables:

```bash
export SFT_MODEL_PATH=/home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/exported_model
export RUN_TS=$(date +%Y%m%d_%H%M%S)
```

## Model Lineage

The model used in the DPO experiments was not initialized directly from an instruction-tuned checkpoint. It was trained as:

```text
Qwen/Qwen2.5-1.5B
  -> full-weight SFT on the SFT partition of UltraFeedback
  -> shared exported SFT model
  -> LoRA DPO variants on the DPO partition of UltraFeedback
```

The completed SFT run is:

```text
runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657
```

The DPO stage should use the exported model directory:

```text
runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/exported_model
```

Do not pass the SFT checkpoint directory to DPO. Use `exported_model`.

## Stage 1: SFT Source Model

The SFT stage starts from the base model:

```text
Qwen/Qwen2.5-1.5B
```

The SFT data source is:

```text
HuggingFaceH4/ultrafeedback_binarized
```

The SFT dataset wrapper is:

```text
examples/data/ultrafeedback_sft.py:create_dataset(...)
```

Implementation details:

- The SFT loader keeps `prompt + chosen`.
- Each sample is converted into a chat-style prompt and target response.
- The tokenizer chat template is applied in the SFT data path.
- Overlength samples are filtered by `max_target_length`.

## Stage 1: SFT Split

The UltraFeedback preference data is split deterministically at the prompt level. The SFT and DPO stages use disjoint prompt partitions.

SFT train module:

```text
examples/data/ultrafeedback_sft.py:create_dataset(split='train_prefs', partition='sft', subset='train', sft_fraction=0.25, eval_fraction=0.1, seed=42)
```

SFT eval module:

```text
examples/data/ultrafeedback_sft.py:create_dataset(split='train_prefs', partition='sft', subset='eval', sft_fraction=0.25, eval_fraction=0.1, seed=42)
```

Split semantics:

- `sft_fraction=0.25`: 25% of prompts from `train_prefs` go to SFT.
- The remaining 75% of prompts are reserved for DPO.
- `eval_fraction=0.1`: 10% of each partition is held out for training-time eval.
- `seed=42`: fixed prompt-level split seed.
- `test_prefs` is not used for SFT training or early stopping.

## Stage 1: SFT Recipe

The SFT config is:

```text
examples/sft/ultrafeedback/qwen2p5_1p5b_ultrafeedback.yaml
```

The completed table uses full-weight SFT, not LoRA SFT.

Important SFT hyperparameters:

| Setting | Value |
|---|---:|
| Base model | `Qwen/Qwen2.5-1.5B` |
| Fine-tuning mode | full-weight |
| Mesh | `(2,2)` with `('fsdp','tp')` |
| Batch size | 2 |
| Eval batch size | 2 |
| Gradient accumulation | 4 |
| Max target length | 768 |
| Optimizer | AdamW |
| Peak LR | `1e-5` |
| Warmup steps | 100 |
| Decay steps | 1500 |
| Weight decay | 0.05 |
| Max grad norm | 1.0 |
| Eval every | 100 steps |
| Save every | 100 steps |
| Max steps | 1500 |

The recorded SFT run finished at step `1493` because the dataset iterator ended before the hard `max_steps=1500` cap.

## Stage 1: Rerun SFT

Only rerun SFT if the shared SFT export is missing or if the experiment intentionally changes the upstream initialization. Otherwise, reuse the existing exported model for all DPO variants.

SFT smoke:

```bash
source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate
cd /home/lhf_hongfu_gmail_com/tunix
set -a
source my_example/.env
set +a

./examples/sft/ultrafeedback/run_qwen2p5_1p5b_ultrafeedback.sh smoke --ft-mode full
```

Full SFT reproduction:

```bash
source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate
cd /home/lhf_hongfu_gmail_com/tunix
set -a
source my_example/.env
set +a

export RUN_TS=$(date +%Y%m%d_%H%M%S)
export RUN_ROOT=/home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_${RUN_TS}

./examples/sft/ultrafeedback/run_qwen2p5_1p5b_ultrafeedback.sh full --ft-mode full
```

SFT outputs:

```text
${RUN_ROOT}/checkpoints
${RUN_ROOT}/tensorboard
${RUN_ROOT}/exported_model
```

After SFT finishes, set:

```bash
export SFT_MODEL_PATH=${RUN_ROOT}/exported_model
```

Then use this `SFT_MODEL_PATH` for every DPO method so all DPO rows share the same initialization.

## Method Names

Use these method identifiers in launcher commands:

| Method label | Preferred launcher variant | Compatibility alias | Meaning |
|---|---|---|---|
| Vanilla DPO | `vanilla_dpo` | none | Uses all preference pairs. |
| Random Pair Filtering (5%) | `random_pair_filtering_filt5` | none | Drops 5%, keeps 95%. |
| Random Pair Filtering (10%) | `random_pair_filtering_filt10` | `random_pair_filtering` | Drops 10%, keeps 90%. |
| Reward-based Filtering (5%) | `reward_based_filtering_filt5` | none | Drops 5%, keeps 95%, ranked by DPO reward margin. |
| Reward-based Filtering (10%) | `reward_based_filtering_filt10` | `reward_based_filtering` | Drops 10%, keeps 90%, ranked by DPO reward margin. |
| Self-DTV | `self_inf` | `self_inf_batch`, `self_dtv` | Batch self-influence / gradient-alignment filtering. |

For clean multi-seed scripts, use the preferred `*_filt10` names. For the legacy GF single-run timestamp `20260417_013847`, the 10% rows are stored under the compatibility aliases `random_pair_filtering` and `reward_based_filtering`, and `eval_qwen2p5_gf_benchmarks.py` discovers them with those names.

## Dataset Names

Use these dataset identifiers with `--corruption-config`:

| Dataset label | Launcher value | Meaning |
|---|---|---|
| Clean | `clean` | No static preference flip. |
| GF20 | `global_flip20` | Globally flips 20% of DPO train preference pairs. |
| GF40 | `global_flip40` | Globally flips 40% of DPO train preference pairs. |

The eval and test preference sets stay clean. Static corruption is applied only to DPO train data.

## Stage 2: DPO Data Split

The DPO stage uses the remaining prompt-disjoint UltraFeedback partition:

```text
examples/data/ultrafeedback_dpo.py:create_dataset(...)
```

Base DPO train module before static corruption is applied:

```text
examples/data/ultrafeedback_dpo.py:create_dataset(split='train_prefs', partition='dpo', subset='train', sft_fraction=0.25, eval_fraction=0.1, seed=42)
```

Base DPO eval module:

```text
examples/data/ultrafeedback_dpo.py:create_dataset(split='train_prefs', partition='dpo', subset='eval', sft_fraction=0.25, eval_fraction=0.1, seed=42)
```

Final test module used for `Test Acc`:

```text
examples/data/ultrafeedback_dpo.py:create_dataset(split='test_prefs', seed=42)
```

Important split/corruption rules:

- DPO train prompts are disjoint from SFT prompts because `partition='dpo'` is the complement of the SFT partition.
- DPO train/eval are also prompt-disjoint through `subset='train'` and `subset='eval'`.
- Static flip corruption is applied only to the DPO train module.
- DPO eval and final `test_prefs` remain clean.
- The launcher disables training-time `late_flip` for the static corruption experiments, so static dataset corruption is not mixed with late training corruption.

## Stage 2: DPO Recipe

The main DPO config is:

```text
examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft_lora.yaml
```

Both actor and reference models are initialized from the same SFT exported model. The actor is trained with LoRA, while the reference model remains fixed.

Important DPO hyperparameters:

| Setting | Value |
|---|---:|
| Actor/reference init | shared SFT `exported_model` |
| Fine-tuning mode | LoRA |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| LoRA rank | 64 |
| LoRA alpha | 64 |
| Mesh | `(2,2)` with `('fsdp','tp')` |
| Batch size | 8 |
| Eval batch size | 8 |
| Gradient accumulation | 32 |
| Max prompt length | 512 |
| Max response length | 512 |
| DPO beta | 0.01 |
| Optimizer | AdamW |
| Peak LR | `1e-6` |
| Warmup steps | 10 |
| Decay steps | 115 |
| Weight decay | 0.1 |
| Max grad norm | 0.1 |
| Eval every | 10 steps |
| Save every | 3 steps |
| Max steps | 115 |

The method variants only change curation behavior. The base optimizer, model, SFT initialization, and train/eval splits should stay fixed for a controlled comparison.

## Single DPO Training Command

This trains one full LoRA DPO run and exports the model.

```bash
RUN_TS=${RUN_TS} \
examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh \
  full \
  <variant> \
  "${SFT_MODEL_PATH}" \
  --corruption-config <clean|global_flip20|global_flip40> \
  --ft-mode lora
```

Example:

```bash
RUN_TS=${RUN_TS} \
examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh \
  full \
  self_inf \
  "${SFT_MODEL_PATH}" \
  --corruption-config global_flip20 \
  --ft-mode lora
```

The run directory is created under `runs/` and encodes method, dataset, fine-tuning mode, profile, and timestamp.

## Run the Full Clean/GF Matrix

This generates the training commands for all selected methods and datasets. Add `--print-only` to inspect commands without running.

```bash
RUN_TS=${RUN_TS} \
examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft_flip_matrix.sh \
  "${SFT_MODEL_PATH}" \
  --profile full \
  --ft-mode lora \
  --methods vanilla_dpo,random_pair_filtering_filt5,random_pair_filtering,reward_based_filtering_filt5,reward_based_filtering,self_inf \
  --datasets clean,global_flip20,global_flip40 \
  --print-only
```

To actually launch the runs sequentially, remove `--print-only`:

```bash
RUN_TS=${RUN_TS} \
examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft_flip_matrix.sh \
  "${SFT_MODEL_PATH}" \
  --profile full \
  --ft-mode lora \
  --methods vanilla_dpo,random_pair_filtering_filt5,random_pair_filtering,reward_based_filtering_filt5,reward_based_filtering,self_inf \
  --datasets clean,global_flip20,global_flip40
```

## Clean 3-Seed Main-Table Pipeline

Use this when the clean table needs true `mean/std` across DPO seeds. This does not rerun SFT. It runs clean-only DPO with seeds `0,1,2`, evaluates each seed, summarizes the table, and by default removes each `exported_model` after evaluation to save SSD.

```bash
export RUN_TS=$(date +%Y%m%d_%H%M%S)

tmux new-session -s clean_main_table_true_std_${RUN_TS}
```

Inside tmux:

```bash
cd /home/lhf_hongfu_gmail_com/tunix
source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate

RUN_TS=${RUN_TS} \
examples/dpo/run_qwen2p5_clean_main_table_pipeline.sh \
  --no-legacy \
  --seeds 0,1,2 \
  --filtering-levels 5,10 \
  --benchmarks clean_test,livebench_if,rewardbench2,ifbench
```

Important seed semantics for this pipeline:

- SFT checkpoint is fixed and reused.
- Prompt-level data split stays fixed.
- `run_seed` is encoded into the run name.
- `train_shuffle_seed` is set to the run seed so DPO train order changes across seeds.
- `curation_seed` is fixed to `0` in the no-legacy pipeline, so random-pair variance mainly comes from train order.

Outputs:

```text
runs/results/qwen2p5_clean_main_table_<RUN_TS>/per_run/
runs/results/qwen2p5_clean_main_table_<RUN_TS>/per_run_metrics.json
runs/results/qwen2p5_clean_main_table_<RUN_TS>/tables/clean_main_table_multiseed.json
runs/results/qwen2p5_clean_main_table_<RUN_TS>/tables/clean_main_table_multiseed.md
runs/results/qwen2p5_clean_main_table_<RUN_TS>/tables/clean_main_table_multiseed.tex
```

## Evaluate Clean Runs Only

If training has already been done and the `exported_model` directories are present, run:

```bash
python examples/dpo/eval_qwen2p5_clean_main_table_runs.py \
  --run-ts <RUN_TS> \
  --no-legacy \
  --profile full \
  --methods vanilla_dpo random_pair_filtering_filt5 random_pair_filtering_filt10 reward_based_filtering_filt5 reward_based_filtering_filt10 self_inf \
  --seeds 0 1 2 \
  --benchmarks clean_test livebench_if rewardbench2 ifbench \
  --output-root runs/results/qwen2p5_clean_main_table_<RUN_TS>
```

Then summarize:

```bash
python examples/dpo/summarize_qwen2p5_clean_main_table.py \
  --run-ts <RUN_TS> \
  --input-json runs/results/qwen2p5_clean_main_table_<RUN_TS>/per_run_metrics.json \
  --output-dir runs/results/qwen2p5_clean_main_table_<RUN_TS>/tables
```

## Evaluate GF20/GF40 Runs

This evaluator expects `exported_model` to exist for the requested runs.

```bash
python examples/dpo/eval_qwen2p5_gf_benchmarks.py \
  --run-ts <RUN_TS> \
  --datasets global_flip20 global_flip40 \
  --methods vanilla_dpo random_pair_filtering_filt5 random_pair_filtering reward_based_filtering_filt5 reward_based_filtering self_inf \
  --benchmarks preference livebench_if rewardbench2 ifbench \
  --output-root runs/results/qwen2p5_gf_benchmarks_<RUN_TS>
```

Outputs:

```text
runs/results/qwen2p5_gf_benchmarks_<RUN_TS>/per_run/
runs/results/qwen2p5_gf_benchmarks_<RUN_TS>/gf_benchmark_summary.json
runs/results/qwen2p5_gf_benchmarks_<RUN_TS>/tables/gf_benchmarks.md
runs/results/qwen2p5_gf_benchmarks_<RUN_TS>/tables/gf_benchmarks.tex
```

## GF20/GF40 5% Filtering Convenience Pipeline

This convenience script was used to fill the GF20/GF40 5% rows:

```bash
RUN_TS=<RUN_TS> \
OUTPUT_ROOT=/home/lhf_hongfu_gmail_com/tunix/runs/results/qwen2p5_gf_benchmarks_<RUN_TS> \
examples/dpo/run_qwen2p5_gf_filt5_benchmarks.sh
```

It runs only:

- `global_flip20 / random_pair_filtering_filt5`
- `global_flip20 / reward_based_filtering_filt5`
- `global_flip40 / random_pair_filtering_filt5`
- `global_flip40 / reward_based_filtering_filt5`

For each row, it trains if needed, evaluates the benchmark metrics, writes one `per_run/*.json`, and deletes that run's `exported_model` after evaluation to avoid filling the SSD.

## Benchmarks / Metrics

The current table benchmarks are all offline and do not require an OpenAI API key.

| Selector | Table column | What it does |
|---|---|---|
| `preference` | `Val Acc-AUC`, `Test Acc` | GF evaluator only. Computes validation accuracy AUC from tensorboard and clean `test_prefs` DPO reward accuracy. |
| `clean_test` | `Test Acc` | Clean evaluator. Evaluates clean held-out `test_prefs` DPO reward accuracy. |
| `livebench_if` | `LiveBench-IF` | Generates answers for the LiveBench instruction-following subset and scores them with the local instruction-following scorer. |
| `rewardbench2` | `RB2 Precise IF` | Evaluates the RewardBench 2 Precise IF subset using the DPO implicit reward scoring path. |
| `ifbench` | `IFBench P-Strict` | Generates IFBench responses and reports prompt-level strict instruction-following accuracy. |

Earlier optional benchmarks such as MT-Bench, AlpacaEval 2, Arena-Hard, and WildBench are not part of the current final table. Those may require external judge setup and, for judge-based runs, an OpenAI API key.

## SSD / tmux Recommendations

The SSD has been tight. A single exported Qwen2.5-1.5B model is roughly several GB, while checkpoints are much smaller. Prefer sequential pipelines and do not keep many `exported_model` directories at the same time.

Recommended pattern:

```bash
df -h /home/lhf_hongfu_gmail_com/tunix

tmux new-session -s dpo_<short_name>_<RUN_TS>
```

Inside tmux, run the relevant pipeline and redirect logs if desired:

```bash
RUN_TS=${RUN_TS} \
examples/dpo/run_qwen2p5_clean_main_table_pipeline.sh \
  --no-legacy \
  --seeds 0,1,2 \
  --filtering-levels 5,10 \
  --benchmarks clean_test,livebench_if,rewardbench2,ifbench \
  > runs/results/qwen2p5_clean_main_table_${RUN_TS}/pipeline.log 2>&1
```

Detach tmux with `Ctrl-b d`, reattach with:

```bash
tmux attach -t dpo_<short_name>_<RUN_TS>
```

Check progress:

```bash
tail -n 100 runs/results/qwen2p5_clean_main_table_${RUN_TS}/pipeline.log
find runs/results/qwen2p5_clean_main_table_${RUN_TS}/per_run -maxdepth 1 -type f | sort
df -h /home/lhf_hongfu_gmail_com/tunix
```

## Quick Smoke Examples

Train one clean self-DTV smoke:

```bash
RUN_TS=${RUN_TS} \
examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh \
  smoke \
  self_inf \
  "${SFT_MODEL_PATH}" \
  --corruption-config clean \
  --ft-mode lora
```

Train one GF20 reward-based 5% full run:

```bash
RUN_TS=${RUN_TS} \
examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh \
  full \
  reward_based_filtering_filt5 \
  "${SFT_MODEL_PATH}" \
  --corruption-config global_flip20 \
  --ft-mode lora
```

Evaluate one GF20 reward-based 5% run:

```bash
python examples/dpo/eval_qwen2p5_gf_benchmarks.py \
  --run-ts "${RUN_TS}" \
  --datasets global_flip20 \
  --methods reward_based_filtering_filt5 \
  --benchmarks preference livebench_if rewardbench2 ifbench \
  --output-root runs/results/qwen2p5_gf_benchmarks_${RUN_TS}
```
