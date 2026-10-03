# DeepSeek 7B AIME extension

Development branch: `codex/aime-deepseek-7b`, based on `c168b7f4` from
`for_GRPO_vLLM_aime`. The pre-existing uncommitted `develop.md` notes are retained.

Server deployment uses the independent directory
`/home/jason_chia925_gmail_com/Project_7B/tunix` on both workers. Keep the original
`Project/tunix` checkout unchanged. Set `REPO` and `VENV` explicitly to the new
paths; the launchers retain their original defaults for existing 1.5B runs.

The shared GRPO, policy-only DTV scoring, LOO selection, masked Policy+KL
updates, dataset processing and reward implementation are unchanged. The
existing 1.5B launcher commands and model configuration remain the default.
The 7B preset reuses this pipeline with an explicit model configuration and
separate `grpo_aime_ds7b_full_*` run/log/cache directories.

## Model

Use `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, configuration id
`deepseek_r1_distill_qwen_7b`. Its 28 layers, hidden size 3584, intermediate size
18944, 28 attention heads, four KV heads, vocabulary 152064, untied embeddings
and RoPE theta 10000 match the official configuration:
https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/blob/main/config.json

This extends the same distillation family as the 1.5B experiment. The two sizes
have distinct backbone architectures and distillation histories, so describe
the result as an extension to a larger model in the same family rather than a
strict experiment changing only parameter count.

## Initial full training gates

Run these on the TPU login worker after deploying identical source to both
workers. Set the actual repository, model cache, data paths, TPU name and zone.
The model and tokenizer cache must be accessible at the configured paths on
both workers. Existing 1.5B checkpoints are not 7B initializations.

```bash
export REPO=/home/jason_chia925_gmail_com/Project_7B/tunix
export VENV="$REPO/.venv"
export PYTHONPATH="$REPO"
unset PYTHON_BIN
export MODEL_PATH=/path/to/deepseek-r1-distill-qwen-7b
export TPU_NAME=your-v5p-16-node
export ZONE=your-zone
export TRAIN_DATA_PATH=/path/to/deepscaler_train.json
export EVAL_DATA_PATH=/path/to/aime_eval.parquet

bash "$REPO/runs_xuesong/scripts/run_aime_7b_full.sh" baseline 0
bash "$REPO/runs_xuesong/scripts/run_aime_7b_full.sh" group_loo_policy 0
```

Each command defaults to two global updates, 16 prompts/update, eight
generations/prompt, train microbatch one prompt group, 8K response limit,
learning rate 1e-6, cosine decay horizon 314, beta 0.001, rollout concurrency
16 and full-parameter training. Run the methods sequentially on one TPU slice.
The 16-prompt gate does not establish that a 128-prompt update fits or is fast
enough; checkpoint save/restore also needs TPU validation.

After the small gates pass, test the target batch while keeping a short budget:

```bash
NUM_BATCHES=2 bash "$REPO/runs_xuesong/scripts/run_aime_7b_full.sh" baseline 0 \
  batch_size=128 rl_training_config.mini_batch_size=128
NUM_BATCHES=2 bash "$REPO/runs_xuesong/scripts/run_aime_7b_full.sh" group_loo_policy 0 \
  batch_size=128 rl_training_config.mini_batch_size=128
```

Measure steady rollout/scoring/update duration, HBM, reward degeneracy,
truncation and LOO retention before increasing `NUM_BATCHES`. Additional CLI
overrides have the same precedence as in the existing seeded launcher; record
the final resolved configuration. Keep full/LoRA settings matched between
baseline and LOO. This initial extension does not add a LoRA preset or change
the exact-vmap LOO backend.

## Evaluation

The existing evaluation launcher retains 1.5B as its default and now accepts
`--model-config`. The 7B wrapper selects the 7B architecture for both frozen
base-model and trained full-checkpoint evaluation. `--dry-run` prints the
resolved evaluation command without creating output directories or launching
workers.

```bash
bash "$REPO/runs_xuesong/scripts/run_aime_7b_eval.sh" \
  --run-root /path/to/7b-run --checkpoint-step 2 --dry-run
```

To check restore and generation after the two-step gate, remove `--dry-run`
and add `--limit 2 --num-samples 2 --max-generation-steps 512`, with a new
output directory. This short check is an integration test, not an accuracy
measurement. For frozen-model profiling, use `--checkpoint-source base_model`
and an explicit new `--output-dir`. Choose length and training settings on a
separate validation set before the final AIME comparison.

## Verification status

Local checks passed: shell/Python syntax, five launcher boundary tests and
default/7B evaluation command selection. These tests intercept TPU transport
and do not load weights or contact remote workers.

The second review verified that 16 existing runtime files remain byte-identical
to `c168b7f4`, including dual-worker orchestration/status handling, JAX rank
discovery, distributed vLLM transport, weight transfer/KV allocation reuse,
resharding, Agentic queues, the LOO trainer, the production schema and the
underlying 1.5B recipe. The two existing shell-backed remote-status regression
tests also passed. A real production-schema/role-inheritance test was added for
the 7B profile and must run in the server environment.

See [deployment and review notes](AIME_7B_DEPLOY.md) for the exact push/pull
commands, new per-worker shim setup and the verified-snapshot procedure for
worker 1. Cloning/updating worker 0 alone does not update worker 1. Do not link
the new `.venv` directly to the old shim: its import path can select old code.
The old SSH archive pipe in `start.md` predates
the archive-corruption diagnosis in `develop.md`; use the persistent archive
procedure instead.

This Mac has no JAX/Flax training dependencies, so architecture routing and
runtime regression tests must still run in the existing TPU Python environment:

```bash
cd "$REPO"
JAX_PLATFORMS=cpu "$REPO/.venv/bin/python" -m pytest -q \
  tests/models/qwen2/deepseek_qwen_config_test.py tests/cli/aime_launchers_test.py \
  tests/scripts/dual_worker_status_test.py tests/cli/grpo_main_distributed_test.py \
  tests/cli/config_test.py tests/cli/recipes/deepscaler_eval_test.py
bash "$REPO/runs_xuesong/scripts/run_aime_cpu_gates.sh"
```

Then verify one unchanged 1.5B training/restore/evaluation gate before starting
the 7B TPU gates. No TPU fit, throughput, checkpoint compatibility or training
improvement has been established by the local launcher checks.
