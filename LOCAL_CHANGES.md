# Local / WIP Changes in This Fork

> 中文说明：这个仓库里有一些为了在多 host TPU（例如 v4-32）上跑 `my_example`/GRPO 做的临时改动。下面列出**哪些是我改的**以及目前已知的**局限/风险**，方便你上传 GitHub 时讲清楚“这不是 Tunix upstream 的默认行为”。

## How to inspect “what I changed”

- `git status` shows modified/untracked files.
- `git diff` shows exact code diffs.
- `git blame <file>` shows whether a line is upstream or local (`Not Committed Yet`).

## Summary of local modifications

### `my_example/config.py`

- Adds `--no-checkpoint` and treats `--checkpoint-root` as optional (`None`/`null`/empty → disabled).

**Limitations**
- Only covers the example entrypoint behavior; checkpoint semantics still depend on Tunix trainer internals.

### `my_example/main.py`

- Adds multi-host JAX init via `jax.distributed.initialize()` when `JAX_COORDINATOR_ADDRESS` is set.
- Prints basic distributed debug info (`process_index`, `process_count`, `device_count`, `local_device_count`).
- Skips checkpoint restore when checkpointing is disabled.

**Limitations**
- `process_id` is inferred from hostname suffix `*-w-<id>`; this is TPU-VM specific and may break on other naming schemes.
- This only initializes JAX; it does not guarantee the rest of the pipeline is “correctly per-host sharded”.

### `my_example/model.py`

- Wraps `qwix.apply_lora_to_model(...)` in an `@nnx.jit` helper to avoid eager-time shape mismatches across processes.

**Limitations**
- Can increase compile time and may hide underlying sharding/shape issues; lightly tested.

### `my_example/run_grpo_gemma.sh`

- Makes `METRICS_LOG_DIR` configurable.
- Adds `CKPT_BUCKET` → `CHECKPOINT_ROOT` logic (warns if running multi-host without shared storage).
- Sets `--save-interval-steps 100000` to effectively avoid frequent saves.

**Limitations**
- Without a shared bucket, multi-host checkpointing is expected to fail (warning is emitted).

### `tunix/generate/sampler.py`

- Adds `jax.experimental.multihost_utils.process_allgather(...)` on `out_tokens/lengths` (when `pad_output=True`) before `jax.device_get(...)`, as a workaround for the error:
  - “`jax.device_get` spans non-addressable devices”

**Limitations / Risks**
- This can change semantics: `process_allgather` may make every host see a full (or duplicated) batch.
- Downstream code in Tunix may assume each process only holds its **local shard**; gathering here can lead to:
  - duplicated samples,
  - inflated effective batch size,
  - higher memory usage / OOM,
  - misleading throughput scaling.
- This is a debugging workaround, not an upstream-quality solution.

### `tunix/sft/sharding_utils.py`

- Updates `shard_input(...)` to handle non-fully-addressable `jax.Array` inputs by using `jax.device_put(..., sharding)` instead of `jax.make_array_from_process_local_data(...)`.

**Limitations**
- Not fully validated across all sharding patterns; may trigger expensive resharding and add overhead.
- Does not replace the need for a correct per-process input pipeline (local shards per host).

### `tunix/rl/grpo/grpo_learner.py`

- Adds conditional prompt/string repetition when `completion_ids` length matches `prompt_ids * num_generations`.
- Adds conditional repetition of `ref_per_token_logps` when shapes suggest it needs to match the completion batch.

**Limitations**
- This can mask the true source of mismatch (e.g. rollout output gathering/replication).
- Only correct if the mismatch is strictly due to `num_generations` expansion; otherwise it may hide bugs.

### `.gitignore`

- Ignores `data/` to avoid accidentally committing local dataset caches/downloads.

## Known issues observed during multi-host runs (context)

- TPU HBM OOM can happen in `log_softmax` / logits-related intermediates with large vocab (e.g. Gemma3 `V=262144`) when `tp=1`. Memory roughly scales with `B×T×V`; doubling `train_micro_batch_size` can easily blow up.
- If outputs are unintentionally gathered/replicated across hosts, effective `B` may silently increase and make OOM more likely.

## What is intentionally *not* committed

- `data/` contains local caches/downloads and should not be pushed.

