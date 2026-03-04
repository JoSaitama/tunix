# Development Log

This file tracks engineering changes made in this repository.

## Update Policy

- Every code/doc/script change must append or update an entry in this file.
- Each entry should include: date, scope, changed files, validation, and known risks.
- If a task has no code changes, note that explicitly.

---

## 2026-03-04: DeepScaler rollout backend integration + sglang_jax stability work

### Scope

- Added rollout backend selection for DeepScaler training (`vanilla`, `vllm`, `sglang_jax`).
- Added sglang_jax rollout support and stability guards.
- Added configurable GRPO rollout orchestration concurrency for `sglang_jax` with safe default.
- Added runnable command documentation and troubleshooting notes.

### Key behavior changes

- `run_train.sh` now supports non-vanilla rollout engines through `train_deepscaler_nb.py` CLI.
- `run_train.sh` now passes `--grpo-max-concurrency`, defaulting to `1` via env var `GRPO_MAX_CONCURRENCY`.
- `sglang_jax` rollout remains conservative by default (sequential episodes + concurrency 1) for stability.

### Changed files

1. `examples/deepscaler/train_deepscaler_nb.py`
2. `examples/deepscaler/run_train.sh`
3. `examples/deepscaler/README.md`
4. `tunix/generate/sglang_jax_sampler.py`
5. `tunix/rl/agentic/pipeline/rollout_orchestrator.py`
6. `tunix/rl/experimental/agentic_rl_learner.py`
7. `tunix/rl/rollout/sglang_jax_rollout.py`

### Detailed change notes

#### 1) `examples/deepscaler/train_deepscaler_nb.py`

- Added rollout-engine CLI surface:
  - `--rollout-engine {vanilla,vllm,sglang_jax,sglang-jax}`
  - `--rollout-model-source`
  - backend-specific flags for vLLM and sglang-jax
- Added helper utilities:
  - rollout engine normalization
  - rollout model source resolution
  - rollout mesh building (separate from training mesh for sglang_jax)
- Added non-vanilla preflight checks:
  - dependency presence (`vllm`, `sgl_jax`)
  - rollout arg sanity checks
  - rollout model source path/repo validation
- Replaced hardcoded rollout config with per-engine `_build_rollout_config(...)`.
- Role mesh now uses:
  - actor/reference -> training mesh
  - rollout -> rollout mesh
- Added `--grpo-max-concurrency`:
  - validation: must be positive integer if set
  - applied to `sglang_jax` path
  - default remains `1` for `sglang_jax` when unset

#### 2) `examples/deepscaler/run_train.sh`

- Added:
  - `GRPO_MAX_CONCURRENCY="${GRPO_MAX_CONCURRENCY:-1}"`
- Added passthrough:
  - `--grpo-max-concurrency "$GRPO_MAX_CONCURRENCY"`

#### 3) `examples/deepscaler/README.md`

- Added rollout backend usage section and env notes.
- Added reproducible commands:
  - vanilla smoke
  - sglang_jax smoke (`--rollout-tp 2`)
  - sglang_jax non-smoke
- Added low-memory fallback profile for `RESOURCE_EXHAUSTED`.
- Added first-step probe command.
- Added disk cleanup note for checkpoint failures (`No space left on device`).
- Added note for overriding default GRPO concurrency via env var.

#### 4) `tunix/generate/sglang_jax_sampler.py`

- Added engine args change to reduce instability:
  - `disable_overlap_schedule=True`

#### 5) `tunix/rl/agentic/pipeline/rollout_orchestrator.py`

- Added `run_episodes_sequentially` option.
- Supports sequential per-pair episode execution mode.

#### 6) `tunix/rl/experimental/agentic_rl_learner.py`

- Enabled `run_episodes_sequentially` only for `sglang_jax`.

#### 7) `tunix/rl/rollout/sglang_jax_rollout.py`

- Updated sync behavior to only sync `nnx.Param` state.

### Runtime validations performed

- CLI/parse/compile checks:
  - `python examples/deepscaler/train_deepscaler_nb.py --help`
  - `./examples/deepscaler/run_train.sh --help`
  - `python -m py_compile examples/deepscaler/train_deepscaler_nb.py`
- Dependency preflight checks verified:
  - `--rollout-engine vllm` fails early if `vllm` missing
  - `--rollout-engine sglang_jax` fails early if `sgl_jax` missing
- Alias check verified:
  - `--rollout-engine sglang-jax`
- Training smoke validation:
  - vanilla smoke path: success
  - sglang_jax smoke path: success
- Non-smoke experiments:
  - Reached training loop on sglang_jax path
  - Encountered/mitigated infra/runtime issues (see below)

### Observed runtime issues and mitigations

1. TPU metadata access failures in sandboxed run  
   - Symptom: repeated `Failed to get TPU metadata (tpu-env)`  
   - Mitigation: rerun outside sandbox restrictions.

2. Checkpoint finalize failure due to disk exhaustion  
   - Symptom: `No space left on device` during Orbax write  
   - Mitigation: cleaned old `/tmp/deepscaler_ckpt_*` directories.

3. Occasional sglang_jax SIGTERM during precompile  
   - Symptom: process received SIGTERM in decode precompile stage  
   - Mitigation: rerun; observed successful completion on retry.

4. Non-smoke default profile OOM (`jit__train_step`)  
   - Symptom: attempted reserve ~81G with ~66.95G available  
   - Mitigation: use lower-memory run profile:
     - `--max-prompt-length 1024`
     - `--total-generation-steps 512` (or controlled increase)
     - `--batch-size 8 --mini-batch-size 8 --train-micro-batch-size 1`

### Commit reference

- Pushed commit: `b5be7bf`
- Branch: `my-changes`
- Remote: `origin` (`yangziao56/tunix`)

---

## 2026-03-04: DeepScaler dtype controls (train + reward/advantage + sglang_jax rollout)

### Scope

- Added configurable dtype controls for DeepScaler training instead of fixed FP32-only behavior.
- Added a single reward/advantage dtype switch (one parameter, not separate reward/advantage flags).
- Added sglang_jax rollout dtype and KV-cache dtype controls with safe defaults and alias support.

### Changed files

1. `examples/deepscaler/run_train.sh`
2. `examples/deepscaler/train_deepscaler_nb.py`
3. `tunix/rl/experimental/agentic_grpo_learner.py`
4. `tunix/rl/rollout/base_rollout.py`
5. `tunix/rl/rollout/sglang_jax_rollout.py`
6. `tunix/generate/sglang_jax_sampler.py`

### Key behavior changes

- `run_train.sh` adds env-based defaults and passthrough:
  - `TRAIN_DTYPE` (default `bf16`, options `fp32|bf16`)
  - `REWARD_ADVANTAGE_DTYPE` (default `bf16`, options `fp32|bf16`)
  - `ROLLOUT_SGLANG_JAX_DTYPE` (default `auto`, supports `float32|bfloat16|float16|half|float|fp32|bf16`)
  - `ROLLOUT_SGLANG_JAX_KV_CACHE_DTYPE` (default `auto`, supports `bf16|fp8_e5m2|fp8_e4m3`)
- `train_deepscaler_nb.py`:
  - new CLI flags for the four dtype knobs above
  - normalization/validation helpers for sglang_jax dtype arguments
  - model load dtype now follows `--train-dtype` instead of hardcoded `jnp.float32`
  - exports `TUNIX_REWARD_ADVANTAGE_DTYPE` for learner-side reward/advantage casting
- GRPO learner path now optionally casts rewards and advantages via `TUNIX_REWARD_ADVANTAGE_DTYPE`.
- sglang_jax rollout config path now forwards `dtype` and `kv_cache_dtype` to sampler engine args.

### Validation

- Syntax/compile check:
  - `python -m py_compile examples/deepscaler/train_deepscaler_nb.py`
- CLI surface check:
  - `python examples/deepscaler/train_deepscaler_nb.py --help`
  - `./examples/deepscaler/run_train.sh --help`
- Static verification by diff inspection:
  - confirmed single reward/advantage dtype switch (`--reward-advantage-dtype`)
  - confirmed rollout dtype knobs are passed end-to-end into `SglangJaxConfig`

### Known risks / TODO

- Runtime numerics differ when using `bf16` vs `fp32`; this is expected and workload-dependent.
- `--rollout-sglang-jax-kv-cache-dtype float32` remains unsupported by sglang_jax API surface (no change).
- End-to-end long-run training validation for all dtype combinations is still pending.
