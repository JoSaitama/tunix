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

## 2026-03-04: Fix empty-pytree offload crash in RL memory-kind transfer

### Scope

- Fixed crash when rollout model variables pytree is empty and offload/load path
  tries to infer memory kind.

### Root cause

- In `tunix/rl/utils.py::put_params_on_memory_kind`, code computed:
  - `original_shardings = jax.tree.map(lambda x: x.sharding, params)`
  - then `tree_reduce(operator.or_, ...)`
- For empty pytrees, `tree_reduce` raised:
  - `TypeError: reduce() of empty iterable with no initial value`

### Fix

- Added early return guard for empty `original_shardings` leaves:
  - if no leaves, log and return original params without transfer.

### Changed files

1. `tunix/rl/utils.py`

### Validation

- `python -m py_compile tunix/rl/utils.py`
- Runtime check in `.venv_sglang312`:
  - `put_params_on_memory_kind({}, 'device')` -> OK
  - `put_params_on_memory_kind({}, 'pinned_host')` -> OK

### Risk

- Minimal/low risk: behavior changes only for empty pytrees.
- Non-empty parameter trees keep previous logic unchanged.

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

---

## 2026-03-04: DeepScaler add configurable CPU offload switch

### Scope

- Added explicit `offload_to_cpu` switch for DeepScaler training flow.
- Kept default behavior unchanged (`offload_to_cpu=False`).
- Exposed shell-level env toggle in `run_train.sh` for easy use.

### Changed files

1. `examples/deepscaler/train_deepscaler_nb.py`
2. `examples/deepscaler/run_train.sh`

### Key behavior changes

- New CLI in training entry:
  - `--offload-to-cpu` / `--no-offload-to-cpu`
  - default: `--no-offload-to-cpu`
- `ClusterConfig.offload_to_cpu` is now wired to CLI argument (previously hardcoded `False`).
- `run_train.sh` now supports:
  - `OFFLOAD_TO_CPU=true|false` (default `false`)
  - auto-maps env to the corresponding boolean flag.

### Validation

- `python -m py_compile examples/deepscaler/train_deepscaler_nb.py`
- `python examples/deepscaler/train_deepscaler_nb.py --help`
- `bash examples/deepscaler/run_train.sh --help`

### Known risks / TODO

- Enabling CPU offload may reduce throughput due to host-device transfer overhead.
- Memory-pressure relief depends on workload shape and rollout backend.

---

## 2026-03-04: run_train.sh expose batch-size env knobs

### Scope

- Added batch-related env knobs to `run_train.sh` for easier shell-level tuning.
- Restored `OFFLOAD_TO_CPU` default in `run_train.sh` to `false` to match documented and CLI default behavior.

### Changed files

1. `examples/deepscaler/run_train.sh`

### Key behavior changes

- New env vars in shell entrypoint:
  - `BATCH_SIZE` (default `32`) -> `--batch-size`
  - `MINI_BATCH_SIZE` (default `32`) -> `--mini-batch-size`
  - `TRAIN_MICRO_BATCH_SIZE` (default `1`) -> `--train-micro-batch-size`
- `OFFLOAD_TO_CPU` default is now `false` in `run_train.sh`.

### Validation

- `bash examples/deepscaler/run_train.sh --help`
- `python examples/deepscaler/train_deepscaler_nb.py --help`

### Known risks / TODO

- Invalid numeric env values (non-integers) will fail in argparse, as expected.

---

## 2026-03-04: Git rollback triage (no code changes)

### Scope

- Investigated whether recent repository history was rolled back too far.
- Focused on branch/HEAD movement and working-tree diffs.

### Changed files

1. `develop.md`

### Key behavior changes

- No code changes.
- No script changes.
- No config changes.

### Validation

- `git status --short --branch`
- `git reflog --all -n 30 --date=iso`
- `git stash list`
- `git diff -- examples/deepscaler/run_train.sh`
- `git diff -- examples/deepscaler/train_deepscaler_nb.py`

### Validation results

- `HEAD` remains at `64d0e7a` on `my-changes`; no `reset`/`rebase` rollback events found in reflog.
- Current differences are uncommitted working-tree edits, not branch history rollback.
- No stash entries available for recovery.

### Known risks / TODO

- If content was lost via IDE undo/local history (never committed/stashed), Git cannot recover it.

---

## 2026-03-04: Re-add OFFLOAD_TO_CPU passthrough in run_train.sh

### Scope

- Restored `OFFLOAD_TO_CPU` env passthrough in `examples/deepscaler/run_train.sh` after local rollback.
- No changes to Python CLI structure (already had `--offload-to-cpu` in training entrypoint).

### Changed files

1. `examples/deepscaler/run_train.sh`
2. `develop.md`

### Key behavior changes

- Added `OFFLOAD_TO_CPU` env knob with accepted values:
  - `true|1|yes|y` -> `--offload-to-cpu`
  - `false|0|no|n` -> `--no-offload-to-cpu`
- Default remains `OFFLOAD_TO_CPU=false`.
- Invalid values fail fast with a clear error.

### Validation

- `bash examples/deepscaler/run_train.sh --help`
- `python examples/deepscaler/train_deepscaler_nb.py --help`

### Known risks / TODO

- None beyond existing offload throughput tradeoff.

---

## 2026-03-04: Fix sglang_jax concurrent event-loop re-entry in rollout

### Scope

- Fixed `RuntimeError: this event loop is already running` when using sglang_jax rollout with concurrency > 1.
- Root cause was concurrent thread access into one `sglang_jax Engine` instance (`engine.generate(...)`) and concurrent param update/generate races.

### Changed files

1. `tunix/generate/sglang_jax_sampler.py`

### Key behavior changes

- Added a sampler-level re-entrant lock guarding engine-facing operations.
- Serialized concurrent calls to:
  - `Engine.generate(...)`
  - parameter updates (`update_params(...)`) that mutate engine model state.
- Replaced shared mutable `self.sampling_params` with a per-call local sampling params object to avoid cross-thread data races.

### Validation

- `python -m py_compile tunix/generate/sglang_jax_sampler.py`

### Known risks / TODO

- This fix prioritizes correctness/stability; heavy concurrent requests will serialize at engine boundary.
- Throughput scaling for high concurrency may still be limited by sglang_jax engine design.

---

## 2026-03-05: DeepScaler GRPO slow-train path review (no code changes)

### Scope

- Read and traced the full DeepScaler training path for:
  - `examples/deepscaler/run_train.sh`
  - `examples/deepscaler/train_deepscaler_nb.py`
  - `tunix/rl/experimental/agentic_rl_learner.py`
  - `tunix/rl/experimental/agentic_grpo_learner.py`
  - `tunix/rl/rl_cluster.py`
  - `tunix/rl/rollout/sglang_jax_rollout.py`
  - `tunix/generate/sglang_jax_sampler.py`
  - related agentic orchestrator/trajectory/reward utilities
- Goal of this task was performance diagnosis only; no behavior change requested.

### Changed files

1. `develop.md`

### Key behavior changes

- 无代码改动。
- 无脚本改动。
- 无配置改动。

### Validation

- `sed -n '1,220p' examples/deepscaler/run_train.sh`
- `sed -n '1,980p' examples/deepscaler/train_deepscaler_nb.py`
- `sed -n '1,820p' tunix/rl/experimental/agentic_rl_learner.py`
- `sed -n '1,620p' tunix/rl/experimental/agentic_grpo_learner.py`
- `sed -n '1,1120p' tunix/rl/rl_cluster.py`
- `sed -n '1,340p' tunix/generate/sglang_jax_sampler.py`
- `sed -n '1,520p' tunix/utils/math_utils.py`

### Known risks / TODO

- Bottleneck ranking is based on static code-path analysis; no runtime profiler trace was collected in this task.

---

## 2026-03-05: Add sglang_jax rollout fast-path (batched generate) for DeepScaler GRPO

### Scope

- Implemented explicit rollout fast-path for agentic GRPO training path to reduce per-sample orchestrator overhead under `rollout_engine=sglang_jax`.
- Added independent rollout prompt batch-size control for fast-path.
- Kept legacy orchestrator path unchanged as default/fallback.

### Changed files

1. `tunix/rl/experimental/agentic_rl_learner.py`
2. `examples/deepscaler/train_deepscaler_nb.py`
3. `tests/rl/experimental/agentic_grpo_learner_test.py`
4. `develop.md`

### Key behavior changes

- New algorithm config fields in agentic RL base config:
  - `enable_rollout_fast_path: bool = False`
  - `rollout_prompt_batch_size: int | None = None`
- New fast-path producer in `AgenticRLLearner`:
  - Uses batched `rl_cluster.generate(...)` calls.
  - Splits full-batch prompts by `rollout_prompt_batch_size`.
  - Expands each prompt by `num_generations` and reconstructs per-prompt groups for existing `_batch_to_train_example(...)` pipeline.
  - Preserves training-side micro-batch consumption and weight-sync behavior.
- OOM handling in fast-path rollout:
  - Detects memory-exhausted errors and raises a clear actionable RuntimeError with parameter tuning suggestions.
- DeepScaler CLI additions:
  - `--enable-rollout-fast-path`
  - `--rollout-prompt-batch-size`
- DeepScaler runtime validation:
  - Fast-path only allowed with `--rollout-engine sglang_jax`.
  - `--rollout-prompt-batch-size` must be positive when set.
  - Warn if rollout prompt batch size is provided without fast-path enabled.
  - Print note that `--grpo-max-concurrency` is ignored when fast-path is enabled.
- New tests:
  - Fast-path chunking + queue count correctness.
  - Fast-path memory error message surfacing.

### Validation

- `python -m py_compile tunix/rl/experimental/agentic_rl_learner.py examples/deepscaler/train_deepscaler_nb.py`
- `python -m py_compile tests/rl/experimental/agentic_grpo_learner_test.py`
- `JAX_PLATFORMS=cpu python -m unittest tests.rl.experimental.agentic_grpo_learner_test.AgenticGrpoLearnerTest.test_fast_path_producer_chunking_and_queue_count tests.rl.experimental.agentic_grpo_learner_test.AgenticGrpoLearnerTest.test_fast_path_producer_memory_error_message`
- `JAX_PLATFORMS=cpu python -m unittest tests.rl.experimental.agentic_grpo_learner_test.AgenticGrpoLearnerTest.test_iterator tests.rl.experimental.agentic_grpo_learner_test.AgenticGrpoLearnerTest.test_grpo_config_validation`

### Validation results

- Python syntax checks passed.
- New fast-path unit tests passed.
- Selected existing agentic GRPO tests passed in CPU backend mode.

### Known risks / TODO

- Fast-path currently targets single-turn prompt->assistant trajectory reconstruction pattern used by DeepScaler GRPO.
- Eval path still uses orchestrator producer; only train rollout production is fast-pathed.
- Throughput gain depends on safe `--rollout-prompt-batch-size`; overly large values can still trigger HBM pressure.

---

## 2026-03-05: TPU fast-path stability fixes (event-loop + irregular output ids)

### Scope

- Debugged and fixed TPU runtime failures for DeepScaler fast-path training command:
  - `GRPO_MAX_CONCURRENCY=4 ./examples/deepscaler/run_train.sh --rollout-engine sglang_jax --rollout-tp 2 --max-steps 100 --total-generation-steps 4096 --max-prompt-length 512 --enable-rollout-fast-path --rollout-prompt-batch-size 4`
- Fixed two independent crash points observed only in end-to-end TPU runs:
  - uvloop nested event loop conflict in fast-path producer
  - irregular sampler output id shapes causing `jnp.array(...)` failure

### Changed files

1. `tunix/rl/experimental/agentic_rl_learner.py`
2. `tunix/generate/sglang_jax_sampler.py`
3. `tests/generate/sglang_jax_sampler_unit_test.py`
4. `develop.md`

### Key behavior changes

- In fast-path producer, moved blocking rollout call into worker thread:
  - `await asyncio.to_thread(self.rl_cluster.generate, ...)`
  - avoids `RuntimeError: Cannot run the event loop while another loop is running` from uvloop-backed sglang_jax engine.
- Added robust normalization for sglang_jax engine outputs before padding/stacking:
  - `_normalize_output_ids(...)`: flatten nested outputs, coerce int32, truncate to `max_generation_steps` when over-length.
  - `_normalize_output_text(...)`: normalize list/scalar output text to scalar string.
- Added lightweight unit tests for normalization helpers.

### Validation

- `python -m py_compile tunix/rl/experimental/agentic_rl_learner.py tunix/generate/sglang_jax_sampler.py tests/generate/sglang_jax_sampler_unit_test.py`
- `source .venv_sglang312/bin/activate && python -m unittest tests.generate.sglang_jax_sampler_unit_test`
- `JAX_PLATFORMS=cpu python -m unittest tests.rl.experimental.agentic_grpo_learner_test.AgenticGrpoLearnerTest.test_fast_path_producer_chunking_and_queue_count tests.rl.experimental.agentic_grpo_learner_test.AgenticGrpoLearnerTest.test_fast_path_producer_memory_error_message`
- TPU实跑（同上完整命令）

### Validation results

- Python syntax checks passed.
- New sampler unit tests passed.
- Existing fast-path producer CPU tests passed.
- TPU command now passes previous two crash points and enters sustained training loop:
  - observed `Actor Training` progression to `2/100` without reproducing prior exceptions.
  - generated checkpoint artifacts under `/tmp/deepscaler_ckpt_20260305_021007/actor/1`.

### Known risks / TODO

- With `--total-generation-steps 4096 --max-prompt-length 512`, per-step wall time is high (observed ~12-15 min/step on current TPU env); full `100` steps will take many hours.
- Current run confirmation is "稳定推进到多步"; full 100-step completion was still in progress at log capture time.

---

## 2026-03-05: Stop running DeepScaler TPU job and handoff run command

### Scope

- Stopped an in-flight DeepScaler TPU training job per user request.
- Prepared the exact command for user-side rerun.

### Changed files

1. `develop.md`

### Key behavior changes

- 无代码改动。
- 无脚本改动。
- 无配置改动。

### Validation

- `ps -eo pid,ppid,cmd | grep -E "examples/deepscaler/run_train.sh|examples/deepscaler/train_deepscaler_nb.py" | grep -v grep`
- `kill -TERM <pids>`
- Rechecked process list is empty.

### Known risks / TODO

- If user re-runs with `--max-steps 100` and current speed, wall time remains long (multi-hour).

---

## 2026-03-05: TPU timing run with longer generation length (no code changes)

### Scope

- Reran DeepScaler fast-path TPU training with:
  - `--max-prompt-length 512`
  - `--total-generation-steps 7680`
- Goal was runtime estimation under longer generation cap.

### Changed files

1. `develop.md`

### Key behavior changes

- 无代码改动。
- 无脚本改动。
- 无配置改动。

### Validation

- Run command:
  - `source .venv_sglang312/bin/activate && GRPO_MAX_CONCURRENCY=4 timeout 14400 ./examples/deepscaler/run_train.sh --rollout-engine sglang_jax --rollout-tp 2 --max-steps 100 --max-prompt-length 512 --total-generation-steps 7680 --enable-rollout-fast-path --rollout-prompt-batch-size 4`
- Checked runtime logs:
  - `/tmp/deepscaler_fastpath_7680_20260305_042535.log`

### Validation results

- Run reached training stage and emitted:
  - `Actor Training: ... 1/100 ... 1190.07s/step`
- Indicates step-1 wall time around 19m50s with this configuration.

### Known risks / TODO

- This run was launched with `timeout 14400` (4h); for full completion remove timeout.
- First-step timing includes warmup effects; steady-state may differ.

---

## 2026-03-05: Stop long run and hyperparameter speed analysis only (no code changes)

### Scope

- Stopped the in-flight DeepScaler TPU run with `--total-generation-steps 7680`.
- Performed static/runtime-log based analysis only; no new training launched.
- Focused on speed-vs-HBM tradeoff knobs while keeping most parameters unchanged.

### Changed files

1. `develop.md`

### Key behavior changes

- 无代码改动。
- 无脚本改动。
- 无配置改动。

### Validation

- Confirmed process stop via:
  - `ps -ef | grep -E "examples/deepscaler/run_train.sh|examples/deepscaler/train_deepscaler_nb.py" | grep -v grep`
- Reviewed relevant arg definitions and fast-path code paths in:
  - `examples/deepscaler/train_deepscaler_nb.py`
  - `tunix/rl/experimental/agentic_rl_learner.py`
- Compared recent runtime logs for 4096 vs 7680 generation lengths.

### Validation results

- No training process remained after termination.
- Analysis-only task completed without code execution changes.

### Known risks / TODO

- Final throughput recommendation still needs controlled A/B runs to quantify exact gains for each knob in current TPU environment.

---

## 2026-03-05: TPU hyperparameter tuning runbook for 7680-step rollout (no code changes)

### Scope

- Explored rollout-speed related hyperparameters while keeping core setup unchanged:
  - fixed `--total-generation-steps 7680`
  - fixed `--max-prompt-length 512`
  - fast-path enabled
- Tuned/checked knobs:
  - `--rollout-prompt-batch-size`
  - `--rollout-sglang-jax-mem-fraction-static`
  - `--rollout-tp`

### Changed files

1. `develop.md`

### Key behavior changes

- 无代码改动。
- 无脚本改动。
- 无配置改动。

### Validation

- Baseline reference log (existing long run):
  - `/tmp/deepscaler_fastpath_7680_20260305_042535.log`
- Candidate A:
  - `--rollout-prompt-batch-size 6 --rollout-sglang-jax-mem-fraction-static 0.25 --rollout-tp 2`
  - log: `/tmp/tune7680_A_rpbs6_mem025_tp2_20260305_060801.log`
- Candidate B:
  - `--rollout-prompt-batch-size 6 --rollout-sglang-jax-mem-fraction-static 0.25 --rollout-tp 4`
  - log: `/tmp/tune7680_B_rpbs6_mem025_tp4_20260305_063824.log`
- Candidate C:
  - `--rollout-prompt-batch-size 4 --rollout-sglang-jax-mem-fraction-static 0.25 --rollout-tp 2`
  - log: `/tmp/tune7680_C_rpbs4_mem025_tp2_20260305_064352.log`

### Validation results

- Baseline 7680 run provides stable step-time references to step 6:
  - step-2..6 around `864, 764, 728, 703, 721 s/step`.
  - median around `~728 s/step`.
- Candidate A did not OOM but was slower (did not finish first step in practical profiling window).
- Candidate B failed early with shape mismatch under current implementation:
  - `ShapeMismatchError ... k_bias: (256,) vs (512,)`.
- Candidate C (mem fraction only increase) also did not show first-step speedup in practical profiling window.

### Known risks / TODO

- For strict apples-to-apples ranking, a controlled profiling harness that captures first completed-step wall time directly (without long producer tail effects) should be added.
- Current recommendation is based on observed practical throughput and failure modes in this TPU runtime.

---

## 2026-03-05: `num-generations=4` parameter exploration under fixed 7680/512 (no code changes)

### Scope

- Objective: keep `--num-generations=4` and search for runnable / fastest settings.
- Fixed core constraints during tests:
  - `--max-prompt-length 512`
  - `--total-generation-steps 7680`
  - `--rollout-engine sglang_jax`
  - fast-path enabled (`--enable-rollout-fast-path --rollout-prompt-batch-size 4`)

### Changed files

1. `develop.md`

### Key behavior changes

- 无代码改动。
- 无脚本改动。
- 无配置改动。

### Validation

- Base candidate (offload true):
  - `OFFLOAD_TO_CPU=true ... --num-generations 4 --rollout-tp 2`
  - log: `/tmp/profile_numgen4_A_offload_true_tp2_rpbs4_mem020_20260305_143844.log`
- Rollout TP reduced:
  - `OFFLOAD_TO_CPU=false ... --num-generations 4 --rollout-tp 1`
  - log: `/tmp/profile_numgen4_B_tp1_rpbs4_mem020_offloadfalse_20260305_144910.log`
- LoRA attempt:
  - `--train-with-lora --lora-rank 32 --lora-alpha 32`
  - log: `/tmp/profile_numgen4_C_lora32_tp2_rpbs4_mem020_20260305_145917.log`
- Mini/micro batch simplification:
  - `--mini-batch-size 1 --train-micro-batch-size 1`
  - log: `/tmp/profile_numgen4_D_minib1_micro1_tp2_rpbs4_mem020_20260305_150019.log`
- Mesh variants:
  - `MESH_FSDP=4 MESH_TP=1 OFFLOAD_TO_CPU=true --rollout-tp 1`
  - log: `/tmp/profile_numgen4_E_mesh4x1_offloadtrue_tp1_rpbs4_mem020_20260305_151234.log`
  - `MESH_FSDP=4 MESH_TP=1 OFFLOAD_TO_CPU=false --rollout-tp 1`
  - log: `/tmp/profile_numgen4_F_mesh4x1_offloadfalse_tp1_rpbs4_mem020_20260305_151720.log`
- KL-off attempt:
  - `--beta 0`
  - log: `/tmp/profile_numgen4_G_beta0_tp2_rpbs4_mem020_20260305_152829.log`

### Validation results

- Most candidates fail at actor compile with TPU sflag OOM,典型报错:
  - `RESOURCE_EXHAUSTED ... Ran out of memory in memory space sflag`
- `MESH_FSDP=4 MESH_TP=1 OFFLOAD_TO_CPU=true` fails earlier in rollout init with memory kind mismatch:
  - `ValueError: Memory kind mismatch ... sharding memory kind 'device' vs buffer 'pinned_host'`
- LoRA candidate fails for current sglang_jax mapping path:
  - `RuntimeError: sglang_jax mappings not available for Qwen2.`
- No tested configuration achieved a successful first actor step with `num-generations=4` under fixed `7680/512`.

### Known risks / TODO

- Current hardware/runtime + graph shape appears incompatible with `num-generations=4` at these lengths without further algorithm/code-level changes.
- If `num-generations=4` is hard requirement, next step should be code-path changes (not hyperparameter-only), e.g. reducing actor graph complexity or changing compile/runtime strategy.

---

## 2026-03-05: Check whether `rollout_prompt_batch_size=1` was tested (no code changes)

### Scope

- Verified whether any existing `num-generations=4` profiling run used `--rollout-prompt-batch-size 1`.

### Changed files

1. `develop.md`

### Key behavior changes

- 无代码改动。
- 无脚本改动。
- 无配置改动。

### Validation

- Inspected `/tmp/profile_numgen4_*` logs and searched for:
  - `rollout_prompt_batch_size=1`
  - `--rollout-prompt-batch-size 1`

### Validation results

- No run with `rollout_prompt_batch_size=1` was found in current `num-generations=4` profiling set.
- Existing runs were with `rollout_prompt_batch_size=4`.

### Known risks / TODO

- `rollout_prompt_batch_size=1` was not empirically run yet in this round.

---

## 2026-03-05: Disk capacity and checkpoint footprint check for deepscaler run (no code changes)

### Scope

- Checked whether running the `num-generations=2` command is likely to run out of disk.
- Verified checkpoint save frequency and observed checkpoint footprint on current machine.

### Changed files

1. `develop.md`

### Key behavior changes

- 无代码改动。
- 无脚本改动。
- 无配置改动。

### Validation

- Checked disk free space: `df -h /tmp`
- Inspected existing checkpoint roots and sizes:
  - `du -sh /tmp/deepscaler_ckpt_*`
  - `du -sh /tmp/deepscaler_ckpt_*/actor`
- Verified checkpoint structure:
  - `find /tmp/deepscaler_ckpt_* -maxdepth 3 -type d`
- Confirmed defaults in code:
  - `SAVE_INTERVAL_STEPS = 500`, `MAX_TO_KEEP = 4`
  - trainer forces final save at close.

### Validation results

- Current `/tmp` filesystem free space is about `3.2G` (`97%` used).
- Observed checkpoint directory sizes vary around:
  - `~2.6G` for bf16-style runs.
  - up to `~5.8G` in some runs.
- With `--max-steps 100` and default `--save-interval-steps 500`, periodic save is typically not triggered; usually only final checkpoint is written.

### Known risks / TODO

- Current free space is near the lower bound for one bf16 checkpoint and not enough for larger (~5.8G) checkpoints.
- Recommend cleanup or redirect `CHECKPOINT_DIR`/`METRICS_LOG_DIR` to a larger volume before launching long runs.

---

## 2026-03-05: Clean `/tmp/deepscaler_ckpt_*` and confirm checkpoint save policy (no code changes)

### Scope

- Removed old temporary DeepScaler checkpoint directories under `/tmp`.
- Verified current checkpoint save cadence and how to keep only final checkpoint.

### Changed files

1. `develop.md`

### Key behavior changes

- 无代码改动。
- 无脚本改动。
- 无配置改动。

### Validation

- Cleanup commands:
  - `rm -rf /tmp/deepscaler_ckpt_*`
- Count before/after:
  - before: `70`
  - after: `0`
- Disk status after cleanup:
  - `df -h /tmp` shows available space about `42G`.
- Save policy references checked:
  - `examples/deepscaler/train_deepscaler_nb.py`:
    - `SAVE_INTERVAL_STEPS = 500`
    - `CheckpointManagerOptions(save_interval_steps=args.save_interval_steps, max_to_keep=args.max_to_keep)`
  - `tunix/sft/peft_trainer.py`:
    - `_save_last_checkpoint()` always force-saves final step if not already saved.

### Validation results

- Temporary checkpoint directories were cleaned successfully.
- Current default policy is periodic save every 500 train steps plus final forced save at trainer close.
- With `--max-steps 100`, periodic save does not trigger; effectively only final save happens.

### Known risks / TODO

- If `--max-steps` exceeds `--save-interval-steps`, periodic checkpoints will be produced unless interval is increased.
- For strict "final-only" behavior in longer runs, set a very large `--save-interval-steps` and optionally `--max-to-keep 1`.

---

## 2026-03-05: Verify reward-curve artifacts for full-dataset run (no code changes)

### Scope

- Confirmed whether current run command writes artifacts for plotting reward curves.
- Verified logging backend/output format and representative metric names.

### Changed files

1. `develop.md`

### Key behavior changes

- 无代码改动。
- 无脚本改动。
- 无配置改动。

### Validation

- Inspected logger implementation in `tunix/sft/metrics_logger.py`.
- Confirmed metrics emission points in:
  - `tunix/rl/experimental/agentic_rl_learner.py`
  - `tunix/rl/experimental/agentic_grpo_learner.py`
  - `tunix/sft/peft_trainer.py`
- Checked latest metrics directory contents under `/tmp/deepscaler_tb_*`.

### Validation results

- Metrics are written as TensorBoard event files in `METRICS_LOG_DIR`.
- Reward-related scalars are logged (e.g., `rewards/sum`, `rewards/min`, `rewards/max`, and reward-fn specific keys).
- Training scalars like `loss`, `steps_per_sec`, and GRPO `kl` are also logged.

### Known risks / TODO

- String fields (`prompts`, `completions`) are filtered by jax.monitoring and will not appear as scalar curves.

---

## 2026-03-05: Snapshot current metric-recording counts for active run (no repo code changes)

### Scope

- Queried current TensorBoard event counts for the active full-dataset run.
- Reported reward-related tag counts and latest recorded step.

### Changed files

1. `develop.md`

### Key behavior changes

- 仓库代码无改动。
- 训练命令无改动。
- 仅做日志统计与运行状态查询。

### Validation

- Active process check:
  - `ps -fp <run_shell_pid>` and child python process check.
- Event file inspected:
  - `/tmp/deepscaler_tb_20260305_180308/events.out.tfevents.1772733809.t1v-n-74398e44-w-0`
- Parsed scalar counts via TensorBoard event accumulator.

### Validation results

- Scalar tags: `21`
- Total scalar points: `3508`
- Reward tags current counts:
  - `global/train/rewards/sum`: `17`
  - `global/train/rewards/min`: `17`
  - `global/train/rewards/max`: `17`
  - `global/train/rewards/math_reward`: `17`
- Actor train metrics current counts:
  - `actor/train/loss`: `16`
  - `actor/train/steps_per_sec`: `16`
- Latest recorded train step in sampled tags: `16`.

### Known risks / TODO

- Counts are a snapshot and will increase while process keeps running.

---

## 2026-03-05: Generate training-curve snapshot image from TensorBoard event (no repo code changes)

### Scope

- Parsed latest DeepScaler TensorBoard event file and rendered a snapshot plot.
- Included reward/loss/kl/throughput/completion-length curves.

### Changed files

1. `develop.md`

### Key behavior changes

- 仓库代码无改动。
- 训练命令无改动。
- 生成了日志目录下的可视化图片产物。

### Validation

- Event source:
  - `/tmp/deepscaler_tb_20260305_180308/events.out.tfevents.1772733809.t1v-n-74398e44-w-0`
- Output image:
  - `/tmp/deepscaler_tb_20260305_180308/training_curves_snapshot.png`
- Tags plotted:
  - `global/train/rewards/sum`
  - `global/train/rewards/math_reward`
  - `actor/train/loss`
  - `actor/train/kl`
  - `actor/train/steps_per_sec`
  - `global/train/completions/mean_length`

### Validation results

- Snapshot generated successfully.
- At snapshot time, latest visible step for these tags was around step `19`.

### Known risks / TODO

- Training still running, so this snapshot is point-in-time and will become stale.

---

## 2026-03-05: Analyze early training curves from active full-dataset run (no code changes)

### Scope

- Performed numerical trend analysis on early-stage training curves from the active run.
- Focused on rewards, actor loss, KL, throughput, and completion length.

### Changed files

1. `develop.md`

### Key behavior changes

- 无代码改动。
- 无脚本改动。
- 无配置改动。

### Validation

- Source event:
  - `/tmp/deepscaler_tb_20260305_180308/events.out.tfevents.1772733809.t1v-n-74398e44-w-0`
- Analyzed tags:
  - `global/train/rewards/sum`
  - `global/train/rewards/math_reward`
  - `actor/train/loss`
  - `actor/train/kl`
  - `actor/train/steps_per_sec`
  - `global/train/completions/mean_length`

### Validation results

- Snapshot window around steps `0~19` (actor tags `1~19`).
- Reward is relatively stable in early stage:
  - mean ~`0.659`, median ~`0.680`, latest ~`0.734`.
- Actor loss / KL show heavy-tailed spikes (notably around steps `7` and `14`), but last-5 values are much smaller:
  - loss last-5 mean ~`0.406`, median ~`0.179`
  - kl last-5 mean ~`475.7`, median ~`182.5`
- Throughput (`actor/train/steps_per_sec`) remains noisy in early phase; median ~`0.169`.
- Completion mean length is high and rising in this window (latest ~`3703`).

### Known risks / TODO

- Current analysis is early-stage only; training is still running and distributions may shift later.
- Large KL/loss spikes should continue to be monitored to rule out mid-run instability.

---

## 2026-03-05: Clarify effect of `GRPO_MAX_CONCURRENCY=4` under fast-path (no code changes)

### Scope

- Answered whether `GRPO_MAX_CONCURRENCY=4` matters for current run command.

### Changed files

1. `develop.md`

### Key behavior changes

- 无代码改动。
- 无脚本改动。
- 无配置改动。

### Validation

- Based on current command using:
  - `--rollout-engine sglang_jax`
  - `--enable-rollout-fast-path`
- Confirmed code path notes that `--grpo-max-concurrency` is ignored in fast-path.

### Validation results

- For this command, `GRPO_MAX_CONCURRENCY` has no runtime effect.
- It becomes relevant again only if fast-path is disabled.

### Known risks / TODO

- None.

---

## 2026-03-05: Bake full-dataset defaults into `examples/deepscaler/run_train.sh`

### Scope

- Updated `examples/deepscaler/run_train.sh` to include the user's current stable full-dataset defaults.
- Kept `"$@"` passthrough so ad-hoc overrides still work.

### Changed files

1. `examples/deepscaler/run_train.sh`
2. `develop.md`

### Key behavior changes

- Added default knobs in script for:
  - `rollout_engine=sglang_jax`
  - `rollout_tp=2`
  - `enable_rollout_fast_path`
  - `rollout_prompt_batch_size=4`
  - `num_generations=2`
  - `max_prompt_length=512`
  - `total_generation_steps=7680`
  - `batch_size=32`
  - `num_batches=1260`
  - `num_epochs=1`
  - `train_fraction=1.0`
  - `max_steps=num_batches`
  - `save_interval_steps=1000000000`
  - `max_to_keep=1`
- Existing env/path and dtype knobs remain intact.

### Validation

- `bash -n examples/deepscaler/run_train.sh`

### Validation results

- Shell syntax check passed.

### Known risks / TODO

- `num_batches=1260` is tied to current dataset size and `batch_size=32`; if dataset size changes, full-epoch equivalence may drift.

---

## 2026-03-06: Confirm active training model path (no code changes)

### Scope

- Verified which model is currently used by the active DeepScaler training process.

### Changed files

1. `develop.md`

### Key behavior changes

- 无代码改动。
- 无脚本改动。
- 无配置改动。

### Validation

- Inspected active process args (`ps`/`pgrep`) for `examples/deepscaler/train_deepscaler_nb.py`.
- Cross-checked script default `DEFAULT_MODEL_PATH` in `examples/deepscaler/run_train.sh`.

### Validation results

- Active process is running with:
  - `--model-path /home/lhf_hongfu_gmail_com/.cache/huggingface/hub/models--agentica-org--DeepScaleR-1.5B-Preview/snapshots/e3f524ce413a296b4d388e7560dd5c82c1c56725`
- This corresponds to `agentica-org/DeepScaleR-1.5B-Preview` snapshot.

### Known risks / TODO

- None.

---

## 2026-03-06: Clarify which training models can be swapped in current DeepScaler script (no code changes)

### Scope

- Clarified model replacement scope for current `examples/deepscaler/train_deepscaler_nb.py` without changing code.

### Changed files

1. `develop.md`

### Key behavior changes

- 无代码改动。
- 无脚本改动。
- 无配置改动。

### Validation

- Checked model config binding in training entrypoint:
  - `ModelConfig.deepseek_r1_distill_qwen_1p5b()`
- Checked Qwen2 loader and mapping path constraints.

### Validation results

- Current script is effectively bound to Qwen2.5-like 1.5B tensor shapes for actor/reference loading.
- Cross-size swaps (0.5B/3B/7B, Llama family) are not directly compatible without code changes.

### Known risks / TODO

- For broader model support, expose model-config selection as CLI and align rollout mapping/runtime checks.

---

## 2026-03-06: Switch DeepScaler default training model to DeepSeek-R1-Distill-Qwen-1.5B

### Scope

- Updated DeepScaler training wrapper default model path to local cached snapshot of `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.

### Changed files

1. `examples/deepscaler/run_train.sh`
2. `develop.md`

### Key behavior changes

- `DEFAULT_MODEL_PATH` now points to:
  - `/home/lhf_hongfu_gmail_com/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`
- All other script defaults remain unchanged.

### Validation

- `bash -n examples/deepscaler/run_train.sh`
- `sed -n '1,8p' examples/deepscaler/run_train.sh`
- `test -d /home/lhf_hongfu_gmail_com/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`

### Validation results

- Shell syntax check passed.
- New default model path is present in script.
- Target local model snapshot directory exists.

### Known risks / TODO

- Existing already-running process keeps using its original launched model path; the new default applies to subsequent runs.

---

## 2026-03-06: Explain dataset pass / ETA / checkpoint count for current `run_train.sh` defaults (no code changes)

### Scope

- Interpreted whether current `./examples/deepscaler/run_train.sh` defaults correspond to one dataset pass.
- Estimated runtime from live event-file step cadence.
- Confirmed expected checkpoint save count.

### Changed files

1. `develop.md`

### Key behavior changes

- 无代码改动。
- 无脚本改动。
- 无配置改动。

### Validation

- Active run process and args verified.
- Parsed latest TensorBoard event scalars from:
  - `/tmp/deepscaler_tb_20260305_180308/events.out.tfevents.1772733809.t1v-n-74398e44-w-0`
- Measured global-step cadence via `global/train/rewards/sum` wall-time deltas.

### Validation results

- Current progress snapshot: step `43/1260`.
- Measured pace:
  - all-so-far average ~`707.4 s/step`
  - last-20 average ~`744.6 s/step`
- Estimated full run (1260 steps) at current last-20 pace:
  - total ~`938,202 s` (~`260.6 h`, ~`10.9 days`)
  - remaining from step 43 ~`906,184 s` (~`251.7 h`, ~`10.5 days`)
- Checkpoint policy with current defaults (`save_interval_steps=1e9`, `max_steps=1260`): periodic save won't trigger; final forced save at close yields one final checkpoint on successful completion.

### Known risks / TODO

- ETA can drift with runtime load and sequence-length mix.
- If process is killed ungracefully, final checkpoint may be missing.

---

## 2026-03-06: Mid-run checkpoint request guidance (no code changes)

### Scope

- Clarified how to keep one halfway checkpoint plus final checkpoint for current full-epoch setup.

### Changed files

1. `develop.md`

### Key behavior changes

- 无代码改动。
- 无脚本改动。
- 无配置改动。

### Validation

- Based on current defaults: `max_steps=1260`.
- Recommended interval target: `630` (halfway).

### Validation results

- Use `--save-interval-steps 630 --max-to-keep 2` to keep step-630 and final checkpoint.

### Known risks / TODO

- Running process cannot hot-update checkpoint interval; needs restart to apply.

---

## 2026-03-06: Update default checkpoint policy to halfway + final in `run_train.sh`

### Scope

- Changed DeepScaler wrapper defaults to save one mid-run checkpoint and keep final checkpoint.

### Changed files

1. `examples/deepscaler/run_train.sh`
2. `develop.md`

### Key behavior changes

- Updated defaults:
  - `SAVE_INTERVAL_STEPS` from `1000000000` to `630`
  - `MAX_TO_KEEP` from `1` to `2`
- This matches the current full-run default `max_steps=1260` and yields save at step `630` plus final save.

### Validation

- `bash -n examples/deepscaler/run_train.sh`
- `rg -n "SAVE_INTERVAL_STEPS|MAX_TO_KEEP" examples/deepscaler/run_train.sh`

### Validation results

- Shell syntax check passed.
- Updated defaults are present and wired into CLI args.

### Known risks / TODO

- Existing already-running process keeps old launch-time args; restart is required for new defaults to take effect.

---

## 2026-03-06: Update `run_train.sh` defaults for grad norm / sampling / weight decay

### Scope

- Applied requested default hyperparameter changes in DeepScaler wrapper script.

### Changed files

1. `examples/deepscaler/run_train.sh`
2. `develop.md`

### Key behavior changes

- Updated defaults:
  - `MAX_GRAD_NORM`: `0.1 -> 1.0`
  - `TOP_P`: default now `1.0`
  - `TOP_K`: default now `-1`
  - `WEIGHT_DECAY`: `0.1 -> 0.01`
- Wired these defaults into CLI args passed to `train_deepscaler_nb.py`:
  - `--max-grad-norm`
  - `--top-p`
  - `--top-k`
  - `--weight-decay`

### Validation

- `bash -n examples/deepscaler/run_train.sh`
- `rg -n "WEIGHT_DECAY|MAX_GRAD_NORM|TOP_P|TOP_K|--top-p|--top-k|--weight-decay|--max-grad-norm" examples/deepscaler/run_train.sh`

### Validation results

- Shell syntax check passed.
- All requested defaults and arg pass-throughs are present.

### Known risks / TODO

- Existing already-running process uses launch-time args; restart needed for new defaults to take effect.

---

## 2026-03-06: Update `run_train.sh` defaults for batch-size 128 one-command run

### Scope

- Applied requested batch-size-128 aligned defaults so `./examples/deepscaler/run_train.sh` can be used directly.
- Added explicit mini-batch and train-micro-batch args passthrough.

### Changed files

1. `examples/deepscaler/run_train.sh`
2. `develop.md`

### Key behavior changes

- Updated defaults:
  - `BATCH_SIZE=128`
  - `MINI_BATCH_SIZE=128`
  - `TRAIN_MICRO_BATCH_SIZE=1`
  - `NUM_BATCHES=315`
  - `MAX_STEPS` remains defaulting to `NUM_BATCHES`
  - `SAVE_INTERVAL_STEPS=158` (halfway for 315-step run)
  - `MAX_TO_KEEP=2`
- Added CLI args emission:
  - `--mini-batch-size "$MINI_BATCH_SIZE"`
  - `--train-micro-batch-size "$TRAIN_MICRO_BATCH_SIZE"`

### Validation

- `bash -n examples/deepscaler/run_train.sh`
- `rg -n "BATCH_SIZE|MINI_BATCH_SIZE|TRAIN_MICRO_BATCH_SIZE|NUM_BATCHES|MAX_STEPS|SAVE_INTERVAL_STEPS|--mini-batch-size|--train-micro-batch-size" examples/deepscaler/run_train.sh`

### Validation results

- Shell syntax check passed.
- New defaults and arg pass-through lines are present.

### Known risks / TODO

- Existing already-running process uses old launch-time args; restart is required for new defaults to take effect.

---

## 2026-03-06: Inspect whether old running process generated many intermediate files (no code changes)

### Scope

- Audited file artifacts produced by the earlier long-running process tied to:
  - `checkpoint_dir=/tmp/deepscaler_ckpt_20260305_180308`
  - `metrics_log_dir=/tmp/deepscaler_tb_20260305_180308`

### Changed files

1. `develop.md`

### Key behavior changes

- 无代码改动。
- 无脚本改动。
- 无配置改动。

### Validation

- Checked process status and child process args.
- Checked directory sizes and file counts:
  - `du -sh ...`
  - `find ... -type f | wc -l`
- Inspected checkpoint tree and file timestamps.
- Scanned `/tmp` top-size directories.

### Validation results

- Checkpoint artifacts are concentrated in a single checkpoint folder:
  - `/tmp/deepscaler_ckpt_20260305_180308` size ~`2.6G`
  - only `14` files under this checkpoint tree
  - only one step directory present: `actor/1/...`
- Metrics artifacts are small:
  - `/tmp/deepscaler_tb_20260305_180308` size ~`532K`
  - `2` files (event + snapshot png)
- No evidence of large accumulation of many intermediate temp files from this run.

### Known risks / TODO

- Current run is still active; if allowed to finish with updated save policy in future runs, additional checkpoints may be added by configured interval.

---

## 2026-03-06: Final pre-push verification for DeepScaler fast-path and wrapper updates

### Scope

- Prepared the accumulated DeepScaler local changes for push on `my-changes`.
- Verified the fast-path / sampler / wrapper updates against syntax and targeted unit tests.

### Changed files

1. `examples/deepscaler/run_train.sh`
2. `examples/deepscaler/train_deepscaler_nb.py`
3. `tests/generate/sglang_jax_sampler_unit_test.py`
4. `tests/rl/experimental/agentic_grpo_learner_test.py`
5. `tunix/generate/sglang_jax_sampler.py`
6. `tunix/rl/experimental/agentic_rl_learner.py`
7. `tunix/rl/utils.py`
8. `develop.md`

### Key behavior changes

- Consolidates the local DeepScaler work since the last push:
  - empty-pytree offload guard
  - sglang_jax fast-path rollout producer
  - sampler normalization and engine locking fixes
  - `run_train.sh` default updates for checkpointing and requested hyperparameters
  - targeted unit coverage for fast-path producer and sampler normalization

### Validation

- `python -m py_compile examples/deepscaler/train_deepscaler_nb.py tunix/rl/experimental/agentic_rl_learner.py tunix/generate/sglang_jax_sampler.py tunix/rl/utils.py tests/rl/experimental/agentic_grpo_learner_test.py tests/generate/sglang_jax_sampler_unit_test.py`
- `bash -n examples/deepscaler/run_train.sh`
- `source .venv_sglang312/bin/activate && JAX_PLATFORMS=cpu python -m unittest tests.rl.experimental.agentic_grpo_learner_test.AgenticGrpoLearnerTest.test_fast_path_producer_chunking_and_queue_count tests.rl.experimental.agentic_grpo_learner_test.AgenticGrpoLearnerTest.test_fast_path_producer_memory_error_message tests.generate.sglang_jax_sampler_unit_test`

### Validation results

- Python syntax checks passed.
- Shell syntax check passed.
- Targeted CPU unit tests passed (`Ran 4 tests ... OK`).

### Known risks / TODO

- Full TPU end-to-end validation still depends on runtime availability and absence of external TPU lock/process interference.
