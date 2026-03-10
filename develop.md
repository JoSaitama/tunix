# Development Log

This file tracks engineering changes made in this repository.

## Update Policy

- Every code/doc/script change must append or update an entry in this file.
- Each entry should include: date, scope, changed files, validation, and known risks.
- If a task has no code changes, note that explicitly.

---

## 2026-03-07: DeepScaler live eval progress inspection

### Scope

- 无代码改动。
- 检查用户当前正在运行的 `examples/deepscaler/run_eval_pass1_avg16.sh` 日志，读取已完成轮次的准确率与当前进度。

### Changed files

1. `develop.md`

### Validation

- `ps -ef | rg 'run_eval_pass1_avg16.sh|examples/deepscaler/math_eval_nb.py'`
- `find /tmp -maxdepth 1 -type d -name 'deepscaler_pass1_avg16_*' -printf '%T@ %p\n' | sort -n | tail -1`
- `rg -n 'Progress:|Correct:|Accuracy:|===== END run=' /tmp/deepscaler_pass1_avg16_20260307_151314/run_*.log`
- 确认：
  - 用户当前活跃目录为 `/tmp/deepscaler_pass1_avg16_20260307_151314`
  - `run_1.log` 结果为 `Correct: 8/30`, `Accuracy: 26.67%`
  - `run_2.log` 结果为 `Correct: 8/30`, `Accuracy: 26.67%`
  - `run_3.log` 已创建，但尚未写出最终 `Correct/Accuracy`
  - 当前 Python 评测进程参数显示 `--seed 2`，与第 3 轮一致

### Known risks / TODO

- 在 `run_3.log` 尚未写出最终 summary 之前，只能确认前两轮结果与当前进程状态，不能把后续均值当成已定结果。

---

## 2026-03-07: DeepScaler sglang-jax fp32 interpretation clarification

### Scope

- 无代码改动。
- 澄清在当前 `eval + sglang-jax` 路径下，“模型权重 fp32 + engine dtype float32”与 `auto` 的关系，以及为什么这里不应混用 `rollout` 概念。

### Changed files

1. `develop.md`

### Validation

- `nl -ba examples/deepscaler/math_eval_nb.py | sed -n '314,335p'`
- `nl -ba tunix/generate/sglang_jax_sampler.py | sed -n '68,123p'`
- `nl -ba tunix/generate/sglang_jax_sampler.py | sed -n '140,143p'`
- `nl -ba tunix/generate/sglang_jax_sampler.py | sed -n '91,113p'`
- 确认：
  - 当前讨论对象是 eval 路径，不是 training rollout 路径。
  - 当前 eval 的 `sglang-jax` 生成实际由 `Engine(**self.args)` 驱动，`args["model_path"] = config.model_version`。
  - Python 侧加载的 `self.model` 只用于构造 mapping；当前 eval 代码没有调用 `update_params()` / `load_checkpoint()` 把这份权重同步到 engine。
  - 因此，若保留 `sglang-jax`，真正决定生成精度的是 engine 侧 `dtype` / `kv_cache_dtype`；`auto` 表示交给 backend 自行选择，并不等于 fp32。

### Known risks / TODO

- 我此前把“模型权重 fp32 + engine dtype float32”说得过于简化；在当前 eval 实现里，更关键的是 engine 侧 dtype，因为 Python 侧权重默认并不直接参与 `sglang-jax` 生成。
- 若不改当前 eval 代码去同步权重到 engine，单独把 Python 侧 `self.model` cast 到 fp32，不足以保证实际生成路径变成 fp32。

---

## 2026-03-07: DeepScaler eval fp32 path feasibility clarification

### Scope

- 无代码改动。
- 说明当前评测链路若要“保证 fp32”，是否可以只通过修改 `examples/deepscaler/run_eval_pass1_avg16.sh` 实现。

### Changed files

1. `develop.md`

### Validation

- `nl -ba examples/deepscaler/math_eval_nb.py | sed -n '292,340p'`
- `nl -ba examples/deepscaler/math_eval_nb.py | sed -n '735,810p'`
- `nl -ba tunix/models/safetensors_loader.py | sed -n '216,223p'`
- `nl -ba tunix/generate/sglang_jax_sampler.py | sed -n '37,55p'`
- `nl -ba tunix/generate/sglang_jax_sampler.py | sed -n '120,143p'`
- `nl -ba examples/deepscaler/train_deepscaler_nb.py | sed -n '269,292p'`
- `nl -ba examples/deepscaler/train_deepscaler_nb.py | sed -n '638,656p'`
- 确认：
  - 当前 eval CLI 没有 `--model-dtype`、`--sglang-jax-dtype` 或 `--sglang-jax-kv-cache-dtype` 参数。
  - `run_eval_pass1_avg16.sh` 只能传递下游已支持的参数；单独在该脚本写“相关参数”如果下游不识别，不会生效。
  - 若要把模型权重强制转成 fp32，需要在 `create_model_from_safe_tensors(..., dtype=...)` 这层显式传入 `jnp.float32`。
  - 若要把 `sglang-jax` engine dtype 显式设为 fp32，需要在 eval 侧增加类似训练侧的 dtype 参数，并传给 `SglangJaxConfig(dtype=\"float32\")`。
  - 训练侧已有规范表明：`sglang-jax` model dtype 可设为 `float32` / `fp32`，但 `kv_cache_dtype` 选项不包含 fp32，仅有 `auto` / `bf16` / `fp8_*`。

### Known risks / TODO

- 因为 `sglang-jax` 的 `kv_cache_dtype` 当前不支持 fp32，且 backend 仍可能有内部混合精度实现，所以在 `sglang-jax` 路径下很难宣称“严格全链路 fp32”。
- 若目标是尽量严格的 fp32 评测，优先级更高的方案通常是：模型权重显式 cast 到 fp32，并使用 `vanilla` sampler，而不是 `sglang-jax`。

---

## 2026-03-07: DeepScaler eval dtype clarification

### Scope

- 无代码改动。
- 核对当前 `examples/deepscaler/run_eval_pass1_avg16.sh` 通过 `sglang-jax` 路径运行时，模型权重和 sampler backend 的 dtype 实际来源。

### Changed files

1. `develop.md`

### Validation

- `nl -ba examples/deepscaler/math_eval_nb.py | sed -n '293,336p'`
- `nl -ba tunix/models/qwen2/params.py | sed -n '84,99p'`
- `nl -ba tunix/models/safetensors_loader.py | sed -n '216,223p'`
- `nl -ba tunix/generate/sglang_jax_sampler.py | sed -n '37,54p'`
- `nl -ba tunix/generate/sglang_jax_sampler.py | sed -n '120,143p'`
- 解析本地 safetensors header，确认模型快照 `/home/lhf_hongfu_gmail_com/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/.../model.safetensors` 中 `339` 个张量全部标记为 `BF16`。
- 确认：
  - `math_eval_nb.py` 调用 `create_model_from_safe_tensors(..., dtype=None)`。
  - `safetensors_loader.py` 仅在 `dtype is not None` 时才会强制 cast。
  - `sglang-jax` 配置里的 `dtype` 和 `kv_cache_dtype` 当前都是默认 `"auto"`。

### Known risks / TODO

- 从当前代码能明确确认“权重没有被改成 fp32，且本地 safetensors 原始 dtype 是 BF16”；`sglang-jax` 的 `"auto"` 在具体底层 kernel 上仍可能对少量内部计算采用更高精度。
- 因此结论应表述为“当前这条命令以 BF16 权重 / BF16 导向配置运行，不是纯 fp32 路径；但某些内部算子可能局部用 float32 计算以保证数值稳定性”。

---

## 2026-03-07: User-run command handoff for DeepScaler eval

### Scope

- 无代码改动。
- 记录交接给用户自行运行 `examples/deepscaler/run_eval_pass1_avg16.sh` 的命令，包括当前 `sglang-jax` eval 路径所需的临时运行时 shim。

### Changed files

1. `develop.md`

### Validation

- 复核当前可运行命令依赖：
  - `.venv_sglang312`
  - `/tmp/tunix_eval_shim/sitecustomize.py`
- 确认当前仓库源码下，若不注入该 shim，`sglang-jax` eval 路径会因缺失 `tunix.google.stubs.sglang_jax_sampler_stub` 模块别名而失败。

### Known risks / TODO

- `/tmp/tunix_eval_shim/sitecustomize.py` 是本地临时运行时文件，不属于仓库内容；若被删除，需要重新创建后才能直接复现当前命令。
- 完整 `NUM_RUNS=16` 会耗时较长。

---

## 2026-03-07: DeepScaler pass1_avg16 runtime check and result analysis

### Scope

- 无仓库代码改动。
- 实际运行 `examples/deepscaler/run_eval_pass1_avg16.sh` 的当前评测链路，记录当前非 smoke 全量 AIME 结果，并与官方公开分数口径做对比分析。

### Changed files

1. `develop.md`

### Validation

- 运行环境检查：
  - `source .venv_sglang312/bin/activate && python -c "import sgl_jax, jax; print('sgl_jax_ok'); print(jax.devices())"`
- 发现当前仓库源码的 `sglang-jax` eval 路径缺少 `tunix.google.stubs.sglang_jax_sampler_stub` 模块别名，直接运行失败：
  - `ModuleNotFoundError: No module named 'tunix.google'`
- 为避免改动仓库代码，使用 `/tmp/tunix_eval_shim/sitecustomize.py` 注入临时运行时 alias，仅用于本次执行。
- smoke 验证：
  - `export PYTHONPATH=/tmp/tunix_eval_shim${PYTHONPATH:+:$PYTHONPATH}; source .venv_sglang312/bin/activate && NUM_RUNS=1 SMOKE_TEST=1 ./examples/deepscaler/run_eval_pass1_avg16.sh`
- 全量非 smoke 单轮执行：
  - `export PYTHONPATH=/tmp/tunix_eval_shim${PYTHONPATH:+:$PYTHONPATH}; source .venv_sglang312/bin/activate && LOG_DIR=/tmp/deepscaler_pass1_avg16_current_full NUM_RUNS=1 ./examples/deepscaler/run_eval_pass1_avg16.sh`
- 本次实际结果：
  - `Correct: 8/30`
  - `Accuracy: 26.67%`
  - `Sampler: sglang-jax`
  - `Seeds: 0..0`
  - 日志目录：`/tmp/deepscaler_pass1_avg16_current_full`

### Known risks / TODO

- 本次拿到的是当前脚本口径下的单轮全量结果；没有完整跑完默认 `NUM_RUNS=16`，因此不能把 `26.67%` 直接当成严格的 16-run average。
- 运行依赖 `/tmp` 下的临时 import shim；它不修改仓库代码，但说明当前仓库的 `sglang-jax` eval 路径仍存在运行时模块别名问题。
- 官方公开分数是多次 sample 平均口径；单轮结果本身会有显著方差。

---

## 2026-03-07: DeepScaler eval seed plumbing for pass1_avg16

### Scope

- 为 DeepScaler 评测入口增加显式 seed 参数。
- 让 `examples/deepscaler/run_eval_pass1_avg16.sh` 在 16 轮运行中明确使用不同 seed，而不是重复使用同一个默认 seed。

### Changed files

1. `examples/deepscaler/math_eval_nb.py`
2. `examples/deepscaler/run_eval.sh`
3. `examples/deepscaler/run_eval_pass1_avg16.sh`
4. `develop.md`

### Validation

- `python -m py_compile examples/deepscaler/math_eval_nb.py`
- `bash -n examples/deepscaler/run_eval.sh`
- `bash -n examples/deepscaler/run_eval_pass1_avg16.sh`
- `sed -n '456,520p' examples/deepscaler/math_eval_nb.py`
- `sed -n '731,805p' examples/deepscaler/math_eval_nb.py`
- `sed -n '1,80p' examples/deepscaler/run_eval.sh`
- `sed -n '1,120p' examples/deepscaler/run_eval_pass1_avg16.sh`
- 确认：
  - `math_eval_nb.py` 新增 `--seed` 参数，默认值为 `0`。
  - 每次采样使用 `sample_seed = seed + pass_idx`，因此默认行为与原来保持一致，而调用方也可以显式构造不同 seed。
  - `run_eval.sh` 新增 `EVAL_SEED` 环境变量透传到 `--seed`。
  - `run_eval_pass1_avg16.sh` 现在为第 `N` 轮设置 `EVAL_SEED=N-1`，即默认使用 seed `0..15`。

### Known risks / TODO

- `sglang-jax` 当前仍配置为 `enable_deterministic_sampling=False`，不同 seed 会提升“独立样本”语义，但不保证跨运行严格可复现。
- 这次只修正了 eval seed 语义，没有去对齐官方完整评测协议中的所有其他细节（例如长度预算、模型版本、backend 差异）。

---

## 2026-03-09: DeepScaler cross-batch reward feasibility review

### Scope

- 无代码改动。
- 审阅 `examples/deepscaler/` 当前训练链路，确认 `num_generations=2` 时 reward/advantage 的计算位置与形态，并评估是否适合增加 cross-batch reward / advantage 分支。

### Changed files

1. `develop.md`

### Validation

- `nl -ba examples/deepscaler/train_deepscaler_nb.py | sed -n '980,1085p'`
- `nl -ba examples/deepscaler/train_deepscaler_nb.py | sed -n '60,140p'`
- `nl -ba examples/deepscaler/train_deepscaler_nb.py | sed -n '540,770p'`
- `nl -ba examples/deepscaler/run_train.sh | sed -n '1,220p'`
- `nl -ba tunix/rl/experimental/agentic_grpo_learner.py | sed -n '1,620p'`
- `nl -ba tunix/rl/experimental/agentic_rl_learner.py | sed -n '1,1100p'`
- `nl -ba tunix/utils/math_rewards.py | sed -n '1,260p'`
- `nl -ba tunix/rl/algorithm_config.py | sed -n '1,180p'`
- `nl -ba tunix/rl/function_registry.py | sed -n '1,260p'`
- 确认：
  - DeepScaler 训练入口当前使用的是 `tunix/rl/experimental/agentic_grpo_learner.py`，不是主线 `tunix/rl/grpo/grpo_learner.py`。
  - `math_reward` 是按 completion 独立计算的 0/1 标量奖励。
  - 当前 `agentic_grpo` advantage 仍是严格按单个 prompt 的 `num_generations` 分组做均值/标准差归一化，没有跨 prompt 统计。
  - `examples/deepscaler/run_train.sh` 默认开启 `--enable-rollout-fast-path`，因此在 fast-path 下天然存在一个 `rollout_prompt_batch_size` 级别的跨 prompt 窗口，可作为未来 cross-batch 分支的最小接入点。

### Known risks / TODO

- 若直接把不同 prompt 的 reward 混到同一个 baseline/std 中，算法语义会从“group-relative”偏移到“batch-relative”；对难度分布不均的数据会引入偏差。
- 更稳妥的做法是优先考虑“per-prompt center + cross-batch scale”这类分支，而不是直接用全局 batch mean 替代 group mean。
- 当前未做实现与跑数，结论仅覆盖代码可接入性与训练信号形态，不代表该方案一定优于调 `beta`、长度约束或 reward 设计。

---

## 2026-03-09: DeepScaler cross-batch advantage explanation

### Scope

- 无代码改动。
- 进一步解释 DeepScaler 在 `num_generations=2`、二值数学 reward 下，保守版与激进版 cross-batch advantage 的区别、收益与风险。

### Changed files

1. `develop.md`

### Validation

- 复用上一条审阅结论，无新增代码执行。
- 关键依据：
  - `tunix/rl/experimental/agentic_grpo_learner.py`
  - `tunix/rl/experimental/agentic_rl_learner.py`
  - `tunix/utils/math_rewards.py`
- 确认：
  - 当前 raw reward 仍是逐 completion 的 `0/1`。
  - 讨论对象是 advantage 的 baseline / scale 设计，而不是修改 reward 定义本身。

### Known risks / TODO

- 保守版主要解决的是 `G=2` 下标准差估计太跳，不解决 `[1,1]` / `[0,0]` 两类组 advantage 为零的问题。
- 激进版能让更多样本带梯度，但更容易把 prompt 难度差异折叠进 advantage，偏离原始 GRPO 的相对比较语义。

---

## 2026-03-09: DeepScaler cross-batch memory impact clarification

### Scope

- 无代码改动。
- 结合当前 DeepScaler fast-path 训练实现，说明为什么把 `num_generations` 从 `2` 提到 `4` 容易 OOM，而 cross-batch advantage 方案在正确实现下通常不会触发同级别显存增长。

### Changed files

1. `develop.md`

### Validation

- `nl -ba tunix/rl/experimental/agentic_rl_learner.py | sed -n '224,324p'`
- `nl -ba tunix/rl/experimental/agentic_grpo_learner.py | sed -n '282,430p'`
- `nl -ba tunix/rl/experimental/agentic_rl_learner.py | sed -n '820,910p'`
- 确认：
  - fast-path rollout 一次会生成 `rollout_prompt_batch_size * num_generations` 条 completion，`num_generations` 直接放大 rollout 侧工作集。
  - 训练消费阶段会按 `train_micro_batch_size * num_generations` 收集样本，再拼成一个 `merged_train_micro_batch` 送入 actor update，`num_generations` 直接放大训练侧 token batch。
  - 若 cross-batch 只额外缓存 reward / completion-level 统计量，而不把更多 prompt 的 token 序列并进单次 actor/ref 前向，则显存主路径不变。

### Known risks / TODO

- 如果实现时为了做 cross-batch 统计而把多个 prompt group 的 `prompt_ids` / `completion_ids` / `logps` 合并后再统一前向，仍然可能 OOM。
- 因此实现上应优先使用“小窗口标量统计 + 原始 group 粒度发射 train example”的方式，避免把 cross-batch 设计误写成更大的 token batch。

---

## 2026-03-09: DeepScaler `num_generations=4` no-OOM feasibility review

### Scope

- 无代码改动。
- 结合当前 DeepScaler fast-path 训练实现，分析把 `num_generations` 从 `2` 提到 `4` 时，哪些参数能真实降低 actor / rollout 显存占用，哪些参数不会改变 OOM 主因。

### Changed files

1. `develop.md`

### Validation

- `rg -n "def update_actor|gradient_accumulation_steps|train_micro_batch_size|mini_batch_size|rollout_micro_batch_size|compute_logps_micro_batch_size" tunix -g '!**/__pycache__/**'`
- `nl -ba tunix/rl/experimental/agentic_rl_learner.py | sed -n '224,324p'`
- `nl -ba tunix/rl/experimental/agentic_rl_learner.py | sed -n '820,910p'`
- `nl -ba tunix/rl/experimental/agentic_grpo_learner.py | sed -n '282,430p'`
- `nl -ba tunix/rl/rl_cluster.py | sed -n '84,150p'`
- `nl -ba tunix/sft/peft_trainer.py | sed -n '180,230p'`
- `nl -ba tunix/sft/peft_trainer.py | sed -n '560,705p'`
- 确认：
  - RL 训练里的梯度累积由 `mini_batch_size // train_micro_batch_size` 自动推导。
  - fast-path rollout 的一次生成规模是 `rollout_prompt_batch_size * num_generations`。
  - actor 训练侧一次送入 train step 的样本数是 `train_micro_batch_size * num_generations`。
  - 因此要让 `num_generations=4` 不 OOM，优先要缩的是这两个乘积，而不是只改逻辑层统计。

### Known risks / TODO

- 当前 `examples/deepscaler/train_deepscaler_nb.py` 在 agentic fast-path 下把 `_rollout_micro_batch_size` 和 `_compute_logps_micro_batch_size` 固定成 `1`，因此额外调这两个值对当前路径帮助有限。
- 如果 `num_generations=4` 后 completion 长度也变长，即使把 `train_micro_batch_size` / `rollout_prompt_batch_size` 降低，仍可能因长序列导致 compile 或 runtime OOM。

---

## 2026-03-09: DeepScaler actor-side chunking explanation

### Scope

- 无代码改动。
- 进一步解释在保持 `num_generations=4` 语义不变的前提下，为什么“actor-side generation chunking / sequence microbatching”能比单纯调参更稳地降低 OOM 风险。

### Changed files

1. `develop.md`

### Validation

- 复用前一条关于 DeepScaler fast-path 训练路径的代码审阅结果，无新增代码执行。
- 关键依据：
  - `tunix/rl/experimental/agentic_rl_learner.py`
  - `tunix/rl/experimental/agentic_grpo_learner.py`
  - `tunix/sft/peft_trainer.py`
- 确认：
  - 当前实现先对一个 prompt 的 `num_generations` 条 completion 算完 reward/advantage，再把这些序列直接拼成一个 `merged_train_micro_batch` 做单次 actor update。
  - 更稳的降内存思路不是减少逻辑组大小，而是把同一组里的多条 completion 分几次前向/反向，再通过现有梯度累积维持等价更新。

### Known risks / TODO

- 这种改法会增加 wall-clock time，因为同一逻辑更新被拆成更多次前向/反向。
- 若实现时没有正确保留“4 条 completion 共用同一组 reward/advantage 统计”，就会把 `num_generations=4` 退化成语义不同的训练。

---

## 2026-03-10: DeepScaler actor-side chunking hyperparameter count clarification

### Scope

- 无代码改动。
- 说明若为 DeepScaler `num_generations=4` 增加 actor-side generation chunking，最小实现需要新增多少超参数，以及哪些参数可以保持内部常量而不暴露给用户。

### Changed files

1. `develop.md`

### Validation

- 无新增代码执行。
- 复用前序对 `tunix/rl/experimental/agentic_rl_learner.py`、`tunix/rl/experimental/agentic_grpo_learner.py`、`tunix/sft/peft_trainer.py` 的审阅结论。

### Known risks / TODO

- 如果后续把过多调度细节都暴露成 CLI，会增加使用复杂度并抬高误配风险。
- 更合适的做法通常是只暴露 1 个主开关或 1 个 chunk 大小参数，其余行为由代码按 `num_generations` 和现有 batch 参数自动推导。

---

## 2026-03-10: DeepScaler actor-side generation chunking for `num_generations=4`

### Scope

- 为 `examples/deepscaler` 增加可选的 actor-side generation chunking。
- 保持 reward / advantage 仍按完整 `num_generations` 分组计算，只在 actor 训练更新阶段按更小的 completion chunk 分批送入前向/反向。
- 同步放大 actor trainer 的梯度累积步数与 weight sync 计数，避免 chunking 改变 optimizer step 频率。

### Changed files

1. `tunix/rl/experimental/agentic_rl_learner.py`
2. `tunix/rl/rl_cluster.py`
3. `examples/deepscaler/train_deepscaler_nb.py`
4. `examples/deepscaler/run_train.sh`
5. `examples/deepscaler/README.md`
6. `develop.md`

### Validation

- `python -m py_compile examples/deepscaler/train_deepscaler_nb.py tunix/rl/experimental/agentic_rl_learner.py tunix/rl/rl_cluster.py`
- `bash -n examples/deepscaler/run_train.sh`
- `source .venv_sglang312/bin/activate && python examples/deepscaler/train_deepscaler_nb.py --help | rg -n "actor-generation-chunk-size|num-generations|rollout-prompt-batch-size"`
- 确认：
  - 新参数 `--actor-generation-chunk-size` 已出现在 DeepScaler CLI help 中。
  - `agentic` 训练循环会把每个 prompt micro-batch 按 `actor_generation_chunk_size` 切成多个 actor train batch。
  - `RLCluster` 会在 actor trainer 初始化时按 chunk 因子放大 `gradient_accumulation_steps`，只影响 actor 分支，不改 critic / baseline 路径。
  - `run_train.sh` 已支持通过环境变量 `ACTOR_GENERATION_CHUNK_SIZE` 传递该新参数。

### Known risks / TODO

- 这次只做了静态验证，没有在当前机器上跑完整 `num_generations=4` 训练，因此还不能宣称已实测消除所有 OOM。
- 若 rollout 侧仍然 OOM，仍需同时降低 `ROLLOUT_PROMPT_BATCH_SIZE`；actor-side chunking 只缓解 actor 训练峰值。
- 当前实现要求 `actor_generation_chunk_size` 必须整除 `num_generations`，这是为了保持 chunk 形状稳定并避免额外 JIT 形状分叉。

---

## 2026-03-10: DeepScaler `G=4` chunking smoke-test status clarification

### Scope

- 无代码改动。
- 回答本次 actor-side chunking 改动是否已经实际跑过 smoke test，并记录当前最接近真实运行态的验证进度。

### Changed files

1. `develop.md`

### Validation

- 运行：
  - `source .venv_sglang312/bin/activate && RUN_TS=$(date +%Y%m%d_%H%M%S) CHECKPOINT_DIR=/tmp/deepscaler_ckpt_${RUN_TS} METRICS_LOG_DIR=/tmp/deepscaler_tb_${RUN_TS} NUM_GENERATIONS=4 ACTOR_GENERATION_CHUNK_SIZE=2 ROLLOUT_PROMPT_BATCH_SIZE=2 ./examples/deepscaler/run_train.sh --smoke-test --rollout-engine sglang_jax --rollout-tp 2`
- 观测到：
  - 启动日志打印了 `actor_generation_chunk_size=2` 与 `actor_grad_acc_factor=2`
  - 数据预处理完成
  - `sglang_jax` 的 extend/decode precompile 完成
  - 进度条进入 `Actor Training: 0%|...| 0/315`
- 后续处理：
  - 未等待该 smoke test 完整收尾；在确认新参数链路已进入真实 actor training 入口后停止继续等待
  - 运行目录已创建：`/tmp/deepscaler_ckpt_20260310_012727`、`/tmp/deepscaler_tb_20260310_012727`

### Known risks / TODO

- 这说明“参数解析 -> rollout precompile -> actor training 入口”是通的，但还不等于完整 smoke test 成功退出。
- 若要把结论升级成“smoke test passed”，还需要再跑一次并等待完整退出码与首步/收尾日志。

---

## 2026-03-07: DeepScaler eval seed flow clarification

### Scope

- 无代码改动。
- 说明 `run_eval_pass1_avg16.sh` 当前评测链路里 seed 的实际来源，以及 `run_idx` 为什么尚未参与采样 seed。

### Changed files

1. `develop.md`

### Validation

- `nl -ba examples/deepscaler/run_eval_pass1_avg16.sh | sed -n '24,40p'`
- `nl -ba examples/deepscaler/run_eval.sh | sed -n '31,46p'`
- `nl -ba examples/deepscaler/math_eval_nb.py | sed -n '501,509p'`
- `nl -ba tunix/generate/sampler.py | sed -n '719,723p'`
- `nl -ba tunix/generate/sglang_jax_sampler.py | sed -n '243,252p'`
- 确认：
  - wrapper 里的 `run_idx` 当前只用于循环轮次、日志文件名和日志输出。
  - `run_eval.sh` 没有单独的 seed 参数透传。
  - `math_eval_nb.py` 里实际采样 seed 来自 `pass_idx`。
  - 当 `EVAL_NUM_PASSES=1` 时，`pass_idx` 恒为 `0`，因此当前每轮实际传给采样器的 seed 都是 `0`。

### Known risks / TODO

- 当前 wrapper 的“16 次平均”仍未显式构造 16 个不同 seed；若需要严格独立采样，应新增 seed 参数并将 `run_idx` 接入该参数。

---

## 2026-03-07: DeepScaler eval wrapper switched to sglang-jax

### Scope

- 将 `examples/deepscaler/run_eval_pass1_avg16.sh` 固定切换到 `sglang-jax` 采样 backend。
- 在 wrapper 日志和最终汇总中输出所用 sampler，避免误读评测口径。

### Changed files

1. `examples/deepscaler/run_eval_pass1_avg16.sh`
2. `develop.md`

### Validation

- `sed -n '1,120p' examples/deepscaler/run_eval_pass1_avg16.sh`
- `bash -n examples/deepscaler/run_eval_pass1_avg16.sh`
- 确认 wrapper 调用 `run_eval.sh` 时固定追加 `--sampler-type sglang-jax`。
- 确认开始日志、结束日志和最终汇总都会显示 `sampler=sglang-jax` / `Sampler: sglang-jax`。

### Known risks / TODO

- 这次修改只切换了 sampler backend，没有显式为每轮 run 注入不同 seed；“16 次独立采样平均”的语义仍不严格。
- 当前 `math_eval_nb.py` 中 `sglang-jax` 配置显式设置 `enable_deterministic_sampling=False`，因此结果可能比 `vanilla` 更容易出现 run-to-run 波动，但这不等价于严格受控的 16 个不同随机 seed。
- 运行该 wrapper 依赖 `sgl_jax` 及其运行时环境可用；若环境缺失，评测会直接失败。

---

## 2026-03-07: DeepScaler repeated-seed determinism clarification

### Scope

- 无代码改动。
- 核对 `run_eval_pass1_avg16.sh` 在默认 `vanilla` 采样 backend 下是否会因固定 `seed=0` 而得到重复结果。

### Changed files

1. `develop.md`

### Validation

- `nl -ba examples/deepscaler/run_eval_pass1_avg16.sh | sed -n '1,120p'`
- `nl -ba examples/deepscaler/math_eval_nb.py | sed -n '495,510p'`
- `nl -ba tunix/generate/sampler.py | sed -n '632,740p'`
- `nl -ba tunix/generate/sampler.py | sed -n '436,452p'`
- `nl -ba tunix/generate/sglang_jax_sampler.py | sed -n '202,255p'`
- `rg -n "sampler_type|vanilla|sglang-jax" examples/deepscaler/math_eval_nb.py`
- 确认：
  - `run_eval_pass1_avg16.sh` 每轮都固定 `EVAL_NUM_PASSES=1`。
  - `math_eval_nb.py` 在 `num_passes=1` 时每轮都用 `seed=0`。
  - 默认 `sampler_type` 为 `vanilla`。
  - `vanilla` 采样器会将整数 seed 转成 `jax.random.PRNGKey(seed)`，并在每个 decoding step 通过 `jax.random.fold_in(sampler_state.seed, decoding_step)` 取样，因此相同模型、相同输入、相同 seed 下属于确定性采样路径。

### Known risks / TODO

- 若调用方改成 `sglang-jax` 或其他 backend，或底层运行时存在非确定性，重复运行结果可能不完全一致。
- 当前结论基于代码路径推断，未在本机对完整大模型评测做 16 次重复实测。

---

## 2026-03-07: DeepScaleR baseline metric interpretation follow-up

### Scope

- 无代码改动。
- 继续核对官方 `22.9%` 与 `28.8%` 的含义差异，确认二者不应被视为“同一模型同一评测预算下仅因 8K/长上下文不同而严格一一对应”的公开结论。

### Changed files

1. `develop.md`

### Validation

- 查阅官方模型卡：
  - `https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview`
- 查阅官方讨论：
  - `https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview/discussions/13`
- 复核本地评测脚本：
  - `nl -ba examples/deepscaler/run_eval.sh | sed -n '1,120p'`
  - `nl -ba examples/deepscaler/run_eval_pass1_avg16.sh | sed -n '1,120p'`
- 确认：
  - 模型卡公开给出 `22.9% -> 33%` 的描述仅标注为 `Initial 8K Context (0-1040 steps)` 训练曲线阶段。
  - 模型卡公开给出 `28.8%` 时仅标注为汇总评测表中的 `DeepSeek-R1-Distill-Qwen-1.5B`，并说明该表为 `Pass@1 accuracy averaged over 16 samples for each problem`。
  - 官方讨论中作者建议复现其公开结果时使用 `max length 2**15, temperature 0.6, top_p 0.95`，这说明公开表分数更接近长推理预算评测，而不是本地 `run_eval.sh` 当前默认的 `8192` 生成上限。

### Known risks / TODO

- 官方公开材料没有把 `22.9%` 与 `28.8%` 的评测脚本、采样次数、长度限制逐项并排写清，因此两者差异来源只能做有限推断，不能当成官方明示结论。

---

## 2026-03-07: DeepScaleR 22.9 vs 28.8 vs 43.1 metric mapping clarification

### Scope

- 无代码改动。
- 进一步核对 DeepScaleR 官方公开数字 `22.9%`、`28.8%`、`33%`、`43.1%` 各自对应的模型/训练阶段，并与本仓库默认评测脚本的默认模型和参数做映射。

### Changed files

1. `develop.md`

### Validation

- `nl -ba examples/deepscaler/run_eval.sh | sed -n '1,120p'`
- `nl -ba examples/deepscaler/run_eval_pass1_avg16.sh | sed -n '1,120p'`
- `nl -ba examples/deepscaler/math_eval_nb.py | sed -n '420,520p'`
- 查阅官方 Hugging Face 模型卡与讨论：
  - `https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview`
  - `https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview/discussions/13`
- 确认：
  - `22.9% -> 33%` 对应的是官方 RL 训练的 `Initial 8K Context (0-1040 steps)` 阶段内，同一训练过程从早期到后期的提升，并非默认基座模型分数。
  - `28.8%` 对应官方表格中的基座模型 `DeepSeek-R1-Distill-Qwen-1.5B`。
  - `43.1%` 对应最终公开模型 `DeepScaleR-1.5B-Preview`。
  - 本地 `run_eval.sh` 默认模型仍是 `DeepSeek-R1-Distill-Qwen-1.5B`，不是官方 8K 阶段中间 checkpoint，也不是最终 DeepScaleR preview checkpoint。

### Known risks / TODO

- 即使切换到相同模型，若评测采样实现与官方“多次独立 pass@1 取平均”不一致，结果仍可能与官方数字有系统偏差。

---

## 2026-03-07: DeepScaleR official eval comparison clarification

### Scope

- 无代码改动。
- 核对 DeepScaleR 官方公开评测口径，并与仓库内 `examples/deepscaler/run_eval_pass1_avg16.sh` / `examples/deepscaler/eval_all.sh` 的实现语义做对比。

### Changed files

1. `develop.md`

### Validation

- `nl -ba examples/deepscaler/run_eval.sh | sed -n '1,120p'`
- `nl -ba examples/deepscaler/run_eval_pass1_avg16.sh | sed -n '1,120p'`
- `nl -ba examples/deepscaler/math_eval_nb.py | sed -n '420,548p'`
- `nl -ba examples/deepscaler/eval_all.sh | sed -n '1,120p'`
- 查阅官方 Hugging Face 模型卡与项目博客：
  - `https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview`
  - `https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2`
- 确认官方模型卡写明：
  - 最终 `DeepScaleR-1.5B-Preview` 在 AIME 2024 上为 `43.1%`
  - 基座 `DeepSeek-R1-Distill-Qwen-1.5B` 为 `28.8%`
  - 指标口径为 `Pass@1 accuracy averaged over 16 samples for each problem`
  - 8K 上下文训练阶段写明 `22.9% -> 33% Pass@1 on AIME 2024`
- 确认本地 `run_eval_pass1_avg16.sh` 默认沿用 `run_eval.sh` 的基座模型 `DeepSeek-R1-Distill-Qwen-1.5B`，并非默认评测 `DeepScaleR-1.5B-Preview`。
- 确认本地 `run_eval_pass1_avg16.sh` 每轮都强制 `EVAL_NUM_PASSES=1`，而 `math_eval_nb.py` 在 `num_passes=1` 时固定使用 `seed=0`，因此“16 次平均”未显式引入 16 个不同采样 seed。

### Known risks / TODO

- 如果底层采样器在相同 seed 下仍存在非确定性，`run_eval_pass1_avg16.sh` 可能仍会出现轻微波动；但从实现看，它并没有显式构造官方口径所需的 16 个不同样本。
- 官方 8K 数字来自训练阶段曲线描述，不应直接等同于“最终公开模型在本仓库 `8192` 生成上限配置下的可复现分数”。

---

## 2026-03-07: DeepScaler pass1_avg16 wrapper clarification

### Scope

- 无代码改动。
- 复核 `examples/deepscaler/run_eval_pass1_avg16.sh` 的评测语义，确认它是否等价于 `pass@1` 独立运行 16 次后取平均。

### Changed files

1. `develop.md`

### Validation

- `nl -ba examples/deepscaler/run_eval_pass1_avg16.sh | sed -n '1,120p'`
- 确认默认 `NUM_RUNS="${NUM_RUNS:-16}"`。
- 确认脚本在每轮强制设置 `EVAL_NUM_PASSES=1` 后调用 `run_eval.sh`。
- 确认脚本逐轮解析 `Correct:` 与 `Accuracy:`，并在结束时输出 `Metric: Pass@1 averaged over ${completed_runs} independent runs`。

### Known risks / TODO

- 汇总逻辑依赖当前评测输出中的 `Correct:` 和 `Accuracy:` 固定文本；若输出格式变更，脚本解析会失效。

---

## 2026-03-07: DeepScaler eval semantics clarification only

### Scope

- 无代码改动。
- 核对 `examples/deepscaler/run_eval.sh`、`examples/deepscaler/math_eval_nb.py` 与 `examples/deepscaler/run_eval_pass1_avg16.sh` 的评测语义，确认 `pass@1` 与 `16` 的含义。

### Changed files

1. `develop.md`

### Validation

- `sed -n '1,220p' examples/deepscaler/run_eval.sh`
- `sed -n '430,640p' examples/deepscaler/math_eval_nb.py`
- `sed -n '1,260p' examples/deepscaler/run_eval_pass1_avg16.sh`
- 确认 `run_eval.sh` 默认 `EVAL_NUM_PASSES="${EVAL_NUM_PASSES:-1}"`，即默认单次采样评测。
- 确认 `math_eval_nb.py` 对每道题执行 `num_passes` 次生成，并在任一次回答正确时将该题记为正确；这是单次评测内的 pass@k 语义，不是多次独立运行求平均。
- 确认 `run_eval_pass1_avg16.sh` 才是固定 `EVAL_NUM_PASSES=1` 并默认独立运行 `16` 次后汇总平均结果的脚本。

### Known risks / TODO

- 结论依赖当前脚本实现；如果后续评测脚本输出格式或 `num_passes` 逻辑变化，需要重新核对。

---

## 2026-03-06: DeepScaler eval default model switch

### Scope

- Switched the default evaluation model in `examples/deepscaler/run_eval.sh` from DeepScaleR preview to DeepSeek R1 Distill Qwen 1.5B.
- Changed the default evaluation `num_passes` in `examples/deepscaler/run_eval.sh` from `16` to `1`.
- Changed the default evaluation `max_generation_steps` in `examples/deepscaler/run_eval.sh` from `32768` to `8192`.
- Added a new wrapper script to run pass@1 evaluation 16 times and report the average over runs.
- Made the averaging wrapper tolerate environments without `rg` by falling back to `grep`.

### Changed files

1. `examples/deepscaler/run_eval.sh`
2. `examples/deepscaler/run_eval_pass1_avg16.sh`
3. `develop.md`

### Validation

- `sed -n '1,40p' examples/deepscaler/run_eval.sh`
- `bash -n examples/deepscaler/run_eval.sh`
- `bash -n examples/deepscaler/run_eval_pass1_avg16.sh`
- Confirmed default values now point to:
  - `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
  - local HF cache snapshot `models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/...`
- Confirmed `EVAL_NUM_PASSES="${EVAL_NUM_PASSES:-1}"`.
- Confirmed `EVAL_MAX_GENERATION_STEPS="${EVAL_MAX_GENERATION_STEPS:-8192}"`.
- Confirmed the new wrapper forces `EVAL_NUM_PASSES=1`, runs `NUM_RUNS` times (default `16`), and parses `Correct:` / `Accuracy:` lines from each run log.
- Confirmed the wrapper now uses `rg` if available and falls back to `grep` otherwise.

### Known risks / TODO

- This only changes the default model selection for `run_eval.sh`; callers that explicitly set `MODEL_PATH` or `MODEL_VERSION` are unchanged.
- This only changes the default `num_passes`; callers that explicitly set `EVAL_NUM_PASSES` are unchanged.
- This only changes the default `max_generation_steps`; callers that explicitly set `EVAL_MAX_GENERATION_STEPS` are unchanged.
- The averaging wrapper assumes each run finishes with `Correct:` and `Accuracy:` summary lines; if the Python evaluator output format changes, parsing will break.
- `grep` fallback matches the current fixed summary lines; if those labels change, parsing still breaks.
- The default local snapshot path is machine-specific and will fail on hosts where that HF cache entry does not exist.

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

## 2026-03-07 - DeepScaler eval progress inspection

### Scope

- No code changes.
- Inspected the live `run_eval_pass1_avg16.sh` evaluation progress and extracted completed-run accuracies from the active log directory.

### Changed files

1. `develop.md`

### Validation

- `ls -td /tmp/deepscaler_pass1_avg16_* | head`
- `ps -eo pid,etimes,cmd | rg 'run_eval_pass1_avg16.sh|examples/deepscaler/math_eval_nb.py'`
- `rg -n 'Correct:|Accuracy:' /tmp/deepscaler_pass1_avg16_20260307_151314/run_1.log /tmp/deepscaler_pass1_avg16_20260307_151314/run_2.log`
- `tail -n 40 /tmp/deepscaler_pass1_avg16_20260307_151314/run_3.log`

### Validation results

- Active log directory detected: `/tmp/deepscaler_pass1_avg16_20260307_151314`.
- Completed runs so far:
  - `run_1.log`: `Correct: 8/30`, `Accuracy: 26.67%`
  - `run_2.log`: `Correct: 8/30`, `Accuracy: 26.67%`
- `run_3.log` exists and is currently in progress (`--seed 2`); no final accuracy written yet at inspection time.

### Known risks / TODO

- The live results can change while the script is still running; check the active log directory again for updated averages after more runs complete.

## 2026-03-07 - DeepScaler run_1 vs run_2 output comparison

### Scope

- No code changes.
- Compared the first two completed `run_eval_pass1_avg16.sh` logs to determine whether identical accuracies came from different answer sets or from identical generations.

### Changed files

1. `develop.md`

### Validation

- `sha256sum /tmp/deepscaler_pass1_avg16_20260307_151314/run_1.log /tmp/deepscaler_pass1_avg16_20260307_151314/run_2.log`
- `diff -u /tmp/deepscaler_pass1_avg16_20260307_151314/run_1.log /tmp/deepscaler_pass1_avg16_20260307_151314/run_2.log`
- Compared full logs again after removing the `Base seed:` line.
- Parsed each log's 30 `model_answer ... IS CORRECT/NOT CORRECT` lines and compared the full sequence.

### Validation results

- `run_1.log` and `run_2.log` differ in the raw file hash, but the only textual diff found was:
  - `Base seed: 0`
  - `Base seed: 1`
- After removing the `Base seed:` line, the two logs are byte-for-byte identical at the line level.
- The full per-problem answer/correctness sequence is identical across both runs.
- Correct positions in dataset order are identical: `1, 7, 8, 9, 10, 16, 20, 25`.

### Known risks / TODO

- This strongly suggests the current `sglang-jax` eval path is effectively deterministic for these runs, or the backend is not varying outputs with the provided `sampling_seed`; confirm after more runs complete before drawing a final conclusion.

## 2026-03-07 - DeepScaler sglang-jax identical-run root cause analysis

### Scope

- No code changes.
- Traced why `run_eval_pass1_avg16.sh` produced identical outputs for `run_1` and `run_2` despite different wrapper-level seeds.

### Changed files

1. `develop.md`

### Validation

- Read [examples/deepscaler/math_eval_nb.py](/home/lhf_hongfu_gmail_com/tunix/examples/deepscaler/math_eval_nb.py) around sampler construction and eval seed usage.
- Read [tunix/generate/sglang_jax_sampler.py](/home/lhf_hongfu_gmail_com/tunix/tunix/generate/sglang_jax_sampler.py) around engine args and `sampling_seed` request wiring.
- Read local installed `sgl_jax` sources under `/tmp/sglang-jax/python/sgl_jax/`:
  - `srt/sampling/sampling_batch_info.py`
  - `srt/layers/sampler.py`
  - `srt/server_args.py`
  - `srt/model_executor/model_runner.py`

### Validation results

- Eval constructs `SglangJaxConfig(enable_deterministic_sampling=False)`.
- The wrapper seed is written into per-request `sampling_seed`, but `sgl_jax` only materializes `sampling_seeds` when `enable_deterministic_sampling=True`; otherwise it sets `sampling_seeds=None`.
- When `sampling_seeds=None`, the sampler falls back to engine RNG-based multinomial sampling instead of request-seeded sampling.
- Engine RNG is initialized from `server_args.random_seed`; if not provided, `sgl_jax` defaults it to `42`.
- Because each eval run starts a fresh engine with the same default RNG seed, same prompts, same order, and same batch size, the random stream is replayed identically and outputs match exactly across runs.

### Known risks / TODO

- If later runs also remain identical, the current `avg16` result should be treated as repeated single-run measurement rather than independent-sample averaging unless engine-level randomness is re-plumbed.

## 2026-03-07 - DeepScaler sglang-jax eval seeding recommendation

### Scope

- No code changes.
- Recorded the recommended fix direction for making `run_eval_pass1_avg16.sh` produce meaningful independent samples while keeping `sglang-jax`.

### Changed files

1. `develop.md`

### Recommendation

- Preferred path if keeping `sglang-jax`:
  - Turn `enable_deterministic_sampling=True` in eval.
  - Keep per-run seed plumbing (`run_idx -> EVAL_SEED`) and let request `sampling_seed` drive sampling.
- Why this is preferred:
  - It makes each run reproducible.
  - Different seeds then map to intentionally different samples, instead of relying on engine-global RNG side effects.
  - It is closer to a defensible `pass@1` averaged-over-runs protocol.
- Less preferred fallback:
  - Keep deterministic sampling off, but plumb `run_idx` into engine/server `random_seed`.
  - This can make runs differ, but the randomness is engine-global and more sensitive to batching/order/runtime details.

### Known risks / TODO

- Even with deterministic sampling enabled, verify at least two adjacent runs produce different logs before treating `avg16` as independent-sample averaging.

## 2026-03-07 - DeepScaler eval enable deterministic sampling for sglang-jax

### Scope

- Enabled deterministic sampling in the `sglang-jax` eval path so per-run `EVAL_SEED` / request `sampling_seed` actually controls sampling.
- Kept the existing `run_eval_pass1_avg16.sh` seed plumbing unchanged.

### Changed files

1. `examples/deepscaler/math_eval_nb.py`
2. `develop.md`

### Validation

- `python -m py_compile examples/deepscaler/math_eval_nb.py`
- `bash -n examples/deepscaler/run_eval.sh`
- `bash -n examples/deepscaler/run_eval_pass1_avg16.sh`
- `export PYTHONPATH=/tmp/tunix_eval_shim${PYTHONPATH:+:$PYTHONPATH}`
- `source .venv_sglang312/bin/activate`
- `LOG_DIR=/tmp/deepscaler_pass1_avg16_detcheck_20260307_161338 NUM_RUNS=2 EVAL_NUM_BATCHES=5 EVAL_MAX_PROMPT_LENGTH=512 EVAL_MAX_GENERATION_STEPS=512 ./examples/deepscaler/run_eval_pass1_avg16.sh`
- Compared `/tmp/deepscaler_pass1_avg16_detcheck_20260307_161338/run_1.log` and `run_2.log`

### Validation results

- Syntax checks passed.
- Short 2-run validation completed successfully with `sglang-jax`.
- Both runs reported `0/5`, `0.00%`, but the raw generations are no longer identical:
  - `run_1.log` and `run_2.log` differ even after removing the `Base seed:` line.
  - All `5/5` recorded raw response entries differ across the two runs.
- This confirms that different eval seeds are now affecting generation, even when the tiny validation slice happens to produce the same final accuracy.

### Known risks / TODO

- The short validation used only 5 questions and a reduced token budget, so equal `acc` there is not meaningful by itself.
- A full `16`-run AIME evaluation is still needed to estimate the new averaged metric under the corrected seeding behavior.

## 2026-03-07 - DeepScaler batch size vs avg16 metric clarification

### Scope

- No code changes.
- Clarified the difference between increasing `EVAL_BATCH_SIZE` and averaging `pass@1` over 16 independent runs.

### Changed files

1. `develop.md`

### Clarification

- `EVAL_BATCH_SIZE=16` only changes how many different questions are processed in parallel inside a single eval run.
- It does not create 16 samples per question and does not replace `NUM_RUNS=16`.
- If the goal is `avg16 pass@1`, the correct setup is still:
  - `run_eval_pass1_avg16.sh`
  - `NUM_RUNS=16`
  - optionally set `EVAL_BATCH_SIZE=16` to speed up each run

### Known risks / TODO

- Larger eval batch sizes may change throughput or memory pressure, but they do not change the metric definition.

## 2026-03-07 - DeepScaler single-run batch16 vs avg16 clarification

### Scope

- No code changes.
- Clarified why a single eval run with `EVAL_BATCH_SIZE=16` is not the same metric as averaging `pass@1` over 16 independent runs.

### Changed files

1. `develop.md`

### Clarification

- `EVAL_BATCH_SIZE` controls how many different questions are processed in parallel within one run.
- `NUM_RUNS=16` in `run_eval_pass1_avg16.sh` controls how many independent seeded runs are averaged.
- A single run with `EVAL_BATCH_SIZE=16` still samples each question once.
- `avg16 pass@1` samples each question 16 times across different runs and averages the resulting accuracies.
- Under ideal batch-invariant sampling, one-run accuracy and avg16 accuracy target the same expectation, but avg16 has much lower variance and is therefore more stable.

### Known risks / TODO

- If the backend is not perfectly batch-invariant, changing `EVAL_BATCH_SIZE` may also slightly change outputs, but that still does not make one run equivalent to 16-run averaging.

## 2026-03-07 - DeepScaler batch1 run command clarification

### Scope

- No code changes.
- Clarified the exact command for `EVAL_BATCH_SIZE=1` with `NUM_RUNS=16`.

### Changed files

1. `develop.md`

### Clarification

- If the desired protocol is `pass@1` averaged over 16 independent runs while keeping per-run eval batch size at 1, use:
  - `EVAL_BATCH_SIZE=1`
  - `NUM_RUNS=16`
  - `./examples/deepscaler/run_eval_pass1_avg16.sh`

### Known risks / TODO

- None beyond the existing long runtime of the full 16-run evaluation.

## 2026-03-07 - PYTHONPATH command clarification for DeepScaler eval

### Scope

- No code changes.
- Clarified the meaning of `export PYTHONPATH=/tmp/tunix_eval_shim${PYTHONPATH:+:$PYTHONPATH}` and why it is needed for the current `sglang-jax` eval path.

### Changed files

1. `develop.md`

### Clarification

- The command prepends `/tmp/tunix_eval_shim` to Python's module search path for the current shell and child processes.
- `${PYTHONPATH:+:$PYTHONPATH}` means:
  - if `PYTHONPATH` is already non-empty, append `:$PYTHONPATH`
  - otherwise append nothing
- In this workspace it is used so Python can find the temporary shim at `/tmp/tunix_eval_shim/sitecustomize.py`, which works around the current missing `tunix.google...` import alias for `sglang-jax` eval.

### Known risks / TODO

- The command relies on the temporary shim existing at `/tmp/tunix_eval_shim`; if that directory is removed, the export no longer helps.

## 2026-03-07 - /tmp/tunix_eval_shim purpose clarification

### Scope

- No code changes.
- Inspected the temporary `/tmp/tunix_eval_shim` directory and clarified what it does and when it is required.

### Changed files

1. `develop.md`

### Validation

- `find /tmp/tunix_eval_shim -maxdepth 2 -type f -o -type d | sort`
- `nl -ba /tmp/tunix_eval_shim/sitecustomize.py`

### Validation results

- `/tmp/tunix_eval_shim` currently contains `sitecustomize.py` plus Python bytecode cache files.
- The shim injects a runtime alias:
  - `tunix.google.stubs.sglang_jax_sampler_stub`
  - mapped to `tunix.generate.sglang_jax_sampler`
- This is a temporary workaround so the current `examples/deepscaler/math_eval_nb.py` import path can resolve without editing repository code.

### Known risks / TODO

- It is required for the current `sglang-jax` eval command path as long as the repo still imports the missing `tunix.google.stubs...` alias.
- It would stop being necessary if that import path were fixed in-repo or if an equivalent real module were added.

## 2026-03-07 - Missing sglang-jax import path clarification

### Scope

- No code changes.
- Clarified the exact missing module path behind the temporary `/tmp/tunix_eval_shim` workaround.

### Changed files

1. `develop.md`

### Validation

- `rg --files | rg '(^tunix/google/|sglang_jax_sampler)'`
- Checked import resolution for:
  - `tunix.google`
  - `tunix.google.stubs`
  - `tunix.google.stubs.sglang_jax_sampler_stub`
  - `tunix.generate.sglang_jax_sampler`

### Validation results

- The eval code imports:
  - `from tunix.google.stubs import sglang_jax_sampler_stub`
- The repository currently has:
  - `tunix/generate/sglang_jax_sampler.py`
- The repository does not currently have:
  - `tunix/google/`
  - `tunix/google/stubs/`
  - `tunix/google/stubs/sglang_jax_sampler_stub.py`
- Therefore the missing piece is a Python module path / compatibility alias, not a model asset or third-party package.

### Known risks / TODO

- The `/tmp` shim is only a temporary workaround; the cleaner long-term fix is in-repo import cleanup or adding a real compatibility stub module.

## 2026-03-07 - One-line fix clarification for sglang-jax import

### Scope

- No code changes.
- Clarified whether the current missing-module issue can be fixed by editing `tunix/generate/sglang_jax_sampler.py`.

### Changed files

1. `develop.md`

### Clarification

- The missing-module issue is not inside `tunix/generate/sglang_jax_sampler.py`.
- The broken import site is in `examples/deepscaler/math_eval_nb.py`, which currently imports `tunix.google.stubs.sglang_jax_sampler_stub`.
- Therefore, the clean one-line fix is to change that import to `from tunix.generate import sglang_jax_sampler`.

### Known risks / TODO

- If other files also depend on the `tunix.google.stubs...` alias, they should be checked before removing the workaround globally.

## 2026-03-07 - DeepScaler one-line sglang-jax import fix

### Scope

- Replaced the broken `tunix.google.stubs...` import in the DeepScaler eval path with the real in-repo `tunix.generate.sglang_jax_sampler` module.

### Changed files

1. `examples/deepscaler/math_eval_nb.py`
2. `develop.md`

### Validation

- `python -m py_compile examples/deepscaler/math_eval_nb.py`
- `source .venv_sglang312/bin/activate && unset PYTHONPATH && python - <<'PY'`
  `from tunix.generate import sglang_jax_sampler`
  `print(sglang_jax_sampler.__file__)`
  `PY`

### Validation results

- Python syntax check passed.
- Direct import of `tunix.generate.sglang_jax_sampler` succeeded without relying on `/tmp/tunix_eval_shim`.
- This fixes the specific missing-module issue caused by `from tunix.google.stubs import sglang_jax_sampler_stub`.

### Known risks / TODO

- I did not rerun a full end-to-end eval after this one-line import cleanup; the fix specifically addresses the previously missing module path.

## 2026-03-08 - DeepScaler README add avg16 eval result

### Scope

- Added the local `sglang-jax` `pass@1` averaged-over-16-runs eval command and the observed AIME 2024 result to the DeepScaler README.

### Changed files

1. `examples/deepscaler/README.md`
2. `develop.md`

### Validation

- `rg -n 'pass@1 averaged over 16|Average Accuracy: 18.9594|run_eval_pass1_avg16.sh' examples/deepscaler/README.md`
- `git diff -- examples/deepscaler/README.md`

### Validation results

- README now includes the `run_eval_pass1_avg16.sh` repro command.
- README now includes the recorded local summary:
  - `Runs: 16`
  - `Average Correct: 5.6875/30.0000`
  - `Average Accuracy: 18.9594%`

### Known risks / TODO

- The recorded result is environment-specific and reflects the local run logged at `/tmp/deepscaler_pass1_avg16_20260307_171010`.

## 2026-03-08 - DeepScaler avg16 command path clarification

### Scope

- No code changes.
- Confirmed the exact command written in the DeepScaler README for the recorded `avg16 pass@1` result and summarized the underlying code path.

### Changed files

1. `develop.md`

### Validation

- `nl -ba examples/deepscaler/README.md | sed -n '114,140p'`
- `nl -ba examples/deepscaler/run_eval_pass1_avg16.sh | sed -n '1,80p'`
- `nl -ba examples/deepscaler/run_eval.sh | sed -n '1,60p'`
- `nl -ba examples/deepscaler/math_eval_nb.py | sed -n '312,338p'`

### Validation results

- README includes the exact repro command for the recorded result.
- The code path is:
  - `examples/deepscaler/run_eval_pass1_avg16.sh`
  - `examples/deepscaler/run_eval.sh`
  - `examples/deepscaler/math_eval_nb.py`

### Known risks / TODO

- None beyond the existing environment dependence of the recorded local result.

## 2026-03-08 - DeepScaler README simplify avg16 command

### Scope

- Simplified the DeepScaler README `avg16 pass@1` command to rely on the existing default values for `EVAL_BATCH_SIZE=1` and `NUM_RUNS=16`.

### Changed files

1. `examples/deepscaler/README.md`
2. `develop.md`

### Validation

- `nl -ba examples/deepscaler/README.md | sed -n '117,124p'`
- `git diff -- examples/deepscaler/README.md`

### Validation results

- README now shows the shorter command:
  - `source .venv_sglang312/bin/activate`
  - `LOG_DIR=/tmp/deepscaler_pass1_avg16_$(date +%Y%m%d_%H%M%S) ./examples/deepscaler/run_eval_pass1_avg16.sh`
- The removed environment variables were redundant because:
  - `EVAL_BATCH_SIZE` already defaults to `1`
  - `NUM_RUNS` already defaults to `16`

### Known risks / TODO

- None beyond the existing environment dependence of the recorded local result.

## 2026-03-08 - DeepScaler DBC micro-batch analysis

### Scope

- No code changes.
- Analyzed whether DeepScaler GRPO training would apply dynamic batch curation over the outer `batch_size=128` or over the inner training micro-batch.
- Confirmed the effective DBC screening window is controlled by `train_micro_batch_size * num_generations`, not by the outer `batch_size`.

### Changed files

1. `develop.md`

### Validation

- `nl -ba examples/deepscaler/run_train.sh | sed -n '1,120p'`
- `nl -ba tunix/rl/rl_learner.py | sed -n '240,520p'`
- `nl -ba tunix/rl/robust_trainer.py | sed -n '1,180p'`
- `nl -ba tunix/rl/self_inf_trainer.py | sed -n '1,220p'`

### Validation results

- `examples/deepscaler/run_train.sh` defaults confirm:
  - `BATCH_SIZE=128`
  - `MINI_BATCH_SIZE=128`
  - `TRAIN_MICRO_BATCH_SIZE=1`
  - `NUM_GENERATIONS=2`
- `tunix/rl/rl_learner.py` confirms GRPO:
  - splits the outer batch into training micro-batches,
  - repeats each micro-batch by `num_generations`,
  - may merge for rollout/inference efficiency,
  - then splits back to original micro-batch boundaries before actor training.
- `tunix/rl/robust_trainer.py` and `tunix/rl/self_inf_trainer.py` both apply per-sample filtering inside a single `_train_step`, so they only see the post-split micro-batch.
- For the current DeepScaler defaults, the effective DBC screening window is `1 * 2 = 2` samples per actor train step, not `128`.

### Known risks / TODO

- With an effective screening window of `2`, outlier-L2 filtering is unlikely to be useful because the mean/std estimate is too small to robustly identify outliers.
- With `train_micro_batch_size=1`, self-influence `batch` and `group` scopes are expected to behave very similarly, because each actor step contains only one GRPO group.
- If stronger DBC behavior is desired, increase `train_micro_batch_size`; this will also increase per-step memory and compile cost because DBC computes per-sample gradients via `jax.vmap`.

## 2026-03-08 - DeepScaler normal gradient accumulation walkthrough

### Scope

- 无代码改动。
- Traced the standard DeepScaler GRPO actor training path to explain exactly how gradients are accumulated and when model weights are updated under the current defaults.

### Changed files

1. `develop.md`

### Validation

- `nl -ba examples/deepscaler/run_train.sh | sed -n '1,120p'`
- `nl -ba tunix/rl/rl_cluster.py | sed -n '100,150p'`
- `nl -ba tunix/rl/rl_learner.py | sed -n '550,760p'`
- `nl -ba tunix/sft/peft_trainer.py | sed -n '180,240p'`
- `nl -ba tunix/sft/peft_trainer.py | sed -n '300,340p'`
- `nl -ba tunix/sft/peft_trainer.py | sed -n '650,720p'`

### Validation results

- `run_train.sh` defaults confirm the active DeepScaler training shape is:
  - `batch_size=128`
  - `mini_batch_size=128`
  - `train_micro_batch_size=1`
  - `num_generations=2`
- `RLTrainingConfig` derives `gradient_accumulation_steps = mini_batch_size // train_micro_batch_size`, so the current setup accumulates for `128` micro-steps before counting one optimizer update.
- `PeftTrainer` wraps the optimizer with `optax.MultiSteps(...)`, which means `optimizer.update(...)` is called every micro-step but the real parameter update is deferred until the accumulation boundary.
- `RLLearner` splits the outer batch into training micro-batches, repeats each micro-batch by `num_generations`, and feeds those micro-batches one by one into actor training.
- With current defaults, each actor micro-step trains on `1 prompt * 2 generations = 2 trajectories`, and `128` such micro-steps make one actor optimizer step.

### Known risks / TODO

- The outer `batch_size=128` can be misleading: it is not the size of a single actor gradient computation under the current defaults.
- If future work changes `train_micro_batch_size`, it will change both the DBC screening window and the memory/compile behavior of each actor micro-step.

## 2026-03-08 - my_example DBC hyperparameter and activation analysis

### Scope

- 无代码改动。
- Inspected `my_example` DBC-related flags, defaults, wrapper scripts, and trainer selection path.
- Confirmed which knobs actually control screening behavior and which metrics verify that screening logic executed.

### Changed files

1. `develop.md`

### Validation

- `rg -n "dbc|dynamic batch curation|curation|self_inf|use_dynamic_batch_curation|use-dbc|skipped_samples|grad_norm_mean|self_inf_kept_fraction" my_example tunix/rl`
- `nl -ba my_example/config.py | sed -n '1,320p'`
- `nl -ba my_example/train.py | sed -n '1,260p'`
- `nl -ba my_example/run_grpo_gemma.sh | sed -n '1,220p'`
- `nl -ba my_example/run_dbc_outlier_l2.sh | sed -n '1,220p'`
- `nl -ba my_example/run_dbc_self_inf_batch.sh | sed -n '1,220p'`
- `nl -ba my_example/run_dbc_self_inf_group.sh | sed -n '1,220p'`
- `nl -ba my_example/main.py | sed -n '81,240p'`
- `nl -ba tunix/rl/grpo/grpo_learner.py | sed -n '160,260p'`

### Validation results

- `my_example` exposes three DBC variants:
  - outlier-L2: `--use-dbc-outlier-l2`
  - self-influence batch: `--use-dbc-self-inf-batch`
  - self-influence group: `--use-dbc-self-inf-group`
- The only exposed DBC numeric threshold is `--curation-threshold` with default `3.0`, and it is used by outlier-L2 only.
- Self-influence does not expose a CLI threshold in `my_example`; it relies on `SelfInfTrainer` default `dot_threshold=0.0`.
- `run_grpo_gemma.sh` defaults to:
  - `--train-micro-batch-size 4`
  - `--num-generations 4`
- `my_example/train.py` sets both `mini_batch_size` and `train_micro_batch_size` to the same value, so `gradient_accumulation_steps=1` and DBC runs on the full prompt micro-batch rather than on a smaller accumulated sub-batch.
- Under those defaults, the effective DBC screening window is `4 prompts * 4 generations = 16` trajectories per actor train step.
- `GRPOLearner` always registers DBC metrics (`skipped_samples`, `grad_norm_mean`, `grad_norm_std`, `self_inf_dot_mean`, `self_inf_dot_std`, `self_inf_kept_fraction`) so TensorBoard/exported logs can confirm whether filtering logic ran and whether any samples were actually dropped.

### Known risks / TODO

- Enabling DBC guarantees the filtering code path runs, but it does not guarantee that any sample will be removed on every step; that still depends on the observed gradient statistics and the chosen threshold.
- If `train_micro_batch_size` or `num_generations` is reduced, the effective DBC window shrinks and filtering becomes less informative.

## 2026-03-08 - my_example DBC screening window breakdown

### Scope

- 无代码改动。
- Broke down the exact screening window for each DBC variant in `my_example`, distinguishing the scoring/comparison window from the final gradient aggregation window.

### Changed files

1. `develop.md`

### Validation

- `nl -ba tunix/rl/robust_trainer.py | sed -n '39,110p'`
- `nl -ba tunix/rl/self_inf_trainer.py | sed -n '50,140p'`
- `nl -ba my_example/run_grpo_gemma.sh | sed -n '103,122p'`
- `nl -ba my_example/train.py | sed -n '101,121p'`
- `nl -ba tunix/rl/rl_cluster.py | sed -n '519,549p'`

### Validation results

- In `my_example`, the default actor-step batch entering DBC is:
  - `train_micro_batch_size=4`
  - `num_generations=4`
  - therefore `4 * 4 = 16` trajectory-level samples per actor step.
- Outlier-L2 (`RobustTrainer`) computes gradient norms, mean, std, and cutoff over all `16` trajectories in the actor step.
- Self-inf batch computes one mean gradient over all `16` trajectories, then scores each trajectory against that same batch mean.
- Self-inf group reshapes the `16` trajectories into `4` prompt groups of size `4`, then scores each trajectory only against its own group mean.
- Even in self-inf group mode, after masking, the final gradient is still averaged over all kept trajectories in the full actor step, not separately per group.

### Known risks / TODO

- The phrase “screening window” is ambiguous for self-inf group because its local scoring window is `4`, but its final update still aggregates across up to `16` kept trajectories from the whole actor step.
- If `batch_size` is not divisible by `num_generations`, self-inf group falls back to batch-scope scoring.

## 2026-03-08 - DeepScaler default-config DBC method recommendation

### Scope

- 无代码改动。
- Evaluated which DBC variants are meaningful under the current DeepScaler defaults (`train_micro_batch_size=1`, `num_generations=2`) and summarized the recommended usage for baseline, outlier-L2, self-inf-batch, and self-inf-group.

### Changed files

1. `develop.md`

### Validation

- `nl -ba examples/deepscaler/run_train.sh | sed -n '19,38p'`
- `nl -ba tunix/rl/robust_trainer.py | sed -n '65,91p'`
- `nl -ba tunix/rl/self_inf_trainer.py | sed -n '78,126p'`
- `nl -ba tunix/rl/rl_cluster.py | sed -n '519,549p'`

### Validation results

- DeepScaler defaults imply an actor-step DBC window of `1 * 2 = 2` trajectories.
- Under a 2-sample window, outlier-L2 with `curation_threshold=3.0` is effectively non-operative for filtering, because the cutoff is at least the larger of the two norms.
- Under the same defaults, self-inf batch and self-inf group are equivalent because there is only one GRPO group per actor step (`train_micro_batch_size=1`, `num_generations=2`).
- Therefore, under strict current defaults:
  - baseline remains the main control run,
  - outlier-L2 should not be prioritized,
  - only one self-inf variant needs to be tried because batch/group collapse to the same behavior.

### Known risks / TODO

- If future DeepScaler experiments increase `train_micro_batch_size`, these conclusions no longer hold: outlier-L2 becomes viable and self-inf batch/group diverge.
- If the user wants DBC to reflect its intended batch-level behavior rather than a 2-sample within-prompt filter, `train_micro_batch_size` must be increased.

## 2026-03-08 - DeepScaler DBC knob distinction: train_micro_batch_size vs num_generations

### Scope

- 无代码改动。
- Clarified which DBC windows are controlled by `train_micro_batch_size` versus `num_generations`, and when each knob should be preferred under DeepScaler defaults.

### Changed files

1. `develop.md`

### Validation

- `nl -ba tunix/rl/robust_trainer.py | sed -n '39,110p'`
- `nl -ba tunix/rl/self_inf_trainer.py | sed -n '50,140p'`
- `nl -ba examples/deepscaler/run_train.sh | sed -n '19,38p'`

### Validation results

- For outlier-L2 and self-inf-batch, the actor-step scoring window scales with `train_micro_batch_size * num_generations`.
- For self-inf-group, the local group-scoring window scales with `num_generations`, while the number of groups scales with `train_micro_batch_size`.
- With DeepScaler defaults (`train_micro_batch_size=1`, `num_generations=2`), increasing only `num_generations` enlarges the within-prompt window but does not create cross-prompt grouping; self-inf-group still collapses to self-inf-batch as long as `train_micro_batch_size=1`.
- Increasing `train_micro_batch_size` is therefore the necessary change when the goal is to recover batch-level behavior or to make self-inf batch/group differ.

### Known risks / TODO

- Increasing either knob raises per-step trajectory count and therefore DBC compute cost.
- Increasing `num_generations` also changes the GRPO algorithmic shape, not just the batching geometry, so it is a less isolated DBC-only intervention than increasing `train_micro_batch_size`.

## 2026-03-08 - DeepScaler focus recommendation for self-inf-group

### Scope

- 无代码改动。
- Clarified the recommended comparison setup when prioritizing only `baseline` and `self-inf-group` for DeepScaler, including the need for matched `num_generations`.

### Changed files

1. `develop.md`

### Validation

- `nl -ba my_example/run_grpo_gemma.sh | sed -n '103,122p'`
- `nl -ba tunix/rl/self_inf_trainer.py | sed -n '88,102p'`
- `nl -ba examples/deepscaler/run_train.sh | sed -n '19,38p'`

### Validation results

- `my_example`’s packaged training script uses `num_generations=4`.
- `self-inf-group` uses `num_generations` as its local group size.
- Therefore, if the goal is to transfer the best-performing `self-inf-group` setup into DeepScaler with minimal moving parts, raising DeepScaler `num_generations` from `2` to `4` is the most direct first knob.
- A fair DBC ablation requires matched GRPO geometry, so the baseline run must also use the same `num_generations=4`; otherwise the result confounds curation with a changed GRPO sampling configuration.

### Known risks / TODO

- With `train_micro_batch_size=1`, `self-inf-group` still remains equivalent to `self-inf-batch`; this is acceptable only if the experiment goal is baseline-vs-group, not group-vs-batch.
- If the matched `num_generations=4` pair shows promise, the next step is to increase `train_micro_batch_size` to make group-vs-batch behavior meaningfully diverge.

## 2026-03-08 - DeepScaler baseline num_generations=4 minimal run

### Scope

- 无代码改动。
- Ran a minimal DeepScaler baseline validation with `num_generations=4`, `num_batches=1`, `num_epochs=1`, and `max_steps=1` to check whether the matched baseline shape fits before trying self-inf-group.

### Changed files

1. `develop.md`

### Validation

- `source .venv_sglang312/bin/activate && RUN_TS=$(date +%Y%m%d_%H%M%S) && CHECKPOINT_DIR=/tmp/deepscaler_ckpt_${RUN_TS} METRICS_LOG_DIR=/tmp/deepscaler_tb_${RUN_TS} ./examples/deepscaler/run_train.sh --rollout-engine sglang_jax --rollout-tp 2 --num-generations 4 --num-batches 1 --num-epochs 1 --max-steps 1`
- `ps -ef | rg 'train_deepscaler_nb|run_train.sh|sglang|python examples/deepscaler'`
- `python - <<'PY' ... EventAccumulator('/tmp/deepscaler_tb_20260308_011741') ... PY`

### Validation results

- `sglang_jax` extend/decode precompile completed successfully.
- The run reached rollout generation, reward evaluation, and actor training startup.
- The run failed on the actor train-step compile path with:
  - `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED`
  - `XLA:TPU compile permanent error`
  - `Ran out of memory in memory space sflag`
- The failure occurred in `partial_train_step(train_example)` after actor training began, not during rollout-engine precompile.
- TensorBoard output was created, but only compile-duration tags were present; no successful actor training scalar was committed before failure.

### Known risks / TODO

- This failure indicates the matched `num_generations=4` baseline is already too large for the current DeepScaler actor-step compile shape, even with only `1` batch and `1` optimizer step.
- The next levers to try should reduce actor-step compile pressure, e.g. lower `total_generation_steps`, lower `max_prompt_length`, or lower the actor-step trajectory count before comparing baseline and self-inf-group.

## 2026-03-08 - DeepScaler alternative sharding checks for num_generations=4

### Scope

- 无代码改动。
- Tested whether changing only the actor training mesh could preserve `num_generations=4` without changing `total_generation_steps`.

### Changed files

1. `develop.md`

### Validation

- `source .venv_sglang312/bin/activate && ... ./examples/deepscaler/run_train.sh --mesh-fsdp 1 --mesh-tp 4 --rollout-engine sglang_jax --rollout-tp 2 --num-generations 4 --num-batches 1 --num-epochs 1 --max-steps 1`
- `source .venv_sglang312/bin/activate && ... ./examples/deepscaler/run_train.sh --mesh-fsdp 4 --mesh-tp 1 --rollout-engine sglang_jax --rollout-tp 2 --num-generations 4 --num-batches 1 --num-epochs 1 --max-steps 1`

### Validation results

- `mesh=(1,4)` failed immediately during model loading with an invalid sharding error because one parameter had full shape `(1536, 2, 128)` and its dimension `2` is not divisible by `tp=4`.
- `mesh=(4,1)` was valid and progressed through rollout precompile and into actor training, but it failed at the same actor compile stage with a worse TPU `sflag` compile OOM:
  - previous `2x2` run: about `2.3K / 2.0K sflag`
  - `4x1` run: about `6.2K / 2.0K sflag`
- Conclusion: changing only the training mesh does not solve the matched `num_generations=4` baseline in this environment.

### Known risks / TODO

- The remaining viable levers, without changing `total_generation_steps`, are now mostly algorithmic or training-regime changes rather than simple sharding changes:
  - revert `num_generations`
  - change prompt count geometry
  - switch to LoRA / lighter trainable state
  - alter the comparison protocol rather than forcing the exact matched baseline shape

## 2026-03-08 - DeepScaler quantization / QLoRA feasibility review

### Scope

- 无代码改动。
- Reviewed whether the current `examples/deepscaler` training path can use quantization to mitigate the `num_generations=4` actor compile HBM/sflag issue without changing `total_generation_steps`.

### Changed files

1. `develop.md`

### Validation

- `rg -n "train-with-lora|lora-rank|lora-alpha|weight_qtype|tile_size|LoraProvider|qwix|quant|qlora|nf4|int8" examples/deepscaler tunix tests -S`
- `sed -n '470,520p' examples/deepscaler/train_deepscaler_nb.py`
- `sed -n '640,760p' examples/deepscaler/train_deepscaler_nb.py`
- `sed -n '840,930p' examples/deepscaler/train_deepscaler_nb.py`
- `sed -n '1,140p' tunix/cli/base_config.yaml`
- `sed -n '1,220p' tunix/cli/utils/model.py`
- `sed -n '55,95p' tests/cli/utils/model_test.py`

### Validation results

- `examples/deepscaler/train_deepscaler_nb.py` currently exposes only standard LoRA flags:
  - `--train-with-lora`
  - `--lora-rank`
  - `--lora-alpha`
- Its `get_lora_model(...)` helper constructs `qwix.LoraProvider` with only `module_path`, `rank`, and `alpha`; it does not pass quantization-related kwargs such as `weight_qtype` or `tile_size`.
- The repository does have a QLoRA-style path in the generic CLI stack:
  - `tunix/cli/base_config.yaml` defines `lora_config.weight_qtype: "nf4"` and `tile_size: 256`
  - `tunix/cli/utils/model.py` forwards `weight_qtype` and `tile_size` into `qwix.LoraProvider`
  - `tests/cli/utils/model_test.py` covers a quantized case using `weight_qtype: "int8"`
- Conclusion: quantized LoRA support exists in the repo in general, but it is not wired into the current DeepScaler example entrypoint.

### Known risks / TODO

- The only no-code-change lever already available in DeepScaler is `--train-with-lora`; true QLoRA for this path would require a new branch in the DeepScaler LoRA model creation path.
- Even with quantized LoRA, this may reduce trainable-state / weight representation pressure but does not guarantee the actor train-step compile issue disappears; the failure was in actor-step TPU compile (`sflag`), not rollout initialization.

## 2026-03-08 - DeepScaler LoRA sanity check under sglang_jax

### Scope

- 无代码改动。
- Tested the only currently exposed lightweight DeepScaler training path (`--train-with-lora`) to see whether it can serve as an immediate substitute for quantized training under `num_generations=4`.

### Changed files

1. `develop.md`

### Validation

- `source .venv_sglang312/bin/activate && ... ./examples/deepscaler/run_train.sh --rollout-engine sglang_jax --rollout-tp 2 --num-generations 4 --num-batches 1 --num-epochs 1 --max-steps 1 --train-with-lora`

### Validation results

- The run did not reach the previous actor compile OOM point.
- It failed earlier during `RLCluster` rollout initialization with:
  - `RuntimeError: sglang_jax mappings not available for Qwen2.`
- This indicates that the currently exposed DeepScaler LoRA path is not yet compatible with the `sglang_jax` rollout setup used by the DeepScaler example.

### Known risks / TODO

- As of this check, `--train-with-lora` is not an immediate no-code workaround for the `num_generations=4` DeepScaler configuration under `sglang_jax`.
- If quantized LoRA is desired here, the practical path is still code changes in the DeepScaler entrypoint, likely together with rollout/model-mapping compatibility work for the LoRA-wrapped actor.

## 2026-03-08 - DeepScaler built-in dtype / quant knob analysis

### Scope

- 无代码改动。
- Reviewed only the existing DeepScaler script knobs related to dtype / quantization-like settings, without using LoRA and without code changes.

### Changed files

1. `develop.md`

### Validation

- `rg -n "dtype|bf16|float16|fp8|auto|reward_advantage|train_model_dtype|rollout.*dtype|kv-cache" examples/deepscaler/train_deepscaler_nb.py examples/deepscaler/run_train.sh -S`
- `sed -n '600,730p' examples/deepscaler/train_deepscaler_nb.py`
- `sed -n '730,860p' examples/deepscaler/train_deepscaler_nb.py`
- `sed -n '1,140p' examples/deepscaler/run_train.sh`

### Validation results

- The current DeepScaler script exposes these precision knobs:
  - `--train-dtype fp32|bf16`
  - `--reward-advantage-dtype fp32|bf16`
  - `--rollout-sglang-jax-dtype auto|float32|bfloat16|float16`
  - `--rollout-sglang-jax-kv-cache-dtype auto|bf16|fp8_e5m2|fp8_e4m3`
- Only `rollout-sglang-jax-kv-cache-dtype=fp8_*` is an actual lower-precision / quantization-like setting in the current no-code DeepScaler path.
- The actor/reference training path does not expose int8/nf4/fp8 weight quantization; it is limited to `fp32` or `bf16`, and the shell default already uses `bf16`.
- Therefore, the current no-code dtype levers mostly help rollout memory, not the actor train-step compile path that previously failed.

### Known risks / TODO

- Because the observed failure was in the actor train-step TPU compile (`sflag`), changing only rollout dtype or rollout KV-cache dtype is unlikely to resolve the core issue.
- The only actor-side precision improvement already available without code changes is ensuring `--train-dtype bf16` and `--reward-advantage-dtype bf16`, which are already the shell defaults.

## 2026-03-08 - DeepScaler rollout vs actor resource-control knob review

### Scope

- 无代码改动。
- Reviewed which existing DeepScaler flags actually change rollout-vs-actor resource usage, versus those that only change rollout throughput.

### Changed files

1. `develop.md`

### Validation

- `rg -n "mesh-fsdp|mesh-tp|rollout-tp|rollout-dp|grpo-max-concurrency|rollout-prompt-batch-size|fast-path|offload-to-cpu|colocated|share" examples/deepscaler/train_deepscaler_nb.py tunix/rl tunix/generate -S`
- `rg -n "mem_fraction_static|rollout_sglang_jax_mem_fraction_static|hbm_utilization|swap_space|rollout_tp_override|create_device_mesh\\(|role_to_mesh" examples/deepscaler/train_deepscaler_nb.py tunix/rl tunix/generate -S`
- `sed -n '190,260p' examples/deepscaler/train_deepscaler_nb.py`
- `sed -n '360,460p' examples/deepscaler/train_deepscaler_nb.py`
- `sed -n '560,640p' examples/deepscaler/train_deepscaler_nb.py`
- `sed -n '770,810p' examples/deepscaler/train_deepscaler_nb.py`
- `sed -n '220,320p' tunix/rl/rl_cluster.py`

### Validation results

- The closest existing "rollout vs actor proportion" knobs for the current `sglang_jax` path are:
  - `--rollout-sglang-jax-mem-fraction-static` with default `0.2`
  - `--rollout-tp`
- `--rollout-sglang-jax-mem-fraction-static` controls how much static memory the `sglang_jax` rollout side reserves.
- `--rollout-tp` controls how many devices are used in the rollout mesh; DeepScaler builds the rollout mesh from a subset of the training-mesh devices.
- `--mesh-fsdp` and `--mesh-tp` change the actor/reference training mesh geometry, not a direct rollout/actor split ratio.
- `--rollout-prompt-batch-size` and `--grpo-max-concurrency` mainly affect rollout pressure / throughput, not the actor train-step compile shape.

### Known risks / TODO

- These rollout-side knobs can reduce rollout memory pressure, but the previously observed failure was in the actor train-step TPU compile (`sflag`), so they may not fix the core issue by themselves.
- If a minimal no-code experiment is desired, the lowest-risk rollout-side levers are lowering `--rollout-sglang-jax-mem-fraction-static` and possibly lowering `--rollout-prompt-batch-size`.

## 2026-03-08 - DeepScaler rollout memory-fraction / prompt-batch experiment

### Scope

- 无代码改动。
- Tested whether reducing rollout-side static memory reservation and rollout prompt batch size helps the `num_generations=4` DeepScaler actor compile failure.

### Changed files

1. `develop.md`

### Validation

- `source .venv_sglang312/bin/activate && ... ./examples/deepscaler/run_train.sh --rollout-engine sglang_jax --rollout-tp 2 --num-generations 4 --rollout-sglang-jax-mem-fraction-static 0.1 --rollout-prompt-batch-size 2 --num-batches 1 --num-epochs 1 --max-steps 1`

### Validation results

- The run completed `sglang_jax` extend/decode precompile and entered rollout/reward computation.
- It also reached `Actor Training: 0/1`, so the reduced rollout settings did not break the training pipeline.
- However, the run still failed in the same actor train-step compile location:
  - `partial_train_step(train_example)`
  - `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED`
  - `Ran out of memory in memory space sflag`
- The reported compile pressure was effectively unchanged from the previous baseline:
  - `Used 2.3K of 2.0K sflag`
  - exceeded by `344B`

### Known risks / TODO

- Lowering `rollout_sglang_jax_mem_fraction_static` from `0.2` to `0.1` and lowering rollout prompt batch size from `4` to `2` did not change the actor compile bottleneck.
- This suggests the core issue is actor-step compile shape rather than rollout-side reserved memory.

## 2026-03-08 - DeepScaler rollout_tp=1 experiment

### Scope

- 无代码改动。
- Tested whether reducing the `sglang_jax` rollout mesh from two devices to one device changes the `num_generations=4` actor compile failure.

### Changed files

1. `develop.md`

### Validation

- `source .venv_sglang312/bin/activate && ... ./examples/deepscaler/run_train.sh --rollout-engine sglang_jax --rollout-tp 1 --num-generations 4 --rollout-sglang-jax-mem-fraction-static 0.1 --rollout-prompt-batch-size 2 --num-batches 1 --num-epochs 1 --max-steps 1`

### Validation results

- The run used `rollout mesh shape: (1, 1)` and completed `sglang_jax` extend/decode precompile successfully.
- It progressed through rollout/reward computation and reached `Actor Training: 0/1`.
- It then failed at the same actor train-step compile point with the same TPU `sflag` compile OOM:
  - `partial_train_step(train_example)`
  - `Used 2.3K of 2.0K sflag`
  - exceeded by `344B`
- Therefore, reducing rollout device count did not materially change the failure mode.

### Known risks / TODO

- The current evidence indicates that shrinking rollout-side resource usage is not enough to make `num_generations=4 + total_generation_steps=7680` fit in the current actor compile shape.
- The next viable levers are likely actor-side or algorithm-shape changes rather than further rollout-side resource tuning.

## 2026-03-08 - DeepScaler next-step recommendation after rollout-side experiments

### Scope

- 无代码改动。
- Summarized the practical next steps after confirming that rollout-side resource knobs do not change the `num_generations=4` actor compile failure.

### Changed files

1. `develop.md`

### Validation

- No new command was run in this step.
- Recommendation is based on the earlier `num_generations=4` DeepScaler runs, including:
  - baseline compile failure
  - alternative mesh checks
  - rollout memory-fraction / prompt-batch reduction
  - `rollout_tp=1`

### Validation results

- All tested rollout-side reductions preserved rollout precompile and training-pipeline progress, but none changed the actor train-step compile error.
- The compile failure remained at the same location with the same TPU `sflag` pressure (`2.3K / 2.0K`, exceeded by `344B`).
- Therefore, further rollout-side tuning is unlikely to solve the problem.

### Known risks / TODO

- The remaining promising no-code lever is actor-shape reduction that does not change `total_generation_steps`, e.g. reducing `max_prompt_length`.
- If `num_generations=4` must be kept and actor-shape reduction still fails, the likely conclusion is that this setup is not feasible in the current full-finetune DeepScaler path without changing the training regime or adding new code support.

## 2026-03-08 - DeepScaler official-8k constraint assessment

### Scope

- 无代码改动。
- Re-assessed the remaining options under the stricter user constraint:
  - keep official 8k-aligned sequence setting
  - keep full finetune
  - keep `num_generations=4`
  - no code changes

### Changed files

1. `develop.md`

### Validation

- No new command was run in this step.
- Recommendation is based on all earlier DeepScaler runs and on the current script defaults / argument semantics.

### Validation results

- Under these constraints, the actor-side single-step shape is already near its minimum:
  - `train_micro_batch_size=1` cannot be reduced further
  - `num_generations=4` is fixed by experiment design
  - the official 8k-aligned prompt+generation length is fixed by user requirement
- The previously tested no-code levers have all failed to change the actor compile bottleneck:
  - rollout-side memory fraction
  - rollout prompt batch size
  - rollout TP
  - training mesh reshaping
  - existing dtype knobs
- Therefore, there is no credible remaining no-code path in the current DeepScaler full-finetune script to make this exact setup fit.

### Known risks / TODO

- If the exact `official-8k + full-finetune + num_generations=4` setup must be preserved, the next step necessarily moves into code or regime changes rather than shell-level tuning.
- The most practical fallback without code changes remains reverting the comparison to `num_generations=2`.

## 2026-03-08 - DeepScaler DBC choice under num_generations=2

### Scope

- 无代码改动。
- Re-evaluated which DBC variant is worth testing after reverting DeepScaler back to `num_generations=2`.

### Changed files

1. `develop.md`

### Validation

- `sed -n '500,560p' tunix/rl/rl_cluster.py`
- `sed -n '1,180p' tunix/rl/self_inf_trainer.py`
- `sed -n '1,140p' tunix/rl/robust_trainer.py`

### Validation results

- `rl_cluster.py` selects:
  - `SelfInfTrainer` only when `use_dynamic_batch_curation` is enabled and `TUNIX_DBC_VARIANT=self_inf`
  - otherwise `RobustTrainer` for standard DBC
- Under the current DeepScaler default geometry:
  - `train_micro_batch_size=1`
  - `num_generations=2`
  - actor-step DBC window size is `2`
- In this geometry:
  - `outlier-l2` has very weak filtering power because it computes `cutoff = mean + threshold * std` over only two samples
  - `self-inf-batch` and `self-inf-group` collapse to the same behavior, because group scope reshapes the two trajectories into exactly one group of size `2`
- Therefore, the only meaningful experiment pair is:
  - `baseline`
  - one self-influence variant (preferably labeled `self-inf-group` for continuity with prior experiments)

### Known risks / TODO

- With only two trajectories per actor step, even self-influence curation is testing a very local signal: agreement between the two generations from the same prompt.
- Running both `self-inf-batch` and `self-inf-group` would be redundant in the current geometry.

## 2026-03-08 - Self-inf-group filtering behavior at group size 2

### Scope

- 无代码改动。
- Clarified whether `self-inf-group` can actually filter samples when `num_generations=2`, i.e. each GRPO group contains only two trajectories.

### Changed files

1. `develop.md`

### Validation

- No new command was run in this step.
- Analysis is based on the existing `SelfInfTrainer` implementation already inspected in `tunix/rl/self_inf_trainer.py`.

### Validation results

- `self-inf-group` computes each sample score as the dot product between that sample gradient and its group-mean gradient.
- With group size `2`, if the two per-sample gradients are `g1` and `g2`, then the group mean is:
  - `m = (g1 + g2) / 2`
- The two keep/drop scores become:
  - `score1 = g1 · m = (||g1||^2 + g1·g2) / 2`
  - `score2 = g2 · m = (||g2||^2 + g1·g2) / 2`
- Because the default threshold is `0.0`, a sample is dropped only when its score is negative.
- Therefore, filtering is still possible with group size `2`, but it requires strong anti-alignment between the two gradients, especially when their norms are similar.
- If the two gradients have equal norm, both scores are always non-negative, so no filtering happens.
- In practice, with group size `2`, `self-inf-group` behaves more like a detector for highly contradictory generation pairs than a broad batch-curation mechanism.

### Known risks / TODO

- Under the current DeepScaler geometry, `self-inf-group` can filter, but the skip rate may be low unless the two generations for a prompt produce sharply conflicting gradients.
- This reinforces the recommendation to compare only `baseline` vs one self-influence variant, and to inspect `skipped_samples` / `self_inf_kept_fraction` rather than assuming the curation is active.

## 2026-03-08 - DeepScaler self-inf-group command feasibility check

### Scope

- 无代码改动。
- Verified whether the current `examples/deepscaler` entrypoint can enable `self-inf-group` purely via command-line arguments or environment variables.

### Changed files

1. `develop.md`

### Validation

- `rg -n "use_dynamic_batch_curation|curation_threshold|TUNIX_DBC_VARIANT|TUNIX_DBC_SELF_INF_SCOPE|TUNIX_GRPO_NUM_GENERATIONS|SelfInfTrainer|RobustTrainer" examples/deepscaler/train_deepscaler_nb.py my_example/train.py my_example/config.py tunix/rl/rl_cluster.py -S`
- `sed -n '900,980p' examples/deepscaler/train_deepscaler_nb.py`
- `sed -n '100,140p' my_example/train.py`
- `sed -n '150,180p' my_example/config.py`

### Validation results

- `examples/deepscaler/train_deepscaler_nb.py` currently does not expose DBC CLI flags.
- Its `RLTrainingConfig(...)` construction does not pass:
  - `use_dynamic_batch_curation`
  - `curation_threshold`
- `rl_cluster.py` only selects `SelfInfTrainer` when both conditions are true:
  - `use_dynamic_batch_curation=True`
  - `TUNIX_DBC_VARIANT=self_inf`
- Therefore, setting only environment variables is insufficient for DeepScaler today; the entrypoint never turns on `use_dynamic_batch_curation`.

### Known risks / TODO

- The current DeepScaler example cannot run `self-inf-group` by command alone.
- To make the comparison runnable, the DeepScaler entrypoint needs additional wiring similar to `my_example`, but this requires code changes.

## 2026-03-08 - DeepScaler DBC CLI wiring and agentic GRPO compatibility fixes

### Scope

- 代码改动。
- Implemented `self-inf-group` CLI wiring for `examples/deepscaler`.
- Added the minimum agentic GRPO compatibility fixes required for DeepScaler DBC to run end-to-end.
- Added a DeepScaler README example for the new DBC command.

### Changed files

1. `examples/deepscaler/train_deepscaler_nb.py`
2. `tunix/rl/experimental/agentic_grpo_learner.py`
3. `examples/deepscaler/README.md`
4. `develop.md`

### Validation

- `source .venv_sglang312/bin/activate && python -m py_compile examples/deepscaler/train_deepscaler_nb.py tunix/rl/experimental/agentic_grpo_learner.py`
- `source .venv_sglang312/bin/activate && python examples/deepscaler/train_deepscaler_nb.py --help | rg "use-dynamic-batch-curation|use-dbc-outlier-l2|use-dbc-self-inf-batch|use-dbc-self-inf-group|curation-threshold" -n`
- `source .venv_sglang312/bin/activate && python examples/deepscaler/train_deepscaler_nb.py --use-dbc-self-inf-batch --use-dbc-self-inf-group`
- `source .venv_sglang312/bin/activate && python examples/deepscaler/train_deepscaler_nb.py --model-path <local model> --train-dataset-path <local train data> --test-dataset-path <local test data> --checkpoint-dir /tmp/deepscaler_ckpt_20260308_035419 --metrics-log-dir /tmp/deepscaler_tb_20260308_035419 --mesh-fsdp 2 --mesh-tp 2 --rollout-engine vanilla --smoke-test --use-dynamic-batch-curation --use-dbc-self-inf-group`
- `source .venv_sglang312/bin/activate && python - <<'PY' ... EventAccumulator('/tmp/deepscaler_tb_20260308_035419') ... PY`
- `source .venv_sglang312/bin/activate && python examples/deepscaler/train_deepscaler_nb.py --model-path <local model> --train-dataset-path <local train data> --test-dataset-path <local test data> --checkpoint-dir /tmp/deepscaler_ckpt_20260308_035838 --metrics-log-dir /tmp/deepscaler_tb_20260308_035838 --mesh-fsdp 2 --mesh-tp 2 --rollout-engine vanilla --smoke-test`

### Validation results

- `examples/deepscaler/train_deepscaler_nb.py` now exposes DBC flags and enforces expected mutual exclusion for self-inf variants.
- DeepScaler now forwards:
  - `use_dynamic_batch_curation`
  - `curation_threshold`
  into `RLTrainingConfig`.
- DeepScaler now sets and clears the self-inf environment variables needed by `rl_cluster.py`:
  - `TUNIX_DBC_VARIANT`
  - `TUNIX_DBC_SELF_INF_SCOPE`
  - `TUNIX_GRPO_NUM_GENERATIONS`
- `agentic_grpo_learner.py` was aligned with standard GRPO behavior by:
  - passing only `train_example` into the actor trainer input dict
  - normalizing single-sample tensors with `jnp.atleast_2d/1d` in agentic GRPO loss
  - registering DBC-related metrics in `with_rl_metrics_to_log(...)`
- The DeepScaler DBC smoke run completed successfully with exit code `0`.
- The resulting event file contained the expected self-inf metrics:
  - `actor/train/self_inf_dot_mean`
  - `actor/train/self_inf_dot_std`
  - `actor/train/self_inf_kept_fraction`
  - `actor/train/skipped_samples`
- A matching vanilla baseline smoke run also completed successfully with exit code `0`, confirming the non-DBC path still works after the compatibility fixes.

### Known risks / TODO

- `examples/deepscaler/run_train.sh` still unconditionally passes `--enable-rollout-fast-path`, so using that wrapper with `--rollout-engine vanilla` remains incompatible; validation therefore used direct `train_deepscaler_nb.py` invocations for vanilla smoke.
- The new DeepScaler CLI also exposes outlier-L2 flags, but this turn only fully validated the `self-inf-group` path that the user requested.

## 2026-03-08 - DeepScaler standard wrapper self-inf-group sanity check

### Scope

- 无代码改动。
- Performed an additional sanity check using the actual user-facing wrapper command shape (`run_train.sh` with `sglang_jax`) after the DBC fixes landed.

### Changed files

1. `develop.md`

### Validation

- `source .venv_sglang312/bin/activate && RUN_TS=$(date +%Y%m%d_%H%M%S) && CHECKPOINT_DIR=/tmp/deepscaler_ckpt_${RUN_TS} METRICS_LOG_DIR=/tmp/deepscaler_tb_${RUN_TS} ./examples/deepscaler/run_train.sh --smoke-test --use-dynamic-batch-curation --use-dbc-self-inf-group`

### Validation results

- The wrapper path successfully passed:
  - argument parsing
  - model / dataset loading
  - `sglang_jax` extend precompile
  - `sglang_jax` decode precompile
  - rollout / reward computation
  - entry into actor training
- The run was then manually interrupted to avoid executing the full `315` steps currently implied by the wrapper defaults.
- This interruption was intentional; it does not indicate a DBC regression.

### Known risks / TODO

- `run_train.sh --smoke-test` still inherits `MAX_STEPS=315` from the wrapper defaults, so it is not currently a true one-step smoke through the wrapper path.
- The strongest completion proof remains the successful direct `train_deepscaler_nb.py --rollout-engine vanilla --smoke-test` runs, which verified both:
  - `self-inf-group` end-to-end completion
  - baseline end-to-end completion

## 2026-03-09 - GitHub sync status check

### Scope

- 无代码改动。
- Checked whether the current working tree contents are fully reflected on GitHub for the active branch.

### Changed files

1. `develop.md`

### Validation

- `git fetch origin`
- `git rev-list --left-right --count origin/my-changes...HEAD`
- `git status --short`
- `git log --oneline --decorate -n 5 HEAD`

### Validation results

- `HEAD` and `origin/my-changes` are at the same commit: `1a357f7c15a1a6ad3582a97dc983698d414a6405`.
- The working tree is not clean, so the current local contents are not fully on GitHub.
- Modified tracked files currently not pushed as committed content:
  - `develop.md`
  - `examples/deepscaler/README.md`
  - `examples/deepscaler/train_deepscaler_nb.py`
  - `tunix/rl/experimental/agentic_grpo_learner.py`
- Untracked paths also exist locally and are not on GitHub unless added and pushed later.

### Known risks / TODO

- Any conclusion about "already synced to GitHub" only applies to committed history on `origin/my-changes`; local uncommitted and untracked content remains outside GitHub.

## 2026-03-09 - DeepScaler max prompt length behavior analysis

### Scope

- 无代码改动。
- Analyzed what happens in `examples/deepscaler` training when a prompt exceeds `--max-prompt-length`, and whether skipping such samples is feasible.

### Changed files

1. `develop.md`

### Validation

- `rg -n "max-prompt-length|max_prompt_length|prompt length|prompt_length" examples/deepscaler tunix -S`
- `sed -n '458,490p' examples/deepscaler/train_deepscaler_nb.py`
- `sed -n '742,800p' examples/deepscaler/train_deepscaler_nb.py`
- `sed -n '896,910p' examples/deepscaler/train_deepscaler_nb.py`
- `sed -n '695,718p' tunix/generate/sampler.py`
- `sed -n '25,55p' tunix/rl/agentic/utils.py`

### Validation results

- The DeepScaler training entrypoint does not currently filter overlong prompts before batching.
- Overlong prompts can reach rollout, where sampler/cache constraints may fail first.
- If execution proceeds, the training-side agentic padding path truncates prompts from the left to `max_prompt_length`.
- Skipping such samples is feasible by adding a DeepScaler-specific filtering branch before batching, without changing baseline CLI structure.

### Known risks / TODO

- Any future fix should preserve the existing baseline command behavior and only add a branch for the new filtering behavior, per repo instructions.

## 2026-03-09 - DeepScaler skip overlong prompts

### Scope

- Added a DeepScaler-specific data filtering branch so prompts longer than `--max-prompt-length` are skipped before batching and rollout.
- Kept the existing CLI shape and did not touch `tunix/rl/robust_trainer.py` or `RLTrainingConfig`.

### Changed files

1. `examples/deepscaler/train_deepscaler_nb.py`
2. `develop.md`

### Validation

- `python -m py_compile examples/deepscaler/train_deepscaler_nb.py`
- `git diff -- examples/deepscaler/train_deepscaler_nb.py`
- `nl -ba examples/deepscaler/train_deepscaler_nb.py | sed -n '458,525p'`

### Validation results

- Syntax check passed.
- `create_datasets()` now computes each prompt's token length after chat templating, filters out rows whose `prompt_length` exceeds `max_prompt_length`, prints skip counts, removes the temporary `prompt_length` column, and only then converts the data to `grain.MapDataset`.
- `run_training()` now passes `runtime["max_prompt_length"]` into `create_datasets()`, so the filter follows the effective runtime value, including smoke-test clamping.

### Known risks / TODO

- I did not run a full DeepScaler training smoke after this change; only syntax and local diff validation were completed.
- If a dataset split is fully filtered out by a very small `max_prompt_length`, downstream training/eval may still fail due to empty batches.

## 2026-03-09 - DeepScaler prompt filter scope clarification

### Scope

- 无代码改动。
- Clarified when the DeepScaler overlong-prompt filter is active and which runtime knobs affect it.

### Changed files

1. `develop.md`

### Validation

- `nl -ba examples/deepscaler/train_deepscaler_nb.py | sed -n '458,518p'`
- `nl -ba examples/deepscaler/train_deepscaler_nb.py | sed -n '742,777p'`
- `nl -ba examples/deepscaler/train_deepscaler_nb.py | sed -n '922,927p'`

### Validation results

- The filter is active in the standard `run_training()` path whenever `max_prompt_length` is set to a positive value.
- In the current CLI, that means it applies for normal runs and smoke runs; smoke mode only changes the effective threshold by clamping it to at most `512`.
- The filter is independent of DBC flags, rollout engine choice, and most other training hyperparameters.

### Known risks / TODO

- A custom caller that bypasses `run_training()` or passes `max_prompt_length=None` / `<=0` into `create_datasets()` would bypass the filter.

## 2026-03-09 - DeepScaler overlong-prompt smoke test

### Scope

- 无代码改动。
- Ran a smoke test to validate that the new overlong-prompt filtering branch executes in the real DeepScaler training path.

### Changed files

1. `develop.md`

### Validation

- Prompt-length distribution probe:
  - `source .venv_sglang312/bin/activate && python - <<'PY' ... PY`
- Direct CPU smoke command:
  - `source .venv_jax081/bin/activate && export JAX_PLATFORMS=cpu && export TOKENIZERS_PARALLELISM=false && python -u examples/deepscaler/train_deepscaler_nb.py --model-path /home/lhf_hongfu_gmail_com/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 --train-dataset-path /home/lhf_hongfu_gmail_com/.cache/huggingface/hub/datasets--agentica-org--DeepScaleR-Preview-Dataset/snapshots/b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29/deepscaler.json --test-dataset-path /home/lhf_hongfu_gmail_com/.cache/huggingface/hub/datasets--HuggingFaceH4--aime_2024/snapshots/2fe88a2f1091d5048c0f36abc874fb997b3dd99a/data/train-00000-of-00001.parquet --checkpoint-dir /tmp/deepscaler_ckpt_filter_smoke_$(date +%Y%m%d_%H%M%S) --metrics-log-dir /tmp/deepscaler_tb_filter_smoke_$(date +%Y%m%d_%H%M%S) --mesh-fsdp 1 --mesh-tp 1 --rollout-engine vanilla --batch-size 1 --mini-batch-size 1 --train-micro-batch-size 1 --num-batches 1 --num-test-batches 1 --num-epochs 1 --max-steps 1 --eval-every-n-steps 1 --num-generations 2 --max-prompt-length 128 --total-generation-steps 8 --top-p 1.0 --top-k 50 --smoke-test`

### Validation results

- TPU-backed smoke could not be used in this session because TPU initialization failed in both local environments:
  - one path reported `libtpu` multi-process lockfile initialization failure
  - another path reported `/dev/vfio/0` busy
- CPU fallback smoke completed successfully with exit code `0`.
- The new filter branch executed in the real training path and printed:
  - `Filtered overlong prompts (max_prompt_length=128): train skipped 7907/40315, test skipped 10/30.`
- The run progressed through:
  - dataset preprocessing
  - overlong prompt filtering
  - model loading
  - rollout / actor training initialization
  - one actor training step
  - normal process exit
- An earlier CPU smoke attempt failed before completion because `--num-generations 1` violates the GRPO config requirement (`num_generations > 1`).
- Another earlier CPU smoke attempt reached rollout and exposed an unrelated vanilla sampling issue with `top_k=-1`; switching to `--top-k 50` resolved that for the smoke path.

### Known risks / TODO

- This validation used a CPU fallback and a deliberately reduced geometry (`mesh 1x1`, `batch-size 1`, `total-generation-steps 8`, `max_prompt_length 128`), so it proves the filter branch and end-to-end control flow, not TPU performance characteristics.
- `run_train.sh` still defaults `TOP_K=-1`; for direct vanilla rollout smoke in the current code, that value is not accepted by the vanilla sampler.

## 2026-03-09 - Pending push status check

### Scope

- 无代码改动。
- Checked which local changes are currently pending commit/push.

### Changed files

1. `develop.md`

### Validation

- `git status --short --branch`
- `git diff --name-only`
- `git ls-files --others --exclude-standard`

### Validation results

- Modified tracked files pending commit/push:
  - `develop.md`
  - `examples/deepscaler/README.md`
  - `examples/deepscaler/train_deepscaler_nb.py`
  - `tunix/rl/experimental/agentic_grpo_learner.py`
- Untracked content also exists locally, including virtual environments, local model/data directories, and test files.

### Known risks / TODO

- The untracked virtual environment and local artifact directories are large and should usually not be committed.

## 2026-03-09 - DeepScaler wrapper total generation steps default

### Scope

- Updated the `examples/deepscaler/run_train.sh` wrapper default `TOTAL_GENERATION_STEPS` from `7680` to `8192` to match the Python entrypoint default.

### Changed files

1. `examples/deepscaler/run_train.sh`
2. `develop.md`

### Validation

- `nl -ba examples/deepscaler/run_train.sh | sed -n '20,30p'`

### Validation results

- `run_train.sh` now defaults `TOTAL_GENERATION_STEPS` to `8192`.

### Known risks / TODO

- This change updates the wrapper default only; callers that explicitly export `TOTAL_GENERATION_STEPS` or pass `--total-generation-steps` still override it.

## 2026-03-10 - DeepScaler G=2/4/8 training-time scaling estimate

### Scope

- 无代码改动。
- Estimated wall-clock training-time multipliers for DeepScaler when only `NUM_GENERATIONS` changes and other parameters stay fixed.

### Changed files

1. `develop.md`

### Validation

- `nl -ba examples/deepscaler/run_train.sh | sed -n '19,30p'`
- `nl -ba tunix/rl/experimental/agentic_rl_learner.py | sed -n '302,392p'`
- `nl -ba tunix/rl/experimental/agentic_rl_learner.py | sed -n '844,912p'`

### Validation results

- Wrapper defaults remain:
  - `ROLLOUT_PROMPT_BATCH_SIZE=4`
  - `NUM_GENERATIONS=2`
  - `TRAIN_MICRO_BATCH_SIZE=1`
  - `NUM_BATCHES=315`
- Fast-path rollout expands each prompt into `rollout_prompt_batch_size * num_generations` requests.
- Actor training consumes `train_micro_batch_size * num_generations` sequences per logical micro-batch, with optional chunking only reducing peak memory, not total sequence count.
- Therefore, with prompt/completion lengths held roughly constant, total training compute scales approximately linearly with `NUM_GENERATIONS`.

### Known risks / TODO

- This is an engineering estimate based on code-path work scaling, not a completed timing benchmark for full `G=4` or `G=8` runs.
- Real wall time can exceed linear scaling if memory pressure lowers throughput or triggers rollout-side instability.

## 2026-03-10 - DeepScaler DBC command clarification

### Scope

- 无代码改动。
- Confirmed which Dynamic Batch Curation flags are wired into the DeepScaler entrypoint and recorded a default command that keeps `ROLLOUT_PROMPT_BATCH_SIZE` at its wrapper default.

### Changed files

1. `develop.md`

### Validation

- `rg -n "use_dynamic_batch_curation|use-dbc|curation_threshold" -S examples/deepscaler tunix/rl`
- `nl -ba examples/deepscaler/README.md | sed -n '117,144p'`
- `nl -ba examples/deepscaler/train_deepscaler_nb.py | sed -n '729,766p'`

### Validation results

- DeepScaler entrypoint supports:
  - `--use-dynamic-batch-curation`
  - `--use-dbc-outlier-l2`
  - `--use-dbc-self-inf-batch`
  - `--use-dbc-self-inf-group`
- Self-influence variants are mutually exclusive, and they cannot be combined with `--use-dbc-outlier-l2`.
- The DeepScaler README currently documents `--use-dbc-self-inf-group` as the DBC command on the default geometry.
- Omitting `ROLLOUT_PROMPT_BATCH_SIZE` keeps the wrapper default at `4`.

### Known risks / TODO

- No end-to-end DBC smoke was run in this step; this entry only confirms the supported CLI and default wrapper behavior.

## 2026-03-10 - DeepScaler G=8 command note

### Scope

- 无代码改动。
- Recorded the recommended `NUM_GENERATIONS=8` launch command for the DeepScaler path.

### Changed files

1. `develop.md`

### Validation

- `nl -ba examples/deepscaler/run_train.sh | sed -n '19,30p'`
- `nl -ba examples/deepscaler/README.md | sed -n '131,150p'`

### Validation results

- `run_train.sh` still defaults `ROLLOUT_PROMPT_BATCH_SIZE=4` when the env var is omitted.
- The actor-side chunking flag remains optional and can be enabled via `ACTOR_GENERATION_CHUNK_SIZE`.
- For `NUM_GENERATIONS=8`, using actor-side chunking is the least invasive way to reduce actor-side peak memory without changing rollout geometry.

### Known risks / TODO

- This note does not prove `NUM_GENERATIONS=8` will fit with default rollout geometry; rollout-side memory pressure can still be the limiting factor.

## 2026-03-10 - DeepScaler wrapper defaults changed to G=8

### Scope

- Updated the DeepScaler shell wrapper so the user no longer needs to prefix the command with `NUM_GENERATIONS=8` and `ACTOR_GENERATION_CHUNK_SIZE=2`.
- Adjusted the README so the documented default geometry matches the wrapper.

### Changed files

1. `examples/deepscaler/run_train.sh`
2. `examples/deepscaler/README.md`
3. `develop.md`

### Validation

- `nl -ba examples/deepscaler/run_train.sh | sed -n '19,26p'`
- `bash -n examples/deepscaler/run_train.sh`
- `nl -ba examples/deepscaler/README.md | sed -n '108,152p'`

### Validation results

- `run_train.sh` now defaults to:
  - `NUM_GENERATIONS=8`
  - `ACTOR_GENERATION_CHUNK_SIZE=2`
  - `ROLLOUT_PROMPT_BATCH_SIZE=4` unchanged
- The wrapper still allows external env vars to override those defaults.
- README references to the wrapper default geometry were updated accordingly.

### Known risks / TODO

- This changes the behavior of invoking `./examples/deepscaler/run_train.sh` with no extra env vars; runs that previously defaulted to `G=2` will now default to `G=8`.
- The wrapper default rollout work profile is now much heavier than before, so rollout-side OOM remains possible.

## 2026-03-10 - DeepScaler wrapper defaults confirmation

### Scope

- 无代码改动。
- Confirmed whether `--rollout-engine sglang_jax` and `--rollout-tp 2` are already covered by wrapper defaults.

### Changed files

1. `develop.md`

### Validation

- `nl -ba examples/deepscaler/run_train.sh | sed -n '19,24p'`

### Validation results

- `run_train.sh` already defaults:
  - `ROLLOUT_ENGINE=sglang_jax`
  - `ROLLOUT_TP=2`
- Those CLI flags can be omitted unless the user wants to override the defaults.

### Known risks / TODO

- None for this clarification step.

## 2026-03-10 - DeepScaler DBC wrapper default confirmation

### Scope

- 无代码改动。
- Confirmed whether the DBC flags are baked into `examples/deepscaler/run_train.sh`.

### Changed files

1. `develop.md`

### Validation

- `nl -ba examples/deepscaler/run_train.sh | sed -n '75,117p'`

### Validation results

- `run_train.sh` always forwards rollout and training geometry flags from wrapper defaults.
- DBC flags such as `--use-dynamic-batch-curation` and `--use-dbc-self-inf-group` are not injected by the wrapper.
- Those DBC flags only reach the Python entrypoint if passed explicitly via the shell command tail (`"$@"`).

### Known risks / TODO

- If the user wants DBC to become the wrapper default, the shell wrapper needs an explicit branch or env-controlled passthrough for those flags.

## 2026-03-10 - Added dedicated DeepScaler DBC wrapper

### Scope

- Added a new `examples/deepscaler/run_train_dbc.sh` wrapper.
- Kept `examples/deepscaler/run_train.sh` untouched in this step.
- The new wrapper hardcodes `--use-dynamic-batch-curation` and `--use-dbc-self-inf-group` so those flags are always forwarded.

### Changed files

1. `examples/deepscaler/run_train_dbc.sh`
2. `develop.md`

### Validation

- `bash -n examples/deepscaler/run_train_dbc.sh`
- `diff -u examples/deepscaler/run_train.sh examples/deepscaler/run_train_dbc.sh`

### Validation results

- New wrapper syntax is valid.
- The new wrapper inherits the same geometry defaults as the current `run_train.sh`.
- The only functional addition is a fixed DBC arg bundle:
  - `--use-dynamic-batch-curation`
  - `--use-dbc-self-inf-group`

### Known risks / TODO

- If callers append a conflicting DBC variant via extra CLI args, the Python entrypoint will reject the combination.
- The new wrapper inherits the current default heavy geometry (`NUM_GENERATIONS=8`, `ROLLOUT_PROMPT_BATCH_SIZE=4`), so rollout-side memory pressure still applies.
