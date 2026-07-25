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

---

## 2026-03-10: DeepScaler first-step progress analysis

### Scope

- 无代码改动。
- 分析用户正在运行的 `examples/deepscaler/run_train.sh` 日志，估算第一个训练 step 在当前配置下的组成与大致进度。

### Changed files

1. `develop.md`

### Validation

- `sed -n '1,220p' examples/deepscaler/run_train.sh`
- `sed -n '1,260p' examples/deepscaler/README.md`
- `sed -n '900,980p' examples/deepscaler/train_deepscaler_nb.py`
- `sed -n '500,580p' tunix/rl/rl_cluster.py`

---

## 2026-03-17: Full-default UltraFeedback SFT -> DPO workflow

### Scope

- 为 `qwen2.5-1.5b` 的 `UltraFeedback` 两阶段 workflow 增加默认 `full`、可选 `lora` 的 stage-wise 分支。
- 打通 `full SFT -> exported safetensors -> full/lora DPO` 的模型交接。
- 保留现有 LoRA-only 导出兼容分支，不改数据切分语义。
- 新增一份端到端 README，明确 `sft` / `dpo` prompt-disjoint 切分和 DPO+DBC 对照命令。

### Changed files

1. `tunix/models/safetensors_saver.py`
2. `tunix/models/qwen2/params.py`
3. `tunix/cli/utils/model.py`
4. `tunix/cli/peft_main.py`
5. `tunix/cli/dpo_main.py`
6. `tests/cli/utils/model_test.py`
7. `tests/cli/peft_main_test.py`
8. `tests/cli/dpo_main_test.py`
9. `tests/models/qwen2/qwen_params_test.py`
10. `examples/sft/ultrafeedback/qwen2p5_1p5b_ultrafeedback.yaml`
11. `examples/sft/ultrafeedback/qwen2p5_1p5b_ultrafeedback_lora.yaml`
12. `examples/sft/ultrafeedback/run_qwen2p5_1p5b_ultrafeedback.sh`
13. `examples/sft/ultrafeedback/README.md`
14. `examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft.yaml`
15. `examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft_lora.yaml`
16. `examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh`
17. `examples/dpo/README.md`
18. `examples/ultrafeedback/README.md`
19. `tunix/cli/README.md`
20. `develop.md`

### Validation

- `python -m py_compile tunix/cli/peft_main.py tunix/cli/dpo_main.py tunix/cli/utils/model.py tunix/models/qwen2/params.py tunix/models/safetensors_saver.py tests/cli/peft_main_test.py tests/cli/dpo_main_test.py tests/cli/utils/model_test.py tests/models/qwen2/qwen_params_test.py`
  - 结果：通过。
- `bash -n examples/sft/ultrafeedback/run_qwen2p5_1p5b_ultrafeedback.sh`
  - 结果：通过。
- `bash -n examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh`
  - 结果：通过。
- 解析 4 份 YAML：
  - `examples/sft/ultrafeedback/qwen2p5_1p5b_ultrafeedback.yaml`
  - `examples/sft/ultrafeedback/qwen2p5_1p5b_ultrafeedback_lora.yaml`
  - `examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft.yaml`
  - `examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft_lora.yaml`
  - 结果：`yaml.safe_load` 全部通过。
- `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && python tests/cli/utils/model_test.py`
  - 结果：`11` 个测试通过。
- `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && python tests/cli/peft_main_test.py`
  - 结果：退出码 `0`。
- `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && python tests/cli/dpo_main_test.py`
  - 结果：`7` 个测试通过。
- `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && JAX_PLATFORMS=cpu python tests/models/qwen2/qwen_params_test.py`
  - 结果：`4` 个测试通过，包含新增 `full model round-trip`。
- 说明：
  - 当前环境没有 `pytest`，因此本轮针对这些 `absltest` 文件直接用 `python <test_file>.py` 执行。

### Known risks / TODO

- 还没有跑依赖 Hugging Face 下载的端到端 SFT/DPO smoke 训练；当前验证覆盖的是脚本入口、配置解析、导出逻辑和单元测试。
- `full model safetensors exporter` 当前只补到了 `Qwen2`，没有顺手扩展到全仓所有模型家族。
- `full DPO` 会独立加载 actor/reference 两份 base model，这保证了语义正确，但实际训练时的 HBM 峰值仍需要在真实 TPU worker 上进一步确认。

---

## 2026-03-17: UltraFeedback ratio and internal validation split

### Scope

- 将 qwen2.5 UltraFeedback recipe 的默认比例从 `0.5/0.5` 调整为 `0.25/0.75`。
- 在 `train_prefs` 内新增 deterministic prompt-level `train/eval` holdout，避免训练期直接使用 `test_prefs` 做 early-stop 或挑 checkpoint。
- 保持 `SFT` 与 `DPO` 的 prompt-disjoint 主切分不变，只增加第二层 `subset=train|eval|all` 过滤。

### Changed files

1. `tunix/examples/data/ultrafeedback_dpo.py`
2. `tunix/examples/data/ultrafeedback_sft.py`
3. `tests/examples/data/ultrafeedback_dpo_test.py`
4. `tests/examples/data/ultrafeedback_sft_test.py`
5. `examples/sft/ultrafeedback/qwen2p5_1p5b_ultrafeedback.yaml`
6. `examples/sft/ultrafeedback/qwen2p5_1p5b_ultrafeedback_lora.yaml`
7. `examples/sft/ultrafeedback/run_qwen2p5_1p5b_ultrafeedback.sh`
8. `examples/sft/ultrafeedback/README.md`
9. `examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft.yaml`
10. `examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft_lora.yaml`
11. `examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh`
12. `examples/dpo/README.md`
13. `examples/ultrafeedback/README.md`
14. `develop.md`

### Validation

- `python -m py_compile tunix/examples/data/ultrafeedback_dpo.py tunix/examples/data/ultrafeedback_sft.py tests/examples/data/ultrafeedback_dpo_test.py tests/examples/data/ultrafeedback_sft_test.py`
  - 结果：通过。
- `bash -n examples/sft/ultrafeedback/run_qwen2p5_1p5b_ultrafeedback.sh`
  - 结果：通过。
- `bash -n examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh`
  - 结果：通过。
- 解析 4 份 YAML：
  - `examples/sft/ultrafeedback/qwen2p5_1p5b_ultrafeedback.yaml`
  - `examples/sft/ultrafeedback/qwen2p5_1p5b_ultrafeedback_lora.yaml`
  - `examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft.yaml`
  - `examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft_lora.yaml`
  - 结果：`yaml.safe_load` 全部通过。
- `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && python tests/examples/data/ultrafeedback_dpo_test.py`
  - 结果：`6` 个测试通过，覆盖 `partition` 与 `subset` 的互斥性。
- `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && python tests/examples/data/ultrafeedback_sft_test.py`
  - 结果：退出码 `0`。

### Known risks / TODO

- 目前只是把 training-time eval 改成 `train_prefs` 内部 holdout；还没有新增一个专门的“最终 test-only 评测脚本”去自动在 `test_prefs` 上汇报最佳 checkpoint。
- `subset=train|eval` 同样是基于原始 prompt 字符串 hash 的严格切分；它能避免同 prompt 泄露，但不会做语义去重。

---

## 2026-03-12: DPO example environment validation

### Scope

- 无代码改动。
- 检查 `examples/dpo/README.md` 的运行前提是否已满足。
- 创建并验证小写虚拟环境 `/home/lhf_hongfu_gmail_com/.venvs/dpo`。
- 直接调用 `tunix.cli.dpo_main` 进行 DPO smoke 运行验证，绕过脚本中硬编码的大写环境路径。

### Changed files

1. `develop.md`

### Validation

- `python3 --version`
- `python3 -c "import jax; print(jax.__version__); print(jax.default_backend(), jax.device_count())"`
  - 结果：系统默认 Python 为 `3.10.12`，且未安装 `jax`，不能直接运行 DPO 示例。
- `python3.11 --version`
- `ls /dev/accel* /dev/vfio/* 2>/dev/null | head`
  - 结果：`python3.11` 可用；机器可见 `/dev/accel0..3`。
- `python3.11 -m venv /home/lhf_hongfu_gmail_com/.venvs/dpo`
- `source /home/lhf_hongfu_gmail_com/.venvs/dpo/bin/activate && python -m pip install -U pip && python -m pip install -e '.[dev]' && python -m pip install 'jax[tpu]==0.8.1'`
  - 结果：安装完成，最终环境包含 `jax 0.8.1`、`jaxlib 0.8.1`、`libtpu 0.0.30`。
- `source /home/lhf_hongfu_gmail_com/.venvs/dpo/bin/activate && python -c "import jax, flax; print('jax', jax.__version__); print('backend', jax.default_backend(), jax.device_count()); print('flax', flax.__version__)"`
  - 结果：`jax 0.8.1`，`backend tpu 4`，`flax 0.12.5`。
- `source /home/lhf_hongfu_gmail_com/.venvs/dpo/bin/activate && set -a && source /home/lhf_hongfu_gmail_com/tunix/my_example/.env && set +a && python -m tunix.cli.dpo_main /home/lhf_hongfu_gmail_com/tunix/examples/dpo/qwen3_4b_ultrafeedback.yaml "train_data_module=examples/data/ultrafeedback_dpo.py:create_dataset(split='train_prefs', limit=512, seed=42)" "eval_data_module=examples/data/ultrafeedback_dpo.py:create_dataset(split='test_prefs', limit=64, seed=42)" training_config.max_steps=20 training_config.eval_every_n_steps=10 training_config.gradient_accumulation_steps=8 training_config.checkpoint_root_directory=/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen3_4b_ultrafeedback_smoke/checkpoints training_config.checkpointing_options.save_interval_steps=250 training_config.checkpointing_options.max_to_keep=4 training_config.metrics_logging_options.project_name=tunix training_config.metrics_logging_options.run_name=qwen3-4b-ultrafeedback-dpo-smoke training_config.metrics_logging_options.log_dir=/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen3_4b_ultrafeedback_smoke/tensorboard training_config.metrics_logging_options.flush_every_n_steps=20 "training_config.data_sharding_axis=['fsdp']" training_config.max_inflight_computations=2 training_config.metrics_prefix=dpo training_config.pbar_description=DPO optimizer_config.opt_type=adamw optimizer_config.schedule_type=warmup_cosine_decay_schedule optimizer_config.init_value=0.0 optimizer_config.peak_value=5e-6 optimizer_config.end_value=0.0 optimizer_config.warmup_steps=2 optimizer_config.decay_steps=20 optimizer_config.b1=0.9 optimizer_config.b2=0.99 optimizer_config.weight_decay=0.1 optimizer_config.max_grad_norm=0.1 dpo_config.beta=0.01 dpo_config.label_smoothing=0.0 dpo_config.max_prompt_length=256 dpo_config.max_response_length=256 merged_model_output_dir=/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen3_4b_ultrafeedback_smoke/merged_lora`
  - 结果：成功读取配置、下载 `Qwen/Qwen3-4B-Instruct-2507` 权重与 tokenizer，并进入 LoRA/训练初始化阶段。

### Known risks / TODO

- `examples/dpo/run_qwen3_4b_ultrafeedback.sh` 当前硬编码 `VENV_PATH="/home/lhf_hongfu_gmail_com/.venvs/DPO"`；如果该大写路径存在但环境不完整，会覆盖已激活的小写 `dpo` 环境，导致 `ModuleNotFoundError`。
- 本次未修改仓库代码；若后续坚持直接使用脚本且环境名必须为小写 `dpo`，需要单独处理这个路径兼容问题。
- 本次 smoke 已确认进入模型初始化，但完整 20 step 训练仍需继续等待运行完成。
- `sed -n '180,320p' tunix/rl/grpo/grpo_learner.py`
- `rg -n "actor_generation_chunk_size|actor_grad_acc_factor|Actor Training|max_steps|num_generations|use-dbc-self-inf-group" examples/deepscaler tunix my_example -S`
- 确认：
  - 本次命令不是 `--smoke-test`，目标训练步数为 `315`。
  - 当前配置为 `batch_size=128`、`num_generations=8`、`actor_generation_chunk_size=2`，因此单个训练 step 需要先完成 `128 * 8 = 1024` 条 completion 的 rollout / reward / advantage 处理。
  - `actor_generation_chunk_size=2` 会把 actor 侧更新拆成 `8 / 2 = 4` 个累积块；进度条只有在整个 step 完成后才会从 `0/315` 跳到 `1/315`。

### Known risks / TODO

- 仅凭用户贴出的 stdout 片段，无法精确给出“第一个 step 已完成百分之多少”；若要精确计数，需要基于完整日志统计当前已打印的 completion 结果条数，或直接观察进程后续 stdout。

---

## 2026-03-10: DeepScaler first-step runtime state check

### Scope

- 无代码改动。
- 检查用户当前运行中的 DeepScaler 训练进程和 TensorBoard event 文件，判断第一个训练 step 是否仍停留在判分阶段。

### Changed files

1. `develop.md`

### Validation

- `ps -eo pid,lstart,etime,cmd | rg 'examples/deepscaler/train_deepscaler_nb.py|run_train.sh' -S`
- `find /tmp/deepscaler_tb_20260310_022623 -maxdepth 3 -type f`
- `find /tmp/deepscaler_ckpt_20260310_022623 -maxdepth 4 -type f`
- `source .venv_sglang312/bin/activate && python - <<'PY' ... event_accumulator ... PY`
- 确认：
  - 训练进程自 `2026-03-10 02:26:22 UTC` 起仍在运行。
  - checkpoint 目录尚无保存产物。
  - metrics event 文件中已存在 `actor/train/tflops_per_step`，其 `count=1` 且 `last_step=0`。
  - 这表明当前运行已经进入第一个训练 step 的 actor-train 阶段；因此第一个 batch 的 rollout / reward / advantage 前置阶段大概率已经完成。

### Known risks / TODO

- 由于当前 stdout 没有重定向到文件，无法事后精确统计“已经打印了多少条 `IS CORRECT/IS NOT CORRECT`”；对“1024 个判分样本已完成多少”的判断只能结合 event 文件阶段信号来推断。

---

## 2026-03-10: DeepScaler first-step liveness recheck

### Scope

- 无代码改动。
- 复测用户当前训练 run 的进程活跃度与 event 文件增长情况，判断距离 `1/315` 是否还很近。

### Changed files

1. `develop.md`

### Validation

- `stat -c '%y %s %n' /tmp/deepscaler_tb_20260310_022623/events.out.tfevents.1773109614.t1v-n-d0f559df-w-0`
- `ps -p 2971628 -o pid,etime,time,%cpu,%mem,stat,cmd`
- 间隔约 12 秒再次执行同样检查
- 确认：
  - 训练进程仍存活，且 `TIME` 在增长，说明仍在消耗 CPU 进行计算或编译。
  - TensorBoard event 文件大小与 mtime 在两次检查间都未变化，仍停留在 `2026-03-10 02:46:55 UTC`。
  - 这说明当前并没有新的 trainer step 指标落盘；距离 `1/315` 至少不是“几秒内就会跳”的状态。

### Known risks / TODO

- 仅靠 host 侧进程状态和 event 文件，无法精确区分“长时间 JAX 编译”与“数值上极慢的 actor 更新”；两者都会表现为 CPU 活跃但 step 指标不前进。

---

## 2026-03-10: DeepScaler step-1 completion confirmation

### Scope

- 无代码改动。
- 再次检查当前运行中的 DeepScaler 训练日志落盘状态，确认是否已经到达第 1 个 step。

### Changed files

1. `develop.md`

### Validation

- `source .venv_sglang312/bin/activate && python - <<'PY' ... event_accumulator ... PY`
- `find /tmp/deepscaler_ckpt_20260310_022623 -maxdepth 4 -type f`
- `ps -p 2971628 -o pid,etime,time,%cpu,%mem,stat`
- 确认：
  - TensorBoard event 文件里已有首批 `global/train/...` 指标，均为 `last_step=0`，表示第一轮训练样本的统计已经落盘。
  - checkpoint 目录已出现 `/tmp/deepscaler_ckpt_20260310_022623/actor/1/...`。
  - 按当前 RL cluster 的 checkpoint 目录约定，这说明第一个 actor step 已经完成并写出 step `1` 的 checkpoint。

### Known risks / TODO

- event 文件中的很多训练指标仍以 `step=0` 记账，而 checkpoint 目录用 `actor/1` 命名；两者是不同的计步口径，不应混为一谈。

---

## 2026-03-10: DeepScaler step-1 elapsed time measurement

### Scope

- 无代码改动。
- 基于当前运行进程启动时间与 `actor/1` checkpoint 文件时间，估算到达 step 1 的实际耗时。

### Changed files

1. `develop.md`

### Validation

- `ps -p 2971628 -o lstart=,etime=,cmd=`
- `stat -c '%y %n' /tmp/deepscaler_ckpt_20260310_022623/actor/1/_CHECKPOINT_METADATA /tmp/deepscaler_ckpt_20260310_022623/actor/1/model_params/manifest.ocdbt`
- 确认：
  - 训练进程启动时间：`2026-03-10 02:26:22 UTC`
  - `actor/1` checkpoint 文件时间：`2026-03-10 04:10:15 UTC`
  - 两者相差约 `1 小时 43 分 53 秒`

### Known risks / TODO

- 这个耗时是按 `actor/1` checkpoint 落盘时间估算的；真正“step 1 训练计算完成”的时刻可能会比写盘时间略早，但通常差距不会很大。

---

## 2026-03-10: DeepScaler checkpoint cadence and disk usage check

### Scope

- 无代码改动。
- 检查当前 DeepScaler 训练 run 的 checkpoint 保存间隔、当前 checkpoint 占用，以及磁盘剩余空间。

### Changed files

1. `develop.md`

### Validation

- `sed -n '1,140p' examples/deepscaler/run_train.sh`
- `du -sh /tmp/deepscaler_ckpt_20260310_022623`
- `du -sb /tmp/deepscaler_ckpt_20260310_022623`
- `find /tmp/deepscaler_ckpt_20260310_022623/actor -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort -n`
- `find /tmp/deepscaler_ckpt_20260310_022623/actor/1/model_params -type f -printf '%s\n' | awk '{s+=$1} END {print s}'`
- `df -h /tmp /home /`
- 确认：
  - 当前脚本默认 `SAVE_INTERVAL_STEPS=158`，`MAX_TO_KEEP=2`。
  - 当前 run 的 checkpoint 根目录 `/tmp/deepscaler_ckpt_20260310_022623` 总占用约 `2.6G`（`2769934936` bytes）。
  - 当前 actor 目录下只有 step `1`。
  - `actor/1/model_params` 文件总和约 `2769901876` bytes，约等于 `2.58 GiB`。
  - 当前根分区剩余空间约 `21G`。

### Known risks / TODO

- `du -sh` 对 Orbax/OCDBT 目录的展示可能不如字节级统计直观；本次以 `du -sb` 和文件字节和作为更可靠口径。
- 实际单次 checkpoint 总大小会随着是否包含额外 metadata/optimizer state 略有波动，但当前 run 的 actor checkpoint 量级可按约 `2.6G` 估算。

---

## 2026-03-10: DeepScaler default checkpoint interval adjustment

### Scope

- 将 `examples/deepscaler/run_train.sh` 的默认 checkpoint 保存间隔从 `158` 调整为 `79`。
- 目标是在默认 `MAX_STEPS=315` 的配置下，训练过程中大约产生 4 次保存点。

### Changed files

1. `examples/deepscaler/run_train.sh`
2. `examples/deepscaler/run_train_dbc.sh`
3. `develop.md`

### Validation

- `sed -n '1,80p' examples/deepscaler/run_train.sh`
- `sed -n '1,80p' examples/deepscaler/run_train_dbc.sh`
- 确认：
  - `examples/deepscaler/run_train.sh` 与 `examples/deepscaler/run_train_dbc.sh` 的默认 `SAVE_INTERVAL_STEPS` 都已从 `158` 改为 `79`
  - 在 `315` step 的默认 run 下，预期保存点大致为 step `79`、`158`、`237`、`315` 附近

### Known risks / TODO

- 当前修改只影响后续新启动的 run；已经在运行中的训练进程不会动态读取这个新默认值。
- 更频繁保存会增加 I/O 和磁盘占用；按当前 actor checkpoint 约 `2.6G` 估算，需要继续关注 `/tmp` 剩余空间。
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

## 2026-03-10 - Pushed-contents summary request

### Scope

- 无代码改动。
- Confirmed the exact contents of pushed commit `24e3cf79dbbdb7193ba2d22c8e099e8bca06510a`.

### Changed files

1. `develop.md`

### Validation

- `git show --stat --summary --format=fuller 24e3cf79`
- `git show --name-only --format=oneline --no-renames 24e3cf79`

### Validation results

- The pushed commit contains 7 tracked-file changes:
  - `develop.md`
  - `examples/deepscaler/README.md`
  - `examples/deepscaler/run_train.sh`
  - `examples/deepscaler/run_train_dbc.sh`
  - `examples/deepscaler/train_deepscaler_nb.py`
  - `tunix/rl/experimental/agentic_rl_learner.py`
  - `tunix/rl/rl_cluster.py`
- Commit summary:
  - added actor-side generation chunking support
  - added a dedicated DeepScaler DBC wrapper
  - changed current `run_train.sh` defaults to `NUM_GENERATIONS=8` and `ACTOR_GENERATION_CHUNK_SIZE=2`
  - updated docs and development log

### Known risks / TODO

- This summary step does not include a fresh end-to-end training run after push.

## 2026-03-14 - DPO environment README check

### Scope

- 无代码改动。
- Checked what runtime environment `examples/dpo` expects and whether the README already documents it.

### Changed files

1. `develop.md`

### Validation

- `sed -n '1,140p' examples/dpo/README.md`
- `sed -n '1,220p' examples/dpo/run_qwen3_4b_ultrafeedback.sh`
- `sed -n '1,220p' pyproject.toml`

### Validation results

- `examples/dpo/README.md` already documents the dedicated DPO environment, Python version, install commands, required `HF_TOKEN`, and TPU assumptions.
- `examples/dpo/run_qwen3_4b_ultrafeedback.sh` matches the documented virtualenv path `/home/lhf_hongfu_gmail_com/.venvs/DPO`.
- `pyproject.toml` confirms the project requires Python `>=3.11`.

### Known risks / TODO

- README documents the intended environment, but this check did not perform a live training run or dependency import test in `/home/lhf_hongfu_gmail_com/.venvs/DPO`.

## 2026-03-14 - DPO environment setup

### Scope

- 无代码改动。
- Reused the existing `/home/lhf_hongfu_gmail_com/.venvs/DPO` virtualenv and aligned it with the `examples/dpo/README.md` setup instructions.

### Changed files

1. `develop.md`

### Validation

- `bash -lc 'source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && python -m pip install -U pip'`
- `bash -lc 'source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && python -m pip install -e ".[dev]"'`
- `bash -lc 'source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && python -m pip install "jax[tpu]==0.8.1"'`
- `bash -lc 'source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && python - <<\"PY\"\nimport jax\nimport tunix\nprint(\"jax\", jax.__version__)\nprint(\"backend\", jax.default_backend())\nprint(\"devices\", jax.device_count())\nprint(\"tunix_import\", tunix.__file__)\nPY'`
- `bash -lc 'source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && python -m pip show jax jaxlib libtpu google-tunix | sed -n "1,120p"'`

### Validation results

- The existing `DPO` virtualenv is active and uses Python `3.11.13`.
- Tunix is installed in editable mode from `/home/lhf_hongfu_gmail_com/tunix`.
- JAX stack is aligned to the documented TPU setup:
  - `jax==0.8.1`
  - `jaxlib==0.8.1`
  - `libtpu==0.0.30`
- Runtime sanity check succeeded with:
  - `backend tpu`
  - `devices 4`

### Known risks / TODO

- This step did not execute `./examples/dpo/run_qwen3_4b_ultrafeedback.sh smoke`; running the trainer still requires a valid `HF_TOKEN` in `.env` or the shell environment.

## 2026-03-14 - DPO smoke run check

### Scope

- 无代码改动。
- Ran the DPO smoke command to verify environment, token loading, model download, trainer startup, and first eval/training transition.

### Changed files

1. `develop.md`

### Validation

- `bash -lc 'source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && ./examples/dpo/run_qwen3_4b_ultrafeedback.sh smoke'`
- `find runs/dpo_qwen3_4b_ultrafeedback_smoke -maxdepth 3 -type f | sort | tail -n 40`
- `du -sh runs/dpo_qwen3_4b_ultrafeedback_smoke`

### Validation results

- Smoke run successfully:
  - loaded `HF_TOKEN` from `my_example/.env`
  - downloaded `Qwen/Qwen3-4B-Instruct-2507`
  - initialized the DPO trainer and checkpoint manager
  - skipped WandB backend cleanly because `wandb` is not installed
  - completed initial eval with `Train step 0 eval loss: 0.691406 - eval perplexity: 2.000000`
  - entered the training loop and created TensorBoard event files under `runs/dpo_qwen3_4b_ultrafeedback_smoke/tensorboard`
- The smoke process was manually stopped after confirming the run had passed initialization and started training, to avoid holding TPU resources for the rest of the compile/train cycle.

### Known risks / TODO

- This verification did not wait for all `20` smoke steps to complete, so it does not yet prove end-of-run cleanup or LoRA merge/export behavior.

## 2026-03-14 - DPO DBC integration assessment

### Scope

- 无代码改动。
- Assessed whether Dynamic Batch Curation should be wired into `examples/dpo` and where it would need to land in the DPO stack.

### Changed files

1. `develop.md`

### Validation

- `sed -n '1,260p' tunix/sft/dpo/dpo_trainer.py`
- `sed -n '1,340p' tunix/sft/peft_trainer.py`
- `sed -n '1,260p' tunix/rl/robust_trainer.py`
- `sed -n '520,610p' tunix/rl/rl_cluster.py`
- `sed -n '1,220p' examples/dpo/qwen3_4b_ultrafeedback.yaml`

### Validation results

- Existing DBC wiring is RL-only:
  - `tunix/rl/robust_trainer.py` subclasses `tunix.rl.trainer.Trainer`
  - `tunix/rl/rl_cluster.py` selects `RobustTrainer` only for RL actor training
- DPO uses `tunix/sft/dpo/dpo_trainer.py`, which subclasses `tunix/sft/peft_trainer.PeftTrainer`, so it does not pass through the RL trainer selection path.
- The current DPO baseline config uses `batch_size: 1` with `gradient_accumulation_steps: 8`, so per-sample DBC on the current micro-batch would have no real filtering effect because each train step sees only one sample before accumulation.
- If DBC is ever added to DPO, the correct landing point is a DPO/SFT-side trainer branch, not the existing RL `RobustTrainer`.

### Known risks / TODO

- A useful DPO DBC design likely needs either:
  - larger per-step micro-batches (`batch_size > 1`), or
  - a new accumulation-aware curation design that filters across multiple micro-steps before optimizer update.

## 2026-03-14 - DPO cross-accumulation DBC implementation

### Scope

- Implemented DPO-side Dynamic Batch Curation that filters per-sample gradient outliers across a full gradient-accumulation window before one optimizer update.
- Switched the `examples/dpo` baseline from `batch_size=1, gradient_accumulation_steps=8` to `batch_size=2, gradient_accumulation_steps=4`, preserving the effective batch size of `8`.
- Added DPO example/docs coverage for enabling DBC via config overrides.

### Changed files

1. `develop.md`
2. `examples/dpo/README.md`
3. `examples/dpo/qwen3_4b_ultrafeedback.yaml`
4. `examples/dpo/run_qwen3_4b_ultrafeedback.sh`
5. `tests/cli/dpo_main_test.py`
6. `tests/sft/dpo/dpo_trainer_test.py`
7. `tunix/__init__.py`
8. `tunix/cli/dpo_main.py`
9. `tunix/sft/dpo/dpo_trainer.py`

### Validation

- `bash -lc 'source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && python -m py_compile tunix/cli/dpo_main.py tunix/sft/dpo/dpo_trainer.py tests/cli/dpo_main_test.py tests/sft/dpo/dpo_trainer_test.py'`
- `bash -lc 'source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && python tests/cli/dpo_main_test.py'`
- `bash -lc 'source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && python tests/sft/dpo/dpo_trainer_test.py'`
- `bash -lc 'source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && RUN_TS=$(date +%Y%m%d_%H%M%S) && ./examples/dpo/run_qwen3_4b_ultrafeedback.sh smoke dpo_config.use_dynamic_batch_curation=true dpo_config.curation_threshold=3.0 training_config.max_steps=2 training_config.eval_every_n_steps=1 training_config.checkpoint_root_directory=/tmp/dpo_dbc_smoke_${RUN_TS}/checkpoints training_config.metrics_logging_options.log_dir=/tmp/dpo_dbc_smoke_${RUN_TS}/tensorboard merged_model_output_dir=/tmp/dpo_dbc_smoke_${RUN_TS}/merged_lora'`

### Validation results

- `tunix/sft/dpo/dpo_trainer.py` now contains:
  - per-sample DPO/ORPO loss helpers
  - accumulation-window curation aggregation
  - `CuratedDPOTrainer` with manual cross-micro-step gradient curation
- `tunix/cli/dpo_main.py` now selects `CuratedDPOTrainer` when `dpo_config.use_dynamic_batch_curation=true`.
- `examples/dpo` baseline now uses `batch_size=2` and `gradient_accumulation_steps=4`; smoke overrides match those values.
- `tests/cli/dpo_main_test.py`: passed.
- `tests/sft/dpo/dpo_trainer_test.py`: passed, including:
  - outlier-filter aggregation math
  - equivalence to standard grad accumulation when threshold is effectively disabled
- End-to-end DBC smoke validation:
  - command reached `CuratedDPOTrainer`
  - passed step-0 eval without the earlier DBC/eval metric crash
  - created TensorBoard output under `/tmp/dpo_dbc_smoke_20260314_191658/tensorboard`
  - the temporary smoke process was then terminated manually to release TPU resources

### Known risks / TODO

- The curated DPO path currently skips TFLOPs measurement because the existing utility assumes a single `train_step(model, optimizer, batch)` signature.
- End-to-end validation confirmed startup, eval, and active metric output for the DBC example path, but it did not wait for the temporary `max_steps=2` smoke run to fully finish and merge/export LoRA outputs.

## 2026-03-15 - DPO baseline command and hyperparameter check

### Scope

- 无代码改动。
- Confirmed the current non-DBC `examples/dpo` full training command and baseline hyperparameters from the launcher, README, and YAML config.

### Changed files

1. `develop.md`

### Validation

- `sed -n '1,220p' examples/dpo/README.md`
- `sed -n '1,220p' examples/dpo/qwen3_4b_ultrafeedback.yaml`
- `sed -n '1,220p' examples/dpo/run_qwen3_4b_ultrafeedback.sh`

### Validation results

- The normal training command remains `./examples/dpo/run_qwen3_4b_ultrafeedback.sh full`.
- The current baseline keeps effective batch size `8` via `batch_size=2` and `gradient_accumulation_steps=4`.
- The recipe still targets `Qwen/Qwen3-4B-Instruct-2507` on `HuggingFaceH4/ultrafeedback_binarized` with LoRA DPO.

### Known risks / TODO

- None for this verification-only task.

## 2026-03-15 - DPO README outlier_l2 full command

### Scope

- Added the missing `full` training example for DPO `outlier_l2` curation to the example README.

### Changed files

1. `develop.md`
2. `examples/dpo/README.md`

### Validation

- `sed -n '1,200p' examples/dpo/README.md`

### Validation results

- `examples/dpo/README.md` now documents three `full` launcher patterns:
  - baseline `full`
  - `full` with `outlier_l2`
  - `full` with `self_inf_batch`

### Known risks / TODO

- None for this documentation-only update.

## 2026-03-15 - DPO README command presence check

### Scope

- 无代码改动。
- Confirmed which DPO training and smoke commands are already documented in the example README.

### Changed files

1. `develop.md`

### Validation

- `sed -n '1,140p' examples/dpo/README.md`

### Validation results

- `examples/dpo/README.md` already documents:
  - normal `full` training
  - normal `smoke`
  - DBC `smoke` with `outlier_l2`
  - DBC `smoke` with `self_inf_batch`
- The README does not currently include a dedicated `full` example for `self_inf_batch`; that path is still run by passing overrides to the existing `full` launcher command.

### Known risks / TODO

- If needed later, the README can add a separate `full + self_inf_batch` example block for discoverability, but it is not required for correctness.

## 2026-03-15 - DPO README self-inf-batch full command

### Scope

- Added the missing `full` training example for DPO `self_inf_batch` curation to the example README.

### Changed files

1. `develop.md`
2. `examples/dpo/README.md`

### Validation

- `sed -n '1,160p' examples/dpo/README.md`

### Validation results

- `examples/dpo/README.md` now documents all four main launcher patterns:
  - normal `full`
  - normal `smoke`
  - DBC `smoke`
  - DBC `full` for `self_inf_batch`

### Known risks / TODO

- None for this documentation-only update.

## 2026-03-15 - DPO README full command inventory

### Scope

- 无代码改动。
- Checked which `full` commands are currently documented in the DPO example README.

### Changed files

1. `develop.md`

### Validation

- `sed -n '1,180p' examples/dpo/README.md`

### Validation results

- `examples/dpo/README.md` currently documents two `full` commands:
  - baseline `full`
  - `full` with `self_inf_batch` DBC
- The README does not currently include a separate `full` example for `outlier_l2`; that path can still be run by appending the corresponding overrides to the baseline `full` command.

### Known risks / TODO

- None for this verification-only task.

## 2026-03-15 - DPO DBC variant scope clarification

### Scope

- 无代码改动。
- Clarified the difference between the DBC variants already present in the RL/GRPO stack and the single DBC variant currently implemented for DPO.

### Changed files

1. `develop.md`

### Validation

- `rg -n "dynamic batch curation|DBC|dbc|use_dynamic_batch_curation|curation_threshold|self_inf|outlier|grad_norm" tunix my_example examples tests`
- `sed -n '1,260p' tunix/rl/robust_trainer.py`
- `sed -n '1,260p' tunix/rl/self_inf_trainer.py`
- `sed -n '150,220p' my_example/config.py`
- `sed -n '540,590p' tunix/rl/rl_cluster.py`

### Validation results

- The RL/GRPO stack exposes multiple DBC variants:
  - outlier-L2 via `RobustTrainer`
  - self-influence batch scope via `SelfInfTrainer(scope="batch")`
  - self-influence group scope via `SelfInfTrainer(scope="group")`
- The DPO stack currently exposes only one DBC variant:
  - gradient-norm outlier filtering with cutoff `mean + threshold * std`
  - applied once per full accumulation window in `CuratedDPOTrainer`
- DPO currently has no `self_inf` variant selector and no environment-variable-based DBC variant switch like RL.

### Known risks / TODO

- If multiple DPO DBC variants are desired later, the clean extension point is the DPO trainer selection path in `tunix/cli/dpo_main.py`, not the existing RL variant wiring.

## 2026-03-15 - DPO self-influence variant design judgment

### Scope

- 无代码改动。
- Assessed whether DPO should expose both `self-inf-batch` and `self-inf-group`, or only one self-influence-style curation variant.

### Changed files

1. `develop.md`

### Validation

- Reused the already inspected DPO trainer implementation in `tunix/sft/dpo/dpo_trainer.py`.
- Reused the already inspected RL self-influence implementation in `tunix/rl/self_inf_trainer.py`.
- Reused the already inspected DPO example recipe in `examples/dpo/qwen3_4b_ultrafeedback.yaml`.

### Validation results

- For the current DPO recipe, `self-inf-group` is not naturally defined because samples are independent preference pairs, not GRPO-style grouped rollouts.
- `self-inf-batch` could be defined for DPO by comparing each pair gradient against the mean gradient of the current curation window.
- `self-inf-group` would only make sense if the DPO dataset loader and batching logic preserved true per-prompt groups with multiple preference pairs per prompt.
- Recommendation: keep one DPO DBC variant for now, or add `self-inf-batch` first; do not add `self-inf-group` unless grouped DPO data is introduced explicitly.

### Known risks / TODO

- If grouped DPO is introduced later, the grouping semantics must be made explicit in the dataset and batching contract before adding a `self-inf-group` variant.

## 2026-03-15 - DPO self-inf-batch variant implementation

### Scope

- Added a second DPO Dynamic Batch Curation variant, `self_inf_batch`, alongside the existing `outlier_l2` path.
- Kept the default DPO DBC behavior unchanged by preserving `outlier_l2` as the default `curation_variant`.
- Added config/docs/tests so DPO can switch between the two DBC variants without touching the RL DBC stack.

### Changed files

1. `develop.md`
2. `examples/dpo/README.md`
3. `examples/dpo/qwen3_4b_ultrafeedback.yaml`
4. `tests/cli/dpo_main_test.py`
5. `tests/sft/dpo/dpo_trainer_test.py`
6. `tunix/cli/dpo_main.py`
7. `tunix/sft/dpo/dpo_trainer.py`

### Validation

- `bash -lc 'source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && python -m py_compile tunix/cli/dpo_main.py tunix/sft/dpo/dpo_trainer.py tests/cli/dpo_main_test.py tests/sft/dpo/dpo_trainer_test.py'`
- `bash -lc 'source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && python tests/cli/dpo_main_test.py'`
- `bash -lc 'source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && JAX_PLATFORMS=cpu python tests/sft/dpo/dpo_trainer_test.py'`
- `bash -lc 'source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && RUN_TS=$(date +%Y%m%d_%H%M%S) && ./examples/dpo/run_qwen3_4b_ultrafeedback.sh smoke dpo_config.use_dynamic_batch_curation=true dpo_config.curation_variant=self_inf_batch dpo_config.self_influence_dot_threshold=0.0 training_config.max_steps=2 training_config.eval_every_n_steps=1 training_config.checkpoint_root_directory=/tmp/dpo_self_inf_smoke_${RUN_TS}/checkpoints training_config.metrics_logging_options.log_dir=/tmp/dpo_self_inf_smoke_${RUN_TS}/tensorboard merged_model_output_dir=/tmp/dpo_self_inf_smoke_${RUN_TS}/merged_lora'`

### Validation results

- `tunix/sft/dpo/dpo_trainer.py` now supports:
  - `curation_variant="outlier_l2"` using the existing grad-norm cutoff
  - `curation_variant="self_inf_batch"` using per-sample gradient dot products against the full accumulation-window mean gradient
- `DPOTrainingConfig` now normalizes DPO DBC variant aliases like `self-inf-batch` to the canonical `self_inf_batch`.
- `CuratedDPOTrainer` now logs variant-specific DBC metrics without polluting eval metrics:
  - common train-side DBC counts and grad-norm stats
  - `dbc/grad_norm_cutoff` for `outlier_l2`
  - `dbc/self_inf_dot_mean`, `dbc/self_inf_dot_std`, and `dbc/self_inf_dot_threshold` for `self_inf_batch`
- `tests/cli/dpo_main_test.py`: passed.
- `tests/sft/dpo/dpo_trainer_test.py`: passed on CPU, including:
  - direct self-influence filtering math
  - normalization of the `self-inf-batch` alias
  - equivalence to standard gradient accumulation when `self_inf_batch` is configured to keep all samples
- The attempted TPU smoke run confirmed that the `self_inf_batch` config overrides were accepted and reached the real DPO launcher/config merge path, but the process stalled in JAX TPU metadata probing before trainer startup and was terminated manually to release resources.

### Known risks / TODO

- The new `self_inf_batch` DPO variant is validated by unit/integration tests on CPU, but not yet by a completed TPU smoke run because this environment stalled during TPU backend metadata probing before trainer startup.

## 2026-03-15 - DPO self-inf-batch smoke rerun and validation

### Scope

- 无代码改动。
- Reran the DPO `self_inf_batch` smoke test, debugged the earlier startup stall, and confirmed the smoke path completes end-to-end when run outside sandbox restrictions.

### Changed files

1. `develop.md`

### Validation

- Sandboxed smoke attempt:
  - `bash -lc 'source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && RUN_TS=$(date +%Y%m%d_%H%M%S) && ./examples/dpo/run_qwen3_4b_ultrafeedback.sh smoke dpo_config.use_dynamic_batch_curation=true dpo_config.curation_variant=self_inf_batch dpo_config.self_influence_dot_threshold=0.0 training_config.max_steps=2 training_config.eval_every_n_steps=1 training_config.checkpoint_root_directory=/tmp/dpo_self_inf_smoke_${RUN_TS}/checkpoints training_config.metrics_logging_options.log_dir=/tmp/dpo_self_inf_smoke_${RUN_TS}/tensorboard merged_model_output_dir=/tmp/dpo_self_inf_smoke_${RUN_TS}/merged_lora'`
- Unsandboxed rerun of the same smoke command:
  - `bash -lc 'source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && RUN_TS=$(date +%Y%m%d_%H%M%S) && ./examples/dpo/run_qwen3_4b_ultrafeedback.sh smoke dpo_config.use_dynamic_batch_curation=true dpo_config.curation_variant=self_inf_batch dpo_config.self_influence_dot_threshold=0.0 training_config.max_steps=2 training_config.eval_every_n_steps=1 training_config.checkpoint_root_directory=/tmp/dpo_self_inf_smoke_${RUN_TS}/checkpoints training_config.metrics_logging_options.log_dir=/tmp/dpo_self_inf_smoke_${RUN_TS}/tensorboard merged_model_output_dir=/tmp/dpo_self_inf_smoke_${RUN_TS}/merged_lora'`
- Post-run artifact checks:
  - `find /tmp/dpo_self_inf_smoke_20260315_030834/checkpoints -maxdepth 2 -mindepth 1 -type d | sort`
  - `find /tmp/dpo_self_inf_smoke_20260315_030834/merged_lora -maxdepth 2 -type f | sort`
  - `find /tmp/dpo_self_inf_smoke_20260315_030834/tensorboard -maxdepth 1 -type f | sort`

### Validation results

- The sandboxed smoke reproduced the earlier issue:
  - repeated `Failed to get TPU metadata (tpu-env)` during startup
  - no progress into model/trainer initialization
- The same smoke command succeeded once rerun outside sandbox restrictions.
- Confirmed runtime milestones on the successful unsandboxed run:
  - model download/init completed
  - overlong train/eval DPO filtering ran
  - `CuratedDPOTrainer` selected with `curation_variant=self_inf_batch`
  - step-0 eval completed
  - train step 1 completed and checkpoint step `1` saved
  - eval at train step `2` completed
  - train step 2 completed and checkpoint step `2` saved
  - merged LoRA output was written successfully
- Successful run outputs:
  - checkpoints under `/tmp/dpo_self_inf_smoke_20260315_030834/checkpoints/1` and `/tmp/dpo_self_inf_smoke_20260315_030834/checkpoints/2`
  - merged model under `/tmp/dpo_self_inf_smoke_20260315_030834/merged_lora`
  - TensorBoard events under `/tmp/dpo_self_inf_smoke_20260315_030834/tensorboard`
- Observed runtime characteristic:
  - the 2-step smoke completed successfully but took about 8 minutes because the first eval/train compilation on TPU was very slow in this environment

### Known risks / TODO

- For TPU-backed smoke/debug in this environment, sandboxed execution is not reliable because TPU metadata probing can stall before trainer startup; prefer unsandboxed execution for real TPU validation.

## 2026-03-15 - DPO DBC parameter and filtering scope check

### Scope

- 无代码改动。
- Confirmed which DPO Dynamic Batch Curation knobs are exposed in config and the exact sample window over which filtering is applied.

### Changed files

1. `develop.md`

### Validation

- `rg -n "use_dynamic_batch_curation|curation_threshold|aggregate_curated_step|CuratedDPOTrainer|gradient_accumulation_steps|batch_size" tunix/sft/dpo/dpo_trainer.py tunix/cli/dpo_main.py examples/dpo/qwen3_4b_ultrafeedback.yaml examples/dpo/run_qwen3_4b_ultrafeedback.sh`
- `sed -n '1,260p' tunix/sft/dpo/dpo_trainer.py`
- `sed -n '260,620p' tunix/sft/dpo/dpo_trainer.py`
- `sed -n '620,820p' tunix/sft/dpo/dpo_trainer.py`
- `sed -n '220,280p' tunix/cli/dpo_main.py`

### Validation results

- DPO DBC is enabled by `dpo_config.use_dynamic_batch_curation=true`.
- The only explicit DBC threshold knob is `dpo_config.curation_threshold`, used as `mean_norm + threshold * std_norm`.
- Filtering is applied once per full accumulation window, not per micro-step:
  - each micro-step computes per-sample gradients and norms
  - those samples are concatenated across `training_config.gradient_accumulation_steps`
  - curation runs over the concatenated window before one optimizer update
- The sample count inside one curation window is determined by `batch_size * gradient_accumulation_steps`.

### Known risks / TODO

- None for this verification-only task.

## 2026-03-16 - Tunix supported base model inventory check

### Scope

- 无代码改动。
- 核对当前仓库中 Tunix 正式支持的 base model 家族与具体变体，基于 `naming` 映射、`ModelConfig` 注册和覆盖测试给出结论。

### Changed files

1. `develop.md`

### Validation

- `rg -n "def (gemma|gemma1p1|gemma2|gemma3|llama3|llama3p1|llama3p2|qwen2p5|deepseek_r1_distill_qwen|qwen3)_" tunix/models/gemma/model.py tunix/models/gemma3/model.py tunix/models/llama3/model.py tunix/models/qwen2/model.py tunix/models/qwen3/model.py`
- `nl -ba tunix/models/naming.py | sed -n '60,110p'`
- `nl -ba tests/models/naming_test.py | sed -n '30,420p'`

### Validation results

- 当前 `naming` 层支持的 model family / category 为：
  - `gemma`, `gemma1p1`, `gemma2`, `gemma3`
  - `llama3`, `llama3p1`, `llama3p2`
  - `qwen2p5`, `deepseek_r1_distill_qwen`
  - `qwen3`
- `tests/models/naming_test.py` 会校验 `_TEST_MODEL_INFOS` 与各家族 `ModelConfig` 方法是双向全覆盖，因此这份清单可视为当前仓库正式支持集合。
- 当前支持的具体 base model 变体共 40 个：
  - Gemma: `gemma-2b`, `gemma-2b-it`, `gemma-1.1-2b-it`, `gemma-7b`, `gemma-7b-it`, `gemma-1.1-7b-it`, `gemma-2-2b`, `gemma-2-2b-it`, `gemma-2-9b`, `gemma-2-9b-it`
  - Gemma 3: `gemma-3-270m`, `gemma-3-270m-it`, `gemma-3-1b-pt`, `gemma-3-1b-it`, `gemma-3-4b-pt`, `gemma-3-4b-it`, `gemma-3-12b-pt`, `gemma-3-12b-it`, `gemma-3-27b-pt`, `gemma-3-27b-it`
  - Llama 3: `llama-3-70b`, `llama-3.1-405b`, `llama-3.1-8b`, `llama-3.1-70b`, `llama-3.2-1b`, `llama-3.2-3b`
  - Qwen 2.5 family: `qwen2.5-0.5b`, `qwen2.5-1.5b`, `qwen2.5-3b`, `qwen2.5-7b`, `qwen2.5-math-1.5b`, `deepseek-r1-distill-qwen-1.5b`
  - Qwen 3: `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b`, `qwen3-4b-instruct-2507`, `qwen3-4b-thinking-2507`, `qwen3-8b`, `qwen3-14b`, `qwen3-30b-a3b`

### Known risks / TODO

- 仓库里目前还没有单独的“模型目录/catalog”源文件；结论依赖 `naming.py`、各家族 `ModelConfig` 和 `tests/models/naming_test.py` 的一致性。

## 2026-03-16 - Qwen base model subset clarification

### Scope

- 无代码改动。
- 针对用户追问，明确区分当前 Tunix 支持的 Qwen 3 / Qwen 2.5 变体中，哪些属于 base model，哪些属于 instruct / thinking / math / distill 变体。

### Changed files

1. `develop.md`

### Validation

- `nl -ba tunix/models/qwen2/model.py | sed -n '100,210p'`
- `nl -ba tunix/models/qwen3/model.py | sed -n '100,215p'`
- `rg -n "Qwen/Qwen2.5|Qwen/Qwen3|DeepSeek-R1-Distill-Qwen" tests/models/naming_test.py`

### Validation results

- Qwen 2.5 family 中当前支持且可视为 plain base model 的是：
  - `qwen2.5-0.5b`
  - `qwen2.5-1.5b`
  - `qwen2.5-3b`
  - `qwen2.5-7b`
- 下列不应归为 plain base model：
  - `qwen2.5-math-1.5b`：math-specialized 变体
  - `deepseek-r1-distill-qwen-1.5b`：distill 变体
- Qwen 3 中当前支持且可视为 plain base model 的是：
  - `qwen3-0.6b`
  - `qwen3-1.7b`
  - `qwen3-4b`
  - `qwen3-8b`
  - `qwen3-14b`
  - `qwen3-30b-a3b`
- 下列不应归为 plain base model：
  - `qwen3-4b-instruct-2507`
  - `qwen3-4b-thinking-2507`

### Known risks / TODO

- 仓库没有单独的 `is_base_model` 标记；本次结论按命名语义与注册名称区分，等价于“非 instruct / 非 thinking / 非 math / 非 distill”的 plain base model。

## 2026-03-16 - Qwen3-4B pretraining-only status clarification

### Scope

- 无代码改动。
- 纠正并澄清 `qwen3-4b` 在 Tunix 本地命名分类与 Qwen 官方训练阶段定义之间的差异。

### Changed files

1. `develop.md`

### Validation

- `nl -ba tests/models/naming_test.py | sed -n '300,330p'`
- 官方来源：
  - `https://huggingface.co/Qwen/Qwen3-4B`
  - `https://huggingface.co/Qwen/Qwen3-4B-Base`

### Validation results

- Tunix 当前注册的 `qwen3-4b` 对应的是官方仓库 `Qwen/Qwen3-4B`，不是 `Qwen/Qwen3-4B-Base`。
- Qwen 官方模型卡标注：
  - `Qwen/Qwen3-4B` 的 `Training Stage` 为 `Pretraining & Post-training`
  - `Qwen/Qwen3-4B-Base` 的 `Training Stage` 为 `Pretraining`
- 因此，若按“只经历 pretrain 的 base model”这个严格定义，`qwen3-4b` 不能算；对应名称应是 `Qwen/Qwen3-4B-Base`。

### Known risks / TODO

- 之前把 `qwen3-4b` 按“非 instruct / 非 thinking”归进 plain base，这个分类只适用于本仓库的命名语义，不等同于官方的“pretrain-only base”定义。

## 2026-03-16 - Pretrain-only base model re-check for Tunix Qwen registrations

### Scope

- 无代码改动。
- 按官方模型卡中的 `Training Stage` 重新核对当前 Tunix 支持的 Qwen 2.5 / Qwen 3 相关模型里，哪些可被严格确认是“只经历过 pretraining”的 base model。

### Changed files

1. `develop.md`

### Validation

- 本地注册清单：
  - `nl -ba tests/models/naming_test.py | sed -n '252,360p'`
- 官方模型卡：
  - `https://huggingface.co/Qwen/Qwen2.5-0.5B`
  - `https://huggingface.co/Qwen/Qwen2.5-1.5B`
  - `https://huggingface.co/Qwen/Qwen2.5-3B`
  - `https://huggingface.co/Qwen/Qwen2.5-7B`
  - `https://huggingface.co/Qwen/Qwen3-0.6B`
  - `https://huggingface.co/Qwen/Qwen3-1.7B`
  - `https://huggingface.co/Qwen/Qwen3-4B`
  - `https://huggingface.co/Qwen/Qwen3-8B`
  - `https://huggingface.co/Qwen/Qwen3-14B`
  - `https://huggingface.co/Qwen/Qwen3-30B-A3B`
  - `https://huggingface.co/Qwen/Qwen3-0.6B-Base`
  - `https://huggingface.co/Qwen/Qwen3-1.7B-Base`
  - `https://huggingface.co/Qwen/Qwen3-4B-Base`
  - `https://huggingface.co/Qwen/Qwen3-8B-Base`
  - `https://huggingface.co/Qwen/Qwen3-14B-Base`
  - `https://huggingface.co/Qwen/Qwen3-30B-A3B-Base`
  - `https://huggingface.co/Qwen/Qwen2.5-Math-1.5B`
  - `https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

### Validation results

- 当前 Tunix 已支持并能被官方模型卡明确确认 `Training Stage: Pretraining` 的 Qwen 系 base model 是：
  - `qwen2.5-0.5b`
  - `qwen2.5-1.5b`
  - `qwen2.5-3b`
  - `qwen2.5-7b`
- 当前 Tunix 注册的 `qwen3-*` 名称都不是 pretrain-only；它们对应的官方模型卡均为 `Training Stage: Pretraining & Post-training`。
- 若按“只经历 pretrain”的严格定义，Qwen 3 对应的官方 base 名称应是：
  - `Qwen/Qwen3-0.6B-Base`
  - `Qwen/Qwen3-1.7B-Base`
  - `Qwen/Qwen3-4B-Base`
  - `Qwen/Qwen3-8B-Base`
  - `Qwen/Qwen3-14B-Base`
  - `Qwen/Qwen3-30B-A3B-Base`
- 但这些 `*-Base` 名称当前不在 Tunix 现有注册清单里。
- `qwen2.5-math-1.5b` 虽然模型卡中被归入 Qwen2.5-Math 的 “base models” 组，但没有像通用 Qwen2.5 / Qwen3 那样给出同样明确的 `Training Stage: Pretraining` 字段；本次未把它放进“已明确确认只经历 pretrain”的严格清单。
- `deepseek-r1-distill-qwen-1.5b` 明确属于 distill 模型，不应归入 pretrain-only base model。

### Known risks / TODO

- 对 `qwen2.5-math-1.5b` 的排除是出于“严格按官方 `Training Stage` 明确字段确认”的保守口径，不代表它一定经历了 post-training；只是当前已查到的官方卡片没有给出同等级别的显式确认。

## 2026-03-16 - UltraFeedback split SFT then DPO feasibility check for qwen2.5-1.5b

### Scope

- 无代码改动。
- 评估基于当前 Tunix 代码，是否适合使用 `qwen2.5-1.5b` 将 `UltraFeedback` 拆分为一部分先做 SFT、另一部分再做 DPO。

### Changed files

1. `develop.md`

### Validation

- `sed -n '1,240p' examples/dpo/README.md`
- `sed -n '1,260p' examples/dpo/qwen3_4b_ultrafeedback.yaml`
- `sed -n '1,260p' tunix/examples/data/ultrafeedback_dpo.py`
- `nl -ba tunix/cli/peft_main.py | sed -n '20,90p'`
- `nl -ba tunix/examples/data/translation_dataset.py | sed -n '35,90p'`
- `nl -ba tunix/cli/dpo_main.py | sed -n '80,155p'`
- `nl -ba tunix/cli/utils/data.py | sed -n '12,80p'`
- `nl -ba tunix/models/qwen2/params.py | sed -n '1,120p'`
- `nl -ba tunix/models/qwen3/params.py | sed -n '140,170p'`

### Validation results

- 从训练方法上看，`qwen2.5-1.5b` 先做 SFT、再做 DPO 是合理路线，尤其因为该模型是 pretrain-only base，而 `UltraFeedback` 是偏 chat/preference 的指令数据。
- 当前 DPO 路径已经支持通过 `train_data_module` / `eval_data_module` 直接读取 `UltraFeedback` preference pairs，并在数据入口统一套 chat template。
- 当前 SFT CLI 入口 `tunix/cli/peft_main.py` 仍然硬编码走 `translation_dataset.create_datasets(...)`，没有像 DPO 那样的通用 `train_data_module` 接口；直接拿 `UltraFeedback` 做 SFT 还不能无改动复用。
- 当前 DPO pipeline 要求 actor/reference 共享同一 base model 标识，并通过“先加载 reference，再在 actor 上额外 apply LoRA”的方式构造模型；这更适合“从一个已定型的 base/full model 再挂一层 DPO LoRA”。
- `qwen3` 已有 `save_lora_merged_model_as_safetensors`，但 `qwen2` 目前没有对应 merge saver；因此若用 `qwen2.5-1.5b` 做 LoRA SFT，SFT 产物不能像 `qwen3` 一样顺滑地 merge 成一个新的 full model 再喂给 DPO。

### Known risks / TODO

- 结论是“方法上可行，但按当前代码不是开箱即用”。
- 若要把这条链路做顺，优先需要两类补充能力：
  - SFT 侧支持通用数据模块或单独的 UltraFeedback SFT 入口；
  - `qwen2` 支持 LoRA merge 导出，或 DPO 侧支持从 SFT checkpoint 正确初始化 actor/reference。

## 2026-03-16 - DBC-on-DPO paper setting discussion

### Scope

- 无代码改动。
- 评估“使用 `qwen2.5-1.5b`，将 UltraFeedback 划分为 SFT 子集和 DPO 子集，用于验证 DBC 在 DPO 上有效性”的实验设定是否适合作为顶会投稿主 setting。

### Changed files

1. `develop.md`

### Validation

- 基于当前仓库中已检查过的：
  - `examples/dpo/qwen3_4b_ultrafeedback.yaml`
  - `tunix/cli/dpo_main.py`
  - `tunix/cli/peft_main.py`
  - `tunix/examples/data/ultrafeedback_dpo.py`
- 以及本轮讨论中已确认的模型口径：
  - `qwen2.5-1.5b` 为 pretrain-only base
  - 当前 Tunix DPO baseline 使用的是 post-trained instruct model

### Validation results

- 该设定作为“辅助实验”是合理的，因为它能测试 DBC 在更弱初始化条件下是否仍有收益。
- 但若把它作为“主实验 setting”去支撑顶会级别的 DBC-on-DPO 核心结论，风险较高，因为 DBC 效果会与以下因素强耦合：
  - pretrain-only base 自身缺乏 instruction tuning
  - SFT 数据切分策略
  - SFT 质量和训练配方
  - 两阶段训练带来的额外超参和初始化差异
- 更强的主实验应优先使用已经 post-trained / instruct 的标准公开模型，在完全相同的 DPO 配方下仅比较“是否启用 DBC”。
- `pretrain-only base -> SFT -> DPO` 更适合作为补充实验，用来回答“DBC 是否也能帮助更弱或更早期的 policy 初始化”。

### Known risks / TODO

- 若只做这一种 setting，审稿人很容易质疑：DBC 的收益是否只是来自于修补一个本来就不够标准的起点，而不是对 DPO 本身更普适的改进。

## 2026-03-16 - Overfitting risk discussion for aligned-model DPO setting

### Scope

- 无代码改动。
- 讨论“instruct/post-trained model + standard DPO + with/without DBC”作为主实验时，是否会因为模型已对齐而在 UltraFeedback 上过拟合，以及这对论文设定的影响。

### Changed files

1. `develop.md`

### Validation

- 基于本轮讨论，不涉及新增脚本或代码执行。
- 参考对象仍为当前仓库中的 DPO baseline recipe：
  - `examples/dpo/qwen3_4b_ultrafeedback.yaml`
  - `examples/dpo/README.md`

### Validation results

- 已对齐模型在 preference 数据上更容易出现“增益空间变小”或“训练后期过拟合”的现象，这个担心是合理的。
- 但这不构成放弃 aligned-model 主 setting 的充分理由；更好的做法是：
  - 把 aligned-model setting 作为主实验，回答 DBC 在标准 DPO 条件下是否有效；
  - 再增加 weaker-initialization setting 作为补充实验，回答 DBC 在更不稳定 regime 下是否更有帮助。
- 若 aligned-model setting 的头部空间较小，反而更能说明 DBC 是否带来稳健且非偶然的改进。

### Known risks / TODO

- 如果 aligned-model setting 完全没有提升，而 weaker-initialization setting 提升明显，则论文主张应收缩为“DBC 对高不稳定 DPO regime 更有效”，而不应宣称普适增益。

## 2026-03-16 - Worker-side DPO artifact inspection for aligned-model setting

### Scope

- 无代码改动。
- 检查当前 worker 上与 `aligned instruct model + standard DPO + with/without DBC` 相关的本地结果文件，确认哪些 full/smoke 结果仍然存在，哪些已经被覆盖。

### Changed files

1. `develop.md`

### Validation

- `find /home/lhf_hongfu_gmail_com -type f -name 'events.out.tfevents.*' 2>/dev/null | rg 'dpo|ultrafeedback|tensorboard'`
- `python - <<'PY' ... EventAccumulator(...) ... PY` on:
  - `runs/dpo_qwen3_4b_ultrafeedback/tensorboard/events.out.tfevents.1773548250.t1v-n-21f197d2-w-0`
  - `runs/dpo_qwen3_4b_ultrafeedback/tensorboard/events.out.tfevents.1773564268.t1v-n-21f197d2-w-0`
  - `runs/dpo_qwen3_4b_ultrafeedback_smoke/tensorboard/events.out.tfevents.*`
  - `/tmp/dpo_self_inf_smoke_20260315_030834/tensorboard/events.out.tfevents.*`
- `sed -n '1,240p' /tmp/dpo_outlier_l2.log`
- `tail -n 120 /tmp/dpo_outlier_l2.log`
- `rg -n "use_dynamic_batch_curation|curation_variant|qwen3-4b-ultrafeedback-dpo-baseline" /tmp/dpo_outlier_l2.log`

### Validation results

- 当前 worker 上仍能明确定位到的 full DPO event 只有：
  - `runs/dpo_qwen3_4b_ultrafeedback/tensorboard/events.out.tfevents.1773548250.t1v-n-21f197d2-w-0`
  - 一个很小的 companion event 文件 `1773564268...`，仅含 JAX compile 指标
- 该 full event 文件包含完整 `dpo/train/dbc/*` 指标，覆盖 `1 -> 5464` 步，说明它是一次启用了 DBC 的 full run，而不是 baseline run。
- `/tmp/dpo_outlier_l2.log` 也明确记录：
  - `dpo_config.use_dynamic_batch_curation=True`
  - `curation_variant='outlier_l2'`
  - 训练完整跑到 `5464` 步并成功保存 checkpoint 与 merged LoRA
- 这次 full DBC run 复用了 baseline 名称和目录：
  - `run_name='qwen3-4b-ultrafeedback-dpo-baseline'`
  - `log_dir=/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen3_4b_ultrafeedback/tensorboard`
  - `checkpoint_root_directory=/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen3_4b_ultrafeedback/checkpoints`
- 因此，当前 worker 上没有找到可直接与之配对的 full non-DBC baseline TensorBoard 留档；现有 full 目录已被 DBC run 占用/覆盖。
- 仍然存在的 baseline 证据只有 smoke 级别：
  - `runs/dpo_qwen3_4b_ultrafeedback_smoke/tensorboard/events.out.tfevents.1773513895...`
  - 其 tags 不含 `dbc/*`
- 仍然存在的 DBC smoke 证据：
  - `self_inf_batch` smoke 在 `/tmp/dpo_self_inf_smoke_20260315_030834/tensorboard` 下，含 `dbc/*` 指标
  - 两个早期 outlier smoke 目录存在，但 event 文件中没有可读标量输出

### Known risks / TODO

- 当前 worker 上的本地数据足以支持“full DBC run 的行为分析”，但不足以支持“full baseline vs full DBC”的严格对照结论，因为 full baseline artifact 目前缺失。
- 后续若要做论文级比较，必须将 baseline / outlier_l2 / self_inf_batch 分别写到独立的 `run_name`、`log_dir`、`checkpoint_root_directory`。

## 2026-03-16 - Interpreting overfitting in aligned-model DPO artifacts

### Scope

- 无代码改动。
- 结合当前 worker 上保留下来的 full DPO artifact，解释“训练 metrics 明显过拟合”对 DBC-on-DPO 论文设定意味着什么。

### Changed files

1. `develop.md`

### Validation

- 基于已解析的 full event 文件：
  - `runs/dpo_qwen3_4b_ultrafeedback/tensorboard/events.out.tfevents.1773548250.t1v-n-21f197d2-w-0`
- 基于已检查的 full 日志：
  - `/tmp/dpo_outlier_l2.log`

### Validation results

- 当前 full run 确实存在明确的 train/eval 分叉：
  - `train/loss`: `0.6934 -> 0.4415`
  - `eval/loss`: `0.3594 -> 0.4004`
  - `train/rewards/accuracy`: `0.375 -> 0.857`
  - `eval/rewards/accuracy`: `0.374 -> 0.712`
- 因此，“明显过拟合”的更强证据不是单独的训练指标，而是 held-out eval 已经随着训练推进而变差。
- 但这条 full run 同时满足：
  - `train/dbc/keep_ratio = 1.0` 全程不变
  - `use_dynamic_batch_curation=True`
  - `curation_variant='outlier_l2'`
- 所以这次 full run 更准确的解释是：
  - aligned-model DPO 确实过拟合
  - 但当前 DBC 配置没有真正介入优化过程
  - 因而这条结果不能用来证明“DBC 无效”，只能说明“当前 `outlier_l2, threshold=3.0, window=8` 这组配置没有起作用”

### Known risks / TODO

- 若论文继续采用 aligned-model setting，核心比较指标不应只看最终点，而应至少加入：
  - best eval checkpoint
  - final-vs-best degradation
  - keep ratio / filtered count
  - train-eval gap

## 2026-03-16 - Qwen2.5-1.5B UltraFeedback SFT -> DPO + DBC implementation

### Scope

- 为 `qwen2.5-1.5b` 落地 `pretrain-only base -> SFT -> DPO` 的最小可运行链路。
- 新增 UltraFeedback 的 prompt-disjoint `sft/dpo` 切分逻辑。
- 给 SFT CLI 增加通用 `train_data_module` / `eval_data_module` 分支，并补齐 `qwen2` 的 merged LoRA safetensors 导出。
- 新增对应的 SFT/DPO recipe、脚本和测试。

### Changed files

1. `develop.md`
2. `examples/data/ultrafeedback_sft.py`
3. `examples/dpo/README.md`
4. `examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft.yaml`
5. `examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh`
6. `examples/sft/ultrafeedback/README.md`
7. `examples/sft/ultrafeedback/qwen2p5_1p5b_ultrafeedback.yaml`
8. `examples/sft/ultrafeedback/run_qwen2p5_1p5b_ultrafeedback.sh`
9. `tests/cli/peft_main_test.py`
10. `tests/examples/data/ultrafeedback_dpo_test.py`
11. `tests/examples/data/ultrafeedback_sft_test.py`
12. `tests/models/qwen2/qwen_params_test.py`
13. `tunix/cli/README.md`
14. `tunix/cli/peft_main.py`
15. `tunix/cli/utils/data.py`
16. `tunix/examples/data/ultrafeedback_dpo.py`
17. `tunix/examples/data/ultrafeedback_sft.py`
18. `tunix/models/qwen2/params.py`

### Validation

- `python -m py_compile tunix/cli/utils/data.py tunix/cli/peft_main.py tunix/models/qwen2/params.py tunix/examples/data/ultrafeedback_dpo.py tunix/examples/data/ultrafeedback_sft.py tests/cli/peft_main_test.py tests/examples/data/ultrafeedback_dpo_test.py tests/examples/data/ultrafeedback_sft_test.py tests/models/qwen2/qwen_params_test.py`
- `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && PYTHONPATH=/home/lhf_hongfu_gmail_com/tunix python tests/cli/dpo_main_test.py`
- `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && PYTHONPATH=/home/lhf_hongfu_gmail_com/tunix python tests/cli/peft_main_test.py`
- `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && PYTHONPATH=/home/lhf_hongfu_gmail_com/tunix python tests/examples/data/ultrafeedback_dpo_test.py`
- `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && PYTHONPATH=/home/lhf_hongfu_gmail_com/tunix python tests/examples/data/ultrafeedback_sft_test.py`
- `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && PYTHONPATH=/home/lhf_hongfu_gmail_com/tunix JAX_PLATFORMS=cpu python tests/models/qwen2/qwen_params_test.py`
- `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && PYTHONPATH=/home/lhf_hongfu_gmail_com/tunix JAX_PLATFORMS=cpu python tests/models/qwen3/qwen_params_test.py`

### Validation results

- `py_compile` 通过，新增和修改的 Python 文件无语法错误。
- `tests/cli/dpo_main_test.py` 通过。
- `tests/cli/peft_main_test.py` 退出码为 `0`。
- `tests/examples/data/ultrafeedback_dpo_test.py` 通过。
- `tests/examples/data/ultrafeedback_sft_test.py` 退出码为 `0`。
- `tests/models/qwen2/qwen_params_test.py` 在 `JAX_PLATFORMS=cpu` 下通过，说明新增的 `qwen2` merged saver 可以完成保存、重载和 forward equivalence 校验。
- `tests/models/qwen3/qwen_params_test.py` 在 `JAX_PLATFORMS=cpu` 下通过，作为对现有 merge 路径的回归验证。
- 新增 recipe 支持：
  - `examples/sft/ultrafeedback/run_qwen2p5_1p5b_ultrafeedback.sh [full|smoke]`
  - `examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh [full|smoke] [baseline|outlier_l2|self_inf_batch] /path/to/sft_merged_model`
- DPO 脚本默认按 `variant/profile/timestamp` 写独立输出目录，避免 baseline 和 DBC 相互覆盖。

### Known risks / TODO

- 本次只验证了静态检查和单测，没有实际执行需要下载 Hugging Face 权重的 SFT/DPO smoke run；端到端训练仍依赖 `HF_TOKEN`、TPU/JAX 环境和真实模型下载权限。
- `tests/cli/peft_main_test.py` 与 `tests/examples/data/ultrafeedback_sft_test.py` 在当前环境中无标准输出，但退出码为 `0`；后续若要统一测试日志风格，可再切到项目统一的 test runner。
- 模型参数测试在默认 TPU backend 下会碰到已有的 `libtpu_lockfile` 环境问题，因此本次显式使用了 `JAX_PLATFORMS=cpu` 进行验证。

## 2026-03-16 - README availability check for SFT -> DPO flow

### Scope

- 无代码改动。
- 确认当前仓库里是否已经有一份把 `qwen2.5-1.5b` 的 `SFT -> DPO -> baseline/DBC` 全流程串起来的 README。

### Changed files

1. `develop.md`

### Validation

- 检查现有文档入口：
  - `examples/sft/ultrafeedback/README.md`
  - `examples/dpo/README.md`
  - `tunix/cli/README.md`

### Validation results

- 当前仓库没有“一份完整串起来的端到端 README”。
- 现有说明是拆开的：
  - `examples/sft/ultrafeedback/README.md` 说明 SFT 部分
  - `examples/dpo/README.md` 说明 DPO 与 DPO-from-SFT 部分
  - `tunix/cli/README.md` 只更新了入口级说明

### Known risks / TODO

- 如果后续需要降低实验复现门槛，建议再补一份单独的端到端 README，把：
  - 数据切分约定
  - SFT 命令
  - SFT 产物路径
  - DPO baseline / `outlier_l2` / `self_inf_batch` 命令
  - 结果目录结构
  串成一条完整流程。

## 2026-03-17 - Qwen2.5-1.5B UltraFeedback full SFT run

### Scope

- 调整 `qwen2.5-1.5b` 的 UltraFeedback full-SFT 配置，确保 full-weight 训练在当前 4-chip TPU worker 上可稳定跑通。
- 修复 SFT launcher 对 `training_config` 的覆盖方式，避免 `peft_main` 因 replace 语义丢失必填字段。
- 完成一轮正式 full-SFT，产出最终 `exported_model`、checkpoint 和训练指标。

### Changed files

1. `develop.md`
2. `examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft.yaml`
3. `examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft_lora.yaml`
4. `examples/sft/ultrafeedback/qwen2p5_1p5b_ultrafeedback.yaml`
5. `examples/sft/ultrafeedback/qwen2p5_1p5b_ultrafeedback_lora.yaml`
6. `examples/sft/ultrafeedback/run_qwen2p5_1p5b_ultrafeedback.sh`

### Validation

- TPU 可用性确认：
  - `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && python -c "import jax; print(jax.default_backend(), jax.device_count())"`
- 数据规模与长度校准：
  - 统计 `sft_fraction=0.25` / `eval_fraction=0.1` 下的 SFT/DPO prompt-disjoint 切分样本数
  - 统计 SFT 子集在 chat template 后的 token 长度分布与 `<=512/768/1024` keep-rate
- launcher 语法检查：
  - `bash -n examples/sft/ultrafeedback/run_qwen2p5_1p5b_ultrafeedback.sh`
- SFT 小预跑：
  - `python -m tunix.cli.peft_main examples/sft/ultrafeedback/qwen2p5_1p5b_ultrafeedback.yaml train_data_module="examples/data/ultrafeedback_sft.py:create_dataset(split='train_prefs', partition='sft', subset='train', sft_fraction=0.25, eval_fraction=0.1, seed=42, limit=128)" eval_data_module="examples/data/ultrafeedback_sft.py:create_dataset(split='train_prefs', partition='sft', subset='eval', sft_fraction=0.25, eval_fraction=0.1, seed=42, limit=32)" training_config.max_steps=2 training_config.eval_every_n_steps=1 training_config.checkpoint_root_directory=/tmp/sft_qwen2p5_preflight3_20260317_003140/checkpoints training_config.metrics_logging_options.run_name=qwen2p5-sft-preflight3-20260317_003140 training_config.metrics_logging_options.log_dir=/tmp/sft_qwen2p5_preflight3_20260317_003140/tensorboard exported_model_output_dir=/tmp/sft_qwen2p5_preflight3_20260317_003140/exported_model`
- 正式 full-SFT：
  - `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && set -a && source my_example/.env && set +a && RUN_TS=20260317_003657 RUN_ROOT=/home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657 ./examples/sft/ultrafeedback/run_qwen2p5_1p5b_ultrafeedback.sh full --ft-mode full`
- TensorBoard / artifact 检查：
  - 用 `tensorboard.backend.event_processing.event_accumulator` 读取 full run 的 `sft/train/*` 和 `sft/eval/*` 指标
  - `find /home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/exported_model -maxdepth 1 -type f | sort`
- chat-template sanity check：
  - `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && python - <<'PY' ... AutoTokenizer.from_pretrained(exported_model).apply_chat_template(...) ... PY`

### Validation results

- TPU backend 可用，设备数为 `4`。
- SFT/DPO 切分下的样本数为：
  - `sft_train=13810`
  - `sft_eval=1475`
  - `dpo_train=41327`
  - `dpo_eval=4523`
- SFT 子集经 chat template 后的长度统计显示：
  - `p50=400`
  - `p75=626`
  - `p90=872`
  - `p95=1050`
  - `p99=1466`
  - `<=768` keep-rate `= 11947 / 13810 = 0.8651`
- 由此将 full-SFT 主配置定为：
  - `mesh.shape="(2,2)"`
  - `max_target_length=768`
  - `peak_value=1e-5`
  - `warmup_steps=100`
  - `decay_steps=1500`
  - `weight_decay=0.05`
  - `max_grad_norm=1.0`
  - `gradient_accumulation_steps=4`
  - `eval_every_n_steps=100`
  - `save_interval_steps=100`
  - `max_steps=1500`
- 正式 full-SFT 成功跑完，训练在数据耗尽时结束于 `step 1493`，总训练时间约 `26m15s`。
- 正式 full run 输出目录：
  - `/home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657`
- 最终导出模型目录：
  - `/home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/exported_model`
- 最终保留 checkpoint：
  - `1200`
  - `1300`
  - `1400`
  - `1493`
- 关键指标：
  - 初始 `eval/loss=1.2246857`
  - 初始 `eval/perplexity=3.4030962`
  - 最后一次 eval（`step 1400`）`loss=1.0714879`
  - 最后一次 eval（`step 1400`）`perplexity=2.9197206`
  - 最佳 eval（`step 1200`）`loss=1.0713655`
  - 最佳 eval（`step 1200`）`perplexity=2.9193630`
  - 最终 train step（`1493`）`loss=1.2489289`
  - 训练过程中记录到的最小 train loss 在 `step 569`，为 `0.5949118`
- 导出的 tokenizer 可以正常应用 chat template，输出格式为 Qwen chat 模板：
  - `<|im_start|>system ... <|im_start|>user ... <|im_start|>assistant`

### Known risks / TODO

- 当前环境缺少 `torch`，因此没有对最终导出模型做一条真实生成的 CPU 推理；本次只验证了 tokenizer/chat template 和导出产物完整性。
- 最优验证点在 `step 1200`，而最终导出模型来自 `step 1493`。如果后续 DPO 更重视验证集最优初始化，而不是“最终一轮训练后权重”，建议直接从 checkpoint `1200` 再导出一份 best-model artifact。
- event file 显示 `step 1200 -> 1400` 基本处于平台区：
  - `1200: 1.0713655`
  - `1300: 1.0713979`
  - `1400: 1.0714879`
  如果后续要把 SFT 当论文主实验，建议再补一个基于 validation loss 的 best-checkpoint export。

## 2026-03-17 - Exported SFT model generation sanity check

### Scope

- 无代码改动。
- 使用 Tunix JAX sampler 对 `/home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/exported_model` 做实际生成验证，确认模型能按 chat 模板进行对话。

### Changed files

1. `develop.md`

### Validation

- 读取 TensorBoard event file，复核 full-SFT 的 `train/eval` 指标序列是否完整、是否存在后段异常反弹。
- 使用 Tunix 的 `automodel.create_model_from_safe_tensors` + `generate.sampler.Sampler` 在 `(2,2)` TPU mesh 上加载导出的 Qwen2.5-1.5B 模型。
- 实际生成测试 1：
  - 中文自我介绍
  - `SFT/DPO` 缩写问答
  - 简单英文算术
- 实际生成测试 2：
  - 中文翻译
  - 中文三条项目符号总结
  - Python 小函数
  - 两轮上下文记忆

### Validation results

- event file 指标完整，`eval/loss` 序列为：
  - `0: 1.2246857`
  - `100: 1.1186268`
  - `200: 1.0916263`
  - `300: 1.0864414`
  - `400: 1.0810978`
  - `500: 1.0770451`
  - `600: 1.0762211`
  - `700: 1.0749636`
  - `800: 1.0735710`
  - `900: 1.0722202`
  - `1000: 1.0718851`
  - `1100: 1.0714262`
  - `1200: 1.0713655`
  - `1300: 1.0713979`
  - `1400: 1.0714879`
- 生成 sanity check 结论：
  - 模型可以正常按 Qwen chat template 回答。
  - 基础对话、翻译、简单代码生成、简单多轮记忆均可用。
  - 专有缩写消歧与严格格式服从仍偏弱，说明这是一份“已能聊天”的 SFT 模型，但还不是强 instruction model。
- 代表性生成结果：
  - 自我介绍：`你好，我是一名AI助手，致力于为用户提供最优质的服务。`
  - 翻译：`Today's weather is nice, let's go for a walk in the park.`
  - Python 函数：能正确给出 `def reverse_string(s): return s[::-1]`
  - 多轮记忆：能记住“最喜欢的颜色是蓝色”，但回答成完整句 `我最喜欢的颜色是蓝色。`，没有完全遵守“只回答颜色”
  - 失败样例：对 `SFT 和 DPO 的区别` 这个 acronym-heavy 提问，模型把缩写错误展开成了无关组织名

### Known risks / TODO

- 这份模型已经具备可用 chat 能力，适合作为后续 DPO 初始化。
- 但如果论文或下游任务特别依赖强 instruction adherence，仍建议通过后续 DPO 进一步强化格式服从和术语理解。

## 2026-03-17 - SFT checkpoint cleanup

### Scope

- 无代码改动。
- 按用户要求，仅保留 `qwen2.5-1.5b` 这轮 full-SFT run 的最后一个 checkpoint，删除更早的中间 checkpoint。

### Changed files

1. `develop.md`

### Validation

- 清理前检查：
  - `find /home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/checkpoints -maxdepth 1 -mindepth 1 | sort`
- 删除中间 checkpoint：
  - 通过 Python `shutil.rmtree(...)` 删除 `1200`、`1300`、`1400`
- 清理后检查：
  - `find /home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/checkpoints -maxdepth 1 -mindepth 1 | sort`

### Validation results

- 清理前 checkpoint 为：
  - `1200`
  - `1300`
  - `1400`
  - `1493`
- 清理后仅保留：
  - `1493`
- `exported_model` 与 TensorBoard 目录未删除，后续仍可直接用于 DPO 初始化与结果回查。

### Known risks / TODO

- 无代码风险。

## 2026-03-17 - Add reproducibility note for Qwen2.5-1.5B SFT experiment

### Scope

- 新增一份独立的实验记录文档，说明这次 `qwen2.5-1.5b` UltraFeedback full-SFT 是如何实现的，以及哪条命令可以直接复现。

### Changed files

1. `develop.md`
2. `examples/ultrafeedback/paper_experiment.md`

### Validation

- 检查新增文档内容是否覆盖：
  - 实现文件
  - 数据切分
  - 超参数
  - 精确复现命令
  - 输出目录
  - 后续 DPO 应使用的模型路径
  - 基本 sanity check 命令

### Validation results

- 已新增独立文档：
  - `examples/ultrafeedback/paper_experiment.md`
- 文档中包含本次实际 full-SFT run 的实现说明与可复制命令。

### Known risks / TODO

- 文档里的固定输出路径引用的是当前已完成的这次 run；若用户重新训练，实际 `RUN_ROOT` 会随新的 `RUN_TS` 改变。

## 2026-03-17 - Rename experiment note to paper_experiment.md

### Scope

- 无逻辑改动。
- 将 `examples/ultrafeedback` 下的实验记录文档重命名为 `paper_experiment.md`。

### Changed files

1. `develop.md`
2. `examples/ultrafeedback/paper_experiment.md`

### Validation

- 检查原文件名引用：
  - `rg -n "qwen2p5_1p5b_sft_experiment\\.md|paper_experiment\\.md" -S .`
- 重命名后检查目录：
  - `ls -l examples/ultrafeedback`

### Validation results

- 实验说明文档现位于：
  - `examples/ultrafeedback/paper_experiment.md`
- `develop.md` 中相关引用已同步更新。

### Known risks / TODO

- 无代码风险。

## 2026-03-17 - DPO-from-SFT smoke validation and local INTERNAL model loading fix

### Scope

- 修复 `qwen2.5-1.5b` 的 DPO-from-SFT launcher，使其像 SFT launcher 一样通过临时 YAML 注入嵌套配置，避免 CLI 覆盖把整段 `training_config` / `dpo_config` 替换坏。
- 为 OSS 模式补充 `ModelSource.INTERNAL` 的本地路径加载分支，使 DPO 可以直接从本地 `exported_model` 目录加载 actor/reference base。
- 运行 DPO smoke 验证：
  - `baseline` 完整通过
  - `outlier_l2` 的 DBC 训练路径通过，但最终导出因磁盘写满失败

### Changed files

1. `develop.md`
2. `examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh`
3. `tunix/models/automodel.py`
4. `tests/models/automodel_test.py`

### Validation

- `bash -n examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh`
- `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && PYTHONPATH=/home/lhf_hongfu_gmail_com/tunix python tests/models/automodel_test.py`
- `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && PYTHONPATH=/home/lhf_hongfu_gmail_com/tunix python tests/cli/dpo_main_test.py`
- `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && set -a && source my_example/.env && set +a && RUN_TS=$(date +%Y%m%d_%H%M%S) && export RUN_TS && export RUN_ROOT=/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen2p5_1p5b_ultrafeedback_from_sft_baseline_full_smoke_${RUN_TS} && ./examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh smoke baseline /home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/exported_model`
- `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && set -a && source my_example/.env && set +a && RUN_TS=$(date +%Y%m%d_%H%M%S) && export RUN_TS && export RUN_ROOT=/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen2p5_1p5b_ultrafeedback_from_sft_outlier_l2_full_smoke_${RUN_TS} && ./examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh smoke outlier_l2 /home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/exported_model`
- `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && set -a && source my_example/.env && set +a && RUN_TS=$(date +%Y%m%d_%H%M%S) && export RUN_TS && export RUN_ROOT=/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen2p5_1p5b_ultrafeedback_from_sft_self_inf_batch_full_smoke_${RUN_TS} && ./examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh smoke self_inf_batch /home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/exported_model`
- `df -h /home/lhf_hongfu_gmail_com/tunix /tmp`

### Validation results

- DPO launcher shell syntax 检查通过。
- `tests/models/automodel_test.py` 通过，新增 `INTERNAL` 本地路径分支测试。
- `tests/cli/dpo_main_test.py` 通过，说明现有 DPO CLI 回归未破坏。
- `baseline` smoke run 成功完成训练、评估、checkpoint 保存与最终导出：
  - run 目录：`/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen2p5_1p5b_ultrafeedback_from_sft_baseline_full_smoke_20260317_032140`
  - exported model：`/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen2p5_1p5b_ultrafeedback_from_sft_baseline_full_smoke_20260317_032140/exported_model`
  - 最后一步 `dpo/eval/loss=0.69140625`
  - 最后一步 `dpo/eval/rewards/accuracy=0.27083334`
  - 最后一步 `dpo/train/loss=0.69140625`
  - 最后一步 `dpo/train/rewards/accuracy=0.5`
- `outlier_l2` smoke run 已进入 `CuratedDPOTrainer`，说明 DBC 代码路径生效：
  - run 目录：`/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen2p5_1p5b_ultrafeedback_from_sft_outlier_l2_full_smoke_20260317_032730`
  - 日志确认：`Dynamic batch curation enabled for DPO: using CuratedDPOTrainer (curation_variant=outlier_l2, curation_threshold=2.0, self_influence_dot_threshold=0.0, gradient_accumulation_steps=2).`
  - 训练与 checkpoint 保存通过，失败点仅在最终 `exported_model` safetensors 序列化
  - 失败原因为：`No space left on device (os error 28)`
- `self_inf_batch` smoke run 已进入 `CuratedDPOTrainer`，并完成实际训练步与 checkpoint 写入，说明另一条 DBC variant 也能跑通：
  - run 目录：`/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen2p5_1p5b_ultrafeedback_from_sft_self_inf_batch_full_smoke_20260317_034030`
  - 日志确认：`Dynamic batch curation enabled for DPO: using CuratedDPOTrainer (curation_variant=self_inf_batch, curation_threshold=2.0, self_influence_dot_threshold=0.0, gradient_accumulation_steps=2).`
  - 已完成 `step 0` eval、`checkpoint step 1` 保存，以及 `train step 1-4`
  - 该 variant 单步明显更慢；本轮 smoke 在确认训练路径正常后手动停止，并清理了临时目录以回收磁盘空间
- 当前根分区磁盘已满：
  - `df -h` 显示 `/dev/root` 可用空间仅 `76K`，使用率 `100%`

### Known risks / TODO

- 目前 baseline smoke 已确认可用，但在释放磁盘空间前，任何新的 full run 或 DBC smoke/final export 都有较高概率因磁盘空间不足失败。
- 本次验证已证明 `DBC` 训练分支本身可进入并执行；剩余问题是环境存储空间，而不是 DPO/DBC 逻辑错误。
- 为继续工作，本轮已清理 baseline/outlier/self-inf 的临时 smoke 目录，当前磁盘空间比验证当时宽裕，但 full DPO 仍建议在运行前确认根分区有充足余量。

## 2026-03-17 - DPO smoke re-validation under sandbox restrictions

### Scope

- 在当前 worker 上重新复跑 `qwen2.5-1.5b` 的 DPO smoke，确认最新代码和最新 SFT 导出模型在真实环境下的可用性。
- 记录 baseline 与 `outlier_l2` DBC 的真实 run 目录、指标和失败点。
- 回收本轮生成的 smoke 产物，为后续 full DPO 释放磁盘空间。
- 本轮无新的代码改动；仅做运行验证、磁盘清理和开发日志补录。

### Changed files

1. `develop.md`

### Validation

- `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && set -a && source /home/lhf_hongfu_gmail_com/tunix/my_example/.env && set +a && RUN_TS=$(date +%Y%m%d_%H%M%S) && export RUN_TS && export RUN_ROOT=/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen2p5_1p5b_ultrafeedback_from_sft_baseline_full_smoke_${RUN_TS} && ./examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh smoke baseline /home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/exported_model`
- `source /home/lhf_hongfu_gmail_com/.venvs/DPO/bin/activate && set -a && source /home/lhf_hongfu_gmail_com/tunix/my_example/.env && set +a && RUN_TS=$(date +%Y%m%d_%H%M%S) && export RUN_TS && export RUN_ROOT=/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen2p5_1p5b_ultrafeedback_from_sft_outlier_l2_full_smoke_${RUN_TS} && ./examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh smoke outlier_l2 /home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/exported_model`
- `python - <<'PY' ... event_accumulator ... PY`
- `df -h /home/lhf_hongfu_gmail_com/tunix /tmp`
- `du -sh /home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen2p5_1p5b_ultrafeedback_from_sft_baseline_full_smoke_20260317_141006 /home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen2p5_1p5b_ultrafeedback_from_sft_outlier_l2_full_smoke_20260317_141540 /home/lhf_hongfu_gmail_com/tunix/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657`
- `rm -rf /home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen2p5_1p5b_ultrafeedback_from_sft_baseline_full_smoke_20260317_141006 /home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen2p5_1p5b_ultrafeedback_from_sft_outlier_l2_full_smoke_20260317_141540`

### Validation results

- `baseline` smoke 在真实 TPU 环境下完整通过：
  - run 目录：`/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen2p5_1p5b_ultrafeedback_from_sft_baseline_full_smoke_20260317_141006`
  - 已写出 checkpoint：`1`、`20`
  - 已成功导出完整模型到 `exported_model`
  - event file：`tensorboard/events.out.tfevents.1773756620.t1v-n-21f197d2-w-0`
  - 最后一步 `dpo/eval/loss=0.69140625`
  - 最后一步 `dpo/eval/rewards/accuracy=0.27083334`
  - 最后一步 `dpo/eval/rewards/margin=-0.00072797`
  - 最后一步 `dpo/train/loss=0.69140625`
  - 最后一步 `dpo/train/rewards/accuracy=0.5`
- `outlier_l2` DBC smoke 在真实 TPU 环境下完成了训练与 checkpoint：
  - run 目录：`/home/lhf_hongfu_gmail_com/tunix/runs/dpo_qwen2p5_1p5b_ultrafeedback_from_sft_outlier_l2_full_smoke_20260317_141540`
  - 日志确认进入 `CuratedDPOTrainer`
  - 已写出 checkpoint：`1`、`20`
  - event file：`tensorboard/events.out.tfevents.1773756955.t1v-n-21f197d2-w-0`
  - 最后一步 `dpo/eval/loss=0.6875`
  - 最后一步 `dpo/eval/rewards/accuracy=0.375`
  - 最后一步 `dpo/eval/rewards/margin=-0.00099945`
  - 最后一步 `dpo/train/loss=0.6904296875`
  - 最后一步 `dpo/train/rewards/accuracy=0.5`
  - 最后一步 `dpo/train/dbc/keep_ratio=1.0`
- `outlier_l2` smoke 的失败点仅在最终导出：
  - `safetensors_rust.SafetensorError: No space left on device (os error 28)`
  - `exported_model` 当时只留下了不完整产物，不能当作成功导出。
- 本轮已删除这两个 smoke run 目录，当前磁盘空间回升到：
  - `/dev/root` 已用 `85G / 97G`
  - 可用空间约 `13G`

### Known risks / TODO

- 当前可以确认：
  - baseline full 命令可直接跑
  - DBC 训练分支可直接跑
- 当前不能确认：
  - 在当前磁盘余量下，长时间 full DPO 加最终导出是否始终稳定
- 若继续 full DPO，建议随时监控根分区空间，避免再次在导出阶段失败。

## 2026-03-17 - README pointers for DPO commands

### Scope

- 核对当前仓库里是否已经有写明 `qwen2.5-1.5b` 的 SFT -> DPO 命令位置。
- 本轮无代码改动，仅确认文档入口。

### Changed files

1. `develop.md`

### Validation

- `rg -n "run_qwen2p5_1p5b_ultrafeedback_from_sft\\.sh|outlier_l2|self_inf_batch|SFT_MODEL|full baseline" examples/ultrafeedback examples/dpo examples/sft -S`
- `sed -n '1,260p' examples/ultrafeedback/README.md`
- `sed -n '1,260p' examples/ultrafeedback/paper_experiment.md`

### Validation results

- `examples/ultrafeedback/README.md` 已写明端到端 workflow，包括：
  - SFT 命令
  - DPO smoke 命令
  - DPO full 命令
  - `baseline` / `outlier_l2` / `self_inf_batch`
- `examples/ultrafeedback/paper_experiment.md` 已写明：
  - 这次实际 SFT 实验的完整复现命令
  - 实际 `SFT_MODEL` 路径
  - 说明后续应把该 `SFT_MODEL` 传给 DPO launcher
- `examples/dpo/README.md` 也保留了 DPO launcher 的单独说明。

### Known risks / TODO

- 当前 `paper_experiment.md` 还没有把这台 worker 上的“带真实 `SFT_MODEL` 路径的 DPO full 命令”逐条写死；如果需要，可以后续补进去。

## 2026-03-17 - DPO hyperparameter summary review

### Scope

- 核对 `qwen2.5-1.5b` 这条 DPO-from-SFT recipe 的当前默认超参数。
- 本轮无代码改动，仅做配置解读。

### Changed files

1. `develop.md`

### Validation

- `sed -n '1,220p' examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft.yaml`
- `sed -n '1,220p' examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft_lora.yaml`
- `sed -n '1,240p' examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh`
- `nl -ba examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft.yaml | sed -n '1,220p'`
- `nl -ba examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft_lora.yaml | sed -n '1,220p'`
- `nl -ba examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh | sed -n '1,220p'`

### Validation results

- 当前 full DPO 默认配置已确认：
  - `batch_size=2`
  - `eval_batch_size=2`
  - `gradient_accumulation_steps=4`
  - `max_steps=2000`
  - `eval_every_n_steps=200`
  - `optimizer=adamw`
  - `peak_value=5e-6`
  - `warmup_steps=200`
  - `decay_steps=2000`
  - `weight_decay=0.1`
  - `max_grad_norm=0.1`
  - `beta=0.01`
  - `max_prompt_length=512`
  - `max_response_length=512`
- 当前 DBC 默认配置已确认：
  - `outlier_l2`: `curation_threshold=2.0`
  - `self_inf_batch`: `self_influence_dot_threshold=0.0`
- launcher 的 smoke profile 会覆盖为：
  - `max_steps=20`
  - `eval_every_n_steps=10`
  - `gradient_accumulation_steps=2`
  - `warmup_steps=2`
  - `decay_steps=20`
- LoRA DPO recipe 与 full DPO 的训练超参数相同，仅额外增加：
  - `rank=64`
  - `alpha=64`
  - `module_path=.*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*up_proj|.*down_proj`

### Known risks / TODO

- 当前这些 DPO 参数属于偏保守的 full-finetuning 设定；后续若 DBC 触发率仍偏低，可以优先从 `curation_threshold` 而不是整体学习率开始调。

## 2026-03-17 - Runtime environment clarification

### Scope

- 说明当前 `qwen2.5-1.5b` SFT/DPO workflow 相比之前的运行环境是否发生变化。
- 本轮无代码改动，仅记录结论。

### Changed files

1. `develop.md`

### Validation

- 核对当前 launcher 与文档中的环境入口：
  - `examples/sft/ultrafeedback/run_qwen2p5_1p5b_ultrafeedback.sh`
  - `examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh`
  - `examples/ultrafeedback/README.md`
  - `examples/ultrafeedback/paper_experiment.md`

### Validation results

- 运行环境本身没有本质变化：
  - 仍然使用 `/home/lhf_hongfu_gmail_com/.venvs/DPO`
  - 仍然在同一台 TPU worker 上运行
  - 仍然需要先加载 `my_example/.env` 或 `.env`
  - 仍然依赖 `HF_TOKEN`
- 相比之前变化的是 workflow 和代码路径，而不是基础环境：
  - 默认改成了 `qwen2.5-1.5b` 的 `SFT -> DPO`
  - DPO 会从本地 `SFT exported_model` 读取 actor/reference base
  - DPO 默认 `ft-mode` 现在是 `full`
- 运行时需要额外注意的环境问题：
  - 当前根分区空间需要在 full DPO 前留出足够余量
  - 在 Codex 沙箱里跑 TPU 任务会被 metadata/network 限制，因此真实训练需要在沙箱外执行

### Known risks / TODO

- 如果后续更换 worker、TPU 拓扑或 venv，再单独补记录。

---

## 2026-07-17 - GSM8K GRPO reproduction guide review

### Scope

- Read `GSM8K_GRPO_Reproduction_Guide.md` and checked the documented GSM8K GRPO reproduction workflow against the current repository.
- No code changes; this entry only records the review.

### Changed files

1. `develop.md`

### Validation

- `git branch --show-current`
- `git rev-parse HEAD`
- `git status --short`
- `sed -n '1,240p' GSM8K_GRPO_Reproduction_Guide.md`
- `sed -n '1,240p' ENV_SETUP.md`
- Reviewed `my_example/run_baseline.sh`, `my_example/run_dbc_self_inf_batch.sh`, `my_example/run_dbc_self_inf_group.sh`, and `my_example/run_dbc_outlier_l2.sh`.

### Validation results

- Current commit is the guide's reference commit: `a448e1f72cd7eafd6e490d66ec1066b10c5a5906`.
- Current local branch is `for_GRPO`, rather than the guide's branch name `my-changes`; the commit identity matches.
- All four documented experiment launchers exist and consistently invoke `my_example/run_grpo_gemma.sh`.
- The workflow expects `.venv_jax081`, JAX 0.8.1, and a TPU backend with four visible JAX devices for `--mesh-counts 4,1`.
- No training was launched during this documentation review.

### Known risks / TODO

- Before running, verify TPU visibility, dependency versions, model/data credentials, and sufficient disk space.
- Use unique checkpoint and metrics directories for every run; do not reuse a completed checkpoint when checking whether training starts normally.
- Run a short smoke test before committing TPU time to all four full experiments.

---

## 2026-07-17 - GSM8K server JAX import failure diagnosis

### Scope

- Diagnosed `ModuleNotFoundError: No module named 'jax'` after activating `.venv_jax081` on the TPU worker.
- Clarified that the `for_GRPO` branch is a user-owned copy of the collaborator's reference commit and is suitable for reproduction.
- No experiment code changes; this entry only records the diagnosis.

### Changed files

1. `develop.md`

### Validation

- Reviewed the dependency declarations in `pyproject.toml`.
- Reviewed the virtual-environment and JAX installation instructions in `ENV_SETUP.md`.
- Reviewed `.gitignore` rules for virtual environments.

### Validation results

- Git branches and clones carry repository files, but do not carry packages installed in a Python virtual environment.
- `jax[tpu]>=0.6.0,!=0.7.2,<=0.8.1` is declared in the `prod` optional dependency group.
- The `dev` optional dependency group is empty, so `pip install -e ".[dev]"` alone does not install JAX.
- The exception occurs before TPU discovery; it proves only that JAX is absent from the active interpreter.
- The previously used DTV-PPO environment is separate unless that exact environment is activated or deliberately reused.

### Known risks / TODO

- Confirm that `python` and `pip` both resolve inside `.venv_jax081` before installing.
- Install the TPU-compatible JAX 0.8.1 build and the project dependencies, then verify `backend=tpu` and four visible devices.
- If TPU detection fails after JAX imports successfully, separately inspect `/dev/accel*`, libtpu initialization, and TPU VM configuration.

---

## 2026-07-17 - TPU device discovery hang diagnosis

### Scope

- Interpreted the server output after JAX 0.8.1 installation: `/dev/accel*` is absent and JAX TPU client initialization hangs until interrupted.
- Accounted for the previously working DTV-PPO environment at `/home/jason_chia925_gmail_com/.venvs/PPO311` as a control environment.
- No experiment code changes; this entry only records the diagnosis.

### Changed files

1. `develop.md`

### Validation

- Compared the reported traceback with JAX TPU backend initialization behavior.
- Consulted the current Google Cloud TPU JAX troubleshooting, TPU FAQ, TPU runtime, and JAX slice documentation.

### Validation results

- JAX 0.8.1 imports successfully, so Python package installation is no longer the blocker.
- `ls: cannot access '/dev/accel*': No such file or directory` is not a Unix permission-denied result; `sudo` alone is therefore not the likely fix.
- The traceback ends in TPU client creation only because the user interrupted a hanging initialization; it does not establish a Python-code failure.
- A TPU slice can also make `jax.device_count()` wait for other hosts, so the resource topology and whether all workers must participate need to be confirmed.
- Testing the same bounded JAX probe in `PPO311` will distinguish a machine/resource problem from a new-environment compatibility problem.

### Known risks / TODO

- Confirm whether `PPO311` can still see TPU devices on this exact worker.
- Check `tpu-info`, PCI/device nodes, TPU-related environment variables, active TPU processes, TPU logs, and the Cloud resource READY/topology state.
- Do not restart TPU services or the VM until the read-only checks identify the resource type and likely failure mode.

---

## 2026-07-17 - TPU topology confirmed with PPO311 control environment

### Scope

- Reviewed the attached server diagnostics from the known-good `PPO311` environment.
- Determined the TPU topology and narrowed the new environment failure to its JAX/libtpu software stack.
- No experiment code changes; this entry only records the diagnosis.

### Changed files

1. `develop.md`

### Validation

- Reviewed the attached output for JAX device discovery, package versions, PCI devices, `/dev/vfio`, TPU agents, and running processes.
- Cross-checked current Google Cloud TPU and libtpu compatibility documentation.

### Validation results

- The worker is a single-host `v5p-8` resource (`ACCELERATOR_TYPE=v5p-8`, `WORKER_ID=0`) with four local TPU devices.
- `PPO311` successfully initializes the TPU with `jax==0.10.2`, `jaxlib==0.10.2`, and `libtpu==0.0.42.1`.
- Four Google accelerator PCI functions and `/dev/vfio` are present; absence of `/dev/accel*` is not evidence of failure on this VM.
- The TPU hardware, runtime, user permissions, and single-host topology are therefore not the blocker.
- The remaining likely difference is the exact `jax`, `jaxlib`, and `libtpu` package set in `.venv_jax081`.

### Known risks / TODO

- Capture `pip show` and `pip check` output from `.venv_jax081` before changing packages.
- Avoid switching the Tunix experiment directly to JAX 0.10.2 because the repository declares JAX `<=0.8.1` and compatibility has not been validated.
- If testing a newer libtpu with JAX 0.8.1, do so only in `.venv_jax081` and retain the known-good PPO311 environment unchanged.

---

## 2026-07-17 - GSM8K JAX 0.8.1 TPU environment validated

### Scope

- Reviewed the final dependency and TPU discovery output from `.venv_jax081`.
- Confirmed that no libtpu replacement is needed before beginning the GSM8K smoke test.
- No experiment code changes; this entry only records the environment validation.

### Changed files

1. `develop.md`

### Validation

- `python -m pip show jax jaxlib libtpu`
- `python -m pip check`
- Bounded JAX TPU probe using `timeout 30s`.

### Validation results

- `.venv_jax081` contains `jax==0.8.1`, `jaxlib==0.8.1`, and `libtpu==0.0.30`.
- `pip check` reports no broken requirements.
- JAX initializes the `tpu` backend and reports four local `TpuDevice` instances on process 0.
- The environment now satisfies the launcher's `--mesh-counts 4,1` device requirement.
- No package change is warranted; the proposed libtpu 0.0.42.1 experiment was not performed.

### Known risks / TODO

- The earlier one-off TPU initialization hang may have been transient startup latency or temporary runtime contention; retain bounded probes if it recurs.
- Before a full experiment, validate imports, credentials, disk capacity, and launch a short baseline smoke test with fresh metrics and checkpoint directories.
- Do not run PPO and GSM8K GRPO concurrently because both workloads may attempt to claim the same four TPU devices.

---

## 2026-07-17 - GSM8K credential-file check interpretation

### Scope

- Explained the meaning and impact of a missing `my_example/.env` file for the default GSM8K launcher.
- Checked the launcher's actual data source and authentication-variable usage.
- No experiment code changes; this entry only records the review.

### Changed files

1. `develop.md`

### Validation

- Reviewed `my_example/run_grpo_gemma.sh`, `my_example/auth.py`, `my_example/data.py`, and `my_example/model.py`.

### Validation results

- `my_example/.env missing` means only that the optional local shell environment file does not exist.
- The default launcher uses TFDS (`--source tfds`), not Kaggle, so `KAGGLE_USERNAME` and `KAGGLE_KEY` are not required for this default command.
- The launcher downloads `google/gemma-3-1b-it` from Hugging Face and reads `HF_TOKEN` or `HUGGINGFACE_TOKEN`; a token or an existing authenticated/cache state may therefore be required.
- WandB is disabled by the default launcher, so `WANDB_API_KEY` is not required.
- The environment and TPU import checks passed; the root filesystem currently has approximately 25 GB free.

### Known risks / TODO

- Verify Hugging Face authentication and acceptance of the gated Gemma model terms without printing the token.
- Verify that the GCS tokenizer path is readable and that enough disk remains for the model, dataset, checkpoints, and logs.
- Keep secrets out of Git, terminal transcripts, and chat messages; `my_example/.env` is intended to remain local.

---

## 2026-07-17 - Hugging Face token permission guidance

### Scope

- Clarified token recovery and minimum permissions needed to download the gated Gemma model for GSM8K reproduction.
- No experiment code changes; this entry only records credential guidance.

### Changed files

1. `develop.md`

### Validation

- Consulted the current official Hugging Face user-token, authentication, and gated-model documentation.

### Validation results

- If the full secret value of an existing token was not retained and the UI exposes only its name, a new token should be created rather than attempting to recover the old secret.
- Downloading `google/gemma-3-1b-it` requires only read access; write permission is unnecessary.
- A fine-grained token restricted to `google/gemma-3-1b-it` is the least-privilege option; a general read token is the simpler acceptable alternative.
- Gated-model access is granted to the individual Hugging Face account, so that same account must first accept/request access to the Gemma repository in the browser.

### Known risks / TODO

- Store the newly displayed `hf_...` secret immediately and do not expose it in chat, screenshots, shell history, or Git.
- Verify authentication with `hf auth whoami` and a small authorized model-file download before launching training.

---

## 2026-07-17 - Local `.env` loading clarification

### Scope

- Explained why `HF_TOKEN` is absent from the current shell even though `my_example/.env` exists.
- No experiment code changes; this entry only records credential-loading guidance.

### Changed files

1. `develop.md`

### Validation

- Rechecked the environment-loading block at the beginning of `my_example/run_grpo_gemma.sh`.

### Validation results

- Creating `my_example/.env` does not automatically export its variables into an already-running interactive shell.
- `run_grpo_gemma.sh` sources the file with `set -a`, so a correctly formatted `HF_TOKEN=hf_...` entry will be exported for the Python process when the launcher runs.
- A pre-source interactive-shell check is therefore expected to report that `HF_TOKEN` is unset.

### Known risks / TODO

- Validate `.env` shell syntax and variable presence without printing the secret.
- Test authenticated access to `google/gemma-3-1b-it/config.json` before starting the full training run.

---

## 2026-07-17 - Gemma gated-repository 403 diagnosis

### Scope

- Interpreted the Hugging Face `GatedRepoError` returned for `google/gemma-3-1b-it/config.json`.
- Clarified that model authorization is account-scoped rather than server-scoped.
- No experiment code changes; this entry only records access guidance.

### Changed files

1. `develop.md`

### Validation

- Reviewed the reported Hugging Face HTTP 403 traceback and its explicit authorization message.
- Rechecked the official Hugging Face gated-model access workflow.

### Validation results

- Network connectivity and the download code path reached Hugging Face successfully.
- Hugging Face rejected the authenticated account because it is not in the authorized list for `google/gemma-3-1b-it`.
- Moving to a colleague's server is not required; the user must request/accept access in the browser using the same account that issued the token.
- Once access is granted, the existing token may work if it has general read access; a repository-scoped fine-grained token may need its resource scope updated or a new token created.

### Known risks / TODO

- Do not use or request a colleague's personal token; gated-model authorization and license acceptance are individual-account matters.
- Repeat the small `config.json` download test after approval before starting training.

---

## 2026-07-17 - Gemma gated access validated

### Scope

- Recorded successful authenticated access to `google/gemma-3-1b-it` from the TPU worker.
- No experiment code changes; this entry only records the validation result.

### Changed files

1. `develop.md`

### Validation

- Loaded `my_example/.env` into the shell.
- Downloaded `google/gemma-3-1b-it/config.json` with `huggingface_hub.hf_hub_download` and `HF_TOKEN`.

### Validation results

- The request completed successfully and cached `config.json` under the user's Hugging Face cache.
- The token is valid, the account has accepted/received Gemma gated access, and the worker can reach Hugging Face.
- Hugging Face model authorization is no longer a blocker for the GSM8K experiment.

### Known risks / TODO

- A successful small-file test does not yet prove that the full model download will fit in the approximately 25 GB of remaining root-disk space.
- Proceed with a short baseline smoke run using fresh checkpoint and metrics directories before launching the complete experiment matrix.

---

## 2026-07-17 - GSM8K baseline smoke command review

### Scope

- Reviewed the guide, baseline wrapper, launcher argument ordering, dataset splitting, max-step calculation, checkpoint configuration, and post-train restore path.
- Derived a one-step baseline smoke command without changing experiment code or scripts.
- No experiment code changes; this entry only records the review.

### Changed files

1. `develop.md`

### Validation

- Reviewed `GSM8K_GRPO_Reproduction_Guide.md`.
- Reviewed `my_example/run_baseline.sh`, `my_example/run_grpo_gemma.sh`, `my_example/config.py`, `my_example/main.py`, and `my_example/train.py`.
- Confirmed the CLI help output supplied from the TPU worker.

### Validation results

- The existing scripts are the intended entrypoints; no new launcher is necessary.
- User arguments occur after launcher defaults, so duplicate argparse options are safely overridden by the final user-provided values.
- With `train_micro_batch_size=4`, `max_train_examples=8`, `train_fraction=0.9`, and one epoch, the effective training split contains one batch and produces `max_steps=1`.
- `save_interval_steps=1` exercises checkpoint save, and retaining post-train evaluation exercises the explicit Orbax restore path that previously failed around WandB monitoring.
- `--num-test-batches 1` assumes a default test micro-batch size of 32 in the compatibility layer; explicitly passing `--test-micro-batch-size 1` makes this smoke evaluate one example.
- The Grain jaxlib-extension warning concerns optional multiprocess worker profiling and is non-fatal for this launcher.

### Known risks / TODO

- The first smoke run still needs to download the full model, tokenizer, and TFDS data, so initialization may be much longer than the single training step.
- Monitor disk usage because only about 25 GB was free before the model/checkpoint download.
- Treat the smoke as successful only if training completes, checkpoint restore completes, post-train evaluation completes, and the wrapper exits with status 0.

---

## 2026-07-17 - First GSM8K baseline smoke failure diagnosis

### Scope

- Reviewed the attached output from the one-step baseline smoke run.
- Identified a missing environment dependency during TFDS GSM8K preparation.
- No experiment code changes; this entry only records the diagnosis.

### Changed files

1. `develop.md`

### Validation

- Read the complete attached traceback from `run_baseline.sh` through `tfds.data_source`.
- Mapped the failure to TFDS/etils importing the backport module `importlib_resources`.

### Validation results

- The run failed before model loading, TPU compilation, training, checkpoint save, or checkpoint restore.
- The fatal error is `ModuleNotFoundError: No module named 'importlib_resources'`.
- The Grain profiling warning is unrelated and non-fatal.
- The `data/train/gsm8k/1.0.0 has no dataset_info.json` warning indicates an absent or incomplete prepared TFDS dataset, which is expected around a failed first preparation attempt.
- Installing the missing `importlib_resources` package in `.venv_jax081` is the minimal next action; no repository code change is warranted.

### Known risks / TODO

- After installing the package, retry with fresh metrics/checkpoint directories while reusing the dataset directory.
- If TFDS then reports corruption or refuses to prepare the incomplete directory, move that specific GSM8K version directory aside before retrying rather than deleting broad data paths.

---

## 2026-07-17 - GSM8K run and log output-path review

### Scope

- Mapped the existing launcher's model, checkpoint, TensorBoard, stdout, and exported-result output controls.
- Derived a command layout that stores run artifacts under repository-root `runs/` and logs under repository-root `logs/` without modifying scripts.
- No experiment code changes; this entry only records the review.

### Changed files

1. `develop.md`

### Validation

- Reviewed the post-train save section in `my_example/main.py`.
- Reviewed output-path handling in `my_example/run_baseline.sh` and `my_example/run_grpo_gemma.sh`.
- Checked `.gitignore` coverage for root `runs/` and `logs/` directories.

### Validation results

- `--checkpoint-root` controls Orbax training checkpoints.
- `--output-dir` controls the final merged LoRA model directory and is destructively recreated by the program, so it must be unique per run.
- `--metrics-log-dir` controls TensorBoard event files.
- `TUNIX_MY_RESULT_DIR` controls wrapper stdout logs and exported summaries/plots.
- Root `runs/` and `logs/` are not currently ignored by Git.

### Known risks / TODO

- Do not stage or commit model/checkpoint/log artifacts from `runs/` or `logs/`.
- Keep timestamped unique run directories because an existing `--output-dir` is removed before saving the merged model.
- Continue monitoring the approximately 25 GB free root filesystem when storing artifacts inside the repository filesystem.

---

## 2026-07-17 - Second GSM8K baseline smoke failure diagnosis

### Scope

- Reviewed the next smoke-run traceback after TFDS dependency repair.
- Identified a missing GCS filesystem plugin while loading the tokenizer.
- No experiment code changes; this entry only records the diagnosis.

### Changed files

1. `develop.md`

### Validation

- Traced the reported failure from `Tokenizer` through `etils.epath`, `fsspec`, and the `gcs` filesystem registry.
- Rechecked the repository's `ENV_SETUP.md` GCS tokenizer dependency note.

### Validation results

- The Gemma model download completed successfully and is now cached locally.
- Mesh creation reported `(4, 1)` successfully.
- The fatal error is `ImportError: Please install gcsfs to access Google Storage` while reading `gs://gemma-data/tokenizers/tokenizer_gemma3.model`.
- `pip check` can still pass because `gcsfs` is an optional runtime plugin not represented as a broken installed-package requirement.
- The Qwix `rngs` message is a warning and did not cause this process termination.

### Known risks / TODO

- Prefer a `gcsfs` version matching the installed `fsspec` version to minimize dependency churn and avoid a `datasets`/`fsspec` conflict.
- Run `pip check` again after installation and perform a direct tokenizer-path read test before rerunning training.
- Reuse the cached model but use fresh run/log paths for the next smoke attempt.

---

## 2026-07-17 - Third GSM8K baseline smoke failure diagnosis

### Scope

- Reviewed the tokenizer initialization failure after GCS access was repaired.
- Identified an installed SentencePiece Python API incompatibility with the reference commit.
- No experiment code changes; this entry only records the diagnosis.

### Changed files

1. `develop.md`

### Validation

- Reviewed `tunix/generate/tokenizer_adapter.py` around tokenizer construction.
- Compared the repository call to the current SentencePiece package/release API information.

### Validation results

- TFDS configuration is correct for the one-step smoke (`max_steps=1`, one train batch, one validation batch, one test example).
- Model cache lookup, GCS tokenizer byte loading, and mesh creation succeeded.
- The fatal call is `SentencePieceProcessor.SetEncodeExtraOptions`, which is absent from the installed processor object.
- The repository has an unpinned `sentencepiece` dependency and expects the legacy CamelCase API.
- Pinning `sentencepiece==0.2.0` in the virtual environment is the minimal compatibility experiment; changing baseline tokenizer code is not yet warranted.

### Known risks / TODO

- Record the currently installed SentencePiece version and exposed method names before changing it.
- After pinning, directly instantiate the repository Tokenizer and encode a short string before restarting training.
- Continue treating the Qwix LoRA RNG message as a warning unless a later failure demonstrates that it affects initialization or training.

---

## 2026-07-17 - Post-train smoke evaluation cache-size diagnosis

### Scope

- Reviewed the smoke failure after one training step completed.
- Traced the post-train sampling length and cache-size calculations.
- No experiment code changes; this entry only records the diagnosis.

### Changed files

1. `develop.md`

### Validation

- Reviewed `my_example/main.py`, `my_example/eval.py`, `my_example/generate.py`, and `tunix/generate/sampler.py`.
- Compared the smoke overrides with launcher/parser defaults.

### Validation results

- The one-step baseline training completed successfully.
- Execution reached `main.py:290`, which is after the explicit checkpoint restore at lines 252-276; checkpoint save and post-train restore therefore completed without the prior WandB/Orbax failure.
- `SamplerWrapper.generate` defaults `max_generation_steps` to 768 and `evaluate` does not override it.
- The smoke's `--total-generation-steps 64` built a cache of `256 + 64 + 256 = 576`, while post-train sampling required a power-of-two padded prompt of 128 plus 768 generation steps, totaling 896.
- The formal launcher default uses 768 generation steps and creates a larger cache, so this specific error is caused by the shortened smoke override rather than the documented baseline command.

### Known risks / TODO

- For a fast clean-exit smoke, skip post-train evaluation while retaining the checkpoint restore and merged-model save path.
- For an end-to-end evaluation smoke, do not reduce `--total-generation-steps` below the standalone evaluator's fixed 768-step request unless the sampler/evaluator interface is changed in a separate branch.
- Do not modify the baseline command or shared sampler behavior solely to accommodate the artificial 64-token smoke override.

---

## 2026-07-17 - Merged LoRA safetensors export failure diagnosis

### Scope

- Reviewed the failure after training, checkpoint restore, and the start of merged-model export.
- Confirmed a JAX-to-NumPy type conversion defect in the shared safetensors saver.
- No experiment code changes; this entry only records the diagnosis.

### Changed files

1. `develop.md`

### Validation

- Reviewed `my_example/model.py`, `tunix/models/gemma3/params.py`, and `tunix/models/safetensors_saver.py`.
- Searched saver implementations and tests for `np.asarray`, `jax.device_get`, and safetensors NumPy serialization.

### Validation results

- Training completed and execution entered final merged-LoRA export.
- `safe_np.load_file` produces NumPy arrays, but `_apply_lora_delta` creates the LoRA delta with `jax.numpy` and augmented assignment can replace a state-dict value with `jaxlib._jax.ArrayImpl`.
- `safetensors.numpy.save_file` requires NumPy-compatible arrays and accesses `.ctypes`; JAX arrays do not expose that NumPy attribute.
- The failure is therefore a repository code compatibility defect, not a missing dependency, TPU problem, or smoke-only length setting.
- The Orbax training checkpoint remains the authoritative successful training artifact even though final merged-model export failed.

### Known risks / TODO

- The documented full baseline will likely encounter the same final export failure unless the saver conversion is fixed or merged export is bypassed.
- Any fix should explicitly transfer updated tensors to host NumPy arrays before `safe_np.save_file` and should include a regression test for JAX LoRA deltas.
- Respect the project constraint against altering protected baseline behavior; implement only after the user authorizes an appropriately scoped branch/fix.

---

## 2026-07-17 - Dependency-only workaround assessment for safetensors export

### Scope

- Assessed whether dependency pinning alone is a sound solution for the JAX-array failure in the NumPy safetensors saver.
- No experiment code changes; this entry only records the assessment.

### Changed files

1. `develop.md`

### Validation

- Rechecked project JAX constraints and the shared saver implementation.
- Consulted the official safetensors NumPy API contract and JAX host-transfer documentation.

### Validation results

- `safetensors.numpy.save_file` explicitly requires a dictionary of `numpy.ndarray` values.
- JAX `jnp.asarray` and JAX `astype` produce JAX arrays, while `jax.device_get` is the supported explicit device-to-host transfer.
- No safetensors version can be relied on to make the NumPy backend accept arbitrary TPU-backed JAX arrays as NumPy arrays.
- Older JAX/NumPy combinations might accidentally change mixed-array dispatch behavior, but relying on that would be fragile and would undermine the validated JAX 0.8.1 reproduction environment.
- A small explicit conversion at the serialization boundary is the correct and version-stable solution.

### Known risks / TODO

- Do not downgrade JAX solely for export; it could invalidate TPU, Flax, Orbax, and Tunix compatibility already established by the smoke run.
- If dependency archaeology is desired for exact historical reproduction, compare the collaborator's complete lockfile or `pip freeze`; absent that evidence, avoid speculative version changes.

---

## 2026-07-17 - Safetensors export fix-scope search

### Scope

- Performed a read-only scope search before modifying the merged-LoRA exporter.
- Compared local call sites, model tests, and the current upstream Tunix implementation.
- No experiment code changes; this entry only records the findings.

### Changed files

1. `develop.md`

### Validation

- Searched all local callers of `save_lora_merged_model_as_safetensors`, `_apply_lora_delta`, and `safe_np.save_file`.
- Reviewed Gemma3, Qwen2, and Qwen3 LoRA merge tests and their shared test base.
- Inspected the current `google/tunix` upstream `tunix/models/safetensors_saver.py`.

### Validation results

- The minimal production fix location is `tunix/models/safetensors_saver.py`, specifically `_apply_lora_delta` around lines 105-117.
- `my_example/model.py` and `tunix/models/gemma3/params.py` only delegate to the shared saver and do not need modification.
- GRPO, trainer, checkpoint, CLI, reward, and launcher files do not need modification.
- The shared saver is also used by Gemma3, Qwen2, and Qwen3, so the conversion fix should be architecture-neutral.
- Existing `test_save_lora_merged_model` coverage in `tunix/tests/lora_params_test_base.py` exercises the save operation; Gemma3-specific coverage is wired through `tests/models/gemma_all/gemma_params_test.py`.
- Current upstream Tunix still performs the same JAX delta into a NumPy state followed by `safe_np.save_file`, so no upstream fix was found to cherry-pick.

### Known risks / TODO

- Keep the patch at the serialization boundary: explicitly transfer only the computed LoRA delta to host NumPy before updating `base_state`.
- Validate both numerical merge correctness and serialized array types; run at least the Gemma3 LoRA merge test before repeating the TPU smoke export.
- Do not touch protected `robust_trainer.py`, CLI structure, reward logic, or baseline launcher behavior.

---

## 2026-07-17 - Fix JAX LoRA delta conversion for safetensors export

### Scope

- Fixed merged-LoRA safetensors export by explicitly transferring the computed JAX delta to host NumPy before updating the NumPy state dictionary.
- Kept the change limited to the shared serialization boundary; training, GRPO, checkpoint, reward, CLI, and launcher logic are unchanged.

### Changed files

1. `tunix/models/safetensors_saver.py`
2. `develop.md`

### Validation

- `python3 -m py_compile tunix/models/safetensors_saver.py`
- `git diff --check`
- Attempted: `python3 -m pytest tests/models/gemma_all/gemma_params_test.py -q`

### Validation results

- Python syntax compilation passed.
- Git whitespace validation passed.
- The local Gemma3 test could not run because the host Python 3.13 environment does not have `pytest` installed (`No module named pytest`).
- The patch now converts `combined_lora` with `np.asarray(jax.device_get(...))` before NumPy in-place addition, preserving `numpy.ndarray` values for `safetensors.numpy.save_file`.

### Known risks / TODO

- Run the existing Gemma3 LoRA merge test in the server's `.venv_jax081` environment if its test dependencies are installed.
- Repeat the one-step TPU smoke export to confirm the merged model writes and reloads successfully with real sharded LoRA parameters.
- The existing Qwix RNG warning remains outside this narrowly scoped serialization fix.

---

## 2026-07-17 - Postprocessing dependency warning diagnosis

### Scope

- Reviewed the successful merged-model export followed by failures in optional result export and plotting helpers.
- No experiment code changes; this entry only records the diagnosis and recovery commands.

### Changed files

1. `develop.md`

### Validation

- Reviewed `my_example/save_results_to_my_result.py` and `my_example/my result/plot_global_eval_rewards_sum.py` imports and CLI arguments.
- Confirmed the reported command exit status and exported model file listing.

### Validation results

- The merged LoRA fix is validated on the TPU worker: `model.safetensors` was successfully written at approximately 1.9 GB with its support files.
- The wrapper exited with status 0 as designed.
- `tensorboard` is required only by the optional metrics-export helper, and Pillow (imported as `PIL`) is required by the optional overlay plot helper.
- Both helper calls are guarded with `|| echo [warn]`, so their failures do not invalidate training, checkpoint restore, or model export.
- Installing `tensorboard` and `Pillow` permits rerunning postprocessing without retraining.

### Known risks / TODO

- Run `pip check` after installing the two helper dependencies because TensorBoard may constrain protobuf-related packages.
- A one-step smoke with skipped evaluation may not contain `global/eval/rewards/sum`; result export can legitimately skip that tag, and plotting may have no eligible series.
- Add these helper packages to a reproducible environment specification later if the paper workflow depends on automatic CSV/plot generation.

---

## 2026-07-17 - Nohup end-to-end baseline smoke preparation

### Scope

- Interpreted the postprocessing output from a checkpoint-reuse/export-only run.
- Prepared a fresh one-step nohup baseline smoke command that includes training-time validation, post-train evaluation, checkpoint restore, merged export, and result postprocessing.
- No experiment code changes; this entry only records the run plan.

### Changed files

1. `develop.md`

### Validation

- Reviewed the reported TensorBoard scalar tags from the export-only run.
- Rechecked smoke parameter interactions with max steps, checkpoint reuse, evaluation cadence, and standalone sampler cache sizing.

### Validation results

- The export-only run reused a step-1 checkpoint and skipped both pre/post evaluation, so the absence of training and evaluation result tags is expected.
- `actor/train/skipped_samples` is DBC-specific and can legitimately be absent from a baseline run.
- A fresh checkpoint plus `--eval-every-n-steps 1` should create training/evaluation metrics during the single training step.
- Retaining post-train evaluation should produce parsable `post-train` accuracy text in stdout.
- Omitting the artificial 64-token override restores the formal 768-step generation/cache relationship and avoids the previous 896-versus-576 cache failure.

### Known risks / TODO

- The end-to-end smoke is materially slower than the export-only check because it performs fresh rollout compilation, one validation rollout, and one post-train generation.
- Confirm completion using the nohup exit-code file, process state, success markers, TensorBoard tags, and exported result files.

---

## 2026-07-17 - GSM8K baseline end-to-end smoke accepted

### Scope

- Reviewed the completed nohup end-to-end baseline smoke evidence and assessed readiness for the first formal GSM8K experiment.
- No experiment code changes; this entry only records acceptance and the transition to the formal baseline run.

### Changed files

1. `develop.md`

### Validation

- Reviewed the nohup exit code, success markers, post-train accuracy, TensorBoard scalar tags, exported result files, and merged-model file listing supplied from the TPU worker.

### Validation results

- Nohup process completed with exit code 0.
- One-step training, training-time validation, post-train evaluation, checkpoint save/restore, and merged safetensors export all completed.
- Post-train single-example result was `num_correct=1/1`, `accuracy=100%`, `partial_accuracy=100%`, and `format_accuracy=0%`; this is a plumbing smoke result, not a paper-quality metric.
- TensorBoard contains actor train/eval metrics plus global train/eval reward and completion metrics.
- CSV, metadata JSON, stdout log, overlay PNG, and an approximately 1.9 GB merged model were generated successfully.
- The environment and pipeline are ready for the documented formal GRPO baseline experiment.

### Known risks / TODO

- The formal baseline uses far more training data and the full evaluation set, so runtime and storage requirements are substantially larger than the smoke.
- Check free disk space and save a package/environment snapshot inside the run directory before launch.
- Run the formal baseline first and inspect its outputs before launching the three DBC variants, avoiding concurrent TPU jobs.

---

## 2026-07-18 - Formal GSM8K baseline completion-check procedure

### Scope

- Prepared a layered acceptance check for the completed formal baseline: process exit, configured/effective steps, errors, pre/post evaluation, checkpoint restore, merged model, TensorBoard metrics, and exported paper artifacts.
- No experiment code changes; this entry only records the verification procedure.

### Changed files

1. `develop.md`

### Validation

- Reused the established timestamped `runs/gsm8k_baseline_full_*` and `logs/gsm8k_baseline_full_*` layout and the launcher's known success markers.

### Validation results

- A valid run should have exit code 0, `max_steps=691`, `Training complete`, pre/post evaluation output, a final actor checkpoint, successful merged-model export, and non-empty TensorBoard/result artifacts.
- Error scanning must distinguish fatal tracebacks from expected warnings and DBC-only missing tags.

### Known risks / TODO

- Exit code 0 alone is insufficient if the wrong run directory is selected or an old completed checkpoint caused training to skip.
- Compare config summary, progress/step metrics, checkpoint step, and wall-time metadata to prove the formal run actually trained.
- Do not start the next DBC experiment until the baseline acceptance evidence is reviewed.

---

## 2026-07-18 - Formal GSM8K baseline accepted and terminology mapped

### Scope

- Reviewed the full formal baseline evidence and accepted the run for the GSM8K experiment series.
- Recorded the terminology mapping: paper name `DTV`, laboratory name `DRPO`, and existing implementation/script name `DBC`.
- No experiment code changes; this entry only records acceptance and naming conventions.

### Changed files

1. `develop.md`

### Validation

- Reviewed configuration/dataset summaries, pre/post evaluation, TensorBoard train/eval series, checkpoint steps and files, merged model readability, result artifacts, and reward CSV endpoints.

### Validation results

- Formal configuration was used: 3072 maximum training examples, 768 batches, 691 train batches, 77 validation batches, 1319 test examples, and 691 effective steps.
- All 691 actor train loss/KL points and all 691 global train reward points were recorded.
- Evaluation reward contains 22 points from step 0 through step 672, matching evaluation every 32 steps.
- Accuracy improved from `623/1319 = 47.232752%` to `638/1319 = 48.369977%`, a gain of 15 correct answers or 1.137225 percentage points.
- Partial accuracy improved from `49.962092%` to `51.250948%`; format accuracy improved from `4.169826%` to `79.681577%`.
- Global eval reward sum increased from `-0.963474` at step 0 to `2.886364` at step 672.
- Checkpoints exist at steps 500 and 691; the final step-691 checkpoint is populated.
- The merged model is approximately 1.9 GB, is readable by safetensors, and contains 340 tensors.
- Stdout, evaluation metadata, reward CSV/metadata, and overlay PNG were generated.
- Baseline is accepted as valid and the first DTV/DRPO variant may proceed.

### Terminology mapping

- Paper/reporting: `DTV`.
- Laboratory discussion: `DRPO`.
- Repository implementation, CLI flags, and launchers: retain `DBC` to avoid unnecessary code changes.

### Known risks / TODO

- The 1.137-point final accuracy gain is a single run and should not yet be presented as a statistically robust effect without the planned variant runs and, if required, multiple seeds.
- Preserve the complete baseline run/log directories and environment snapshot as the comparison reference.
- Keep DBC names in commands and raw artifact metadata, while documenting the DTV/DRPO alias in paper tables and experiment notes.

---

## 2026-07-18 - First DTV/DRPO variant launch preparation

### Scope

- Confirmed the documented GSM8K experiment matrix and prepared the next full run: batch-level self-influence curation.
- Retained the repository's DBC launcher/flag naming while using DTV terminology in run directories.
- No experiment code changes; this entry only records the launch plan.

### Changed files

1. `develop.md`

### Validation

- Rechecked `GSM8K_GRPO_Reproduction_Guide.md` and `my_example/run_dbc_self_inf_batch.sh`.

### Validation results

- The guide defines four runs total: baseline, self-influence batch-level, self-influence GRPO-group-level, and L2 outlier curation.
- Baseline is complete, leaving three DTV/DRPO variants.
- The next launcher is `my_example/run_dbc_self_inf_batch.sh` with `TUNIX_REWARD_MODE=accuracy`.
- Output-path arguments and `TUNIX_MY_RESULT_DIR` can be added without altering the launcher's algorithm defaults.

### Known risks / TODO

- Run variants sequentially because each claims all four TPU devices.
- Confirm free disk space before each run; every successful run exports another approximately 1.9 GB merged model plus checkpoints and logs.
- Apply the same acceptance checks used for baseline before moving to the group-level variant.

---

## 2026-07-18 - Self-influence batch result evaluation procedure

### Scope

- Prepared technical acceptance and baseline-comparison checks for the completed DTV/DRPO self-influence batch run.
- Included checks proving that the self-influence filter actually activated, not merely that training completed.
- No experiment code changes; this entry only records the evaluation procedure.

### Changed files

1. `develop.md`

### Validation

- Reviewed `tunix/rl/self_inf_trainer.py`, `tunix/rl/rl_cluster.py`, the batch launcher, and result-export conventions.

### Validation results

- The self-influence trainer exposes `skipped_samples`, `self_inf_dot_mean`, `self_inf_dot_std`, and `self_inf_kept_fraction` through training metrics.
- A valid variant must match the baseline data/step configuration, finish at step 691, save/restore checkpoints, export a readable model, and produce evaluation artifacts.
- Mechanism activation requires SelfInfTrainer batch scope plus nontrivial filtering metrics; a run with all-zero skipped samples and kept fraction always 1 would not demonstrate DTV curation.
- Outcome quality should compare pre/post accuracy, final evaluation reward, reward trajectory, KL/loss stability, and filtering intensity against the accepted baseline.

### Known risks / TODO

- A single run can rank below baseline due to stochasticity; it is evidence for this seed/run, not a statistically robust conclusion.
- Extreme filtering (kept fraction near zero) can make updates unstable even if a final metric happens to improve.
- Do not launch the group variant until the batch run's technical validity and mechanism activation are both confirmed.

---

## 2026-07-18 - Self-influence batch outcome accepted and group variant prepared

### Scope

- Compared the completed batch-level DTV/DRPO run against the accepted baseline using identical pre-train evaluation and 22-point eval reward series.
- Prepared the next documented comparison: GRPO-group-level self-influence curation.
- No experiment code changes; this entry only records the result and launch transition.

### Changed files

1. `develop.md`

### Validation

- Reviewed baseline and batch-level pre/post accuracy, correct counts, format accuracy, and reward CSV summary statistics supplied from the TPU worker.

### Validation results

- Both runs have the same pre-train result: `623/1319`, `47.232752%`, supporting a direct paired configuration comparison.
- Batch-level DTV reached `711/1319 = 53.904473%`, versus baseline `638/1319 = 48.369977%`.
- Batch-level DTV exceeds baseline by 73 correct answers and 5.534496 percentage points post-train.
- Pre-to-post gain was `+6.671721` points for batch DTV versus `+1.137225` for baseline.
- Final eval reward was `3.451299` versus `2.886364` (+0.564935); mean across 22 eval points was `2.534792` versus `2.423259` (+0.111533).
- Batch DTV format accuracy was `73.161486%`, lower than baseline `79.681577%` by 6.520091 points; this tradeoff must be reported alongside the accuracy gain.
- The batch-level run is accepted as clearly stronger than baseline for this single run.

### Known risks / TODO

- Do not describe a single-run result as statistically significant without repeated seeds or an appropriate paired statistical analysis.
- Preserve mechanism-activation metrics (`skipped_samples`, kept fraction, dot statistics) for the eventual methods table even though outcome metrics are already favorable.
- Run the group-level method next, then the L2-outlier method, using the same baseline comparison protocol.

---

## 2026-07-18 - Group-level self-influence startup-log interpretation

### Scope

- Interpreted the absence of the `scope=group` trainer log during the first minute of the group-level run.
- No experiment code changes; this entry only records the runtime-state explanation and verification method.

### Changed files

1. `develop.md`

### Validation

- Rechecked `my_example/main.py` execution order and `tunix/rl/rl_cluster.py` trainer-selection logging.

### Validation results

- The run is currently in the pre-train evaluation (`3/1319`), before `RLCluster` and `SelfInfTrainer` construction.
- The `scope=group` log is emitted only after the complete pre-train evaluation finishes and the trainer is initialized.
- The outer nohup bash process can show 0% CPU while its Python child performs model evaluation.
- Immediate launch-mode verification should inspect the Python child command line for `--use-dbc-self-inf-group`.

### Known risks / TODO

- Do not terminate or restart solely because the trainer-selection line is absent during pre-train evaluation.
- Once pre-train evaluation completes, verify the explicit `SelfInfTrainer (scope=group, num_generations=4)` log before accepting that training phase configuration.

---

## 2026-07-17 - Export-only smoke metrics interpretation

### Scope

- Interpreted the result export output after installing TensorBoard and Pillow.
- Distinguished checkpoint/model-export validation from an evaluation-metrics smoke run.
- No experiment code changes; this entry only records the interpretation.

### Changed files

1. `develop.md`

### Validation

- Reviewed the reported installed package versions, `pip check`, event file path, stdout path, and result-export messages.
- Rechecked the flags used for the export-only rerun and the metrics tags exported by the helper.

### Validation results

- `tensorboard==2.21.0` and `Pillow==12.3.0` installed successfully, imports pass, and `pip check` reports no broken requirements.
- The export-only rerun reused a checkpoint already at `max_steps=1`, so it performed no new training step.
- Both pre- and post-train evaluation were explicitly skipped, so `no eval metrics found` is expected.
- `global/eval/rewards/sum` is absent because no evaluation ran; `actor/train/skipped_samples` is a DBC-specific signal and need not exist in a baseline run.
- The existing event file proves logging initialization, but this particular run is not suitable for accuracy/reward validation.

### Known risks / TODO

- Run a fresh one-step end-to-end baseline with a new checkpoint root, `eval_every_n_steps=1`, post-train evaluation enabled, and the default 768 generation length.
- Do not reuse a completed checkpoint when validating metrics, or training and its scheduled evaluation will be skipped.
- Evaluate only one test example for smoke purposes; do not interpret its accuracy as a paper result.

---

## 2026-07-18 - GRPO self-influence scope and leave-one-out feasibility analysis

### Scope

- Analyzed the mathematical and implementation differences between batch-level and group-level self-influence DTV in the current GRPO pipeline.
- Assessed the feasibility and experimental risks of adding batch-level and group-level DTV leave-one-out scoring.
- No code changes; this entry records analysis only.

### Changed files

1. `develop.md`

### Validation

- Read the current self-influence score implementation, GRPO sample-repeat/group construction, runtime trainer selection, configuration mapping, and launcher defaults.

### Validation results

- With the current defaults, one actor step contains 4 prompts times 4 generations, producing 16 per-completion gradients arranged as four contiguous prompt groups.
- Batch scope compares each completion gradient with the mean of all 16 completion gradients.
- Group scope compares each completion gradient only with the mean of the four generations belonging to the same prompt.
- Batch and group LOO scores can be computed from a total/group gradient sum minus the current sample gradient, without additional gradient evaluations.
- The core score change is locally small, but integration and scientific validation are medium difficulty because LOO can filter substantially more samples and requires explicit handling of singleton, malformed-group, and all-filtered cases.

### Known risks / TODO

- Preserve the existing batch/group behavior exactly and expose LOO only through new branches; do not modify the CLI argument structure.
- Define whether the LOO normalization uses `N-1`/`G-1` or preserves the original `N`/`G` scale. The zero-threshold mask is unchanged by this denominator choice, but logged scores and nonzero thresholds are not.
- Avoid silently falling back from malformed group LOO to batch LOO in paper experiments; fail clearly or emit an unmistakable configuration warning.
- Add exact synthetic-gradient tests and one-step TPU smoke tests before full LOO runs.

---

## 2026-07-18 - DTV LOO retention-cap and observability requirements

### Scope

- Refined the proposed GRPO DTV LOO design based on the established DPO/PPO protocol.
- No training code changes; this entry records requirements only.

### Changed files

1. `develop.md`

### Validation

- Checked the requested strict leave-one-out normalization and minimum-retention semantics against the current 4-prompts-by-4-generations GRPO batch layout.

### Validation results

- LOO must use the strict `N-1` definition for batch scope and `G-1` for group scope.
- Initial threshold selection keeps samples with nonnegative LOO score.
- A 75% filtering cap requires retaining at least `ceil(0.25 * population_size)` highest-score samples, implementable with vectorized top-k/rank selection and no Python sample loop.
- With current defaults, batch LOO retains at least 4 of 16 completions; group LOO should retain at least 1 of 4 completions in each prompt group to preserve group-level semantics.
- Existing methods and launchers must remain unchanged; LOO is exposed only through new `_loo` branches and independent launch scripts.

### Known risks / TODO

- Specify metric names and normalization explicitly so ordinary-DTV self/cross decomposition is not confused with the strict LOO score scale.
- Preserve aggregate TensorBoard metrics and add structured per-step/per-sample decision records containing score, terms, selection reason, group/generation indices, filtered counts, and optimizer-update status.
- Ensure the retention-cap implementation has deterministic tie handling and does not accidentally retain more or fewer samples at equal cutoff scores.

---

## 2026-07-18 - Add independent batch/group DTV LOO methods

### Scope

- Added strict leave-one-out self-influence curation as two opt-in GRPO methods: batch LOO and prompt-group LOO.
- Preserved all existing methods, CLI arguments, trainers, and launch scripts.
- Added a 25% minimum-retention cap; group LOO applies the cap independently within each prompt group.
- Added aggregate TensorBoard metrics and per-step JSONL selection records.

### Changed files

1. `tunix/rl/self_inf_loo_trainer.py` (new)
2. `tunix/rl/rl_cluster.py`
3. `tunix/rl/grpo/grpo_learner.py`
4. `my_example/run_dbc_self_inf_batch_loo.sh` (new)
5. `my_example/run_dbc_self_inf_group_loo.sh` (new)
6. `tests/rl/self_inf_loo_trainer_test.py` (new)
7. `develop.md`

### Implementation details

- Batch score uses strict `N-1`: `dot(g_i, sum_{j != i}(g_j) / (N - 1))`.
- Group score uses strict `G-1` over the other generations belonging to the same contiguous prompt group.
- Standard DTV self/cross decomposition and strict LOO score are recorded separately to keep their normalizations unambiguous.
- The normal mask keeps finite scores greater than or equal to zero. If fewer than `ceil(0.25 * population)` remain, a vectorized stable top-score mask supplies the minimum retained population.
- Batch LOO applies the cap over the actor batch. Group LOO vmaps the cap independently over prompt groups; with four generations, every prompt retains at least one completion.
- Existing `SelfInfTrainer` remains the default. `SelfInfLooTrainer` is selected only when a new launcher sets `TUNIX_DBC_SELF_INF_LOO=1` for its child process.
- JSONL records contain train step, scope, group/generation indices, raw self/cross values, standard score components, strict LOO scores, threshold/final/cap masks, cap status, filtered counts, and optimizer/effective-update indicators.

### Validation commands and results

- `python3 -m py_compile tunix/rl/self_inf_loo_trainer.py tests/rl/self_inf_loo_trainer_test.py tunix/rl/rl_cluster.py tunix/rl/grpo/grpo_learner.py`: passed.
- `bash -n my_example/run_dbc_self_inf_batch_loo.sh my_example/run_dbc_self_inf_group_loo.sh`: passed.
- `git diff --check`: passed.
- Verified no diff in `my_example/config.py`, `tunix/rl/self_inf_trainer.py`, `run_dbc_self_inf_batch.sh`, or `run_dbc_self_inf_group.sh`.
- Compared launcher argument sets: each LOO launcher matches its corresponding existing launcher, with only the scoped LOO environment and decision-log path added.
- Added formula/cap tests covering strict `N-1`, strict `G-1`, prompt-group isolation, highest-quartile fallback, nonnegative-mask preservation, and invalid group sizes.

### Known risks / TODO

- The local Codex environment does not contain JAX/Flax, so the new JAX unit tests could not be executed locally; run them in `.venv_jax081` on the TPU worker before the smoke test.
- Run one-step batch-LOO and group-LOO TPU smoke tests and verify `SelfInfLooTrainer` startup logs, TensorBoard tags, JSONL decisions, checkpoint save, post-train evaluation, and model export before full experiments.
- The structured log identifies samples by train step plus prompt-group/generation position; it intentionally avoids storing full prompt/completion token sequences to control log size.

---

## 2026-07-19 - Group self-influence result-check and next-run guidance

### Scope

- Prepared server-side checks for the completed `self_inf_group` GSM8K run.
- Confirmed from the reproduction guide that the next original-method experiment is L2 outlier curation with threshold `3.0`.
- No training code changes; this entry records operational guidance only.

### Changed files

1. `develop.md`

### Validation

- Re-read the experiment order in `GSM8K_GRPO_Reproduction_Guide.md`.
- Rechecked `my_example/run_dbc_outlier_l2.sh` defaults and argument forwarding.

### Validation results

- The documented original-method order is baseline, batch self-influence, group self-influence, then L2 outlier curation.
- The L2 launcher already fixes `--curation-threshold 3.0`, disables WandB, uses train micro-batch size 4, and accepts explicit metrics/checkpoint/output paths without changing the paper-comparison parameters.

### Known risks / TODO

- Accept the group run only after confirming exit code 0, the explicit `SelfInfTrainer (scope=group, num_generations=4)` line, complete pre/post evaluation, saved model/checkpoint outputs, and absence of traceback/error markers.
- Compare group post accuracy and reward trajectory against the same baseline and batch runs before drawing conclusions.
- Start L2 with fresh timestamped run/log/checkpoint directories so a completed checkpoint cannot silently skip training.

---

## 2026-07-19 - Accept completed group self-influence GSM8K run

### Scope

- Reviewed the complete server-side status, metrics, checkpoints, results, and model export for the group self-influence run.
- No code changes; this entry records result acceptance and interpretation.

### Changed files

1. `develop.md`

### Validation results

- Run root: `runs/gsm8k_dtv_selfinf_group_full_20260718_121640`.
- Log root: `logs/gsm8k_dtv_selfinf_group_full_20260718_121640`.
- Exit code is 0; training reached all 691 steps, post-train evaluation completed, and no traceback/error marker was found.
- Pre-train result: `623/1319 = 47.232752%`.
- Post-train result: `682/1319 = 51.705838%`; gain is 59 correct answers and 4.473086 percentage points.
- Group DTV is 44 answers and 3.335861 points above baseline post-train accuracy, but 29 answers and 2.198635 points below batch DTV.
- Post-train format accuracy is `66.034875%`, below baseline (`79.681577%`) and batch DTV (`73.161486%`).
- Curation metrics contain 691 points. Mean kept fraction is `0.984171`, mean skipped count is `0.253256` out of 16, and maximum skipped count is 5; group curation was active but mild.
- Eval reward contains 22 points; final reward is `2.780032` and mean reward is `1.997159`.
- Checkpoints 500 and 691 exist; merged `model.safetensors` is approximately 1.9 GB; exported result artifacts are present.

### Known risks / TODO

- The captured log does not contain the informational `SelfInfTrainer (scope=group, num_generations=4)` line. Group identity is supported by the previously recorded group launcher command and the launcher-generated `selfinf-group__...` result label, but the missing runtime line should be noted as an audit limitation.
- Continue with the documented L2 outlier experiment using threshold 3.0 and fresh paths.

---

## 2026-07-19 - Explain missing group-scope log line

### Scope

- Diagnosed why the completed group run did not contain the expected `scope=group` line.
- No code changes; this entry records the logging explanation only.

### Changed files

1. `develop.md`

### Validation results

- The grep expression was correct and would match the expected `SelfInfTrainer (scope=group, num_generations=4)` message if present.
- The message is emitted with `absl.logging.info` inside `tunix/rl/rl_cluster.py`.
- `my_example/main.py` does not raise Abseil verbosity to INFO; the captured run visibly includes warning-level Abseil output but not INFO-level trainer-selection output.
- Therefore the line was suppressed before reaching `nohup.log`; grep did not discard it.

### Known risks / TODO

- For future audit-grade runs, explicitly enable Abseil INFO logging or print the resolved DBC method in the launcher/main configuration summary before training.

---

## 2026-07-19 - L2 outlier run-check commands

### Scope

- Prepared the post-run validation procedure for the completed GSM8K L2 outlier experiment.
- No code changes; this entry records operational guidance only.

### Changed files

1. `develop.md`

### Validation results

- Confirmed `RobustTrainer` filters samples whose per-sample gradient L2 norm exceeds `mean + 3.0 * std`.
- Confirmed the L2-specific TensorBoard mechanism tags are `actor/train/skipped_samples`, `actor/train/grad_norm_mean`, and `actor/train/grad_norm_std`.
- Self-influence dot and kept-fraction tags are not expected for the L2 method.

### Known risks / TODO

- INFO-level `RobustTrainer` selection logs may be absent for the same Abseil verbosity reason observed in the group run; use launcher/result labels plus L2-specific metrics as supporting evidence.
- Compare the completed L2 result against baseline, batch DTV, and group DTV using accuracy, correct count, format accuracy, reward trajectory, and actual filtering frequency.

---

## 2026-07-19 - Accept L2 run and prepare DTV LOO launches

### Scope

- Reviewed the completed L2 outlier run, diagnosed the four-method comparison script syntax error, and prepared sequential batch/group LOO launch guidance.
- No code changes; this entry records result interpretation and operational commands only.

### Changed files

1. `develop.md`

### Validation results

- L2 exit code is 0; all 691 steps, post-train evaluation, checkpoints 500/691, result exports, and merged model completed without traceback.
- L2 post-train result is `733/1319 = 55.572403%`, a gain of 110 correct answers and 8.339651 points over its identical pre-train result.
- L2 filtered 257 samples over 691 x 16 opportunities (`2.324530%`) and activated on 257 steps; L2-specific grad-norm metrics are present and self-influence metrics are absent as expected.
- The comparison failure came from an invalid nested f-string on Python 3.11. The attempted replacement was pasted at the Bash prompt instead of inside a Python heredoc.
- Diffed each `_loo` launcher against its original counterpart: effective training arguments are identical; differences are limited to the LOO environment selector, structured decision-log path, and result label.

### Known risks / TODO

- Execute the JAX LOO unit test on the server before consuming a full TPU run.
- Run batch LOO and group LOO sequentially, never concurrently on the single-task TPU host.
- Use fresh timestamped checkpoint roots for both methods and verify the selection JSONL plus LOO TensorBoard metrics after the first optimizer step.

---

## 2026-07-19 - Verify batch DTV LOO launch

### Scope

- Reviewed the server preflight tests and initial batch-LOO process/log output.
- No code changes; this entry records runtime verification only.

### Changed files

1. `develop.md`

### Validation results

- Server is up to date with `for_GRPO`.
- All eight LOO helper tests passed under Python 3.11/JAX 0.8.1, including strict N-1/G-1 scores and independent group caps.
- No pre-existing Python training task was present before launch.
- The launched command uses `run_dbc_self_inf_batch_loo.sh`; run/log roots and labels contain `gsm8k_dtv_selfinf_batch_loo_full`.
- Configuration matches the original full runs: 3072 examples, 691 steps, train micro-batch size 4, and full pre/post evaluation.
- At the captured 22-second point the process was still loading/preparing for pre-train evaluation, so no optimizer step or selection JSONL was expected yet.

### Known risks / TODO

- After pre-train evaluation completes, require a nonempty `selfinf-batch_loo__...__selection.jsonl` and LOO TensorBoard tags before treating the algorithm selection as runtime-confirmed.
- Do not launch group LOO while this process is alive.

---

## 2026-07-19 - Batch DTV LOO completion-check guidance

### Scope

- Prepared end-to-end validation commands for the completed batch-level DTV LOO GSM8K run.
- No code changes; this entry records operational guidance only.

### Changed files

1. `develop.md`

### Validation coverage

- Process exit, full 691-step lifecycle, pre/post evaluation, errors, checkpoints, model export, and result artifacts.
- LOO TensorBoard metric presence and summary statistics.
- JSONL record count, batch scope, 16-sample shape, strict `raw_cross_sum / 15` score identity, standard self/cross decomposition, finite-score status, threshold versus cap selections, minimum four-sample retention, and optimizer/effective-update counts.
- Five-method comparison against baseline, batch DTV, group DTV, and L2 outlier using a Python-3.11-compatible formatter.

### Known risks / TODO

- Accept the run only if the structured LOO checks pass with no malformed records and all 691 optimizer steps are represented.
- Do not launch group LOO until batch LOO has exit code 0 and no surviving Python training process.

---

## 2026-07-19 - Batch LOO outcome interpretation and group LOO launch guidance

### Scope

- Recorded the completed Batch LOO GSM8K outcome and prepared structured-filter analysis plus Group LOO launch commands.
- No code changes; this entry records result interpretation and operational guidance only.

### Changed files

1. `develop.md`

### Validation results

- Batch LOO post-train result is `646/1319 = 48.9765%`, only 8 answers and 0.6065 points above baseline.
- Its pre-to-post gain is 1.7437 points, substantially below batch DTV (6.6717), group DTV (4.4731), and L2 outlier (8.3397).
- Batch LOO format accuracy is `63.9879%`, the lowest among the five completed methods.
- Current ranking is L2 outlier, batch DTV, group DTV, Batch LOO, baseline.

### Known risks / TODO

- Diagnose Batch LOO using the structured selection records before attributing the weak result to filtering aggressiveness: quantify cap frequency, retained fractions, ordinary-DTV versus LOO decision disagreement, self-term rescue rate, score distribution, and training-stage drift.
- Start Group LOO only after confirming Batch LOO exit code 0 and no active TPU Python process.
- Compare Group LOO against Group DTV as the primary matched comparison, while also retaining the global five/six-method ranking.

---

## 2026-07-19 - Interpret Batch LOO filtering summary

### Scope

- Analyzed all 691 structured Batch LOO selection records to explain the weak final GSM8K result.
- No code changes; this entry records experimental interpretation only.

### Changed files

1. `develop.md`

### Validation results

- Batch LOO filtered `4569/11056 = 41.325977%` of completion gradients and retained `58.674023%` after cap.
- The 25% cap triggered on only 17/691 steps (`2.460203%`) and restored only 25 samples, so the cap is not the cause of the weak result.
- No step had all-negative scores, no nonfinite scores occurred, and filtering showed almost no step correlation (`r=0.0377` post-cap), ruling out progressive filtering collapse.
- Counterfactual ordinary-DTV scores would retain `84.270984%`, while strict LOO thresholding retained `58.447902%`.
- `2855/11056 = 25.823082%` of samples were ordinary-DTV-positive but LOO-negative; there were zero reverse disagreements. The removed positive self term therefore accounts for the entire one-way decision gap.
- `41.552098%` of cross terms were negative, showing substantial disagreement between a completion gradient and the other 15 cross-prompt completion gradients.
- Standard self terms dominate the score scale: median self term is `0.279459`, versus median cross term `0.011871`; means and standard deviations are heavily distorted by rare very large gradient outliers.
- Filtering remains broadly stable across all ten training bins at roughly 37%-45%, so the behavior is structural rather than confined to early or late training.

### Interpretation / TODO

- The strongest supported explanation is cross-prompt gradient heterogeneity: strict batch LOO discards many useful prompt-specific updates once their positive self contribution is removed.
- The cap design worked as intended and did not materially alter the run; changing the 25% floor would not explain or likely repair this result.
- Group LOO is the correct next diagnostic because it replaces the 15 cross-prompt peers with the other three completions from the same prompt.
- When Group LOO completes, compare its filtering/disagreement summary directly with Batch LOO and its accuracy directly with Group DTV.

---

## 2026-07-19 - Clarify GRPO prompt groups versus optimizer batch

### Scope

- Clarified the distinction between a GRPO reward/advantage group and the actor optimizer batch used by the current experiment.
- No code changes; this entry records conceptual interpretation only.

### Changed files

1. `develop.md`

### Validation results

- The current configuration processes four prompts per training micro-batch and generates four completions per prompt, yielding 16 per-completion gradients in one actor update.
- GRPO reward comparison and group-relative advantage computation occur independently within each prompt's four completions.
- Standard GRPO does not simply retain the single best answer; all sampled completions can contribute, with positive or negative group-relative advantages.
- Group-level DTV compares a completion gradient with the four-gradient mean of its own prompt group; batch-level DTV compares it with the mean of all 16 gradients across four prompt groups.

### Known risks / TODO

- Keep the terms `prompt group`, `completion/trajectory`, and `optimizer batch` distinct in experiment documentation and paper descriptions.

---

## 2026-07-19 - Interpret actor/reference backbone-sharing warning

### Scope

- Diagnosed the Abseil warning about colocated actor/reference models not sharing a backbone.
- No code changes; this entry records runtime interpretation only.

### Changed files

1. `develop.md`

### Validation results

- Actor, reference, and rollout roles are intentionally mapped to the same TPU mesh with CPU offload disabled.
- The warning is emitted when Tunix expects a LoRA actor and colocated reference to share a backbone but runtime identity checks find separate backbone objects.
- The consequence stated by the code is an unnecessary model copy and increased HBM usage; it does not change reward, advantage, DTV/LOO scores, or optimizer mathematics.

### Known risks / TODO

- Treat the warning as harmless if training proceeds without `RESOURCE_EXHAUSTED`, OOM, or process termination.
- Do not alter model-sharing behavior mid-comparison because all methods should retain the same memory/runtime configuration.
- Consider backbone-sharing optimization only as a separate engineering change after the reproduction experiments are complete.

---

## 2026-07-20 - Group DTV LOO completion-check guidance

### Scope

- Prepared end-to-end validation and six-method comparison commands for the completed Group DTV LOO GSM8K run.
- No code changes; this entry records operational guidance only.

### Changed files

1. `develop.md`

### Validation coverage

- Exit status, 691-step lifecycle, pre/post evaluation, runtime errors, checkpoints, merged model, and result exports.
- Group LOO structured records: batch shape 16, four contiguous prompt groups, strict `G-1=3` score identity, independent per-group cap, at least one retained completion per prompt, nonfinite scores, and optimizer/effective-update counts.
- TensorBoard LOO metrics and final six-method accuracy/format ranking.

### Known risks / TODO

- Compare Group LOO primarily against Group DTV to isolate self-term removal within the same prompt-group scope.
- Compare Group LOO filtering rate and ordinary-DTV/LOO disagreement against Batch LOO to evaluate the cross-prompt heterogeneity hypothesis.

---

## 2026-07-20 - Six-method gap analysis and GRPO DTV research direction

### Scope

- Analyzed the six completed GSM8K outcomes and developed a prioritized direction for improving DTV-style GRPO curation.
- Reviewed primary GRPO, Dr. GRPO, DAPO, U-statistic/pruning-bias, and gradient-projection literature.
- No code changes; this entry records research analysis only.

### Changed files

1. `develop.md`

### Validation results

- L2 outlier leads batch DTV by 22 answers / 1.6679 points, group DTV by 51 / 3.8666 points, Group LOO by 68 / 5.1554 points, and Batch LOO by 87 / 6.5959 points.
- Group LOO improves over Batch LOO by 19 answers / 1.4405 points, supporting cross-prompt gradient heterogeneity, but remains 17 answers / 1.2889 points below Group DTV, showing that self-term removal/hard trajectory deletion is also harmful within prompt groups.
- Batch LOO's 25% minimum-retention cap is not a useful tuning knob at its current value: it triggered on only 2.46% of steps and changed only 25/11056 decisions.
- Aggregate test accuracy alone cannot establish significance; per-question paired predictions and multiple training seeds are required.

### Recommended direction

- First obtain the missing Group LOO filtering/disagreement summary and analyze the intersection between LOO-negative samples and L2 norm outliers.
- For a minimal ablation, replace the weak 25% floor with explicit maximum-filter budgets (batch keep 12/16, 14/16, or 15/16; group keep 3/4) and tune only on held-out validation data.
- The primary algorithmic direction should preserve GRPO group structure and avoid deleting complete trajectory gradients: aggregate completion gradients into prompt-group gradients, detect harmful direction at group level, then soft-weight or project only the conflicting component.
- A high-potential DTV/L2 synthesis is to act only on gradients that are both strongly anti-aligned and unusually large, rather than treating every negative dot product as harmful.

### Known risks / TODO

- Do not claim guaranteed superiority over L2 or tune repeatedly on the GSM8K test set.
- Hard pruning changes the GRPO gradient estimator and can introduce bias; any selective method needs an explicit bias discussion, correction, or conservative projection/weighting design.
- Preserve all existing six methods and add every new proposal as an isolated branch with its own launcher and diagnostics.

---

## 2026-07-20 - Add reproducible multi-seed runs and configurable LOO keep floor

### Scope

- Added an environment-based experiment seed without changing the CLI argument structure or existing launcher behavior.
- Added an environment override for LOO minimum retention, defaulting to the existing 25% behavior.
- Added a unified single-task seeded launcher supporting the six existing methods and batch/group LOO keep-75 ablations.

### Changed files

1. `my_example/seeding.py` (new)
2. `my_example/data.py`
3. `my_example/train.py`
4. `my_example/main.py`
5. `tunix/rl/rl_cluster.py`
6. `my_example/run_seeded_full.sh` (new)
7. `tests/my_example/seeding_test.py` (new)
8. `develop.md`

### Implementation details

- `TUNIX_EXPERIMENT_SEED` is optional. When absent, dataset shuffle remains 42 and rollout sampling retains its implicit PRNG key 0.
- Explicit seed 0 uses dataset shuffle 42 and rollout PRNG key 0, matching the random values used by all completed legacy runs; those runs are recorded as the common paired seed 0.
- Explicit seed `s` uses dataset shuffle `42+s` and rollout PRNG key `s`. The config summary prints the resolved seed mapping.
- `TUNIX_DBC_SELF_INF_MIN_KEEP_FRACTION` defaults to 0.25 and is read only by the opt-in LOO trainer branch.
- `run_seeded_full.sh METHOD SEED` creates fresh seed-labelled run/log/checkpoint/model paths, refuses to start alongside an active `my_example` Python task, and supports `batch_loo_keep75` / `group_loo_keep75` without modifying the original LOO launchers.

### Validation commands and results

- `python3 -m unittest tests/my_example/seeding_test.py`: four tests passed.
- Python compilation passed for seeding, data, train, main, cluster, and test modules.
- `bash -n my_example/run_seeded_full.sh`: passed.
- `git diff --check`: passed.
- Confirmed no diffs in any of the six existing method launchers.
- Unified launcher usage/argument validation returns the expected usage text and exit code 2.

### Known risks / TODO

- Run seed tests and a one-step TPU seeded smoke test after pulling this commit to the server.
- Use identical seed sets across methods. Treat the completed legacy run as paired seed 0 (`dataset=42`, `rollout=0`) and use seeds 5 and 21 (`dataset=47/63`, `rollout=5/21`) for the preliminary three-seed comparison; use five seeds for final paper claims if compute permits.
- Never launch more than one seeded method concurrently on the current TPU host.

---

## 2026-07-20 - Fix direct execution of seeding test

### Scope

- Fixed the new seed unit test's import path when invoked directly as `python tests/my_example/seeding_test.py` on the TPU server.
- Recorded the selected preliminary paired seed set: completed legacy seed 0 plus new experiment seeds 5 and 21.

### Changed files

1. `tests/my_example/seeding_test.py`
2. `develop.md`

### Validation commands and results

- `python3 tests/my_example/seeding_test.py`: four tests passed.
- `python3 -m unittest tests/my_example/seeding_test.py`: four tests passed.
- `bash -n my_example/run_seeded_full.sh`: passed.
- `git diff --check`: passed.

### Known risks / TODO

- The existing completed runs used dataset shuffle seed 42 and implicit rollout seed 0; they should not be described as a single global seed 42.
- Run all compared methods with the same paired seed set. Do not compare a method on seed 5 with another method only on seed 21.

---

## 2026-07-21 - Second TPU worker GRPO environment bootstrap guidance

### Scope

- Reconstructed the validated `.venv_jax081` installation procedure for a second, equivalent single-host four-device TPU worker.
- Included the runtime dependencies discovered during the first worker bring-up: `importlib_resources`, GCS filesystem support, SentencePiece legacy API compatibility, TensorBoard, and Pillow.
- No training code or launcher changes; this entry records operational guidance only（无代码改动）.

### Changed files

1. `develop.md`

### Validation basis

- Reviewed `ENV_SETUP.md`, `pyproject.toml`, `GSM8K_GRPO_Reproduction_Guide.md`, and the prior first-worker environment/debug records.
- Target package stack: Python 3.11, `jax==0.8.1`, `jaxlib==0.8.1`, `libtpu==0.0.30`, `sentencepiece==0.2.0`, matching `gcsfs`/`fsspec`, `tensorboard==2.21.0`, and `Pillow==12.3.0`.
- Required preflight checks cover editable Tunix imports, `pip check`, TPU backend with four local devices, GCS tokenizer access, gated Gemma access, and LOO unit tests.

### Known risks / TODO

- The second server must use a Hugging Face token belonging to an account that has accepted the Gemma gated-model terms; never commit `my_example/.env`.
- Do not start two jobs on the same TPU worker. Use distinct seed-labelled run, log, checkpoint, and model directories.
- If the second worker reports a different TPU topology, stop before training rather than forcing the existing `--mesh-counts 4,1` configuration.

---

## 2026-07-21 - Second worker Flax/JAX compatibility diagnosis

### Scope

- Diagnosed `ImportError: cannot import name 'Effect' from jax.extend.core` while importing Tunix on the new worker.
- No training code or launcher changes; this entry records dependency repair guidance only（无代码改动）.

### Changed files

1. `develop.md`

### Validation basis and result

- The repository fixes TPU JAX at 0.8.1 but declares only `flax>=0.11.1`, so a fresh resolver can install a much newer Flax release.
- Official Flax 0.12.7 metadata requires `jax>=0.10.0`, which is incompatible with this experiment's fixed JAX 0.8.1 environment.
- Official Flax 0.11.1 metadata requires `jax>=0.6.0`; pinning `flax==0.11.1` is the minimal compatible repair while retaining JAX/JAXLIB 0.8.1 and libtpu 0.0.30.

### Known risks / TODO

- Install the Flax and JAX pins in one resolver command so pip cannot silently upgrade JAX.
- Run `pip check`, Tunix/NNX imports, the bounded four-device TPU probe, and both seed/LOO tests before starting a full run.
- If `pip check` reports another package requiring JAX newer than 0.8.1, resolve that package explicitly rather than upgrading JAX.

---

## 2026-07-21 - Correct Flax pin for Qwix LoRA on second worker

### Scope

- Diagnosed the Batch LOO Keep-75 startup failure after the provisional Flax 0.11.1 downgrade.
- Corrected the environment target to the first worker's validated `flax==0.12.5` with JAX/JAXLIB 0.8.1.
- No training code or launcher changes; this entry records dependency repair guidance only（无代码改动）.

### Changed files

1. `develop.md`

### Validation basis and result

- The run resolved method, seed, dataset seed, rollout seed, and 0.75 keep floor correctly, then failed before evaluation/training inside Qwix LoRA parameter creation.
- Fatal error: Qwix calls `get_raw_value()` through the NNX variable API, but Flax 0.11.1 falls through to the wrapped JAX array and raises `AttributeError`.
- The first worker record shows successful Tunix/Qwix training with Flax 0.12.5 and JAX 0.8.1.
- Official Flax 0.12.5 metadata requires `jax>=0.8.1` and pins `optax==0.2.6`; unlike Flax 0.12.7, it does not require JAX 0.10.

### Known risks / TODO

- Pin `flax==0.12.5` and `jax[tpu]==0.8.1` together, then verify `jax`, `jaxlib`, `libtpu`, Flax, Optax, Qwix, and `pip check`.
- Use fresh run/log/checkpoint paths for the retry; retain the failed run as provenance but do not reuse its directory.
- Perform a one-step LOO smoke before committing to the full pre-evaluation/training run on the new worker.

---

## 2026-07-21 - GRPO DTV/LOO score-loss and Keep-75 analysis

### Scope

- Audited the current GRPO DTV, strict LOO, GRPO objective, advantage estimator, and L2 outlier implementations after the seed-0 Keep-75 results completed.
- Compared the GRPO implementation with the separate DTV-PPO project's explicit `score_loss=policy|total` design.
- No training code or launcher changes; this entry records algorithm analysis only（无代码改动）.

### Changed files

1. `develop.md`

### Results and code findings

- Batch LOO Keep-75 reached 632/1319 (47.9151%): 6 fewer than baseline, 79 fewer than original Batch DTV, and 101 fewer than L2.
- Group LOO Keep-75 reached 680/1319 (51.5542%): 42 above baseline, 2 below original Group DTV, and 53 below L2. It improved 15 answers over default Group LOO but essentially converged to original Group DTV rather than surpassing it.
- Both original DTV and LOO differentiate the actor trainer's registered `grpo_loss_fn` per completion. The score gradient is not separately configurable.
- The registered function is named a policy loss, but its differentiable objective is clipped GRPO policy surrogate plus `beta * KL`; the experiment uses `beta=0.08`. Thus the score is based on the total actor objective, not a KL-free policy-surrogate-only gradient.
- With `num_iterations=1`, the per-completion score is approximately driven by `-advantage * mean(log probability)` plus the KL gradient; clipping is usually not the principal first-iteration distinction.
- L2 and DTV use the same total-objective per-completion gradients, but L2 removes only extreme norm outliers (about 2.7%-2.9% in the recorded counterfactual analysis), whereas LOO sign filtering targets about 30.6% group / 41.6% batch before caps.
- Batch size at the trainer is 16 completions: four prompts times four generations. Batch Keep-75 can delete 4/16. Group Keep-75 can delete one of four within every prompt group, making 75% the highest nontrivial per-group hard-retention level; 100% becomes no filtering.
- GRPO intentionally creates centered positive and negative advantages within each prompt group. Pairwise/LOO anti-alignment can therefore represent useful contrastive structure rather than harmful contamination, especially with only three leave-one-out peers.

### Recommended research order

- Highest-priority ablation: decouple the DTV scoring loss from the update loss, compute DTV/LOO scores using the KL-free clipped GRPO policy surrogate, but apply the selected mask to the unchanged full GRPO update. This most closely transfers the successful PPO policy-loss lesson while preserving the DTV idea.
- Analyze policy-only versus KL-only gradient dot products and mask disagreement before another full run; verify whether KL cross terms dominate or flip selection decisions as training progresses.
- Do not continue treating the minimum keep floor as the primary solution. Batch 15/16 is a possible conservative diagnostic, but Group LOO has no nontrivial level between 3/4 and 4/4 without filtering only selected groups/steps.
- A later pure-direction alternative is to treat the full four-completion prompt gradient as the atomic GRPO unit and compare prompt-group gradients across the batch; this preserves within-prompt relative learning without introducing an L2 norm criterion.

### Known risks / TODO

- Seed-0 differences alone are not a paper-level significance result, but the large and mechanistically consistent LOO gaps justify changing the ablation direction before spending compute on all LOO seeds.
- Current LOO JSONL records scores and masks but not rewards, advantages, completion lengths, correctness, or KL/policy gradient decomposition, so they cannot identify the semantic class of removed completions without additional instrumentation.
- Any future score-loss-only branch must leave baseline, original DTV, L2, and existing LOO update objectives untouched and must use independent launchers/records.

---

## 2026-07-21 - GRPO versus PPO policy-score / total-update analysis

### Scope

- Compared the exact policy and total objectives in the current Tunix GRPO implementation with the separate DTV-PPO implementation.
- Assessed whether a future Batch/Group strict-LOO policy-score branch would remain faithful to the DTV principle and preserve existing methods.
- No code or launcher changes; this entry records design analysis only（无代码改动）.

### Changed files

1. `develop.md`

### Findings

- GRPO pure policy objective is a clipped likelihood-ratio surrogate weighted by prompt-group-relative advantages. Its full actor update adds `beta * KL(reference)` and has no learned value-function loss in this implementation.
- The current GRPO DTV/LOO score and update both use the full objective (`policy + beta*KL`), because the trainer differentiates one registered loss and reuses those gradients for masking and optimization.
- The DTV-PPO project's pure policy score uses the clipped PPO surrogate only. Its total objective adds entropy regularization and value regression; its code explicitly supports `score_loss=policy|total`.
- PPO advantages come from return/value estimation and are not structurally centered inside each small action group. GRPO advantages are centered and standardized among four completions from the same prompt, making within-group opposing policy gradients intentional rather than necessarily harmful.
- A policy-score/full-update GRPO branch remains within the DTV family: DTV attribution measures alignment with the task-improvement objective, while the unchanged full update retains the KL trust-region constraint. It should be described as an objective-decoupled or policy-attribution DTV variant, not as mathematically identical to the original total-gradient DTV.

### Proposed isolated experiment contract (not implemented)

- Add two opt-in methods only: strict Batch LOO policy-score and strict Group LOO policy-score.
- Score gradients: KL-free clipped GRPO policy surrogate.
- Selection: the existing strict `N-1` / `G-1` dot-product rule and the same configurable minimum-keep floor/cap behavior.
- Update gradients: unchanged full GRPO objective (`policy + beta*KL`) averaged over selected completions.
- Preserve all existing baseline, L2, original DTV, total-loss LOO, CLI behavior, launchers, and result formats.
- Add independent launchers/names and log both score-gradient and update-gradient diagnostics so the experiment is auditable.

### Known risks / TODO

- Do not implement policy-only scoring by simply setting `beta=0` on the existing loss, because that would also silently change the optimizer update objective.
- The new branch requires distinct score and update gradients (or an equivalent correct decomposition), so it may increase compute/memory versus the existing one-gradient-set trainer.
- Policy-only scoring removes KL contamination but does not by itself solve intentional GRPO positive/negative-advantage conflict; Batch and Group variants must therefore both be evaluated and mask behavior analyzed before full multi-seed runs.

---

## 2026-07-21 - Static-shape masking confirmation for GRPO DTV

### Scope

- Compared current GRPO DTV/LOO masking with the DTV-PPO policy-mask update path for JAX/TPU shape stability.
- No code or launcher changes; this entry records implementation analysis only（无代码改动）.

### Changed files

1. `develop.md`

### Findings

- Current GRPO original DTV, strict LOO, and L2 curation preserve the full completion batch and all per-sample gradient tensor shapes.
- They compute every per-sample forward/backward gradient first, create a fixed-length Boolean mask, broadcast-multiply the gradient leaves by that mask, sum, and divide by the number kept. No Boolean indexing or variable-size tensor is passed to the optimizer.
- For the standard full run, the trainer-side completion axis stays 16 (`4 prompts x 4 generations`); Group DTV only reshapes this statically to `[4, 4, ...]` for scoring and flattens it back.
- This is the same TPU-friendly principle as the DTV-PPO `policy_mask` path. It is not identical to the older PPO `DTVBatchFilter.filter_batch` path, which physically Boolean-indexes the batch and can change the leading dimension.
- GRPO masking currently removes all total-objective gradient contribution (policy and KL) for a dropped completion. It does not save per-sample forward/backward compute, because selection occurs after all gradients are materialized.

### Future policy-score branch constraint (not implemented)

- Preserve fixed shapes for both policy-score gradients and full-update gradients.
- Compute a fixed `[16]` Batch mask or `[4,4]` Group mask from policy-only gradients, then broadcast that mask over unchanged full (`policy + KL`) gradient leaves before reduction.
- Do not gather/compact selected samples and do not use data-dependent Python branching inside the jitted training step.

---

## 2026-07-21 - Add isolated Batch/Group Policy-LOO GRPO methods

### Scope

- Added opt-in strict Batch and Group Policy-LOO methods.
- Policy-only (`beta=0` score view) GRPO gradients are used only for DTV/LOO attribution; the unchanged full GRPO gradients (`policy + configured beta*KL`) are masked and passed to the optimizer.
- Reused the existing strict `N-1` / `G-1` score, per-group cap, static-shape mask, diagnostics, JSONL export, TensorBoard metric, and postprocessing implementation.
- Preserved baseline, L2, original DTV, and total-loss LOO commands and behavior.

### Changed files

1. `tunix/rl/self_inf_loo_trainer.py`
2. `tunix/rl/self_inf_loo_policy_trainer.py` (new)
3. `tunix/rl/rl_cluster.py`
4. `tunix/rl/grpo/grpo_learner.py`
5. `my_example/run_dbc_self_inf_batch_loo_policy.sh` (new)
6. `my_example/run_dbc_self_inf_group_loo_policy.sh` (new)
7. `my_example/run_seeded_full.sh`
8. `tests/rl/self_inf_loo_policy_trainer_test.py` (new)
9. `develop.md`

### Implementation details

- `PolicySelfInfLooTrainer` is selected only when both `TUNIX_DBC_SELF_INF_LOO=1` and `TUNIX_DBC_SELF_INF_LOO_POLICY=1` are set by the new launchers.
- `GRPOLearner` detects only the new trainer's policy-score setter and supplies a copy of the existing GRPO config with `beta=0.0`; the regular update loss closure retains the original configured beta.
- Existing LOO uses no score-loss override, so `score_grads is per_sample_grads` and retains its one-gradient-set total-loss path.
- New Policy-LOO computes fixed-shape policy score gradients and fixed-shape total update gradients. The mask is derived from policy gradients and broadcast over total gradient leaves before the existing masked mean.
- Default minimum keep remains 0.25. The existing `TUNIX_DBC_SELF_INF_MIN_KEEP_FRACTION` override is reused; seeded aliases for 0.75 are available without changing the method implementation.
- Policy JSONL records contain the same arrays/scalars as existing LOO plus `score_objective="policy"`. Existing total-loss LOO JSONL schema is unchanged.
- New Policy-LOO launchers own their environment selection, labels, stdout logs, and JSONL paths; existing Batch/Group LOO launchers are unchanged.

### New methods and launchers

- `batch_loo_policy` -> `my_example/run_dbc_self_inf_batch_loo_policy.sh`
- `group_loo_policy` -> `my_example/run_dbc_self_inf_group_loo_policy.sh`
- Optional later aliases: `batch_loo_policy_keep75`, `group_loo_policy_keep75`

### Validation commands and results

- Python compilation passed for both LOO trainers, RL cluster routing, GRPO learner wiring, and the new policy trainer test.
- `bash -n` passed for original LOO launchers, new policy launchers, and `run_seeded_full.sh`.
- New launchers are executable.
- `git diff --check`: passed.
- Confirmed no diffs in `tunix/rl/robust_trainer.py` or any existing baseline/L2/original DTV/total-loss LOO launcher.
- `tests/my_example/seeding_test.py`: four tests passed locally.
- New policy trainer test contains four checks, including an actual tiny NNX train step proving that a policy-derived mask is applied to a distinct set of total update gradients.
- Local JAX/absl tests could not run because the local interpreter lacks `absl`; run existing and new LOO tests in `.venv_jax081` on the TPU worker.

### Known risks / TODO

- Policy-LOO performs a second per-sample gradient pass, so compilation time, step time, and peak HBM can exceed existing total-loss LOO. Validate with a one-step smoke before a full run.
- The score/update split removes KL from attribution but cannot eliminate intentional conflict between positive- and negative-advantage GRPO completions.
- On the TPU worker, verify class selection (`PolicySelfInfLooTrainer`), default min keep 0.25, JSONL `score_objective=policy`, 16-sample static masks, successful optimizer/checkpoint/model export, and unchanged original LOO smoke behavior.

---

## 2026-07-22 - Fix Policy-LOO score-loss input adaptation

### Scope

- Diagnosed the new Policy-LOO TPU unit-test failure before launching a smoke/full experiment.
- Fixed score-only loss invocation so dictionary model inputs are unpacked identically to the existing total/update loss path.
- Removed deprecated NNX `.value` access from the tiny policy/update separation test.
- No change to baseline, L2, original DTV, total-loss LOO, CLI structure, strict LOO math, cap behavior, or total update objective.

### Changed files

1. `tunix/rl/self_inf_loo_trainer.py`
2. `tests/rl/self_inf_loo_policy_trainer_test.py`
3. `develop.md`

### Root cause

- The existing total loss is called through `per_sample_loss_fn`, which expands dictionary inputs as keyword arguments.
- The initial policy-score hook passed the entire dictionary positionally into the configured score loss. The synthetic test therefore reported missing `total`; a real GRPO example would likewise receive a dictionary instead of the expected `train_example`.
- The failure occurred before optimizer update and invalidates no experiment result because no Policy-LOO smoke/full run had started.

### Validation

- Server before fix: seed tests 4/4 passed, existing strict LOO tests 8/8 passed, Policy-LOO tests 3/4 passed; the fourth failed at score-loss argument binding.
- Local Python compilation, shell syntax, protected-file diff audit, and `git diff --check` are required after the patch.
- Rerun all four Policy-LOO tests on the TPU worker; only after 4/4 pass should the one-step Batch Policy-LOO smoke begin.

### Known risks / TODO

- The TPU smoke remains necessary to validate dual per-sample gradient compilation, HBM, class routing, JSONL records, checkpointing, and model export.

---

## 2026-07-22 - Batch Policy-LOO one-step smoke interpretation

### Scope

- Reviewed the completed Batch Policy-LOO seed-0 smoke output after the score-loss input fix.
- No code or launcher changes; this entry records validation interpretation only（无代码改动）.

### Changed files

1. `develop.md`

### Validation results

- The unified launcher selected `batch_loo_policy`, seed 0, dataset seed 42, rollout seed 0, and the new independent policy launcher.
- Smoke overrides intentionally resolved to `max_train_examples=8`, one train split batch, `max_steps=1`, and both pre/post evaluation disabled.
- One optimizer step completed in 170.72 seconds, training completed, the merged model exported successfully, and the wrapper exited with code 0.
- Run, log, checkpoint/model, stdout label, and result paths use a fresh `dtv_selfinf_batch_loo_policy` timestamp; there is no evidence of an old Batch LOO checkpoint or result being restored.
- The missing stdout eval summary is expected because `--skip-eval-before --skip-eval-after` was used. Any one-point TensorBoard eval reward generated internally is not a post-train GSM8K accuracy result and must not be used for method comparison.

### Known risks / TODO

- Inspect the smoke selection JSONL before the full run: require `scope=batch`, `score_objective=policy`, `min_keep_fraction=0.25`, 16 scores/mask entries, finite scores, and an applied/effective optimizer update.
- The first dual-gradient step includes JIT compilation and is slower than existing one-gradient methods; measure steady-state full-run steps before deciding whether implementation optimization is necessary.
- The smoke model is a disposable one-step artifact, not an experimental result. Use the no-override seeded launcher for the full 691-step run with pre/post evaluation.

---

## 2026-07-22 - L2 outlier versus strict DTV-LOO mechanism audit

### Scope

- Audited the mathematical and code-level differences between L2 outlier curation and strict Batch/Group DTV-LOO after the professor meeting.
- Distinguished counterfactual mask overlap on identical LOO-run gradients from retrospective comparison with the separately trained L2 run.
- Developed a prioritized future direction without modifying code or launchers（无代码改动）.

### Changed files

1. `develop.md`

### Mathematical findings

- L2 filters by magnitude only: `||g_i|| > mean(||g||) + 3*std(||g||)`. It removes both aligned and anti-aligned extreme gradients and ignores direction.
- Strict LOO filters by sign only: `g_i^T mean(g_-i) < 0`. Removing the self term means the score contains no direct norm-outlier test; norm scales the dot-product magnitude but does not determine its sign.
- Original DTV decomposes as `(self + cross)/N`; it filters only when `cross < -self`. Strict LOO filters whenever `cross < 0`. Thus LOO catches reverse extremes that self-protection would rescue, but also catches every mild negative conflict.
- A principled DTV continuum is `cross < -alpha*self`, `alpha in [0,1]`: alpha 0 is strict LOO, alpha 1 matches the original DTV negative-score condition, and intermediate alpha values target strong anti-alignment without adopting an L2 outlier rule.

### Code-level protection audit

- Both implementations compute fixed-shape per-completion gradients, construct a fixed mask, multiply and average selected total gradients, and always call `optimizer.update`; both share the same downstream Optax clipping/AdamW configuration.
- L2 has no explicit minimum-keep cap, top-k rescue, finite-score guard, per-group protection, or skip-update branch. Its practical protection is the conservative three-standard-deviation cutoff; for finite values at least one observation must be at or below the mean/cutoff.
- LOO explicitly excludes nonfinite scores, defaults to a 25% minimum keep, uses deterministic top-k rescue when the threshold retains too few, and applies the cap independently per prompt for Group LOO.
- Both renormalize by the number retained, rather than dividing zero-masked gradients by the original batch size. Consequently aggressive LOO selection changes both direction and the relative influence of survivors.

### Existing identical-gradient counterfactual evidence

- Batch LOO: 4594/11056 LOO-negative versus 316/11056 counterfactual L2 outliers, with only 134 samples in both. The negative-set precision for L2 is 2.92%, L2 recall is 42.41%, and Jaccard overlap is about 2.81%.
- Group LOO: 3385/11056 LOO-negative versus 303/11056 counterfactual L2 outliers, with 151 in both. Precision is 4.46%, L2 recall is 49.83%, and Jaccard overlap is about 4.27%.
- These low overlaps establish that strict LOO and L2 do not select approximately the same samples even when evaluated on identical gradients.
- They do not identify the exact samples removed by the independently trained L2 trajectory after step 1: the model, rollout, reward, and gradient streams diverge after different first updates, and the completed L2 run logged only aggregate norm/skipped statistics, not per-sample identities.

### Prioritized research direction

1. Finish the isolated policy-score LOO seed-0 experiment and measure whether removing KL from attribution changes mask rate and accuracy.
2. Offline-sweep `alpha` in `cross < -alpha*self` using existing JSONL `raw_cross_sum` and `raw_self`, reporting filter rate, cap activation, overlap with counterfactual L2, score margins, and training-stage stability. Choose on train/validation behavior, not GSM8K test accuracy.
3. If policy-only LOO remains broad, test an isolated strong-conflict Policy-LOO branch using the alpha margin while preserving the full policy+KL update and all existing methods.
4. Add shadow diagnostics that compute L2 and DTV decisions on the same live gradients without changing the chosen update; record prompt/sample identity, reward, advantage sign/magnitude, correctness, format, completion length, KL, norm, self/cross terms, and both masks.
5. Evaluate three paired seeds first and expand the key baseline/L2/best-DTV comparison to five seeds for paper claims.

### Known risks / TODO

- Matching L2's filter rate alone does not make the selected samples equivalent; report overlap and semantic/outcome categories.
- Hard completion deletion can disturb GRPO's intentional within-prompt positive/negative-advantage contrast. Group-level results must separately report whether a removed completion had positive, negative, or zero advantage.
- Do not claim exact actual-run L2/LOO sample overlap from current historical logs. Only the counterfactual L2 mask reconstructed on LOO gradients is exact for identical samples.

---

## 2026-07-22 - Correct DTV objective framing and inspect Batch LOO/L2 sample CSV

### Scope

- Corrected the interpretation of DTV as step-local gradient-direction curation: samples whose gradients oppose the intended aggregate update are masked before optimization to reduce drag on that update.
- Verified the exact minimum-retention implementation and analyzed the supplied 11,056-row Batch LOO versus counterfactual-L2 sample CSV.
- No training code, method, launcher, or analysis artifact was changed（无代码改动）.

### Changed files

1. `develop.md`

### Verification commands and results

- Inspected `_stable_top_k_mask`, `_capped_mask`, Batch/Group routing, masked gradient aggregation, and optimizer update in `tunix/rl/self_inf_loo_trainer.py`.
- Confirmed scores are sorted descending (`argsort(-safe_scores)`); when the nonnegative set is below the minimum, the final mask keeps the highest finite scores. Negative samples are therefore rescued from closest-to-zero downward, never from the most negative upward.
- Parsed the supplied CSV: 691 steps and 11,056 samples; 316 counterfactual L2 drops, 4,594 LOO-negative samples, and 4,569 final LOO drops.
- The 25% Batch cap rescued only 25 samples across 17 steps. Rescued scores ranged from about -0.438 to -0.0000394, with median about -0.00860.
- Exact categories were 134 both-drop, 182 L2-only, 4,435 LOO-only, and 6,305 kept-by-both.
- L2-only samples had median norm 23.82 and median positive LOO score 1.34; both-drop samples had median norm 19.47 and median LOO score -1.08; LOO-only samples had median norm 2.08 and median score -0.111; kept-by-both samples had median norm 1.94 and median score 0.135.

### Interpretation

- L2 magnitude filtering and DTV directional filtering solve different step-local problems. L2 removes extreme leverage regardless of whether it assists or opposes the aggregate direction; strict LOO removes directional opposition regardless of magnitude.
- The supplied data show L2 removes 182 extreme but directionally positive samples and misses 4,435 directionally negative non-outliers. Thus overlap alone is not a quality target; the important question is which rule better predicts the subsequent useful optimizer update.
- Because the 25% cap almost never activates in Batch LOO, it cannot explain the weak default result and changing the cap primarily changes behavior on steps with fewer than the requested retained count.

### Known risks / TODO

- Gradient dot products are measured in raw gradient space, whereas AdamW applies moment-based coordinate-wise preconditioning and global clipping. Raw alignment is a proxy for, not exactly equal to, contribution along the realized parameter update.
- The supplied CSV contains gradient and mask fields but not prompt/completion text, reward, advantage, KL, or correctness, so it supports exact numerical overlap analysis but not semantic sample-quality attribution.
- Evaluate total-loss versus policy-loss scoring according to which gradient best approximates the intended GRPO update direction; do not reframe DTV as an outlier detector.

---

## 2026-07-22 - Policy-component masking and GRPO trajectory-unit analysis

### Scope

- Analyzed whether PPO-style policy-score plus policy-component-only masking preserves the DTV idea and whether GSM8K GRPO needs rollout-to-trajectory segmentation.
- Inspected the active GRPO loss construction, completion masks, policy-score configuration, and current Policy-LOO gradient application.
- No training code, method, or launcher was changed（无代码改动）.

### Changed files

1. `develop.md`

### Verification results

- `grpo_loss_fn` builds a per-token clipped policy surrogate and, when beta is nonzero, adds `beta * KL`; the configured default beta is 0.08. There is no critic or value loss in this GRPO actor objective.
- The policy-score branch constructs the same GRPO loss with beta set to zero, so score gradients are policy-only.
- Current Policy-LOO uses policy-only gradients to compute the selection score but applies the resulting sample mask to complete per-sample total gradients. It is policy-score plus full-sample mask, not policy-component-only masking.
- Each GSM8K GRPO sample is one generated completion from a prompt until EOS or the generation limit. Its token losses are aggregated using the completion mask, and DTV masking removes the entire completion gradient while preserving static tensor shapes.
- A train micro-batch contains multiple prompt groups and multiple complete generations, but not a continuing environment rollout containing several independently terminated episodes. No additional stop-based trajectory splitting is required for the current GSM8K experiment.

### Interpretation

- Policy-score plus policy-component-only masking remains a direction-aware DTV variant if the target direction is explicitly defined as policy improvement: harmful policy contributions are removed while value/regularization components retain their own training signal. It is component-selective DTV rather than full-gradient DTV and must be named and reported separately.
- This distinction is important in PPO because the value objective is essential and semantically different from the policy objective. In current GRPO there is no value loss; the only non-policy component to preserve would be KL regularization, so the motivation is weaker but still testable.
- For GRPO policy-only masking, the intended aggregation must be specified carefully: average selected policy gradients over selected samples while averaging KL gradients over all samples, rather than accidentally changing either component's scale through a shared denominator.

### Known risks / TODO

- A filtered completion can still move actor parameters through its retained KL gradient under policy-component-only masking; claims must therefore concern harmful policy contribution, not complete sample removal.
- Multi-turn, tool-using, or environment-interacting GRPO would require revisiting trajectory boundaries and credit assignment, but that concern does not apply to the present single-completion GSM8K setup.
- If implemented later, policy-component-only masking must be an isolated new method; current Policy-LOO and all existing full-mask methods must remain unchanged.

---

## 2026-07-22 - Batch Policy-LOO seed-0 full result and Group run decision

### Scope

- Reviewed the completed Batch Policy-LOO seed-0 full-run result and verified the effective minimum-retention setting selected by the seeded launcher.
- Compared the result with the existing seed-matched baseline, total-loss Batch LOO, original Batch DTV, Group LOO, and L2 results.
- No training code, method, or launcher was changed（无代码改动）.

### Changed files

1. `develop.md`

### Verification results

- Command `run_seeded_full.sh batch_loo_policy 0` selects the independent Batch Policy-LOO launcher and does not match the `_keep75` suffix branch.
- The seeded wrapper explicitly unsets `TUNIX_DBC_SELF_INF_MIN_KEEP_FRACTION` for this method; `rl_cluster.py` therefore supplies the method default of 0.25.
- Post-train Batch Policy-LOO seed-0 result: 708/1319 exact, 53.6770% accuracy, 56.7854% partial accuracy, and 71.4936% format accuracy.
- Relative to total-loss Batch LOO, Policy-LOO gains 62 correct answers and 4.7005 percentage points exact accuracy.
- Relative to baseline, it gains 70 correct answers and 5.3071 points; it is 3 answers and 0.2274 points below original Batch DTV, and 25 answers and 1.8954 points below L2.

### Interpretation / next action

- The large controlled gain from changing score objective while retaining full masking and the same 25% cap is evidence that total-loss attribution was a major source of poor Batch LOO selection.
- Run Group Policy-LOO seed 0 next with the same seeded wrapper and default 25% group-local cap. This is the necessary paired scope comparison before trying Keep75 or policy-component-only masking.

### Known risks / TODO

- One seed does not establish superiority or equivalence; inspect the selection JSONL and complete the paired Group Policy-LOO result before seed expansion.
- Confirm the pre-train evaluation, config summary, exit code, and model export in the full log before including the result in aggregate tables.

---

## 2026-07-22 - Group Policy-LOO performance hypothesis

### Scope

- Assessed whether Group Policy-LOO could plausibly match L2 given the observed scope reversal between original DTV and strict total-loss LOO, plus the Batch Policy-LOO gain.
- No code, method, launcher, or result artifact was changed（无代码改动）.

### Changed files

1. `develop.md`

### Quantitative hypothesis

- Total-loss Group LOO exceeds total-loss Batch LOO by 1.4405 percentage points.
- Applying that observed scope gap to Batch Policy-LOO gives a rough Group Policy-LOO estimate of 55.1175%, only 0.4549 points (approximately 6/1319 questions) below the seed-0 L2 result of 55.5724%.
- The same estimate follows by adding the Batch policy-score gain of 4.7005 points to total-loss Group LOO.

### Interpretation

- Matching or exceeding L2 at seed 0 is plausible, because Group LOO compares completions within the same prompt and therefore aligns naturally with GRPO's group-relative advantages, while policy scoring removes KL-induced attribution from the selection rule.
- The estimate is not additive evidence: changing score objective changes masks and optimization trajectories, and the Group reference uses only three peers, making its score noisier and its 25% cap discrete at one completion per prompt.

### Known risks / TODO

- Wait for the actual Group Policy-LOO seed-0 result before deciding on Keep75, policy-component masking, or seed expansion.
- Treat a single-seed tie as candidate evidence only; use paired seeds for the final L2 comparison.

---

## 2026-07-22 - Group Policy-LOO seed-0 result and method-matrix decision

### Scope

- Recorded the completed Group Policy-LOO seed-0 result and compared all current GSM8K seed-0 methods, including default and Keep75 total-loss LOO variants.
- Evaluated the observed score-objective, scope, and retention-cap trends to prioritize the next experiments.
- No code, method, launcher, or result artifact was changed（无代码改动）.

### Changed files

1. `develop.md`

### Results

- Group Policy-LOO: 694/1319, 52.6156% exact, 54.7384% partial, 73.8438% format.
- It improves total-loss Group LOO by 29 correct answers and 2.1986 percentage points, exceeds original Group DTV by 12 answers and 0.9098 points, but trails Batch Policy-LOO by 14 answers and 1.0614 points.
- Batch Policy-LOO remains the best DTV-family LOO variant at 708/1319 (53.6770%), only 3 answers below original Batch DTV but 25 below L2.
- Group total-loss Keep75 improves default Group LOO by 15 answers and 1.1372 points, whereas Batch total-loss Keep75 reduces default Batch LOO by 14 answers and 1.0614 points.

### Interpretation / next action

- Policy scoring is strongly beneficial in both scopes, but its gain is larger for Batch (+4.7005 points) than Group (+2.1986 points). It restores Batch-over-Group ordering rather than creating a Group breakthrough.
- Keep75 is scope-dependent and is not a general cure for strict LOO. Group Policy-LOO Keep75 is worth one seed-0 run only to complete the interaction cell; Batch Policy-LOO Keep75 has low priority because the analogous total-loss Batch experiment degraded.
- Before adding another algorithm, compare policy versus total selection logs: filtering rate, cap activation, mask overlap, score-sign flips, and group advantage composition.
- For robust claims, prioritize paired seed 5/21 runs for L2 and the strongest DTV-family candidates rather than optimizing exclusively against seed 0.

### Known risks / TODO

- Policy and total variants follow different optimization trajectories, so historical masks are comparable diagnostically but not as identical post-step sample populations.
- A Group Policy-LOO Keep75 gain would complete an ablation but is unlikely by itself to close the full L2 gap based on current scope/cap trends.

---

## 2026-07-22 - Decide on PPO-aligned Policy-Only masking

### Scope

- Evaluated whether to inspect existing Policy-LOO logs or immediately add a PPO-aligned `dtv_loo_policy_only` method for Batch and Group scopes.
- Defined the intended component-selective update and an evidence-preserving implementation direction without changing training code.
- No code, method, launcher, or test was changed（无代码改动）.

### Changed files

1. `develop.md`

### Decision

- Perform a short existing-log audit first, then add the isolated Policy-Only variants. The audit can verify policy-score filtering/cap behavior but cannot counterfactually estimate the effect of retaining KL gradients, so it is not a substitute for the new experiment.
- The proposed method uses policy-only per-sample gradients for DTV-LOO scores, masks and averages only selected policy gradients, retains the KL-gradient mean over all samples, and applies their sum in the optimizer update.
- This matches the component-selective principle used in the PPO experiments while adapting PPO's retained value component to GRPO's retained KL regularization component.

### Implementation constraints for a later code-change turn

- Add independent Batch/Group method identities and launchers ending in `_loo_policy_only`; keep the default 25% minimum retention and all seeded hyperparameters unchanged.
- Preserve every existing baseline, L2, original DTV, total-loss LOO, and policy-score full-mask path exactly.
- Reuse the already computed total and policy per-sample gradients: the weighted KL component can be obtained as `total_gradient - policy_gradient`, avoiding a third per-sample gradient pass.
- Aggregate selected policy gradients over selected samples and the KL component over all samples with separate denominators; do not allow filtering rate to rescale the effective KL coefficient.
- Retain the existing selection records and add an explicit mask-application identity plus policy/KL update-norm diagnostics.

### Known risks / TODO

- Because Policy-Full and Policy-Only runs diverge after their first update, later mask differences reflect both update semantics and trajectory changes.
- A filtered sample still affects actor parameters through KL under Policy-Only masking, so the paper must claim removal of harmful policy contribution rather than complete sample removal.

---

## 2026-07-22 - Audit Policy-LOO selection strength before Policy-Only masking

### Scope

- Analyzed the completed Batch and Group Policy-LOO seed-0 selection audits to assess whether Policy-Only masking could plausibly close the L2 gap.
- No code, method, launcher, or result artifact was changed（无代码改动）.

### Changed files

1. `develop.md`

### Audit findings

- Batch Policy-LOO filters 2,346/11,056 samples (21.2192%), with no cap activation or rescued sample; mean retained fraction is 78.7808%.
- Group Policy-LOO has 2,370 negative samples but rescues 136 through group-local caps, finally filtering 2,234/11,056 (20.2062%) and retaining 79.7938%.
- Policy scoring substantially reduces selection aggressiveness relative to total-loss LOO: Batch drops from about 41.33% to 21.22%, and Group drops from about 28.45% to 20.21%.
- Batch permits 88/2,764 prompt groups to retain zero completions because its cap is batch-global. Group retains at least one completion per prompt; 377 groups retain one, 319 retain two, 465 retain three, and 1,603 retain all four.
- Despite filtering fewer samples and protecting every prompt group, Group Policy-LOO trails Batch Policy-LOO by 1.0614 points. Filter rate and group survival alone therefore do not explain performance ordering.
- Policy-score quartiles and median at exactly zero indicate substantial zero-score mass, plausibly from zero policy gradients when group-relative advantages collapse/tie; exact zero counts and reward/advantage linkage require additional diagnostics.

### Policy-Only assessment

- Policy-Only will not reduce the number of policy gradients removed: it preserves the same Policy-LOO mask and restores only the all-sample weighted KL-gradient contribution.
- It can close the L2 gap only if a material part of the remaining deficit is caused by removing KL regularization for the filtered approximately 20% of completions. It cannot fix loss caused by incorrectly masking useful policy gradients.
- The PPO analogy supports the experiment, but the expected effect may be smaller in GRPO because PPO preserves an essential value-learning objective, whereas current GRPO preserves only beta=0.08 KL regularization.
- The experiment remains well motivated as a cross-algorithm consistency ablation, but it should not be presented as likely to recover the entire 1.8954-point Batch Policy-LOO-to-L2 gap before evidence.

### Known risks / TODO

- Measure exact-zero policy-score frequency and, if available, relate it to tied rewards/zero advantages before interpreting score distributions.
- Add policy/KL component gradient-norm and cosine diagnostics to a future Policy-Only method so any gain or loss can be attributed to restored regularization rather than filter-count changes.

---

## 2026-07-22 - Policy-score zero mass and L2 scale comparison

### Scope

- Analyzed exact zero-score counts for Batch and Group Policy-LOO and reframed their comparison with L2 around update contribution rather than mask imitation.
- No code, method, launcher, or result artifact was changed（无代码改动）.

### Changed files

1. `develop.md`

### Findings

- Batch Policy-LOO has 2,346 negative (21.22%), 4,535 zero (41.02%), and 4,175 positive (37.76%) scores; 35.98% of nonzero policy gradients are negative.
- Group Policy-LOO has 2,370 negative (21.44%), 4,558 zero (41.23%), and 4,128 positive (37.34%) scores; 36.47% of nonzero policy gradients are negative before group-cap rescue.
- About 41% exact-zero mass is consistent with frequent tied group rewards/zero GRPO advantages. These samples are retained but contribute no policy gradient; under full masking they can still contribute KL because their score is nonnegative.
- Relative to the approximately 2.8% counterfactual L2 outlier rate observed on total-loss LOO gradients, Policy-LOO removes roughly seven times as many total samples and more than one third of samples with active policy signal.
- The current Policy selection records contain policy-gradient self/cross terms, whereas the actual L2 method thresholds total-loss gradient norms. Exact actual-L2 versus Policy-LOO sample overlap cannot be recovered from these historical runs; only a policy-gradient counterfactual L2 mask can be reconstructed.

### Interpretation

- L2 sparsely controls extreme gradient leverage regardless of alignment; Policy-LOO removes directional opposition among active policy gradients regardless of norm. Their mask rates and overlap are not objectives that DTV should match.
- Batch Policy-LOO reaching within 1.8954 points of L2 despite removing a much larger, direction-defined set supports the usefulness of policy-alignment curation, but also leaves open whether useful minority policy directions are being discarded.
- Policy-Only masking preserves the same policy selection and therefore cannot reduce directional filtering; it tests whether restoring all-sample KL regularization improves the outcome enough to match or exceed L2.

### Known risks / TODO

- Exact-zero score alone does not prove the cause is tied reward; record reward and advantage alongside future selection decisions to verify.
- Use a same-gradient shadow diagnostic for exact L2/DTV sample comparisons in future runs; do not infer actual historical overlap from separately trained trajectories.

---

## 2026-07-22 - Interpret Policy-LOO/L2 overlap and clarify KL versus group baseline

### Scope

- Analyzed counterfactual policy-gradient L2 overlap for Batch/Group Policy-LOO.
- Clarified the distinction between GRPO group-relative reward/advantage evaluation and KL regularization when defining a Policy-Only mask.
- No code, method, launcher, or result artifact was changed（无代码改动）.

### Changed files

1. `develop.md`

### Overlap findings

- Batch: 253 policy-gradient L2 outliers (2.29%) versus 2,346 Policy-LOO drops (21.22%), with 98 in both; precision 4.18%, L2 recall 38.74%, and Jaccard 3.92%.
- Group: 243 policy-gradient L2 outliers (2.20%) versus 2,234 final Policy-LOO drops (20.21%), with 83 in both; precision 3.72%, L2 recall 34.16%, and Jaccard 3.47%.
- Policy-LOO is therefore not an L2 proxy: more than 95% of its filtered samples are not policy-norm outliers, and most policy-norm outliers are retained because they are not directionally negative.

### Objective clarification

- All generated completions already participate in reward computation and group-relative advantage normalization before DTV selection. A later full-gradient mask does not retroactively remove a filtered completion from the prompt group's reward mean/standard deviation or from peers' advantages.
- KL does not evaluate which completion is good relative to its prompt group. It measures deviation between the current policy and reference policy for each completion and supplies a regularization gradient.
- The precise Policy-Only objective is therefore: compute group-relative advantages from all completions; compute DTV-LOO scores from policy gradients; mask harmful policy-gradient contributions; retain the KL regularization gradients from all completions.

### Paper framing

- Describe the proposed method as `policy-gradient selective masking with unmasked KL regularization`, not as retaining filtered samples in the group baseline (which already occurs in existing Full Mask methods).
- Student analogy: every student's score remains in class mean/standard-deviation evaluation; advice that conflicts with the chosen teaching direction is excluded from the policy change; nevertheless every answer still contributes a constraint that teaching should not drift too far from the reference curriculum.

### Known risks / TODO

- If the intended requirement is only to retain filtered completions in GRPO advantage normalization, no new Policy-Only method is required because current methods already satisfy it.
- A new method is justified only to test unmasked all-sample KL regularization, and its result should be interpreted as a component-masking ablation rather than a change to group-relative credit assignment.

---

## 2026-07-22 - Add Batch/Group DTV-LOO Policy-Only methods

### Scope

- Added isolated Batch and Group `dtv_loo_policy_only` methods while preserving every existing baseline, L2, original DTV, total-loss LOO, and policy-score full-mask method and launcher behavior.
- Policy-only methods compute strict LOO selection from policy gradients, mask selected policy-gradient contributions, and retain the beta-weighted KL-gradient mean over all samples.

### Changed files

1. `tunix/rl/self_inf_loo_trainer.py`
2. `tunix/rl/self_inf_loo_policy_only_trainer.py` (new)
3. `tunix/rl/rl_cluster.py`
4. `my_example/run_dbc_self_inf_loo_policy_only.sh` (new)
5. `my_example/run_dbc_self_inf_batch_loo_policy_only.sh` (new)
6. `my_example/run_dbc_self_inf_group_loo_policy_only.sh` (new)
7. `my_example/run_seeded_full.sh`
8. `tests/rl/self_inf_loo_policy_only_trainer_test.py` (new)
9. `develop.md`

### Implementation details

- Added backward-compatible aggregation hooks to `SelfInfLooTrainer`; their defaults reproduce the existing full-mask gradient, loss, and auxiliary aggregation paths.
- Reused the existing total and policy per-sample gradient passes. The already beta-weighted KL component is `total_gradient - policy_gradient`, avoiding a third backward pass.
- Final Policy-Only update is `mean(selected policy gradients) + mean(all weighted KL gradients)`, with independent denominators so filtering does not rescale effective beta.
- Selection JSONL retains every existing score/self/cross/mask/cap record and adds `mask_application=policy_only`; `score_objective=policy` remains unchanged.
- Added the gated environment selector `TUNIX_DBC_SELF_INF_LOO_POLICY_ONLY`; it requires both existing LOO and policy-score selectors, so no existing environment combination routes to the new trainer.
- Added seeded method names `batch_loo_policy_only` and `group_loo_policy_only`, both using the existing default 25% minimum retention and the same full-run parameters as their Policy-Full counterparts.

### Validation commands and results

- `bash -n` passed for the generic Policy-Only launcher, Batch/Group wrappers, and `run_seeded_full.sh`.
- `python3 -m py_compile` passed for all modified/new Python modules and the new test.
- `git diff --check` passed.
- The three JAX/Flax unit-test commands were attempted locally but could not start because the local desktop Python environments do not contain `absl` (and the repository has no local TPU/JAX virtualenv). This is an environment limitation, not a test assertion failure.

### Known risks / TODO

- Run existing total-LOO and Policy-Full unit tests plus the new Policy-Only test in `.venv_jax081` on the server.
- Run a one-step TPU smoke for Batch Policy-Only and inspect trainer selection, `mask_application`, finite scores, effective update, checkpointing, and export before the full seed-0 run.
- Historical methods are protected by default aggregation hooks and gated routing, but the server regression tests remain required before experimental use.

---

## 2026-07-23 - Reduce Policy-Only TPU compilation peak memory

### Scope

- Fixed a Group Policy-Only first-step TPU compile/load failure: XLA attempted to reserve 29.79 GiB with only 28.70 GiB reservable.
- Changed only the new Policy-Only aggregation implementation; all existing methods and routing remain unchanged.

### Changed files

1. `tunix/rl/self_inf_loo_policy_only_trainer.py`
2. `develop.md`

### Root cause and fix

- The first implementation explicitly constructed the full per-sample weighted-KL gradient pytree as `total_per_sample_grads - policy_per_sample_grads`. That introduced another model-sized, batch-leading intermediate at the memory-critical compiled step.
- Replaced it with the algebraically identical aggregate expression `selected_policy_mean + all_total_mean - all_policy_mean`. This computes the all-sample weighted-KL contribution without materializing a third per-sample gradient pytree.
- Applied the same algebraic rewrite to the reported loss: `selected_policy_loss + mean(total_loss) - mean(policy_loss)`.
- The update remains exactly `mean(selected policy gradients) + mean(all beta-weighted KL gradients)` with independent selected/all-sample denominators.

### Validation commands and results

- Local syntax/compile checks and the Policy-Only unit test must be rerun after this rewrite.
- TPU one-step Group smoke is required to confirm the peak drops below the 28.70 GiB device limit.

### Known risks / TODO

- XLA may still choose a device-specific buffer schedule with a high peak. If the optimized expression remains above capacity, inspect other TPU processes and compiler memory analysis before changing experimental batch parameters.

---

## 2026-07-22 - Diagnose TPU log permission failure on second server

### Scope

- Diagnosed a `/tmp/tpu_logs` permission failure before the Group Policy-Only smoke/run on the second TPU server.
- No repository code, method, launcher, or test was changed（无代码改动）.

### Changed files

1. `develop.md`

### Findings / recommended repair

- `chmod -r 777` is syntactically wrong for recursive chmod (`-R` is recursive), but recursively making log files mode 777 is unnecessary and unsafe.
- Recommended inspecting ownership, setting the shared temporary log directory itself to mode 1777, and changing ownership/permissions only for the stale current-user TPU log that blocks opening.
- Verify no existing Python/TPU task is active before retrying Group Policy-Only.

### Known risks / TODO

- If the directory is managed by a TPU service and permissions are recreated on reboot/process start, identify the service user/umask rather than repeatedly applying broad permissions.
- Run unit tests and a one-step Group Policy-Only smoke after the permission repair before launching the full experiment.

---

## 2026-07-22 - Transfer ownership of stale TPU logs on second server

### Scope

- Interpreted the follow-up `/tmp/tpu_logs` state: directory mode 1777 is correct, the originally reported timestamped log no longer exists, and remaining logs/directories belong to a previous server user.
- No repository code, method, launcher, or test was changed（无代码改动）.

### Changed files

1. `develop.md`

### Recommended operation

- Before ownership transfer, check that the previous user has no active Python/TPU process.
- If the host has been transferred exclusively to the current user, recursively change ownership of `/tmp/tpu_logs` to the current user, normalize files to user-readable/writable mode, and restore the top-level temporary directory to mode 1777.
- Do not delete the old logs; ownership transfer is sufficient and preserves diagnostic history.

### Known risks / TODO

- Recursively taking ownership affects another user's historical files. Perform it only after confirming the machine/log directory is no longer actively shared.
- Retry JAX TPU initialization after ownership transfer; new timestamped log names should be discovered dynamically rather than copied from an earlier error.

---

## 2026-07-23 - Reduce Group Policy-Only compile-time HBM peak

### Scope

- Diagnosed Group Policy-Only XLA compilation failure requiring 29.79 GiB with only 28.70 GiB reservable, while Batch Policy-Only completed on the same method family.
- Optimized only the new Policy-Only aggregation; no baseline, L2, original DTV, total-loss LOO, or Policy-Full path was changed.

### Changed files

1. `tunix/rl/self_inf_loo_policy_only_trainer.py`
2. `tests/rl/self_inf_loo_policy_only_trainer_test.py`
3. `develop.md`

### Root cause and implementation

- The original Policy-Only aggregation explicitly materialized a third full per-sample gradient tree, `weighted_kl_grads = total_grads - policy_grads`, while Group LOO also kept group statistics/cap intermediates live. The Group XLA buffer schedule exceeded available HBM by about 1.09 GiB; Batch had a more favorable fused/liveness schedule and fit.
- Replaced the materializing form `selected_mean(policy) + mean(total - policy)` with the algebraically equivalent reduction-first form `mean(total) + selected_mean(policy) - mean(policy)`.
- Applied the same algebraic rewrite to the displayed loss. The selection scores, masks, cap, default parameters, retained all-sample KL semantics, and optimizer update mathematics are unchanged apart from ordinary floating-point reduction ordering.

### Validation

- Static Python compilation, shell syntax, and diff checks are required locally; TPU Group smoke remains the decisive memory validation.
- Existing numerical unit expectation remains 27.0 update gradient and 54.0 displayed loss for the synthetic policy/KL case.

### Known risks / TODO

- XLA buffer assignment is compiler/version/device dependent, so the algebraic rewrite is expected to remove the extra full tree but cannot guarantee the Group graph fits until rerun on the 28.70 GiB TPU worker.
- If Group still misses HBM, collect compiler memory diagnostics before changing micro-batch size because reducing it would change the experiment's Batch/Group comparison parameters.

---

## 2026-07-23 - Record Batch Policy-Only seed-0 result and pause method changes

### Scope

- Recorded the completed Batch DTV-LOO Policy-Only seed-0 result and consolidated all currently completed seed-0 GSM8K results for analysis while Group Policy-Only remains running.
- No code, method, launcher, or test was changed（无代码改动）.

### Changed files

1. `develop.md`

### Result

- Batch Policy-Only: 679/1319, 51.4784% exact, 52.9947% partial, and 64.0637% format; improvement over the common pre-train result is 4.2456 percentage points and 56 correct answers.
- Relative to Batch Policy-Full, Policy-Only loses 29 correct answers and 2.1986 points exact accuracy; format accuracy falls by 7.4299 points.
- Relative to original Batch DTV it loses 32 answers and 2.4261 points; relative to L2 it loses 54 answers and 4.0940 points.

### Interpretation

- Policy-Only leaves the policy score/mask unchanged and restores only all-sample KL gradients. The degradation is evidence that, in this seed/run, retaining KL from policy-negative completions harms the final trajectory or removes a beneficial implicit KL curation effect of Full Mask.
- This does not prove Policy-Only is universally worse from one seed, but the effect is too large to describe as equivalent to Policy-Full.
- Do not change code or add another method until Group Policy-Only completes and selection/update diagnostics are compared.

### Known risks / TODO

- Verify the completed run's pre-train metrics, exit code, method identity, and `mask_application=policy_only` before final tabulation.
- Group Policy-Only is pending and must be reported separately rather than inferred from the Batch result.

---

## 2026-07-23 - Select the highest-priority next method under compute limits

### Scope

- Analyzed counterfactual policy-gradient ordinary-DTV scores reconstructed from completed Policy-LOO logs under a 12-hour-per-method compute constraint.
- Selected the single highest-priority next experiment without changing code, methods, launchers, or tests（无代码改动）.

### Changed files

1. `develop.md`

### Counterfactual findings

- Batch Policy-LOO has 2,346 negative samples (21.22%); restoring the ordinary DTV self term leaves only 277 Policy-DTV negatives (2.51%) and rescues 2,069 samples.
- Group Policy-LOO has 2,370 negatives (21.44%); restoring self leaves 180 Policy-DTV negatives (1.63%) and rescues 2,190 samples.
- Policy-DTV negatives are strict subsets of Policy-LOO negatives, as required mathematically; there are zero DTV-only negatives.
- Rescued Batch samples have median raw self 11.64 versus median raw cross -1.67; rescued Group samples have median raw self 13.77 versus raw cross -1.59. Most strict-LOO conflicts are therefore mild relative to their self contribution.
- Policy-DTV filter rates are close in scale to counterfactual policy-L2 outlier rates (Batch 2.29%, Group 2.20%) while retaining a direction-based criterion rather than a magnitude criterion.

### Decision

- The single highest-priority new experiment is Batch Policy-DTV with Full Mask: policy gradients for ordinary DTV score including self term, and the selected mask applied to the full policy+KL sample gradient.
- Do not prioritize Group: Batch wins in original DTV and Policy-LOO Full, and Group Policy-DTV would filter only 1.63%, reducing the likelihood of a material gain.
- Do not prioritize Policy-Only: the completed Batch result loses 2.1986 points versus Policy-Full.
- Do not alter PPO solely to mirror GRPO. PPO Policy-Only preserves essential value learning; GRPO Policy-Only preserves KL regularization and has empirically degraded. The unified paper principle can be policy-gradient scoring with algorithm-appropriate treatment of auxiliary objectives.

### Known risks / TODO

- Counterfactual masks do not predict the post-first-step training trajectory; Batch Policy-DTV Full is the best evidence-based bet, not a guaranteed L2 win.
- Before spending 12 hours, run a one-step smoke and verify the Policy-DTV filter count/identity. If implemented, keep it isolated and preserve every existing method.
- Do not rerun the deleted PPO seed or change PPO masking until the GRPO candidate result is known and the paper's cross-algorithm ablation plan is fixed.

---

## 2026-07-23 - Define comprehensive seed-0 filtering audit

### Scope

- Clarified the Policy-LOO versus Policy-DTV self-term rescue interpretation and designed a no-code server-side audit for all completed seed-0 filtering methods.
- No project code, method, launcher, or test was changed（无代码改动）.

### Changed files

1. `develop.md`

### Clarification

- Batch policy self term rescues 2,069 of 2,346 strict-LOO negatives. The rescued samples do not have small self terms: median raw self is 11.64 versus median raw cross -1.67, so the negative interaction is mild relative to self contribution and the ordinary DTV total score remains positive.
- Group similarly rescues 2,190 of 2,370 negatives, with median raw self 13.77 versus raw cross -1.59.

### Audit design

- Historical original-DTV and L2 runs record aggregate `skipped_samples`/score or norm summaries in TensorBoard but do not contain per-sample identities, self/cross values, or masks.
- LOO, Policy-LOO, and Policy-Only runs contain selection JSONL with per-sample self/cross/standard/LOO scores and masks.
- The audit therefore reports actual aggregate filtering for DTV/L2 and exact actual filtering for LOO-family JSONL runs, plus explicitly labeled counterfactual DTV and L2 masks reconstructed on the same LOO-run gradients.
- Exact overlap/disagreement categories are computed only among masks reconstructed from the same JSONL gradient population; separately trained trajectories are not aligned after their first different update.

### Known risks / TODO

- Log-directory discovery must be reviewed in the generated output because both older unseeded-name and newer seed-0-name conventions exist.
- TensorBoard flush/aggregation can make actual aggregate event counts differ from 691; the report must print event counts and avoid claiming per-sample identity from scalar events.

---

## 2026-07-23 - Analyze comprehensive seed-0 filtering audit

### Scope

- Analyzed the generated audit output across L2, original total-loss DTV, strict total-loss LOO, Policy-LOO Full, and Policy-LOO Policy-Only.
- Compared actual filtering intensity, counterfactual same-trajectory mask overlap, self/cross-term distributions, GRPO loss decomposition, and seed-0 accuracy.
- No training code, method, launcher, or configuration was changed（无代码改动）.

### Main findings

- Batch Policy-LOO Full filters 21.22% versus L2's historical 2.32% (about 9.13 times as many), yet trails L2 by only 1.90 percentage points and 25 correct GSM8K examples.
- Filter rate is not monotonic with accuracy: Batch Policy-Only filters less than Batch Policy-LOO Full but is 2.20 points worse, so the treatment of the KL component and the identity of selected samples dominate raw filtering intensity.
- On the Batch Policy-LOO trajectory, counterfactual Policy-DTV and policy-gradient L2 each filter about 2% but have zero sample overlap. Policy-DTV targets moderate-norm samples with strongly negative cross terms; L2 targets very large-norm samples whose directional score is usually positive.
- Strict LOO removes many mildly conflicting samples whose positive self term would protect them under ordinary DTV. The 2,069 batch self-rescued samples have median raw self 11.64 and raw cross -1.67.

### Recommendation

- Highest-priority single new experiment, if authorized later: Batch Policy-DTV score with Full mask. It preserves DTV's original self-plus-cross criterion, uses the reward-driving policy gradient for scoring, and follows the empirically stronger Full-mask behavior in current GRPO seed-0 results.
- Do not infer that PPO must switch from Policy mask to Full mask; PPO value loss and GRPO KL regularization serve different roles.

### Known risks / TODO

- L2/original-DTV historical sample identities were not logged, so their exact mask intersections are counterfactual reconstructions on LOO-family trajectories rather than intersections of independently trained historical runs.
- All accuracy comparisons are currently seed 0 and require matched-seed replication before statistical claims.

---

## 2026-07-23 - Clarify DTV policy naming and minimum-retention semantics

### Scope

- Clarified the proposed `dtv_policy` definition: retain the ordinary DTV self term, compute selection scores from policy loss gradients, and apply the resulting mask to the full/total GRPO update.
- Corrected the description of `min_keep_fraction=0.75`: every batch, or every prompt group under group scope, retains at least 75% and therefore filters at most 25%.
- No training code, method, launcher, or configuration was changed（无代码改动）.

### Code verification

- `_capped_mask` computes `ceil(population_size * min_keep_fraction)` and retains the highest scores.
- Scores are sorted descending, so when the cap activates, nonnegative samples remain selected and negative samples closest to zero are restored first; for example `-2` is restored before `-20`.
- `_group_capped_mask` applies the same cap independently to each prompt group.
- `run_seeded_full.sh` sets `TUNIX_DBC_SELF_INF_MIN_KEEP_FRACTION=0.75` for methods ending in `_keep75`; the runtime default remains 0.25.

### Validation commands and results

- Attempted `python -m pytest tests/rl/self_inf_loo_trainer_test.py -q`; local shell has no `python` command.
- Attempted `python3 -m pytest tests/rl/self_inf_loo_trainer_test.py -q`; the local Python installation has no `pytest`.
- Static inspection confirms the intended ordering, and the existing test `test_cap_keeps_highest_quarter_without_sample_loop` explicitly expects `-1` to be retained from `[-4, -3, -2, -1]`.

### Known risks / TODO

- `dtv_policy` does not yet exist; implementing it requires explicit authorization and must be additive so all existing methods remain unchanged.
- The currently running `dtv_loo_policy_keep75` experiment should finish before deciding whether `dtv_policy` needs a non-default retention cap.

---

## 2026-07-23 - Add ordinary DTV policy-score full-update methods

### Scope

- Added independent Batch and Group `dtv_policy` methods.
- `dtv_policy` retains the ordinary DTV self term, computes selection scores
  from KL-free policy-loss gradients, and applies the selected mask to the
  full/total GRPO gradients and loss.
- Existing baseline, L2, original total-loss DTV, total-loss LOO,
  Policy-LOO Full, and Policy-LOO Policy-Only launch paths remain selected by
  their existing environment variables and commands.

### Changed files

1. `tunix/rl/self_inf_trainer.py`
2. `tunix/rl/self_inf_policy_trainer.py`
3. `tunix/rl/rl_cluster.py`
4. `my_example/run_dbc_self_inf_policy.sh`
5. `my_example/run_dbc_self_inf_batch_policy.sh`
6. `my_example/run_dbc_self_inf_group_policy.sh`
7. `my_example/run_seeded_full.sh`
8. `tests/rl/self_inf_policy_trainer_test.py`
9. `develop.md`

### Implementation details

- Added an optional score-only loss hook to `SelfInfTrainer`. When unset, the
  original DTV code uses total gradients for both scoring and updating exactly
  as before.
- Added `PolicySelfInfTrainer`, which exposes
  `with_policy_score_loss_fn`; `GRPOLearner` therefore supplies the existing
  `beta=0` policy loss for scoring while the ordinary total loss remains the
  optimizer objective.
- Added runtime selection through `TUNIX_DBC_SELF_INF_POLICY=1`. Combining it
  with the LOO selector is rejected to prevent ambiguous method selection.
- Added seeded method names `batch_policy` and `group_policy`.
- The new launchers explicitly disable all LOO selectors in their child
  process and do not set a minimum-retention cap; selection is ordinary
  DTV `score >= 0`.

### Validation commands and results

- `python3 -m compileall -q ...`: passed for all changed Python modules and the
  new test.
- `bash -n ...`: passed for the common policy launcher, both scope wrappers,
  and `run_seeded_full.sh`.
- `git diff --check`: passed.
- Full JAX unit tests could not run locally: the host shell has no `python`
  command and `/usr/local/bin/python3` has no `pytest`. Server-side commands
  are required in `.venv_jax081`.

### Test coverage

- The new trainer test verifies that policy gradients determine the mask while
  total gradients determine the optimizer update.
- Its fixture includes a sample with a negative LOO cross term but a positive
  ordinary DTV score after adding the self term, guarding against accidental
  regression to strict LOO.
- Setter separation and missing-policy-loss validation are covered.

### Known risks / TODO

- TPU/JIT memory behavior must be checked with a one-step smoke test before a
  691-step run. The method computes policy and total per-sample gradients, so
  its peak profile should be comparable to Policy-LOO Full.
- Run Batch seed 0 first; only start Group after the Batch smoke/full outcome
  justifies the additional compute.

---

## 2026-07-23 - Interpret Batch DTV-Policy seed-0 result

### Scope

- Analyzed the completed Batch `dtv_policy` seed-0 GSM8K result.
- Compared it with L2, original DTV, and Policy-LOO results and assessed the
  proposed six-method, five-seed GRPO experiment design.
- No training code, launcher, configuration, or method was changed（无代码改动）.

### Result

- Batch `dtv_policy` reached 733/1319 exact answers (55.5724%), exactly tying
  the seed-0 L2 exact-accuracy result.
- Relative to L2, Batch `dtv_policy` had 0.3033 percentage points lower partial
  accuracy but 7.7331 points higher format accuracy, indicating meaningfully
  different learned behavior despite equal exact accuracy.

### Interpretation

- GRPO group-relative advantages already encode competition among completions
  from the same prompt. Strict LOO removes the positive self contribution and
  treats every negative cross interaction as harmful, including weak conflicts
  that can represent useful minority/difficult-answer signal.
- Ordinary Policy-DTV requires the negative cross term to exceed the positive
  policy self term before filtering. In GRPO this self term acts as an adaptive
  confidence/margin rather than merely undesirable self-protection.
- Policy-LOO consequently filters much more aggressively than Policy-DTV and
  can reduce completion diversity, effective data coverage, and format/partial
  learning, especially with four-generation prompt groups.

### Experiment-design recommendation

- The six methods Baseline, L2, Batch/Group DTV-Policy, and Batch/Group
  Policy-LOO Full form a coherent main comparison, provided all use matched
  seeds and identical hyperparameters.
- Report mean, standard deviation, per-seed paired differences, exact correct
  counts, partial accuracy, format accuracy, and filtering intensity.
- Five matched seeds are preferable for the main table; avoid claiming a tie
  or superiority from seed 0 alone.

### Known risks / TODO

- Verify the Batch `dtv_policy` filtering rate and score distribution from its
  TensorBoard log before finalizing the mechanistic explanation.
- Group `dtv_policy` seed 0 remains necessary before committing all Group runs.
- Policy-LOO keep75 is a useful diagnostic ablation but should not replace the
  default Policy-LOO method in the main six-method table unless chosen before
  examining all seed outcomes.

---

## 2026-07-23 - Finalize six-method GRPO main experiment matrix

### Scope

- Analyzed the completed Group `dtv_policy` seed-0 result and finalized the
  proposed six-method matched-seed GRPO main experiment.
- Prepared a staged seed-5/seed-21 execution plan for Batch and Group
  `dtv_policy` across two single-task TPU servers.
- No training code, launcher, configuration, or method was changed（无代码改动）.

### Seed-0 result

- Group `dtv_policy` reached 731/1319 exact answers (55.4208%), two answers and
  0.1516 percentage points below L2.
- It reached 58.3776% partial accuracy, equal to the recorded L2 partial
  accuracy to displayed precision, and 80.2881% format accuracy, 10.8415
  points above L2.
- Batch `dtv_policy` remained at 733/1319 (55.5724%), exactly tying L2 exact
  accuracy while exceeding its format accuracy by 7.7331 points.

### Final main methods

1. Baseline GRPO
2. L2 outlier
3. Batch DTV-Policy
4. Group DTV-Policy
5. Batch DTV-LOO-Policy Full
6. Group DTV-LOO-Policy Full

### Execution plan

- Run matched experiment seeds 5 and 21 next.
- Server A runs one single-TPU queue in paired-seed order:
  Batch seed 5, Group seed 5, Batch seed 21, then Group seed 21.
- Each next run starts only after the preceding run exits successfully. A
  failure or missing per-run exit-code file stops the queue.
- Existing per-run `runs/` and `logs/` directories remain authoritative; the
  queue adds a master log and status files under its own `logs/` directory.
- The second server remains available for the other four retained methods once
  its current work finishes.

### Known risks / TODO

- Seed 0 alone cannot establish a statistical tie with L2.
- Verify every run records the requested experiment seed, dataset seed
  (`42 + experiment seed`), rollout seed, full 1319-example evaluation, and
  exit code 0.
- After seeds 0, 5, and 21, calculate paired per-seed differences before
  committing compute to seeds 42 and 84 for all six methods.

---

## 2026-07-23 - Define unified seed-0 filtering audit without code changes

### Scope

- Clarified the Policy-LOO versus Policy-DTV self-rescue interpretation and audited which historical methods contain sample-level filtering records.
- Designed a standalone server-side analysis workflow for L2, original DTV, total LOO, Policy-LOO, and Policy-Only seed-0 runs without modifying repository code（无代码改动）.

### Changed files

1. `develop.md`

### Clarification

- Batch Policy-LOO marks 2,346 samples negative; adding the nonnegative self term leaves 277 Policy-DTV negatives, so exactly 2,069 are self-rescued.
- The rescued samples do not have small self terms. Their median raw self is 11.64 versus median raw cross -1.67, meaning self is typically about seven times the opposing cross magnitude; the directional conflict is mild relative to self contribution.

### Logging audit

- Historical `RobustTrainer` (L2) and `SelfInfTrainer` (original DTV Batch/Group) record aggregate TensorBoard metrics only; they do not persist per-sample norms, self/cross terms, identities, or masks.
- LOO-family runs persist selection JSONL with per-sample raw/standard self and cross terms, ordinary-DTV score, strict-LOO score, final/cap masks, and group/generation indices.
- Therefore actual historical L2/DTV filtering intensity can be recovered from TensorBoard, but exact historical sample intersections cannot. Exact intersections can only be reconstructed counterfactually on each LOO run's identical gradients and must be labeled by score objective (total or policy).

### Known risks / TODO

- Logs may reside on different servers; the standalone script accepts explicit `METHOD=LOG_ROOT` arguments and skips missing paths.
- TensorBoard scalar sums assume one `skipped_samples` event per optimizer step; the script reports event counts so incomplete logging is visible.

---

## 2026-07-22 - Assess policy-score plus policy-only masking for GRPO

### Scope

- Analyzed the distinction between the loss component used to compute DTV scores and the loss components affected by the resulting mask.
- Compared the prior PPO policy-mask rationale with the loss decomposition in the current GRPO implementation.
- No training code, method, launcher, or configuration was changed（无代码改动）.

### Changed files

1. `develop.md`

### Verification commands and results

- Inspected `tunix/rl/grpo/grpo_learner.py`: the actor loss is the clipped GRPO policy surrogate plus optional `beta * KL`; setting `beta=0` creates the current policy-only score loss.
- Confirmed GRPO has no critic/value loss in this experiment. The repository explicitly describes the critic role as PPO-style only, not GRPO.
- Inspected `tunix/rl/self_inf_loo_policy_trainer.py` and `tunix/rl/self_inf_loo_trainer.py`: the current Policy-LOO variant computes the mask from KL-free policy gradients but applies that mask to the full per-sample actor gradient, including policy and KL contributions.
- Inspected Tunix PPO: actor policy and critic value losses use separate trainers. Conceptually, policy-only masking preserves critic/value learning; in GRPO the closest analogous preserved component would be KL regularization, not a value loss.

### Interpretation

- Policy-score plus policy-only masking does not discard DTV's core directional criterion; it is a component-selective DTV variant whose declared target is acceleration of the policy-improvement component rather than the entire composite objective.
- For PPO, preserving value learning has a strong independent rationale because critic targets remain informative even when an action-policy gradient conflicts. For GRPO there is no value branch, so policy-only masking means retaining only the filtered completion's KL gradient.
- KL is a reference-policy constraint, not the GRPO analogue of value learning. Therefore policy-only masking is technically applicable but should be tested as a separate ablation rather than assumed to transfer from PPO.

### Known risks / TODO

- Retaining KL for a policy-filtered completion means the sample still changes actor parameters; this no longer removes its full contribution to the realized total update.
- Compare at least policy-score/full-mask and policy-score/policy-mask under identical seeds. Log policy, KL, and total gradient contributions separately so the interpretation remains identifiable.
## 2026-07-21 - Summarize current GSM8K GRPO experiment

### Scope

- Read `GSM8K_GRPO_Reproduction_Guide.md` and inspected the current GSM8K launchers, configuration, dataset loader, and reward definitions.
- Summarized the experiment design using the requested paper-style structure.
- No experiment code or launcher changes（无代码改动）.

### Changed files

1. `develop.md`

### Validation commands and results

- Confirmed the reproduction guide defines four core runs: GRPO baseline, Batch Self-Influence, Group Self-Influence, and L2 outlier curation.
- Confirmed current defaults in `run_grpo_gemma.sh`: Gemma 3 1B IT, GSM8K via TFDS, LoRA GRPO, 4 prompts per training micro-batch, 4 generations per prompt, 3,072 training examples, one epoch, learning rate `1e-6`, and TPU mesh `4,1`.
- Confirmed `TUNIX_REWARD_MODE=accuracy` selects the accuracy-focused reward combination.
- Confirmed additional total-loss LOO and policy-score LOO launchers exist as extension experiments beyond the four core guide runs.

### Known risks / TODO

- The guide identifies commit `a448e1f72cd7eafd6e490d66ec1066b10c5a5906` as the reference; exact reproduction should verify the active branch and commit on the TPU worker.
- Full runtime results were not produced in this documentation-only task; TPU environment, dataset/model access, checkpoint restore, and end-to-end metrics still require execution checks.

---
## 2026-07-22 - Explain GSM8K GRPO table metrics

### Scope

- Inspected `my_example/eval.py`, `my_example/main.py`, and the supplied GRPO result table to document the exact meanings of Pre Acc, Post Acc, Delta, Correct, Partial, and Format.
- No experiment code or launcher changes（无代码改动）.

### Changed files

1. `develop.md`

### Validation commands and results

- Confirmed exact-match accuracy is computed after extracting the number following the solution marker and comparing it numerically with the GSM8K answer.
- Confirmed Partial counts answers whose extracted number is between 90% and 110% of the reference value; it includes exact answers and is not a disjoint error category.
- Confirmed Format counts responses matching the required reasoning/solution-tag regular expression, independently of numerical correctness.
- Confirmed the table's `638/1319` equals `48.3700%`, so Correct is the numerator/denominator form of Post Acc; Delta is Post Acc minus Pre Acc in percentage points.

### Known risks / TODO

- For `num_passes > 1`, each question is counted as successful when at least one generated response satisfies a metric; the displayed runs normally use one evaluation pass.
- Partial uses a ratio check rather than absolute or symmetric relative error, so interpretation around zero or negative reference answers requires care, although GSM8K answers are ordinarily non-negative.

---
## 2026-07-22 - Document Mac-to-TPU VS Code SSH setup

### Scope

- Prepared a command-by-command workflow to discover a new Google Cloud TPU VM, authenticate a Mac with `gcloud`, initialize SSH access, and configure VS Code Remote-SSH.
- Included separate discovery guidance for Cloud TPU API nodes and newer Compute Engine-managed TPU VMs, plus public-IP and private-IP caveats.
- No experiment code or launcher changes（无代码改动）.

### Changed files

1. `develop.md`

### Validation commands and results

- Cross-checked the current official `gcloud compute tpus tpu-vm list`, `describe`, and `ssh` command interfaces.
- Confirmed Cloud TPU VM SSH supports worker selection and `--dry-run`, which can expose the effective SSH username, host, key, and options needed by VS Code Remote-SSH.
- No connection command was executed because the project ID, zone, TPU name, account, and network reachability belong to the new external server and were not supplied in this task.

### Known risks / TODO

- Direct VS Code Remote-SSH requires a reachable SSH endpoint; a TPU without an external IP needs an organization-approved bastion/VPN/private-network route or, for Compute Engine-managed instances, an IAP-compatible setup.
- An ephemeral external IP can change after resource recreation; update the matching `HostName` in `~/.ssh/config` if that occurs.
- OS Login, IAM roles, organization policies, and custom firewall rules can prevent SSH even when the local configuration is correct.

---
## 2026-07-22 - Diagnose missing TPU zone in gcloud list

### Scope

- Diagnosed the Mac `gcloud compute tpus tpu-vm list` failure after selecting project `tunix-testing`.
- Provided read-only Cloud Asset Inventory and Compute Engine discovery commands to locate the TPU name and zone before retrying the zone-scoped TPU command.
- No experiment code or launcher changes（无代码改动）.

### Changed files

1. `develop.md`

### Validation commands and results

- Confirmed from the command output that `core/project=tunix-testing` is set successfully and the failure is specifically an unset TPU location/zone.
- Confirmed the Cloud TPU list operation is zone-scoped and officially requires `--zone` when `compute/zone` has not been configured.
- No external server command was executed from this workspace; the user must run the discovery commands on the authenticated Mac terminal.

### Known risks / TODO

- Cloud Asset Inventory search requires `cloudasset.assets.searchAllResources`; if unavailable, use the Google Cloud TPU console or ask the project administrator for the TPU zone.
- The resource might be a Compute Engine-managed TPU rather than a legacy Cloud TPU API node, so both inventories must be checked.

---
## 2026-07-22 - Refine existing SSH-config workflow for new TPU

### Scope

- Refined the Mac-to-TPU connection instructions for a Mac that already connects to two servers through `~/.ssh/config` and does not need Homebrew installation.
- Documented how to reuse an existing host stanza, map TPU inventory fields to connection fields, obtain the missing SSH endpoint, validate with terminal SSH, and select the alias in VS Code Remote-SSH.
- No experiment code or launcher changes（无代码改动）.

### Changed files

1. `develop.md`

### Validation commands and results

- Confirmed `DISPLAY_NAME` can be used as the TPU resource name and `LOCATION` as its zone, but `STATE` and `ASSET_TYPE` do not provide the SSH endpoint.
- Confirmed a working SSH stanza still needs a local alias plus the effective host/IP, user, and authentication route/key; these can be copied from an existing stanza only when the new TPU uses the same account and access path.
- No remote connection was attempted because the new TPU values and current SSH stanza were not supplied.

### Known risks / TODO

- Do not replace the whole SSH config; append a new uniquely named `Host` stanza and preserve both existing server entries.
- The new TPU might use a different username, key, project, worker, external IP, bastion, or proxy command even when it belongs to the same organization.
- Verify `STATE=READY` and obtain the effective endpoint before testing VS Code.

---
## 2026-07-22 - Clarify reuse of existing Google TPU SSH config

### Scope

- Explained how to add a third TPU entry to an existing Mac `~/.ssh/config` while reusing the established `google_compute_engine` identity.
- Distinguished resource-inventory columns (`DISPLAY_NAME`, `LOCATION`, `STATE`, `ASSET_TYPE`) from the actual queued-resource/node/worker SSH target.
- Documented the purpose and necessity of the additional SSH options present in one older generated host block.
- No experiment code or launcher changes（无代码改动）.

### Changed files

1. `develop.md`

### Validation commands and results

- Confirmed current queued-resource SSH supports `--dry-run`, node/worker selection, the default `~/.ssh/google_compute_engine` key, and optional IAP tunneling.
- Confirmed that reusing one Google Compute Engine SSH identity across multiple authorized TPU hosts is expected; each host still needs its own alias and resolved hostname/IP.
- No local SSH configuration was changed because `~/.ssh/config` is outside the repository and the exact new resource values were not supplied.

### Known risks / TODO

- `StrictHostKeyChecking no` together with an empty or `/dev/null` known-hosts file disables meaningful server identity verification; retain it only when it is an intentional cluster policy or exact `gcloud --dry-run` output.
- Queued resources may contain multiple nodes/workers, so VS Code must target one concrete node/worker, normally node 0 and worker 0 for a single-host experiment.

---
## 2026-07-22 - Explain editing macOS SSH config

### Scope

- Provided safe commands to create, back up, edit, permission-check, and validate `~/.ssh/config` on macOS.
- No experiment code, launcher, or local SSH configuration changes（无代码改动）.

### Changed files

1. `develop.md`

### Validation commands and results

- No commands were executed against the user's home SSH configuration.
- Documented `ssh -G` as a non-connecting syntax/effective-configuration check before attempting login.

### Known risks / TODO

- Preserve existing Host blocks and use the exact hostname, user, identity file, and optional host-key alias returned for the new TPU.

---
## 2026-07-22 - Explain editing an existing Mac SSH config

### Scope

- Provided safe commands to back up, open, edit, validate, and test an existing `~/.ssh/config` when adding a new TPU host.
- No SSH configuration, experiment code, or launcher changes were made（无代码改动）.

### Changed files

1. `develop.md`

### Validation commands and results

- Documented `code ~/.ssh/config` as the VS Code editing path and `nano ~/.ssh/config` as a terminal fallback.
- Documented `ssh -G` for configuration expansion and ordinary `ssh` for connection verification.

### Known risks / TODO

- The new host's exact hostname/IP and optional host-key alias must come from its own `gcloud ... ssh --dry-run` output, not from another server's block.

---
## 2026-07-23 - Diagnose Policy-LOO TPU compile OOM on new server

### Scope

- Inspected the supplied first-step failure log for Batch and Group `dtv_loo_policy_only` runs.
- Traced the failure to the shared Policy-LOO two-gradient compilation path; no code or launcher changes（无代码改动）.

### Changed files

1. `develop.md`

### Validation commands and results

- The fatal exception is `RESOURCE_EXHAUSTED: XLA:TPU compile permanent error`, not the preceding TFLOPs measurement message.
- XLA reports a 39.21 GiB program requirement against 30.75 GiB available HBM, exceeding capacity by 8.46 GiB; 39.20 GiB is HLO temporary storage with only 0.8% fragmentation.
- Largest allocations come from the vmapped JVP/transpose-JVP path for per-token log-probability gradients with shape `bf16[4,4,1,1024,6912]`.
- Confirmed Batch and Group Policy-LOO share the same two vmapped gradient passes; their scope differs only after gradients are produced, so both are expected to have nearly identical compile-time HBM pressure.

### Known risks / TODO

- Compare TPU type/topology, JAX/jaxlib/libtpu/qwix versions, repository commit, mesh, and effective CLI arguments between a successful old server and the new server before attributing the difference to hardware.
- A reduced sequence length or micro-batch can establish an HBM diagnosis, but results from such a run are not directly comparable to the existing full-configuration experiment.
- Long-term same-configuration support may require rematerialization, sequentialized score/update gradient computation, or different sharding; these are code changes and were not attempted.

---
## 2026-07-23 - Refine Policy-LOO OOM diagnosis after smoke success

### Scope

- Incorporated the report that the new TPU server passes smoke testing but fails during the real Policy-LOO run.
- Clarified that smoke success establishes general server health only if it does not compile the exact same Policy-LOO per-step shape and objective.
- No code or launcher changes（无代码改动）.

### Changed files

1. `develop.md`

### Validation commands and results

- The failing real run compiled a Policy-LOO graph with completion activation shape `bf16[4,4,1,1024,6912]` and required 39.21 GiB HBM versus 30.75 GiB available.
- `max_train_examples` and number of optimizer steps do not normally reduce the memory of one identically shaped jitted step; therefore an exact Policy-LOO smoke with the same micro-batch, generations, sequence lengths, mesh, and environment should reproduce the compile result.
- If only baseline or ordinary LOO smoke passed, it does not test the second vmapped policy-score gradient pass responsible for Policy-LOO peak HBM.

### Known risks / TODO

- Retrieve the exact successful smoke command and its first compiled tensor shapes before concluding that smoke and full runs are equivalent.
- Check for argument overrides, environment-variable leakage, different checkout/venv, shorter generation length, smaller micro-batch, or a different Policy/Policy-LOO method between the two runs.

---
## 2026-07-23 - Refine new-server Policy-LOO OOM diagnosis after smoke success

### Scope

- Incorporated the observation that Policy-LOO smoke tests pass on the new TPU while the full run fails, and that the same full experiment succeeds on two older TPU servers.
- No code or launcher changes（无代码改动）.

### Changed files

1. `develop.md`

### Findings

- Smoke success rules out a general TPU/JAX failure but does not establish full-shape HBM capacity unless it compiled the same `[4,4,1,1024,6912]` path with identical flags and environment.
- The full run fails at global step 0 during compilation, so the number of planned steps (691) is not itself consuming memory; the first real rollout batch is triggering a 1024-token worst-case static shape.
- If the smoke used identical nominal CLI arguments, generated completion length or padding/bucketing may still have produced a smaller compiled shape.
- If an old server demonstrably compiles the exact same shape and commit, the leading cause becomes JAX/jaxlib/libtpu/qwix/runtime/XLA configuration drift rather than faulty hardware.

### Validation status

- No server commands were executed locally. A three-server environment and effective-shape comparison remains required.

### Known risks / TODO

- Preserve the failing full-run log and compare its shape, exact invocation, commit, package versions, TPU device kind/count, and XLA-related environment variables with a successful old-server run.
- Do not treat host RAM cleanup or reboot as a primary fix: the report is a permanent compile-time HBM requirement with zero argument memory and negligible fragmentation.

---
## 2026-07-23 - Refine new-server Policy-LOO OOM diagnosis after smoke success

### Scope

- Refined the diagnosis after learning that the new TPU server passes smoke tests and the other two nominally identical servers complete full Policy-LOO training.
- Prepared a three-server fingerprint comparison covering effective command/config, repository state, Python packages, TPU topology, runtime variables, and XLA compilation inputs.
- No experiment code or launcher changes（无代码改动）.

### Changed files

1. `develop.md`

### Validation commands and results

- A Baseline or ordinary-LOO smoke only establishes basic TPU health; it does not exercise Policy-LOO's two-gradient compile graph.
- If a Policy-LOO smoke preserves micro-batch size 4, four generations, and sequence length 1024, changing only total steps should not change the first-step static compile shape. A pass/fail difference therefore indicates a hidden effective-input or environment difference.
- The supplied failing graph is unambiguously `[4,4,1,1024,6912]` and requires 39.21 GiB versus 30.75 GiB available HBM.

### Known risks / TODO

- “Installed from the same instructions” does not prove binary/runtime identity; exact package hashes, TPU runtime image, XLA/libtpu variables, repository diff, and effective launcher arguments still need comparison.
- Compilation-cache reuse can obscure whether an old server is recompiling the same executable; capture fresh logs and exact graph shapes on all servers before drawing a hardware conclusion.

---
## 2026-07-23 - Refine Policy-LOO full-run OOM diagnosis after smoke success

### Scope

- Incorporated the clarification that both Batch and Group Policy-LOO-only smoke tests passed on the new server while full training fails.
- Refined the diagnosis toward smoke/full compile-shape or execution-path differences rather than a universally unsupported method or defective TPU.
- No experiment code or launcher changes（无代码改动）.

### Changed files

1. `develop.md`

### Validation commands and results

- The failing full run compiles a worst-case-looking activation shape `bf16[4,4,1,1024,6912]` and requires 39.21 GiB HBM.
- A passing smoke only proves parity if it actually completed an actor optimizer step with the same method class, 4x4 completion batch, sequence length 1024, mesh, and dependency/runtime build.
- Because JAX specializes compiled programs by input shape, a smoke batch with shorter generated sequences can pass and a later/full batch reaching the 1024-token padded length can trigger a new, larger compilation.

### Known risks / TODO

- Verify the smoke was not training-skipped by an existing checkpoint or zero effective batches and that its log contains a completed Actor Training step plus Policy-LOO selection metrics.
- Capture effective arguments, first compiled sequence shape, TPU device kind/count, runtime build, XLA flags, commit, and dirty diff on all three servers; nominally identical installation instructions do not prove bit-for-bit runtime parity.
- If old full runs truly compiled the identical `[4,4,1,1024,6912]` graph on the same TPU topology and software builds, compare XLA dump/memory reports and environment flags before changing experiment settings.

---
## 2026-07-23 - Identify smoke/full method mismatch

### Scope

- Inspected the supplied successful smoke log and compared its method/launcher with the failing Policy-LOO-only path.
- No experiment code or launcher changes（无代码改动）.

### Changed files

1. `develop.md`

### Validation commands and results

- The smoke genuinely completed one Actor Training optimizer step and model export with exit code 0.
- Its header identifies `Method: group_policy` and `run_dbc_self_inf_group_policy.sh`, selecting `PolicySelfInfTrainer` (ordinary group DTV policy scoring).
- The failing target described by the user is `group_loo_policy_only`/`batch_loo_policy_only`, which selects `PolicyOnlySelfInfLooTrainer` through a different launcher and compiled aggregation graph.
- Therefore the supplied smoke does not establish that either LOO Policy-only method compiles on the new server.

### Known risks / TODO

- Run exact one-step smokes named `batch_loo_policy_only` and `group_loo_policy_only`, then verify the headers and selection JSONL identify LOO Policy-only rather than ordinary Policy-DTV.
- If exact-method smokes pass but full runs fail, compare generated/padded sequence shapes; if they fail with the same 1024-length HBM report, the method mismatch explains the prior apparent contradiction.

---
## 2026-07-23 - Prepare Group Policy-DTV full experiment command

### Scope

- Confirmed the successful smoke maps to `group_policy` and prepared the corresponding seed-0 full-run procedure using the existing seeded launcher.
- No experiment code or launcher changes（无代码改动）.

### Changed files

1. `develop.md`

### Validation commands and results

- Confirmed `run_seeded_full.sh group_policy 0` routes to `my_example/run_dbc_self_inf_group_policy.sh`.
- Confirmed the seeded launcher creates timestamped run/log roots and supplies isolated TensorBoard, checkpoint, merged-model, stdout, and exit-code paths.
- Confirmed no extra smoke-only arguments are needed for the full run; omitting them restores the standard 3,072-example and pre/post evaluation configuration.

### Known risks / TODO

- Verify available disk space and that no other `my_example` Python training process is active before launch.
- Monitor the first Actor step for HBM compilation success and retain the generated `nohup.log`, `exit_code`, checkpoints, model export, TensorBoard, and evaluation metadata.
- Complete seed 0 before dispatching additional seeds so the new server's full-run stability is established.

---
## 2026-07-23 - Confirm Group Policy full-run HBM failure on new TPU

### Scope

- Inspected the full `group_policy` failure after a successful one-step smoke on the same new server.
- Compared the full path with the smoke configuration and current dataset splitting/evaluation order.
- No experiment code or launcher changes（无代码改动）.

### Changed files

1. `develop.md`

### Validation commands and results

- Full pre-train evaluation completed over all 1,319 GSM8K examples before the failure.
- The first Actor step then failed at XLA compile with 39.16 GiB required versus 30.75 GiB available HBM (8.41 GiB over capacity).
- The failing graph again contains `bf16[4,4,1,1024,6912]` vmapped JVP/transpose-JVP activations and only 0.7% fragmentation.
- The earlier smoke completed one real `group_policy` Actor step but skipped pre/post evaluation and used `max_steps=1`; therefore the TPU is functional, while an execution-context or compile-shape/runtime difference remains between smoke and full.
- Current code performs pre-train evaluation with the same model before constructing the RL cluster and compiling the actor training step, making `--skip-eval-before` the smallest no-code isolation test.

### Known risks / TODO

- Compare accelerator type/device kind and HBM across old and new servers; identical source and Python package installation do not imply identical TPU hardware/runtime.
- Re-run full `group_policy` with only `--skip-eval-before`. If it succeeds, pre-evaluation leaves compilation/runtime state that causes the new TPU to cross its HBM limit; if it still fails, compare exact XLA/runtime fingerprints and smoke/full shapes.
- Skipping pre-evaluation does not change the training dataset or optimizer configuration, but the resulting artifact lacks run-local Pre Acc metadata and should use a separately recorded common base-model evaluation if reported.

---
## 2026-07-23 - Explain cross-server divergence under nominally identical setup

### Scope

- Documented why identical source/configuration and similarly installed Python environments can still produce different TPU XLA HBM outcomes across servers.
- No experiment code or launcher changes（无代码改动）.

### Changed files

1. `develop.md`

### Validation commands and results

- The observed failure is below the Python training layer: XLA compiled a 39.16 GiB program for a 30.75 GiB-per-chip target.
- The successful smoke proves the new TPU is operational, while peer-server full success makes a universal source-code failure unlikely.
- Remaining differentiators include accelerator/HBM/topology, TPU runtime and libtpu/XLA build, sharding/compiler flags, actual generated sequence shapes, and pre-evaluation-to-training executable/buffer lifetime.

### Known risks / TODO

- No memory report from a successful peer-server full compile is available, so it is not yet known whether the peer compiled the identical 1024-length graph, used different rematerialization/sharding, or had more per-chip HBM.
- If this server is revisited, capture a side-by-side runtime fingerprint and successful-server XLA shape/memory evidence before changing code or training hyperparameters.

---
## 2026-07-23 - Add server A DTV-Policy seed 5/21 one-shot queue

### Scope

- Recorded the completed Group `dtv_policy` seed-0 result and finalized the
  proposed six-method GRPO main experiment matrix.
- Added an independent one-shot server A queue for Batch/Group DTV-Policy
  seeds 5 and 21.
- No trainer, score, mask, GRPO configuration, or existing launcher changed.

### Seed-0 result and main matrix

- Group `dtv_policy` reached 731/1319 exact answers (55.4208%), 58.3776%
  partial accuracy, and 80.2881% format accuracy.
- The main matrix is Baseline, L2, Batch/Group DTV-Policy, and Batch/Group
  Policy-LOO Full, all with identical hyperparameters and matched seeds.
- Total-score DTV/LOO, keep75, and Policy-Only remain diagnostic ablations.

### Changed files

1. `my_example/run_server_a_dtv_policy_seeds_5_21.sh`
2. `develop.md`

### Queue behavior

- Runs sequentially in this order: `batch_policy/5`, `group_policy/5`,
  `batch_policy/21`, `group_policy/21`.
- Reuses `run_seeded_full.sh` without overriding experiment parameters.
- Refuses to start while another `my_example` Python task is active and stops
  at the first failed run.
- Writes PID, combined log, tab-separated run status, and final exit code under
  `logs/server_a_dtv_policy_seeds_5_21_<timestamp>/`.

### Validation commands and results

- `bash -n my_example/run_server_a_dtv_policy_seeds_5_21.sh`: passed.
- `git diff --check`: passed.
- Confirmed the queue script is executable.

### Known risks / TODO

- The queue does not skip completed runs. After a partial failure, launch only
  the remaining method/seed pairs rather than blindly restarting the queue.
- Four runs can occupy server A for roughly 48 hours and create four complete
  checkpoint/model/log artifact sets; verify disk capacity first.

---
## 2026-07-24 - Clarify preference mismatch noise in DPO versus GRPO

### Scope

- Explained whether online GRPO has a direct analogue of DPO chosen/rejected response mismatch noise.
- Distinguished preference-label corruption from GRPO reward, rollout, grouping, and prompt noise.
- No experiment code or launcher changes（无代码改动）.

### Changed files

1. `develop.md`

### Validation commands and results

- Confirmed the current GSM8K GRPO pipeline samples multiple responses online from each prompt and computes group-relative advantages from their rewards; it does not consume a fixed chosen/rejected preference pair.
- Therefore a DPO-style operation that swaps or mismatches chosen/rejected responses is not natively defined for this experiment.
- Identified the closest GRPO corruption analogues as reward corruption, response-to-prompt/group reassignment, prompt/answer corruption, and rollout-distribution changes; these test different failure modes and should not be labeled preference mismatch without a precise definition.

### Known risks / TODO

- Offline GRPO variants or GRPO implementations trained from stored rollouts can define response/group mismatch corruption, but that remains distinct from DPO preference-label mismatch.
- Any cross-algorithm noise comparison should match the corrupted semantic signal (for example reward correctness) rather than reuse the same mechanical corruption operation.

---
## 2026-07-24 - Add separate seed queues for Baseline/L2 and Batch Policy-LOO

### Scope

- Added one independent queue for Baseline and L2 seed 21.
- Added a second independent queue for Batch Policy-LOO Full seeds 5 and 21.
- Both reuse `run_seeded_full.sh`; no trainer, score, mask, hyperparameter, or
  existing method launcher was modified.

### Changed files

1. `my_example/run_baseline_l2_seed21.sh`
2. `my_example/run_batch_loo_policy_seeds_5_21.sh`
3. `develop.md`

### Queue definitions

- `run_baseline_l2_seed21.sh`: `baseline/21` followed by `l2/21`.
- `run_batch_loo_policy_seeds_5_21.sh`: `batch_loo_policy/5` followed by
  `batch_loo_policy/21`.
- Each queue rejects an already-running `my_example` Python task, executes
  sequentially, stops after the first failure, and records PID, combined log,
  status TSV, and exit code under its timestamped `logs/` directory.

### Validation commands and results

- `bash -n` passed for both new queue scripts.
- `git diff --check` passed.
- Both scripts have executable permissions.
- Static inspection confirmed the exact method/seed arrays and order.

### Known risks / TODO

- Do not launch both scripts concurrently on the same single-task TPU host.
- After a partial failure, run only the unfinished method/seed pair instead of
  restarting a queue and duplicating completed experiments.
- Check disk capacity before each queue because every run exports a full merged
  model plus checkpoints and logs.

---
## 2026-07-24 - Diagnose TPU client initialization hang on second server

### Scope

- Diagnosed a JAX TPU probe hanging in `make_tpu_client()` on the second
  server after it had previously trained successfully.
- No repository code, environment, process, service, or TPU resource was
  changed（无代码改动）.

### Evidence

- JAX imports successfully from `.venv_jax081`; the hang occurs while creating
  the TPU backend client, not while importing the package.
- The narrow process check only excluded Python commands containing
  `my_example`; it did not exclude PPO, JAX, notebook, other-user, or stale TPU
  processes.
- Root disk has 18 GiB free. This is tight for multiple exported runs but does
  not explain a TPU client initialization hang.
- The probe was manually interrupted, so backend health remains unconfirmed.

### Diagnostic order

1. Restore the intended `.venv_jax081` environment and inspect JAX/libtpu
   versions and TPU-related environment variables.
2. Inspect all users' Python/JAX/TPU processes, rather than only
   `python.*my_example`.
3. Run bounded CPU and TPU probes and inspect `/tmp/tpu_logs`.
4. Inspect TPU-related services/runtime only if no process owns the device.
5. Restart a runtime or TPU VM only after confirming no useful task is active.

### Known risks / TODO

- Do not launch a training queue until a bounded TPU probe reports four TPU
  devices.
- Do not kill processes or restart services based only on the narrow grep;
  identify PID, user, elapsed time, and command first.
- 18 GiB free may be insufficient for two complete sequential model/checkpoint
  exports; clean only explicitly verified obsolete artifacts.

---
## 2026-07-25 - Add Group Policy-LOO seeds 5/21 queue

### Scope

- Added an independent sequential queue for Group Policy-LOO Full seeds 5 and
  21, completing the queued seed coverage for both Policy-LOO scopes.
- No trainer, score, mask, retention cap, hyperparameter, or existing launcher
  was modified.

### Changed files

1. `my_example/run_group_loo_policy_seeds_5_21.sh`
2. `develop.md`

### Queue definition

- Runs `group_loo_policy/5` followed by `group_loo_policy/21`.
- Reuses `run_seeded_full.sh`, so the method remains policy-gradient scoring,
  strict LOO without the self term, Full mask, and default minimum keep 25%.
- Refuses to start while another `my_example` Python task is active, stops at
  the first failure, and records PID, combined log, status TSV, and exit code.

### Validation commands and results

- `bash -n my_example/run_group_loo_policy_seeds_5_21.sh`: passed.
- `git diff --check`: passed.
- Confirmed executable permissions and exact method/seed ordering.

### Known risks / TODO

- Do not launch Batch and Group Policy-LOO queues concurrently on the same
  single-task TPU host.
- After a partial failure, launch only the unfinished seed rather than
  restarting and duplicating the completed run.

---
