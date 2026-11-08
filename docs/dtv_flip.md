# dtv-flip（GSM8K / GRPO）

`dtv-flip` 沿用普通 DTV / DTV-LOO 的完整 sample loss 梯度评分，
将原先置零的负分梯度改为取反。policy 与 KL 均参与评分并一起翻转。
不改 advantage、reward 或答案，也不采用 policy-only 评分。

对于对应 batch/group 内的完整梯度 `g_i`：

- DTV：`s_i = dot(g_i, mean_j(g_j))`。
- DTV-LOO：`s_i = dot(g_i, sum_{j != i}(g_j) / (N - 1))`。
- `s_i < 0` 使用 `w_i = -1`，否则 `w_i = +1`，零分保留。
- 更新梯度为 `sum_i(w_i * g_i) / N`，不除以权重和。
- LOO 最低保留比例不适用于 flip；所有负分都翻转。

batch 指现有 trainer 一次 `_train_step` 的 micro-batch；group 指其中同一
prompt 的 `num_generations` 个 sample。与原方法一样，在当前可训练参数上
计算梯度（启用 LoRA 时为 LoRA 参数）。不跨梯度累积步骤重新计算全局分数。
非法 group 布局或 LOO population 小于 2 会报错；非有限 score 会中止该次执行。

## 启动

在已配置好 GSM8K、模型和依赖的服务器仓库根目录执行，使用对应训练环境的 Python。
以下每条是一个独立实验，请选所需组合，不要同时启动四个任务：

```bash
# 普通 DTV score，batch 范围
./my_example/run_dtv_flip.sh dtv batch --num-test-batches 1

# 普通 DTV score，group 范围
./my_example/run_dtv_flip.sh dtv group --num-test-batches 1

# DTV-LOO score，batch 范围
./my_example/run_dtv_flip.sh dtv-loo batch --num-test-batches 1

# DTV-LOO score，group 范围
./my_example/run_dtv_flip.sh dtv-loo group --num-test-batches 1
```

`--num-test-batches 1` 限制评测数量，不限制训练步数。
其余已有 GRPO 参数可追加。脚本默认使用 `--no-wandb`、train micro-batch size 4，
与原 self-inf launcher 一致。每次生成包含 score/scope/时间/PID 的独立 checkpoint
与 metrics 路径，也支持 `--checkpoint-root`、`--metrics-log-dir` 两种参数写法。
传入旧 checkpoint 路径仍可能恢复已有步数；正式比较应使用独立新路径。

脚本沿用现有 `run_grpo_gemma.sh`，不修改 CLI / RLTrainingConfig。
内部通过 `TUNIX_DBC_SELF_INF_FLIP=1` 选择新 trainer，
`TUNIX_DBC_SELF_INF_LOO` 区分评分，现有 self-inf batch/group flag 确定 scope。
这些变量仅对脚本启动的子进程设置，不要求在 shell 中全局 export。
不要在 `.env` 中设置冲突的方法变量；原 launcher 会加载 `.env`。
开启任何 policy-score/policy-only 环境变量时，新 trainer 明确拒绝运行。
原过滤方法和 baseline 不开启 flip 时仍走原路径。

## 核验结果

stdout、结果导出和 `__decisions.jsonl` 默认保存到 `my_example/my result/`，
可通过 `TUNIX_MY_RESULT_DIR` 更改。文件名前缀为
`dtv-flip_<dtv 或 dtv-loo>_<batch 或 group>_...`。
JSONL 包含原始 `scores`、实际 `weights`、scope、score 类型及
`score_objective=total`、`flip_application=full`。

训练指标新增 `dtv_flip_count`、`dtv_flip_fraction` 和 `dtv_flip_raw_loss`。
`skipped_samples=0`，`self_inf_kept_fraction=1`。
训练 loss 显示 signed loss；原 KL/reward 等 aux 指标仍是未加符号的普通均值，
不能将日志里的 KL 均值误认为 KL 梯度未翻转。

## 验证状态

完整训练环境中运行：

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests/rl -p dtv_flip_trainer_test.py
```

本机缺少 `metrax`，采用显式隔离模式替换 Trainer 基类 import，
使用真实 CPU JAX、Flax autodiff、JIT、LoRA 和 Optax 验证：

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
TUNIX_FLIP_TEST_ISOLATED=1 JAX_PLATFORMS=cpu \
python -m unittest discover -s tests/rl -p dtv_flip_trainer_test.py
```

本地上述四 CPU 设备隔离模式 17 项测试全部通过（含 sharded JIT 更新）。
隔离测试不代表完整依赖导入、TPU 显存、checkpoint restore 或完整 GSM8K 训练已验证。
服务器仍需运行正常模式测试与实际训练；本次未在服务器执行或推送代码。
