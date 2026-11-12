# dtv-flip（GSM8K / GRPO）

`dtv-flip` 沿用普通 DTV / DTV-LOO 的完整 sample loss 梯度评分，
将原先置零的负分梯度改为取反。policy 与 KL 均参与评分并一起翻转。
不改 advantage、reward 或答案，也不采用 policy-only 评分。

对于对应 batch/group 内的完整梯度 `g_i`：

- DTV：`s_i = dot(g_i, mean_j(g_j))`。
- DTV-LOO：`s_i = dot(g_i, sum_{j != i}(g_j) / (N - 1))`。
- 仅 `s_i < 0` 的样本有资格翻转；在本批配额内选 score 最低的样本，使用 `w_i=-1`。其他样本（包括未入选负分）使用 `w_i=+1`。同分按原 completion 索引稳定选择。
- 更新梯度为 `sum_i(w_i * g_i) / N`，不除以权重和。
- LOO 最低保留比例不适用于 flip；DTV 和 DTV-LOO 共用累计比例预算。

batch 指现有 trainer 一次 `_train_step` 的 micro-batch；group 指其中同一
prompt 的 `num_generations` 个 sample。与原方法一样，在当前可训练参数上
计算梯度（启用 LoRA 时为 LoRA 参数）。不跨梯度累积步骤重新计算全局分数。
非法 group 布局或 LOO population 小于 2 会报错；非有限 score 会中止该次执行。

## 累计翻转配额（默认模式）

`TUNIX_DTV_FLIP_MAX_FRACTION` 默认 `0.1`，表示累计参与 flip 评分的
completion 的翻转上限。默认每次更新都启用 flip。
评分仍按所选 batch/group 范围；翻转配额统一在当前 `_train_step` 的全部样本中分配。

令 `S_t` 为截至当前批次累计参与 flip 评分的 completion 数，`r` 为配置比例：

```text
本批配额 q_t = floor(r*S_t) - floor(r*S_(t-1))
实际翻转数 = min(q_t, 当前批次负分数)
```

只保留取整产生的小数余量；负分不足导致的未用整数名额作废，不会在后续追补。
因此实际累计翻转数始终不超过 `floor(r*S_t)`，负分充足时长期接近 r，
负分不足时实际比例更低，不为了凑数翻转零分或正分。
每批16个 completion 时，10%配额前五批为 `1、2、1、2、2`，
共80个 completion 最多翻8个。个别批次2/16=12.5%符合累计上限定义。

比例使用精确有理数取整，支持 0 到 1，分母最多 1000000。累计模式中，0
使配额为零，1 给启用批次的每条负分 completion 一个名额。配额按实际
处理的 completion 数推进。
每次训练实际再次处理的completion都计数，不按去重后的唯一文本数计数。

例如显式指定10%（原CLI不变）：

```bash
TUNIX_DTV_FLIP_MAX_FRACTION=0.1 ./my_example/run_dtv_flip.sh dtv batch
TUNIX_DTV_FLIP_MAX_FRACTION=0.1 ./my_example/run_dtv_flip.sh dtv-loo batch
```

## 逐 batch 配额和更新频率

设置 `TUNIX_DTV_FLIP_BATCH_FRACTION=k` 时，当前启用 batch 的配额为 `k × N`
取最近的偶数，等距时向上，最大为当前 batch 中可容纳的偶数。`N` 是这次
trainer 调用的实际 completion 数。比如 `k=0.1` 时，16、32、64、512 条
分别给 2、4、6、52 个名额。配额只用于当前批次，实际翻转数仍受负分候选数
限制，可以是奇数。16 条对应的配额比例为 12.5%。

`TUNIX_DTV_FLIP_FREQUENCY=f` 适用于两种配额模式。`f` 是正整数，默认 1，
启用第 `1, 1+f, 1+2f, …` 次 actor optimizer update。若使用梯度累积，属于
同一次 update 的 micro-batch 共用启用状态。跳过步走普通 batch loss 梯度与
原 optimizer，并暂停累计模式的配额计数和小数余量。

```bash
# 旧累计配额，仅在第 1、3、5… 步启用
TUNIX_DTV_FLIP_MAX_FRACTION=0.1 TUNIX_DTV_FLIP_FREQUENCY=2 \
  ./my_example/run_dtv_flip.sh dtv batch

# 逐 batch 偶数配额，仅在第 1、3、5… 步启用
TUNIX_DTV_FLIP_BATCH_FRACTION=0.1 TUNIX_DTV_FLIP_FREQUENCY=2 \
  ./my_example/run_dtv_flip.sh dtv batch
```

## Random Flip

`random batch` 从当前启用 batch 的所有 completion 均匀无放回抽满固定配额。
默认逐 batch 比例为 `0.1`，同样取最近偶数。当前 16 条 completion 时，每个
启用步翻 2 条。采样由 `TUNIX_EXPERIMENT_SEED` 与训练 micro-step 决定。
选中条目的完整 policy + KL loss 乘 `-1`，再按原 batch 大小平均并反传一次。
跳过步使用普通 GRPO 更新。Random Flip 使用标准的 `sequence-mean-token-mean`
GRPO loss。

```bash
TUNIX_EXPERIMENT_SEED=5 TUNIX_DTV_FLIP_BATCH_FRACTION=0.1 \
TUNIX_DTV_FLIP_FREQUENCY=2 ./my_example/run_dtv_flip.sh random batch
```

决策 JSONL 中 `method` 为 `random-flip`，记录 `random_flip_seed` 与
`flipped_indices`。DTV 评分和负分计数为 `null`。配额、实际翻转数及两种累计
比例沿用 flip 日志字段。Random Flip checkpoint 同时校验采样 seed。

## Checkpoint恢复

周期保存和最终保存都包含 `dtv_flip_budget` metadata，记录配额模式、比例、
频率、梯度累积步数、评分配置、累计计数和取整余量。恢复时根据已完成的
actor step 继续原来的频率相位。原有 v1 metadata 可按累计模式、频率 1 恢复。
变更配额模式、比例、频率或评分配置时使用新的 checkpoint 目录。
现有框架只恢复模型参数，不恢复Optax动量；本次保证配额计数恢复，不宣称
Adam训练轨迹完整等价。测试使用无动量SGD验证保存恢复后的模型和配额一致。

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
与原 self-inf launcher 一致。每次生成包含 score、scope、配额、时间和 PID 的独立 checkpoint
与 metrics 路径，也支持 `--checkpoint-root`、`--metrics-log-dir` 两种参数写法。
逐 batch 模式的路径带 `batchcap` 和 `freq`，累计模式在频率大于 1 时也附加 `freq`。
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
`dtv-flip_<dtv 或 dtv-loo>_<batch 或 group>_cap0p1_...`。
JSONL 记录 `quota_policy`、`configured_fraction`、`frequency`、`flip_enabled`，
以及每批配额和实际翻转数。启用步包含原始 `scores`、实际 `weights` 和负分数。
跳过步的这些评分字段为 `null`，实际翻转数与配额均为零。

训练指标包括 `dtv_flip_count`、`dtv_flip_fraction`、`dtv_flip_raw_loss`、
`dtv_flip_negative_count/fraction`、`dtv_flip_quota`、`dtv_flip_seen`、
`dtv_flip_total_seen`、`dtv_flip_total`、`dtv_flip_enabled` 和两个实测比例。
`dtv_flip_active_fraction` 的分母是启用步处理的 completion 数（`dtv_flip_seen`）。
`dtv_flip_cumulative_fraction` 的分母是全部训练步的 completion 数（`dtv_flip_total_seen`）。
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

本地 JAX 0.8.1 四 CPU 设备隔离模式 30 项测试通过，覆盖两种配额、
频率调度、跳过步与普通 AdamW 的一致性、梯度累积、Orbax 保存恢复，
以及 Random Flip 单次反传与逐 completion 完整梯度取反的一致性。
隔离测试不代表完整依赖导入、TPU 显存、checkpoint restore 或完整 GSM8K 训练已验证。
服务器仍需运行正常模式测试与实际训练；本次未在服务器执行或推送代码。
