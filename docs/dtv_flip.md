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

## 累计翻转配额（默认10%）

`TUNIX_DTV_FLIP_MAX_FRACTION` 默认 `0.1`，表示整个训练累计 completion 的
翻转上限，不是每个 prompt 或每个 batch 的硬上限。
评分仍按所选 batch/group 范围；翻转配额统一在当前 `_train_step` 的全部样本中分配。

令 `S_t` 为截至当前批次累计处理的 completion 数，`r` 为配置比例：

```text
本批配额 q_t = floor(r*S_t) - floor(r*S_(t-1))
实际翻转数 = min(q_t, 当前批次负分数)
```

只保留取整产生的小数余量；负分不足导致的未用整数名额作废，不会在后续追补。
因此实际累计翻转数始终不超过 `floor(r*S_t)`，负分充足时长期接近 r，
负分不足时实际比例更低，不为了凑数翻转零分或正分。
每批16个 completion 时，10%配额前五批为 `1、2、1、2、2`，
共80个 completion 最多翻8个。个别批次2/16=12.5%符合累计上限定义。

比例使用精确有理数取整，支持0到1（分母最多1000000）；0禁用翻转但保留评分，
1恢复“全部负分都翻转”的旧行为。JIT动态计数存于optimizer的独立NNX状态，
按真实处理completion数推进，不依赖optimizer update step或梯度累积次数。
每次训练实际再次处理的completion都计数，不按去重后的唯一文本数计数。

例如显式指定10%（原CLI不变）：

```bash
TUNIX_DTV_FLIP_MAX_FRACTION=0.1 ./my_example/run_dtv_flip.sh dtv batch
TUNIX_DTV_FLIP_MAX_FRACTION=0.1 ./my_example/run_dtv_flip.sh dtv-loo batch
```

## Checkpoint恢复

周期保存和最终保存都包含 `dtv_flip_budget` metadata：累计处理/翻转数、取整余量、
精确比例、评分scope、LOO开关和num_generations。恢复后继续原配额序列。
不包含配额状态的旧checkpoint、不同配额比例或不同评分配置均拒绝继续；
请使用新的checkpoint目录，避免把之前无上限的实验与10%实验混合。
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
与原 self-inf launcher 一致。每次生成包含 score/scope/cap/时间/PID 的独立 checkpoint
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
`dtv-flip_<dtv 或 dtv-loo>_<batch 或 group>_cap0p1_...`。
JSONL 包含原始 `scores`、实际 `weights`、scope、score 类型及
`score_objective=total`、`flip_application=full`、`quota_scope=batch`、
配置比例及本批配额/负分数/实际翻转数/累计处理数/累计翻转数和比例。

训练指标包括 `dtv_flip_count`、`dtv_flip_fraction`、`dtv_flip_raw_loss`、
`dtv_flip_negative_count/fraction`、`dtv_flip_quota`、`dtv_flip_seen`、
`dtv_flip_total` 和 `dtv_flip_cumulative_fraction`。
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

本地上述四 CPU 设备隔离模式 25 项测试全部通过（含 sharded/cached JIT、
累计配额、未用名额作废、恢复配置校验及真实生产CheckpointManager/Orbax保存恢复）。
隔离测试不代表完整依赖导入、TPU 显存、checkpoint restore 或完整 GSM8K 训练已验证。
服务器仍需运行正常模式测试与实际训练；本次未在服务器执行或推送代码。
