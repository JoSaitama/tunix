# GRPO rebuttal：材料准备、核心诊断与新增对照

日期：2026-11-07。对应工作文档：`rebuttal.md`。本文不是已完成实验报告。

## 1. 优先级：先回答有效性，再补方法覆盖

| 项目 | 直接回答的问题 | 优先级与局限 |
|---|---|---|
| 固定checkpoint的独立clean utility | batch/group内部一致性是否真的对应外部clean objective改善？ | 核心优先；回应R2，也回应R1对Assumption和minority signal的疑问；实验支持不是数学证明 |
| Norm top-50% | DTV是否只是挑选较大梯度，而不需要direction？ | R1明确点名的直接对照；只看最终accuracy仍不足以验证reference可靠性 |
| GRPO＋DS与正在运行的PODS | 是否胜过直接相关的数据/rollout选择方法？ | 回应missing baselines；即便胜出也不能直接证明核心Assumption |
| PCGrad / CAGrad | 为什么过滤样本，而不是修改冲突更新方向？ | 先准确related-work/算法对照；PCGrad适配优先于CAGrad完整训练矩阵 |
| 清楚的协议、预算、模型与任务范围 | 是否可复现、公平，是否过度声称general RL收益？ | R3核心，同时影响R1/R2；无需等新实验即可完善 |

不应把三位评审简化为“都要更多baseline”。R1/R2对效用解释的担忧更根本，R3的实验定位和scope问题也不是单靠加一个方法能解决。

## 2. Norm top-50%：用户准备什么？

### 2.1 必需材料

- 一个当前冻结的DTV/LOO正式启动命令或run配置，尤其`group` scope、policy-only打分、full-update、min-keep/cap、KL和序列长度。
- 可访问的原base model/tokenizer与原GSM8K训练/validation/test数据，不需要复制数据进git。
- 原实验seeds、结果JSON/TensorBoard路径，以便配对比较；保存实际命令，不仅提供目录截图。
- 若要历史行为分析，可再提供每completion日志，至少包含step、prompt/group/unit ID、self或norm、DTV/LOO mask、advantage；有clean/observed rewards和correctness更好。

**不需要历史完整梯度、不需要已训练的最终模型来启动这个baseline。** 它需要从与其他方法相同的初始化进行新的训练；只用旧DTV日志离线改变mask不能得到norm方法的最终性能。

### 2.2 需要新增什么代码？

计划新增独立norm trainer/launcher/tests，只有opt-in接入。不修改已有DTV、PODS或robust trainer算法。

score为`q_i=||g_i^score||²`。默认建议同prompt4条回答保留top2；评审未明确median范围，若其意为全batch，则应另做16条中的top8。两者分别标记group/batch，不能默默互换。ties采用明确确定性规则；不能把有ties的top50称严格`>median`。

保持原4条回答算出的advantages、原完整update loss、KL、mask归一化、optimizer和clipping，不按保留子集重算advantage。这样控制的是“是否需要方向信息”，而非同时变换多个算法组件。

### 2.3 最终输出

- `results/*eval_accuracy__meta.json`：与现有方法一致的pre/post accuracy、partial、format、num_correct、total。
- `selection/norm_selection.jsonl`（计划）：unit ID、score norm²、selected、tie信息、observed advantage、是否噪声组等。
- norm/filter summary、TensorBoard与wall-time/实际保留率/HBM报告。
- 正文/回复：paired seed的最终性能表与实际保留率，说明大小-only能否解释DTV收益。结果可能支持，也可能削弱“方向信息必要”这一主张，不能预写结论。

## 3. 固定checkpoint：用户准备什么？

### 3.1 最优材料包

建议首先提供一个可恢复的Vanilla GRPO中期run，而不是每方法拿一个不同模型：

1. run根目录、完整启动命令、code revision或代码备份、真实训练step。
2. `checkpoints/actor/<step>/`的目录清单。先只提供目录树，不需要上传几十GB。需要定位`model_params`、optimizer moments/count、训练step等具体保存项，目录名不能保证optimizer一定存在。
3. 原base model、tokenizer、LoRA rank/alpha、参考模型信息；恢复LoRA参数必须对应原backbone。
4. 原training pool与validation分区规则/seed/IDs。当前代码的数据shuffle由`my_example/seeding.py`决定，不能想当然所有seed都固定shuffle42；按实际run配置审计。
5. 可选已有候选token/answer记录。如果没有，可以在固定checkpoint生成一次新的共同候选并保存，不必bitwise重放整个旧训练过程。
6. 投稿Methods/Theory中Assumption及定理的原始PDF页或TeX。当前工作记录只有评审整理意见，不能据此假装已逐行核查正式定理。

只存在最终`model/`导出、旧optimizer已删除时，并非完全无法诊断：可以做明确标注的SGD-direction或新optimizer代理，但不能称“历史AdamW的一步真实效果”。若只剩最终step且原LR为0，照原scheduler做一步没有辨识力；优先恢复中期checkpoint，或把非零步长方向实验明确标为独立probe。

### 3.2 最小规模与过程

- 固定1个非零LR中期checkpoint。
- 固定8批×4 prompts×4 answers，共32 prompts/128 completions；每批只生成一次。
- 固定64–128个不在训练中的clean validation prompts；不用1319条benchmark test调诊断。验证prompt与候选做去重。
- Vanilla/DTV/LOO从相同参数和相同AdamW状态开始，分别临时更新一步；恢复后再跑下一个方法/候选batch。相同原reward污染、advantages、old/ref logps，不能为一个方法重新rollout。
- 每次更新前后计算独立clean reference-answer NLL：`U_b(M)=L_H(theta)-L_H(theta+Delta_b,M)`。U>0表示该proxy下降。
- clean标签只参与诊断，不参与任何正式selector或训练；正确答案teacher-forced NLL只是proxy，不等于生成数学正确率。

### 3.3 要新增的独立脚本与输出（尚未实现）

建议`audit_grpo_fixed_checkpoint.py` + `.sh` + CPU/toy/TPU smoke tests；复用模型/数据/梯度逻辑，仅在诊断进程内克隆/恢复状态，不修改正式trainer。

建议输出：

```text
audit_manifest.json            # checkpoint、model、optimizer可恢复性、数据划分与全部配置
fixed_candidates.npz           # 必要tokens/masks/old/ref logps/advantages，避免重新rollout
unit_utility.csv               # score、mask、clean/noisy、advantage、correctness、utility proxy
batch_utility.csv              # 每batch每方法更新前后NLL、U、kept count、update norm
audit_summary.json             # 配对差、相关性、分层计数、零/近零统计、限制
hbm_report.json                # 单worker拓扑与逐阶段实际内存
```

NPZ只是存储设计，尚未编码。不要事先删除checkpoint。完整梯度树不必保存到磁盘。一次只驻留一个模型/当前optimizer，按小批累计heldout梯度，三个counterfactual依次恢复；需要先验证恢复前后的参数与optimizer状态校验值一致。

最终展示优先两个小表，而非只画一个相关系数：

- 每方法的clean NLL单步改善U及相对Vanilla配对差。
- kept/dropped unit的独立一阶utility、clean/noisy和advantage符号分层，特别看rare-correct units是否更容易被删除。

8批用于可行性，不当作5个独立seed；有信号再扩checkpoint/seed。相关性受ties和零梯度影响，分别报告全量与非近零子集及样本量。

## 4. 理论回应应如何与这两个实验连接？

设独立clean loss为F，其梯度为h；某更新为`theta'=theta-eta*d`。若F是L-smooth，则

\[
F(\theta-\eta d)\le F(\theta)-\eta\langle h,d\rangle
+\frac{L\eta^2}{2}\|d\|^2.
\]

因此“梯度有用”需要相对于clean h的方向和有限步长条件，不只是相对于内部reference的正内积。

写内部reference为r，误差`e=h-r`，则由Cauchy–Schwarz：

\[
\langle h,d\rangle\ge\langle r,d\rangle-\|e\|\|d\|.
\]

所以一个充分条件为

\[
\langle r,d\rangle>
\|e\|\|d\|+\frac{L\eta}{2}\|d\|^2.
\]

这是用于解释条件缺口的基本推导，不冒充投稿定理或声称已估计L。对于AdamW，直接使用实际参数位移Delta时，平滑性界写成`F(theta+Delta)<=F(theta)+<h,Delta>+(L/2)||Delta||²`；不能把未预条件的raw gradient内积当AdamW的实际改善。

各项证据的职责：

- norm baseline控制梯度大小因素，但不检验r是否接近h。
- 固定checkpoint probe直接观察独立F的一步变化与一阶联系，检验reference有效性的经验范围。
- baseline胜负检验实用竞争力，不证明r可靠，也不能证明普遍的降噪能力。
- 精确Assumption/定理文字需要原稿核查；理论可以保持条件性，经验陈述应限定测试场景。

尤其在GRPO中整组policy梯度反号的构造下，`g_i'=-g_i`、`r'=-r`会保持内部DTV/LOO内积不变，而与固定clean h的内积反号。因此内部一致性不能普遍识别整组污染。该构造不是说实际dense rank reversal每次都等于整体梯度反号，而是指出“内部agreement等价于clean utility”的无条件说法不成立。不能仅用DPO的多数clean直觉覆盖GRPO组内reference。

可用英文框架（需与正式定理核对，不能预填实验结果）：

> Our guarantee is conditional on the relationship between the endogenous reference and the target objective; positive internal alignment alone is not a universal certificate of clean utility. We therefore distinguish a magnitude-only control from a fixed-checkpoint evaluation on an independent clean objective, and report the conditions and limitations under which the alignment signal is informative.

## 5. 三个新增方法的区别与论文命名

这里的三个方法指DS、PCGrad、CAGrad；norm top50是额外的评审点名对照，PODS已在运行。

| 方法 | 单元/输入信号 | 处理动作 | 是否refill | 论文中准确名称 |
|---|---|---|---|---|
| GRPO+DS | prompt组的observed reward方差 | 删常数reward组，生成新组补满；每个入选prompt保留4条回答 | 是，必须报告额外token/时间 | GRPO + DAPO-style dynamic sampling，不称完整DAPO |
| PCGrad | 原本是多个任务loss梯度；本实验拟同prompt completion gradients | 投影冲突分量后合成梯度，通常不删回答 | 否 | per-completion/groupwise PCGrad adaptation |
| CAGrad | 原本多个任务梯度与平均梯度 | 解约束/权重问题，合成兼顾最差局部改善的方向 | 否 | per-completion/groupwise CAGrad adaptation |
| Norm top50（对照） | 相同打分参数空间的gradient norm | 保留top2/4，不看cross方向 | 否 | gradient-norm top-50% (group)，明确ties |

PCGrad/CAGrad原文已有multi-task RL实验，不能称不适合RL；但是将completion当task是实质适配，需要披露。CAGrad还需要预先确定c、solver和rescaling。两者属于更新规则变化，不能伪装成mask selector直接接上。共同KL项如何处理需在实现前明确，避免重复加KL或让KL遮蔽policy conflicts。

写法：把DS/PODS归为reward/rollout选择，把norm归为magnitude-only，把PCGrad/CAGrad归为gradient manipulation，DTV/LOO归为内生reference下的unit filtering。不要声称四类方法优化同一个估计量或有相同生成预算。

一手来源：

- DAPO原文：https://arxiv.org/html/2503.14476v1#S3.SS2
- DS代码：https://github.com/verl-project/verl-recipe/blob/0a9360bb366d8bd71e26fb86da1acf077da9e6d9/dapo/dapo_ray_trainer.py
- PCGrad：https://arxiv.org/html/2001.06782v4#S2.SS3 ；https://github.com/tianheyu927/PCGrad
- CAGrad：https://arxiv.org/html/2110.14048v1#S3.SS1 ；https://github.com/Cranial-XIX/CAGrad

## 6. 本轮已开发：GRPO＋DS，其他项目仍为设计

主要文件：

- `my_example/dynamic_sampling.py`：std>0 selector、训练分区循环、同策略refill、完整组gather/padding、cap报错、独立selection日志。
- `my_example/train.py`：仅新增`TUNIX_DYNAMIC_SAMPLING=1`时启用的factory分支；CLI结构和冻结baseline入口不变。
- `my_example/run_ds_baseline.sh`、`run_ds_baseline_suite.sh`：single-worker拓扑、原4×4/691步预算与评测、规范rebuttal命名。
- `tests/my_example/dynamic_sampling_test.py`：算法、refill、对齐、production rank-noise host函数、cap/queue、CPU JAX padding、dry-run。
- `my_example/test_ds_smoke.sh`：服务器clean/mismatch小规模完整pre/post训练与结果结构验证。
- `my_example/test_ds_hbm.py`：服务器一次真实生成更新、两次256/768满长度强制两轮refill的压力测试。

正式输出目录：`logs/rebuttal_grpo_gsm8k_ds_seed5_mismatch0.2_YYYYMMDD_HHMMSS/`，包含selection、tensorboard、checkpoints、model、results和exit_code。selection保存metadata、每组observed rewards/eligible/selected/reason/noise标志、summary。`prepared_updates`明确是已准备batch数，不能替代完成actor updates/exit_code检查。

默认cap20是防止无限采样的安全上限，不是声称论文最优超参数。达到上限仍不足4个有效组即报错，无部分更新、无静默fallback。自动resume尚未实现，因为需要一致恢复额外候选cursor与rollout RNG；新run拒绝历史selection文件或已恢复的训练step。checkpoint仍可用于后续诊断。

尚未在服务器测试前，不能称已通过TPU/HBM、已可正式训练或保证不会OOM。当前实现先为每个候选4×4块计算原有logps再筛选，便于最小适配，不能声称省掉被丢组的reference inference成本。

本地验证：20项DS测试通过（包括真实CPU JAX/Flax gather与padding、production host noise函数、factory opt-in隔离）；bash语法、Python编译与diff空白检查通过。PODS轻量环境13项中11通过/2个框架测试跳过，alignment轻量环境37项中35通过/2个autodiff测试跳过。另一次尝试在本地不完整JAX环境运行框架回归，因缺`metrax`等训练依赖在导入阶段失败，不是已通过的完整框架回归；服务器需运行完整测试。

本轮未实现norm trainer、fixed-checkpoint probe、PCGrad或CAGrad。它们的所需材料和验收已在本文写明。

## 7. 服务器测试与提交操作

不要在正在跑PODS的同一worker执行TPU测试。空闲worker上先pull，再CPU单测，然后运行smoke/HBM；脚本将physical worker1作为独立logical process0，使用4个本地设备，不等待worker0。

```bash
cd /home/jason_chia925_gmail_com/Project_8k/tunix
git pull --ff-only origin for_GRPO_8kBaseline
source .venv_jax081/bin/activate
env JAX_PLATFORMS=cpu python -m unittest discover -s tests/my_example -p 'dynamic_sampling*test.py'
env JAX_PLATFORMS=cpu python -m unittest discover -s tests/my_example -p 'pods*test.py'
env JAX_PLATFORMS=cpu python -m unittest discover -s tests/my_example -p 'alignment_*test.py'
bash my_example/test_ds_smoke.sh
```

测试全部通过后另行启动正式seed5，再扩5seeds；本轮不自动启动训练。worker1没有git时需从worker0同步本次精确文件，不要在工作中的worker0启动额外TPU进程。
