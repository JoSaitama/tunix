# Fixed-checkpoint reference / utility audit：实施前协议

日期：2026-11-08。状态：定义与材料清单已确认；本文件不是已完成的诊断实现或实验结果。正式训练代码不变，norm baseline 仍暂停开发。

## 1. 需要检验的命题

教授提出的长期直觉可表述为：在训练访问的状态上，训练可见梯度的期望方向通常与干净目标方向相容。它不是“污染率低于50%”自动保证的结论。

固定参数 theta，令 h 为干净目标梯度、b 为污染条件下梯度均值，且污染组比例为 epsilon。若确实可将当前训练分布分解为这两个条件分布，则

`E[g_observed] = (1-epsilon) h + epsilon b`，

`<h, E[g_observed]> = (1-epsilon)||h||² + epsilon<h,b>`。

第二式为正是期望方向相容的条件，不是仅由 epsilon<0.5 得出的事实。实际GRPO条件分布、组内相关、有限样本方差以及模型状态变化须明确；这里不声称所有prompt都共享同一梯度分布。独立heldout估计h仍有采样误差。

固定checkpoint诊断只能提供访问状态上的局部实证。早/中/晚多个checkpoint可检查时间覆盖，不能把某个final模型的正值称为整条训练轨迹的证明，也不保证少数正确/负advantage单元不被误删。

## 2. 材料及恢复限制

优先一个Vanilla GRPO共同checkpoint：同一模型上比较各selector，避免拿不同已训练方法的模型混为筛选效应。

- 一个实际run目录、原启动命令/配置、seed、step、源码版本。
- `checkpoints/actor/<step>/model_params/`的真实目录清单；先不上传模型。优先中期，早/晚可后补。
- backbone版本、tokenizer、LoRA rank/alpha；若模型保存项不是LoRA-only，应按实际保存结构恢复，不能强行套LoRA target。
- 原training/validation划分及shuffle/noise seeds。独立heldout不得和候选重叠，不用最终test选步长或方法。
- 历史completion不是必需：恢复模型后生成一次共同候选，固定tokens、mask、observed reward、advantage、old/ref logps后供所有方法复用。

当前`tunix/sft/checkpoint_manager.py`只保存model_params及metadata，不保存AdamW optimizer moments。`my_example/audit_alignment_projection.py`已有LoRA Orbax恢复示例，但其run metadata属于alignment baselines，不能原样要求旧Vanilla/DTV run存在同样文件。

只有最终merged `model/`时需按导出格式另写恢复检查；并非不能诊断。若模型及checkpoint都没有，仅CSV/JSONL不足以重建真实梯度。预训练模型可做t=0诊断，但不能替代历史训练状态。

### 2.1 现有保存链路核查补充（2026-11-08）

- `run_seeded_full.sh` 的原Vanilla/DTV实验将参数和导出存到 `runs/<run_name>/checkpoints/actor/<step>/model_params`、`runs/<run_name>/model`；results和TensorBoard在 `logs/<run_name>`。不能只搜索logs。LearnAlign/GradAlign launcher则把参数与日志放在同一个logs run下。历史手工override和已迁移归档还可能使用其他路径，真实stdout中的Run/Logs或checkpoint-root为准。
- `TrainingConfig`及parser当前默认save_interval_steps=500、max_to_keep=4；`build_cluster_config`传入Orbax。PEFT trainer周期保存`self.model`，LoRA模式只存`nnx.LoRAParam`；close时force保存最后一步。691步且无override/保存失败/清理时，预计可有step500和691，但不是对历史目录存在性的保证；500是中后期，不声称存在250/350快照。
- `CheckpointManager.save`不保存optimizer或rollout/dataset RNG。Vanilla默认`custom_checkpoint_metadata()`为空，不能期待metadata自动包含完整启动参数、seed、dataset split、backbone revision。需日志/命令/代码版本补足。确切历史运行版本仍需云端核查。
- `main.py`训练后已有恢复最终LoRA checkpoint、重新构建sampler做post-eval、导出merged模型的路径。这说明保存格式设计支持模型恢复，但本轮没有执行真实历史checkpoint restore；不能把静态检查当作完整性验收。
- 梯度/一阶utility/独立SGD-direction probe无需改变原保存逻辑；新增独立诊断脚本即可。若未来要求真实AdamW重放，需要另外保存/恢复moments、step、scheduler及相关运行状态；只可支持新快照，不能从旧LoRA参数反推出旧moments。当前不为此改训练代码。

### 2.2 已定位的首个输入（用户云端输出，尚未下载/恢复）

- GCS：`gs://node-v5-jason/GRPO_GSM8K/clean/grpo_gsm8k_baseline_seed0_clean_20260717_144514/checkpoints/actor/500/`。
- 用户测得整个step500为68.93MiB，层级含`_CHECKPOINT_METADATA`和`model_params/`；仅由压缩磁盘大小不能推断参数量、rank或完整性。同run另有691但初次无需下载。
- 归档git revision：`6fba11c8d2e4cb00ee19e8ed45e196521d3f93d0`，branch `for_GRPO`。本地git object已存在，已用只读git show核查；不需要checkout/切换当前开发分支。该commit的`my_example/model.py`与当前文件无diff，config的LoRA默认64/64、Gemma3-1B；实际run是否CLI覆盖仍需元数据/日志。
- 历史parser默认completion length=768（当前parser为512），历史data.py固定shuffle42且无seeding.py，reward也有环境变量分支。不能用当前默认参数或仅run名自动复现历史采样、reward和split；seed0的数据shuffle恰与当前seed0相同也不代表完整运行状态等价。
- 计划下载到服务器独立`logs/utility_audit_inputs/vanilla_seed0_clean_20260717_144514/checkpoints/actor/500/`并保存五个版本/环境txt。只复制一个完整step和小记录，不取merged模型。先做CPU元数据/形状核查（不初始化JAX），再空闲worker真实恢复/HBM，未实施。

## 3. 两类外部目标：明确区分

### 3.1 独立clean GRPO surrogate（优先用于reference检查）

在独立clean heldout prompts上按当前checkpoint生成回答，用原干净reward构造冻结advantages，保持训练中的policy score loss、token reduction和可训练参数空间。以loss-gradient记号定义

`h_PG = mean(grad ell_policy_clean_heldout)`。

这是当前on-policy快照的独立clean策略梯度估计，目标比正确答案NLL更接近GRPO的训练机制；仍是有噪声的局部surrogate，不是长期return或exact success-rate gradient的保证。不得用本次heldout rollout去正式筛选或更新训练模型。组内全零advantage保留计数；h接近零时alignment应报NA/低辨识力，不补造方向。

### 3.2 正确答案NLL（可选有限步代理）

独立clean prompts及gold numeric answers的teacher-forced NLL，需预先固定答案模板和token mask。定义 `F_NLL(theta)` 与 `h_NLL=grad F_NLL`。若原数据加载只保留最终数字，不能把它描述为完整解题推理链监督。

NLL下降只说明该独立proxy改善，不能写成greedy accuracy、数学推理成功率或期望RL return提升。不得混用h_PG和h_NLL统计不标注目标。第一阶段可仅实现3.1及其固定rollout surrogate的有限步检查；NLL不是强制先做两套的要求。

## 4. 初版规模和输出数字

可行性：1个checkpoint；候选32 prompts = 8 batches x 4 prompts x 4 completions；独立clean heldout先64 prompts x 原4 rollouts。原配置从run读取并核对，不把当前config默认当历史配置。候选随机子集及其seed在看结果前确定。

用原policy gradients `g_i`、原group/batch scope计算DTV、LOO原始分数和实际mask；minimum-retention cap、policy-only/full-update分支必须对应原配置并在结果中记录。对group reference `r_p=mean_i g_pi`，主要报告：

1. `cos(r_p,h_PG)`的分布及非近零reference的正对齐比例；同时报告所有零reference的数量/比例，不能仅删去零组后报一个高比例。
2. 平均reference与h_PG的cosine，按采样prompt加权汇总；不能代替上一项分布。
3. `u_i=<h_PG,g_i>`与DTV/LOO score的Spearman，全量及非近零子集，注明样本数和ties。
4. 各method kept/dropped的positive-utility比例、mean utility、保留数。
5. DTV-keep / LOO-drop的conflict集合：count、positive-utility比例、clean correctness与advantage符号分层；理论阈值分歧和cap后实际分歧分开。
6. 组内唯一正确回答、negative-advantage且positive-utility单元的删除率；仅为observed rare-correct，不外推所有稀有推理模式。

reference与其自身均值/同批g_i的正内积不能作为独立验证。完整batch内 `sum_i <g_i,mean_j g_j> >= 0` 是代数恒等式，不证明clean utility。

不确定性以prompt为cluster。多checkpoint同seed不是多独立训练seeds；heldout h的估计噪声也须检查，可分heldout子集累计h再算结果，不将completion数当独立样本量。第一批仅可行性，不能保证已有统计power。

## 5. 有限步检查及可比性

原始参数theta0固定。每个候选batch、每个方法从theta0开始，取对应实际聚合的完整loss-gradient d_M（policy-only variant另依其规则保留KL），做独立SGD-direction probe：

`theta_M = theta0 - eta_probe * d_M`，

`U_M = F_clean(theta0) - F_clean(theta_M)`。

F_clean可先使用3.1中固定独立rollout/advantage的GRPO surrogate，另记录其定义、old/ref logps与是否包含KL；不要称其为新的on-policy expected reward。若采用NLL则列名必须带NLL。

这不需要历史optimizer，但不是历史AdamW重放。eta_probe预先固定，FP32检查参数delta确实非零，不将最终cosine scheduler的零LR当probe。可预声明小范围步长敏感性，不看结果挑一个有利点。不同保留数导致更新范数差异，必须同时报告update norm；若另做equal-norm方向控制，明确它不是原始算法更新。

每次probe结束恢复theta0并核对参数校验值，不能接着上一个方法继续训练。若确有独立optimizer快照才增加历史AdamW分支，不能伪造旧moments。

## 6. 计划文件与安全边界

- `my_example/audit_grpo_fixed_checkpoint.py`：独立进程恢复、数据划分验证、固定rollout、原分数/mask、外部utility、有限步probe及输出；尚未创建。
- `my_example/audit_grpo_fixed_checkpoint.sh`：只启动诊断，默认限制当前worker，fresh输出目录，不重用训练checkpoint_root写入；尚未创建。
- `tests/my_example/fixed_checkpoint_utility_test.py`及TPU/HBM小测：覆盖符号、toy true utility、零梯度/空集合、mask分支、恢复和scope；尚未创建。

输出计划：audit_manifest.json、fixed_candidates、unit_utility.csv、batch_utility.csv、audit_summary.json、hbm_report.json；少量tokens与标量，不保存整个候选池50M维梯度到磁盘。逐completion/小分块计算梯度、精确tree点积，不沿用有碰撞误差的哈希投影。分块能否严格匹配生产batch形状需测试，不能仅凭线性梯度公式声称数值等价；原训练文件不改。

先CPU toy和参数恢复测试，再单worker真实4x4及满长度HBM检查；尚未测量耗时/峰值，不能承诺安全、运行时长或诊断结果。不得与同worker现有TPU训练并行。

## 7. 恢复前置自检已编码（尚未服务器验证）

用户已下载上述step500：本地约70MiB，顶层有step metadata，model_params含`_METADATA`、`_sharding`及`manifest.ocdbt`。归档环境Python3.11.15、JAX/JAXLIB0.8.1、Flax0.12.6、Orbax0.12.1、qwix0.1.8。这不等于实际恢复已通过。

新增三个独立文件，不改现有训练：

- `my_example/audit_grpo_checkpoint_restore.py`：`metadata`模式只用标准库解析元数据、检查payload存在、版本记录和全step SHA256；不import JAX。`restore`模式在单process四local TPU加载base+LoRA，按当前mesh抽象target恢复，检查tree/shape/dtype/finite/actor参数digest、源文件未变，并记录恢复阶段内存。没有optimizer、trainer、rollout、gradient或utility计算。HF backbone可用`--backbone-path`显式指定现有snapshot，恢复尚不能独立认证历史HF revision。
- `my_example/audit_grpo_checkpoint_restore.sh`：默认metadata，restore时先拒绝常见并发my_example模型任务、固定单worker拓扑，不复用训练checkpoint-root；使用服务器`.venv_jax081/bin/python`，输出独立目录，不改source。
- `tests/my_example/fixed_checkpoint_restore_test.py`：12项标准库测试，包括缺payload/metadata、commit/JSON非法、输出覆盖拒绝、symlink拒绝、输入hash变化、metadata模式无JAX导入和失败报告。按用户要求只给服务器运行命令，本机未执行这些测试。

服务器命令：

```bash
python -m unittest discover -s tests/my_example -p 'fixed_checkpoint_restore_test.py'
bash my_example/audit_grpo_checkpoint_restore.sh metadata
# 仅当前worker空闲时
bash my_example/audit_grpo_checkpoint_restore.sh restore
```

PASS必须带mode解读：metadata PASS不等于restore PASS；restore PASS也不等于utility有效或梯度程序不OOM。`restore_report.json`始终带`utility_audit_completed=false`，恢复内存结果明确不代表rollout/backward峰值。报告目录存在即拒绝覆盖，默认带mode/seed/step/时间戳；默认input/commit只对应本次归档，其他run需显式override。
