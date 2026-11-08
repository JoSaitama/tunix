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

## 8. R2原文核对、诊断主次与恢复bug修复（2026-11-08）

已完整阅读`../ICLR2027.pdf`四页并视觉核对p2。R2不仅问reference是否同向，明确要求fixed-checkpoint filtering decisions能否predict improvement on a separate clean objective。R2还点出same-prompt reference与whole-group rank reversal、IF reference替换不保证utility ranking。R1要求assumption条件和minority信号，R3要求区分算法/参数化/任务协议与收窄一般RL外推。本文实验只能直接支持前两者的局部机制，不能独自解决跨模型/泛化和论文呈现问题。

### 8.1 三种方向不能混用

- `r_B`：当前候选batch或同prompt group的内部policy梯度reference。LOO还需对每个unit去自身，不能把DTV/LOO说成完全相同reference。
- `d_V(B)`：同一候选batch上Vanilla的实际更新方向；若包含KL/AdamW，不必等于`r_B`。
- `h_H`：独立clean heldout目标在共同theta上的梯度，不是候选batch的Vanilla方向。由Vanilla训练所得的checkpoint只是共同模型状态，不将外部检查变成与Vanilla自己的比较。

`cos(r_B,h_H)`支持reference质量，但不能单独回答score ranking和筛选后utility。主要证据应为如下两层，第一层alignment仅辅助：

1. 每单元score与独立一阶utility相关：`u_i^PG=<h_H,g_i^PG>`（纯policy attribution）与`u_i^full=<h_H,g_i^full>`（full-mask variant的实际loss梯度方向）分别命名；finite-step更新按真实variant聚合policy/KL，不把policy utility冒称full-update utility。
2. 同checkpoint、同候选tokens/rewards/advantages、共享old/ref logps，各方法从theta0独立出发的一步clean objective变化`U_M=F_H(theta0)-F_H(theta0+Delta_M)`；恢复参数并核验，不串行累计训练。缺optimizer时仍是显式SGD-direction probe，不称历史AdamW。

在Vanilla/DTV/LOO外，加入按各方法同样scope/每组实际保留数的random keep匹配对照（独立多个mask seed，提前固定），区分选择质量与仅减少数据/改变范数的效果。实际聚合归一化一致，并报告retained count/update norm；若有equal-norm sensitivity，标成额外方向控制，不冒称原算法。norm对照可作为offline方向诊断扩展，不据此改变仍暂停的norm训练baseline开发状态。

summary应至少包括score/utility的全量与非近零Spearman和计数、kept/dropped的正utility比例、matched-random的utility、finite-step U及DTV-LOO配对差，分clean/noisy groups与正/负/零advantage。conflict单位的utility符号以及唯一正确回答误删率用于解释差异；不预设GSM8K必然DTV较好。

heldout独立不等于无噪声真梯度：PG surrogate仍受有限rollouts与稀疏reward影响，NLL若另用只能称supervised utility proxy。需要零方向辨识力检查、prompt-cluster不确定性，后续扩checkpoints。独立clean数据仅用于audit，不能接入正式filter或选取最有利结果。单clean step500可行性不等于验证污染训练全过程。

### 8.2 整组污染的结构限制

作为数学构造，若同一group所有`g_i`同时变成`-g_i`，DTV与LOO的内部内积不变，而对固定外部h的utility反号。这说明组内agreement不可能无条件识别所有whole-group方向反转。实际dense reward rank reversal不一定恰等于逐advantage取反；不能把构造当作已测数据事实。诊断必须允许不支持排序的结果，不能为证明assumption而挑checkpoint或外部目标。

### 8.3 用户restore失败的原因及最小修复

异常在`qwix.apply_lora_to_model`的dummy forward：Gemma `get_model_input()`的batch=2，被自检脚本外层`with mesh`置于fsdp=4物理mesh下，activation `(2,1,1152)`无法按第一维4分片；尚未执行Orbax参数恢复，不是已发现checkpoint损坏或HBM OOM。

仅修改独立`audit_grpo_checkpoint_restore.py`：与正式`my_example/main.py`一致，`load_model`自己进入mesh，LoRA dummy forward不额外包外层mesh；原`apply_lora`仍在自己的内部mesh中重分片参数，后续restore仍使用4x1 mesh。没有修改dummy/训练batch、LoRA rank、权重、mesh大小或正式model.py。新增静态AST回归测试，防止初始化再次被包进mesh；现13项测试需服务器执行。CPU Gemma的shard函数会直接跳过约束，普通CPU运行无法复现该TPU专属报错。

本机仅语法检查；服务器尚未重新restore。utility计算仍未实现，当前修复只是前置恢复检查。

### 8.4 服务器恢复已通过（用户反馈）

用户回传修复后命令`bash my_example/audit_grpo_checkpoint_restore.sh restore`的摘要：`passed=true`、`mode=restore`、`utility_audit_completed=false`。报告为服务器`logs/utility_checkpoint_checks/restore_seed0_step500_20261108_040038/restore_report.json`。按当前脚本，这支持参数恢复/结构与有限值/应用后digest/输入源未变等既定检查成功；本机未读取完整报告，不填造参数维度或headroom数字。

这解除首个clean Vanilla step500作为audit输入的恢复阻碍，不需为获得模型重训或重下载。后续同模型候选rollout、独立heldout梯度、score→utility和finite-step更新仍未实现/验收；恢复期间的内存检查不代表反传或rollout HBM压力。用户这次只提供restore摘要，不据此声称13项单元测试或utility已通过。输入目录和成功报告保留，不要求另外方法checkpoint；正式训练代码/参数保持冻结。

## 9. Utility诊断独立实现（2026-11-08，待服务器测试）

按用户要求，新代码全部放`my_example/rebuttal/`，不改生产trainer、model/config/CLI、checkpoint保存、旧恢复自检。完整算法定义/假设/命令/限制见该目录`README.md`。新增入口`audit_grpo_utility.py`及launcher `run_utility_audit.sh`，NumPy统计/对照`utility_math.py`，TPU执行`utility_runtime.py`和两份CPU测试文件/launcher。

已实现preflight、smoke、hbm、score、probe五种模式：独立TFDS train/heldout、共享固定回答、原dense reward分支与原prompt-hash mismatch、原GRPO loss/advantage、policy/full梯度区分、精确LoRA tree点积、生产风格DTV/LOO分数与cap、同保留数random控制、reference辅助cosine、unit相关性及分层、有限步SGD-direction surrogate改善/参数恢复、零delta与数值分辨率、源hash/阶段HBM。clean F的forward采用同样per-unit/vmap形状，避免不同batch AD路径混用。没有梯度压缩、不写全梯度池、不创建optimizer/trainer、不运行正式benchmark test或保存模型。

输入模型来自clean；candidate-noise0.2仅为此状态敏感性，不替代mismatch训练模型证据。强制显式reward-mode，保存数据版本及未认证历史split/config/HF revision警告。matched random的mask不读取heldout h/clean audit标签。统计按prompt/batch聚类，但区间条件于当前h/checkpoint，不伪装训练seed不确定性。stress全长数据不进入科学汇总。程序完成不自动宣称assumption/实用性已验证。

本机仅AST、bash -n、diff检查；按用户要求所有CPU/JAX toy、真实smoke及满长度HBM必须在服务器执行，当前不能声称通过。正式数字须恢复/CPU/smoke/HBM和配置核对后产生、检查，先勿启动大规模probe或据此写结论。

### 9.1 NNX状态快照别名修复（2026-11-08，待服务器回归）

用户nohup smoke到最终SHA256失败，退出码1。确定原脚本存在快照别名缺陷：`self.snapshot=nnx.state(actor,LoRAParam)`保留可变Variable引用，原逐probe比较可能自比较，不能保证每branch从同一theta0开始。核对Flax v0.12.6官方graphlib的state/flatten逻辑（https://github.com/google/flax/blob/v0.12.6/flax/nnx/graphlib.py）后，改成新增`parameter_snapshot.py`中的独立copied-array tuple；每次生成新State包装以恢复，前后按路径/shape/dtype/bitwise检查，保留最终初始SHA检查，不放宽allclose。失败记录具体变化，不自动fallback。

新增`roundtrip`模式用于实际checkpoint两次扰动/恢复，不取数据、不rollout/反传；只检查恢复且utility标false。新增6项`parameter_snapshot_test.py`服务器CPU回归，覆盖快照不随actor变化、包装可丢弃、signed zero与微小bit变化、真实probe的成功/异常finally、dirty起点拒绝、真实roundtrip；其余utility测试仍保留。额外占用仅一份LoRA参数数组，byte数记录到manifest。只改rebuttal独立文件/launcher/docs，不改训练参数、loss、原数据或源checkpoint。旧失败run的有限步比较不作为有效证据，保留作debug；无须重训或重新下载。全部模型/CPU测试仍仅服务器执行，本机静态检查不算回归通过。

## 10. 评审原文与实现再审查（2026-11-08）

已重新完整读取同级ICLR2027.pdf四页并渲染核对p2。当前用户回传roundtrip及smoke通过，修复后的参数恢复已验收；smoke4候选/8heldout prompts不足以构成机制证据，full-length HBM仍未验收。本文前面“计划/未实施”的段落保留为历史协议，当前实现以my_example/rebuttal/README.md和本节为准。

| 评审问题 | 本诊断能回答的部分 | 必须保留的限制 |
| --- | --- | --- |
| R1 W2/Q3：Assumption1是否有依据 | 各prompt reference与独立clean h的cosine、正对齐比例、零reference及不确定性 | 不是全程/所有任务保证；不能用全数据污染<50%直接解释整组GRPO reference |
| R2 W1：agreement是否预测有用信号/排名 | 更新前score与dot(h,g_i)的关联、kept/drop效用及norm对照 | 一阶梯度代理不是原influence-function utility的排名保真定理，也不是accuracy |
| R1 minority风险/R2 Q1过滤分层 | clean/selected/effective noise、advantage、correctness、unique-correct、观察到的组成功率分层 | 需分母/数量与联合noise×advantage统计；4回答成功率只是难度代理，unique-correct不是所有rare reasoning |
| R2 Q3：fixed-checkpoint independent objective improvement | 同theta0/同rollout下有限步U与matched-retention随机的配对差 | 当前实测与一阶预测约1097倍，不满足数值解释门槛；仅cos/rho不能声称完整完成此问 |
| R1：不看test如何选DTV/LOO | conflict的独立utility可解释相对行为、提出局部诊断依据 | 尚无验证过的无验证集事前选法规则；若用heldout选方法须披露额外验证资源 |
| R1 baselines/cost/scaling、R2 Table/Figure与跨分布、R3 scope/protocol | 本诊断不能替代这些回答 | 分别需要独立baseline/成本统计/结果口径核查/写作和claim范围修订 |

实现已具备：原始LoRA梯度精确FP32内积，无哈希；heldout不进入mask；policy打分/full-KL更新口径分开；真实mask/cap和matched随机；prompt-cluster相关性区间；源只读、精确参数恢复、单worker限制。score模式可不做临时参数更新。

正式统计前最小核查项（本轮未改算法或执行）：

1. 数据及配置：固定候选/heldout IDs、源checkpoint实际训练条件与reward分支；相同sample_seed在候选规模变化时不保证heldout不变。当前无跨run完整NPZ回放入口，不能冒称已支持；clean模型+候选污染只报告固定状态敏感性。优先预先固定128候选/128heldout、scope group和原cap，不按有利结果选样本。
2. h可靠性：当前informative仅表示norm>1e-12，并不表示方向稳定或接近真实目标。建议同一次heldout backward中分开累计h_A/h_B，报告两半方向及关键结论稳定性；当前bootstrap只针对候选，未包含heldout/训练seed不确定性。
3. 数值门槛：复用固定smoke tokens，先检查零步/同值参数更新、恢复后loss、heldout自身负梯度及预定正反向步长，比较实际delta和Taylor预测。old logps在16回答batched forward生成，loss/AD用per-completion vmap，形状/混合精度差异是应测量的具体风险，不是已确认根因；不能只放宽误差门槛或挑有利步长。若不能通过则停止有限步科学结论，明确只交付梯度对齐证据。
4. 统计补齐：主报全量，辅报仅按policy_gradient_norm非零界定的有效梯度子集；目前non-near-zero子集同时按score和utility筛选，须标注条件，不能独立承担主结论。补noise×advantage联合表、所有子集数量/分母、reference区间及paired first-order差的区间；数据已大部分存在CSV，不需额外训练。

必须承认的结构边界：若一整个prompt组的policy梯度同步变为-g，组内DTV和LOO分数均不变，但相对固定外部h的utility反号。现有toy测试覆盖此代数例子；真实dense rank reversal不总是严格梯度反号，不能泛化为每个污染组必然不变，也不能把方法描述成有保证的整组噪声检测器。

最小交付是两张数值表（reference/score与独立梯度对齐；过滤率与minority/advantage/noise分层），数值门槛通过后加一张局部finite-step表。不要求新训练seed、所有方法checkpoint或新coefficient；不把gradient-norm baseline、PODS/DS或R2图表口径问题视为已由此解决。score与gradient norm的相关性目前仅诊断，不能代替R1点名的norm-top50训练baseline。

## 11. Utility v2已补齐代码，等待服务器验收（2026-11-08）

- 正式trainer/config/训练CLI/PODS/DS完全不改，仅my_example/rebuttal中的独立模块、启动与测试及说明变更。
- 新utility_data.py用独立permutation流固定candidate/heldout源ID前缀，支持partition-file及prompt哈希核对；replay-rollouts读取原manifest/NPZ，禁止pickle/输入输出重叠，核对生产source hash/配置/恢复theta0和backbone snapshot身份。V1旧档案仅同污染条件回放，V2保存完整prompt strings可按原哈希重施candidate污染，heldout始终clean；不重采样原回答、不改源文件、不把新随机IDs冒充旧划分。
- h在原16回答反传路径内交替按完整4prompt批累计两半，无额外backward；新增两半cosine、reference/单元对两半的效用、norm-only有效子集、全量主统计、reference含零的mean dot/条件bootstrap区间及分污染组的reference统计。两半方向正相关不等于真实方向已知，区间仍条件于估计h，不伪装training-seed CI。
- 补noise-selected/effective × advantage三类的filter_strata.csv（units/prompts/dropped/correct分母）、既有minority风险、method vs Vanilla/LOO/matched-random的一阶配对区间（随机重复先在batch内平均）。assumption_report.json逐项报告正/负/跨零/不可计算的局部证据；assumption_proven永远false表示不宣称普适定理，不是软件失败。没有自动事前选法规则。
- 新numerics模式可复用用户成功smoke（20261108_061613）：不跑候选梯度扫描；检查旧full16 logps与per-unit前向差、零步、恢复后的F，以及预声明1e-6/1e-4的+/-heldout自身梯度响应，使用实际delta预测。预设rtol.1/atol1e-7及10倍分辨率门槛，不能据DTV结果挑步长或改精度/old logps。无通过步长则numerics退出2并保留报告；执行完成与数值验收区分。smoke/probe只在自检通过的步长上做method probe，每条仍需一致性，失败时保留raw但finite_step_validated=false；score只交付梯度几何。HBM合成probe仍只作资源压力测试，不作收益数字。
- 新test_utility_v2.sh只排CPU→metadata→numerics，失败停，不自动大规模probe。当前worker单process/四localTPU沿用，h两半新增聚合树的HBM需重测，不以旧91.53%余量替代；不改变16回答AD来绕过峰值。
- 新增数据/证据/微型JAX runtime/NumPy假runtime编排测试，现共54项已定义。遵用户要求本机不执行任何单测/模型/JAX计算，仅AST、ruff E9/F、bash -n、diff检查；服务器实际通过尚未获得，也未声称约1097倍偏差根因已消除。先在空闲worker跑小回放自检，回传numerical/heldout报告后再定正式128/128统计，不重新训练seed。

## 12. 一次有止损边界的FP32测量对照（2026-11-08）

用户已回传v2数值失败：零步/恢复通过，但1e-6不分辨、1e-4正反两向loss都增加；原full16与unit前向logps存在差异。不能据此确诊BF16为唯一根因，更不能更换候选到结果为正。

新增独立 `my_example/rebuttal/utility_fp32_control.py` 与 `run_utility_fp32_control.sh`，复用旧smoke的8heldout prompts/32回答，原mask、reward、advantage和token不变。无candidate AD、rollout、训练或模型导出；所有生产路径与原v2保持不变。该进程将actor浮点state扩为FP32并以highest matmul precision运行，核对实际loss/logps dtype与嵌套前向JAXPR的浮点中间类型。

**新目标必须单独标注**：用新FP32 unit forward在theta0生成独立old-policy logps，保存到新目录，原BF16源文件只读。h是探测中同一个mean(vmap(per-unit loss))标量的梯度，不再为这个数值自检保留16份逐样本参数梯度。这改变诊断目标的精度、old-policy与计算口径，既不宣称与生产排名等价，也不能单独归因某一种改动。只测原预声明1e-6/1e-4、zero与正反SELF，原容差/分辨率不放宽。

先服务器CPU回归，再当前物理worker独立四TPU进行一次小对照。TPU子进程45分钟预算，TERM后60秒可强制结束自己的子进程；不停止其他作业。实际HBM至少5%余量且计数完整才允许exit0，原91.53%余量不当作FP32安全保证；这不是全长度全valid合成压力测试。失败/超时不自动重试或排大扫描，先转向gradient-norm top50%训练baseline。通过也只说明FP32测量对照有效，不直接宣称R2问题已解答/长期效用/Assumption成立；后续科学证据仍需另外决定。
