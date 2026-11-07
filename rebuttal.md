# Rebuttal 工作记录与逐项行动方案

日期：2026-11-07。当前分支：`for_GRPO_8kBaseline`。

本轮仅分析和记录，不修改训练代码、冻结参数、PODS 启动配置或运行中的任务。本文区分“本地代码/数据已核实”“待运行验证”和“解释性假说”，不把计划实验写成已完成结果。先前整体评审分析见 `REBUTTAL_REVIEW_ANALYSIS_20261107.md`。

## 1. “复用逐样本梯度”需要什么历史记录？

### 1.1 这里应纠正措辞

复用的是现有训练器的**在线逐 completion 梯度计算能力**，不是读取历史保存的完整梯度。正常的 summary/JSONL/TensorBoard 标量通常不足以还原完整梯度树，也不应要求为训练新 baseline 先恢复全部历史梯度。

已核实本仓库：

- `tunix/rl/self_inf_trainer.py:64` 开始通过 `nnx.value_and_grad` 与 `jax.vmap` 计算逐样本梯度，支持单独的 score loss。
- `tunix/rl/self_inf_loo_trainer.py:_group_loo_statistics` 在线计算 `raw_self`、`standard_self_term`、`standard_cross_term` 与 LOO score。
- `tunix/rl/self_inf_policy_trainer.py`、`self_inf_loo_policy_trainer.py` 区分“KL-free policy gradient 打分”和“含原训练项的梯度更新”。新 baseline 必须匹配被比较 DTV 运行的实际梯度定义，而不是随意选含/不含 KL。

### 1.2 不同目的需要的文件不同

| 目的 | 最少需要 | 不足以替代它的记录 |
|---|---|---|
| 新训练 norm baseline | 当前代码、原启动配置、模型/数据可访问、固定 seeds | 不需要历史完整梯度或旧 checkpoint |
| 回看旧样本的 norm/filter 关系 | 同一 unit 的 norm 或 self、mask、step/group ID，最好有 advantage/correctness/noise 标志 | 只有跨样本 mean/std 无法恢复单样本选择 |
| 重算旧时刻的分数 | 对应参数 checkpoint、相同 token IDs/mask/advantages/old logprobs 等输入及配置 | 只有 prompt 文本与 seed 不保证重新 rollout 完全一致 |
| 实际 AdamW 单步效用诊断 | 参数 + optimizer moments/count + scheduler step，及固定候选输入 | 最终 model 导出通常不能替代 optimizer checkpoint |
| Figure/Table 对账 | 生成图表的原 CSV、实际运行配置/manifest、eval JSON、最终 checkpoint step | 仅截图或目录时间戳不能核对指标定义 |

如果已有 `standard_self_term = ||g_i||²/G` 且梯度定义和单位严格一致，在同一个 G=4 组中按该 self 排序等价于按 norm 排序。但旧 DTV 轨迹上重建 norm mask，只是离线对照，不能代替 norm baseline 从初始化开始训练后的性能。

下一步用户优先提供：一个完整的 DTV/LOO run 的配置或启动命令、selection/sample JSONL 的表头及少量记录、checkpoint 目录结构；不需要先传巨大的梯度文件。

## 2. Gradient-norm median 到底检验什么？

### 2.1 问题不是再提出一个复杂方法

评审在问：DTV 的提升，是否仅仅因为偏向较大梯度，而不是使用了梯度方向/一致性信息？这是一个必须直接回应的 magnitude-only 对照。

设打分梯度为 `g_i = ∇θ ℓ_i^score`，参数空间与对应 DTV 一致。令

\[
q_i=\|g_i\|_2^2.
\]

平方范数和范数排名完全相同，不需要额外开平方。DTV 的组内分数包含

\[
\langle g_i,\bar g\rangle
=\frac1G\|g_i\|^2+
\frac1G\sum_{j\ne i}\langle g_i,g_j\rangle.
\]

norm baseline 只用第一部分的大小，不使用 cross/direction。

建议首先定义为 `Norm top-50% (group)`：每个 prompt 的 4 条回答中，保留 norm 最大的 2 条。每个 update 仍生成 4 prompts × 4 completions =16 条，通常仅 8 条参加更新。举例：同组 norm 为 `[0.2, 1.0, 0.4, 2.0]`，保留第 2、4 条，不看它们与其余梯度同向还是反向。

**范围说明：评审的“median”没有在现有摘要里明确是全 batch 还是组内。组内 2/4 是我们为对齐当前 group-DTV 的适配选择，不是论文原方法或评审指定细节。** 若需字面全 batch median，则 16 条中保留 top8，可能某组全被删除；两者必须分别命名，不能混称。实施前确认主配置。

### 2.2 必须固定的公平性细节

- 不能因为 gradient norm 大，就声称该样本必然更有用；两个很大的梯度也可能相互冲突。
- 相同范数/零梯度时采用明确、可复现的 tie-breaking；严格 `norm > median` 在 ties 下不一定保留 50%，因此建议如实命名 top-50%。
- 保留原 4 条回答计算出的 advantages，与 DTV 式过滤对齐；不额外导入 PODS 的“入选子集重算 advantage”。
- norm 从与 DTV 相同的未做最终全局 clipping 的 score gradients 计算；更新 loss、mask 后归一化、KL、optimizer、LR schedule 和 clipping 顺序保持原语义。
- 不使用历史有问题的哈希投影；直接在当前可训练梯度树上按叶累加平方范数。
- fixed 50% 与 DTV 自动保留率不同，必须报告实际保留率。若预算允许，补 random-50% 有助于排除纯保留数量差异；已有 random-5%/10% 不能冒充 matched-retention 控制。
- 预期可以复用已有梯度开销，但不能承诺比 DTV 快多少或不 OOM；需要独立真实 rollout 与满长度 HBM 测试。

新代码方案（尚未实施）：新增独立 norm trainer/selector 文件、最小 opt-in 训练入口、独立 launcher/测试，不改 `robust_trainer.py` 或冻结 baseline 行为。

## 3. GRPO + DAPO-style dynamic sampling

### 3.1 原方法与代码来源已核实

原文 DAPO §3.2：二值 reward 下，全对或全错的组没有有效组内 policy advantage；去掉这样的 prompt groups，继续生成，直到凑足有效训练 batch。

作者项目仓库链接到 verl 实现。旧 `volcengine/verl/recipe/dapo` 路径已迁移，本次直接核实了维护仓库 `verl-project/verl-recipe/dapo`：

- `dapo_ray_trainer.py:239–264`：按配置 metric 对 prompt UID 分组，计算 `np.std`，保留 std>0 的组；代码另允许仅一条记录的特殊情况，当前 G=4 不适用。
- `:277–294`：不足 train batch 就继续生成；达到上限仍不足时抛错；充足后按 `train_batch_size × rollout.n` 截取。
- `README.md` 的 Dynamic Sampling 节明确 `filter_groups.metric` 可配置，并说明补满 batch 与 `max_num_gen_batches`。

不是“只删掉组但不补充”。也不是把 DAPO 的非对称 clipping、token loss、overlong reward 等全部搬进来。

来源：

- [DAPO 原文](https://arxiv.org/html/2503.14476v1#S3.SS2)
- [作者项目](https://github.com/BytedTsinghua-SIA/DAPO)
- [当前维护的 DS 实现](https://github.com/verl-project/verl-recipe/blob/main/dapo/dapo_ray_trainer.py)
- [实现说明](https://github.com/verl-project/verl-recipe/blob/main/dapo/README.md)

### 3.2 对当前 dense-reward GSM8K 的最小适配

主建议命名 `GRPO + DS (observed-reward variance)`：依据**训练可见的污染后总 reward**，保留组内 reward 非常数的 prompt；继续从同一 training pool 取 prompt、生成 4 条回答，直到 4 个有效 groups，再按原 GRPO 更新。

不能直接把“全对/全错”二值规则当成当前 dense reward 的同义词：即使四条都错，格式等分数不同仍可能产生有效 advantage。也不能在 mismatch 中用隐藏 clean correctness 做 selector，造成 oracle 优势。

当前污染是 `my_example/reward_rank_noise.py:reverse_reward_ranks` 重新排列同组 reward 值；在固定同组回答上，**reward 方差不变**。因此 DS 不会仅凭方差识别这种 rank mismatch。这是机制差异，不是预先宣称 DS 最终一定无效；新训练会改变模型和后续采样轨迹。

### 3.3 实现边界与预算

- 新增独立 `dynamic_sampling.py` 等文件封装 group 筛选/补采样，通过 opt-in 引入训练准备流程。具体 hook 需在编码时核实奖励计算、旧 logprob、prefetch 与 update 的先后顺序，不能只在 dataset iterator 层删 prompt 就声称已完成 DS。
- 同一次 update 的 refill 都要使用该 update 对应的策略状态；保存入选回答直接用于训练，不能筛完后另 rollout 一次而失去已筛内容。
- 保持每次有效 update 的 4×4、总 691 updates、模型、LR/KL/clip/长度/评测不变。为了补足有效 batch，需要允许消耗更多 training prompt 访问；691 是 update budget，而不是访问原池一次的保证。
- 原池可以循环，但不能读 test；候选耗尽、重复 prompt 的 occurrence/group ID 与缓存失效必须明确。
- 建议沿官方上限机制：达到预先设置的 refill cap 仍不足即保存诊断并停止，不能静默改成小 batch 或无限等待。上限数值在 smoke 观察有效组率后、正式结果出来前固定并记录，不按 test 调整。
- 需逐步记录 generated/kept/dropped groups、refill rounds、额外 rollouts/tokens、候选噪声率与入选噪声率、wall time、cap hit。
- 若单次组有效概率为 a，粗略需要 `16/a` 条生成回答/update；独立同分布仅是时间估计近似，不能保证 GSM8K 一定很快。HBM 应按固定小生成块顺序 refill，而非扩大一次性生成 batch。
- CPU 单测、single-worker TPU smoke、真实生成/满长度 HBM 通过后再先跑 seed5 mismatch20；确认日志/评测保存后扩到 5 seeds `5 0 13 21 42`。clean 随后补齐，同表比较不能把缺失设置当已完成。

后续进展（2026-11-07）：已新增独立GRPO+DS路径与launcher、单测、服务器smoke/HBM脚本；详见`GRPO_REBUTTAL_PREPARATION.md`。本地验证不等于服务器TPU验证，未启动正式训练；worker1可用也不绕过这些检查。

## 4. PCGrad / CAGrad：先理解，再决定是否实现

### 4.1 PCGrad

原问题是多任务：对每个任务梯度，遇到冲突内积就去掉沿另一梯度的负向分量：

\[
\tilde g_i\leftarrow\tilde g_i-
\frac{\min(\langle\tilde g_i,g_j\rangle,0)}{\|g_j\|^2}g_j.
\]

按随机顺序处理其他任务，最后合成更新。被对照的 `g_j` 是原始梯度，`tilde g_i` 会迭代变化；零范数需保护。

已读作者 [PCGrad_tf.py](https://github.com/tianheyu927/PCGrad/blob/master/PCGrad_tf.py)，是 TensorFlow optimizer wrapper，不是可直接插入本仓库的 JAX GRPO 代码。还发现其 `tf.random.shuffle(loss)` 返回值没有赋回，不能盲目复制这行就声称实现了论文随机顺序；应以原文算法为准，并记录与作者代码的差异。

在本实验可以将同 prompt 的 completions 作为多个局部目标，但这是 **per-completion PCGrad adaptation**，不是原论文 multi-task benchmark 的严格复现。不能用 clean/noisy 标签定义任务。需匹配 LoRA 参数空间、loss normalization、梯度符号、随机顺序 seed、KL 加入位置与实际 optimizer。

PCGrad 修改向量，不是只产生 mask；因此不能仅塞进当前 sample selector。建议独立 trainer，在 TPU 上按树计算 dot/norm/投影，不依赖大梯度 CPU 导出或哈希压缩。测试：两向量解析例、无冲突时退化原聚合、零梯度、sum/mean尺度、随机顺序、KL与 HBM。

优先级：适合补充“为何筛样本而非修改冲突梯度”，但排在 R1 明确点名的 norm median / DS 和 R2 独立效用验证之后。原论文已有多任务 RL 实验，不能回应为“不适合 RL”。

来源：[PCGrad 原文](https://arxiv.org/html/2001.06782v4#S2.SS3)。

### 4.2 CAGrad

原方法在平均梯度 `g0` 附近寻找兼顾最差任务局部改善的更新方向。一个原始形式是

\[
\max_d\min_i\langle g_i,d\rangle,
\qquad \|d-g_0\|\le c\|g_0\|.
\]

通过小规模 simplex 权重优化求方向，不是简单删除负内积样本。c 是实质方法超参数，选择和缩放必须披露；其理论也有条件，不能移植到非平稳 GRPO 后直接宣称原保证成立。

已核实作者 [mtrl/mtrl_files/cagrad.py](https://github.com/Cranial-XIX/CAGrad/blob/main/mtrl/mtrl_files/cagrad.py) 同时有迭代/精确/fast 路径；精确版本将 Gram 矩阵放到 CPU 后用 SciPy `minimize`。toy、RL 代码的返回缩放也不完全同一种写法，应指定对照哪一个实现，不能只抄一个公式忽略 rescaling。

若适配，4条 completion 的 Gram 只有4×4，但逐样本梯度、solver、TPU 同步、共享 KL、归一化和 optimizer 接口仍需验证。可借鉴官方小矩阵求解，不应为避免 CPU 而仓促换一个未验证近似求解器。也不需把完整梯度树传 CPU。

适用性：有机制比较价值，且原文有多任务 RL；当前单任务多 completion 的适配比“引用比较”更重。建议先完成充分 related-work 对照，再按剩余时间决定是否做完整 seeds，不能先承诺完整训练矩阵。

来源：[CAGrad 原文](https://arxiv.org/html/2110.14048v1#S3.SS1)、[作者仓库](https://github.com/Cranial-XIX/CAGrad)。

## 5. 固定 checkpoint 的独立 clean utility 检查

### 5.1 它回答的正是 R2 的关键问题

“与 batch 内其他梯度一致”并不等价于“改善独立 clean objective”。额外 self/cross corr 或漂亮的分布图不能替代这项检查。先做小规模诊断，不要求重训完整 seeds。

### 5.2 最小可执行设计（待批准/实现）

1. 先取一个可恢复的中期 checkpoint，优先同一个 Vanilla GRPO 轨迹，避免方法各自 checkpoint 差异干扰比较；有资源再扩 early/mid/late 和多个 seeds。避开 LR 恰好为0的初始化或结束点。
2. 从 training pool 固定采样8个 batches，每 batch4 prompts，每题4回答，共32 prompts/128 completions。对每批只 rollout 一次，冻结 token IDs、masks、old/ref logprobs、dense clean/observed noisy rewards、advantages、prompt/group/occurrence IDs。后续方法共享这些输入，不重新生成。
3. 独立 clean held-out objective 使用训练未用过的64–128个 GSM8K prompts，不使用1319条 benchmark test。当前 `my_example/main.py` 按 batch 切分，3072/4=768 batches，训练691 batches=2764 prompts，剩77 batches=308 prompts，可从这个 validation pool 固定选择；需先核实实际 run 配置和 prompt 去重。不宣称这个已经周期评估过的 validation 是从未使用过的数据。
4. 主 objective 建议 clean reference-answer 的 teacher-forced mean token NLL，只在 answer tokens 上计分。它是独立正确答案效用代理，不是 greedy accuracy、数学推理正确率或期望 reward；模板、答案格式、截断和多种正确解的局限需说明。可另做小规模 common-random-number rollout reward 检查，但更昂贵，不作为最小前置条件。
5. 从完全相同参数、AdamW moments/count、schedule step 出发，分别临时执行 Vanilla、DTV、DTV-Loo 的一次真实更新，含原 KL、归一化、weight decay/clipping。每次结束后恢复，下一候选 batch 也从相同 checkpoint 出发；不串行积累成新的训练轨迹。无 test 调参。
6. 定义

\[
U_b(M)=L_H(\theta)-L_H(\theta+\Delta_{b,M}).
\]

U>0 表示该次更新降低 clean held-out proxy loss。报告每批 U，及配对差 `U(DTV)-U(Vanilla)`、`U(LOO)-U(Vanilla)`，不要只展示有利 batch。8 batches 是可行性检查，不足以冒充跨5个训练 seeds的稳健结论。

### 5.3 单元层面的低成本补充

同一 checkpoint 求 clean loss 梯度 `h=∇L_H`，对每 unit 的 policy-score gradient 计算 `u_i^PG=<h,g_i^score>`。在纯 SGD 小步近似中，`η u_i^PG` 对应 clean loss 改善；这是 first-order policy-gradient alignment proxy，不是实际 AdamW utility。

若要和含 KL 的实际 unit loss 对照，再单独定义 `u_i^total=<h,∇ℓ_i^total>`，不要混用两个名称。报告 DTV/LOO score 与 u 的 Spearman、kept/dropped u、clean/noisy × advantage符号 × clean correctness 的分层计数；零梯度、近零与 ties 单独统计。unit 内/同 prompt 内存在相关性，不把128条 completion 当128次独立训练。

三个临时更新的 U 才用于真实 optimizer 检验，unit proxy 提供较便宜解释。mask mean 导致更新尺度不同，建议有预算时加 matched-count random mask 或 equal-update-norm 辅助诊断；主结果仍保留算法真实更新，不为获得正结果调尺度。

### 5.4 checkpoint 与 HBM 风险管理

- 只有导出 `model/`、没有 optimizer：可以做明确标注的 SGD-direction/fresh-optimizer proxy，不能声称复现了历史 AdamW 更新；优先恢复归档的完整 checkpoint。
- 一次只驻留一个 actor/当前优化器，不并排复制3个模型或同时缓存128份完整梯度。
- 固定候选按原4×4微批顺序处理；unit dot 按梯度树归约只传标量，必要时分块并做等价测试。
- held-out NLL 与 gradient 也分微批累计；累积梯度数值/归一化固定。先真实输入 smoke + 最大长度 HBM，报告实测峰值，不能只用静态估计保证。
- 诊断保存 JSON/CSV、输入 IDs/必要 tokens 和配置；不需常态保存几十 GB 原始梯度。
- 若结果不支持“内部一致性代表有用性”，据实收窄结论与 Assumption 的经验适用范围，不能只换图。

## 6. Figure 2 / Table 3：已找到什么，还缺什么？

本轮只读核查外部 DPO 仓库，路径根目录：`/Users/jason/Documents/1. Education/9. Codex/DTV_DPO`。未修改该仓库。当前代码不自动等于投稿时归档配置，需要实际 manifest 交叉核实。

### 6.1 已确认：周期 validation 不是正式 test 的同一个集合

`tunix/examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft_lora.yaml:37–38`：

- train：`train_prefs`, partition=dpo, subset=train。
- periodic validation：`train_prefs`, partition=dpo, subset=eval，eval_fraction=.1、seed42，按 prompt hash 分区。
- `training_config` 每10步eval，max_steps115。

`tunix/examples/dpo/eval_qwen2p5_no_pref_main_table_runs.py:_build_test_module_spec`：默认 clean Test Acc 来自 `test_prefs`, seed42；但允许环境变量 `DPO_CLEAN_TEST_DATA_MODULE` 覆盖，归档运行必须核查。

两者可使用相同的 DPO reward-accuracy 定义，但同一种评估操作/同分布不等于同一批 test 数据。不能把“每隔几步评估一次”当成数据集相同的证据。

`tunix/tunix/sft/dpo/dpo_trainer.py:239–245` 的常规 DPO accuracy 为比较 actor 相对 SFT reference 的隐式 chosen/rejected reward：

\[
r_\theta(x,y)=\beta[\log\pi_\theta(y|x)-\log\pi_{ref}(y|x)],
\quad Acc=\mathbb E[\mathbf1(r_{chosen}>r_{rejected})].
\]

这是偏好排序准确率，不是生成回答的 instruction-following accuracy。

### 6.2 已重算本地绘图 CSV 的 clean 末端

文件：`dpo_convergence_raw_scalars.csv`，tag=`dpo/eval/rewards/accuracy`，seeds=0/1/2/3/4，最后可见 eval step=110。

| 方法 | step110 五seed均值 | 对同seed Vanilla 的均值差 |
|---|---:|---:|
| Vanilla DPO | 0.6164801 | 0 |
| DTV | 0.6195896 | +0.0031094 |
| DTV-Loo | 0.6133706 | -0.0031095 |

所以“clean 图末端接近0”的视觉印象与该 CSV 相容；不能把它说成+0.0292，也不能直接判定它与 test 结果矛盾。

`plot_dpo_paper_figs.py:make_delta_to_vanilla` 按 setting/同seed，用 `np.interp` 把 Vanilla 对齐到方法 step 后相减，再跨 seed 求 mean/std。没有共同 step 时会插值，超范围时 numpy 会取边界值，审计时需单独标出，不能默认为同一时刻真值。当前 clean110点恰有共同step，不涉及插值误差。

训练配置115步，而原始曲线最后eval110步，说明至少“最后周期eval”和“配置训练终点”不一定相同；**不能只凭配置断定最终表确实评估115步**。必须看实际run和checkpoint。图的 smoothing 也需看当时命令，不能把默认值当实际值。

### 6.3 尚未完成的逐run对账

需要 Figure2/Table3 实际引用的图文件/表格、生成命令、manifest/run IDs 与 table raw eval JSON，逐项列出：metric、split及样本数、reference model、checkpoint实际step、seed集合、baseline减法、平滑、batch aggregation、是否截断/只评部分数据、数据或模型路径覆盖。

若表的+0.0292是 clean test gain，而图是 validation gain，二者不同可以成立，但应清楚标注并报告其差异；若发现用了错run、错metric、错checkpoint或统计错误，要明确更正，不用“validation和test差不多”掩盖。

## 7. 三个额外 benchmark 是不是“混合数据集”？

当前 main-table evaluator 默认额外列为 `livebench_if_score`、`rewardbench2_precise_if`、`ifbench_prompt_strict`。这需要最终 Table8 原文件确认，但已定位到对应数据与评分实现。

### 7.0 澄清：不是在GSM8K上额外算三个指标

2026-11-07补充：用户此前未参与这部分评测设置，容易混淆“模型的训练数据”“独立评测数据”和“评测指标”。当前已核对的是外部DPO仓库的Qwen2.5/UltraFeedback评测流程，不是Gemma/GSM8K GRPO评测器。因此这三列不能自动被解释为GSM8K结果，也不能把DPO checkpoint与GSM8K checkpoint混称同一个模型。

这里的流程为：在UltraFeedback上完成DPO训练 → 固定run的导出模型 → 分别加载LiveBench-IF、RewardBench2、IFBench各自的问题/回答资产 → 计算各自分数。三个名字是benchmark/子集与其指标的组合，而不是三种套在同一批GSM8K题上的accuracy计算方法。

- LiveBench-IF：`_load_livebench_rows`显式加载`livebench/instruction_following`的test集，取`turns[0]`为新prompt。`_evaluate_livebench_if`用`run['exported_model_path']`生成新回答（此路径temperature=0），再调用约束检查器。不是复用GSM8K的生成答案。
- IFBench：加载`allenai/IFBench_test`独立测试资产，用同一个run导出模型生成该集的新回答，再计算prompt-strict（每题所有要求都满足才成功）。HF split字段名为train不改变资产IFBench_test的身份，也不说明本实验用它训练。
- RB2 Precise IF：加载`allenai/reward-bench-2`；数据已包含prompt及多个chosen/rejected回答，不需该模型重新生成。代码对给定回答计算DPO隐式reward，以排名最高的回答是否属于正确集合来计分；最终只取Precise IF子集。其隐式reward依赖该run actor和原SFT reference，因此不只是对actor生成文本检查格式。

RB2的打分可写作`r(x,y)=β[logπ_actor(y|x)−logπ_SFT(y|x)]`。这是评估阶段的回答评分，不是GSM8K训练中“数值答案是否正确”的dense reward。

最终checkpoint的理解：这些新评测函数不再更新模型，均加载`exported_model_path`；它通常对应run导出的训练后模型，但不能只从变量名证明它一定是最后一步而非手工指定的其他checkpoint。还需核对实际run路径、导出step与归档结果。主表入口允许复用legacy/cache结果，所以也不能把“脚本执行过”当成每一列当次都重新评测的证据。

“综合/混合”应分清三件事：多种题材或约束、多benchmark分数汇总、clean/noisy训练数据混合。三者互不等价。这三个额外评测都不是把GSM8K clean和mismatch题混起来计算三个分数。它们检测的是UltraFeedback训练之后的外部能力迁移：两项是新回答的约束遵循，一项是已有回答的排序识别。

例子（仅作解释，不代表数据原题）：外部prompt要求“两段、每段三句、包含指定词”。模型回答内容合理但少满足一条约束，IFBench严格计分仍可能判失败；模型在UltraFeedback偏好对上的ranking更好不保证这里更好。RB2则是给它若干已写好的回答，看它是否把满足要求的回答排第一。生成能力与排序能力有关，但不是同一指标。

因此，当前能确认的是“评测数据/任务/评分方式不同”；不能据此已确定下降是正常domain shift，更不能未经归档映射就说GSM8K训练使这三个benchmark下降。正式回应前需要同作者确认Table8的模型、训练设置、run与metric列一一对应。

| 当前列 | 实际被评内容 | 是否能简单称“全是混合任务 benchmark” |
|---|---|---|
| LiveBench-IF | `livebench/instruction_following`，只取 instruction-following category；代码用 strict prompt/约束满足率组合的逐题均分 | 不能借用整个 LiveBench 的 math/coding等六大类来解释这一子集 |
| RB2 Precise IF | RewardBench2 的 `Precise IF` subset；对给定回答用 DPO implicit reward 排序，top候选是否在correct集合 | 完整 RB2 的确有6个子类，但本列不是它们的混合总分，也不是模型重新生成答案的准确率 |
| IFBench prompt-strict | `allenai/IFBench_test`（HF标记的split名为train，但资产本身是IFBench测试集）；新颖可验证约束、每题全部约束满足才算strict成功 | 有多种约束/题材，但仍专测 instruction-following generalization，不能等同无关混合任务 |

本地证据：

- `tunix/examples/dpo/eval_qwen2p5_no_pref_main_table_runs.py:107`, `:294`, `:377`, `:417`。
- `score_instruction_following_dataset.py:65–106`：LiveBench当前实现score为每题 `0.5×(all_constraints_success + fraction_constraints_success)` 再平均；不是数学QA exact match。
- 主表实际加载`eval_qwen2p5_rewardbench_v2.py:_summarize_rewardbench_v2_rows`：按subset汇总，Precise IF来自自己的分类准确率，不是macro总分；另一个no_pref版本中也有相同汇总逻辑。
- `eval_qwen2p5_clean_benchmarks.py:_prepare_ifbench_assets` 加载 IFBench_test。

外部一手来源：

- [LiveBench IF dataset card](https://huggingface.co/datasets/livebench/instruction_following)
- [RewardBench2 dataset card](https://huggingface.co/datasets/allenai/reward-bench-2)
- [IFBench 原论文](https://arxiv.org/abs/2507.02833)：58种新的可验证 out-of-domain constraints。

### 7.1 可以合理解释什么，不能声称什么

可以说：本方法的内生 reference 反映当前训练任务/数据的局部优化一致性，并不自动优化独立 benchmark 或保证跨分布约束满足率。UltraFeedback preference ranking、给定回答的 Precise IF ranking、自由生成的 constraint following 不是同一 objective。in-domain收益与跨分布指标下降可以并存。

**这只是解释框架，不是已经证实的下降原因。** 不能说“因为是混合数据集，所以下降正常/与我们无关”。UltraFeedback 本身也是广泛指令偏好数据，与指令遵循并非完全无关；这些 benchmark 正是在测试其相关泛化能力。

对评审引用 `0.4938 → 0.4238`，是7.00个百分点下降，而不是0.07个百分点。若表数值核实无误，应承认这个真实代价，删/收窄“普遍提升下游能力”之类表述。优先按同seed/同sample复核，并查看per-example outputs、约束类型、生成长度/截断、prompt模板和reference设置；未分析前不把长度/格式/负迁移某一种原因说成事实。

可用英文回应骨架（完成对账后填入实测结果）：

> We agree that the gains are not uniform across evaluation objectives. Our endogenous reference is computed from the current training units and does not guarantee improvement on independent instruction-following benchmarks. We will distinguish in-domain preference accuracy from out-of-distribution generation and response-ranking metrics, and explicitly report the observed trade-offs rather than claim universal downstream improvement.

“任务局部性”是scope限定，不是证明方法有效性的替代品；仍需第5节的独立clean objective检查。

## 8. Experimental settings 的修改承诺

承认当前正文呈现不够明确，补可复现协议表，而不是笼统“会放入附录”。至少包括：

- 每任务的模型全名/大小/初始化checkpoint、RL或offline preference objective；LoRA仅是参数化，不是SFT目标。
- train/validation/test的来源、划分、样本数、是否 clean、是否参与选择或调参。
- 4 prompts/update、4 answers/prompt、691updates、各 selector 的额外 rollouts/参考数据/保留比例；不要混淆候选rollouts和真正训练rollouts。
- 原 dense reward 与污染后reward的算法、按prompt hash选污染组的seed与比例；chosen/noise-selected不等于reward实际上改变。
- LR峰值1e-6、warmup比例0.1与cosine衰减、KL0.08、clip、AdamW、LoRA rank、真实序列长度/截断、评测解码。
- 本地 launcher 的真实默认长度继承当前config；不能把HBM压力测试的768长回答自动写成所有正式run都使用768。以归档实际配置为准。
- seeds：GRPO目前配对集合0/5/13/21/42；本地DPO曲线是0/1/2/3/4，不能混写。
- DTV打分是否policy-only、实际更新是否含KL；self/cross/LOO分母、scope、min-keep/cap/零分tie处理。
- 硬件、single-worker拓扑、wall time（含selection/refill/eval）、真实峰值内存；模型参数大不等于设备架构优劣的证据。
- 原论文方法到JAX/TPU的适配差异与代码来源/commit，包括LearnAlign/GradAlign各自预算，不冒充所有原文超参数严格复现。

回应模板：感谢指出 → 明确承认哪一项表述缺失 → 给出当前确切值/定义 → 指出修订正文/表/附录位置 → 解释是否改变已报告实验。没有实际改稿时说“will clarify”，不要说“have revised”。

## 9. 下一步顺序与验收

1. 保持运行中的 PODS 不动；先确认本文件方案与关键选择。
2. 补齐可读run配置/sample记录/checkpoint目录，以及投稿Figure2/Table3/Table8原始结果映射；完成图表对账，不等待新训练。
3. 独立实现 GRPO+DS（用户已表达想优先利用空闲worker），通过CPU单测、当前worker TPU smoke/HBM、输出结构检查，再先跑seed5。
4. norm top50% baseline 先确认 group / batch median 范围，再新增独立实现；与PODS互补，不是重复同一个实验。
5. 同步做一个固定checkpoint的clean utility可行性诊断，再决定是否扩3阶段/多seed；恢复不到optimizer必须改变结果名称和解释。
6. 完成分层过滤率、rare-correct retention、resource/protocol信息；这些可直接回应效用和少数信号风险。
7. PCGrad先完成最小toy与接口设计再决定训练，CAGrad先写准确机制比较；不把rebuttal变成无止境加baseline。

任何实验结果不支持预期都应保留并解释边界。上述优先级旨在回答明确质疑，不承诺评审提分或论文录取。

## 本轮验证记录

- 只读检查GRPO梯度、污染、数据切分、launcher及外部DPO绘图/评测/数据配置。
- 使用Python标准库重新汇总DPO CSV clean step110五seed数值；未新增训练数值。
- 阅读DAPO/PCGrad/CAGrad原文与上述作者/维护者核心代码；核实三个benchmark一手资料。
- 文档检查：`git diff --check`；未运行TPU、修改现有训练代码或停止任何任务。

## 后续进展：GRPO材料清单与核心假设检验（2026-11-07）

已新增`GRPO_REBUTTAL_PREPARATION.md`，单独列出norm对照与固定checkpoint诊断的所需文件、是否需要重训/新增代码、预期输出、可支持及不可支持的结论。增加L-smooth独立clean objective的方向/步长条件及reference误差界，用于澄清为什么内生正内积不能无条件保证clean utility；没有投稿理论原文，不冒充已完成正式定理核查。

新增baseline按DS/PCGrad/CAGrad分别梳理，norm作为额外直接对照。已开发DS独立路径；未实现其余三项。当前变更只有`train.py`的opt-in分支与新增文件，不修改通用learner、robust trainer、PODS/DTV实现、RLTrainingConfig或原CLI。

DS测试进展（2026-11-07）：用户回传显示真实mismatch20 smoke和pre/post结果结构通过；最终HBM夹具因重复简单题的observed reward全组内常数而达到cap20，并非HBM OOM。已仅修测试夹具，从首阶段起明确使用受控reward覆盖有效/无效组及两轮refill，保持生产算法与上限不变；23项本地测试通过，待服务器只重试HBM。真实生成+受控reward的内存检查不代替生产reward链路的端到端验证。详见准备文档修复记录。
