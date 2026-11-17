# UrbanGene V2 — Hierarchical Compositional Urban Gene Learning
## 研究方案与 Codex 增量实施需求（PRD）

**版本**：2.0（研究设计稿，待实验验证）  
**日期**：2026-11-09  
**目标读者**：Codex / 研究工程开发者 / 项目研究人员  
**优先级**：先验证实例发现与共享 Gene，再做组合生成；不一次性集成全部高级模型。

> **一句话任务**：在**不使用人工实例类别、边框或分割 Mask 作为训练监督**的条件下，从街景自动推断相对独立的潜在单元，让多个单元由一个受限的、跨场景共享的 Gene 字典表达，并通过可控的空间组合重新构建与生成街景。最终以 **Minimality、Reusability、Compositionality** 三个维度验证，而非仅以 PSNR 判断是否成功。

---

## 0. 决策与范围（Codex 必须首先阅读）

### 0.1 本次架构决定

**核心主线（必须实现）**：

```text
输入街景 I
  ↓
冻结的预训练空间视觉编码器（DINOv2 / DINOv3，可配置）
  ↓
Adaptive Spatial Slot Discovery（无标注实例发现）
  ↓                 输出初始 soft masks、slot vectors、geometry
Shared Sparse/Discrete Gene Dictionary（每实例调用少数共享 Gene）
  ↓                 输出有限、可序列化的实例程序
Shared Spatial Compositional Decoder（共享局部生成器 + 受限空间放置 + 融合）
  ↓
重建 AE latent → 冻结的图像 AE Decoder → 重建图像
  ↓
重组实例程序 A + B → 新场景 C（生成时不读取 C）
```

**非主线但必须可比较**：
- **V1.1**：现有 Fixed Average、Residual CNN 和 GRCT 实验保持可追溯，作为历史对照；禁止覆盖已有结果。
- **V1.2 固定网格 VQ/RVQ**：作为**表征容量与优化诊断基线**，不得称为自动发现的景观单元。
- **SAE**：只作为可插拔特征分解**消融实验**；先实现不用 SAE 的主模型，再决定是否值得加入。
- **GRCT**：作为复杂关系交互的升级/消融，不是新主线的先决依赖。

### 0.2 硬性原则

1. **不预先定义 Gene 的语义**：Gene 可以是抽象向量，不要求对应“树/建筑/道路”。
2. **实例发现与 Gene 发现分开评价**：实例是可定位的图像成分，Gene 是共享的可复用表示；一实例可由多个 Gene 构成，一个 Gene 可复用到多个实例。
3. **训练的实例发现不使用语义类别/实例 Mask/框标签**；合成 scene_spec 和人工标注只用于预先制定的划分、诊断、测试与评价。任何使用标注训练的实验必须另列为“监督上限”，不可冒称无标注方法。
4. **生成通道不可绕过瓶颈**：最终生成器只能接收可序列化、定量计费的 Gene/实例程序与全局共享参数，不能读取目标图、DINO feature map、AE 原始 latent、连续 Slot 向量或未经压缩的像素级 Mask。训练用的 Slot soft mask 不等于可以免费传给生成器的程序。
5. **模型训练阶段分离**：先测试 DINO 特征可用性，再独立验证实例发现，再验证共享 Gene，最后验证空间重建与真正重组。
6. **保持科学诚实**：重建未见组合 C（给定完整 C）与 A+B→C（不提供 C）是不同任务；Mask 像真实对象、Gene 语义可解释和严格最小基因数都不是架构自动保证的性质。
7. **先做消融再扩大规模**：小规模验证不成立则停止新增参数/复杂模块，按诊断门槛回溯。不得默默运行大规模实验矩阵。

### 0.3 “高效”的定义

本 PRD 优先优化 **科学识别效率**（能否区分失败来源、验证科学假设），其次考虑计算效率。特别要求记录：
- 场景质量与程序比特数的 Rate–Distortion；
- Gene 字典与生成器共享参数成本；
- 实例复用、空间可控性、未见组合与 A+B→C 成功率；
- 训练过程稳定性、跨随机种子的方差。

---

## 1. 科学问题、研究假设及可证伪标准

核心科学问题：

> **Can a compact, shared vocabulary of learned latent genes, composed into spatially instantiated units, represent and generate a combinatorially rich set of urban scenes?**

**H1 — Discovery（实例发现）**：不使用实例监督，模型可以将场景分解为相对完整、稳定、可定位的潜在单元，并对重复实例保持区分。  
**H2 — Shared Genes（共享与紧凑性）**：在相近质量和场景编码预算下，多个单元共享的少量离散 Gene 能比非共享/自由连续表示更节省描述成本或提供更稳定复用。  
**H3 — Composition（组合泛化）**：保持有限 Gene 字典和共享解码器，重组来自不同场景的实例程序，可以生成训练中未出现的合理组合，且非目标区域相对稳定。  
**H4 — Architecture（必要性）**：加入额外 SAE 或 GRCT 的收益应通过严谨消融证明；若简单模型足够，不保留无增益的模块。

**允许的负结果**：模型仅找到 Patch、把多棵树合并、码本塌缩、A+B→C 失败都必须如实报告；不能仅凭重建改善宣称发现了“视觉基因”。

---

## 2. 模型概念与数学定义

### 2.1 三个表示层级

- **Gene type**：共享字典中的一个基向量 `g_k`；跨图片共享，无固定人工标签。
- **Gene instance / latent unit**：模型在某张图中发现的可定位单元，其形态由少量 Gene 的组合表示。
- **Scene program**：多个实例的受限编码及关系/布局；将其执行后生成场景。

定义共享 Gene 词典：

\[
\mathcal G=\{g_k\}_{k=1}^{K_g},\quad g_k\in\mathbb R^{d_g}.
\]

对第 `i` 个候选单元，Slot Discovery 输出**仅供推断与训练诊断使用**的临时变量：

\[
(s_i,m_i,p_i,e_i)=D_{\rm slot}(F),
\]

其中 `s_i` 为连续 Slot，`m_i` 为软区域，`p_i` 为几何估计，`e_i` 为存在门控。它们不意味着已学出真实物理对象。

**正式生成程序**使用受限编码（不能原样保存 `s_i,m_i`）：

\[
P_x=\left\{\left(e_i,\{k_{it}\}_{t=1}^{r},q_i^{\rm geom},q_i^{\rm attr}\right)\right\}_{i=1}^{M} \cup P_x^{\rm bg},
\]

其中：
- `M` 为最大候选实例数，不是实际实例数量；
- `k_it` 是选出的 Gene ID，可重复引用；`r` 是每实例最大 Gene 调用数；
- `q_geom` 为**量化**中心、尺度（必要时角度/层次）；
- `q_attr` 是可选、明确计费的小规模离散属性码；V2 首先不启用；
- `P_bg` 是受限的背景编码（或固定、共享的背景原型），不得携带未计费的全分辨率背景。

实例表示（最简版本）：

\[
z_i=\operatorname{Combine}_\psi(g_{k_{i1}},\ldots,g_{k_{ir}}),
\]

其中 `Combine` 是**跨实例共享**的定长算子（先尝试求和/小 MLP），不是每张图单独优化的网络或向量。若引入可变系数，必须量化、计费并消融。

场景生成：

\[
(\hat z_{\rm AE},\{\hat m_i\})=R_\theta(\mathcal G,P_x),\qquad
\hat I=D_{\rm AE}(\hat z_{\rm AE}).
\]

`R_theta` 用 Gene 编码在局部坐标中预测局部特征/形状，按量化空间参数放置、遮挡与融合。**生成阶段不得读取 `D_slot` 的预测 Mask，除非该 Mask 完整编码入 `P_x` 并计费。**

### 2.2 信息预算与 Minimality

**不再承诺 V2 固定为 272 bits/image。** 旧值仅属于 V1.1 的特定程序协议；新的模型含实例数量、多个 Gene ID、几何以及可能的背景编码，必须重新核算。

优先使用完整实测序列化码长：

\[
B_x=\operatorname{Bits}(\operatorname{Serialize}(P_x)).
\]

同时报告：
- `scene_bits_avg / scene_bits_p95`：每图数据成本，包含禁用槽位与数据结构开销（按约定编码格式）；
- `dictionary_bits`：共享 Gene 词典权重（按声明精度）；
- `renderer_bits`：共享生成器权重；
- `backbone_bits`：预训练 DINO 与 AE 的权重，单独列示（并说明哪些对照共同使用）；
- `amortized_total_bits(N)`：将共享模型成本摊销到 `N` 张图后的成本。报告明确采用何种摊销范围与权重精度。

实际优化目标首先为重建质量，使用**硬性容量控制**（`K_g`、`r`、`M`、量化几何、受限解码器），不一开始设计复杂 MDL Loss。不要使用一个没量化、可任意精度存储的连续实例变量隐藏场景信息。

---

## 3. 数据与划分

### 3.1 复用合成城市数据（第一阶段）

尽量复用现有 synthetic city 数据、manifest 和 scene_spec；**先检查现有路径与格式，不假设 PRD 里示意路径在当前项目一定存在**。

推荐继续使用已建立的：`train / val_seen / test_seen / test_unseen_combo / test_higher_complexity`。原有 3×3×3 因子组合中 21 个训练、6 个保留作测试的规则应维持一致，若重新生成应保存 manifest 和 seed，并证明泄漏检查。

**训练时**可用 RGB 原图及其由冻结网络提取的自监督特征；**训练实例发现时不可使用** scene_spec 的对象数量、位置、Mask、类别作为监督。

**评价时**可使用 scene_spec 构造实例 GT、区域、因素组合、反事实场景及属性标签。务必在日志中记录标注从未进入训练路径。

### 3.2 数据适配的关键难题

- 旧合成数据可能过于规则/简单，也可能与预训练 DINO 的自然图像分布差异很大；先做 DINO 特征可用性诊断。
- 256×256 的 patch 特征可能不足以区分十棵密集树；实现更高输入分辨率或多尺度特征作为可配置方案（不要预设一定有效）。
- 设置至少三种可见实例密度场景：单独实体、相互靠近但不接触、相互遮挡/相接；仅评价可观察的实例，不能要求单视角恢复完全遮挡物体。
- 道路/天空等连续区域可以进入受限背景分支，不应被强制拆成“十个物体”。

### 3.3 真实街景迁移（后续）

在合成实验通过后引入真实街景的无标注训练子集和**小规模独立人工评价集**，检查对象发现的域外稳定性。未提供真实数据路径时，不得伪造已准备或已运行数据。

---

## 4. 分阶段技术路线与验收关口

### M0 — 安全基线、环境与失败诊断归档

**任务**
1. 核验现有代码根目录、训练配置、DINO/AE 实现、CLI、GPU 与 NAS 路径；创建隔离 `v2` 工作区/分支。
2. 只读汇总 V1.1 报告的 AE≈30.7 dB、Gene+GRCT≈13.7–14.6 dB、程序塌缩及覆盖率下降等**此前观察**；直接读取日志前不可认定已独立复核。
3. 保留 V1.1 / V1.2 的所有检查点、队列状态、测试协议；不要更改正在运行的旧队列源码或共享配置。
4. 检查并复用 V1.2 fixed-grid VQ/RVQ 诊断结果；若尚未实现或未运行，先如实标注。

**交付**：`docs/V2_DECISIONS.md`、`outputs/v2/baseline_audit.json`、新配置与目录隔离策略。  
**工程 Gate**：所有旧结果仍可定位；新实验能独立运行且不能写入旧输出目录。

### M1 — DINO 特征可用性与无标注 Spatial Slot Discovery

**模型**
- `FrozenVisualBackbone`：DINOv2/DINOv3 通过同一接口输出 `[B,C,Hf,Wf]` 特征及坐标；默认使用可用的已验证预训练权重，不在此阶段重训 DINO；记录权重版本。
- `AdaptiveSpatialSlotEncoder`：具有竞争归属的 Slot Attention；空间位置/相对尺度偏置；可变存在门控；足够高的 slot 上限（初始建议 32，作为配置而非事实最优）；背景可单独建模。
- `FeatureReconstructionDecoder`：共享参数，从 Slot 与可学习软 Mask 重建**冻结 DINO 空间特征**。这是**实例发现内部训练**，不受最终序列化规则约束；但不能把这些 Mask 直接给最终场景生成器。

**训练**
- 训练输入仅图像及冻结 DINO 特征，不输入实例标签/数量。
- 主损失 `L_discovery = mean ||F - F_hat||²`；必要时采用标准 Slot Attention 的数值稳定策略、mask 竞争、梯度裁剪、合理初始化。正则化仅在诊断有依据时加入，单独记录消融。
- 在此阶段**不要引入**共享 Gene 字典、SAE、GRCT，也不要训练完整 RGB 生成模型。

**输出接口**
- `slot_features`: `[B,M,d_s]`
- `soft_masks`: `[B,M,Hf,Wf]`（近似竞争归一化）
- `exist_logits`: `[B,M]`
- `estimated_geometry`: `[B,M,d_geom]`
- `reconstructed_features`: `[B,C,Hf,Wf]`
- `background_state`：仅用于实例发现诊断；若进入生成，须受限重新编码。

**评价（标签只用于 eval）**
- matched mask IoU / ARI 或 FG-ARI（清楚说明背景口径）；
- 计数误差、split/merge 比例、有效 Slot 数量；
- 实例在平移/尺度变换下的匹配稳定性（先进行跨增强 Slot 匹配，不依赖槽位编号）；
- 特征重建误差；含十棵树的场景要单独做可见实例分离评估。

**科学 Gate**：至少在预定义验证组上产生显著优于“单 Slot/规则网格/随机分组”的实例分组证据，且不只是低重建误差。未达标则优先检查特征分辨率、背景占比、slot 上限、相邻实体分离，再决定是否使用多尺度和空间归纳偏置。**不要因为输出了漂亮 Mask 就直接通过。**

### M2 — 每实例的 Shared Sparse/Discrete Gene Dictionary

**核心**：将上一步的每个实例连续 `slot_features` 压缩为共享字典中**少数几个离散 Gene ID**；一实例可调用多个 Gene，不要求一个 Gene 对应一个对象类别。

**第一版主方案**
- `SharedGeneBank(K_g,d_g)`：`K_g` 先试 `{64,128,256}`；`d_g` 建议 `{128,256}` 中先固定一个；
- `InstanceGeneEncoder`：从 Slot 特征输出最多 `r` 个离散 Gene ID；优先试 `r∈{1,2,4}`；
- `SharedGeneCombiner`：使用简单可交换求和或共享小型 MLP 生成实例码；若重复 ID 无意义，要记录并决定是否允许重复；
- 首选可训练的 VQ/FSQ 式离散化方案，明确其索引与**跨图片共享 Gene 参数**的对应关系。可使用 straight-through/EMA/重置死码本等标准做法，记录 codebook 使用情况。FSQ 与普通 learnable codebook 的性质不同，不可只改名字等同。
- 不允许未量化的连续形态/外观旁路、自由每图残差或隐式 Mask 直通。

**训练**
- 先冻结 M1 的分组模块，检验实例码能否重构 Slot 表示/对象局部特征；可先只训练 Gene encoder/combiner/decoder。
- 然后在训练稳定基础上做有控制的联合微调；**分开报告冻结和联合两种结果**，防止端到端训练改写实例含义。
- 第一版以对象特征或 AE-latent 重建为主目标；若用 VQ commitment/EMA 等辅助机制，需注明是为了量化优化而非人为定义语义，并记录它们的效果。

**关键指标**
- 字典活跃 Gene 数量、每基因样本覆盖、perplexity、dead codes；
- 不同场景的同一 Gene 重复调用情况、相似实例的 Gene 组合稳定性；
- `K_g × r` 在同等失真下的 Pareto 曲线；
- 无字典的自由连续 Slot、每实例不共享原型、固定网格 RVQ 的对照（公平处理码率/共享成本）。

**科学 Gate**：共享离散码在不同图像间具备真实复用，且其表示损失和编码率可测；若字典极度塌缩、实例全用相同序列或仅靠背景路径重建，必须停止扩展到组合器。

### M3 — Shared Spatial Compositional Decoder（先简单后复杂）

**目标**：仅凭**完整序列化后的实例程序**，重建 AE latent / 图像。

**第一版实现**
1. `GeneInstanceDecoder`：从少量 Gene ID 组合码，在**实例本地标准坐标系**输出局部 AE-latent 模板与预测透明度/软形状；必须共享同一组权重。
2. `QuantizedGeometry`：从实例发现得到的中心、尺度转换为离散码；需要明确坐标网格、尺度档位、禁用槽位和背景码率。
3. `SpatialPlacer`：把局部模板按离散几何放到固定 AE latent 画布。
4. `SharedComposer`：首先固定 alpha/遮挡或规范化合成；可以引入共享的受限局部卷积作简单改进，但**不应让网络直接接入原始 DINO 特征或 target latent**。
5. `FrozenAEImageDecoder`：使用 V1.1 的已验证 AE 进行 RGB 重建；若无法兼容，增加显式训练的 latent adapter 并计入共享参数，不能偷偷重训 AE 且声称与旧版可直接比较。
6. `BackgroundPath`：优先固定背景原型或少数离散背景 tokens；背景若需要图像特定细节，必须通过完整序列化计费，并在消融中检查是否掩盖了前景实例贡献。

**训练与验证**
- 主目标先使用 `L_scene = mean |AE(I) - R_theta(P_I)|`，不预先加入复杂组合/语义/解耦 Loss。
- 对像素级质量做旁路评估 RGB L1、PSNR、SSIM；若 latent loss 与 RGB 效果失配，再另列 `RGB L1` 单损失训练对照。
- `serialize(P_I) → deserialize → R_theta → AE decoder` 的输出必须与直接执行一致（容差由实现数值精度定义）。
- 测试：全部实例禁用、同一 Gene 多次调用、交换两个实例、移动一个实例、尺度变化、背景消融、未覆盖区域、顺序/遮挡一致性。

**重要技术风险**：如果 `M1` 的软 Mask 在 `M3` 被直接拿来作为图像级 alpha，且未被压缩入 `P_I`，就发生了**信息绕过**；这违反核心要求。

**科学 Gate**：相较同码率的固定网格 RVQ 或其他离散表示，重建具备可解释的优劣比较；同时 Gene 的插入、删除、移动和重复具有可验证的局部作用。不能把高 PSNR 当成通过全部测试。

### M4 — Novel Composition / A+B→C（独立于重建测试）

**核心定义**：在生成/重组新场景 C 时，模型**不允许输入 C 的 RGB、AE latent、DINO features、Slot 或任何派生结果**。C 只可用于评价。

**实验 1：Held-out-combination reconstruction（表征泛化）**
- 输入 C，解析其程序后重建 C；使用 train 中未出现的因子组合。
- 明确仅能得出“未见组合的编码/重建能力”，不能代替真正生成实验。

**实验 2：Paired scene recombination（生成组合能力）**
- 由合成生成器构造空间相机、道路与必要背景对齐的 A、B、反事实 C；例如 A 的建筑形态 + B 的植被布局。
- 仅从 A、B 推断程序，选择/交换某类或某区域的实例子程序；第一阶段可**仅在组合控制器**使用合成编辑区域真值，但不能用于训练实例发现和 Gene Dictionary，且不得读取目标 C。记录这是一种“给定编辑区域的程序重组”。
- 对应实例选择应基于**空间区域相交/几何匹配**，不能假设 Slot 0 在两图里都代表同类实体。
- 构成的 `P_C^new` 必须满足相同程序格式、最大实例容量和编码预算。若超出容量，输出“不可表示/超预算”并计为有效失败结果，不能自动扩容。
- 与由生成器得到的目标 C 比较：编辑区域属性/结构正确性、非目标区域差异、边界融合、关系合理性、RGB/latent 质量。
- 如果生成质量不佳，可另设“使用 A/B/C 配对进行重组训练”的实验，但必须与**不使用组合监督**的主实验区分，不能混合结论。

**实验 3：Instance-count extrapolation（重复实例外推）**
- 训练样本 1–3 个可见重复构件，测试增加到 4–5 个或更高密度（具体场景需确保程序容量足够或按超容量报告）。
- 测量重复调用共享 Gene 后的重建/生成质量；不能声称已经测试递归层级组合，除非程序真的支持嵌套 Repeat/Group。

**科学 Gate**：必须产生**未经 C 编码的输出**并量化评价。只做 C→重建 C，不算完成 M4。

### M5 — 结构消融：SAE 与 GRCT 只有增益才保留

比较以下**具有明确实验意义**的配置：

| 版本 | 实例发现 | Gene 编码 | 组合器 | 回答的问题 |
|---|---|---|---|---|
| A | DINO + Adaptive Slot | 自由连续 Slot（仅诊断上限） | 共享空间 decoder | 实例发现本身是否可行？ |
| B（主模型） | DINO + Adaptive Slot | 共享离散 Gene 字典 | 共享空间 decoder | 共享 Gene 是否有贡献？ |
| C | DINO + **SAE** + Adaptive Slot | 同 B | 同 B | SAE 是否增加稳定复用/分组质量？ |
| D | 同 B | 同 B | **GRCT** | 关系 Transformer 是否在真正组合测试中改善？ |
| E（诊断） | 固定网格 | RVQ | 直接拼接 / 相同渲染基线 | 普通离散空间编码达到什么上限？ |

优先在稳定主模型后做 C/D，勿一次性联训。C 的 SAE 特征、D 的 GRCT 结构和所有额外参数都需计入模型总成本；结论从性能—码率—参数—组合泛化的联合比较中得出。

---

## 5. 推荐初始配置（建议值，非性能保证）

```yaml
experiment:
  name: urbangene_v2
  seed: 42
  output_root: outputs/v2
  train_labels_for_instance_discovery: false

data:
  image_size: 256
  splits: [train, val_seen, test_seen, test_unseen_combo, test_higher_complexity]
  test_split_locked: true

visual_encoder:
  family: dinov2  # 或有已验证权重的 dinov3
  pretrained: true
  frozen: true
  feature_layer: configurable
  input_resolution: configurable
  multi_scale: false

instance_discovery:
  method: adaptive_spatial_slot_attention
  max_slots: 32
  slot_dim: 256
  iterations: 3
  background_slot: true
  use_class_labels: false
  use_mask_labels: false
  use_box_labels: false

gene_dictionary:
  method: discrete_shared_dictionary
  num_genes: 128
  gene_dim: 256
  codes_per_instance: 2
  use_continuous_side_channel: false
  dead_code_recovery: true

scene_program:
  quantize_position: true
  quantize_scale: true
  serialize_all_scene_specific_fields: true
  no_unbilled_masks: true
  no_target_latent_skip: true
  max_instance_count: 32

renderer:
  method: shared_local_decoder_alpha_composition
  output_space: frozen_ae_latent
  use_grct: false

training:
  staged: true
  selection_metric: val_seen_reconstruction_and_discovery
  save_best_and_last: true
  report_codebook_usage: true
  report_serialized_bits: true

ablations:
  use_sae: false
  use_grct: false
  compare_fixed_grid_rvq: true
```

配置参数由 Codex 在源码检查后对应到真实项目字段；不要假装本 YAML 与现有 CLI 已兼容。

---

## 6. 建议的代码模块与接口

优先新建隔离 namespace，复用通过测试的现有模块。示意目录：

```text
urbangene_v2/
├── configs/
│   ├── m1_discovery.yaml
│   ├── m2_gene_dictionary.yaml
│   ├── m3_renderer.yaml
│   ├── m4_recombination.yaml
│   └── ablations/
├── src/urbangene_v2/
│   ├── data/                 # manifest 复用、合成配对/反事实、仅评价使用 GT
│   ├── models/
│   │   ├── visual_backbone.py
│   │   ├── adaptive_spatial_slots.py
│   │   ├── slot_feature_decoder.py
│   │   ├── shared_gene_bank.py
│   │   ├── instance_gene_encoder.py
│   │   ├── scene_program.py
│   │   ├── local_instance_decoder.py
│   │   ├── spatial_composer.py
│   │   └── optional/{sae.py,grct.py}
│   ├── train/{discovery.py,gene_bank.py,renderer.py}
│   ├── eval/{instances.py,reconstruction.py,recombination.py,bits.py}
│   └── diagnostics/{pipeline_export.py,collapse.py}
├── tests/
├── docs/{V2_DECISIONS.md,EXPERIMENT_PROTOCOL.md,RESULTS.md}
└── outputs/v2/
```

推荐接口（具体可在阅读现有代码后调整，必须对齐张量 shape 与数据语义）：

```python
features = visual_backbone(images)             # [B, C, Hf, Wf]
slots = instance_discovery(features)            # slot_feats, masks, geometry, presence
program = instance_gene_encoder(slots)          # all scene-specific fields quantized
blob = program.serialize()
program2 = SceneProgram.deserialize(blob)
latent_hat = spatial_renderer(program2, gene_bank)  # [B, 8, 32, 32]
image_hat = frozen_ae.decoder(latent_hat)       # [B, 3, 256, 256]
```

**审计约束**：`spatial_renderer` 的函数签名不应接收 `images/features/slots/raw_masks/target_latent`；训练中可通过明确的独立 loss 分支访问这些数据，但最终场景生成的前向路径不得访问。

**建议的可复现命令约定**（新 CLI 具体由 Codex 实现/写进 README）：

```bash
python -m pytest tests/ -q
python -m urbangene_v2.cli diagnose --config configs/m1_discovery.yaml
python -m urbangene_v2.cli train-discovery --config configs/m1_discovery.yaml
python -m urbangene_v2.cli eval-discovery --checkpoint <m1_checkpoint> --split val_seen
python -m urbangene_v2.cli train-gene --config configs/m2_gene_dictionary.yaml
python -m urbangene_v2.cli train-renderer --config configs/m3_renderer.yaml
python -m urbangene_v2.cli eval-recombination --config configs/m4_recombination.yaml
```

`<m1_checkpoint>` 为命令行占位符，非实际路径；源码未实现 CLI 前不可把命令标成已运行。

---

## 7. 评估协议、可视化和错误定位

### 7.1 每阶段必须保存的诊断图

**M1 — 实例发现**：原图、DINO patch 可视化、每 Slot 的 soft mask（单独色图）、argmax 分组图、实例数量分布、跨视角/增强的匹配、tree crowded case split/merge 拼图。  
**M2 — Gene 编码**：每实例 Gene ID 列表、Gene 激活热图、字典使用直方图、不同图片共享同一 Gene 的案例、同一场景码率统计、死码诊断。  
**M3 — 场景重建**：原图 → AE baseline → 发现的实例 → **量化后的程序** → 局部预测模板 → 放置/覆盖 → 重建 latent → 重建 RGB → RGB/latent error。  
**M4 — 组合**：A 原图、B 原图、各自程序、重组程序、生成 C、目标 C（仅评价显示）、目标区域指标与非目标区域差异。

### 7.2 一组测试样本必须可全链条跟踪

固定若干无泄漏验证样本，保存 JSON、实际序列化二进制与可视化；每个中间变量注明 shape、取值范围、是否连续/量化、是否计入程序码率。优先覆盖以下失败：
- 不同图片输出几乎相同程序；
- 大面积区域未被实例/背景表示；
- 固定 Gene 库却只调用极少数码字；
- Slot 数量少于可辨认重复实体且大量合并；
- 程序重组后局部修改影响整张图；
- 单独的 AE / DINO 表征已经在未见场景严重失真。

### 7.3 训练/验证/测试纪律

- 用 `val_seen` 做模型选择和工程排错；预留 `test_seen`、`test_unseen_combo` 和 `test_higher_complexity` 的正式结果。
- 预定义 `TEST_PROTOCOL_LOCK.json` 后才进行最终测试；不得用正式测试结果反复修改架构，若发生则开启新版本并明确说明。
- 正式论文比较至少对主模型与最强基线运行多个随机种子（建议≥3），报告均值/方差或置信区间。
- 保留真实 bitstream 和复原一致性测试；不能只用理论 `log2(K)` 替代真实码长。
- 对照既看相同**每图编码成本**，也看相近**共享模型参数/容量**；若无法同时匹配，绘制 Pareto/曲线并明确限制。

---

## 8. 必须通过的工程自动化测试

1. **No-label training**：训练 DataLoader 不传入 `scene_spec` 的对象标签/Mask/框/计数，测试断言标签路径不可访问。
2. **Shape / dtype / device**：Slot 输入输出、K 个 Gene、r 个 ID、可变 M、空槽位/背景路径均可 CPU 小样本前向与反向。
3. **Slot competition**：Mask 在指定归一化维度符合概率约束，空槽位不会导致 NaN；多图批量结果稳定。
4. **Quantization**：所有场景特定参数可以量化/序列化/反序列化；禁止未量化残差在 Python object 中偷渡。
5. **No-leak**：给 renderer 任意两张不同目标图片，但相同 `program_bytes` 时，输出应完全相同（允许浮点容差）；后续 DINO/Slot/Mask 不参与生成。
6. **Gradient flow**：M1 Slot、M2 Gene 和 M3 Renderer 该训练的参数梯度非零且有限；冻结 DINO/AE 权重始终不更新。
7. **Codebook collapse monitoring**：死码、使用熵、实例码重复率与程序多样性纳入每 epoch 日志。
8. **Recomposition integrity**：A+B→C 使用 A/B 的程序生成；改变或移除目标 C 文件不能影响生成 C 的结果，仅影响评价过程。
9. **Capacity enforcement**：超出最大实例数或预算不能静默截断造成伪成功；必须记录拒绝/错误/容量不足统计。
10. **Legacy isolation**：V2 不能写入 V1.1/V1.2 的日志、检查点、队列配置和 output 根目录。

`pytest` 单元测试 + CPU smoke 完成，才允许启动 GPU 的小规模验证；GPU 完整训练必须等科学 Gate 通过。

---

## 9. 实施优先级（非常重要）

| 优先级 | 任务 | 不要做的事 |
|---|---|---|
| **P0** | 审计并保留 V1.1/V1.2，确认 DINO 在合成数据上的空间特征质量 | 不覆盖旧队列，不重跑整套 14 组 |
| **P0** | 完成 M1 DINO + Adaptive Spatial Slot 的无标注实例发现与评价 | 暂不加入 SAE、Gene、GRCT |
| **P1** | 完成 M2 少量共享离散 Gene 表示，报告码率和真实复用 | 不用自由连续向量绕过字典 |
| **P1** | 完成 M3 受限空间实例解码与完整 bitstream 重建 | 不直接把 Slot mask 当生成输入 |
| **P1** | 完成 M4 真正的 A+B→C 组合生成验证 | 不把 C→C 重建冒称组合生成 |
| **P2** | 完成 SAE / GRCT 消融和真实街景初步迁移 | 主模型未成立时不叠加复杂模块 |

**硬停条件**（发现任一项先诊断）：输入无区分、全部同一码字/程序、背景或 Mask 泄漏、实例数明显塌缩、程序字节数统计错误、训练中偷偷使用实例标签、用 C 的输入生成 C 却称为 A+B→C。

---

## 10. Codex 的具体交付格式

每个里程碑必须同时交付：

- 可运行代码、最小配置、README 的实际执行命令；
- 独立验证/单元测试结果（测试数量、通过/失败原因）；
- 当前最佳检查点路径（若尚未训练必须明确 `not_run`）；
- JSON/CSV 指标 + 失败案例可视化 + 中间变量/码率审计；
- `STATUS.md`：本阶段做了什么、有什么证据、是否满足进入下阶段的 Gate；
- `RESULTS.md`：只写**真实运行**的数字，清楚区分工程 Smoke、验证集与正式锁定测试；
- 下一阶段计划及其依赖，不使用模糊“以后再做”替代具体验收项。

不承诺任何模型一定成功；不写伪造的论文结果。不允许只提交漂亮示意图作为实例发现或组合泛化证据。

---

## 11. 可直接复制给 Codex 的执行任务

> 请基于 `UrbanGene_V2_Hierarchical_Compositional_Gene_PRD.md`，在当前项目中新建隔离的 UrbanGene V2 研发线。不要覆盖或影响 V1.1/V1.2 的代码、训练队列、检查点及评估结果。
>
> 首先完成 M0 的代码/配置/日志审计并提出实际文件改动计划。随后只实现 **M1：冻结 DINO 空间特征 + Adaptive Spatial Slot Discovery + 无标签特征重建 + 实例发现评价**。不要立即集成 SAE、GRCT 或完整图像生成器。
>
> 完成 M1 的自动化测试、CPU smoke 和少量验证图后，给出具体的实例计数误差、mask 匹配质量、slot split/merge 与跨增强稳定性结果；若实例发现不稳定，优先定位问题，不能仅凭重建 loss 下降继续 M2。
>
> 当 M1 通过再依次开发 M2 共享离散 Gene 字典、M3 受限空间组合生成和 M4 独立 A+B→C 测试。严格阻断所有未经计费的连续场景信息侧路，并对每张图的可序列化程序做实际 bitstream 审计。最后将 SAE 与 GRCT 作为可插拔消融实验，只有表现出明确增益才升级成主模型。
>
> 每个阶段都报告真实已运行的测试和结果，未执行的训练标注为 `not_run`。所有数值结论必须能回溯到具体 checkpoint、manifest、seed、代码版本和日志。

---

## 12. 最终成功的定义

**工程成功**：训练链路可复现、生成链路仅依赖序列化程序、输入依赖与容量约束不塌缩、测试协议严格隔离。  
**方法成功**：能够发现相对稳定的可定位单元，多个单元调用共享 Gene，跨图像复用率可测，并有明显的实例级操作效果。  
**科学成功**：在公平信息预算与模型成本对照下，重建/生成的质量、复用能力与未见组合 A+B→C 具有可重复、可解释的优势。  

**不应声称**：数学意义上的全局最少 Gene、必然对应人工语义对象、完全语义解耦、或未经目标图输入就能生成任意真实街景——除非单独设计并通过了对应实验。
