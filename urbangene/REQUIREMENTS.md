# UrbanGene 项目需求报告（交付 Codex 实现）

**项目名称**：UrbanGene — Minimal Compositional Representation of Urban Visual Scenes  
**文档版本**：v1.0（研究原型需求）  
**项目类型**：机器学习方法研究 / 可复现实验原型  
**语言与技术栈**：Python 3.11+、PyTorch、配置文件 YAML、CLI、pytest；训练以单 GPU 可运行的轻量原型为目标。

## 1. 项目目标与核心科学问题

**核心问题**：能否利用数量有限、跨场景共享的基本潜在表征（Urban Visual Genes），通过明确且可复用的组合操作，以较低的信息成本重建丰富的城市视觉场景，并泛化到训练中未出现过的组合？

模型不必把每个 Gene 解释为“树木、建筑、道路”等人类语义。Gene 可以是不可命名的潜在模板。最重要的是：**可复用、可实例化、可组合、总体紧凑、具有未见组合泛化能力**。

项目希望回答以下可检验问题：

- **Q1 表达能力**：固定小基因库时，模型能在多大程度上重建不同场景？
- **Q2 表示效率**：在相近重建质量和模型规模下，它是否比普通离散潜在编码使用更少的场景信息？
- **Q3 组合性**：是否能够在训练未见过的“成分 × 空间关系”组合上重建或生成正确场景？
- **Q4 扩展性**：当场景中的重复实例增加时，能否维持基因库不变、只增加组合编码？
- **Q5 机制贡献**：组合结构带来的收益是否独立于编码预算和解码器容量？

**科学边界**：不声称发现唯一真实的城市基因；不把“基因数量少”直接等同于“总信息量少”；不把能重建未见图片直接等同于“可以无条件生成未见组合”。

## 2. 与前期讨论一致的设计决策

1. **不以 Neural Superposition 或 Sparse Autoencoder 为必选前提**。它们可作为后续比较的特征编码方式；本项目核心是组合表示。
2. **参考 SCION 的 Gene 与 Instance 分离思想**：同一 Gene 可在多场景、多位置、多尺度重复调用；但本项目主要面向跨场景共享及未知组合，而不只是单场景压缩。
3. **先用固定、简洁的组合算子，不从第一天实现完整的自动语法搜索**。
4. **第一版只使用重建损失**，通过基因数量、实例数量、离散参数与无旁路结构限制信息容量。
5. **把组合泛化作为独立测试集的指标**。有明确实验需要时，第二阶段可加入成对场景重组训练，但仍使用统一重建损失。
6. **先可控合成场景、再真实城市数据**。先验证研究命题，再追求逼真生成。

## 3. 数学定义

全数据集共享一个潜在基因库：

\[
\mathcal G=\{g_1,\ldots,g_K\},\quad g_k\in\mathbb R^{C\times r\times r}.
\]

每张场景由编码器产生一个有限长度的**场景组合程序**：

\[
P_x=\{(k_j,v_j,u_j,s_j)\}_{j=1}^{m_x},\quad m_x\le M.
\]

- \(k_j\)：离散 Gene ID；\(v_j\)：实例是否启用；\(u_j\)：量化二维位置；\(s_j\)：量化尺度（初版不开放任意精度的连续参数）。
- 组合器按照 `Place -> Transform -> Overlay`，将基因模板写入潜在画布；相同 Gene ID 可在不同 slot 重复出现。
- 模型输出：\(\hat x =D_{\mathrm{frozen}}(\operatorname{Render}(\mathcal G,P_x))\)。
- 第一版优化：

\[
\min_{\theta,\mathcal G}\;\mathbb E_{x}[d(x,\hat x)]
\quad\text{s.t.}\quad K\le K_{max},\;m_x\le M,\;\operatorname{Bits}(P_x)\le B.
\]

其中参数预算与离散位数必须在代码中**硬约束**，不能只作为文字描述。

**重要区分**：\(K\) 是基因类型数量，\(m_x\) 是某张图片中的实例数；两者不可混用。

## 4. MVP 模型架构（必须实现）

### 4.1 Frozen Image Autoencoder（固定图像编码/解码器）

- 推荐先准备一个训练好的轻量卷积 Autoencoder，将 `256×256 RGB` 映射至 `C×32×32` 潜在图，例如 `C=4` 或 `8`。可使用可合法获取的预训练 VAE，或**先单独训练并冻结**一个轻量 AE。
- Gene 实验阶段冻结 `E_img`、`D_img`，使各方法共享同一个图像解码器。
- 该 AE 的预训练属于独立准备阶段，不影响 Gene 阶段“单一重建目标”的规定。
- 必须报告冻结 AE 自身的重建上限，避免把 AE 的损失误判为 Gene 的局限。

### 4.2 Scene Parser（场景解析器）

- 输入原图或冻结 AE 的目标 latent；推荐第一版直接输入目标 latent，降低不必要的模型复杂度。
- 轻量 CNN/Transformer 输出最多 `M` 个 slots。
- 每个 slot 预测：启用/关闭、Gene ID（`K` 类）、网格位置（如 `16×16`）、离散尺度（如 `{0.5,1,1.5}`）。
- 离散选择可以使用 Straight-through Gumbel-Softmax。FSQ 是后续可选替代；**不要求同时实现 VQ/FSQ/SAE 三套方案**。
- 不允许输出另一个可传给解码器的自由连续 latent、隐藏全图 residual 或输入图像 skip-connection。否则会绕过 Gene 表示瓶颈。

### 4.3 Gene Library（共享基因词典）

- 初版直接使用 `nn.Parameter` 学习大小为 `[K,C,r,r]` 的 latent 模板，例如 `r=8`，避免额外引入大型生成网络。
- 相同 Gene ID 在不同位置、不同图片共享完全同一模板。
- 若需学习透明度图，它属于共享模板参数，不允许每个实例传递无限精度、任意内容的额外通道。
- 初始 `K={32,64,128}`；不要预设 Gene 具有可读语义。

### 4.4 Compositional Renderer（组合渲染器）

- `Place`：将模板放在离散坐标。
- `Transform`：支持有限、可控的尺度变换；旋转可留待后续。
- `Overlay`：用固定且可微的加权融合/alpha compositing 将多个实例写入 `C×32×32` latent canvas。
- `Repeat` 的第一版含义是多个 slots 选择同一个 Gene ID；更节省编码的显式 Repeat 操作在 V2 再做。
- slot 顺序、重叠处理和空 slot 行为必须确定且可测试。
- 最终输出 latent canvas，送入冻结 `D_img` 重建 RGB。

### 4.5 后续功能，不进入 MVP

- 可变长程序与显式 Repeat 语法；Group / 递归组合；通过程序压缩自动学习 Macro Genes；生成式 diffusion decoder；3D 多视角场景；跨城市文化概念解释；Superposition-aware 特征恢复。这些仅作为后续扩展点保留接口。

## 5. 数据需求与严格划分

### 5.1 第一阶段：程序化合成城市数据

- 初始目标：`20,000–50,000` 张训练图，分辨率 `256×256`；另建验证集、已见组合测试集、未见组合测试集；先提供可在 CPU 跑通的少量数据 smoke 模式。
- 可以先用 Python/Pillow/简单 2D 程序化渲染器生成建筑、道路、树木、立面、重复图案和空间关系；**不要求一开始搭 Blender/CARLA**。
- 数据生成器至少具备：建筑形态、立面属性、道路宽度/布局、植被类型与布局、相对位置/重复数量等可控因素。
- 存储每张图像的生成参数/scene_spec；这些真值用于测试与生成配对编辑，不必输入无监督 Gene Encoder。

### 5.2 数据拆分

- `train`：只包含部分因素组合；
- `val_seen` / `test_seen`：训练规则内但图像实例、随机种子不同；
- `test_unseen_combo`：刻意保留若干“建筑形态 × 立面 × 植被布局”组合，只在测试出现；保证每个单独因素在训练中出现过；
- `test_higher_complexity`：训练中较少重复实例，测试增加实例数，但不得超出模型设置的最大 `M`，以免形成不公平的容量外测试。
- 所有 split 的 scene_spec、随机种子与生成规则可复现；不允许相同的场景模板/随机种子泄漏到训练与测试。

### 5.3 第二阶段：真实城市街景（后续）

- 支持用户提供本地图片目录或公开、许可允许的数据；先不自动抓取可能存在版权限制的街景。
- 真实数据必须支持按城市/街道分组划分；只在合成实验验证核心方法后开展。

## 6. Loss 与训练策略

### V0：只用一个主 Loss

\[
\mathcal L_{\mathrm{V0}}=\|E_{img}(x)-\operatorname{Render}(G,P_x)\|_1.
\]

可配置是否改成冻结 `D_img` 上的 RGB L1/感知距离，但**默认只保留一种主重建目标**。对离散选择使用 straight-through 估计，不增加独立的 Composition、Modularity 或 MDL 正则项。

- 用硬参数限制代替 MDL：限制 `K`、最大实例 `M`、位置/尺度精度、是否存在旁路、固定解码器规模。
- 用明确的组合式 Renderer 代替结构性的 Composition Loss。
- 用空间放置与共享实例机制，先替代 Modularity Loss；编辑稳定性留待独立测量。
- 如果 V0 发现 slot collapse、梯度不稳定等问题，允许引入最低限度的数值稳定措施，但必须记录，不能默默改变研究假设。

### V1：可选的配对组合训练（仍使用同一损失形式）

- 构造 `(source_A, source_B, operation, target_C)` 配对。
- 根据已知合成 scene_spec 指定交换的区域/子结构，由模型使用共享基因与组合器完成重组。
- 将普通重建样本与组合重建样本混合训练，统一计算 `d(target, prediction)`；不需额外定义一套复杂的损失权重。
- **必须保留仅 V0 训练、不使用组合配对的基线**，以检验组合性是否自然出现。

### V2（只有在必要时）：自适应 Rate–Distortion

\[
\mathcal L=\mathcal L_{recon}+\beta\,\mathbb E[-\log_2 p(P_x)].
\]

仅当希望让不同场景动态使用不同的实例数与编码成本时添加；不是 MVP 的先决条件。

## 7. 核心实验与基线

### 7.1 必做实验

| 实验 | 变量 | 指标 | 要回答的问题 |
|---|---|---|---|
| E1：容量—重建 | K=32/64/128，M=8/16/32 | latent L1、RGB PSNR/SSIM、可选 LPIPS | 小基因库能表达多少场景？ |
| E2：已见/未见组合 | seen vs held-out | 属性匹配准确率、结构关系准确率、泛化差距 | 是否真的有组合泛化？ |
| E3：重复实例扩展 | 重复数量/布局复杂度 | 重建、属性保真度、失败率 | 是否能复用固定的 Gene？ |
| E4：信息预算 | 改变 Gene ID、坐标、尺度精度 | bits/scene vs 质量；共享参数量 | 是否只是使用更多信息？ |
| E5：结构消融 | 无空间放置、仅加和、带 Place/Overlay、带 Repeat（后续） | 同 E1-E3 | 哪些组合结构真正有效？ |

**注意**：`test_unseen_combo` 的输入图像重建只能证明**对未见组合的表征泛化**；要声称“生成了训练未见的组合”，必须另做不提供目标图像的合成程序/配对编辑生成测试。两种实验分开报告。

### 7.2 至少实现的 Baselines

- `B0`：相同或接近编码预算的普通离散图像表示/Autoencoder（无共享空间实例的显式组合）。
- `B1`：共享基因词典 + 简单线性求和，不包含位置变换。
- `B2`：共享基因词典 + Place/Overlay，但禁止同一 Gene ID 被重复使用（研究复用作用；若容量不匹配需说明）。
- `UrbanGene`：共享基因 + 重复实例 + Place/Transform/Overlay。

尽可能固定图像解码器、目标分辨率、训练集、计算预算；报告模型参数量与场景实际编码比特数，避免不公平比较。

### 7.3 辅助诊断

- Gene 使用频率、有效利用率、冗余性/塌缩；每张场景平均实例数。
- 可视化共享 Gene 模板、空间实例、重建图、失败案例。
- 编辑试验可以做，但不把语义可解释性设为必须通过的验收标准。

## 8. 工程目录与接口

建议创建：

```text
urbangene/
├── README.md
├── pyproject.toml
├── configs/
│   ├── smoke.yaml
│   ├── v0_64genes.yaml
│   └── ablations/
├── src/urbangene/
│   ├── data/synthetic.py
│   ├── data/dataset.py
│   ├── models/image_autoencoder.py
│   ├── models/parser.py
│   ├── models/gene_library.py
│   ├── models/renderer.py
│   ├── models/urbangene.py
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   └── visualize.py
├── scripts/
│   ├── generate_synthetic.py
│   ├── train_image_ae.py
│   ├── train_urbangene.py
│   ├── evaluate_all.py
│   └── run_ablation.py
├── tests/
│   ├── test_data_splits.py
│   ├── test_renderer.py
│   ├── test_budget.py
│   ├── test_no_bypass.py
│   └── test_smoke_train.py
└── outputs/  # gitignored
```

### 必要接口

- `SyntheticSceneGenerator.sample(spec, seed) -> (image, metadata)`
- `SceneParser.forward(latent) -> discrete_program`
- `GeneLibrary.lookup(gene_ids) -> templates`
- `CompositionalRenderer.forward(program, library) -> reconstructed_latent`
- `UrbanGene.forward(images) -> {latent_hat, image_hat, program}`
- `evaluate(checkpoint, split) -> metrics.json + visualizations`

### 配置示例

```yaml
seed: 42
data:
  image_size: 256
  train_samples: 20000
  val_samples: 2000
  test_seen_samples: 2000
  test_unseen_samples: 2000
model:
  latent_channels: 4
  latent_hw: 32
  gene_count: 64
  gene_patch_hw: 8
  max_instances: 16
  position_grid: 16
  scale_levels: [0.5, 1.0, 1.5]
  image_autoencoder: frozen
  allow_continuous_scene_residual: false
training:
  loss: latent_l1
  batch_size: 32
  learning_rate: 0.0003
  epochs: 30
  amp: true
logging:
  save_every_epoch: true
  visualization_every_n_steps: 200
```

具体数值作为初始默认值，并非经验证的最优值；显存不足时可降低 batch size。

## 9. Codex 交付顺序与验收标准

### Milestone 0：可运行项目骨架

- 配置解析、依赖、命令行入口、随机种子、设备选择、日志、checkpoint、pytest。
- CPU smoke 训练能处理一个 tiny dataset，完成 forward/backward，loss 有限、shape 正确。

### Milestone 1：合成数据与冻结 AE

- 生成可复现的训练/验证/测试数据及 scene_spec；提供 held-out 组合划分与可视化。
- 准备并冻结图像 AE；报告 AE 的重建误差与示例图像。

### Milestone 2：UrbanGene V0

- 实现固定 K/M 的离散 Scene Parser、共享 GeneLibrary 与可微 Renderer。
- 严格检查 `no bypass`：重建器只接受离散程序及共享基因；修改输入图但保持程序不变，不得改变重建结果。
- 跑通训练并保存原始/重建/实例分布图、loss 曲线和 checkpoint。

### Milestone 3：正式评估与消融

- 至少跑 E1/E2/E3/E4 的基础版本与 B0/B1 两个关键 Baseline。
- 输出 CSV/JSON 指标、可视化、率失真图、见过/未见组合性能对比图。
- 明确记录未成功的实验与已知限制，不以“漂亮重建图”代替定量结论。

### Milestone 4（可选）：配对组合与高级语法

- 当 V0 达到基本重建能力后，再加入配对重组训练、显式 Repeat 和递归 Macro Gene。
- 不得把这些复杂模块作为 V0 能跑通的前置条件。

### 工程验收（必须满足）

- 一条命令生成少量样本、一条命令启动 smoke 训练、一条命令评估与导出图片。
- 数据拆分固定且无组合泄漏；重复运行同种子得到可复现结果（允许 GPU 数值误差）。
- 模型使用的基因类型、实例上限、位置/尺度位数可以从配置文件直接修改。
- 基因库必须跨场景共享，同一个 Gene ID 可重复实例化。
- 不存在隐式高维场景连续残差或图像 skip-connection。
- 输出包含：`metrics.json`、`recon_grid.png`、`gene_usage.png`、`rate_distortion.png`、`seen_unseen_comparison.png`、配置快照、checkpoint。
- README 记录数据、命令、训练设置、评价口径和已知问题。

## 10. 重要技术风险及验证口径

1. **Gene Collapse**：模型可能只使用少数或固定几个 Gene；监控利用率与效果，先排查训练与采样，不立即堆新 Loss。
2. **信息旁路**：位置、尺度、透明度或 skip-connection 可能隐藏大量场景信息；所有场景特定信息必须量化计费。
3. **Decoder 背锅**：固定 AE 本身的失真可能限制重建；独立报告其性能上限。
4. **组合性被夸大**：未见组合图片的重建不等于无目标输入的生成；必须分开评价。
5. **SCION 类工作的已有贡献**：不能仅把“可重复使用的神经基元”当作全新发明；重点在跨场景共享、总体表示效率、学习组合规则、未见组合泛化。
6. **没有真正最优性保证**：固定 K/M 只是受限表示研究，不能证明找到数学上唯一、最少的基因；应报告 K/M/bit-budget 的性能前沿。
7. **数据捷径**：合成数据应包含位置/颜色/种类的多样化与混合；避免模型只记住一个固定背景。

## 11. 明确不做的事情（第一版）

- 不要求基因语义可解释、人工命名或与物体类别一一对应。
- 不要求神经叠加、SAE、物理规律或文化标签。
- 不从头训练大型扩散/视觉基础模型。
- 不上来做大型 3D 场景、自动 Macro Gene 搜索、全套场景语法。
- 不引入四项复杂 Loss；先保持重建目标简单。
- 不默认联网批量下载大规模模型或图片；如需外部权重与数据，优先提供配置接口和离线 fallback。

## 12. 可以直接发送给 Codex 的执行指令

> 请以本需求报告为唯一当前规格，在一个新仓库中实现 **UrbanGene V0** 的可运行研究原型。按 Milestone 0 → 1 → 2 → 3 递进。优先完成可控 2D 合成城市数据、冻结轻量 AE、固定 K/M 的离散 Gene Parser、共享 Gene Library、Place/Transform/Overlay 潜在渲染器、**仅重建损失**的训练循环，以及 seen/unseen-combination 评估。先做最小 smoke 训练与单元测试，再进行全量实验。不要引入复杂正则、SAE、Diffusion、3D 或 Macro Gene。所有场景特定参数都必须离散化并计入信息预算，禁止连续 latent 旁路。提供 README、CLI、YAML、测试、样例重建图、指标、失败案例，并说明哪些科学主张尚未验证。若具体实现有多个可行方案，请优先选择最简单、稳定、可复现、便于消融的方案，并在文档里记录取舍。不得声称尚未完成的实验已经成功。

---

**研究主张一句话版**：在可控信息容量下，学习一套跨场景共享的潜在基因及组合操作，系统检验它们能否比非组合式表示更高效地重建复杂城市景观，并泛化到训练未出现过的视觉组合。
