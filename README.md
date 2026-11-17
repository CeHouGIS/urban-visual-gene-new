# Urban Visual Gene：四层顺序重建

从真实街景中学习跨图共享的局部视觉字典，按 L0 → L1 → L2 → L3 的顺序重建街景。

- 当前分支：[`hierarchical-sequential`](https://github.com/CeHouGIS/urban-visual-gene-new/tree/hierarchical-sequential)，固定空间单元、多尺度残差量化与逐层重建。
- 旧版分支：[`additive`](https://github.com/CeHouGIS/urban-visual-gene-new/tree/additive)，保留原 `main` 的加性 AE、多尺度残差 ISTA 等代码和提交历史。

本分支在旧版历史上增加新实验，旧实验目录继续作为对照保留。

## 当前实验

[实验说明](experiments/hierarchical_sequential_gene_10k_20261117/README.md) · [模型](experiments/hierarchical_sequential_gene_10k_20261117/source/model.py) · [训练](experiments/hierarchical_sequential_gene_10k_20261117/source/train.py) · [配置](experiments/hierarchical_sequential_gene_10k_20261117/configs/experiment.json)

|层|整图格网|输出分辨率|共享字典条目|每单元索引数|
|---|---|---|---:|---:|
|L0|1×1|32×32|64|4|
|L1|2×2|64×64|128|4|
|L2|4×4|128×128|128|4|
|L3|8×8|256×256|256|4|

L0 在中性画布上生成底稿。每个后续层使用已解码父图，编码剩余误差，再在对应空间单元内解码修正并加回父图。
解码端只接收字典索引和已解码的父上下文。没有原图 skip、任意位置的稠密激活图或每单元 2×2 像素层。

先逐层训练，冻结已完成父层的参数和码本 EMA；完成后才以十分之一学习率联合微调。逐层目标为：

$$\mathcal L_\ell=\operatorname{MAE}(\hat I_\ell,I_\ell)+0.25\,\mathcal L_{\mathrm{commit},\ell}.$$

联合阶段对四层相同目标等权平均。LPIPS 仅用于最终评价。码本采用 EMA 更新，同层共享字典进行四步残差量化。
85 个空间单元、576 种共享条目、每图 340 次索引调用；固定长度索引预算 329 byte/图，不含共享模型和容器开销。

10,000/1,000/1,000 张真实街景用于训练/验证/测试；格网、全景和图片 ID 在切分间隔离，同城可跨切分。
原测试集已被查看，用作诊断集；另报告其中 120 张训练未见城市图像。
两卡 DDP、全局 batch 256、随机初始化。功能是否形成“氛围、布局、纹理”分工，需要训练后的响应、替换和复用实验验证。

## 检查与运行

原运行已通过 CPU 数据协议检查、12 项两卡 GPU 结构/分布式检查和完整阶段 smoke；历史记录位于实验目录的 `evidence/`。
这些检查验证执行协议，不代表正式训练已经完成或层级语义分工已经成功。

先安装匹配 CUDA 的 PyTorch，再安装 [requirements.txt](requirements.txt)。准备外部数据和独立运行目录后，按 [复现说明](docs/REPRODUCIBILITY.md) 启动。
代码、原始配置、设计文档和小型检查记录保存在本仓库；动态训练状态、街景图像、缓存、检查点及凭据保留在原存储位置。
[存储说明](docs/STORAGE.md) 列出服务器路径；[新实验来源清单](experiments/hierarchical_sequential_gene_10k_20261117/SOURCE_MANIFEST.json) 记录复制来源。

旧版完整说明和历史结果请查看 [`additive` 分支](https://github.com/CeHouGIS/urban-visual-gene-new/tree/additive)。
