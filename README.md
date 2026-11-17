# Urban Visual Gene

从真实街景中学习可跨图片共享的视觉字典，并用图像相关的空间表达系数组合、重建街景。
本仓库保存项目代码、实验配置、研究文档和小型结果记录，包含多个阶段的实验快照。

## 当前讨论的模型：多尺度残差 ISTA

入口：[实验说明](experiments/residual_ista_gene_10k_scratch_20261116/README.md) · [模型代码](experiments/residual_ista_gene_10k_scratch_20261116/source/model.py) · [配置](experiments/residual_ista_gene_10k_scratch_20261116/configs/experiment.json) · [结果](experiments/residual_ista_gene_10k_scratch_20261116/evidence/comparison_metrics.json)

对每个尺度，以共享字典近似求解非负稀疏编码：

$$A_s(x) \approx \arg\min_{A\ge0}\;\frac12\|T_s(x)-D_s*A\|_F^2+\lambda\|A\|_1.$$

$$F_s(x)=D_s*A_s(x),\qquad \hat x=\operatorname{Decoder}(F_c,F_m,F_f).$$

- **gene 本体**：训练得到的共享字典原子 $D_{s,g}$；训练完成后固定。
- **gene 表达**：输入图片经共享编码器和 6 步展开 ISTA 得到的空间系数图 $A_{s,g}(x)$；随图片变化。
- **可加性**：同尺度内的特征合成严格线性；最终 RGB 解码器为非线性网络。
- 当前研究重点是有效的跨图片复用；激活频率和重建质量本身不证明语义独立或功能必要性。

|尺度|字典原子数|每个原子的权重形状|256×256 输入对应表达网格|
|---|---:|---|---|
|粗|256|64×1×1|16×16|
|中|128|64×3×3|32×32|
|细|128|64×3×3|64×64|

该版本从随机初始化训练编码器、字典和解码器，无编码器绕过字典的路径，无每图 64-gene 硬上限。
训练/验证/测试分别为 10,000/1,000/1,000 张真实街景，训练 6,000 更新，有效 batch 256。
洲→国家→城市均衡采样，使用 GHSL 城市边界、1 km 格网，每格 5 个全景点、每点全部 4 张图像；切分检查隔离格网、全景与图片 ID。

## 已有结果

以下为同轮 1,000 张测试图的验证集选优检查点结果，原始记录见上方 `comparison_metrics.json`：

|模型|MAE（RGB 范围 0–1）|PSNR（逐图均值）|SSIM|LPIPS|
|---|---:|---:|---:|---:|
|单尺度 ISTA|0.03199|25.73|0.7154|0.1061|
|多尺度残差 ISTA|0.01742|31.47|0.9056|0.0270|

两组参数量和设计有差异，不能仅凭这组对照独立归因于多尺度、卷积或残差监督。结果为单随机种子的可行性实验。
中细尺度表达仍较稠密，部分粗尺度原子未使用；尚不能据此声称找到少量、语义独立、不可替代的城市视觉基因。

另一个 `group_sparse_gene_10k_20261116` 实验研究每图 8/16/32/64 个 gene 的容量限制、量化和功能复用。
它使用不同的表达预测架构与网格，属于另一版本，不应与当前 ISTA 检查点混用。

## 目录

|路径|用途|
|---|---|
|`experiments/residual_ista_gene_10k_scratch_20261116/`|当前 512-gene、多尺度残差 ISTA 模型、训练、测试和完整 gene 图册导出代码|
|`experiments/group_sparse_gene_10k_20261116/`|8/16/32/64-gene 容量、编码与跨城市功能复用实验|
|`experiments/gene_validation_three_tests_20261116/`|字典学习、率失真和组合对照实验|
|`experiments/multiscale_gene_10k_20261116/`|多尺度向量/卷积字典与旧加性模型对照|
|`experiments/additive_gene_*/`|最初加性 AE、百万级数据采样及大 batch 训练代码|
|`urbangene/`、`urbangene_v11/`、`urbangene_v2/`|早期视觉基因、GRCT 组合器、DINOv3/分组实验|
|`experiments/object_centric_pilot/` 等|早期分组与生成可行性实验|
|`archive/`|V2 的 20k/1m 数据阶段代码快照|
|`docs/`|研究说明、原始需求、发布清单与复现说明|

## 环境与使用

建议 Python 3.11+、Linux 和支持 CUDA 的 GPU。先按运行环境安装匹配的 PyTorch/torchvision，再安装当前 ISTA 依赖：

```bash
python -m pip install -r requirements.txt
```

这是实验代码快照。配置保留原始 NAS/机械盘绝对路径，跨服务器运行前需要修改；数据、缓存、检查点及图册不在 GitHub。
克隆仓库后不会自动启动训练。GPU 测试、推理与训练只在取得空闲 GPU 租约后运行；CPU 仅用于数据检查、绘图与代码校验。
详细入口及外部资源准备见 [复现说明](docs/REPRODUCIBILITY.md)。

## 文件范围与来源

保存项目源代码、配置、测试、说明及小型历史结果记录。原始图像、训练清单、模型权重、特征数组、环境和凭据保留在原存储位置。
历史验证记录存入 `evidence/`，避免被训练监督器误认为新环境已经通过启动检查。
源文件保持快照内容，路径映射及 SHA256 记录在 [SOURCE_MANIFEST.json](docs/SOURCE_MANIFEST.json)。
完整 NAS 实验资源位置见 [存储说明](docs/STORAGE.md)。
