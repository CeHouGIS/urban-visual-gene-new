# DINOv3 + SAE：接续原框架的后续流程

## 本轮做什么

使用当前从头训练的 2048 / 4096 atom SAE 的最终结果，接续原框架的图像解码、街景检索、跨图片与 batch 的激活检查和实际算术编码。DINOv3 与两个 SAE 均冻结，只新训练一个共享 RGB 解码器；保留原实验和所有检查点。

```text
真实街景 → 冻结 DINOv3 ViT-L/16 → F [1024,32,32]
                                   ├→ 原始 F ──────────────────┐
                                   ├→ SAE2048 → Q → 字典合成 F̂ ├→ 同一个新 RGB 解码器 → 256×256 街景
                                   └→ SAE4096 → Q → 字典合成 F̂ ┘
                                              ├→ 空间池化 → 街景检索
                                              ├→ atom 激活/覆盖率/跨 batch 分析
                                              └→ 算术文件 → 精确回读 Q → 字典 → RGB
```

## 编码与新解码器

DINO 输入 512×512，使用完整 1024 通道、32×32 patch 网格，不做 PCA 压缩。特征采用训练集通道均值与一个标量 RMS 归一化。SAE 使用已训练的非负整数编码：

\[
Q=\operatorname{round}\!\left[\operatorname{clip}\!\left(\operatorname{ReLU}(W*F+b)/\Delta,0,63\right)\right],\qquad\Delta=0.125.
\]

字典合成是线性的卷积相加，atom 本身为 1024×3×3 的权重：

\[
\hat F=\sum_{g=1}^{M}D_g*(\Delta Q_g),\qquad \hat I=R_\phi(\hat F).
\]

RGB 解码器为 1024→128 的 3×3 卷积、残差块、三次两倍上采样及 RGB sigmoid 输出。只有字典合成为线性相加；SAE 编码与 RGB 解码均有非线性。没有原图跳跃连接。

新 RGB 解码器只用一个损失：

\[
\mathcal L_{RGB}=\operatorname{MAE}(R_\phi(F_{\text{source}}), I_{256}).
\]

三种输入源为原始 DINO、2048 SAE 合成、4096 SAE 合成。连续三次更新使用完全相同的 32 张图，依次切换输入源；共 1602 次更新，每源 534 次。AdamW，初始学习率 0.001，40 次 warmup，余弦衰减。用真实特征实测 microbatch 8/16/32，选显存预算内最快的值；实测后主解码器重新随机初始化。

原始 DINO 是参考输入，帮助区分 SAE 压缩丢失与 RGB 逆映射能力的限制。共享解码器本身仍有训练预算与源之间折中的影响。这不是为每种表示分别充分优化的重建上限比较。RGB 目标使用原框架的 256×256 缓存；与 DINO 的 512×512 输入分辨率分别报告。

## 后续评估与输出

- **重建：** 同一批 5 张未见街景的原图、DINO 参考重建、两个 SAE 重建；逐图绝对误差热图。128 张验证图 MAE / 平均逐图 PSNR 与训练曲线。
- **检索：** 1000 张验证图 gallery，5 个固定 query。单一网格做 1/2/4 空间平均池化、拼接、L2 归一化；这是读出方式，不增加可学习多尺度分支。排除同一全景及同一格网，返回不同格网的 top 3。无人工相关性标签，余弦值不能当准确率，也不能跨表示直接比高低。
- **激活：** 所有 atom 的平均权重、非零位置比例、出现图片比例；响应强的 atom 在不同格网上的热图。用同一个最终 SAE 重新编码原训练第 1/400/800/1200 步的图片 batch，比较平均空间激活；不是历史检查点变化。
- **实际压缩：** 验证图 0/499/999 写入真实算术文件并精确回读；再用 GPU 检查直接整数编码和码流回读编码的字典输出、RGB 输出完全相同。文件大小分别报告以 512 输入像素和 256 输出像素为分母的 bpp。共享神经网络与熵表不计入单图码流，熵表大小单列；这不表示照片无损压缩。
- **NAS 留存：** 所有 1000 张验证图的量化激活、检索向量、完整 atom 统计、算术码流、新解码器检查点、图片和报告。

激活稀疏、空间聚拢和语义可解释是不同性质。本轮展示和统计响应，不预先给 atom 命名。

## 自动执行和资源

独立队列先等上游 `progress.json` 为 `complete`，严格要求最终 SAE 为 step 1200 且配置和特征签名匹配。之后依次进行 GPU 单元测试 → 两个 SAE 导出 → 新解码器训练（GPU）与算术编码归档（CPU）→ GPU 码流解码一致性检查 → 最终报告。

GPU 通过共享 UUID flock 租约与二次空闲检查获取，同时最多两张；上游完成前本队列不占 GPU。不使用 CPU 神经推理或训练。CPU 仅用于元数据、绘图、符号编码和归档。

队列日志和临时运行数据先写 `/workspace/urban-visual-gene/dinov3_single_scale_sae_20261121/downstream`，检查点和结果保存到 NAS。代码快照、配置、最终 SAE 哈希均记录；恢复时检查兼容性。上游训练配置保持原样。

启动（避免重复启动；队列有独立锁）：

```bash
PYTHONDONTWRITEBYTECODE=1 UVG_DOWNSTREAM_CONFIG=/host/mnt/nas/urban-visual-gene/dinov3_single_scale_sae_20261121/downstream/configs/experiment.json /root/code/urban-visual-gene/experiments/dinov3_generation_revision/.venv/bin/python3 /host/mnt/nas/urban-visual-gene/dinov3_single_scale_sae_20261121/downstream/source/launch.py
```

队列状态：`/host/mnt/nas/urban-visual-gene/dinov3_single_scale_sae_20261121/downstream/progress.json`。

最终报告：`reports/RESULT_zh.md`；主要图为 `reports/decode_overview.png`、`reports/error_query_*.png`、`reports/retrieval_query_*.png` 和 `reports/reusable_atoms_*.png`。报告在导出完成和 RGB 中间验证后更新，图标题注明实际步数。

单独停止这个后续队列：在本 downstream 目录创建 `STOP_REQUESTED`；不影响上游队列。恢复前删除这个文件，重新启动 `source/launch.py`；完成的作业根据本队列收据跳过。

## 已完成模型的激活查看

以下脚本只读取保存的量化编码，CPU 绘图，不执行神经推理或重新训练。用本实验的 Python 执行，默认读取 `configs/experiment.json`；也可通过 `UVG_DOWNSTREAM_CONFIG` 指定配置：

- `source/compare_activations.py`：同一街景上比较 2048/4096 SAE 的高均值 atom 和每位置激活数量。
- `source/inspect_weights.py`：验证图 666 的网格 `(16,16)` 完整权重向量、CSV、单个 atom 的 32×32 矩阵和 NPZ。
- `source/overlay_features.py --index 666 --row 16 --col 16`：该位置权重最大的 12 个 atom 在原图上的激活叠加，以及前 4 个 atom 在其他格网的高响应街景。零权重保留原图，颜色显示权重；不将热图当作物体掩膜或语义标签。

这些图输出到 NAS 本目录 `reports/activation_comparison`、`reports/exact_weights`、`reports/feature_overlays`。2048 与 4096 模型的 atom 编号分别定义，不能按相同编号推定语义一致。
