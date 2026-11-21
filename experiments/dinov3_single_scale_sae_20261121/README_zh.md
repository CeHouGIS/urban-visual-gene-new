# DINOv3 全维特征 + 单尺度 SAE

本实验将原先从零训练的 64 通道 CNN 替换为冻结的预训练 DINOv3 ViT-L/16。
保留完整 1024 通道，不使用 PCA 降维训练 SAE。单尺度 32×32 空间网格；
2048/4096 两种字典宽度分别从随机初始化训练，旧实验代码、缓存和检查点不修改。

## 数据与预处理

- 使用相同的 10000 张训练街景和 1000 张验证街景清单；不使用 test 训练或选择模型。
- 从 manifest 中的原始真实照片读取，EXIF 转正、RGB、双线性缩放至 512×512，ImageNet 均值/标准差归一化。
- 原照片不都是 512×512（有 400×400、400×300）；缩放不会创造新的图像细节，长宽比处理延续此前方形输入的做法。
- 移除 1 个 CLS 和 4 个 register token，只取 1024 个 patch token，形状 `[N,1024,32,32]`。
- 冻结预训练骨干。其每个 token 已包含上下文，不等于仅观察一个 16×16 像素块。
- 原始特征 FP16 缓存在服务器机械盘 `/workspace/urban-visual-gene/dinov3_single_scale_sae_20261121/features`，完成的分片同步至 NAS 本实验 `features/`。
- 训练集的全部 patch 拟合通道均值 μ 和单个标量 RMS s，`F=(F_DINO-μ)/s`。没有逐通道白化，也没有降维；验证集使用同一 μ、s。
- 由于更换骨干、输入分辨率及归一化，这不是只改变“维度”的受控消融。不同骨干 NMSE 的数值不能直接比较。

## SAE 与 loss

```
RGB [B,3,512,512]
  → 冻结 DINOv3 ViT-L/16
  → 原始特征 [B,1024,32,32]
  → 固定训练集中心化及标量缩放
  → Conv3×3(1024→M) → ReLU → 量化
  → Q [B,M,32,32], M ∈ {2048,4096}
  → 线性卷积字典 D（每个 atom 为 1024×3×3）
  → 重建特征 F_hat [B,1024,32,32]
```

`q = round(clamp(ReLU(W*F+b)/0.125, 0, 63))`，训练时对重建使用 STE；
字典按 atom 归一化到单位 L2 范数，重建为 `F_hat = D*(0.125*q)`。
SAE 编码非线性，固定字典的合成为线性可加；不强制相邻位置聚拢、互斥分组或固定 top-k。

`L_SAE = NMSE(F_hat,F) + λ·bpp`；`bpp = Σ[-log2 p_g(q_g,h,w)]/512²`。
熵模型独立用量化符号的负对数似然拟合，SAE 的 rate 梯度沿用原版分段线性代理。
没有增加语义标注、实例掩膜、空间连通损失或 RGB 重建任务。

λ=0.015，是原版 0.00375 的四倍，因为 bpp 分母由 256² 改为 512²，
而系数网格仍为 32²；这样每个编码 bit 的目标函数权重保持相同。
这不代表新旧骨干的失真度相同，也不代表不同字典达到相同码率。

## 训练与验证

- 最多两张空闲 GPU，合作式 UUID 文件锁；仅 GPU 神经网络推理/训练，GPU 不可用时排队。
- 两个提取进程 → 训练集归一化统计 → 两个独立 SAE。
- 每个 SAE 1200 updates，effective batch 128，通过实测选择 microbatch 4/8/16。
- Adam，SAE/字典 lr=5e-4，熵模型 lr=2e-3，warmup 60，余弦衰减，梯度范数限制 5。
- 固定验证集 128 张评估 NMSE、中心化特征 cosine、非零率、每位置激活 atom 数和未使用 atom 数。
- 先运行 GPU 形状/梯度/rate/可加性测试，再真实特征 smoke 与显存测试，主训练重新随机初始化。
- step 50 导出明确标注步数的早期例子，1200 导出最终例子；不要将 step 50 当最终结果。
- 原版 64 通道 RGB decoder 不适用于该特征空间；RGB 重建由后续独立模块训练新的共享 1024 通道输入解码器，见 [downstream/README_zh.md](downstream/README_zh.md)。

## 输出和复现

NAS 实验根目录：`/host/mnt/nas/urban-visual-gene/dinov3_single_scale_sae_20261121`。
`progress.json` 为队列状态，`features/*.progress.json` 为提取状态，`runs/atoms_*/progress.json` 为训练状态。
`reports/dino_query_features.npz` 返回 5 张验证图完整原始特征、RGB 和验证集索引。
`reports/dinov3_features_five_scenes.png` 展示原图、特征 PCA 伪彩和固定参考点的特征相似度。
PCA 仅用 32 张训练图片拟合三维可视化坐标，所有图共享坐标及颜色范围，绝不送入 SAE。
PCA 伪彩不是分割标签，也不是图像重建。相似度参考点固定在同一图内的网格 `(23,16)`。

`runs/atoms_M/query_outputs.npz` 返回 5 张验证图的完整量化 SAE 激活 `Q` 和重建特征；
`reports/sae_M_same_atoms.png` 展示相同 atom 在不同图像中的响应，`sae_M_feature_reconstruction.png` 展示同一 PCA 下的特征重建。
扩大特征维度和换预训练骨干能改善上游表示，但不能保证 SAE 自动得到连通区域或纯净语义。

运行：使用 `configs/experiment.json` 中的 Python 执行 `source/launch.py`。
可恢复中断实验；`touch <实验根目录>/STOP_REQUESTED` 请求停止。恢复前移除该停止文件。
所有新模型独立保存，不复用旧 CNN SAE 权重。

运行环境版本见 [requirements.txt](requirements.txt)。配置中的 Python、NAS、机械盘、权重和清单路径对应当前服务器；移植到其他环境需先调整路径。下游配置包含完整上游配置副本和上游 JSON 文件的 SHA256，修改后需同步副本并重新计算 `parent_config_sha256`。本轮摘要和验证记录见 [RESULTS_zh.md](RESULTS_zh.md)。
