# UrbanGene V2：M1 主线实验记录（2026-11-09）

当前完成的是“冻结 DINOv3 → 无标注 Slot Discovery → 重建 DINO 特征”的小规模验证。初版和空间初始化对照都按独立目录保存。最终结论以对应 `metrics.json` 中的科学 Gate 为准；训练 loss 下降不等于发现了景观单元。

## 实际完成

- 从已校验的 ModelScope DINOv3 ViT-L/16 权重提取 256 张训练图和 64 张验证图；448×448 输入得到 28×28、1024 维特征。缓存、manifest 和权重都有哈希。
- 训练/验证没有重复图片；逐张再生成验证图与实例掩码并对齐。正式测试集保持锁定。
- 两轮各训练 10 epoch、640 步，seed=42；使用 CPU。训练只输入 DINO 特征，检查点只按验证重建误差选择。
- 第二轮仅加入坐标驱动的初始空间位置，后续位置、尺度和区域仍自适应学习；没有预先定义树或建筑单元，没有实例监督。
- 工程测试 24 项通过；新增解码计算与旧计算的输出/梯度等价。

## 普通验证组结果

| 指标 | 初版 | 空间初始化对照 |
| --- | ---: | ---: |
| FG-ARI ↑ | 0.009723 | 0.085687 |
| 匹配 Mask IoU ↑ | 0.015334 | 0.040773 |
| 实例计数 MAE ↓ | 1.984375 | 2.093750 |
| 特征重建 MSE ↓ | 0.025411 | 0.025414 |
| 特征翻转匹配 IoU ↑ | 0.462006 | 0.359602 |

固定 4×4 网格的 FG-ARI 为 0.261881、匹配 IoU 为 0.056344。只用训练图计算的空间平均特征图在验证集的 MSE 为 0.017190，因此低重建误差本身也不足以证明模型使用了实例结构。

初版 Gate：**未通过**；空间初始化 Gate：**未通过**。判据为：相对单 Slot、固定网格和随机分组，在 FG-ARI 与匹配 IoU 上的逐图配对 bootstrap 95% 差值下界都大于零。该统计只描述此随机种子与此验证分布。

## 十棵树的独立诊断组

另建 12 张验证诊断图，每张十棵树，超出训练图的 1–3 棵树密度；十棵树在 RGB 和特征网格的评价标注上均可见。它不属于正式 `test_higher_complexity`，其标注和 scene_spec 只进入评价。

| 指标 | 初版 | 空间初始化对照 |
| --- | ---: | ---: |
| 树的匹配 IoU | 0.001627 | 0.005002 |
| 树独立匹配召回（IoU≥0.5） | 0.000000 | 0.000000 |

## 失败定位与解释边界

初版编码端的最大 Slot 平均占据 95.69% 注意力质量，解码端掩码熵接近均匀分配；32 个门控全部打开但非空前景组数均值仅 2.78。特征中仍可见实例并不保证 Slot 会将其分开。图像中天空和道路占比较大、模型可通过共享位置模式降低重建损失，是后续应检验的因素，尚不能认定为唯一原因。

空间初始化对照的平均中心离散度为 0.011639，初版为 0.010378；解码掩码熵分别为 3.452585 与 3.452243。需要联合查看这些数值、实例匹配和可视化，不能仅凭初始化分散就认为有效。

这两轮只有 10 epoch，不能据此断言 DINOv3 或所有 Slot Attention 方法不可行；当前证据只说明这个配置/预算尚未满足 M1。

## 下一步主线决策

维持在 M1，先解决竞争归属和局部重建的退化，再进入共享 Gene。下一项应为有单独协议的竞争/局部解码对照：检查背景分支与前景竞争、slot geometry 更新是否抹掉初始差异、以及位置信息是否让解码器绕开内容分组。对照仍只使用特征重建训练；标签只用于验证。若局部分组改善，再检验训练预算和多尺度是否有增益。M2、M3、M4 当前均未运行。

## 可追溯文件

- 初版：`/host/mnt/nas/urban-visual-gene/urbangene_v2_20261109/outputs/v2/m1_dinov3_cpu_screen_20261109`
- 空间初始化：`/host/mnt/nas/urban-visual-gene/urbangene_v2_20261109/outputs/v2/m1_dinov3_cpu_spatial_seed_20261109`
- 每轮 `validation/metrics.json`、`validation_ten_tree/metrics.json`、`tree_separation.json`、`slot_diagnostics.json`、`sample_*.npz`。
- 每轮的 `discovery_first6.png`、`discovery_worst6.png`、`soft_masks_0000.png` 等、`training_and_counts.png`。
- 初始协议：`M1_VALIDATION_PROTOCOL_20261109.json`；消融协议：`M1_SPATIAL_SEED_ABLATION_PROTOCOL_20261109.json`。
- 源码快照、执行状态、训练 stdout 日志、特征 sidecar、模型下载校验记录均在 `/host/mnt/nas/urban-visual-gene/urbangene_v2_20261109/outputs/v2`。
- 对照图：`/host/mnt/nas/urban-visual-gene/urbangene_v2_20261109/outputs/v2/m1_validation_comparison_20261109.png`。

翻转、平移、尺度稳定性在缓存特征图上计算，不等同于对变换后的 RGB 重新执行 DINO。所有模型都使用 CPU；后续 GPU 实验仍最多两张卡。原 V0/V1.1 队列与检查点未被本实验修改。
