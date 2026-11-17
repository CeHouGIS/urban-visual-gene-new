# 预训练图像编解码对照与方案修订

已完成五个城市预先固定的第一张验证/测试图片对照，无新增训练。CPU float32 推理，448×448 输入，关闭 tiling，VAE 使用后验 mode。

![原图、上一轮 RGB 探针、预训练 VAE 图像重建](/root/code/urban-visual-gene/experiments/dinov3_generation_revision/results/vae_control/comparison.png)

## 实测结果

| 路径 | 实际输入 | 448×448 共同评价尺度 PSNR |
| --- | --- | ---: |
| 上一轮 RGB 探针 | PCA 压缩后的 DINOv3 特征，经 32 个 slot；原生输出 112×112 再上采样 | 17.65 dB |
| 预训练 VAE 编解码 | 原始 RGB，直接编码与解码，448×448 输出 | 28.08 dB |

PSNR 由五张图片的平均 MSE 计算，非完整验证集统计。两条路径输入信息不同：数值差异包含表示与解码器两方面的影响，不能归因于仅替换解码器。

对五张固定图片做了视觉检查。预训练 VAE 能恢复车辆轮廓、树干、建筑边界和道路标线；细树枝、车牌等高频细节仍有损失。上一轮探针主要输出平滑的色彩与大致布局。

## 能支持的判断

这些街景通过预训练图像编解码路径可以达到明显更好的恢复质量。上一轮小型 RGB 探针与强语义瓶颈不足以检验用户想要的高保真可控生成，应替换生成路径并分步检查条件信息丢失。

本对照没有证明 DINOv3 条件扩散可行、没有发现实例、没有拆出单棵树、没有验证形态与外观解耦。完整扩散主干、对象适配器及新的分组网络尚未训练。

下一轮按 [DESIGN.md](/root/code/urban-visual-gene/experiments/dinov3_generation_revision/DESIGN.md) 执行：密集 DINOv3 条件扩散对照 → 联合对象分组 → 对象局部外观与空间编辑 → 新组合。训练不使用实例类别、框或人工 mask；单树标签仅用于独立评价。

## 文件与复现

- [运行说明](/root/code/urban-visual-gene/experiments/dinov3_generation_revision/README.md)
- [逐图指标与推理设置](/root/code/urban-visual-gene/experiments/dinov3_generation_revision/results/vae_control/metrics.json)
- [固定 Austin/London 预览](/root/code/urban-visual-gene/experiments/dinov3_generation_revision/results/vae_control/preview.png)
- [VAE 权重来源和文件校验](/root/code/urban-visual-gene/experiments/dinov3_generation_revision/models/sd-vae-ft-mse/source.json)
