# 从当前连续 Slot 编码输出真实 RGB 的诊断实验

用户要求直接查看目前的编码能解出什么街景，因此在 M1 主实验之外增加一个小规模、独立的 RGB 转换诊断。主训练源代码、原有检查点、实验队列和迁移程序均不改动。

## 数据流

训练目标来自真实 GoogleSV 街景。冻结主实验 step9000 的模型及预训练 `sd-vae-ft-mse`。

```text
已有 DINO 特征 → 冻结 Slot 编码器 → 连续 Slot + geometry
→ 冻结特征解码器 → Slot 重建的 DINO 特征
→ 共享、固定的 QR 投影 → 新训练的小型 CNN
→ 4×56×56 的 VAE latent → 冻结 VAE decoder → 448×448 RGB
```

Slot 特征解码器最后是 128→1024 的共享线性映射。QR 投影仅保留该映射列空间的坐标，降低缓存和计算成本；这个投影不接收额外图像信息。

原始 RGB 和从 RGB 编码得到的 VAE latent 仅用于训练监督及单独的评价对照。**生成图像时，仅读取已保存的 Slot 程序和共享模型权重**，不读取源照片、源 DINO 特征或源 VAE latent。

这是连续 Slot 的信息保留能力诊断，并未实现正式离散 Gene 字典或验证对象层面的组合泛化。

## 首轮设置

- 训练：train split 首个特征分片中的 2,048 张真实照片。
- 验证：validation split 前 64 张，按既有划分与训练全景及内容去重。
- 显示：固定索引 0、16、32、48，对应 Austin、Bogota、Lagos、London（加拿大）。
- 优化：共享 CNN，1,200 个 latent 回归更新，batch16；再做100个 RGB 监督更新，batch2。
- latent 损失：MSE + 0.25×四倍平均池化后的 MSE。
- RGB 微调：像素 L1 + 八倍平均池化的 L1 + 0.1×latent 损失。
- 冻结 VAE 仍允许梯度经过 decoder 到达 CNN，不更新 VAE 权重。
- VAE 使用原始 posterior mode，直接 decode，无 diffusion scheduler 的缩放。
- 验证对照：VAE 对原图的还原、正确 Slot 编码、固定错配编码、训练均值编码。
- 组合：重放先前保存的 identity / delete / move / duplicate / replace_B 程序。

训练过程中通过 latent 验证损失保存最优转换模块，再从其初始化 RGB 微调。展示的是最后一次微调结果，未按图片观感挑选。

## 计算与输出

仅与原主实验共享物理 GPU2，仍限制项目使用的物理卡集合为 GPU1、GPU2；不暂停主训练。诊断进程有显存上限与外部30分钟超时，禁止 CPU 模型运算。数据准备缓存位于本地机械硬盘；模型、图片、协议、指标、日志及代码快照保存在 NAS 的独立输出目录。

主程序：`tools/run_slot_rgb_probe.py`。独立重放：`tools/render_slot_rgb.py`。核心模块：`src/urbangene_v2/rgb_probe.py`。

独立重放需要显式设置 `CUDA_VISIBLE_DEVICES` 为原分配中的单张卡 UUID，然后提供 `--program`、`--bridge`、`--discovery`、`--vae` 和一个尚未存在的 `--output` PNG 路径。这个命令没有原图输入参数。

## 如何解释结果

VAE 对照说明预训练图像压缩器自身能保留多少内容；Slot RGB 说明当前连续编码经小型转换模块能还原多少。错配和均值编码用于检查输出是否确实依赖输入编码。PSNR 不能证明发现真实实例，组合后有变化也不能证明完成正确的对象编辑。对照仍无法完全区分 DINO 信息损失、Slot 信息损失及转换器训练不足。

正式 V2 还需要受限离散 Gene 程序，以及独立的空间组合生成器和 A+B→C 评价。
