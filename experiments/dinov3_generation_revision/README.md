# 生成方案修订与图像编解码诊断

[DESIGN.md](DESIGN.md) 给出对上一轮失败的分析与下一轮方案。此目录保留上一轮结果，没有覆盖其检查点。

`vae_diagnostic.py` 用每个城市预先固定的第一张验证/测试图片做三列对照：原图、上一轮 DINOv3 对象瓶颈后的 RGB 探针、预训练 VAE 对原图的重建。

**VAE 直接看到原图，因此这个对照没有检验 DINOv3 条件生成，也不能单独估计换解码器的收益。** 它检查图像编解码路径在当前街景上的恢复能力，为后续生成主干提供一个诊断起点。没有新的训练，也没有实例监督。

## 重现

隔离安装生成组件，复用环境中既有 torch/numpy/Pillow/scipy：

```bash
python -m venv --system-site-packages experiments/dinov3_generation_revision/.venv
experiments/dinov3_generation_revision/.venv/bin/python -m pip install \
  'diffusers==0.35.1' 'safetensors==0.6.2' 'huggingface_hub==0.35.0'
python experiments/dinov3_generation_revision/download_vae.py
experiments/dinov3_generation_revision/.venv/bin/python \
  experiments/dinov3_generation_revision/vae_diagnostic.py --device cpu
```

VAE 使用固定公开仓库提交的 safetensors 权重，对每个文件验证服务提供的 SHA256。推理采用 float32、潜变量后验 mode、448×448 原始输入，不在编码器与解码器之间额外缩放。默认关闭 tiling。旧探针输出 112×112，在图中上采样展示；同时记录 112 和 448 两个共同评价尺度。

结果保存到 `results/vae_control/comparison.png`、各城市的全尺寸 VAE PNG 与 `metrics.json`。指标仅对应这五张固定图片，不是完整验证集均值；两条路径输入信息不同。
