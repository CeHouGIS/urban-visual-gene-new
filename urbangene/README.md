# UrbanGene V0

用共享 latent 模板和有位数上限的离散场景程序，研究合成城市图像的组合表征。
当前规格是 [REQUIREMENTS.md](REQUIREMENTS.md)，训练安排见 [TRAINING_PLAN.md](TRAINING_PLAN.md)。

## 安装与最小运行

需要 Python3.11+、PyTorch、NumPy、Pillow、PyYAML、Matplotlib；没有联网下载模型的步骤。

```bash
cd /root/code/urban-visual-gene/urbangene
python -m pip install -e '.[test]'
urbangene generate --config configs/smoke.yaml
python scripts/run_smoke.py
urbangene evaluate --checkpoint outputs/smoke/urbangene/best.pt --split test_unseen_combo
python -m pytest -q
```

smoke 一条命令完成 tiny 合成数据、AE 准备、四种模型训练、评价和导图。
它不是可用模型，样本和训练量不足以判断研究假设。

## 分步训练

```bash
urbangene generate --config configs/v0_64genes.yaml
urbangene train-ae --config configs/v0_64genes.yaml --output outputs/manual/ae --device cuda:0
urbangene train --config configs/v0_64genes.yaml --ae outputs/manual/ae/best.pt --output outputs/manual/main --device cuda:0
urbangene evaluate --checkpoint outputs/manual/main/best.pt --split val_seen --device cuda:0
```

命令中的 GPU 只是示例，请先看实际负载；自动队列会检查资源。
`train` 的 `--method` 支持 `urbangene/b0/b1/b2`；`train-ae/train` 的 `--resume` 指向 `last.pt`。
同配置与种子可以复现 CPU 结果；CUDA grid_sample 反向可能有数值非确定性，不能保证逐位一致。

## 正式队列

```bash
urbangene plan --config configs/v0_64genes.yaml --output outputs/v0
urbangene queue --plan outputs/v0/plan.json
```

所有任务串行，有 AE 与主模型验证门。资源不足时写入状态后每45秒检查一次。
`--no-wait` 仅检查资源，无资源则退出。源代码改变后需重新生成并核对计划；避免用变化后的代码继续同名实验。
不要同时开两个队列：本地文件锁会阻止同一计划的重复 worker。
没有在 shell 中无条件串起全部实验，也不会失败后静默改变目标函数。

## 结构与编码

`src/urbangene/data`：确定性合成数据、held-out 因素组合与独立 seed。
`models`：轻量 AE、离散 Parser、共享 GeneLibrary 与固定渲染器。
`budget.py`：硬位数检查、固定长度二进制序列化与反序列化。
`train/evaluate/probe/schedule`：训练、评价、独立属性评价器和串行训练队列。

每张图只有 Gene ID、启用位、量化位置与尺度。Gumbel-Softmax 仅提供训练期直通梯度，前向仍为离散选择；保存的程序只有整数。
渲染器不接受目标图像或目标 latent；固定程序可独立复现输出。背景为零 latent，空 slot 不贡献内容。
Overlay 是固定权重归一化融合，顺序无关；不等同于带遮挡层次的 alpha painter。模板可被不同 slot 重复调用。

## 产物与已知限制

每次训练：`config.yaml`、`history.json`、`loss_curve.png`、`last.pt/best.pt`、`status.json`。
每次评价：`metrics.json/metrics.csv`、逐图结果、`recon_grid.png`、`failure_cases.png`、`gene_usage.png`、实例图、`programs.bin`。
实验根目录汇总 `rate_distortion.png`、`seen_unseen_comparison.png`；完整正式测试结束后有 `results.csv`。
属性指标要额外指定 `--probe`；未运行时明确标记 `probe_used=false`。

当前未验证：优于离散网格基线、组合泛化优势、高重复实例稳定性、独立可控因素、无目标组合生成、真实街景效果或最小 Gene 数。
固定 K/M 与固定码长不构成最优性证明。B1 预算较低，不能把它的性能差距直接归因于组合结构。
第一版只收集单训练种子；检查点选取只看 val_seen；测试打开前保存协议锁。
