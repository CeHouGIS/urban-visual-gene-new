# 百万张 GHSL 街景：四卡、batch 1024 续训

用户于 2026-11-15 授权本轮直接使用四张 GPU，并扩大 batch。本实验使用两张 L20 训练 `sparse_n256_s4` 主模型，另外两张训练 `dense_ae` 对照。每模型使用 NCCL DistributedDataParallel（DDP）同步梯度。

| 参数 | 原阶段 | 四卡续训阶段（每个模型） |
|---|---:|---:|
| GPU 数 | 1 | 2 |
| 每卡 microbatch | 32 | 128 |
| 梯度累积次数 | 8 | 4 |
| 有效全局 batch | 256 | 1024 |
| 全部模型合计 GPU | 2 | 4 |

原阶段已优雅停止并保存最后一步：主模型 9,938 步（2,544,128 张次），对照 10,188 步（2,608,128 张次）。模型权重、AdamW moment buffers、累计基因使用量与随机状态均从最后恢复检查点继续，不从头重训。

原数据与实验位于：
`/host/mnt/nas/urban-visual-gene/additive_gene_1m_20261115`

本目录 `data` 为原百万张采样目录的引用，机械盘 RGB 缓存已全部准备完成并直接复用。采样不变：50,000 个 1 km 投影格网、每格 5 点、每点全部 4 张照片，共 1,000,000 张训练图；验证、测试各 5,000 张。原代码、日志、配置和检查点保留，停止时权重额外快照于 `origin_checkpoints/`。

## 训练预算与续接

- 两个阶段累计各训练 10,000,128 张次，约 10 遍训练集；扩大 batch 后剩余更新次数相应减少。
- 主模型剩余 7,282 次更新，对照剩余 7,219 次更新，最后一批按剩余图片数处理。
- 确定性采样器以累计已见图片数恢复 shuffle 流；两个 rank 分到不同图片，不各自重复同一个 batch。
- 保留原学习率范围 2e-4 → 2e-5，按累计已见图片数延续原 cosine schedule。没有重新 warmup，也没有将学习率乘四。
- 每 250 次新更新验证，相当于每 256,000 张次；每 125 次新更新保存恢复检查点，相当于 128,000 张次。续训第 50 次更新先验证一次。
- 验证/测试按 rank 划分，不填充重复图片，指标按全体实际图片数求平均。
- 原阶段最优验证检查点仍有资格成为最终最优模型；测试集只在完整训练结束并选定检查点后评估一次。

## 启动门槛

监督器先获取两张空闲 GPU 的共享 lease，不够一对时释放并等待。最多四张卡，不占用已有 compute process 的卡。没有 CPU 模型训练/推理 fallback。

每个模型正式续训前都执行双卡 GPU Smoke Test：

1. 双卡梯度累积的梯度与相同全局 batch 的单进程 GPU 计算比较。
2. 四张真实照片上进行 20 步学习，要求损失下降。
3. 实际运行完整 1,024 张全局 batch 的前向、反向和优化器更新，检查有限梯度。
4. 检查两个 rank 的模型参数一致，并记录每张卡峰值显存。

已有 24 项 CPU 数据/调度检查通过，报告见 `reports/preflight_tests.xml`。GPU 结果见 `reports/{variant}.distributed_smoke.json`。

## 文件入口

- `configs/experiment.json`：实际配置。
- `queue_status.json`：模型进程、GPU 列表及当前阶段。
- `runs/{variant}/progress.json`：`step` 为本次续训更新数，`images_seen` 包括原阶段。
- `runs/{variant}/history.jsonl`、`validation_history.jsonl`：续训曲线记录。
- `runs/{variant}/checkpoint_resume.pt`：四卡阶段可恢复权重及各 rank 的随机状态。
- `runs/{variant}/checkpoint_best.pt`：本次续训优于原阶段最佳验证指标时保存。
- `runs/{variant}/previews/`：重建、基因激活和移除图。
- `source/train_distributed.py`、`source/distributed_schedule.py`、`source/supervisor_distributed.py`：续训实现。

`source/train.py` 等文件是原单卡实现的保留依赖，不作为本轮入口。启动本轮：

```bash
python source/supervisor_distributed.py --config configs/experiment.json
```

每个实验有独立进程锁。创建本目录的 `STOP_REQUESTED` 后，两 rank 在同步边界保存并退出。需要恢复时，确认上一进程退出、移除该标志，然后重启监督器。
