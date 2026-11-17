# 百万样本实验的本地机械硬盘缓存

用户要求将训练读取的数据迁移到服务器机械硬盘，并明确要求不校验特征文件。

## 数据与位置

- NAS 原始数据：`/host/mnt/nas/urban-visual-gene/urbangene_v2_20261109/outputs/v2/data/real1m_20261110`
- 本地缓存：`/workspace/urban-visual-gene/cache/real1m_20261110`
- 迁移记录：`/host/mnt/nas/urban-visual-gene/urbangene_v2_20261109/outputs/v2/m1_real1m_bs128_20261110/local_cache_migration_20261112`
- 训练与验证特征共 491 个 NPY 分片、1,612,208,731,520 字节，约 1.47 TiB。
- `/workspace` 是独立的本地 SATA 机械硬盘；启动前可用约 8.57 TiB。

## 复制方式

`tools/cache_real1m_local.py` 按原目录结构顺序直接复制分片，不计算特征内容哈希，不回读特征内容。原索引中已有的内容哈希仅作为数据集元数据保留，不代表此次重新校验。

每个分片先写入 `.copying` 文件，写完并刷盘后再改为正式文件名。仅记录长度、时间、inode 和完成状态，避免未完成的分片供训练读取。完成记录支持按分片恢复，异常中断最多重新复制当前分片。最后写入 `CACHE_COMPLETE.json`。NAS 原件不删除，不重建特征。

迁移脚本只为两个很小的 JSON 索引计算身份标识，用于保持原训练的断点恢复约束；不扫描 NPY 内容。复制期间当前训练继续读取 NAS。

## 自动切换

`tools/resume_real1m_local.py` 独立等待完整缓存的完成记录，之后等待当前训练再次保存 `checkpoint_resume.pt`。它核对旧训练进程的 PID、启动时间与命令，只停止登记的本项目父进程和训练子进程，不操作其他任务。

切换前将滚动检查点及执行记录另存为迁移目录中的 `before_local_*` 文件。旧进程完全退出后，重新取得原来的两张 GPU 租约，再恢复训练；两卡未空闲则等待，不使用 CPU 训练。

模型、优化器、学习率、batch、随机状态、样本顺序和原始配置均沿用已有检查点。训练仍导入 `source_real1m_bs128_20261110` 的冻结代码，仅由独立 I/O wrapper 将 `ShardedFeatures` 内存中的文件路径指向本地缓存。原始 JSON 索引与配置不改写，因此恢复时的数据身份保持一致。等待检查点和停止进程间，尚未写入检查点的少量计算可能在恢复后重复。

模型与结果继续写入原 NAS 实验目录，训练预览所需的少量原照片仍从原路径读取。此次未搬迁百万张原照片。

## 状态文件

- `status.json`：复制进度，包含 `copied_shards`、`copied_bytes`、当前文件复制字节数。
- `migration.log`：每个完成分片的进度日志。
- `handoff.json`：`waiting_for_copy` → `waiting_for_next_checkpoint` → `waiting_for_original_two_gpus` → `training_from_local_cache` → `complete`。
- `handoff.log`：切换进程日志。
- `local_training.log`：切换后训练输出。
- `CACHE_COMPLETE.json`：所有分片完成的记录，不表示内容校验通过。

## 验证

CPU 数据操作测试和原百万样本数据管线测试共 11 项通过。覆盖直接复制不哈希特征、分片恢复、半文件不发布、本地与 NAS 读取的特征值和采样顺序一致、原索引保持不变、GPU 忙碌时释放部分租约，以及两卡数量限制。没有启动额外模型训练或占用第三张 GPU。实际吞吐变化需在切换后观察。
