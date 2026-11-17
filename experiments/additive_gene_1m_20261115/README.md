# Urban Visual Gene：百万张真实街景、GHSL 格网采样实验

本实验承接 10K 加性视觉基因实验，使用独立目录、独立数据清单和独立检查点。目标是扩大训练集，并减少少数城市或密集拍摄点主导训练的现象。

## 1. 已确认的采样规则

- 使用 GHSL UCDB R2024A V1.2 的 **2025 城市中心边界**，不是行政区边界。
- 在 World Mollweide 等面积投影（ESRI:54009）上建立 1,000 m × 1,000 m 方格，再与城市边界相交。这里的边长是投影坐标中的长度；地表实际方向长度有投影形变。
- 只纳入边界内有至少 5 个不同物理位置、且每点 0°/90°/180°/270° 四张照片均在已有 QC keep 清单中的格网。
- 全球 panorama ID 去重；经纬度四舍五入至 1e-5 度后，对相同位置保留最新记录，避免历史照片充当不同采样点。
- 每个格网先按固定随机种子选一点，再通过最远点策略补足 4 点，使 5 点尽量分散。每点保留全部四张照片，每格 20 张。
- 按洲 → 国家 → GHSL 城市逐级等额分配；某一级容量不足时，将余额分给其他还有容量的成员。因此“尽量均衡”不等于最终各洲数量相同。
- 跨城市边界的全球方格由合格候选点最多的城市拥有；并列取最小 GHSL ID。同一全球方格不会被重复计数。

## 2. 数据量与隔离

已有质量清单扩展后共有 50,726 个合格格网，覆盖 6 大洲、57 个国家、489 个 GHSL 城市。该数量是候选池统计；最终选中城市数见 `data/READY.json`。

| 集合 | 格网 | 采样点 | 照片 |
|---|---:|---:|---:|
| 训练 | 50,000 | 250,000 | 1,000,000 |
| 验证 | 250 | 1,250 | 5,000 |
| 测试 | 250 | 1,250 | 5,000 |

验证和测试的大小是本次实施选择。整格网分配到单个集合，5 点和 4 朝向全部留在该集合。旧数据集的 4,096 个受保护 panorama ID 不参与新采样。不同集合仍可能存在相邻格网、同一街道和视觉相似内容；这是未见格网上的重建评估，不是未见城市泛化评估。

没有合成图、没有重复格网补数。没有重新计算原始图像或特征文件的哈希。已有 QC 记录是质量依据，实际图片在预处理和训练读入时解码；解码失败会报告错误，不静默替换。

## 3. 模型与训练

主模型 `sparse_n256_s4`：RGB 256×256 → 从头训练的 residual CNN encoder → 16×16 空间位置上的 256 个非负激活，每位置保留 Top4 → 256×64 归一化共享字典相加 → 非线性 residual RGB decoder。没有跳过字典的旁路。

\[
z(x)_{h,w}=\sum_{j=1}^{256}a_j(x)_{h,w}\,d_j,\qquad
\hat x=D(z(x)),\quad a_j\geq0,\quad \|a(x)_{h,w}\|_0\leq4.
\]

这是特征空间的可加分解。非线性 decoder 不保证单独解码基因得到的 RGB 图像也能直接相加，更不保证两幅完整街景编码相加就能得到自然融合的新街景；组合泛化需要后续独立实验。

对照 `dense_ae`：相同 encoder/decoder，通过稠密瓶颈重建，不经过稀疏字典。两模型分别从 seed 42 随机初始化，不续训旧模型。

- 有效 batch size 256，microbatch 32，梯度累积 8 次。
- 39,063 次更新，约 10 次完整数据遍历。
- AdamW，学习率 2e-4，warmup 1,000 次，cosine 衰减至 2e-5。
- 损失：L1 + 0.1 × LPIPS。
- 第 250 次更新首次验证，之后每 1,000 次验证；每 500 次保存可恢复检查点。
- 以验证集选择最优权重，训练完成后仅对该权重做一次最终测试。
- 指标：L1、MSE、PSNR、SSIM、LPIPS、基因使用数和激活分布。

## 4. 执行与资源

`source/bootstrap.py` 先完成新空间清单和验证/测试 RGB 缓存，之后启动训练监督器和单个训练缓存写入进程。

`source/supervisor.py` 使用共享 GPU lease，并检查显存、利用率及 compute processes；最多两张物理 GPU，每模型一张。无空闲 GPU 时排队，不进行 CPU 模型训练或推理。每个模型先通过 GPU 单元测试、4 图 20 步 smoke test 和 microbatch 32 梯度检查，再从头正式训练。

CPU 负责 GIS、采样、照片解码和绘图。机械盘缓存位于 `/workspace/urban-visual-gene/additive_gene_1m_20261115/cache`，完整训练 RGB 数组约 196.6 GB，另加约 2 GB 的验证/测试缓存。后台单写者先写 RGB、flush，再发布 ready 标记；训练只读已发布的行，未缓存的行从 NAS 原图解码。

所有代码、配置、采样清单、GIS 输出、日志、图表与模型检查点都在本 NAS 实验目录。原始 NAS 照片直接引用，未复制。

## 5. 结果入口

- `data/READY.json`：只有清单、精确计数和验证/测试缓存完成后才写入。
- `data/images.{train,validation,test}.jsonl`：照片路径、坐标、GHSL 城市、格网和朝向。
- `data/selected_grids.parquet`：裁剪后的选中格网，含 CRS。
- `data/sampled_points.parquet`：5 点/格的最终采样点。
- `data/distribution_by_{continent,country,city}.csv`：逐级分布。
- `data/spatial_candidate_index.json`：候选池统计和来源。
- `data/source_city_audit.json`：来源城市坐标匹配记录。
- `data/cache_progress.json`：机械盘 RGB 缓存进度。
- `launch_status.json` / `queue_status.json`：准备、排队、GPU 检查和训练状态。
- `runs/{variant}/progress.json`、`history.jsonl`、`validation_history.jsonl`：训练状态和曲线数据。
- `runs/{variant}/checkpoint_{resume,best}.pt`、`previews/`：权重和重建预览。
- `reports/preflight_spatial_tests.xml`：CPU 数据与调度检查报告（19 项通过；不含模型前向训练）。

## 6. 复现

使用 `configs/experiment.json` 中的 Python 路径运行：

```bash
python source/bootstrap.py --config configs/experiment.json
```

进程锁阻止同一实验重复启动。运行前为 `TMPDIR` 指定机械盘临时目录。停止本实验可在本目录创建 `STOP_REQUESTED`，不会修改其他实验的队列。

相关源文件：`spatial_catalog.py` 建立完整四朝向 QC 点清单；`spatial_sampling.py` 执行 GHSL 边界匹配、格网抽样与集合划分；`spatial_balance.py` 执行容量约束下的逐级配额。
