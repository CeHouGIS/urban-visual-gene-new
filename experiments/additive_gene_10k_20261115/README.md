# UrbanGene：10,000 张真实街景的共享可加基因实验

建立时间：2026-11-15（北京时间）。此目录为独立实验；旧队列、源码和检查点不修改。

## 要检验什么

从真实街景集合共同学习共享字典 d_k，每张图片有自己的空间系数 A_i(k,u,v)。把基因贡献相加，再由非线性解码器重建 RGB：

\[
\hat I_i=D\left(\sum_k d_k\otimes A_i(k,:,: )\right).
\]

基因不预设类别或实例含义。求和发生在特征空间；自然图像重建本身不证明语义独立或任意组合生成有效。

## 数据与模型

- 10,000 张真实 Google Street View 训练照片，10,000 个全景，覆盖原清单的 296 座城市，城市数量由 data/READY.json 的实际记录确认。
- 验证 1,024 张、测试 1,024 张；来源于原有受保护的验证照片池，三份清单的全景互不相交。这里只保证已知全景隔离，未声称做过新的地理邻近去重。
- 原始照片留在 NAS。256×256 RGB 缓存在服务器本地机械硬盘，可从 NAS 清单重新生成。未重新校验/哈希原图片和特征文件。
- 无类别、框、实例 Mask、融合目标图或教师生成图。
- 普通 RGB AE 与共享稀疏字典 AE 使用同样的编码器/解码器主干；从头训练，所有权重参与图像重建优化。
- 编码器 4 次下采样，latent 为 64×16×16；解码器为非线性残差上采样网络，无输入跳连。
- 字典 AE 使用 256 个单位范数字典向量，非负激活，每个位置最多 4 个非零系数。没有额外残差路径绕过字典。

## 首轮两组

| 名称 | 目的 |
|---|---|
| dense_ae | 检查普通 RGB 重建能力 |
| sparse_n256_s4 | 检查共享稀疏可加分解的重建与跨图像复用 |

两组每组 6,000 次更新、有效 batch64、microbatch16、AdamW，学习率 2e-4，200 步 warmup 后余弦衰减至 2e-5。每组共 384,000 次训练图像曝光，约 38.4 遍训练集。损失为 L1 + 0.1 LPIPS（现有 AlexNet 感知权重，固定不训练）。

共享主干的初始化、图片顺序和优化预算对齐；字典头额外参数量单独报告。两组编码容量不同，不能以这一轮单独证明字典方法在相同比特预算下更好。后续容量曲线、随机字典与无稀疏消融根据首轮结果安排。

## 执行与检查

先完成轻量数据/队列单元测试与静态语法检查；模型单元测试和真实 4 图、20 步学习 smoke 在已租用的 GPU 上执行，通过才进入正式训练。smoke 权重丢弃，正式训练重新按相同种子初始化。

一组模型一张空闲 GPU，两组最多并行两张物理卡；只有一张可用则先跑一组。每次要求无计算进程、显存占用 <1 GiB、利用率 <10%，同时取得与旧项目相同的 GPU 协作锁。没有空闲 GPU 则等待，禁止 CPU 模型训练或推理。

启动入口：source/supervisor.py --config configs/experiment.json。队列每 30 秒更新 queue_status.json。已运行的队列由 queue.lock 保证不重复启动。

停止方式：在本实验根目录创建 STOP_REQUESTED。训练完成当前更新后保存恢复检查点并退出；队列释放其锁。也可对本实验 supervisor 发送 SIGTERM。不要改动历史实验的停止标记。

每 250 步在完整验证集评价 PSNR/SSIM/LPIPS，并保存恢复检查点、最佳检查点、loss 曲线和固定预览。仅根据验证集目标选择最佳检查点，完成后在独立测试集评价一次。

## 输出位置

- queue_status.json：排队或训练的实际状态、GPU 和进程。
- data/READY.json 与 images.*.jsonl：来源、划分和预处理规则。
- reports/preflight_tests.xml：不执行模型计算的测试结果。
- reports/*.gpu_tests.xml、*.gpu_smoke.json：GPU 测试和学习检查。
- runs/<组名>/progress.json、history.jsonl、validation_history.jsonl：训练进度与指标。
- runs/<组名>/loss_curves.png：损失与验证指标曲线。
- runs/<组名>/previews/：原图、重建、误差；字典模型另外输出基因激活、基因删除效果和实际稀疏代码。
- runs/<组名>/checkpoint_resume.pt：模型、优化器和随机状态；checkpoint_best.pt：验证最优模型。
- runs/<组名>/test_metrics.json：验证选优后的最终测试结果。
- reports/comparison_curves.png、comparison_metrics.json、test_reconstruction_comparison.png：两组完成后自动汇总。

输出图中的激活不是实例分割。删除实验用于检查基因的视觉作用，未以删除后仍重建原整图的目标训练。

研究方案见原项目 source/docs/DATASET_SHARED_ADDITIVE_GENE_EXPERIMENT_20261115.md。训练耗时与完成时间需等 GPU 实测吞吐后估计。
