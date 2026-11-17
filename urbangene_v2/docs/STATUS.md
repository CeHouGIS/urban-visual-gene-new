# UrbanGene V2 Status

Updated 2026-11-15（北京时间）。

## 当前：10k 共享可加基因 RGB 实验已完成

用户授权按共享稀疏字典 AE 方案启动 10,000 张真实街景实验。新根目录为
`/host/mnt/nas/urban-visual-gene/additive_gene_10k_20261115`。
训练/验证/测试分别为 10,000/1,024/1,024 张，296 城，互斥全景，RGB 缓存已准备。
两组为普通 RGB AE 和 n=256、每位置最多 4 个激活的共享字典 AE；每组 6,000 更新、有效 batch64。
8 项轻量预检和 4 项 GPU 模型测试通过，普通 AE 的 4 图 20 步 GPU 学习 smoke 通过。
主模型的独立 GPU 学习 smoke 在其正式训练前自动执行；所有模型计算只使用空闲 GPU。
2026-11-15 03:02（北京时间）两组均已完成 6,000 次正式训练更新及各 1,024 张测试图评价，队列状态为 `complete`。
普通 AE：测试平均逐图 PSNR 26.31 dB、SSIM 0.7454、LPIPS 0.0920。
共享字典 AE：PSNR 23.19 dB、SSIM 0.6056、LPIPS 0.1752；测试集激活 140/256 个基因。
基因模型保留大体布局，但细节质量弱于普通 AE。当前不能据此认定基因已实现稳定解耦或任意组合生成。
报告为新根目录的 `reports/RESULTS_20261115.md`，对照图为 `reports/original_dense_sparse_first4.png`。
最新状态以新根目录的 `queue_status.json` 为准；当前无本实验运行中的训练进程。
每组一张 GPU、总共最多两张；单卡可用时顺序推进，禁止 CPU fallback。
代码、配置、日志和检查点均在新 NAS 目录，原百万张任务继续保持用户要求的停止状态。
详情见新目录 `README.md`；旧实验停止记录见
`outputs/v2/m1_real1m_bs128_20261110/STOP_ALL_20261112.json`。

以下是历史状态，不能用于推断当前训练进度。

## 历史：20k 已停止，百万张新实验启动

2026-11-10 16:33（北京时间）：用户要求停止当前实验并训练新实验。
20k 轮在第 27 轮进行中、step 133,500 停止，原检查点保留，
`m1_real20k_20261110/STOPPED_BY_USER.json` 已记录。

独立新 run：`outputs/v2/m1_real1m_bs128_20261110`；队列 PID 713590。
目标为 296 城、1,000,000 张训练照片、4,096 张验证照片、有效 batch 128。
已完成 8 项数据读取、累积梯度、调度、GPU 排队及特征身份检查测试。
两张 L20（物理 GPU 1、2）的 batch-128 smoke 已通过，完成 3 次更新，损失及梯度有限。
当前两张 GPU 正在提取 DINOv3 特征；训练照片已核验超过 100,000 张、验证照片 4,096 张。
待百万张全量数据及分片核验后，自动从头训练 100,000 更新。
不回退 CPU 模型计算；原实验不会恢复运行。

执行源码冻结为 `source_real1m_bs128_20261110`，源文件 SHA256 在新 run 的
`source_snapshot.json` 中；方案见 `docs/REAL1M_PROTOCOL_20261110.md`。
实时状态以新 run 的 `execution.json`、`gpu_batch_smoke.json`、
`extraction_worker_*.json`，以及数据目录的 `preparation.json` 为准。
目前不宣称已完成百万张训练，也没有新模型的科学效果结论。

以下为历史阶段记录，其启动状态已由以上新流程替代。

## 2026-11-10：20,000 张扩大实验已启动

**最新用户要求：只用 GPU，无卡则排队。** 已停止本轮 CPU 特征计算，保留完成的
2,560 张训练图特征。GPU-only 队列 PID 433135；状态为 `waiting_for_gpu` 时仅轻量
轮询，无模型计算子进程。特征提取、训练及最终模型推理均无 CPU fallback，最多两张卡。
修订记录见新 run 的 `gpu_only_policy_20261110/policy.json`。

用户要求检验训练不充分的可能性，已完成新的 20,000 张真实训练照片、512 张验证
照片核验。四城各 5,000 张训练图；训练来自 10,000 个全景、各两个相反视角。
训练/验证全景重叠和图像内容 hash 重叠均为零；原验证 64 张保留用于前后比较。

沿用原架构、MSE 目标、batch 4、学习率与种子，从头训练 30 轮（150,000 更新），
在第 1/5/10/20/30 轮输出固定图像分组预览。当前启动阶段为特征提取，
不能称为已经完成 20k 主模型训练。最新状态以新 run 的 `execution.json` 为准。

原 CPU 流程 PID 432375 已停止；新结果 `outputs/v2/m1_real20k_20261110`；
数据 `outputs/v2/data/real20k_20261110`。执行源代码冻结在同实验根的
`source_real20k_20261110`。27 项测试和真实特征尺寸的 CPU 续训 smoke 通过。
GPU 全有计算任务时排队，每 30 秒重新检查，空闲后自动接续到最多两张 GPU。
方案见 `docs/REAL20K_PROTOCOL_20261110.md`。
本轮同时增加数据量、视角数和更新预算，不单独归因；无真实实例标注，不报告实例准确率。

## 当前主线：真实街景（2026-11-09 数据方向修正）

用户指出此前展示的是合成街景，因此主线已切换到 NAS 的真实 Google 街景。
512 张训练照片、64 张验证照片均已核对源文件 SHA256、可读性和全景级隔离。
新的 CPU 流程已完成：重新提取全部 576 张的 DINOv3 特征 → 从头训练 10 轮、
1280 次更新 → 完成 64 张验证照片的无标签诊断和逐步可视化。
完成时间为 2026-11-09 17:15（北京时间）。该 512 张首轮已结束；20k 新流程另行推进。
特征重建误差 0.037830，比训练集空间均值基线 0.039362 低 3.89%，
但编码器最大 Slot 占平均 86.19% 的注意力质量，解码器软权重接近均匀。
当前输出未达到逐棵树分离目标；没有独立真实实例标注，不能报告实例准确率。
真实报告为新 run 的 `validation/REPORT.md`；简图为
`validation/real_street_result_simple.png`。尚未开展本轮 Gene 组合或 RGB 生成。
实时状态见 `outputs/v2/m1_real_streets_cpu_20261109_execution.json`；
训练阶段的 epoch/step 见新 run 内 `progress.json`。
方案见 `docs/REAL_STREET_PROTOCOL_20261109.md`。

下文两轮 256/64 图、十棵树、FG-ARI 与 IoU 结果均属于**合成图诊断**，
不能作为真实街景效果的证据。真实照片没有独立实例标注，实例发现能力尚待评价。
新流程使用独立 NAS 目录，原队列和检查点保留。

## M0

- [x] Separate NAS-backed source/output root created.
- [x] V1.1 and V1.2 artifacts inspected read-only.
- [x] Existing dense feature cache verified as readable 1024-D DINOv3-derived
  features; it is a feature cache, not a model checkpoint.
- [x] New reproducible local extraction pipeline implemented because no
  reusable extraction script with verified provenance was found.
- [x] Download and checksum DINOv3 ViT-L/16 from ModelScope repository
  `facebook/dinov3-vitl16-pretrain-lvd1689m`, pinned revision
  `99b6205b0321c4d9726dd072d8c6a80aa6be72df`. All seven files match the
  ModelScope SHA256 records. Weights are in `models/dinov3-vitl16` (1,212,559,808
  bytes of safetensors); Transformers loaded all parameters on CPU with no
  missing, unexpected, or mismatched keys. See `outputs/v2/dinov3_download.json`.
  Hugging Face remains unreachable; this public download never reads `.env`
  or sends an HF token. The ModelScope hashes were not independently compared
  against Hugging Face.

## M1

- [x] Local pretrained DINOv3 loader and extraction API implemented.
- [x] Adaptive Spatial Slot Discovery and feature reconstruction decoder
  implemented.
- [x] Image-only/feature-only training datasets and eval-only mask metrics
  implemented.
- [x] Focused unit tests: 24 passed, including two-GPU cap, fail-before-output,
  decoder output/gradient equivalence, mask decoding, bootstrap comparison,
  unmatched-instance stability, and spatial initialization checks.
- [x] CPU engineering smoke: 2 update steps on an existing real-image DINOv3
  feature cache; reconstruction MSE decreased from 0.01332 to 0.01301.
- [x] Repeated the CPU smoke after adding the GPU selection path; the trainer
  recorded CPU with an empty GPU list and completed without CUDA use.
- [x] Synthetic image-only manifests and separate evaluation annotations
  generated for train, validation, seen-combination, unseen-combination, and
  higher-complexity splits.
- [x] CPU integration with the downloaded checkpoint: two training images at
  448x448 produced finite float16 features of shape `[2,1024,28,28]` using the
  existing frozen backbone/extractor. Manifest IDs match and no labels were
  loaded. Results: `outputs/v2/dinov3_modelscope_cpu_smoke_20261109/summary.json`.
- [x] Added explicit two-GPU `DataParallel` support, an at-most-two-GPU
  validation guard, and a separate GPU training config. This path has not run
  on GPU because all visible GPUs are occupied.
- [x] Complete CPU extraction of 256 training and 64 validation images into
  `[N,1024,28,28]` float16 caches, with hashes and matching image IDs. All 64
  validation RGB/mask pairs were regenerated and matched. Train/validation
  image hash overlap is zero; no instance disappeared at the patch grid.
- [x] Initial M1 validation screen: 10 epochs, 640 updates, seed 42, CPU.
  Best fixed-RNG validation reconstruction MSE is 0.025412. Evaluation FG-ARI
  is 0.009723 and matched IoU is 0.015334, below the regular-grid diagnostic
  (FG-ARI 0.261881, IoU 0.056344). Scientific grouping gate **failed**.
- [x] Saved per-image tensors, all 32 soft masks, first/failure case montages,
  count distributions, and feature-level flip/translation/scale diagnostics.
- [x] Added a separately identified, reproducible 12-image ten-tree validation
  stress group. All ten trees remain visible in its 28x28 GT maps. Distinct
  tree recall at matched IoU >= 0.5 is zero in the initial model.
- [x] Identified collapse: encoder's largest slot has 95.69% mean attention
  mass; decoder entropy is 3.4522 versus log(32)=3.4657; all presence gates
  are open but mean nonempty foreground groups are only 2.78125.
- [x] Completed the bounded spatial-initialization ablation with the same
  data, loss, 10-epoch budget, and selection rule. Validation FG-ARI improved
  to 0.085687 and matched IoU to 0.040773, but remained below grid grouping;
  scientific gate **failed** again. Ten-tree distinct recall remains zero.
  Soft-mask entropy (3.45259) and center spread (0.01164) still show collapse.
- [x] Saved a Chinese experiment report at
  `docs/M1_EXPERIMENT_REPORT_20261109.md` and a combined comparison figure and
  JSON under `outputs/v2/m1_validation_comparison_20261109.*`.

## Compute Status

The V1.1 queue remains active (PID 399150). The GPU availability audit found
all four L20 GPUs occupied. V2 feature extraction, training, and validation
screens run on CPU with no selected GPU. Future V2 GPU runs are capped at two
GPUs and must record the devices used. All new artifacts are on NAS.

## Synthetic Diagnosis: Next Gate (archived direction)

Both bounded screens have finished. M1 is not passed, so M2-M4 remain
`not_run`. The next bounded diagnosis should test competitive/local decoding
and the geometry update, rather than assuming a dispersed initialization is
enough. A train-only spatial average baseline has validation MSE 0.017190,
better than either model, so reconstruction reduction alone is especially
unconvincing here. Formal test splits stay locked. No V2 training process is
left running after these screens; GPU training has not been launched.
