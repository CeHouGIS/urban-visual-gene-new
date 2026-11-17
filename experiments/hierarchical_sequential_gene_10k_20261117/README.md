# 四层顺序视觉基因 AE：10k 首轮实验

日期：2026-11-17。初始运行目录独立，不覆盖旧 ISTA、V0/V1/V2 模型和检查点。代码、配置、日志、报告和新模型保存在此 NAS 目录。

## 已确定的架构

|层|格网|单元数|原生输出|原图窗口|共享字典条目|每单元索引数|
|---|---|---:|---|---|---:|---:|
|L0|1×1|1|32×32|256×256|64|4|
|L1|2×2|4|64×64|128×128|128|4|
|L2|4×4|16|128×128|64×64|128|4|
|L3|8×8|64|256×256|32×32|256|4|

共 85 个单元，576 种跨图共享字典条目，340 次调用。固定长度逐图索引预算为 2,632 bit = 329 byte，不含共享模型及容器开销。每个量化模板为 512 维，视作 32×4×4；同层字典供四步残差量化重复使用。

L0 在固定中性底色 0.5 上生成底稿。后续层先取得真实预测父图并放大，再编码原图目标减父图的剩余误差，在各自窗口中解码修正并加回。
编码端可看目标残差；解码端只使用索引和已解码父图。没有原图 skip、额外连续风格码、全图精修器或每单元 2×2 像素层。
局部解码器使用每个 patch 独立的 GroupNorm；父上下文只取对应 patch。bilinear 升采样的有限边缘传播明确纳入局部支持，不宣称没有插值边缘。

## Loss 与训练

- 逐层预训练：`MAE(current reconstruction, current target) + 0.25 * commitment`。
- 顺序为 L0→L1→L2→L3。训练当前层时，冻结父层全部参数与码本 EMA，不使用 GT 父图。
- 字典采用分布式汇总统计后的 EMA 更新，每个 micro batch 进行一次；衰减 0.99。每 100 次 EMA 更新检查低使用码并重置，记录重置数；评估、冻结阶段禁止更新和重置。
- 各层最多 1,200 更新，至少 400 更新；每 100 更新验证，首个验证在第 25 更新。改善门槛为 MAE 0.0001，连续 4 次不改善且达到最短训练量后停止当前阶段。下一层继承该父层的最佳验证模型。
- 保存完整逐层模型后，以十分之一学习率联合微调，最多 1,200 更新。联合目标为四层等权 MAE＋四层等权 commitment，始终顺序前向。
- 主学习率 0.0002，联合学习率 0.00002；全局 batch256，每卡128，micro batch32，梯度累积4次；两张 GPU，BF16。
- 全程随机初始化。LPIPS 只作为最终评价指标，不参与训练。

固定容量不保证最少原子数或唯一语义分解。“氛围、布局、外观”目前是待检验的层级功能，不是训练标签。

## 数据

复用原真实街景缓存 `/workspace/urban-visual-gene/multiscale_gene_10k_20261116/cache`，不重复复制 NAS 已有图像或检查源图哈希。
本目录保留对应的 10,000 train / 1,000 validation / 1,000 test 清单。图片、全景及格网在 split 间隔离；同城可跨 split。
每个训练全局 batch 按既有 grid-balanced 顺序采样，避免同格网邻近视角挤占同一个 batch。
现有 test 曾被查看，作为诊断集；其中 120 张来自训练集未出现的城市，单独报告，不称为新外部留出集。

## 队列与输出

队列使用项目共享 GPU flock，仅租用两张空闲卡；没有 GPU 则等待，不做 CPU 模型训练或推理。数据与采样协议的非模型检查可在 CPU 执行。

顺序门禁：`test_protocol.py` → 两卡 `test_gpu.py` → 两卡 Smoke → 正式逐层训练 → 训练后诊断。
Smoke 使用与正式训练独立的 `runs/smoke`，只检查实际数据上的整个训练/重放/保存流程，不作为模型质量结论。

- `queue_status.json`：当前队列、GPU、工作进程。
- `reports/cpu_protocol.json`、`reports/gpu_tests.json`、`reports/distributed_smoke.json`：门禁结果。
- `runs/main/progress.json`：当前层级和更新数。
- `runs/main/history.jsonl`、`loss_curves.png`：训练与验证曲线。
- `runs/main/best_L0.pt`…`best_L3.pt`：逐层最佳模型。
- `runs/main/checkpoint_stagewise.pt`：联合微调前模型。
- `runs/main/best_joint.pt`、`checkpoint_last.pt`：联合阶段及断点。
- `runs/main/preview_*.png/.npz`：原图、逐层前缀、绝对误差和有符号修正，另存原始数组与索引。
- `runs/main/test_metrics.json`：各层 MAE、PSNR、SSIM、码本使用、修正改善比例；最终层另报告 LPIPS。
- `reports/final_diagnostics/`：10 张测试图的逐层重建；曝光、白平衡、几何、模糊变化的编码替换响应；同 gene 在跨图/城市 crop 中的调用样例；逐层模型与联合模型验证比较。

绝对误差图黑色为 0、红色为 0.2 及以上；修正图灰色为 0，±0.25 对应黑白。中间输出不裁剪；MAE 和训练损失使用原始预测，PSNR/SSIM/LPIPS 使用 [0,1] 裁剪预测，并报告越界比例。

暂停本次队列：在本目录创建 `STOP_REQUESTED`。worker 在更新边界保存断点后退出，队列释放租约。继续时移除本实验的停止标记，再使用既有配置启动 `source/supervisor.py --config configs/experiment.json`；配置严格一致才允许恢复。

CPU 协议与 GPU 门禁检查的是顺序、加性、实际父图、父层冻结、索引重放、局部支持、EMA 跨 rank 一致和 DDP 梯度等价，不把这些当作语义分工已成功的证据。

## GitHub 分支快照

本实验位于 `hierarchical-sequential`；原 `main` 的旧版代码保存在 `additive`。
本目录的 `source/` 和原始配置与 NAS 运行版本逐字节一致，来源见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。
`configs/DESIGN_SNAPSHOT.md` 保留实施前的历史文本，其中“尚未实现”描述的是方案形成时的状态；实现入口和训练协议以本 README、代码及配置为准。

原运行通过的 CPU 协议、12 项两卡 GPU 检查和完整训练阶段 smoke 记录收在 `evidence/`，不会被监督器误当成本次复现的通过标记。
这里的 smoke 记录不是最终重建质量报告。正式训练的动态状态、图像、权重、缓存和完整图片清单保留在 NAS，不在此分支中。
`continue_analysis.py` 是原队列版本更新时使用的一次性衔接脚本；新运行直接使用当前 `supervisor.py` 即可。

配置保留运行时的绝对路径。复现前建立独立运行目录，复制 `source/`、`configs/`，准备真实清单与 RGB 缓存，更新路径，再执行 CPU 协议检查与监督器。
详细步骤见仓库 [复现说明](../../docs/REPRODUCIBILITY.md)，不要将克隆后的配置直接指向正在运行的 NAS 实验。
