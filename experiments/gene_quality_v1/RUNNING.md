# 当前实际训练

以 runtime.json、runs/*/status.json、latest.json 为实时依据。本说明不代替运行状态。

## 当前任务

- smoke1024_DV_seed17：32张训练图、200个optimizer updates、batch16、SDXL1024、DINOv3+VAE。每25步保存检查点，每50步运行30步纯噪声采样（2训练图+2验证图，原/零/打乱条件）。
- GPU2训练，GPU3制作剩余训练与验证集VAE缓存。不会终止其他人的GPU作业。
- 512版已停止，保留25步检查点与对照；不用于正式比较。

## 文件

- 训练日志：smoke1024.log；进度：runs/smoke1024_DV_seed17/status.json。
- 权重：runs/smoke1024_DV_seed17/checkpoint_*.pt（仅适配器、条件器与优化器）。
- 样本：runs/smoke1024_DV_seed17/samples_*/panel.png 与 metrics.json。
- 来源与配置变更：EXECUTION.md；设计：EXPERIMENT.md；当前配置：config.json。

## 后续正式阶段（尚未运行）

先检查200步纯噪声结果；不能仅凭去噪损失宣布Dense通过。

训练入口train.py：--smoke-images 0 --updates 6000 --mode D/V/DV --resolution 1024，各组从相同种子和原始主干初始化。
评估入口evaluate.py：固定64验证图、seeds101/211、30步，按输入图平均后配对bootstrap；测试需显式锁定协议且另行缓存。
蒸馏入口distill_gene.py：必须传入正式密集teacher检查点，分别random_s8/learned_s8/learned_s4。
联合训练train.py：--teacher共同Dense检查点；Gene额外--gene及--distilled-gene，自动冻结输入融合；Dense+只更新同一起点的适配器。

这些后续脚本已实现但尚未接受正式数据端到端运行验证；不能把代码存在视为实验完成。当前没有自动启动全量62k训练的队列；小样本质量检查后再决定进入下一阶段或修正条件路径。
