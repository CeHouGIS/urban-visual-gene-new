# 复现与运行范围

## 当前 ISTA 实验的入口

目录：`experiments/residual_ista_gene_10k_scratch_20261116/`。

|文件|功能|
|---|---|
|`source/model.py`|金字塔编码器、归一化字典、6 步非负 ISTA、多尺度 RGB 解码器|
|`source/losses.py`|RGB、LPIPS、特征稀疏编码目标与残差金字塔监督|
|`source/data.py`|读取已有 RGB 缓存，格网采样及 DDP 数据划分|
|`source/train.py`|两卡 DDP、smoke、训练与评估|
|`source/supervisor.py`|GPU 租约、检查、smoke 和独立实验队列|
|`source/inspect_five_street_expressions.py`|严格载入 step 6000 检查点，导出五张图实际表达|
|`source/plot_all_gene_responses_five_streets.py`|512 个 gene 的响应数与表达强度|
|`source/export_all_gene_cards.py`|逐 gene PNG、64 页 PDF、离线 HTML 与完整权重数值导出|

## 外部资源

本仓库提供源代码快照，不包含训练数据和检查点。原始实验运行目录与 GitHub 克隆目录是两个独立位置。
历史配置的 `root`、`python`、`cache`、`temporary_directory`、`shared_gpu_leases` 及数据来源路径需要按环境修改。

ISTA 的 `source/data.py` 使用与脚本相邻的实验目录下 `data/images.train.jsonl`；训练、验证、测试的
`data/images.{train,validation,test}.jsonl` 和已有缓存中的 `{train,validation,test}.rgb.npy` 必须与历史顺序一致。
缓存形状为 NHWC uint8 RGB、256×256；分别 10,000/1,000/1,000 张。`data/READY.json` 是数据准备完成标记，不能在缺少数据时伪造。
原始采样和构建百万级清单的代码见 `experiments/additive_gene_1m_20261115/source/`，依赖外部街景源和 GHSL 边界资源。

如使用已有数据在新位置运行，先建立独立实验目录并复制 source/configs，按真实资源更新配置，
再把匹配的清单放进该目录的 `data/`。创建 `logs/`、`reports/`、`runs/` 和临时目录。
监督器用源码相对位置读取清单，所以仅修改 JSON 中的 `root` 不足以改变所有相对资源路径。

不要把 GitHub 克隆目录的配置直接指向已有完成实验来启动；新实验使用新的 root 和独立输出目录。
复用历史检查点的严格恢复逻辑要求配置兼容；随意修改路径后加载旧状态可能被恢复兼容检查拒绝。

## CPU 允许的检查

代码可用 Python `compile(source, filename, 'exec')` 检查语法，避免生成缓存。
`source/tests/test_data_schedule.py` 中只有配置/学习率用例不依赖外部清单：

```bash
python -m pytest -q -o addopts= -p no:cacheprovider \
  experiments/residual_ista_gene_10k_scratch_20261116/source/tests/test_data_schedule.py::test_exact_updates_learning_rate_and_resume_protocol
```

完整采样测试需要原始清单；GPU 模型测试需要 GPU，不能把测试跳过理解为通过。
本次上传仅进行源文件校验与不含模型推理的 CPU 协议检查，不重新训练模型。

## GPU 检查、smoke 和训练

配置及真实数据准备完成后，原监督器的入口是：

```bash
python /absolute/new_experiment/source/supervisor.py \
  --config /absolute/new_experiment/configs/experiment.json
```

监督器检查 GPU 空闲状态并持有共享 flock 租约，先 GPU 单测，再两卡 smoke，再正式训练；没有 GPU 则等待。
默认两个 variant 各两张卡，上限四张物理卡；改变 GPU 预算前核对该版本的 `common.py` 协议。
直接运行 torchrun 不会自动申请 GPU 租约，不应绕过共享服务器的调度。

## 结果的适用范围

仓库内 `evidence/` 是原实验的历史记录，并非这次上传重新完成的训练或测试。
当前 ISTA 测试指标与 cap64 功能复用指标来自不同架构和评估集，应分别引用。
字典特征的可加性由线性合成保证；不能把它等同于独立 RGB gene 图像相加，或自动等同于语义实例分解。
