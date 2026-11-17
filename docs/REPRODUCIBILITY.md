# 当前四层顺序实验的复现

入口：`experiments/hierarchical_sequential_gene_10k_20261117/`。其源码与 NAS 运行版本一致，配置保留原服务器路径。
当前分支为 `hierarchical-sequential`，旧版原 `main` 已命名为 `additive`。下半部分的 ISTA 说明是历史对照。

1. 建立新的独立实验目录，将本实验的 `source/` 和 `configs/` 复制进去；创建 `data/`、`logs/`、`reports/`、`runs/` 和临时目录。
2. 修改 `configs/experiment.json` 中的 `root`、`python`、`cache`、`temporary_directory`、`shared_gpu_leases`。`root` 应是包含复制后 `source/` 的目录；监督器实际使用 `root/configs/experiment.json`。所有共用服务器任务必须使用相同 GPU 租约目录。
3. 准备 `data/images.{train,validation,test}.jsonl` 与顺序严格匹配的 `{train,validation,test}.rgb.npy` 缓存，分别 10,000/1,000/1,000 张 NHWC uint8 RGB、256×256。训练清单每格 20 张，即 5 个全景点 × 4 个朝向。清单至少包含 `id`、`pano`、`grid_id`、`country`、`city`。使用自己的真实数据，不从历史 `evidence/` 复制门禁标记。
4. 检查 split 间图片/全景/格网隔离。`data/READY.json` 至少需包含根据真实清单计算的 `unseen_city_test_indices`（相对于 test 清单的零起始行号，无该子集时为 `[]`）。原运行完整元信息在 NAS 的 `data/READY.json`；不能在数据缺失或未检查时伪造 ready 状态。
5. 使用配置里的 Python 运行 `source/test_protocol.py`（仅数据协议，无 CPU 神经执行）。通过后运行监督器，自动等待两张空闲 GPU，依次执行 GPU 检查、完整阶段 smoke、正式训练、训练后分析。

```bash
python /absolute/new_experiment/source/test_protocol.py
python /absolute/new_experiment/source/supervisor.py \
  --config /absolute/new_experiment/configs/experiment.json
```

两条命令中的 `python` 都需指向安装好依赖的同一解释器，并与配置中的 `python` 一致。
新监督器已包含 analysis 阶段，无需执行一次性 `continue_analysis.py`。没有 GPU 时排队，CPU 仅做数据/语法检查和绘图。
原运行通过记录放在 `evidence/`，真正启动门禁读取新运行目录的 `reports/`；两者用途不同。
不要将新运行配置指向正在训练的 NAS 目录。创建 `STOP_REQUESTED` 可请求当前实验保存并停止；恢复要求配置严格兼容。

---

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
