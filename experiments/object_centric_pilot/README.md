# DINOv3 无实例标注街景表示实验

本目录是针对“从表示学习中自动发现景观单元，再修改表示生成图像”的小规模可行性实验。它使用已经提取的 DINOv3 ViT-L/16 密集特征，学习新的对象中心表示。DINOv3 本体保持冻结；未使用 BatchTopK、SAE、类别标签、检测框或人工实例 mask。

## 范围

此处实现的是 DINOSAUR 风格的轻量特征重建模型及一个 RGB 重建探针，不是 DINOSAUR 的正式复现，也不是完整的 SLATE、SlotAdapt、MagicDrive 或街景生成系统。模型没有预先指定“树”“房子”的类别。

```
整张街景的 DINOv3 特征
  → 训练集 PCA（1024 → 64 维）
  → Slot Attention（16 或 32 个 slot，每个 96 维）
  → 共享空间解码器
  → 重建特征与学习到的区域权重
  → RGB 重建器
```

实例数量、形状与外观没有显式解耦。图中不同颜色表示 slot 的区域分配；同一种颜色不固定表示某一语义类别。区域分配也不是已确认的实例分割。

## 数据与划分

- 来源：已有 NAS 街景及原始 DINOv3 特征缓存，路径见 `data/dinov3/manifest.json`。
- 训练：Austin、Bogota、Lagos、London，各 128 个全景的一张视角，共 512 张。
- 同城市验证：以上城市各 16 个不重叠全景，共 64 张。
- 跨城市测试：Adelaide 的 32 张；该城市不参与训练、PCA 或检查点选择。
- 每个全景最多选一个视角，训练和验证的全景 ID 不重叠。
- 输入特征为 28×28 patch、1024 维。PCA 仅在训练图像上拟合，保留约 75.35% 的特征方差。
- 选用历史质检没有删除全景的城市，确认缓存行数为全景数 × 4 × 784。图片顺序沿用既有清单，视角顺序为 0/90/180/270。未重新运行 DINOv3 验证每张图片的特征。
- 源图文件 SHA256、图片路径与缓存偏移均记录在 manifest 中；实验只读取源目录，输出单独保存于本目录。

## 训练与控制

发现模型以原始自监督视觉特征为重建目标。RGB 探针在发现模型冻结后训练，因此 RGB 损失没有用于发现实例。初始探针是共享 MLP 产生的 28×28 图像；改进探针从 slot 重建的特征图出发，使用小型卷积网络输出 112×112 图像，没有原始 RGB 的跳连。

检查点由同城市验证误差选择。所有跨城市指标只在最终评估时计算。

实验包含：

1. 特征与 RGB 重建，以及训练图像均值布局的基线。
2. 将编码替换成其他图像的编码，检查是否依赖表示。
3. 改变 slot 初始化，匹配区域后检查分组稳定性。
4. 镜像已缓存的 patch 特征，检查区域分配的变换一致性。这里没有重新提取镜像 RGB 的 DINOv3 特征，不能视为完整图像增强一致性。
5. 跨图像替换一个占据区域的 slot：分别重新生成 mask 和固定原 mask。候选区域不赋予“树”等标签，供体选用距离较大的编码作为压力测试。

固定 mask 的空间约束是人工施加的模型操作，不证明形态和外观已经解耦。卷积重建器还可能使变化扩散到区域边界附近。slot 数量不等于对象数量，重建误差降低不等于逐棵树发现成功。

## 运行

从项目根目录运行，依赖版本见 `environment.json`。无需新增大型模型下载。

```bash
python experiments/object_centric_pilot/prepare.py
python experiments/object_centric_pilot/run.py --device cuda:3
python experiments/object_centric_pilot/evaluate_pilot.py --device cuda:3
```

本次初始训练：16 slots / batch 8 和 32 slots / batch 4，均为 2,500 次发现更新及 1,500 次 RGB 更新。其训练样本暴露量不相同，不能据此单独归因 slot 数量的影响。

为检查训练不足和弱解码器造成的影响，又继续训练并使用卷积重建器：

```bash
python experiments/object_centric_pilot/run.py \
  --device cuda:3 --slots 16 --batch 8 --steps 7500 --rgb-steps 3000 \
  --lr 0.0002 --rgb-mode features \
  --initial-checkpoint experiments/object_centric_pilot/results/k16/model.pt \
  --output experiments/object_centric_pilot/results/k16-refined

python experiments/object_centric_pilot/run.py \
  --device cuda:3 --slots 32 --batch 8 --steps 8750 --rgb-steps 3000 \
  --lr 0.0002 --rgb-mode features \
  --initial-checkpoint experiments/object_centric_pilot/results/k32/model.pt \
  --output experiments/object_centric_pilot/results/k32-refined

python experiments/object_centric_pilot/evaluate_pilot.py \
  --device cuda:3 --output experiments/object_centric_pilot/results/k16-refined
python experiments/object_centric_pilot/evaluate_pilot.py \
  --device cuda:3 --output experiments/object_centric_pilot/results/k32-refined
```

改进版的发现模型累计各见到约 80,000 个训练样本（含重复采样），RGB 阶段各约 24,000 个。两条发现训练历史和初始 batch 大小仍有差异，属于探索性对照。训练采用单个随机种子，尚未建立统计置信区间。

在预览图中观察到空间条带后，还从 16-slot 改进版继续训练了去掉编码器显式坐标的对照。解码器仍使用空间坐标，DINOv3 特征本身也可能含位置信息，因此不能称为完全移除位置的模型。它仅检验分组阶段额外坐标输入的影响，且属于不同训练历史的探索性对照：

```bash
python experiments/object_centric_pilot/run.py \
  --device cuda:3 --slots 16 --batch 8 --steps 5000 --rgb-steps 3000 \
  --lr 0.0002 --position-strength 0 --rgb-mode features \
  --initial-checkpoint experiments/object_centric_pilot/results/k16-refined/model.pt \
  --output experiments/object_centric_pilot/results/k16-no-position
python experiments/object_centric_pilot/evaluate_pilot.py \
  --device cuda:3 --output experiments/object_centric_pilot/results/k16-no-position
```

所有最终模型已检查输出为有限值，区域权重在 slot 维度求和为 1。源图 SHA256 检查确认训练与验证/测试没有完全相同的原始图像文件。

训练进程设有约 1.8 GiB 的 PyTorch GPU allocator 上限。实际峰值记录在每个结果目录的 `metrics.json` 中，此数不包括全部驱动和 CUDA 上下文开销。

## 输出

- `results/*/model.pt`：包含发现模型和 RGB 重建器的检查点。
- `results/*/metrics.json`：重建误差、稳定性、替换压力测试与训练历史。
- `results/*/per_image.json`：每张验证/测试图片的诊断。
- `results/*/overview.png`：每个城市固定展示第一张验证或测试样本，无成功案例筛选。
- `results/*/gallery_*.png`：全部 96 张验证/测试图的分组与重建结果。
- `results/*/interventions.png`：4 个跨图像 slot 替换例子。
- `RESULTS.md`：完成实验后整理的结论与限制。

未经单棵树实例标注的独立评价，不能报告树实例准确率；未经形态/外观独立干预验证，不能宣称语义完全解耦。后续若使用评价标注，它应只用于测试，不进入无标注训练。

## 方法参考

- [DINOv3 官方实现](https://github.com/facebookresearch/dinov3)
- [DINOSAUR](https://arxiv.org/abs/2209.14860)
- [SLATE：对象条件自回归生成](https://arxiv.org/abs/2110.11405)
- [SlotAdapt：对象表示与预训练扩散模型](https://kaanakan.github.io/SlotAdapt/)
