# 全部 512 个 ISTA 基因

打开 index.html 可按尺度、编号和响应图片数筛选。每个 gene 单独 PNG 保存在 coarse/medium/fine。PDF 按粗、中、细尺度及编号顺序展示全部 gene，每页 8 个，共 64 页。

每张卡最左列为实际归一化字典权重 D：粗尺度 64×1×1，中细尺度 64×3×3，64 个通道按 8×8 拼排。其余五列是此前固定五张街景的实际表达图 A。表达热力图、权重热力图在每个尺度内均采用统一线性色标。红蓝表示权重正负，灰色为卷积核通道间展示间隙；表达图亮度表示系数大小。

图片顺序：Sofia、Tallinn、Tauranga、Cancún、Santo Domingo。mean 包含零，max 为最大表达，A>0 是原生表达网格中的非零位置数。未响应基因亦完整保留。字典图不对应 RGB 外观；表达强度不等于最终重建贡献。

actual_gene_weights.txt 打印全部实际权重数值；actual_gene_weights.npz 为无损数组；gene_catalog.csv 为全部表达统计；metadata.json 记录来源和色标。使用已有第 6000 步检查点严格验证的 FP32 导出结果，CPU 仅绘图和统计，没有新模型推理或训练。
