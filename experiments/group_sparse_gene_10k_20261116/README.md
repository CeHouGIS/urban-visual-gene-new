# Reusable Additive Visual Gene — 10k

实验说明见 [PROTOCOL.md](PROTOCOL.md)，固定参数见 [configs/experiment.json](configs/experiment.json)。

研究优先级：**跨城市有效复用第一、数量精简第二；可组合按特征可加性定义**。主档为每图64个基因上限，8/16/32为次要容量实验。固定字典在未见城市上逐基因删除，验证同一基因是否反复帮助重建；并不要求其具有人类可命名的语义。

保留整张表达图选择、低分辨率8位表达码和两个种子的learned/frozen-random对照。此次在任何GPU模型执行前修订；原方案备份在 `revisions/20261117_reuse_priority/before/`。

实时进度读取 `queue_status.json`。CPU 测试通过不代表 GPU 测试、冒烟或训练已经通过；GPU 检查只有在原实验完成且获得空闲卡后才启动。

NAS：`/host/mnt/nas/urban-visual-gene/group_sparse_gene_10k_20261116`

服务器临时与缓存目录：`/workspace/urban-visual-gene/group_sparse_gene_10k_20261116/tmp`，原 RGB 缓存继续复用，不复制 NAS 已有资源。
