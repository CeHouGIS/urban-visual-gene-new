# Group Sparse Visual Gene — 10k

实验说明见 [PROTOCOL.md](PROTOCOL.md)，固定参数见 [configs/experiment.json](configs/experiment.json)。

核心改动：整张基因表达图选择、每图 8/16/32/64 档上限、低分辨率 8 位表达码、从头训练的 learned/frozen-random 字典对照，两个种子。

实时进度读取 `queue_status.json`。CPU 测试通过不代表 GPU 测试、冒烟或训练已经通过；GPU 检查只有在原实验完成且获得空闲卡后才启动。

NAS：`/host/mnt/nas/urban-visual-gene/group_sparse_gene_10k_20261116`

服务器临时与缓存目录：`/workspace/urban-visual-gene/group_sparse_gene_10k_20261116/tmp`，原 RGB 缓存继续复用，不复制 NAS 已有资源。
