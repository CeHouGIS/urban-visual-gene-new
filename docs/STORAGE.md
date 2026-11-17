# 实验存储

原始代码、数据、模型检查点和报告保留原位置。GitHub 上传没有移动或修改已有实验。

- NAS 项目根目录：`/host/mnt/nas/urban-visual-gene/`
- 当前 ISTA 实验：`/host/mnt/nas/urban-visual-gene/residual_ista_gene_10k_scratch_20261116/`
- 当前模型：`runs/multiscale_residual_ista/checkpoint_best.pt`，step 6000。
- 完整 512-gene 图册：`reports/all_512_gene_cards_20261117_100042/`，512 张 PNG、64 页 PDF、HTML 和数值权重。
- 空间表达导出：`reports/five_street_expressions_20261117_062418/`。
- 已有 10k RGB 缓存：`/workspace/urban-visual-gene/multiscale_gene_10k_20261116/cache/`。
- GitHub 工作副本：`/host/mnt/nas/urban-visual-gene/github/urban-visual-gene-new/`。

这些位置为本服务器记录，不是公开下载地址。`/workspace` 为服务器本地磁盘，NAS 挂载点为 `/host/mnt/nas`。
GitHub 仓库与以上运行目录分开保存；源快照不能替代外部数据和检查点。
