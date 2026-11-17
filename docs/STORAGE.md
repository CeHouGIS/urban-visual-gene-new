# 当前四层顺序实验存储

- 当前运行：`/host/mnt/nas/urban-visual-gene/hierarchical_sequential_gene_10k_20261117/`。
- 代码、配置、设计快照：该目录的 `source/`、`configs/`、`README.md`。
- 正式训练输出：`runs/main/`；检查和训练后诊断：`reports/`；队列动态状态：`queue_status.json`。
- GitHub 分支：`hierarchical-sequential`；原 `main` 代码保存为 `additive`。
- GitHub 工作副本与实际运行目录分开；发布代码不修改正在训练的源码、数据或检查点。

以下为原加性/ISTA 实验的历史存储记录。

---

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
