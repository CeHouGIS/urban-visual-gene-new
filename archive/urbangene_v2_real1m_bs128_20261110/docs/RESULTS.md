# UrbanGene V2 Results

## 2026-11-10：扩大到 20,000 张（进行中）

新的 20,000 张训练 / 512 张验证真实照片已完成核验，原架构与损失保持一致。
从头训练预算为 30 epochs / 150,000 updates，原 64 张验证图保留用于相同图像比较。
当前流程已启动，正在提取新增数据的特征，还没有本轮最终训练结果。
结果独立保存于 `outputs/v2/m1_real20k_20261110`，方案见 `REAL20K_PROTOCOL_20261110.md`。

## 数据来源说明与主线修正

下文两轮正式诊断训练采用**程序生成的简化街景**，不是实拍照片。
这些结果不能回答真实街景上的生成质量或逐棵树发现能力。

2026-11-09 已按用户纠正完成独立真实街景流程：512 张训练照片、64 张验证照片，
全部来自 NAS 的 Google 街景。源文件已逐张核验，沿用全景级划分。
启动记录为 `outputs/v2/m1_real_streets_cpu_20261109_execution.json`，
真实结果已写入新 run 的 `validation/REPORT.md`。

## 真实街景首轮结果（2026-11-09）

CPU；10 epochs / 1280 updates；单种子；从头训练 Slot 模型；DINOv3 冻结。
完整提取、训练、验证已于北京时间 17:15 完成。本轮未开展 Gene 组合或 RGB 生成。

| 无标签诊断 | 结果 | 含义 |
| --- | ---: | --- |
| 验证特征重建 MSE | 0.037830 | 比训练集空间均值基线 0.039362 低 3.89% |
| 编码器最大 Slot 平均注意力质量 | 86.19% | 一个候选 Slot 占据大部分注意力，分组明显集中 |
| 解码器归一化软权重熵 | 0.9893 | 接近完全均匀的 1，区域竞争弱 |
| 解码器最大软权重平均值 | 4.82% | 32 个 Slot 均分时为 3.125%，最大者也没有明显占优 |

固定展示各城市验证集前两张，未按输出效果挑图。输出有大块区域、条带和
中心小块，没有形成可靠对象边界；不能将彩色最大值分区当成逐棵树发现成功。
这只是当前架构和预算的诊断，不是对 DINOv3 或无监督对象发现的总体否定。
没有真实实例标注，未计算实例准确率或真实 FG-ARI；M1 实例能力尚未确认。

简图为 `validation/real_street_result_simple.png`，全部中间结果见
`validation/real_street_steps_1.png`、`real_street_steps_2.png` 和 `sample_*.npz`。

## Engineering Checks

- Focused unit tests: 24 passed, including GPU-selection cap, fail-before-output,
  exact decoder output/gradient equivalence, metric robustness, and spatial
  initialization checks. The first pipeline version had 17 passing tests.
- CPU smoke: 2 updates on the existing 16x16, 1024-D real-image feature cache.
  Reconstruction MSE changed from 0.0133196 to 0.0130067. No annotation files
  were opened and no object labels were used. This verifies execution only.
- A second CPU smoke after the GPU-selection change completed with `device=cpu`
  and no selected GPUs, confirming the CPU path remains independent of CUDA.
- Synthetic dataset: 256 training images and 64 images in each of four
  validation/test splits. Image manifests contain only IDs, paths, and split
  names; evaluation instance masks are stored separately.
- Synthetic image/mask dimensions were checked at 256x256. The checkpoint
  was subsequently downloaded and both synthetic screens below completed.
- Training can explicitly use two GPUs through `DataParallel`; requests for
  more than two GPUs are rejected. GPU execution remains untested while all
  visible GPUs are occupied.
- ModelScope download completed: `facebook/dinov3-vitl16-pretrain-lvd1689m`,
  revision `99b6205b0321c4d9726dd072d8c6a80aa6be72df`, 303,129,600 parameters.
  The 1,212,559,808-byte safetensors SHA256 is
  `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.
  All snapshot files match ModelScope's hashes. Local Transformers loading
  on CPU reported no missing, unexpected, or mismatched keys. ModelScope
  provenance is recorded; equality to the Hugging Face snapshot was not
  independently checked. See `outputs/v2/dinov3_download.json`.
- Frozen DINOv3 CPU integration passed on the first two synthetic training
  images, using 448x448 inputs and the existing extractor. Output shape is
  `[2,1024,28,28]`, float16, finite; feature mean is -0.00697 and standard
  deviation is 0.26110 (statistics computed in float32). IDs align with the
  manifest; no labels or GPUs were used. Results and provenance are under
  `outputs/v2/dinov3_modelscope_cpu_smoke_20261109/`. This is an engineering
  check, not evidence of instance discovery.

## Scientific M1 Gate

The first single-seed CPU validation screen is complete and **failed** the
scientific grouping gate. This is a bounded 10-epoch diagnostic, not a claim
that all training budgets or all Slot Attention designs fail.

- Data: 256 training and 64 validation images, 448x448 DINO inputs,
  `[N,1024,28,28]` float16 caches. No duplicate image hashes or train/val hash
  overlap. All validation images and masks were regenerated and matched;
  no visible GT instance disappeared at 28x28. Formal tests stay unopened.
- Initial run: `outputs/v2/m1_dinov3_cpu_screen_20261109`, seed 42,
  10 epochs / 640 updates / 381.22 training seconds. Checkpoint selection
  used fixed-RNG validation reconstruction only, with no instance annotations.
- Best validation feature MSE: 0.0254117; separate evaluation sampling gives
  0.0254110. Train-only channel-mean baseline: 0.0515620; train-only spatial
  mean baseline: **0.0171902**, better than the trained model.

| Method | Validation FG-ARI | Matched mask IoU | Count MAE |
| --- | ---: | ---: | ---: |
| Initial M1 | 0.009723 | 0.015334 | 1.984375 |
| One slot | 0.000000 | 0.033605 | 2.921875 |
| Fixed 4x4 grid | 0.261881 | 0.056344 | 12.078125 |
| Random grouping | -0.000441 | 0.007742 | 26.078125 |

Matched mask IoU sums one-to-one Hungarian IoUs and divides by the larger
GT/predicted instance count. GT background is excluded; model slot 0 is
treated as background in the primary protocol. FG-ARI uses GT foreground
pixels and retains predicted background as a cluster label. Split/merge
fractions use any nonzero overlap and can be sensitive to small boundaries.

The paired image-bootstrap 95% interval for M1 minus grid FG-ARI is
[-0.265235,-0.238331]; for matched IoU it is [-0.045506,-0.036465]. These are
image-level uncertainty estimates for this seed and synthetic distribution,
not across-seed scientific confirmation.

Collapse diagnostics: all 32 presence gates open, but mean nonempty foreground
groups are 2.78125 (GT 3.921875). Mean encoder largest-slot attention mass is
0.956882. Decoder entropy is 3.45224, close to uniform log(32)=3.46574; slot
center spread is only 0.01038 in normalized coordinates. Re-evaluating all
slots as positive foreground leaves FG-ARI essentially unchanged, so the
poor grouping is not explained by the primary background designation alone.

Feature-map transform consistency: flip IoU 0.462006, translation by two
patches in x/y 0.900252 on the valid overlap, and center zoom 1.15 at 0.571637.
These transform **cached features**, not RGB with DINO re-encoding. Stable
collapsed partitions do not establish object discovery.

Ten-tree validation stress group: 12 images, ten visible trees at both RGB
and patch-grid resolution. Full-scene FG-ARI 0.005669, matched IoU 0.008317,
count MAE 9.5. Tree-specific matched IoU is 0.001627 and distinct-tree recall
at IoU >= 0.5 is **0**. This is an extra validation diagnostic, not the locked
`test_higher_complexity` split. It can be reproduced with
`tools/prepare_ten_tree_diagnostic.py`; its scene metadata stays evaluation-only.

The bounded follow-up changed only initial spatial seed geometry and retained
learned masks/adaptive geometry. Its config and hypothesis were recorded
before training in
`outputs/v2/M1_SPATIAL_SEED_ABLATION_PROTOCOL_20261109.json`. It completed
10 epochs / 640 updates in 392.52 training seconds.

| Spatial initialization screen | Value |
| --- | ---: |
| Validation FG-ARI | 0.085687 |
| Matched mask IoU | 0.040773 |
| Count MAE | 2.093750 |
| Mean nonempty foreground groups | 4.890625 |
| Feature reconstruction MSE (evaluation) | 0.025414 |
| Feature flip stability IoU | 0.359602 |
| Mean mask entropy | 3.452585 |
| Mean slot-center spread | 0.011639 |
| Ten-tree matched tree IoU | 0.005002 |
| Ten-tree distinct recall at IoU >= 0.5 | 0.000000 |

This improves grouping over the initial model but still **fails** the
predefined gate. The model-minus-grid FG-ARI paired 95% interval remains
negative: [-0.201864,-0.148523]. The matched IoU interval is
[-0.021513,-0.009111]. A better initialization alone did not resolve the
near-uniform decoder masks or small center spread. The count MAE also did
not improve; a larger number of nonempty groups is not sufficient evidence
of correct instance grouping.

The ten-tree group exceeds the training density of 1-3 trees. It is an
evaluation stress group, not training supervision or a formal test result.
Both runs use the same validated feature caches and keep all final test
splits locked. M2-M4 remain `not_run`.

Figures, metrics, per-image tensors, slot diagnostics, and checkpoint/feature/
annotation hashes are under the initial run's `validation/` and
`validation_ten_tree/`. Source snapshots and protocol hashes are in
`outputs/v2`. No M2 Gene dictionary or M3/M4 result is claimed.

The Chinese consolidated report is `docs/M1_EXPERIMENT_REPORT_20261109.md`;
the side-by-side figure is `outputs/v2/m1_validation_comparison_20261109.png`
and its numerical summary is the adjacent JSON. All numbers above come from
completed runs and remain single-seed validation observations.

## Download Network History

The `.env`-based download of `facebook/dinov3-vitl16-pretrain-lvd1689m` was
attempted. The proxy accepts CONNECT to Hugging Face but stalls after the TLS
ClientHello; the same happens through SOCKS, while direct access times out.
GitHub is reachable. This locates the current blocker in network egress, before
token authentication. That Hugging Face attempt downloaded no model files and
recorded no token value. The configured egress needs access to `huggingface.co`,
`cdn-lfs.huggingface.co`, and `cas-bridge.xethub.hf.co` over HTTPS. See
`outputs/v2/model_search.json` for the audit. The later public ModelScope
download resolved model availability without reading or sending the HF token.
Do not interpret PCA64 pilot masks or V1.1 RGB metrics as V2 M1 results.
