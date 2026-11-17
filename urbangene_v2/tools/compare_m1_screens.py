#!/usr/bin/env python3
"""Compare the archived initial M1 screen and its bounded spatial ablation."""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent / "outputs/v2"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / "matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
sys.path.insert(0, str(ROOT / "tools"))
from report_m1_validation import label_rgb


def main():
    runs = {"initial": OUTPUT / "m1_dinov3_cpu_screen_20261109",
            "spatial_seed": OUTPUT / "m1_dinov3_cpu_spatial_seed_20261109"}
    summaries = {}
    for name, directory in runs.items():
        metrics = json.loads((directory / "validation/metrics.json").read_text())
        tree = json.loads((directory / "validation_ten_tree/tree_separation.json").read_text())
        slots = json.loads((directory / "validation/slot_diagnostics.json").read_text())
        summaries[name] = {"run": str(directory),
            "checkpoint_sha256": metrics["checkpoint_sha256"],
            **{key: value for key, value in metrics.items() if key.startswith("mean_")},
            "slot_diagnostics_means": {key: float(np.mean([row[key] for row in slots])) for key in slots[0] if key != "id"},
            "tree_separation": tree["means"], "scientific_gate": metrics["scientific_gate"]}
    data = OUTPUT / "data/synthetic_20261109"
    records = [json.loads(line) for line in (data / "images.val_seen.jsonl").read_text().splitlines()]
    fig, axes = plt.subplots(3, 4, figsize=(10, 9), layout="constrained")
    for row in range(3):
        original = np.load(runs["initial"] / f"validation/sample_{row:04d}.npz")
        spatial = np.load(runs["spatial_seed"] / f"validation/sample_{row:04d}.npz")
        with Image.open(records[row]["image"]) as image:
            plots = [np.asarray(image), label_rgb(original["gt"]),
                     label_rgb(original["prediction"]), label_rgb(spatial["prediction"])]
        for col, (plot, title) in enumerate(zip(plots, [records[row]["id"], "Visible GT (eval only)", "Initial M1", "Spatial initialization"])):
            axes[row, col].imshow(plot, interpolation="nearest")
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis("off")
    fig.suptitle("M1 validation: first three images in manifest order\nColors identify arbitrary groups within each map", fontsize=12)
    figure = OUTPUT / "m1_validation_comparison_20261109.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    gate_pass = all(value["scientific_gate"]["grouping_evidence_pass"] for value in summaries.values())
    summary = {"updated_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Two bounded, single-seed CPU validation screens, not final testing or across-seed evidence",
               "runs": summaries, "comparison_figure": str(figure),
               "mainline_phase": "M1", "m2_started": False, "test_splits_opened": False,
               "decision": "M1 requires further diagnosis; do not advance to M2" if not gate_pass else "Review density and stability gates before deciding on M2",
               "source_script": str(Path(__file__).resolve())}
    (OUTPUT / "m1_validation_comparison_20261109.json").write_text(json.dumps(summary, indent=2) + "\n")
    a, b = summaries["initial"], summaries["spatial_seed"]
    table = "\n".join(f"| {label} | {a[key]:.6f} | {b[key]:.6f} |" for label, key in [
        ("FG-ARI ↑", "mean_foreground_ari"), ("匹配 Mask IoU ↑", "mean_matched_mask_iou"),
        ("实例计数 MAE ↓", "mean_count_abs_error"), ("特征重建 MSE ↓", "mean_feature_reconstruction_mse"),
        ("特征翻转匹配 IoU ↑", "mean_flip_stability_iou")])
    report = f"""# UrbanGene V2：M1 主线实验记录（2026-11-09）

当前完成的是“冻结 DINOv3 → 无标注 Slot Discovery → 重建 DINO 特征”的小规模验证。初版和空间初始化对照都按独立目录保存。最终结论以对应 `metrics.json` 中的科学 Gate 为准；训练 loss 下降不等于发现了景观单元。

## 实际完成

- 从已校验的 ModelScope DINOv3 ViT-L/16 权重提取 256 张训练图和 64 张验证图；448×448 输入得到 28×28、1024 维特征。缓存、manifest 和权重都有哈希。
- 训练/验证没有重复图片；逐张再生成验证图与实例掩码并对齐。正式测试集保持锁定。
- 两轮各训练 10 epoch、640 步，seed=42；使用 CPU。训练只输入 DINO 特征，检查点只按验证重建误差选择。
- 第二轮仅加入坐标驱动的初始空间位置，后续位置、尺度和区域仍自适应学习；没有预先定义树或建筑单元，没有实例监督。
- 工程测试 24 项通过；新增解码计算与旧计算的输出/梯度等价。

## 普通验证组结果

| 指标 | 初版 | 空间初始化对照 |
| --- | ---: | ---: |
{table}

固定 4×4 网格的 FG-ARI 为 0.261881、匹配 IoU 为 0.056344。只用训练图计算的空间平均特征图在验证集的 MSE 为 0.017190，因此低重建误差本身也不足以证明模型使用了实例结构。

初版 Gate：**{'通过' if a['scientific_gate']['grouping_evidence_pass'] else '未通过'}**；空间初始化 Gate：**{'通过' if b['scientific_gate']['grouping_evidence_pass'] else '未通过'}**。判据为：相对单 Slot、固定网格和随机分组，在 FG-ARI 与匹配 IoU 上的逐图配对 bootstrap 95% 差值下界都大于零。该统计只描述此随机种子与此验证分布。

## 十棵树的独立诊断组

另建 12 张验证诊断图，每张十棵树，超出训练图的 1–3 棵树密度；十棵树在 RGB 和特征网格的评价标注上均可见。它不属于正式 `test_higher_complexity`，其标注和 scene_spec 只进入评价。

| 指标 | 初版 | 空间初始化对照 |
| --- | ---: | ---: |
| 树的匹配 IoU | {a['tree_separation']['matched_tree_iou_mean']:.6f} | {b['tree_separation']['matched_tree_iou_mean']:.6f} |
| 树独立匹配召回（IoU≥0.5） | {a['tree_separation']['distinct_tree_recall_iou50']:.6f} | {b['tree_separation']['distinct_tree_recall_iou50']:.6f} |

## 失败定位与解释边界

初版编码端的最大 Slot 平均占据 95.69% 注意力质量，解码端掩码熵接近均匀分配；32 个门控全部打开但非空前景组数均值仅 2.78。特征中仍可见实例并不保证 Slot 会将其分开。图像中天空和道路占比较大、模型可通过共享位置模式降低重建损失，是后续应检验的因素，尚不能认定为唯一原因。

空间初始化对照的平均中心离散度为 {b['slot_diagnostics_means']['mean_geometry_center_std_across_slots']:.6f}，初版为 {a['slot_diagnostics_means']['mean_geometry_center_std_across_slots']:.6f}；解码掩码熵分别为 {b['slot_diagnostics_means']['mean_mask_entropy']:.6f} 与 {a['slot_diagnostics_means']['mean_mask_entropy']:.6f}。需要联合查看这些数值、实例匹配和可视化，不能仅凭初始化分散就认为有效。

这两轮只有 10 epoch，不能据此断言 DINOv3 或所有 Slot Attention 方法不可行；当前证据只说明这个配置/预算尚未满足 M1。

## 下一步主线决策

维持在 M1，先解决竞争归属和局部重建的退化，再进入共享 Gene。下一项应为有单独协议的竞争/局部解码对照：检查背景分支与前景竞争、slot geometry 更新是否抹掉初始差异、以及位置信息是否让解码器绕开内容分组。对照仍只使用特征重建训练；标签只用于验证。若局部分组改善，再检验训练预算和多尺度是否有增益。M2、M3、M4 当前均未运行。

## 可追溯文件

- 初版：`{runs['initial']}`
- 空间初始化：`{runs['spatial_seed']}`
- 每轮 `validation/metrics.json`、`validation_ten_tree/metrics.json`、`tree_separation.json`、`slot_diagnostics.json`、`sample_*.npz`。
- 每轮的 `discovery_first6.png`、`discovery_worst6.png`、`soft_masks_0000.png` 等、`training_and_counts.png`。
- 初始协议：`M1_VALIDATION_PROTOCOL_20261109.json`；消融协议：`M1_SPATIAL_SEED_ABLATION_PROTOCOL_20261109.json`。
- 源码快照、执行状态、训练 stdout 日志、特征 sidecar、模型下载校验记录均在 `{OUTPUT}`。
- 对照图：`{figure}`。

翻转、平移、尺度稳定性在缓存特征图上计算，不等同于对变换后的 RGB 重新执行 DINO。所有模型都使用 CPU；后续 GPU 实验仍最多两张卡。原 V0/V1.1 队列与检查点未被本实验修改。
"""
    destination = ROOT / "docs/M1_EXPERIMENT_REPORT_20261109.md"
    destination.write_text(report)
    print(json.dumps({"report": str(destination), "summary": str(OUTPUT / "m1_validation_comparison_20261109.json"),
                      "decision": summary["decision"]}, indent=2))


if __name__ == "__main__":
    main()
