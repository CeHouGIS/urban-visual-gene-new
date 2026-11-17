#!/usr/bin/env python3
"""Save real-image discovery outputs without inventing instance labels."""
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from urbangene_v2.common import fresh_dir, sha256, write_json
from urbangene_v2.discovery import DiscoveryModel


@torch.inference_mode()
def report(config, checkpoint_path, output, device="cpu"):
    torch.set_num_threads(config["training"].get("cpu_threads", 4))
    output = fresh_dir(output)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DiscoveryModel(**checkpoint["config"]["model"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    train = np.load(config["features"]["train"], mmap_mode="r")
    validation = np.load(config["features"]["validation"], mmap_mode="r")
    records = [json.loads(line) for line in Path(config["manifests"]["validation"]).read_text().splitlines()]
    if len(validation) != len(records):
        raise ValueError("Validation feature count mismatch")
    for split in ("train", "validation"):
        path = Path(config["features"][split])
        meta = json.loads(path.with_suffix(".npy.json").read_text())
        if meta["feature_sha256"] != sha256(path) or meta["manifest_sha256"] != sha256(config["manifests"][split]):
            raise ValueError("Feature identity mismatch")

    # All PCA/statistics fits use training data only, with bounded memory.
    spatial_mean = np.zeros(train.shape[1:], dtype=np.float64)
    samples = []
    pca_indices = set(np.linspace(0, len(train) - 1, min(len(train), 512), dtype=int).tolist())
    for index, item in enumerate(train):
        spatial_mean += item.astype(np.float64)
        if index in pca_indices:
            samples.append(item[:, ::4, ::4].reshape(train.shape[1], -1).T.astype(np.float32))
    spatial_mean = (spatial_mean / len(train)).astype(np.float32)
    channel_mean = spatial_mean.mean((1, 2), keepdims=True)
    x = torch.from_numpy(np.concatenate(samples))
    mean = x.mean(0)
    centered = x - mean
    values, vectors = torch.linalg.eigh(centered.T @ centered / (len(x) - 1))
    basis = vectors[:, -3:]
    projected = (centered @ basis).numpy()
    lo, hi = np.quantile(projected, [.01, .99], axis=0)
    np.savez_compressed(output / "train_only_display_pca.npz", mean=mean.numpy(),
                        basis=basis.numpy(), lo=lo, hi=hi,
                        spatial_mean=spatial_mean, channel_mean=channel_mean)
    del x, centered, samples

    rng = np.random.default_rng(777)
    palette = rng.integers(35, 240, (config["model"]["max_slots"], 3), dtype=np.uint8)
    provenance = json.loads((Path(config["manifests"]["train"]).parent / "provenance.json").read_text())
    preview_ids = provenance["preview_ids"]
    previews, metrics = {}, []
    for i, (item, row) in enumerate(zip(validation, records)):
        feature = torch.from_numpy(item.astype(np.float32))[None]
        torch.manual_seed(10000 + i)
        out = model(feature.to(device))
        masks = out["soft_masks"][0].cpu()
        attention = out["attention_masks"][0].cpu()
        reconstruction = out["reconstructed_features"][0].cpu().numpy()
        decoder_ids, encoder_ids = masks.argmax(0).numpy(), attention.argmax(0).numpy()
        entropy = -(masks.clamp_min(1e-12) * masks.clamp_min(1e-12).log()).sum(0).mean().item()
        # Cyclically shifted validation target is a label-free image dependence diagnostic.
        other = np.asarray(validation[(i + 1) % len(validation)], dtype=np.float32)
        metrics.append({"id": row["id"],
            "feature_reconstruction_mse": float(np.mean((reconstruction - item.astype(np.float32)) ** 2)),
            "train_spatial_mean_mse": float(np.mean((spatial_mean - item.astype(np.float32)) ** 2)),
            "train_channel_mean_mse": float(np.mean((channel_mean - item.astype(np.float32)) ** 2)),
            "shifted_target_mse": float(np.mean((reconstruction - other) ** 2)),
            "decoder_soft_mask_entropy": entropy,
            "decoder_normalized_entropy": entropy / math.log(len(masks)),
            "decoder_mean_max_weight": masks.amax(0).mean().item(),
            "encoder_max_mean_attention_mass": attention.mean((1, 2)).max().item(),
            "encoder_nonempty_regions": len(np.unique(encoder_ids)),
            "decoder_nonempty_regions": len(np.unique(decoder_ids))})
        np.savez_compressed(output / f"sample_{i:04d}.npz", id=np.array(row["id"]),
            soft_masks=masks.numpy().astype(np.float16),
            attention_masks=attention.numpy().astype(np.float16),
            decoder_ids=decoder_ids, encoder_ids=encoder_ids,
            slots=out["slot_features"][0].cpu().numpy(), geometry=out["estimated_geometry"][0].cpu().numpy(),
            presence_probabilities=out["exist_logits"][0].sigmoid().cpu().numpy())
        if row["id"] in preview_ids:
            with Image.open(row["image"]) as image:
                rgb = image.convert("RGB").resize((240, 240), Image.Resampling.BICUBIC)
            f = feature[0].flatten(1).T
            colors = np.clip(((f - mean) @ basis).numpy().reshape(28, 28, 3) - lo,
                             0, hi - lo) / np.maximum(hi - lo, 1e-8)
            pca = Image.fromarray((colors * 255).astype(np.uint8)).resize((240, 240), Image.Resampling.NEAREST)
            enc = Image.fromarray(palette[encoder_ids]).resize((240, 240), Image.Resampling.NEAREST)
            dec = Image.fromarray(palette[decoder_ids]).resize((240, 240), Image.Resampling.NEAREST)
            weight = Image.fromarray((masks.amax(0).numpy() * 255).astype(np.uint8)).convert("RGB").resize((240, 240), Image.Resampling.NEAREST)
            previews[row["id"]] = (rgb, pca, Image.blend(rgb, enc, .5),
                                     Image.blend(rgb, dec, .5), weight)
        if (i + 1) % 16 == 0:
            print(json.dumps({"event": "real_validation", "samples": i + 1}), flush=True)
    titles = ["1 Original photograph", "2 DINO PCA (not RGB)", "3 Encoder regions",
              "4 Decoder regions", "5 Max soft weight [0,1]"]
    for page in (0, 1):
        canvas = Image.new("RGB", (5 * 248, 30 + 4 * 278), "white")
        draw = ImageDraw.Draw(canvas)
        for col, title in enumerate(titles):
            draw.text((col * 248 + 4, 8), title, fill="black")
        for r, identifier in enumerate(preview_ids[page::2]):
            city = identifier.split("_", 1)[0]
            for col, tile in enumerate(previews[identifier]):
                canvas.paste(tile, (col * 248 + 4, 60 + r * 278))
            draw.text((5, 36 + r * 278), f"{city} | fixed validation sample {page + 1}", fill="black")
        canvas.save(output / f"real_street_steps_{page + 1}.png")
    fields = [k for k in metrics[0] if k != "id"]
    summary = {"dataset_kind": config["dataset_kind"], "samples": len(metrics),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "means": {k: float(np.mean([m[k] for m in metrics])) for k in fields},
        "source_images": "real Google Street View RGB files on NAS",
        "annotations_loaded": False, "pca_fit_split": "train", "inference_device": str(device),
        "scientific_instance_gate": "not_evaluated_no_real_instance_annotations",
        "nonempty_regions_are_not_instance_counts": True,
        "argmax_warning": "Near-uniform soft masks can still produce colored argmax regions",
        "image_generation_performed": False, "per_image": metrics}
    write_json(output / "metrics.json", summary)
    m = summary["means"]
    text = f"""# 真实街景实验输出

本轮使用真实 Google 街景照片：{len(train)} 张训练，{len(validation)} 张同城市验证。
冻结 DINOv3 后，从头训练 Slot 模型；未使用对象标签、框或实例 mask。
PCA 显示仅在训练集内最多 512 张照片的图块上拟合；空间均值基线使用全部训练照片。

## 图怎么读

`real_street_steps_1.png` 和 `real_street_steps_2.png` 固定展示每城市验证集的前两张。
从左到右：原始照片、训练集 PCA 显示的 DINO 特征、编码器区域、解码器区域、
解码器最大软权重（黑色 0，白色 1）。PCA 图不是生成图片。
区域颜色仅表示 Slot 编号，不能直接叫作单棵树或独立景观对象。
软权重接近均匀时，取最大值仍会画出彩色分区，应同时查看最后一列。

## 本轮无标签诊断

- 验证特征重建 MSE：{m['feature_reconstruction_mse']:.6f}
- 训练集空间均值基线 MSE：{m['train_spatial_mean_mse']:.6f}
- 训练集通道均值基线 MSE：{m['train_channel_mean_mse']:.6f}
- 重建结果与错位验证图特征的 MSE：{m['shifted_target_mse']:.6f}
- 解码器归一化权重熵：{m['decoder_normalized_entropy']:.4f}（1 为完全均匀）
- 解码器最大软权重的平均值：{m['decoder_mean_max_weight']:.4f}
- 编码器最大 Slot 的平均注意力质量：{m['encoder_max_mean_attention_mass']:.4f}

没有独立真实实例标注，因此未评价单棵树发现准确率，M1 实例发现能力尚未确认。
这是一次单种子、有限预算的真实数据诊断，不是完整街景生成实验。
原始数组逐图保存在 `sample_*.npz`，源图及特征索引见配置和 sidecar。
"""
    (output / "REPORT.md").write_text(text)
    return summary
