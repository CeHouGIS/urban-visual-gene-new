#!/usr/bin/env python3
"""Render static scientific diagnostics from a completed M1 evaluation."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / "outputs/v2/matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def label_rgb(labels):
    palette = plt.get_cmap("gist_ncar")(np.linspace(.07, .95, 64))[:, :3]
    result = palette[(labels.astype(int) * 13) % len(palette)]
    result[labels == 0] = 0
    return result


def pca_rgb(features, mean, basis, limits):
    height, width = features.shape[-2:]
    flat = features.transpose(1, 2, 0).reshape(-1, features.shape[0]).astype(np.float32)
    projected = (flat - mean) @ basis
    projected = np.clip((projected - limits[0]) / np.maximum(limits[1] - limits[0], 1e-6), 0, 1)
    return projected.reshape(height, width, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--training-features", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    evaluation, run = Path(args.evaluation), Path(args.run)
    metrics = json.loads((evaluation / "metrics.json").read_text())
    records = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    assert len(records) == metrics["samples"]
    train_features = np.load(args.training_features, mmap_mode="r")
    sample = np.asarray(train_features[:16], dtype=np.float32).transpose(0, 2, 3, 1).reshape(-1, 1024)[::6]
    mean = sample.mean(0)
    _, _, vt = np.linalg.svd(sample - mean, full_matrices=False)
    basis = vt[:3].T
    projected = (sample - mean) @ basis
    limits = np.quantile(projected, [.01, .99], axis=0)
    np.savez_compressed(evaluation / "pca_training_basis.npz", mean=mean, basis=basis, limits=limits)

    def contact(indices, filename, title):
        fig, axes = plt.subplots(len(indices), 5, figsize=(12.5, 2.35 * len(indices)), squeeze=False)
        for row, index in enumerate(indices):
            item = np.load(evaluation / f"sample_{index:04d}.npz")
            record = records[index]
            with Image.open(record["image"]) as image:
                images = [np.asarray(image), pca_rgb(item["features"], mean, basis, limits),
                          label_rgb(item["gt"]), label_rgb(item["prediction"]),
                          label_rgb(item["prediction_flip_back"])]
            score = metrics["per_sample"][index]
            labels = [record["id"], "Frozen DINOv3 PCA", f"Eval GT: {score['gt_instances']} instances",
                      f"Prediction: {score['pred_instances']} instances",
                      f"Feature flip IoU {score['flip_stability_iou']:.3f}"]
            for col, (image, label) in enumerate(zip(images, labels)):
                axes[row, col].imshow(image, interpolation="nearest")
                axes[row, col].set_title(label, fontsize=8)
                axes[row, col].axis("off")
            axes[row, 3].set_xlabel(f"FG-ARI {score['foreground_ari']:.3f}; IoU {score['matched_mask_iou']:.3f}", fontsize=8)
        fig.suptitle(title + "\nColors identify slots within each map; labels are used for evaluation only", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .965))
        fig.savefig(evaluation / filename, dpi=140)
        plt.close(fig)

    contact(list(range(min(6, len(records)))), "discovery_first6.png", "M1: image -> frozen DINO features -> instance grouping")
    worst = np.argsort([row["foreground_ari"] for row in metrics["per_sample"]])[:6].tolist()
    contact(worst, "discovery_worst6.png", "M1 validation failure cases, ranked by FG-ARI")

    diagnostics = []
    for index in range(len(records)):
        item = np.load(evaluation / f"sample_{index:04d}.npz")
        masks = item["soft_masks"].astype(np.float32)
        encoder_labels = item["attention_masks"].argmax(0)
        diagnostics.append({"id": records[index]["id"],
                            "mean_mask_entropy": float(-(masks * np.log(masks + 1e-8)).sum(0).mean()),
                            "encoder_argmax_nonempty_slots": int(len(np.unique(encoder_labels))),
                            "decoder_argmax_nonempty_slots": int(len(np.unique(masks.argmax(0)))),
                            "present_gates": int((item["exist_probabilities"] >= .5).sum()),
                            "mean_probability_of_background_slot": float(masks[0].mean()),
                            "mean_geometry_center_std_across_slots": float(item["geometry"][:, :2].std(0).mean())})
        if index < 3:
            fig, axes = plt.subplots(4, 8, figsize=(14, 7))
            for slot, ax in enumerate(axes.flat):
                if slot < len(masks):
                    ax.imshow(masks[slot], cmap="magma", vmin=0, vmax=max(float(masks.max()), .05))
                    ax.set_title(f"Slot {slot}{' (BG)' if slot == 0 else ''}; e={item['exist_probabilities'][slot]:.2f}\nmax={masks[slot].max():.3f}", fontsize=8)
                ax.axis("off")
            fig.suptitle(f"{records[index]['id']}: decoder soft masks, common color range")
            fig.tight_layout()
            fig.savefig(evaluation / f"soft_masks_{index:04d}.png", dpi=140)
            plt.close(fig)
    (evaluation / "slot_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")

    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
    epochs = sorted(set(row["epoch"] for row in history))
    train_loss = [np.mean([row["loss"] for row in history if row["epoch"] == e]) for e in epochs]
    val_loss = [next(row["validation_reconstruction_mse"] for row in reversed(history)
                     if row["epoch"] == e and "validation_reconstruction_mse" in row) for e in epochs]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].plot(epochs, train_loss, label="Train epoch mean")
    axes[0].plot(epochs, val_loss, label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Feature reconstruction MSE")
    axes[0].legend(fontsize=8)
    gt_counts = [row["gt_instances"] for row in metrics["per_sample"]]
    pred_counts = [row["pred_instances"] for row in metrics["per_sample"]]
    bins = np.arange(-.5, max(gt_counts + pred_counts) + 1.5)
    axes[1].hist([gt_counts, pred_counts], bins=bins, label=["Visible GT", "Predicted"])
    axes[1].set(xlabel="Nonempty foreground instances", ylabel="Images")
    axes[1].legend(fontsize=8)
    names = ["Model", "One slot", "4x4 grid", "Random"]
    scores = [metrics["mean_foreground_ari"], metrics["mean_one_slot_ari"], metrics["mean_regular_grid_ari"], metrics["mean_random_ari"]]
    axes[2].bar(names, scores)
    axes[2].set(ylabel="Validation FG-ARI")
    fig.tight_layout()
    fig.savefig(evaluation / "training_and_counts.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"report": str(evaluation), "figures": ["discovery_first6.png", "discovery_worst6.png",
                      "soft_masks_0000.png", "soft_masks_0001.png", "soft_masks_0002.png", "training_and_counts.png"]}, indent=2))


if __name__ == "__main__":
    main()
