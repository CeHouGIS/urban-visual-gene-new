#!/usr/bin/env python3
"""Label-free translation/scale consistency diagnostic on cached feature maps."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from urbangene_v2.common import sha256, writable
from urbangene_v2.discovery import DiscoveryModel
from urbangene_v2.train import prediction_from_slots
from urbangene_v2.metrics import stability_iou


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    torch.set_num_threads(4)
    model = DiscoveryModel(**checkpoint["config"]["model"]).eval()
    model.load_state_dict(checkpoint["model"])
    data = np.load(args.features, mmap_mode="r")
    background = model.background_slot
    rows = []
    for i, array in enumerate(data):
        features = torch.from_numpy(np.array(array, dtype=np.float32))[None]
        torch.manual_seed(10000 + i)
        original = model(features)
        original_prediction = prediction_from_slots(original["soft_masks"][0], original["exist_logits"][0], background)
        h, w = features.shape[-2:]
        transforms = {
            "translate_2_patches_xy": torch.tensor([[[1., 0., -4 / w], [0., 1., -4 / h]]]),
            "center_zoom_1_15": torch.tensor([[[1 / 1.15, 0., 0.], [0., 1 / 1.15, 0.]]]),
        }
        row = {"index": i}
        for name, theta in transforms.items():
            grid = F.affine_grid(theta, features.shape, align_corners=False)
            transformed = F.grid_sample(features, grid, align_corners=False, padding_mode="border")
            torch.manual_seed(10000 + i)
            result = model(transformed)
            homogeneous = torch.eye(3)
            homogeneous[:2] = theta[0]
            inverse = torch.linalg.inv(homogeneous)[:2][None]
            inverse_grid = F.affine_grid(inverse, features.shape, align_corners=False)
            masks = F.grid_sample(result["soft_masks"], inverse_grid, align_corners=False)[0]
            valid = F.grid_sample(torch.ones(1, 1, h, w), inverse_grid, align_corners=False)[0, 0] > .999
            aligned = prediction_from_slots(masks, result["exist_logits"][0], background)
            valid_np = valid.numpy()
            row[name] = {"matched_iou_on_valid_overlap": stability_iou(original_prediction[valid_np], aligned[valid_np]),
                         "valid_fraction": float(valid.float().mean())}
        rows.append(row)
    summary = {
        "scope": "Cached DINO feature-map transforms; no fresh RGB DINO forward and no labels loaded",
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "feature_file": str(Path(args.features).resolve()), "feature_sha256": sha256(args.features),
        "diagnostic_source_sha256": sha256(__file__), "samples": len(rows),
        "device": "cpu", "gpu_ids": [], "annotation_files_opened": False,
        "transform_sampling": "bilinear with border padding; masks aligned by inverse affine, IoU on valid overlap only",
        "comparison": "Hungarian slot matching with unmatched-instance penalty",
        "means": {name: float(np.mean([r[name]["matched_iou_on_valid_overlap"] for r in rows])) for name in transforms},
        "per_sample": rows,
    }
    output = writable(args.output)
    with output.open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(output), "means": summary["means"]}, indent=2))


if __name__ == "__main__":
    main()
