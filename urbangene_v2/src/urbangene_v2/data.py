import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .common import LEGACY_ROOT, writable


class ImageManifestDataset(Dataset):
    """Image-only input. It intentionally has no annotation/spec argument."""
    def __init__(self, manifest):
        self.records = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line]
        if not self.records or any(set(row) != {"id", "image", "split"} for row in self.records):
            raise ValueError("Image manifest must contain only id, image and split")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        row = self.records[i]
        with Image.open(row["image"]) as image:
            image = image.convert("RGB")
            tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        return {"image": tensor, "id": row["id"]}


class FeatureDataset(Dataset):
    """Feature-only input. Does not open images, specs, masks, boxes or counts."""
    def __init__(self, path, grid_hw=None):
        self.path = Path(path)
        self.array = np.load(self.path, mmap_mode="r")
        if self.array.ndim == 3:
            if grid_hw is None or self.array.shape[1] != grid_hw[0] * grid_hw[1]:
                raise ValueError("Flattened [N,HW,C] features need a matching grid_hw")
            self.grid_hw = tuple(grid_hw)
            self.layout = "nhwc_flat"
        elif self.array.ndim == 4:
            self.grid_hw = tuple(self.array.shape[-2:])
            self.layout = "nchw"
        else:
            raise ValueError("Features must be [N,C,H,W] or flattened [N,HW,C]")

    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        item = np.array(self.array[i], dtype=np.float32, copy=True)
        if self.layout == "nhwc_flat":
            item = item.reshape(*self.grid_hw, -1).transpose(2, 0, 1)
        return {"features": torch.from_numpy(item)}


def _instance_mask(meta, image_size):
    masks = Image.new("I", (image_size, image_size), 0)
    draw = ImageDraw.Draw(masks)
    shape, next_id = meta["building_shape"], 1
    for left, top, right, horizon in meta["building_boxes"]:
        if shape == 0:
            draw.rectangle((left, top, right, horizon), fill=next_id)
        elif shape == 1:
            mid = (left + right) // 2
            draw.polygon([(left, top + 23), (mid, top + 23), (mid, top),
                          (right, top), (right, horizon), (left, horizon)], fill=next_id)
        else:
            draw.rectangle((left, top - 18, right, horizon), fill=next_id)
            draw.polygon([(left - 5, top - 18), ((left + right) // 2, top - 32),
                          (right + 5, top - 18)], fill=next_id)
        next_id += 1
    for x0, y0, x1, y1 in meta["tree_boxes"]:
        radius = max(1, round((x1 - x0) / 2))
        cx, cy = round((x0 + x1) / 2), y0 + 2 * radius
        draw.rectangle((cx - 2, cy - 4, cx + 2, y1), fill=next_id)
        draw.ellipse((cx - radius, y0, cx + radius, cy + 3), fill=next_id)
        next_id += 1
    return np.asarray(masks, dtype=np.uint8)


def prepare_synthetic(config, output):
    """Reuse V0 renderer; keep target masks outside image manifests."""
    output = writable(output)
    sys.path.insert(0, str(LEGACY_ROOT / "urbangene" / "src"))
    from urbangene.data.synthetic import SPLITS, SyntheticSceneGenerator, make_specs

    output.mkdir(parents=True, exist_ok=False)
    renderer = SyntheticSceneGenerator(config["image_size"])
    source = LEGACY_ROOT / "urbangene/src/urbangene/data/synthetic.py"
    renderer_hash = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    for split in SPLITS:
        image_dir = output / "images" / split
        image_dir.mkdir(parents=True)
        records, gt_rows = [], []
        specs = make_specs(config["legacy_config"], split)
        for index, spec in enumerate(specs):
            image, meta = renderer.sample(spec)
            sample_id = f"{split}_{index:05d}"
            image_path = image_dir / f"{sample_id}.png"
            image.save(image_path)
            records.append({"id": sample_id, "image": str(image_path), "split": split})
            gt_rows.append(_instance_mask(meta, config["image_size"]))
        (output / f"images.{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records))
        if split != "train":
            np.savez_compressed(output / f"annotations.{split}.npz",
                                instance_ids=np.stack(gt_rows),
                                visible_instance_counts=np.asarray([len(np.unique(mask)) - 1
                                                                    for mask in gt_rows], dtype=np.uint8))
    (output / "generation.json").write_text(json.dumps({
        "seed": config["legacy_config"]["seed"], "renderer": "urbangene.data.synthetic",
        "renderer_sha256": renderer_hash,
        "training_manifests": "image id/path/split only; no scene specifications or labels",
        "annotation_use": "evaluation only; stored separately from training inputs",
        "splits": config["legacy_config"]["data"]["counts"]}, indent=2) + "\n")
