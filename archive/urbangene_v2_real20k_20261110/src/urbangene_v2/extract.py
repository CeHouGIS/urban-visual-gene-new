import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .backbone import FrozenVisualBackbone
from .common import sha256, writable
from .data import ImageManifestDataset


def _collate_rgb(batch, resolution):
    # Real photographs have different aspect ratios. Resize each image with
    # the backbone's exact policy before stacking; same-size resize inside
    # the backbone is an identity operation.
    images = [torch.nn.functional.interpolate(
        row["image"][None], (resolution, resolution), mode="bicubic",
        align_corners=False, antialias=True).clamp(0, 1)[0] for row in batch]
    return {"image": torch.stack(images), "id": [row["id"] for row in batch]}


@torch.inference_mode()
def extract_manifest(manifest, output, backbone_config, batch_size=1, workers=0, device="cpu",
                     progress_callback=None):
    output = writable(output)
    manifest = Path(manifest).resolve()
    dataset = ImageManifestDataset(manifest)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.startswith("cuda"),
                        collate_fn=partial(_collate_rgb, resolution=backbone_config["resolution"]))
    model = FrozenVisualBackbone.from_config(backbone_config).to(device).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Frozen backbone has trainable parameters")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".partial")
    if output.exists() or temp.exists():
        raise FileExistsError(f"Refusing to overwrite feature cache: {output}")
    cache = None
    row = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=device.startswith("cuda"))
        features = model(images)["features"].cpu().numpy().astype(np.float16)
        if cache is None:
            cache = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float16,
                                             shape=(len(dataset), *features.shape[1:]))
        cache[row:row + len(features)] = features
        row += len(features)
        if progress_callback is not None:
            progress_callback(row, len(dataset))
    if cache is None or row != len(dataset):
        raise RuntimeError("Feature extraction ended before all manifest rows")
    cache.flush()
    del cache
    temp.replace(output)
    sidecar = output.with_suffix(output.suffix + ".json")
    records = dataset.records
    metadata = {
        "manifest": str(manifest), "manifest_sha256": sha256(manifest),
        "feature_file": str(output), "feature_sha256": sha256(output),
        "shape": list(np.load(output, mmap_mode="r").shape), "dtype": "float16",
        "records": len(records), "ids": [row["id"] for row in records],
        "backbone": model.provenance, "resolution": model.resolution,
        "patch_size": model.patch_size, "normalization": {"mean": [.485, .456, .406],
                                                           "std": [.229, .224, .225]},
        "image_input": "RGB [0,1], square bicubic resize, no crop",
        "training_labels_loaded": False,
    }
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
