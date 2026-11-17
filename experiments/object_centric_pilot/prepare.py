"""Make a small, RGB-only training set from existing DINOv3 patch caches.

No SAE activations, labels, masks, or boxes are read. Split by panorama before
selecting a view. Only cities without historical QC removals are supported,
so the 4-view/784-patch offsets can be verified against the source manifest.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def prepare(source: Path, output: Path, seed: int = 42):
    torch.set_num_threads(4)
    rng = np.random.default_rng(seed)
    cities = ["Austin", "Bogota", "Lagos", "London", "Adelaide"]
    groups = {city: OrderedDict() for city in cities}
    parts = sorted((source / "formal_city_qc").glob("formal_sample_images.part*.tsv"))
    if not parts:
        raise FileNotFoundError("Missing ordered image manifests")
    for part in parts:
        for line in part.open():
            city, name = line.rstrip("\n").split("\t", 1)
            if city not in groups:
                continue
            pano, heading = Path(name).stem.rsplit("_", 1)
            path = Path(name.replace("/host/root/mnt/nas/", "/host/mnt/nas/"))
            groups[city].setdefault(pano, {})[int(heading)] = path
    cache = source / "formal_batchtopk_307city_balanced_3000"
    qc = json.loads((cache / "sae_qc_manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    all_features, all_rgb, records, checks = [], [], [], []
    for city in cities:
        entries = list(groups[city].items())
        status = qc["cities"][city]
        features = np.load(cache / f"dict_{city}_p784.f16.npy", mmap_mode="r")
        assert status["black"] == 0 and status["tunnel"] == 0, city
        assert features.shape == (len(entries) * 4 * 784, 1024), (city, features.shape)
        assert all(set(views) == {0, 90, 180, 270} for _, views in entries)
        checks.append({"city": city, "panoramas": len(entries),
                       "feature_rows": int(features.shape[0]), "removed_panoramas": 0})
        order = rng.permutation(len(entries))
        selections = [("cross_city", order[:32])] if city == "Adelaide" else [
            ("train", order[:128]), ("validation", order[128:144])]
        for split, indices in selections:
            for index in indices:
                pano, views = entries[int(index)]
                view_index = int(rng.integers(4))
                heading = [0, 90, 180, 270][view_index]
                original = views[heading]
                start = (int(index) * 4 + view_index) * 784
                f = np.asarray(features[start:start + 784], dtype=np.float32)
                assert f.shape == (784, 1024) and np.isfinite(f).all()
                image = Image.open(original).convert("RGB").resize((448, 448), Image.Resampling.BILINEAR)
                identifier = f"{city}_{pano}_{heading}"
                relative = f"images/{identifier}.jpg"
                image.save(output / relative, quality=92)
                rgb = np.asarray(image.resize((112, 112), Image.Resampling.BILINEAR), dtype=np.float32) / 255
                all_features.append(f)
                all_rgb.append(rgb)
                records.append({"id": identifier, "city": city, "pano": pano,
                                "heading": heading, "split": split, "image": relative,
                                "source_image": str(original), "cache_start": start,
                                "cache_file": str(cache / f"dict_{city}_p784.f16.npy"),
                                "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest()})
        print(f"loaded {city}: {len(records)} views total", flush=True)
    train = [i for i, r in enumerate(records) if r["split"] == "train"]
    # PCA statistics use training images only; validation cities cannot influence them.
    fit = torch.from_numpy(np.concatenate([all_features[i][::8] for i in train]))
    mean = fit.mean(0)
    centered = fit - mean
    covariance = centered.T @ centered / (len(fit) - 1)
    values, vectors = torch.linalg.eigh(covariance)
    basis = vectors[:, -64:]
    retained = float(values[-64:].sum() / values.clamp_min(0).sum())
    raw = np.stack(all_features)
    projected = (raw - mean.numpy()) @ basis.numpy()
    scale = float(projected[train].std())
    projected /= scale
    np.savez_compressed(output / "dataset.npz", features=projected.astype(np.float16),
                        rgb=np.stack(all_rgb).astype(np.float16), mean=mean.numpy(),
                        basis=basis.numpy(), scale=np.array(scale))
    train_panos = {(records[i]["city"], records[i]["pano"]) for i in train}
    assert not any((r["city"], r["pano"]) in train_panos
                   for r in records if r["split"] != "train")
    manifest = {"seed": seed, "backbone": "DINOv3 ViT-L/16 (existing dense patch cache)",
                "patch_grid": [28, 28], "input_features": 1024, "pca_features": 64,
                "pca_retained_variance": retained, "pca_fit_split": "train",
                "split_unit": "panorama", "labels_used_for_training": False,
                "cache_checks": checks, "records": records,
                "alignment_note": "Offsets follow the ordered panorama manifest, then headings 0/90/180/270. No historical QC removals in the selected cities. Source cache is reused; backbone was not rerun."}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"views": len(records), "train": len(train),
                      "retained_variance": retained}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/host/mnt/nas/huangyj/GoogleSV/urban_visual_gene"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data/dinov3")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.source, args.output, args.seed)
