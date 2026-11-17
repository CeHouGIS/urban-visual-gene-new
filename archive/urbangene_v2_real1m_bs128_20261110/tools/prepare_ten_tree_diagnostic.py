#!/usr/bin/env python3
"""Reproduce the separate ten-tree validation stress group; never train on it."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/root/code/urban-visual-gene/urbangene/src")
from urbangene_v2.common import writable, sha256
from urbangene_v2.data import _instance_mask
from urbangene.data.synthetic import SyntheticSceneGenerator, FACTORS, SEEN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    output = writable(args.output)
    if not args.verify_existing:
        output.mkdir(exist_ok=False)
        (output / "images").mkdir()
    renderer = SyntheticSceneGenerator(256)
    records, masks, metas = [], [], []
    for index in range(12):
        spec = dict(zip(FACTORS, SEEN[(index * 2) % len(SEEN)]))
        spec.update(seed=99000000000 + index, split="diagnostic_validation_ten_tree",
                    tree_count=10, road_width=index % 3, building_relation=(index // 3) % 3)
        image, meta = renderer.sample(spec)
        sample_id = f"validation_ten_tree_{index:03d}"
        path = output / f"images/{sample_id}.png"
        if args.verify_existing:
            from PIL import Image
            with Image.open(path) as saved:
                if not np.array_equal(np.asarray(saved), np.asarray(image)):
                    raise ValueError(f"RGB reproduction differs: {path}")
        else:
            image.save(path)
        records.append({"id": sample_id, "image": str(path), "split": "diagnostic_validation_ten_tree"})
        masks.append(_instance_mask(meta, 256))
        metas.append({"id": sample_id, **meta, "mask_building_ids": [1, 2], "mask_tree_ids": list(range(3, 13))})
    manifest = "".join(json.dumps(row) + "\n" for row in records)
    if args.verify_existing:
        if (output / "images.jsonl").read_text() != manifest:
            raise ValueError("Diagnostic manifest differs")
        with np.load(output / "annotations.npz") as saved:
            if not np.array_equal(saved["instance_ids"], masks):
                raise ValueError("Diagnostic instance masks differ")
    else:
        (output / "images.jsonl").write_text(manifest)
        np.savez_compressed(output / "annotations.npz", instance_ids=np.stack(masks),
                            ids=np.asarray([row["id"] for row in records]))
        (output / "evaluation_only_meta.json").write_text(json.dumps({
            "scope": "Additional validation stress group; not a locked test split; no training use",
            "samples": metas}, indent=2) + "\n")
    print(json.dumps({"status": "verified_existing" if args.verify_existing else "generated",
                      "images": len(records), "output": str(output),
                      "script_sha256": sha256(__file__),
                      "annotations_sha256": sha256(output / "annotations.npz")}, indent=2))


if __name__ == "__main__":
    main()
