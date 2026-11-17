#!/usr/bin/env python3
"""Verify real RGB files and preserve the existing panorama-level split.

This imports image identities, never historical feature arrays or annotations.
The old cross-city test RGB is deliberately not opened.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from urbangene_v2.common import fresh_dir, sha256, write_json


def prepare(manifest, output):
    manifest = Path(manifest).resolve()
    source = json.loads(manifest.read_text())
    selected = [r for r in source["records"] if r["split"] in {"train", "validation"}]
    if Counter(r["split"] for r in selected) != {"train": 512, "validation": 64}:
        raise ValueError("Expected the established 512/64 real-street split")
    identities, hashes, records, sizes = {}, {}, {}, Counter()
    for split in ("train", "validation"):
        rows = [r for r in selected if r["split"] == split]
        identities[split] = {(r["city"], r["pano"]) for r in rows}
        if len(identities[split]) != len(rows):
            raise ValueError("More than one view selected per panorama")
        hashes[split] = set()
        records[split] = []
        for r in rows:
            image = Path(r["source_image"]).resolve()
            digest = sha256(image)
            if digest != r["source_sha256"]:
                raise ValueError(f"Original image changed: {r['id']}")
            if digest in hashes[split]:
                raise ValueError("Duplicate image within split")
            hashes[split].add(digest)
            with Image.open(image) as opened:
                opened.convert("RGB").load()
                sizes[str(opened.size)] += 1
            records[split].append({"id": r["id"], "image": str(image), "split": split})
        print(json.dumps({"verified": split, "images": len(rows)}), flush=True)
    if identities["train"] & identities["validation"] or hashes["train"] & hashes["validation"]:
        raise ValueError("Train/validation overlap")
    output = fresh_dir(output)
    for split, rows in records.items():
        (output / f"images.{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
    # First two validation samples per city; no result-dependent selection.
    cities = list(dict.fromkeys(r["city"] for r in selected))
    examples = [r for city in cities for r in selected
                if r["city"] == city and r["split"] == "validation"]
    examples = [r for city in cities for r in
                [r for r in examples if r["city"] == city][:2]]
    canvas = Image.new("RGB", (4 * 288, 2 * 310), "white")
    draw = ImageDraw.Draw(canvas)
    for i, r in enumerate(examples):
        x, y = (i // 2) * 288, (i % 2) * 310
        with Image.open(r["source_image"]) as opened:
            preview = opened.convert("RGB").resize((280, 280), Image.Resampling.BICUBIC)
        canvas.paste(preview, (x + 4, y + 26))
        draw.text((x + 8, y + 6), f"{r['city']} | validation {i % 2 + 1}", fill="black")
    canvas.save(output / "real_street_inputs.png")
    provenance = [{k: r[k] for k in ("id", "city", "pano", "heading", "split",
                  "source_image", "source_sha256")} for r in selected]
    write_json(output / "provenance.json", {
        "dataset_kind": "real_google_street_view_photographs",
        "source_manifest": str(manifest), "source_manifest_sha256": sha256(manifest),
        "split_counts": {k: len(v) for k, v in records.items()},
        "city_counts": {split: dict(Counter(r["city"] for r in selected if r["split"] == split))
                        for split in records},
        "split_unit": "panorama", "one_view_per_panorama": True,
        "panorama_overlap": 0, "image_hash_overlap": 0, "image_sizes": dict(sizes),
        "historical_feature_cache_reused": False,
        "historical_test_rgb_opened": False,
        "training_annotations_used": False,
        "preview_selection": "first two validation samples per city in manifest order",
        "preview_ids": [r["id"] for r in examples], "records": provenance})
    print(json.dumps({"prepared": str(output), "train": 512, "validation": 64}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source-manifest", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    prepare(a.source_manifest, a.output)
