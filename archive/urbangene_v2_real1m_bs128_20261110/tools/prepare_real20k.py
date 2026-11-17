#!/usr/bin/env python3
"""Deterministically expand four real-street cities, preserving old validation."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from urbangene_v2.common import fresh_dir, sha256, write_json

CITIES = ("Austin", "Bogota", "Lagos", "London")


def verify(row):
    path = Path(row["source_image"])
    digest = sha256(path)
    if row.get("source_sha256") and digest != row["source_sha256"]:
        raise ValueError(f"Previously verified image changed: {row['id']}")
    with Image.open(path) as im:
        im.convert("RGB").load()
        size = list(im.size)
    return {**row, "source_sha256": digest, "source_size": size}


def prepare(source_directory, old_manifest, output, seed=42):
    output = fresh_dir(output)
    old = json.loads(Path(old_manifest).read_text())
    old_records = [r for r in old["records"] if r["split"] in {"train", "validation"}]
    fields = ("id", "city", "pano", "heading", "split", "source_image", "source_sha256")
    old_records = [{k: r[k] for k in fields} for r in old_records]
    if Counter(r["split"] for r in old_records) != {"train": 512, "validation": 64}:
        raise ValueError("Expected the 512/64 original split")
    groups = defaultdict(dict)
    parts = sorted(Path(source_directory).glob("formal_sample_images.part*.tsv"))
    for part in parts:
        with part.open() as stream:
            for line in stream:
                city, name = line.rstrip("\n").split("\t", 1)
                if city not in CITIES:
                    continue
                pano, heading = Path(name).stem.rsplit("_", 1)
                groups[city].setdefault(pano, {})[int(heading)] = name.replace(
                    "/host/root/mnt/nas/", "/host/mnt/nas/")
    if any(len(groups[c]) < 2628 for c in CITIES):
        raise ValueError("Not enough panoramas for disjoint 2500/128 split per city")
    rng = np.random.default_rng(seed)
    rows = {s: [r for r in old_records if r["split"] == s] for s in ("train", "validation")}
    sizes, used_hashes, rejected = Counter(), set(), []
    with ThreadPoolExecutor(max_workers=8) as pool:
        checked = list(pool.map(verify, rows["train"] + rows["validation"]))
        if len({r["source_sha256"] for r in checked}) != len(checked):
            raise ValueError("Original split has duplicate RGB files")
        used_hashes.update(r["source_sha256"] for r in checked)
        sizes.update(str(tuple(r["source_size"])) for r in checked)
        rows = {s: [r for r in checked if r["split"] == s] for s in rows}

        def make(city, pano, heading, split):
            return {"id": f"{city}_{pano}_{heading}", "city": city, "pano": pano,
                    "heading": heading, "split": split,
                    "source_image": groups[city][pano][heading]}

        def add_batch(candidates):
            try:
                verified = list(pool.map(verify, candidates))
                digests = [r["source_sha256"] for r in verified]
                if len(set(digests)) != len(digests) or used_hashes.intersection(digests):
                    raise ValueError("Duplicate image content")
            except (OSError, ValueError) as error:
                rejected.append({"ids": [r["id"] for r in candidates], "reason": str(error)})
                return False
            for r in verified:
                rows[r["split"]].append(r)
                sizes[str(tuple(r["source_size"]))] += 1
                used_hashes.add(r["source_sha256"])
            return True

        for city in CITIES:
            existing_train = [r for r in rows["train"] if r["city"] == city]
            existing_val = [r for r in rows["validation"] if r["city"] == city]
            forbidden = {r["pano"] for r in existing_train + existing_val}
            choices = sorted(set(groups[city]) - forbidden)
            choices = [choices[int(i)] for i in rng.permutation(len(choices))]
            # Complete the protected training panoramas with an opposite view.
            for r in existing_train:
                if not add_batch([make(city, r["pano"], (r["heading"] + 180) % 360, "train")]):
                    raise ValueError("Could not add the second view of a protected training panorama")
            position = 0
            while sum(r["city"] == city for r in rows["validation"]) < 128:
                pano = choices[position]
                position += 1
                add_batch([make(city, pano, int(rng.choice([0, 90, 180, 270])), "validation")])
            added_panos = 128
            while added_panos < 2500:
                if position >= len(choices):
                    raise ValueError("Exhausted candidates while verifying images")
                pano = choices[position]
                position += 1
                heading = int(rng.choice([0, 90, 180, 270]))
                if add_batch([make(city, pano, heading, "train"),
                              make(city, pano, (heading + 180) % 360, "train")]):
                    added_panos += 1
                if added_panos % 250 == 0:
                    write_json(output / "preparation_progress.json", {
                        "status": "verifying", "city": city,
                        "train_images": len(rows["train"]), "validation_images": len(rows["validation"]),
                        "rejected_groups": len(rejected)})
            print(json.dumps({"city": city, "train_images": 5000, "validation_images": 128}), flush=True)
    assert len(rows["train"]) == 20000 and len(rows["validation"]) == 512
    sets = {s: {(r["city"], r["pano"]) for r in items} for s, items in rows.items()}
    assert len(sets["train"]) == 10000 and len(sets["validation"]) == 512
    assert not sets["train"] & sets["validation"]
    assert set(Counter((r["city"], r["pano"]) for r in rows["train"]).values()) == {2}
    for split, items in rows.items():
        (output / f"images.{split}.jsonl").write_text("".join(json.dumps({
            "id": r["id"], "image": r["source_image"], "split": split}) + "\n" for r in items))
    write_json(output / "provenance.json", {
        "dataset_kind": "real_google_street_view_photographs", "seed": seed,
        "split_counts": {s: len(v) for s, v in rows.items()},
        "panorama_counts": {s: len(v) for s, v in sets.items()},
        "city_counts": {s: dict(Counter(r["city"] for r in v)) for s, v in rows.items()},
        "source_manifest": str(Path(old_manifest).resolve()), "source_manifest_sha256": sha256(old_manifest),
        "source_inventory_files": {str(p): sha256(p) for p in parts},
        "split_unit": "panorama", "train_views_per_panorama": 2,
        "view_pair": "opposite headings", "panorama_overlap": 0, "image_hash_overlap": 0,
        "original_training_prefix": 512, "original_validation_prefix": 64,
        "preview_ids": [r["id"] for city in CITIES for r in
                        [r for r in rows["validation"] if r["city"] == city][:2]],
        "test_images_opened": False, "test_city_used": False, "training_annotations_used": False,
        "image_sizes": dict(sizes), "rejected_groups": rejected,
        "records": rows["train"] + rows["validation"]})
    write_json(output / "preparation_progress.json", {"status": "complete", "train_images": 20000,
               "validation_images": 512, "unique_training_panoramas": 10000})
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source-directory", required=True)
    p.add_argument("--old-manifest", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    prepare(a.source_directory, a.old_manifest, a.output)
