#!/usr/bin/env python3
"""Build resumable, content-checked RGB shards for one million real images."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from urbangene_v2.common import load_config, sha256, writable, write_json


def inspect_rgb(row):
    try:
        raw = Path(row["image"]).read_bytes()
        with Image.open(io.BytesIO(raw)) as opened:
            opened.convert("RGB").load()
            size = list(opened.size)
        return {**row, "sha256": hashlib.sha256(raw).hexdigest(), "size": size}
    except (OSError, ValueError) as error:
        return {**row, "error": str(error)}


def stable_key(seed, city, pano):
    return hashlib.sha256(f"{seed}/{city}/{pano}".encode()).digest()


def prepare(config_path):
    cfg = load_config(config_path)
    directory = writable(cfg["data_root"])
    directory.mkdir(exist_ok=True, parents=True)
    state_path = directory / "preparation.json"
    if (directory / "READY.json").exists():
        print("Data already complete", flush=True)
        return
    seed = cfg["seed"]
    groups = defaultdict(dict)
    inventories = sorted(Path(cfg["inventory_root"]).glob("qc_part*/keep.part0000.txt"))
    if not inventories:
        raise FileNotFoundError("Quality-screened image inventories not found")
    write_json(state_path, {"status": "indexing_inventory", "training_target": 1000000})
    for part in inventories:
        with part.open() as stream:
            for line in stream:
                city, name = line.rstrip("\n").split("\t", 1)
                if city in cfg["excluded_cities"]:
                    continue
                parts = Path(name).stem.rsplit("_", 1)
                if len(parts) != 2 or not parts[1].isdigit() or int(parts[1]) not in {0, 90, 180, 270}:
                    continue
                pano, heading = parts
                groups[city].setdefault(pano, {})[int(heading)] = name.replace(
                    "/host/root/mnt/nas/", "/host/mnt/nas/")
    # Old validation identities are kept; their entire panoramas are excluded from training.
    old = json.loads(Path(cfg["comparison_provenance"]).read_text())
    protected = [r for r in old["records"] if r["split"] == "validation"]
    protected_panos = {(r["city"], r["pano"]) for r in protected}
    old_train_panos = {(r["city"], r["pano"]) for r in old["records"] if r["split"] == "train"}
    candidates = {}
    for city, entries in groups.items():
        eligible = [(pano, views) for pano, views in entries.items()
                    if (city, pano) not in protected_panos and
                    any((h + 180) % 360 in views for h in views)]
        if len(eligible) >= 1800:
            candidates[city] = sorted(eligible, key=lambda p: stable_key(seed, city, p[0]))
    cities = sorted(candidates)
    if len(cities) < 250:
        raise ValueError("Insufficient city coverage for one-million-image target")
    del groups
    db = sqlite3.connect(directory / "verification.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS accepted (id TEXT PRIMARY KEY, city TEXT, pano TEXT, "
               "split TEXT, sha256 TEXT UNIQUE, row_json TEXT, position INTEGER UNIQUE)")
    db.execute("CREATE TABLE IF NOT EXISTS rejected (id TEXT PRIMARY KEY, reason TEXT)")
    accepted = {s: [] for s in ("train", "validation")}
    hashes, panos = set(), {s: set() for s in accepted}
    for split, digest, row_json in db.execute("SELECT split,sha256,row_json FROM accepted ORDER BY position"):
        row = json.loads(row_json)
        accepted[split].append(row)
        hashes.add(digest)
        panos[split].add((row["city"], row["pano"]))
    seen_ids = {r["id"] for rows in accepted.values() for r in rows}
    position = len(seen_ids)
    rejected_count = db.execute("SELECT COUNT(*) FROM rejected").fetchone()[0]
    shard_size = cfg["data_shard_size"]

    def emit(split, finish=False):
        rows = accepted[split]
        destination = directory / "rgb_shards" / split
        destination.mkdir(parents=True, exist_ok=True)
        complete = (len(rows) // shard_size) * shard_size
        if finish:
            complete = len(rows)
        for start in range(0, complete, shard_size):
            end = min(start + shard_size, complete)
            if not finish and end - start != shard_size:
                continue
            path = destination / f"images_{start:07d}_{end:07d}.jsonl"
            if path.exists():
                continue
            subset = rows[start:end]
            raw = "".join(json.dumps({"id": r["id"], "image": r["image"], "split": split}) + "\n"
                          for r in subset)
            temporary = path.with_suffix(".jsonl.partial")
            temporary.write_text(raw)
            write_json(path.with_suffix(".provenance.json"), {
                "split": split, "start": start, "end": end, "records": subset,
                "image_checks": "SHA256, RGB decode, unique file content; no object annotations"})
            temporary.replace(path)

    def add(checked):
        nonlocal position, rejected_count
        new = [r for r in checked if r["id"] not in seen_ids]
        if not new:
            return True
        reason = next((r["error"] for r in new if "error" in r), None)
        digests = [r.get("sha256") for r in new]
        if not reason and (len(set(digests)) != len(digests) or hashes.intersection(digests)):
            reason = "duplicate RGB content"
        if reason:
            for r in new:
                db.execute("INSERT OR REPLACE INTO rejected VALUES (?,?)", (r["id"], reason))
            db.commit()
            rejected_count += len(new)
            return False
        for r in new:
            other = "validation" if r["split"] == "train" else "train"
            if (r["city"], r["pano"]) in panos[other]:
                raise ValueError("Panorama leakage")
            db.execute("INSERT INTO accepted VALUES (?,?,?,?,?,?,?)",
                       (r["id"], r["city"], r["pano"], r["split"], r["sha256"], json.dumps(r), position))
            accepted[r["split"]].append(r)
            seen_ids.add(r["id"])
            hashes.add(r["sha256"])
            panos[r["split"]].add((r["city"], r["pano"]))
            position += 1
        return True

    def row(city, pano, heading, views, split):
        return {"id": f"{city}_{pano}_{heading}", "city": city, "pano": pano,
                "heading": heading, "image": views[heading], "split": split}

    with ThreadPoolExecutor(max_workers=cfg["verification_workers"]) as pool:
        if len(accepted["validation"]) < len(protected):
            checked = list(pool.map(inspect_rgb, [{"id": r["id"], "image": r["source_image"],
                       "city": r["city"], "pano": r["pano"], "heading": r["heading"],
                       "split": "validation"} for r in protected]))
            if not add(checked):
                raise ValueError("Protected validation content invalid")
            db.commit()
        # Add a deterministic city-balanced validation set, excluding old training panoramas.
        needed = cfg["validation_images"] - len(protected)
        for city_index, city in enumerate(cities):
            target = needed // len(cities) + int(city_index < needed % len(cities))
            fixed_count = sum(r["city"] == city for r in protected)
            count = sum(r["city"] == city for r in accepted["validation"]) - fixed_count
            for pano, views in candidates[city]:
                if count >= target:
                    break
                key = (city, pano)
                if key in panos["validation"] or key in old_train_panos:
                    continue
                heading = sorted(views)[stable_key(seed + 1, city, pano)[0] % len(views)]
                if add(list(pool.map(inspect_rgb, [row(city, pano, heading, views, "validation")]))):
                    count += 1
            if count != target:
                raise ValueError("Validation target not reached")
        db.commit()
        emit("validation", finish=True)
        write_json(directory / "validation_ready.json", {
            "images": len(accepted["validation"]), "original_validation_prefix": len(protected)})
        needed_panos = cfg["train_images"] // 2
        cursors = {city: 0 for city in cities}
        city_counts = Counter(r["city"] for r in accepted["train"])
        targets = {city: needed_panos // len(cities) + int(i < needed_panos % len(cities))
                   for i, city in enumerate(cities)}
        # Round robin keeps every committed shard geographically mixed.
        while len(accepted["train"]) < cfg["train_images"]:
            pair_groups = []
            for city in cities:
                if city_counts[city] // 2 >= targets[city]:
                    continue
                choices = candidates[city]
                while cursors[city] < len(choices):
                    pano, views = choices[cursors[city]]
                    cursors[city] += 1
                    key = (city, pano)
                    if key in panos["train"] or key in panos["validation"] or key in protected_panos:
                        continue
                    headings = sorted(h for h in views if (h + 180) % 360 in views)
                    heading = headings[stable_key(seed + 2, city, pano)[0] % len(headings)]
                    pair_groups.append([row(city, pano, heading, views, "train"),
                                        row(city, pano, (heading + 180) % 360, views, "train")])
                    break
                else:
                    raise ValueError(f"Training candidates exhausted: {city}")
            checked = list(pool.map(inspect_rgb, [r for pair in pair_groups for r in pair]))
            for i in range(0, len(checked), 2):
                pair = checked[i:i + 2]
                if add(pair):
                    city_counts[pair[0]["city"]] += 2
            db.commit()
            emit("train")
            write_json(state_path, {
                "status": "verifying_images", "train_verified": len(accepted["train"]),
                "validation_verified": len(accepted["validation"]), "train_target": cfg["train_images"],
                "cities": len(cities), "rejected_images": rejected_count,
                "updated_utc": datetime.now(timezone.utc).isoformat()})
        emit("train", finish=True)
    if panos["train"] & panos["validation"]:
        raise ValueError("Final panorama overlap")
    for split in accepted:
        manifest = directory / f"images.{split}.jsonl"
        with manifest.open("w") as stream:
            for r in accepted[split]:
                stream.write(json.dumps({"id": r["id"], "image": r["image"], "split": split}) + "\n")
    summary = {"dataset_kind": "real_google_street_view_photographs", "seed": seed,
        "split_counts": {s: len(v) for s, v in accepted.items()},
        "panorama_counts": {s: len(v) for s, v in panos.items()}, "eligible_cities": cities,
        "city_counts": {s: dict(Counter(r["city"] for r in v)) for s, v in accepted.items()},
        "image_hash_overlap": 0, "panorama_overlap": 0, "training_annotations_used": False,
        "excluded_cities": cfg["excluded_cities"], "old_validation_prefix": len(protected),
        "preview_ids": old["preview_ids"], "source_inventory_hashes": {str(p): sha256(p) for p in inventories},
        "records_database": str(directory / "verification.sqlite"),
        "manifests": {s: {"path": str(directory / f"images.{s}.jsonl"),
                         "sha256": sha256(directory / f"images.{s}.jsonl")} for s in accepted}}
    write_json(directory / "READY.json", summary)
    write_json(state_path, {"status": "complete", **{s: len(v) for s, v in accepted.items()}})
    db.close()
    print(json.dumps({"status": "complete", "splits": summary["split_counts"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    try:
        prepare(args.config)
    except BaseException as error:
        cfg = load_config(args.config)
        write_json(Path(cfg["data_root"]) / "preparation.json", {
            "status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise
