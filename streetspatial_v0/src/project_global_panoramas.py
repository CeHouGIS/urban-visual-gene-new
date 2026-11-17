"""Project downloaded 360-degree panoramas into fixed perspective source views."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import py360convert
from PIL import Image

from .build_dataset import write_jsonl


DATA_ROOT = Path("/host/mnt/nas/urban-visual-gene/streetspatial_v0/data")
PANORAMA_ROOT = DATA_ROOT / "global200_panoramas"
OUTPUT_ROOT = DATA_ROOT / "global200_sources"
SOURCE_SIZE = (800, 600)
PANORAMA_WORK_SIZE = (4096, 2048)
HORIZONTAL_FOV = 90.0
VERTICAL_FOV = math.degrees(2 * math.atan(SOURCE_SIZE[1] / SOURCE_SIZE[0]))
HEADING_OFFSETS = (0, 90, 180, 270)


def load_panorama(path):
    with Image.open(path) as source:
        source.draft("RGB", PANORAMA_WORK_SIZE)
        resized = source.convert("RGB").resize(PANORAMA_WORK_SIZE, Image.Resampling.LANCZOS)
    return np.asarray(resized)


def project(panorama, heading):
    perspective = py360convert.e2p(
        panorama, (HORIZONTAL_FOV, VERTICAL_FOV),
        u_deg=float(heading), v_deg=0.0,
        out_hw=(SOURCE_SIZE[1], SOURCE_SIZE[0]), mode="bilinear",
    )
    return Image.fromarray(perspective, mode="RGB")


def source_records(target, panorama_path, output_dir):
    city, panoid = target["city"], target["panoid"]
    scene = f"{city}_{panoid}"
    for offset in HEADING_OFFSETS:
        heading = (target["heading"] + offset) % 360
        image_id = f"{scene}_{offset}"
        destination = output_dir / city / f"{panoid}_{offset}.jpg"
        yield destination, {
            "image_id": image_id, "scene_id": scene, "city_id": city,
            "image_path": str(destination), "image_size": list(SOURCE_SIZE),
            "panorama_path": str(panorama_path), "country": target["country"],
            "source_city": target["source_city"], "lat": target.get("lat"),
            "lon": target.get("lon"), "heading": heading, "pitch": 0,
            "horizontal_fov": HORIZONTAL_FOV, "vertical_fov": VERTICAL_FOV,
            "objects": [],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=PANORAMA_ROOT / "selected_200.jsonl")
    parser.add_argument("--panorama-root", type=Path, default=PANORAMA_ROOT)
    parser.add_argument("--outdir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    targets = [json.loads(line) for line in args.targets.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        targets = targets[:args.limit]
    records = []
    for index, target in enumerate(targets, 1):
        panorama_path = args.panorama_root / target["city"] / f"{target['panoid']}.jpg"
        if not panorama_path.is_file():
            if args.limit is None:
                raise FileNotFoundError(panorama_path)
            continue
        options = list(source_records(target, panorama_path, args.outdir / "images"))
        panorama = load_panorama(panorama_path) if any(not path.exists() for path, _ in options) else None
        for destination, record in options:
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                image = project(panorama, record["heading"])
                image.save(destination, "JPEG", quality=95)
            records.append(record)
        if index % 25 == 0:
            print(f"Projected {index}/{len(targets)} panoramas", flush=True)
    manifest = args.outdir / ("candidates.jsonl" if args.limit is None else "smoke_candidates.jsonl")
    write_jsonl(manifest, records)
    print(f"Projected {len(records)} perspective candidates from {len(records) // 4} panoramas: {manifest}")


if __name__ == "__main__":
    main()
