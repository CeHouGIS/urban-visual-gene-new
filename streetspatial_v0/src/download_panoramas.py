"""Download full equirectangular panoramas for a StreetSpatial follow-up run."""

import argparse
import io
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageStat
import requests


ROOT = Path(__file__).resolve().parents[1]
NAS_ROOT = Path("/host/mnt/nas/urban-visual-gene/streetspatial_v0/data/panoramas_v0_2")
PILOT = Path("/host/mnt/nas/urban-visual-gene/streetspatial_v0/data/splits/pilot200_final/transformations.jsonl")
SOURCE = Path("/host/mnt/nas/urban-visual-gene/urbangene_v2_20261109/outputs/v2/data/real20k_20261110/images.train.jsonl")
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
LEGACY_TILE_COLUMNS = 26
LEGACY_TILE_ROWS = 13
LEGACY_TILE_SIZE = 512
TILE_HEADERS = {
    "Referer": "https://www.google.com/maps/",
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "Chrome/125.0.0.0 Safari/537.36"),
}


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def image_identity(image_id):
    city_and_panoid, separator, heading = image_id.rpartition("_")
    if not separator or heading not in {"0", "90", "180", "270"}:
        raise ValueError(f"Unexpected image ID: {image_id}")
    city, separator, panoid = city_and_panoid.partition("_")
    if not separator or not ID_PATTERN.fullmatch(city) or not ID_PATTERN.fullmatch(panoid):
        raise ValueError(f"Unsafe image ID: {image_id}")
    return city, panoid


def prepare_targets(pilot_path, source_path, count, seed):
    chosen = {}
    for row in read_jsonl(pilot_path):
        if row["transformation"] != "original":
            continue
        city, panoid = image_identity(row["image_id"])
        chosen[(city, panoid)] = {"city": city, "panoid": panoid,
                                   "pilot_scene_id": row["scene_id"]}
    if count < len(chosen):
        raise ValueError(f"Count {count} is smaller than the {len(chosen)} pilot panoramas")
    pool = {}
    for row in read_jsonl(source_path):
        city, panoid = image_identity(row["id"])
        if (city, panoid) not in chosen:
            pool[(city, panoid)] = {"city": city, "panoid": panoid,
                                     "pilot_scene_id": None}
    needed = count - len(chosen)
    if needed > len(pool):
        raise ValueError(f"Only {len(pool)} additional panorama IDs available")
    extras = random.Random(seed).sample(sorted(pool), needed)
    return list(chosen.values()) + [pool[key] for key in extras]


def valid_panorama(path, expected_size=None):
    try:
        with Image.open(path) as image:
            width, height = image.size
            if ((expected_size is not None and (width, height) != expected_size) or
                    width < 1024 or height < 512 or abs(width / height - 2) >= 0.08):
                return False
            image.draft("RGB", (256, 128))
            image.load()
            return max(ImageStat.Stat(image.convert("RGB")).stddev) >= 6
    except (OSError, ValueError):
        return False


def validate_target(row):
    city, panoid = row.get("city"), row.get("panoid")
    if not isinstance(city, str) or not isinstance(panoid, str):
        raise ValueError(f"Invalid panorama target: {row}")
    if not ID_PATTERN.fullmatch(city) or not ID_PATTERN.fullmatch(panoid):
        raise ValueError(f"Unsafe panorama target: {row}")


def legacy_tile_url(panoid, x, y):
    # The historical cbk0 endpoint is unavailable here; this is its current tile API.
    return ("https://streetviewpixels-pa.googleapis.com/v1/tile"
            f"?cb_client=maps_sv.tactile&panoid={panoid}&zoom=5&x={x}&y={y}")


def fetch_legacy_tile(panoid, x, y):
    url = legacy_tile_url(panoid, x, y)
    for attempt in range(3):
        try:
            response = requests.get(url, headers=TILE_HEADERS, timeout=20)
            response.raise_for_status()
            with Image.open(io.BytesIO(response.content)) as tile:
                tile.load()
                if tile.width > LEGACY_TILE_SIZE or tile.height > LEGACY_TILE_SIZE:
                    raise ValueError(f"Oversized tile at {x},{y}: {tile.size}")
                if len(response.content) < 4096 and tile.getextrema() == (0, 0):
                    raise ValueError(f"Blank Street View placeholder tile at {x},{y}")
            return response.content
        except (requests.RequestException, OSError) as error:
            if attempt == 2:
                raise RuntimeError(f"Tile {x},{y} failed after 3 attempts: {error}") from error


def stitch_legacy_tiles(panoid, destination, tile_workers=8,
                        columns=LEGACY_TILE_COLUMNS, rows=LEGACY_TILE_ROWS,
                        tile_size=LEGACY_TILE_SIZE):
    """Fetch the historical 26x13 tile grid and paste it into one 2:1 image."""
    positions = [(x, y) for x in range(columns) for y in range(rows)]
    panorama = Image.new("RGB", (columns * tile_size, rows * tile_size))

    def paste_tile(x, y, content):
        with Image.open(io.BytesIO(content)) as tile:
            tile.load()
            if not (0 < tile.width <= tile_size and 0 < tile.height <= tile_size):
                raise ValueError(f"Invalid tile size at {x},{y}: {tile.size}")
            if (tile.width != tile_size and x != columns - 1 or
                    tile.height != tile_size and y != rows - 1):
                raise ValueError(f"Incomplete interior tile at {x},{y}: {tile.size}")
            panorama.paste(tile.convert("RGB"), (x * tile_size, y * tile_size))

    try:
        # Probe one tile before submitting the full grid when the endpoint is down.
        paste_tile(0, 0, fetch_legacy_tile(panoid, 0, 0))
        with ThreadPoolExecutor(max_workers=tile_workers) as executor:
            futures = {executor.submit(fetch_legacy_tile, panoid, x, y): (x, y)
                       for x, y in positions[1:]}
            try:
                for future in as_completed(futures):
                    x, y = futures[future]
                    paste_tile(x, y, future.result())
            except Exception:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
        panorama.save(destination, "JPEG", quality=92)
    finally:
        panorama.close()


def download_one(row, outdir, zoom, backend="tiles", tile_workers=8):

    city, panoid = row["city"], row["panoid"]
    destination = outdir / city / f"{panoid}.jpg"
    expected_size = ((LEGACY_TILE_COLUMNS * LEGACY_TILE_SIZE,
                      LEGACY_TILE_ROWS * LEGACY_TILE_SIZE)
                     if backend == "tiles" else None)
    if valid_panorama(destination, expected_size):
        return {**row, "backend": backend, "status": "existing", "path": str(destination)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part.jpg")
    try:
        if backend == "tiles":
            stitch_legacy_tiles(panoid, temporary, tile_workers=tile_workers)
        else:
            from streetlevel import streetview
            pano = streetview.find_panorama_by_id(panoid)
            if pano is None:
                raise ValueError("Panorama ID no longer resolves")
            streetview.download_panorama(pano, str(temporary), zoom=zoom,
                                         pil_args={"quality": 92})
        if not valid_panorama(temporary, expected_size):
            raise ValueError("Downloaded panorama is blank, corrupt, or not 2:1")
        with Image.open(temporary) as image:
            width, height = image.size
        os.replace(temporary, destination)
        return {**row, "backend": backend, "status": "downloaded", "path": str(destination),
                "width": width, "height": height}
    except Exception as error:
        temporary.unlink(missing_ok=True)
        return {**row, "backend": backend, "status": "failed", "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-manifest", type=Path, default=PILOT)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE)
    parser.add_argument("--outdir", type=Path, default=NAS_ROOT)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=["tiles", "streetlevel"], default="tiles",
                        help="tiles uses the historical auto-svi-download zoom-5 tile grid")
    parser.add_argument("--zoom", type=int, default=5,
                        help="streetlevel zoom; tiles always uses zoom 5")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel panoramas (tiles should use one due to image memory)")
    parser.add_argument("--tile-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Attempt only the first N targets for a network smoke test")
    parser.add_argument("--success-count", type=int, default=None,
                        help="Stop after this many valid panoramas; use spare target IDs for failures")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if (args.count <= 0 or not 0 <= args.zoom <= 5 or not 1 <= args.workers <= 8
            or not 1 <= args.tile_workers <= 16
            or args.limit is not None and args.limit <= 0
            or args.success_count is not None and args.success_count <= 0):
        parser.error("count, limit, and success-count must be positive; zoom 0..5, workers 1..8, tile-workers 1..16")
    if args.backend == "tiles" and (args.zoom != 5 or args.workers != 1):
        parser.error("tiles requires --zoom 5 and --workers 1")
    if args.success_count is not None and args.backend != "tiles":
        parser.error("success-count currently requires the sequential tiles backend")
    args.outdir.mkdir(parents=True, exist_ok=True)
    targets_path = args.outdir / "targets.jsonl"
    if targets_path.exists():
        targets = list(read_jsonl(targets_path))
        if len(targets) != args.count:
            raise ValueError("Existing targets.jsonl count differs; use a new outdir")
    else:
        targets = prepare_targets(args.pilot_manifest, args.source_manifest,
                                  args.count, args.seed)
        with targets_path.open("w", encoding="utf-8") as handle:
            for row in targets:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Prepared {len(targets)} panorama IDs in {targets_path}", flush=True)
    for row in targets:
        validate_target(row)
    if args.prepare_only:
        return
    if args.limit is not None:
        targets = targets[:args.limit]
    if args.success_count is not None and args.success_count > len(targets):
        parser.error("success-count exceeds the number of available targets")
    results_path = args.outdir / "download_results.jsonl"
    with results_path.open("a", encoding="utf-8") as handle:
        def record(result, index):
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if result["status"] == "failed":
                print(f"{index}/{len(targets)} failed: {result['city']}/{result['panoid']}: {result['error']}", flush=True)
            elif index % 25 == 0:
                print(f"{index}/{len(targets)} processed", flush=True)
        if args.backend == "tiles":
            successful = []
            for index, row in enumerate(targets, 1):
                result = download_one(row, args.outdir, args.zoom, args.backend,
                                      args.tile_workers)
                record(result, index)
                if result["status"] != "failed":
                    successful.append(result)
                if index == min(3, len(targets)) and not successful:
                    raise SystemExit("First panorama requests all failed; inspect download_results.jsonl before retrying")
                if args.success_count is not None and len(successful) == args.success_count:
                    selected = args.outdir / f"selected_{args.success_count}.jsonl"
                    with selected.open("w", encoding="utf-8") as selected_file:
                        for item in successful:
                            selected_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                    print(f"Selected {len(successful)} complete panoramas: {selected}", flush=True)
                    break
            else:
                if args.success_count is not None:
                    raise SystemExit(f"Only {len(successful)}/{args.success_count} panoramas succeeded; add spare targets and resume")
        else:
            probe_count = min(3, len(targets))
            probes = [download_one(row, args.outdir, args.zoom, args.backend,
                                   args.tile_workers) for row in targets[:probe_count]]
            for index, result in enumerate(probes, 1):
                record(result, index)
            if all(result["status"] == "failed" for result in probes):
                raise SystemExit("First panorama requests all failed; inspect download_results.jsonl before retrying")
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(download_one, row, args.outdir, args.zoom,
                                           args.backend, args.tile_workers)
                           for row in targets[probe_count:]]
                for index, future in enumerate(as_completed(futures), probe_count + 1):
                    record(future.result(), index)
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
