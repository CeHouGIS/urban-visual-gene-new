"""Sample Google Street View panorama IDs across the available world metadata."""

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as parquet


GSV_ROOT = Path("/host/mnt/nas/huangyj/GoogleSV")
OUTPUT = Path("/host/mnt/nas/urban-visual-gene/streetspatial_v0/data/global200_panoramas")
PANOID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def safe_city(country, city):
    def ascii_part(value):
        normalized = unicodedata.normalize("NFKD", value)
        return re.sub(r"[^A-Za-z0-9]+", "-", normalized.encode("ascii", "ignore").decode()).strip("-")[:24]

    digest = hashlib.sha1(f"{country}/{city}".encode()).hexdigest()[:8]
    return f"{ascii_part(country)}_{ascii_part(city)}_{digest}"


def city_catalog(root):
    cities = defaultdict(list)
    with (root / "city_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["Total_Panoids"]) > 0:
                cities[row["Country"]].append(row["City"])
    return dict(cities)


def random_panoid(path, rng):
    source = parquet.ParquetFile(path)
    if source.metadata.num_rows == 0:
        raise ValueError(f"Empty panorama table: {path}")
    columns = [name for name in ("panoid", "lat", "lon", "year", "month")
               if name in source.schema.names]
    row_index = rng.randrange(source.metadata.num_rows)
    for group in range(source.metadata.num_row_groups):
        group_size = source.metadata.row_group(group).num_rows
        if row_index < group_size:
            return source.read_row_group(group, columns=columns).slice(row_index, 1).to_pylist()[0]
        row_index -= group_size
    raise AssertionError("Parquet row-group counts do not match the total")


def sample_targets(root, count, seed):
    rng = random.Random(seed)
    cities = city_catalog(root)
    countries = sorted(cities)
    if not countries:
        raise ValueError("No countries in city catalog")
    selected, seen = [], set()
    failures = defaultdict(int)
    rounds = []
    while len(selected) < count:
        if not rounds:
            rounds = countries.copy()
            rng.shuffle(rounds)
        country = rounds.pop()
        city = rng.choice(cities[country])
        path = root / "metadata" / country / city / "panoids.parquet"
        if not path.is_file():
            failures["missing_parquet"] += 1
            continue
        try:
            metadata = random_panoid(path, rng)
        except (OSError, ValueError):
            failures["unreadable_parquet"] += 1
            continue
        panoid = metadata.get("panoid")
        if not isinstance(panoid, str) or not PANOID_PATTERN.fullmatch(panoid):
            failures["invalid_panoid"] += 1
            continue
        if panoid in seen:
            failures["duplicate_panoid"] += 1
            continue
        seen.add(panoid)
        selected.append({
            "city": safe_city(country, city), "country": country,
            "source_city": city, "panoid": panoid,
            "lat": metadata.get("lat"), "lon": metadata.get("lon"),
            "year": metadata.get("year"), "month": metadata.get("month"),
            "heading": rng.randrange(360), "sample_index": len(selected),
        })
        if len(selected) % 25 == 0:
            print(f"Sampled {len(selected)}/{count} panorama IDs", flush=True)
    return selected, dict(failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=GSV_ROOT)
    parser.add_argument("--outdir", type=Path, default=OUTPUT)
    parser.add_argument("--count", type=int, default=240,
                        help="candidate IDs, including replacements for 200 successful downloads")
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--extend", action="store_true",
                        help="extend an existing manifest after verifying its deterministic prefix")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("count must be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)
    targets_path = args.outdir / "targets.jsonl"
    existing = []
    if targets_path.exists():
        if not args.extend:
            raise FileExistsError(targets_path)
        with targets_path.open(encoding="utf-8") as handle:
            existing = [json.loads(line) for line in handle if line.strip()]
        if len(existing) >= args.count:
            raise ValueError("Existing target count already meets or exceeds --count")
    targets, failures = sample_targets(args.root, args.count, args.seed)
    if targets[:len(existing)] != existing:
        raise ValueError("Existing targets differ from this seed; refusing to replace them")
    with targets_path.open("w", encoding="utf-8") as handle:
        for target in targets:
            handle.write(json.dumps(target, ensure_ascii=False) + "\n")
    summary = {"seed": args.seed, "candidate_count": len(targets),
               "countries": len({row["country"] for row in targets}),
               "cities": len({row["city"] for row in targets}),
               "sampling": "shuffled country cycles; uniform city and panorama within country",
               "metadata_root": str(args.root / "metadata"), "skipped": failures}
    (args.outdir / "sampling.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
