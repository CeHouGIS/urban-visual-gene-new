"""Download historical London panoramas inside selected 100 m circles."""

import argparse
import json
import math
import re
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .download_panoramas import download_one, read_jsonl, validate_target


HISTORY_DB = Path("/host/mnt/nas/GSV/metadata/London/history_3periods/history_3periods.db")
PANORAMA_DB = Path("/host/mnt/nas/GSV/metadata/London/gsv_50m.db")
DEFAULT_OUTDIR = Path(
    "/host/mnt/nas/urban-visual-gene/streetspatial_v0/data/london_100m_panoramas_3sites"
)
DEFAULT_SAMPLE_IDS = (207192, 262745, 342355)
EARTH_RADIUS_M = 6_371_000.0
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def haversine_m(a, b):
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(value))


def load_anchors(database, sample_ids):
    uri = f"file:{database}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        placeholders = ",".join("?" for _ in sample_ids)
        rows = connection.execute(
            f"SELECT sample_id,lat,lon,status FROM samples WHERE sample_id IN ({placeholders})",
            sample_ids,
        ).fetchall()
    finally:
        connection.close()
    anchors = {row[0]: row for row in rows}
    missing = sorted(set(sample_ids) - anchors.keys())
    unfinished = sorted(row[0] for row in rows if row[3] != "done")
    if missing or unfinished:
        raise ValueError(f"Missing or unfinished anchor samples: missing={missing}, unfinished={unfinished}")
    return [
        {"sample_id": row[0], "lat": row[1], "lon": row[2], "status": row[3]}
        for row in sorted(rows)
    ]


def period_for_year(year):
    if year is None:
        return "unknown"
    if 2010 <= year <= 2015:
        return "2010_2015"
    if 2016 <= year <= 2020:
        return "2016_2020"
    if 2021 <= year <= 2025:
        return "2021_2025"
    return "outside_study_periods"


def select_targets(panorama_database, anchors, radius_m):
    # Use a bounding box to reduce Python distance checks, then apply the exact circle.
    clauses, parameters = [], []
    for anchor in anchors:
        lat_delta = (radius_m + 1) / 110_574
        lon_delta = (radius_m + 1) / (111_320 * math.cos(math.radians(anchor["lat"])))
        clauses.append("(lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?)")
        parameters.extend((
            anchor["lat"] - lat_delta,
            anchor["lat"] + lat_delta,
            anchor["lon"] - lon_delta,
            anchor["lon"] + lon_delta,
        ))

    uri = f"file:{panorama_database}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        query = "SELECT panoid,lat,lon,year,month FROM gsv WHERE " + " OR ".join(clauses)
        rows = connection.execute(query, parameters)
        targets = {}
        for panoid, lat, lon, year, month in rows:
            if not isinstance(panoid, str) or not ID_PATTERN.fullmatch(panoid):
                continue
            memberships = []
            for anchor in anchors:
                distance = haversine_m((anchor["lat"], anchor["lon"]), (lat, lon))
                if distance <= radius_m:
                    memberships.append({
                        "sample_id": anchor["sample_id"],
                        "distance_to_anchor_m": round(distance, 3),
                    })
            if not memberships:
                continue
            target = targets.setdefault(panoid, {
                "city": "London",
                "panoid": panoid,
                "lat": lat,
                "lon": lon,
                "year": year,
                "month": month,
                "period": period_for_year(year),
                "memberships": [],
            })
            target["memberships"].extend(memberships)
    finally:
        connection.close()

    for target in targets.values():
        target["memberships"].sort(key=lambda row: row["sample_id"])
    return sorted(targets.values(), key=lambda row: (
        row["memberships"][0]["sample_id"],
        row["year"] is None,
        row["year"] or 0,
        row["month"] or 0,
        row["panoid"],
    ))


def summarize_targets(anchors, targets, radius_m, panorama_database, history_database):
    sites = []
    for anchor in anchors:
        site_targets = [
            target for target in targets
            if any(member["sample_id"] == anchor["sample_id"] for member in target["memberships"])
        ]
        year_counts = Counter(
            str(target["year"] if target["year"] is not None else "unknown")
            for target in site_targets
        )
        sites.append({
            **anchor,
            "radius_m": radius_m,
            "panorama_count": len(site_targets),
            "year_counts": dict(sorted(year_counts.items())),
        })
    all_years = [target["year"] for target in targets if target["year"] is not None]
    return {
        "city": "London",
        "radius_m": radius_m,
        "anchor_count": len(anchors),
        "unique_panorama_count": len(targets),
        "year_range": [min(all_years), max(all_years)] if all_years else None,
        "period_counts": dict(sorted(Counter(target["period"] for target in targets).items())),
        "source_panorama_database": str(panorama_database),
        "source_anchor_database": str(history_database),
        "sites": sites,
    }


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_or_prepare(outdir, history_db, panorama_db, sample_ids, radius_m):
    targets_path = outdir / "targets.jsonl"
    summary_path = outdir / "sampling_summary.json"
    if targets_path.exists() or summary_path.exists():
        if not targets_path.is_file() or not summary_path.is_file():
            raise FileExistsError(f"Incomplete existing dataset directory: {outdir}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_ids = sorted(sample_ids)
        actual_ids = sorted(site["sample_id"] for site in summary["sites"])
        if summary["radius_m"] != radius_m or actual_ids != expected_ids:
            raise ValueError("Existing manifest uses a different radius or anchor set")
        return list(read_jsonl(targets_path)), summary

    outdir.mkdir(parents=True, exist_ok=False)
    anchors = load_anchors(history_db, sample_ids)
    targets = select_targets(panorama_db, anchors, radius_m)
    if not targets:
        raise ValueError("No panoramas found for the selected anchors")
    summary = summarize_targets(anchors, targets, radius_m, panorama_db, history_db)
    write_jsonl(targets_path, targets)
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return targets, summary


def update_download_summary(outdir, targets):
    results_path = outdir / "download_results.jsonl"
    latest = {}
    if results_path.exists():
        for row in read_jsonl(results_path):
            latest[row["panoid"]] = row
    statuses = Counter(row["status"] for row in latest.values())
    downloaded = sum(row.get("status") in {"downloaded", "existing"} for row in latest.values())
    result = {
        "target_count": len(targets),
        "panoramas_with_results": len(latest),
        "successful_or_existing": downloaded,
        "latest_status_counts": dict(sorted(statuses.items())),
    }
    (outdir / "download_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-db", type=Path, default=HISTORY_DB)
    parser.add_argument("--panorama-db", type=Path, default=PANORAMA_DB)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--sample-ids", type=int, nargs="+", default=DEFAULT_SAMPLE_IDS)
    parser.add_argument("--radius-m", type=float, default=100)
    parser.add_argument("--workers", type=int, default=2,
                        help="Panoramas downloaded concurrently; 1..4")
    parser.add_argument("--tile-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Attempt only the first N panoramas, useful for a smoke test")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if (args.radius_m <= 0 or not 1 <= args.workers <= 4
            or not 1 <= args.tile_workers <= 16):
        parser.error("radius must be positive, workers 1..4, and tile-workers 1..16")
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    if len(set(args.sample_ids)) != len(args.sample_ids):
        parser.error("sample IDs must be unique")

    targets, summary = load_or_prepare(
        args.outdir, args.history_db, args.panorama_db, args.sample_ids, args.radius_m
    )
    for target in targets:
        validate_target(target)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.prepare_only:
        return

    selected = targets[:args.limit] if args.limit else targets
    panorama_dir = args.outdir / "panoramas"
    results_path = args.outdir / "download_results.jsonl"
    panorama_dir.mkdir(parents=True, exist_ok=True)
    results_path.touch(exist_ok=True)
    with results_path.open("a", encoding="utf-8") as handle:
        def record(index, target, result):
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"{index}/{len(selected)} {target['year']}-{target['month'] or 0:02d} "
                f"{target['panoid']} {result['status']}",
                flush=True,
            )

        probe_count = min(3, len(selected))
        probe_results = []
        for index, target in enumerate(selected[:probe_count], 1):
            result = download_one(
                target,
                panorama_dir,
                zoom=5,
                backend="tiles",
                tile_workers=args.tile_workers,
            )
            probe_results.append(result)
            record(index, target, result)
        if probe_results and all(row["status"] == "failed" for row in probe_results):
            raise SystemExit("First three panorama downloads failed; check download_results.jsonl")

        remaining = selected[probe_count:]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    download_one,
                    target,
                    panorama_dir,
                    5,
                    "tiles",
                    args.tile_workers,
                ): target
                for target in remaining
            }
            for index, future in enumerate(as_completed(futures), probe_count + 1):
                target = futures[future]
                record(index, target, future.result())
    update_download_summary(args.outdir, targets)
    print(f"Download results: {results_path}", flush=True)


if __name__ == "__main__":
    main()
