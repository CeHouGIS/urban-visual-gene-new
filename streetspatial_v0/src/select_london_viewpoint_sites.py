"""Select dispersed, cross-time London sites with real camera-position changes."""

import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path


DATA_ROOT = Path("/host/mnt/nas/urban-visual-gene/streetspatial_v0/data")
HISTORY_ROOT = Path("/host/mnt/nas/GSV/metadata/London/history_3periods")
PERIODS = ("2010_2015", "2016_2020", "2021_2025")
HEADINGS = (0, 90, 180, 270)
EARTH_RADIUS_M = 6_371_000.0


def haversine_m(a, b):
    lat1, lon1 = a
    lat2, lon2 = b
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)
    value = (math.sin(dlat / 2) ** 2
             + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(value))


def load_completed_rows(history_root):
    data = {}
    for period in PERIODS:
        path = history_root / f"{period}.csv"
        rows = {}
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                image_paths = {heading: row.get(f"image_{heading}", "")
                               for heading in HEADINGS}
                if row.get("status") != "complete" or not all(image_paths.values()):
                    continue
                sample_id = int(row["sample_id"])
                rows[sample_id] = {
                    "sample_id": sample_id,
                    "period": period,
                    "panoid": row["panoid"],
                    "year": int(row["year"]),
                    "month": int(row["month"] or 0),
                    "anchor_lat": float(row["lat"]),
                    "anchor_lon": float(row["lon"]),
                    "source_views": image_paths,
                }
        data[period] = rows
        print(f"{period}: {len(rows)} complete records", flush=True)
    return data


def load_camera_metadata(database, sample_ids):
    uri = f"file:{database}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    result = {}
    try:
        for start in range(0, len(sample_ids), 3000):
            batch = sample_ids[start:start + 3000]
            placeholders = ",".join("?" for _ in batch)
            query = (
                "SELECT ch.sample_id,ch.period,ch.panoid,ca.lat,ca.lon,"
                "ca.year,ca.month,ca.distance_m "
                "FROM choices ch JOIN candidates ca "
                "ON ca.sample_id=ch.sample_id AND ca.panoid=ch.panoid "
                f"WHERE ch.sample_id IN ({placeholders})"
            )
            for sample_id, period, panoid, lat, lon, year, month, distance in connection.execute(query, batch):
                result.setdefault(sample_id, {})[period] = {
                    "panoid": panoid,
                    "camera_lat": float(lat),
                    "camera_lon": float(lon),
                    "year": int(year),
                    "month": int(month or 0),
                    "distance_to_anchor_m": float(distance),
                }
            print(f"Read camera positions for {min(start + 3000, len(sample_ids))} / {len(sample_ids)} sites",
                  flush=True)
    finally:
        connection.close()
    return result


def collect_candidates(data, metadata, min_baseline_m, min_near_far_delta_m):
    common = set.intersection(*(set(data[period]) for period in PERIODS))
    candidates = []
    for sample_id in common:
        if sample_id not in metadata or any(period not in metadata[sample_id] for period in PERIODS):
            continue
        period_rows = []
        for period in PERIODS:
            source = data[period][sample_id]
            camera = metadata[sample_id][period]
            if source["panoid"] != camera["panoid"]:
                break
            period_rows.append({**source, **camera})
        if len(period_rows) != len(PERIODS):
            continue
        positions = [(row["camera_lat"], row["camera_lon"]) for row in period_rows]
        baselines = [haversine_m(positions[i], positions[j])
                     for i in range(len(positions)) for j in range(i + 1, len(positions))]
        nearest = min(period_rows, key=lambda row: row["distance_to_anchor_m"])
        farthest = max(period_rows, key=lambda row: row["distance_to_anchor_m"])
        baseline = max(baselines)
        near_far_delta = farthest["distance_to_anchor_m"] - nearest["distance_to_anchor_m"]
        if baseline < min_baseline_m or near_far_delta < min_near_far_delta_m:
            continue
        for row in period_rows:
            row["viewpoint_role"] = (
                "near" if row["panoid"] == nearest["panoid"] else
                "far" if row["panoid"] == farthest["panoid"] else "intermediate"
            )
        anchor = data[PERIODS[0]][sample_id]
        candidates.append({
            "sample_id": sample_id,
            "anchor_lat": anchor["anchor_lat"],
            "anchor_lon": anchor["anchor_lon"],
            "periods": period_rows,
            "max_camera_baseline_m": baseline,
            "near_far_delta_m": near_far_delta,
            "nearest_distance_m": nearest["distance_to_anchor_m"],
            "farthest_distance_m": farthest["distance_to_anchor_m"],
        })
    return common, candidates


def select_spatially_dispersed(candidates, count, min_spacing_m):
    ordered = sorted(candidates,
                     key=lambda row: (row["max_camera_baseline_m"], row["near_far_delta_m"]),
                     reverse=True)
    selected = []
    spacing = min_spacing_m
    while spacing >= 0:
        selected = []
        for candidate in ordered:
            point = (candidate["anchor_lat"], candidate["anchor_lon"])
            if all(haversine_m(point, (row["anchor_lat"], row["anchor_lon"])) >= spacing
                   for row in selected):
                selected.append(candidate)
                if len(selected) == count:
                    return selected, spacing
        if spacing == 0:
            break
        spacing = 100 if spacing > 100 else 0
    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} qualified sites for requested count {count}")
    return selected[:count], spacing


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-root", type=Path, default=HISTORY_ROOT)
    parser.add_argument("--database", type=Path, default=HISTORY_ROOT / "history_3periods.db")
    parser.add_argument("--outdir", type=Path, default=DATA_ROOT / "london_viewpoints_100")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--min-baseline-m", type=float, default=5.0)
    parser.add_argument("--min-near-far-delta-m", type=float, default=20.0)
    parser.add_argument("--min-site-spacing-m", type=float, default=250.0)
    args = parser.parse_args()
    if args.count <= 0 or args.min_baseline_m < 0 or args.min_near_far_delta_m < 0:
        parser.error("count must be positive and distance thresholds nonnegative")
    if args.outdir.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset directory: {args.outdir}")

    data = load_completed_rows(args.history_root)
    common = set.intersection(*(set(data[period]) for period in PERIODS))
    metadata = load_camera_metadata(args.database, sorted(common))
    common, candidates = collect_candidates(
        data, metadata, args.min_baseline_m, args.min_near_far_delta_m)
    print(f"Three-period complete sites: {len(common)}; near/far and baseline qualified: {len(candidates)}",
          flush=True)
    sites, actual_spacing = select_spatially_dispersed(
        candidates, args.count, args.min_site_spacing_m)
    incomplete = [row["sample_id"] for site in sites for row in site["periods"]
                 if not all(Path(path).is_file() for path in row["source_views"].values())]
    if incomplete:
        missing_sites = set(incomplete)
        candidates = [row for row in candidates if row["sample_id"] not in missing_sites]
        print(f"Excluded {len(missing_sites)} sites with missing perspective files", flush=True)
        sites, actual_spacing = select_spatially_dispersed(
            candidates, args.count, args.min_site_spacing_m)

    output = args.outdir
    panorama_dir = output / "panoramas"
    panorama_dir.mkdir(parents=True)
    site_rows, targets = [], []
    for index, candidate in enumerate(sites, 1):
        site_id = f"LON_{index:03d}"
        periods = []
        for row in candidate["periods"]:
            period_row = {
                "period": row["period"],
                "panoid": row["panoid"],
                "year": row["year"],
                "month": row["month"],
                "camera_lat": row["camera_lat"],
                "camera_lon": row["camera_lon"],
                "distance_to_anchor_m": row["distance_to_anchor_m"],
                "viewpoint_role": row["viewpoint_role"],
                "source_views": row["source_views"],
            }
            periods.append(period_row)
            targets.append({
                "city": "London",
                "source_city": "London",
                "country": "United Kingdom",
                "panoid": row["panoid"],
                "lat": row["camera_lat"],
                "lon": row["camera_lon"],
                "year": row["year"],
                "month": row["month"],
                "heading": 0,
                "sample_id": candidate["sample_id"],
                "site_id": site_id,
                "period": row["period"],
                "viewpoint_role": row["viewpoint_role"],
                "camera_distance_m": row["distance_to_anchor_m"],
                "heading_offsets": list(HEADINGS),
            })
        site_rows.append({
            "site_id": site_id,
            "sample_id": candidate["sample_id"],
            "anchor_lat": candidate["anchor_lat"],
            "anchor_lon": candidate["anchor_lon"],
            "search_radius_m": 50,
            "periods": periods,
            "max_camera_baseline_m": candidate["max_camera_baseline_m"],
            "near_far_delta_m": candidate["near_far_delta_m"],
            "nearest_camera_distance_m": candidate["nearest_distance_m"],
            "farthest_camera_distance_m": candidate["farthest_distance_m"],
        })

    write_jsonl(output / "sites.jsonl", site_rows)
    write_jsonl(panorama_dir / "targets.jsonl", targets)
    baselines = sorted(row["max_camera_baseline_m"] for row in sites)
    deltas = sorted(row["near_far_delta_m"] for row in sites)
    summary = {
        "city": "London",
        "anchor_count": len(site_rows),
        "panorama_count": len(targets),
        "period_counts": {period: sum(x["period"] == period for x in targets) for period in PERIODS},
        "headings_per_panorama": list(HEADINGS),
        "source_perspective_images_per_panorama": 4,
        "common_three_period_sites": len(common),
        "qualified_sites": len(candidates),
        "selection_thresholds": {
            "minimum_camera_baseline_m": args.min_baseline_m,
            "minimum_near_far_distance_delta_m": args.min_near_far_delta_m,
            "requested_site_spacing_m": args.min_site_spacing_m,
            "actual_site_spacing_m": actual_spacing,
        },
        "max_camera_baseline_m": {
            "min": baselines[0], "median": baselines[len(baselines) // 2], "max": baselines[-1],
        },
        "near_far_delta_m": {
            "min": deltas[0], "median": deltas[len(deltas) // 2], "max": deltas[-1],
        },
        "source_database": str(args.database),
        "source_image_root": "/host/mnt/nas/GSV/images/London",
        "download_root": str(panorama_dir),
    }
    with (output / "sampling_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
