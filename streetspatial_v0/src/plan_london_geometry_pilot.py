"""Audit the available London data and illustrate the proposed geometry pilot."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


DEFAULT_DATA = Path(
    "/host/mnt/nas/urban-visual-gene/streetspatial_v0/data/london_100m_panoramas_3sites"
)
COHORTS = {
    207192: ("2025-08", "2024-08", "2012-06"),
    262745: ("2025-06", "2024-07", "2012-09"),
    342355: ("2025-10", "2024-07", "2012-08"),
}
TIME_COHORTS = {
    207192: ("2019-04", "2025-08"),
    262745: ("2016-08", "2024-07"),
    342355: ("2018-04", "2024-07"),
}


def distance(a, b):
    p, q = map(math.radians, (a["lat"], b["lat"]))
    dl = math.radians(b["lon"] - a["lon"])
    v = math.sin((q - p) / 2) ** 2 + math.cos(p) * math.cos(q) * math.sin(dl / 2) ** 2
    return 12_742_000 * math.asin(min(1.0, math.sqrt(v)))


def date_key(row):
    return f"{row['year']}-{row['month']:02d}"


def match_candidates(first, second, threshold=2.0):
    """Deterministic greedy disjoint matches; GPS-only screening, not pose truth."""
    edges = sorted((distance(a, b), a["panoid"], b["panoid"])
                   for a in first for b in second)
    used_first, used_second, result = set(), set(), []
    for dist, a, b in edges:
        if dist <= threshold and a not in used_first and b not in used_second:
            result.append({"first_panoid": a, "second_panoid": b,
                           "metadata_distance_m": round(dist, 3)})
            used_first.add(a)
            used_second.add(b)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "docs")
    args = parser.parse_args()
    font = Path("/root/.cache/streetspatial_fonts/NotoSansCJKsc-Regular.otf")
    use_chinese = font.exists()
    if use_chinese:
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
        plt.rcParams["axes.unicode_minus"] = False
    summary = json.loads((args.data / "sampling_summary.json").read_text())
    targets = [json.loads(line) for line in (args.data / "targets.jsonl").read_text().splitlines()]
    audit = {
        "status": "design_and_metadata_audit_only",
        "source_manifest": str(args.data / "targets.jsonl"),
        "panorama_count": len(targets),
        "metadata_fields": sorted(targets[0]),
        "missing_for_geometry": ["camera_orientation", "camera_elevation", "landmark_tracks", "3d_points"],
        "time_matching_warning": "GPS-only candidates require pose and unchanged-landmark verification",
        "sites": [],
    }
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.3), constrained_layout=True)
    angles = np.linspace(0, 2 * math.pi, 361)
    main_ids, geometry_ids, time_ids = set(), set(), set()
    for axis, site in zip(axes, summary["sites"]):
        sid = site["sample_id"]
        rows = [r for r in targets if any(m["sample_id"] == sid for m in r["memberships"])]
        dates = defaultdict(list)
        for row in rows:
            dates[date_key(row)].append(row)
        main_date = COHORTS[sid][0]
        main = dates[main_date]
        east = np.array([math.radians(r["lon"] - site["lon"]) * 6_371_000
                         * math.cos(math.radians(site["lat"])) for r in rows])
        north = np.array([math.radians(r["lat"] - site["lat"]) * 6_371_000 for r in rows])
        axis.plot(100 * np.cos(angles), 100 * np.sin(angles), color="#919ead", ls="--", lw=1)
        axis.scatter(east, north, s=13, c="#9fb5c8", alpha=0.45,
                     label="全部采集日期" if use_chinese else "All available dates")
        chosen = np.array([date_key(r) == main_date for r in rows])
        axis.scatter(east[chosen], north[chosen], s=40, c="#d44c36", edgecolors="white", lw=0.5,
                     label=f"主采集月：{main_date}" if use_chinese else f"Main cohort: {main_date}", zorder=3)
        axis.scatter([0], [0], marker="*", s=150, c="#152a43",
                     label="锚点" if use_chinese else "Anchor", zorder=4)
        title = (f"地点 {sid}：{len(rows)} 张全景\n主采集月含 {len(main)} 个相机位置" if use_chinese
                 else f"Site {sid}: {len(rows)} panoramas\n{len(main)} main camera positions")
        axis.set(xlim=(-112, 112), ylim=(-112, 112),
                 xlabel="锚点以东（米）" if use_chinese else "East of anchor (m)",
                 ylabel="锚点以北（米）" if use_chinese else "North of anchor (m)",
                 aspect="equal", title=title)
        axis.grid(alpha=0.15)
        axis.legend(loc="lower left", fontsize=7.5)
        time_first, time_second = TIME_COHORTS[sid]
        matches = match_candidates(dates[time_first], dates[time_second])
        geometry = [r for date in COHORTS[sid] for r in dates[date]]
        main_ids.update(r["panoid"] for r in main)
        geometry_ids.update(r["panoid"] for r in geometry)
        time_ids.update(r["panoid"] for date in TIME_COHORTS[sid] for r in dates[date])
        audit["sites"].append({
            **site,
            "capture_month_count": len(dates),
            "main_date": main_date,
            "main_count": len(main),
            "main_max_camera_baseline_m": round(max(distance(a, b) for a in main for b in main), 3),
            "geometry_cohort_counts": {date: len(dates[date]) for date in COHORTS[sid]},
            "time_cohorts": [time_first, time_second],
            "metadata_close_time_pairs": matches,
        })
    audit["main_panorama_count"] = len(main_ids)
    audit["geometry_cohort_panorama_count"] = len(geometry_ids)
    audit["geometry_and_time_union_panorama_count"] = len(geometry_ids | time_ids)
    audit["proposed_yaw_count"] = 8
    audit["main_perspective_view_count"] = len(main_ids) * 8
    figure.suptitle("伦敦三维表征先导实验：三个 100 米范围内的真实相机位置" if use_chinese
                   else "London geometry pilot: real camera positions inside three 100 m circles", fontsize=14)
    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "images"
    image_dir.mkdir(exist_ok=True)
    figure.savefig(image_dir / "london_100m_experiment_sampling.png", dpi=180)
    plt.close(figure)
    (args.output / "london_100m_design_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in audit.items() if key != "sites"}, indent=2))


if __name__ == "__main__":
    main()
