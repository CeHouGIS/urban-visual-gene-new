"""Draw the actual global-panorama-to-V0.1 crop pipeline for one example."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from py360convert.utils import uv2coor, xyz2uv, xyzpers


DATA = Path("/host/mnt/nas/urban-visual-gene/streetspatial_v0/data")
SPLIT = DATA / "splits/global200_v01/transformations.jsonl"
CANDIDATES = DATA / "global200_sources/candidates.jsonl"
OUTPUT = Path(__file__).resolve().parents[1] / "outputs/global200_v0_1/figures/fig_panorama_crop_pipeline.png"
EXAMPLE_ID = "Chile_Temuco_ee2d9a97_BqFtEl-l6useWW1TzIi9Rg_0"
COLORS = {"original": "#167c80", "shift_left": "#cc503e", "shift_right": "#dc9828",
          "shift_up": "#577ab5", "shift_down": "#31916c", "zoom_in": "#985ca1",
          "zoom_out": "#394b59", "mirror": "#d16c35"}
FRAME_TRANSFORMS = ("original", "shift_left", "shift_right", "shift_up",
                    "shift_down", "zoom_in", "zoom_out")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def panorama_xy(points, heading, hfov, vfov, source_size, panorama_size):
    """Map perspective pixel coordinates back onto the equirectangular panorama."""
    source_width, source_height = source_size
    pano_width, pano_height = panorama_size
    x = np.asarray(points[:, 0], dtype=np.float32)
    y = np.asarray(points[:, 1], dtype=np.float32)
    xx = (x / (source_width - 1) * 2 - 1) * np.tan(np.deg2rad(hfov) / 2)
    yy = (1 - y / (source_height - 1) * 2) * np.tan(np.deg2rad(vfov) / 2)
    rays = np.column_stack((xx, yy, np.ones_like(xx)))
    angle_u = -np.deg2rad(heading)
    cos_u, sin_u = np.cos(angle_u), np.sin(angle_u)
    world = rays @ np.array([[cos_u, 0, -sin_u], [0, 1, 0], [sin_u, 0, cos_u]], dtype=np.float32)
    u, v = xyz2uv(world)
    px, py = uv2coor(u, v, pano_height, pano_width)
    return np.column_stack((px, py))


def crop_outline(box, heading, hfov, vfov, source_size, panorama_size, samples=100):
    left, top, right, bottom = box
    top_edge = np.column_stack((np.linspace(left, right, samples), np.full(samples, top)))
    right_edge = np.column_stack((np.full(samples, right), np.linspace(top, bottom, samples)))
    bottom_edge = np.column_stack((np.linspace(right, left, samples), np.full(samples, bottom)))
    left_edge = np.column_stack((np.full(samples, left), np.linspace(bottom, top, samples)))
    points = np.concatenate((top_edge, right_edge, bottom_edge, left_edge))
    mapped = panorama_xy(points, heading, hfov, vfov, source_size, panorama_size)
    return mapped


def main():
    views = {row["transformation"]: row for row in read_jsonl(SPLIT)
             if row["image_id"] == EXAMPLE_ID}
    candidates = {row["image_id"]: row for row in read_jsonl(CANDIDATES)}
    candidate = candidates[EXAMPLE_ID]
    panorama_path = Path(candidate["panorama_path"])
    heading, hfov = candidate["heading"], candidate["horizontal_fov"]
    vfov = candidate["vertical_fov"]
    source_size = tuple(views["original"]["source_size"])

    fig, pano_ax = plt.subplots(figsize=(16, 10), facecolor="white")
    display_width, display_height = 3000, 1500
    with Image.open(panorama_path) as panorama:
        pano_width, pano_height = panorama.size
        panorama_display = panorama.convert("RGB").resize(
            (display_width, display_height), Image.Resampling.LANCZOS)
        pano_ax.imshow(panorama_display)
    pano_ax.set_xlim(0, display_width)
    pano_ax.set_ylim(display_height, 0)
    pano_ax.set_title("A. Full panorama with perspective crop windows mapped onto it",
                      loc="left", fontsize=14)
    pano_ax.set_xticks([])
    pano_ax.set_yticks([])
    for name in FRAME_TRANSFORMS:
        box = views[name]["crop_xyxy"]
        outline = crop_outline(box, heading, hfov, vfov, source_size, (pano_width, pano_height))
        outline *= np.array([display_width / pano_width, display_height / pano_height])
        pano_ax.plot(outline[:, 0], outline[:, 1], color=COLORS[name], linewidth=2.6,
                     label=f"{name}: {box[2]-box[0]} x {box[3]-box[1]} px")
    footprint = crop_outline([0, 0, source_size[0] - 1, source_size[1] - 1],
                             heading, hfov, vfov, source_size, (pano_width, pano_height))
    footprint *= np.array([display_width / pano_width, display_height / pano_height])
    pano_ax.plot(footprint[:, 0], footprint[:, 1], color="#f0d34a", linewidth=3.0,
                 label=f"selected perspective view: heading {heading}°, FOV {hfov:.0f}°")
    pano_ax.legend(loc="lower left", fontsize=10, framealpha=0.94, ncol=2)
    fig.text(0.01, 0.015,
             "Colored boundaries are the actual V0.1 windows mapped back through the perspective projection. "
             "They curve on the panorama because the crop is defined in perspective coordinates.",
             fontsize=10, color="#454545")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
