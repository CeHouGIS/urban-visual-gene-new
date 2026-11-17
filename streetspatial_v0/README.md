# StreetSpatial V0

Frozen Qwen2.5-VL spatial representation study. See `outputs/v0/report_zh.md`
for the Chinese report or `outputs/v0/report.md` for the English report.

## V0.1 position controls

The completed V0.1 analysis is in `outputs/v0_1/report.md`. It tests whether
coordinate decoding exceeds what is available from the known object box and
selected visual Token positions. Run `bash scripts/run_v0_1.sh` to reproduce
the audit, E1-E4 probes, two-step image transforms, and figures using the
existing NAS images, V0 feature cache, and local model snapshot. The script
does not modify V0 outputs. Most pilot annotations remain detector proposals;
see the V0.1 report before interpreting category or latent-coordinate results.

## Full-panorama follow-up

`src/download_panoramas.py` supports complete panorama downloads. Its `tiles`
backend adapts the 26 x 13, zoom-5 tile download and stitching code
from the historical `GSVdownload_maoran.py` in `auto-svi-download` commit
`e9ea8d8e88`. That historical file's `--mode panorama` actually calls the
four-heading thumbnail path; its tile functions were not wired into the CLI.
The current tile API succeeded on this host after requests included the Google
Maps referer and browser user agent. The sampler and projector for the global
V0.1 run are `src/sample_global_panoramas.py` and
`src/project_global_panoramas.py`.

The global follow-up sampled IDs by cycling through countries, then choosing
random cities and panorama IDs within each country. It saved 200 complete
panoramas covering 97 countries and 167 cities. Each was projected to four
800 x 600 perspective views before unreviewed Grounding DINO proposals selected
an eligible direction. 177 scenes produced 1,199 transformed views. The frozen
Qwen2.5-VL features and E1-E4 results are stored in separate global200 paths;
the original V0 and V0.1 outputs were not overwritten. See
`outputs/global200_v0_1/report_zh.md` for the Chinese report and measured
results. All global object boxes remain unreviewed machine proposals.

The tile backend validates complete 2:1 images and can resume without
redownloading completed panoramas. The alternate `streetlevel` backend remains
available for other runs.

```bash
cd /root/code/urban-visual-gene/streetspatial_v0
python -m src.download_panoramas --backend tiles --limit 1
python -m src.download_panoramas --backend tiles
# Alternate implementation, if the legacy tile endpoint is unavailable:
PYTHONPATH=/host/mnt/nas/urban-visual-gene/streetspatial_v0/vendor python -m src.download_panoramas --backend streetlevel --zoom 4 --workers 4
```

The original source manifest and pilot remain available below for reproducing
the earlier experiment. The global panorama data and new analysis are separate
artifacts on the NAS at `/host/mnt/nas/urban-visual-gene/streetspatial_v0/`.

## Current pilot

The completed 200-scene pilot uses
`/host/mnt/nas/urban-visual-gene/streetspatial_v0/data/annotations/pilot200_annotations.jsonl`.
Ten scenes were visually reviewed; the remaining 190 contain detector
proposals. Eight numbered contact sheets in
`/host/mnt/nas/urban-visual-gene/streetspatial_v0/data/annotations/pilot200_review_pages/`
show every scene with its current boxes. Inspect the source image at full
resolution before accepting or correcting a box. Do not change
`annotation_status` to `visually_reviewed` until that scene is checked.

The analysis uses the finalized manifest in
`/host/mnt/nas/urban-visual-gene/streetspatial_v0/data/splits/pilot200_final/`.
Corrections to the source annotations require building a new split and
re-extracting features; the existing pilot output is a fixed, exploratory
snapshot. See `outputs/v0/report.md` for measured results and limitations.

## Data preparation for a new run

The source manifest in `configs/v0.yaml` points to existing real Google Street
View files. The script only reads those files. It samples one camera heading per
panorama so later train/validation/test splits cannot share a panorama. Generated
templates and previews go to the NAS `data_root` in the same config because the
local filesystem is full.
The templates below have already been generated. `sample` refuses to overwrite
an existing annotation file; use new output paths for a new run.

```bash
cd /root/code/urban-visual-gene/streetspatial_v0
python -m src.build_dataset sample --count 10 --annotations /host/mnt/nas/urban-visual-gene/streetspatial_v0/data/annotations/smoke10.jsonl --contact-sheets /host/mnt/nas/urban-visual-gene/streetspatial_v0/data/annotations/smoke10_sheets
python -m src.build_dataset sample --count 200
```

Fill each image record's `objects` array with 2-3 verified objects, e.g.:

```json
{"object_id":"car_1","object_category":"Car","bbox":[95,125,180,205]}
```

Coordinates are source-image pixels. Keep `image_id`, `scene_id`, `city_id`,
`image_path`, and `image_size` as generated. Check every box in the source image
and use distinct object IDs within each image. Prefer Car, Tree, Traffic Sign,
and visually confirmed Lamp Post. Use `Pole` when the function of an unreviewed
vertical pole is uncertain. Exclude ambiguous or partly visible instances.

```bash
python -m src.build_dataset review --annotations /host/mnt/nas/urban-visual-gene/streetspatial_v0/data/annotations/smoke10_reviewed.jsonl --output /host/mnt/nas/urban-visual-gene/streetspatial_v0/data/annotations/smoke10_review
python -m src.build_dataset build --annotations /host/mnt/nas/urban-visual-gene/streetspatial_v0/data/annotations/smoke10_reviewed.jsonl --output /host/mnt/nas/urban-visual-gene/streetspatial_v0/data/splits/smoke10_new
```

`build` requires at least two labeled objects per image. It writes transformed
JPEGs, flattened `annotations.jsonl`, `transformations.jsonl`, and scene-level
`train.jsonl`, `val.jsonl`, `test.jsonl`. Each view records the exact integer
crop window, output size, mirrored state, updated boxes, centers, and pair
relations. Views clipping any labeled object are skipped. No padding or image
generation is used.

Run geometry tests with `python -m unittest discover -s tests -v`.

## Model experiment

The ModelScope snapshots and generated data are under
`/host/mnt/nas/urban-visual-gene/streetspatial_v0/`. Each model file is checked
against the source SHA256 and listed in `models/*/snapshot_manifest.json`.
The tested Python environment is
`/root/code/urban-visual-gene/experiments/dinov3_generation_revision/.venv/bin/python`;
package versions are in `requirements.txt`. Use `bash scripts/run_v0.sh` to
repeat the pilot quality checks, frozen feature extraction, and probe analysis.
Feature extraction skips completed view caches, allowing resume after a GPU or
I/O interruption.

The 200-scene pilot has 10 visually reviewed scenes and 190 detector-proposed
scenes. `data/splits/pilot200_final` contains the analysis manifests and a
label-balanced scene split chosen before model feature analysis. Unreviewed
vertical poles are labeled `Pole`, since the detector cannot reliably tell a
streetlight from an electrical pole. Treat pilot semantic results as exploratory
until those 190 scenes receive individual review.
