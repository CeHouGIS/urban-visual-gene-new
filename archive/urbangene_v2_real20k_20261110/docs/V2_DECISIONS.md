# UrbanGene V2 Decisions

Updated 2026-11-09. This is a new NAS-backed experiment line. V1, V1.1,
their queues, and their checkpoints are read-only inputs to the audit.

## Scope

Implement only M0 and M1 from the V2 PRD. M1 uses frozen DINOv3 spatial
features, competitive Adaptive Spatial Slot Discovery, and feature
reconstruction. No SAE, shared Gene dictionary, GRCT, or image renderer is
added before the M1 scientific gate.

The current checkout already has a deterministic synthetic renderer and a
1024-channel DINOv3 ViT-L/16 cache. The cache resides at
`/host/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_batchtopk_307city_balanced_3000/dict_<city>_p784.f16.npy`.
For Austin its shape is `(9408000, 1024)`, dtype `float16`; the 3200-item V1
cache is `(3200, 256, 1024)`, also `float16`. Both were opened and sampled.
The feature arrays are not model weights. Their manifests identify DINOv3,
but do not record a weight hash or a fully reproducible extraction pipeline.

The V2 source adds a versioned local-weight loader and feature extraction
pipeline. It refuses random initialization and records the exact weight/config
hashes. The original backbone checkpoint search is documented in
`outputs/v2/model_search.json`; an unrelated MoCo-v3 or MAE checkpoint is not
treated as DINOv3.

## Data and label boundary

Synthetic scene images reuse the V0 renderer and its split-generation rules.
Image manifests contain only sample ID, image path, and split. Evaluation
instance masks are stored separately and are not loaded by `FeatureDataset`,
feature extraction, or `train_discovery`. The M1 reconstruction loss uses only
frozen DINO features. A feature-level horizontal flip is a stability diagnostic;
it is not reported as a fresh DINO forward pass on a flipped RGB image.

The existing 1024-D cache supports an engineering CPU smoke test without
pretending to provide synthetic instance labels. Scientific synthetic
evaluation requires the reproducible DINOv3 pipeline to extract features for
the labeled synthetic validation images.

## Isolation

All code is stored under the NAS directory `urbangene_v2_20261109/source` and
all new outputs are restricted to `urbangene_v2_20261109/outputs/v2`. V2 code
rejects output paths outside that root. No V1/V1.1/V1.2 output directory is a
valid write target. The source is reachable from the current checkout through
the `urbangene_v2` symlink.

## Known audit findings

- V1.1 has a formal 14-run queue reported active at PID 399150. Prior status
  reports 3/14 completed at 00:50 UTC on 2026-11-09; inspect the NAS queue
  status for the current state before any GPU work.
- V1.1's prior report identifies a frozen AE at 30.674 dB, but M1 does not use
  that AE.
- No V1.2 fixed-grid VQ/RVQ implementation or run artifact was found in the
  source tree or V1.1 output inventory. It remains `not found`, not a completed
  baseline.
- Current GPU queues are left untouched. CPU tests/smoke run first; any GPU
  screen waits until the V1.1 queue and device availability have been checked.
