# UrbanGene V2

2026-11-10 已启动 **20,000 张真实训练照片 / 512 张验证照片** 的独立扩大实验。
沿用原分组模型与重建目标，从头训练 30 轮，最多两张空闲 GPU；按用户最新要求，
无空闲卡时排队，取消 CPU 模型计算。
入口 `tools/run_real20k.py`，配置 `configs/m1_real20k_20261110.yaml`，
方案 `docs/REAL20K_PROTOCOL_20261110.md`，状态在新 run 的 `execution.json`。

**当前主线使用真实街景照片。** 2026-11-09 启动了独立的 512 张训练 / 64 张验证
真实 Google 街景实验，方案见 `docs/REAL_STREET_PROTOCOL_20261109.md`。
入口为 `tools/prepare_real_streets.py` 与 `tools/run_real_street_screen.py`。
此前两轮可视化与实例指标来自合成图，仅保留为诊断；不代表真实街景效果。

This NAS-backed workspace implements M0 and M1 from the V2 PRD. The current
scientific pipeline is frozen DINOv3 features -> Adaptive Spatial Slot
Discovery -> feature reconstruction. M2-M4 are gated on held-out instance
discovery evidence.

## DINOv3 extraction

The checkpoint is available on NAS at
`/host/mnt/nas/urban-visual-gene/urbangene_v2_20261109/models/dinov3-vitl16`.
The public ModelScope download is pinned to commit
`99b6205b0321c4d9726dd072d8c6a80aa6be72df`, verifies every SHA256 against
that snapshot, and checks full local CPU loading. It does not read `.env` or
send Hugging Face credentials. To reproduce or resume it with the prepared
Python environment:

```bash
cd /root/code/urban-visual-gene/urbangene_v2
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
  /root/code/urban-visual-gene/experiments/dinov3_generation_revision/.venv/bin/python3 \
  tools/download_dinov3_modelscope.py
```

File identities and loading results are in `outputs/v2/dinov3_download.json`
under the experiment root. The original Hugging Face downloader remains
available for an independent source comparison when that route is reachable.

The verified 1024-D NAS arrays are cached features, not weights. They are
usable for the CPU engineering smoke. For reproducible synthetic validation,
the extraction command requires a local DINOv3 Transformers snapshot with
`config.json` and its complete weight shards. The loader is local-only, checks
the model family, refuses missing/mismatched weights, freezes every parameter,
and records hashes plus preprocessing in a sidecar.

```bash
cd /root/code/urban-visual-gene/urbangene_v2
python -m pip install -e '.[backbone,test]'
python run.py prepare-synthetic --config configs/synthetic_data.yaml \
  --output /host/mnt/nas/urban-visual-gene/urbangene_v2_20261109/outputs/v2/data/synthetic_20261109
python run.py extract-features --config configs/m1_dinov3.yaml \
  --manifest /host/mnt/nas/urban-visual-gene/urbangene_v2_20261109/outputs/v2/data/synthetic_20261109/images.train.jsonl \
  --output /host/mnt/nas/urban-visual-gene/urbangene_v2_20261109/outputs/v2/data/features.train.npy
```

The val/test image manifests follow the same extraction command with distinct
output files. Instance annotations are separate `.npz` files. Extraction and
training load only the image manifest / feature cache.

## Tests and CPU smoke

```bash
python -m pytest -q tests
python run.py train-discovery --config configs/m1_cpu_smoke.yaml
```

The smoke uses existing 1024-D DINOv3 features after their existing 16x16
average pooling. It checks engineering execution only; it does not evaluate
instance masks or satisfy the M1 scientific gate.

## M1 training and evaluation

After verified synthetic extraction, set `features.train` and
`features.validation` in a fresh config under this directory, set its `output`
to a new child of `outputs/v2`, and train with `train-discovery`. Evaluate a
selected checkpoint against `annotations.val_seen.npz` using `eval-discovery`.
The validation selection criterion is feature reconstruction. Test manifests
remain locked until `docs/EXPERIMENT_PROTOCOL.md` and a config hash are saved.

`configs/m1_dinov3_gpu2.yaml` is the separate two-GPU training configuration.
The trainer rejects selections larger than two GPUs and writes the chosen
device IDs and names to `run.json`. Keep the CPU smoke config separate; do not
launch the GPU config until synthetic feature extraction is complete and two
GPUs are available.

## Recorded M1 validation screens (2026-11-09)

The complete feature caches are `outputs/v2/data/features.train.npy`
(`[256,1024,28,28]`) and `features.val_seen.npy` (`[64,1024,28,28]`). Their
sidecars record model hashes, preprocessing, image IDs, and feature hashes.
The experiment protocol and config hash were saved before instance evaluation
in `outputs/v2/M1_VALIDATION_PROTOCOL_20261109.json`.

`configs/m1_dinov3_cpu_screen_20261109.yaml` is the initial 10-epoch CPU
validation screen. Its run is `outputs/v2/m1_dinov3_cpu_screen_20261109`; the
run has 640 updates and validation metrics, sample tensors, and static figures
under `validation/`. It did not pass the instance-discovery gate. The separate
`validation_ten_tree/` group contains 12 additional validation images and is
not a formal held-out test split. No evaluation annotations enter training.

The bounded follow-up changes only coordinate-based initial slot geometry
(`spatial_initial_scale: 0.18`) to diagnose the observed collapse. It retains
adaptive geometry and learned assignments. Its config is
`configs/m1_dinov3_cpu_spatial_seed_20261109.yaml`; its rationale and protocol
are `outputs/v2/M1_SPATIAL_SEED_ABLATION_PROTOCOL_20261109.json`. A spatial
initialization does not establish that the resulting regions are instances.

The runner `tools/run_m1_validation_screen.py` chains training, evaluation,
ten-tree diagnosis, and figures. It forces CPU, validates feature checksums,
archives source without credentials, and refuses existing run directories.
Reproductions require a fresh config output and an unused `--tag`. Useful
diagnostic tools are `report_m1_validation.py`, `diagnose_m1_transforms.py`,
and `diagnose_m1_tree_separation.py`. The ten-tree images and masks can be
reproduced or checked with `prepare_ten_tree_diagnostic.py`.

All final test splits remain locked. `docs/STATUS.md`, `docs/RESULTS.md`,
and the run-specific `progress.json` distinguish completed work from ongoing
work and scientific gates from engineering tests.
