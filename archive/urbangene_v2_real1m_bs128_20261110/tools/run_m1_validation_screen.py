#!/usr/bin/env python3
"""Run the bounded CPU M1 validation screen without unlocking final tests."""

import argparse
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent / "outputs/v2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    from urbangene_v2.common import load_config, write_json, sha256
    from urbangene_v2.train import train_discovery, evaluate_discovery
    from urbangene_v2.extract import extract_manifest
    import torch

    config = load_config(args.config)
    if config["device"] != "cpu" or config["training"].get("gpu_ids"):
        raise ValueError("This bounded CPU runner must not use any GPUs")
    for key in ("train", "validation"):
        feature = Path(config["features"][key])
        sidecar = feature.with_suffix(feature.suffix + ".json")
        meta = json.loads(sidecar.read_text())
        if sha256(feature) != meta["feature_sha256"]:
            raise ValueError(f"Feature checksum mismatch: {feature}")
    if args.tag and not args.tag.replace("_", "").isalnum():
        raise ValueError("tag must contain only letters, numbers, and underscores")
    suffix = "_" + args.tag if args.tag else ""
    state_path = OUTPUT / f"m1_screen_execution_20261109{suffix}.json"
    if state_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior execution: {state_path}")
    archive = OUTPUT / f"source_snapshot_m1_cpu_screen_20261109{suffix}.tar.gz"
    with tarfile.open(archive, "x:gz") as stream:
        for directory in ("src", "configs", "tests", "tools", "docs"):
            for path in sorted((ROOT / directory).rglob("*")):
                if path.is_file() and path.suffix in {".py", ".yaml", ".md"}:
                    stream.add(path, arcname=str(path.relative_to(ROOT)))
    state = {"pid": os.getpid(), "config": str(Path(args.config).resolve()),
             "config_sha256": sha256(args.config), "source_snapshot": str(archive),
             "source_snapshot_sha256": sha256(archive),
             "started_utc": datetime.now(timezone.utc).isoformat(),
             "device": "cpu", "gpu_ids": [], "test_splits_opened": False}

    def phase(name, **extra):
        state.update(phase=name, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        write_json(state_path, state)
        print(json.dumps(state), flush=True)

    try:
        phase("training")
        run = train_discovery(config)
        checkpoint = run / "checkpoint_best_recon.pt"
        phase("validation_evaluation", run=str(run), checkpoint=str(checkpoint))
        summary = evaluate_discovery(checkpoint, config["features"]["validation"],
            OUTPUT / "data/synthetic_20261109/annotations.val_seen.npz",
            device="cpu", output_dir=run / "validation")
        phase("ten_tree_feature_extraction", validation_gate=summary["scientific_gate"])
        diagnostic = OUTPUT / "data/diagnostic_ten_tree_20261109"
        torch.set_num_threads(8)
        backbone = {"family": "dinov3", "backend": "transformers", "resolution": 448,
                    "weights": str(ROOT.parent / "models/dinov3-vitl16")}
        if (diagnostic / "features.npy").exists():
            cached = json.loads((diagnostic / "features.npy.json").read_text())
            if (cached["feature_sha256"] != sha256(diagnostic / "features.npy")
                    or cached["manifest_sha256"] != sha256(diagnostic / "images.jsonl")):
                raise ValueError("Existing diagnostic cache checksum/manifest mismatch")
        else:
            extract_manifest(diagnostic / "images.jsonl", diagnostic / "features.npy", backbone,
                             batch_size=2, workers=0, device="cpu")
        phase("ten_tree_evaluation")
        ten_tree = evaluate_discovery(checkpoint, diagnostic / "features.npy",
            diagnostic / "annotations.npz", device="cpu", output_dir=run / "validation_ten_tree")
        phase("rendering_diagnostics")
        for evaluation, manifest in [
                (run / "validation", OUTPUT / "data/synthetic_20261109/images.val_seen.jsonl"),
                (run / "validation_ten_tree", diagnostic / "images.jsonl")]:
            subprocess.run([sys.executable, str(ROOT / "tools/report_m1_validation.py"),
                "--evaluation", str(evaluation), "--manifest", str(manifest),
                "--training-features", config["features"]["train"], "--run", str(run)], check=True)
        phase("complete", validation_metrics=str(run / "validation/metrics.json"),
              ten_tree_metrics=str(run / "validation_ten_tree/metrics.json"),
              validation_grouping_gate=summary["scientific_gate"]["grouping_evidence_pass"],
              ten_tree_grouping_gate=ten_tree["scientific_gate"]["grouping_evidence_pass"])
    except BaseException as error:
        phase("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
