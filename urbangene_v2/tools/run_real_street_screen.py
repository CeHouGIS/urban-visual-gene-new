#!/usr/bin/env python3
"""Real RGB -> verified DINOv3 -> CPU discovery -> label-free diagnostics."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tarfile
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from urbangene_v2.common import load_config, sha256, write_json, writable
    from urbangene_v2.extract import extract_manifest
    from urbangene_v2.train import train_discovery
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    config = load_config(a.config)
    if config["device"] != "cpu" or config["training"].get("gpu_ids"):
        raise ValueError("This screen is CPU only")
    if config.get("dataset_kind") != "real_google_street_view_photographs":
        raise ValueError("Explicit real-street data identity required")
    output = writable(config["output"])
    execution = output.with_name(output.name + "_execution.json")
    if execution.exists() or output.exists():
        raise FileExistsError("Refusing to overwrite an execution or training run")
    data_dir = Path(config["manifests"]["train"]).parent
    provenance = json.loads((data_dir / "provenance.json").read_text())
    if provenance["dataset_kind"] != config["dataset_kind"]:
        raise ValueError("Data provenance mismatch")
    for split in ("train", "validation"):
        rows = [json.loads(line) for line in Path(config["manifests"][split]).read_text().splitlines()]
        expected = [{"id": r["id"], "image": r["source_image"], "split": split}
                    for r in provenance["records"] if r["split"] == split]
        if rows != expected:
            raise ValueError(f"Image manifest/provenance mismatch: {split}")
    archive = output.with_name(output.name + "_source.tar.gz")
    with tarfile.open(archive, "x:gz") as stream:
        for folder in ("src", "tools", "configs", "docs", "tests"):
            for file in sorted((ROOT / folder).rglob("*")):
                if file.is_file() and file.suffix in {".py", ".yaml", ".md"}:
                    stream.add(file, arcname=str(file.relative_to(ROOT)))
    state = {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
             "dataset_kind": config["dataset_kind"], "device": "cpu", "gpu_ids": [],
             "config": str(Path(a.config).resolve()), "config_sha256": sha256(a.config),
             "provenance_sha256": sha256(data_dir / "provenance.json"),
             "source_snapshot": str(archive), "source_sha256": sha256(archive),
             "test_rgb_opened": False, "training_annotations_used": False,
             "phase": "prepared"}

    def phase(name, **fields):
        state.update(phase=name, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
        write_json(execution, state)
        print(json.dumps(state), flush=True)

    try:
        torch.set_num_threads(8)
        for split in ("train", "validation"):
            phase("feature_extraction", split=split, extracted=0,
                  total=provenance["split_counts"][split])
            feature = Path(config["features"][split])
            manifest = Path(config["manifests"][split])
            if feature.exists():
                meta = json.loads(feature.with_suffix(".npy.json").read_text())
                if meta["manifest_sha256"] != sha256(manifest) or meta["feature_sha256"] != sha256(feature):
                    raise ValueError("Existing cache identity mismatch")
                if meta["backbone"]["files"]["model.safetensors"] != sha256(Path(config["backbone"]["weights"]) / "model.safetensors"):
                    raise ValueError("Existing cache backbone mismatch")
            else:
                started = time.monotonic()

                def progress(done, total):
                    if done % 8 == 0 or done == total:
                        elapsed = time.monotonic() - started
                        phase("feature_extraction", split=split, extracted=done, total=total,
                              elapsed_split_seconds=round(elapsed, 2),
                              estimated_remaining_split_seconds=round(elapsed / done * (total - done), 2))

                extract_manifest(manifest, feature, config["backbone"], batch_size=2,
                                 workers=0, device="cpu", progress_callback=progress)
        phase("training", output=str(output), extracted=576, total=576)
        run = train_discovery(config)
        phase("label_free_validation", checkpoint=str(run / "checkpoint_best_recon.pt"))
        from report_real_streets import report
        report(config, run / "checkpoint_best_recon.pt", run / "validation")
        phase("complete", output=str(run),
              report=str(run / "validation/REPORT.md"),
              scientific_instance_gate="not_evaluated_no_real_instance_annotations")
    except BaseException as error:
        phase("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
