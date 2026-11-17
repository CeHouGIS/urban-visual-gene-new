#!/usr/bin/env python3
"""Persistently run resumable feature shards and training, with <=2 idle GPUs."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from urbangene_v2.common import load_config, sha256, write_json, writable


def idle_gpus():
    """Require both an empty process list and low memory/utilization."""
    try:
        info = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                        "--format=csv,noheader,nounits"], text=True, timeout=15)
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                                        "--format=csv,noheader"], text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    busy = {s.strip() for s in apps.splitlines()}
    selected = []
    for line in info.splitlines():
        index, uuid, memory, utilization = [s.strip() for s in line.split(",")]
        if uuid not in busy and int(memory) < 1024 and int(utilization) < 10:
            selected.append({"index": int(index), "uuid": uuid})
    return selected[:2]


def lease_gpus(output, limit=2):
    handles, chosen = [], []
    lock_dir = writable(output.parent / "gpu_leases")
    lock_dir.mkdir(exist_ok=True)
    for gpu in idle_gpus()[:limit]:
        handle = (lock_dir / (gpu["uuid"] + ".lock")).open("a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        handles.append(handle)
        chosen.append(gpu)
    return chosen, handles


def wait_for_gpus(output, limit, on_wait, poll_seconds=30):
    """Queue only; never launch model work with an empty GPU selection."""
    while True:
        chosen, leases = lease_gpus(output, limit=limit)
        if chosen:
            return chosen, leases
        on_wait()
        time.sleep(poll_seconds)


def worker(config_path, phase, gpu_ids, split=None, start=None, end=None):
    if not gpu_ids or len(gpu_ids) > 2:
        raise ValueError("GPU-only experiment: select one or two GPUs; CPU fallback is disabled")
    config = load_config(config_path)
    if phase == "train":
        from urbangene_v2.scaling import train_epoch
        return train_epoch(config, gpu_ids)
    if phase == "report":
        from report_real_streets import report
        return report(config, Path(config["output"]) / "checkpoint_best_recon.pt",
                      Path(config["output"]) / "validation", device=f"cuda:{gpu_ids[0]}")
    import numpy as np
    import torch
    from urbangene_v2.extract import extract_manifest
    torch.set_num_threads(config["extraction"]["cpu_threads"])
    directory = Path(config["manifests"][split]).parent
    shards = directory / "shards" / split
    shards.mkdir(parents=True, exist_ok=True)
    path = shards / f"features_{start:05d}_{end:05d}.npy"
    manifest = shards / f"images_{start:05d}_{end:05d}.jsonl"
    rows = [json.loads(line) for line in Path(config["manifests"][split]).read_text().splitlines()]
    selected = rows[start:end]
    text = "".join(json.dumps(r) + "\n" for r in selected)
    if manifest.exists() and manifest.read_text() != text:
        raise ValueError("Shard manifest changed")
    manifest.write_text(text)
    sidecar = path.with_suffix(".npy.json")
    weight_hash = sha256(Path(config["backbone"]["weights"]) / "model.safetensors")
    if path.exists():
        meta = json.loads(sidecar.read_text())
        if (meta["feature_sha256"] != sha256(path) or meta["manifest_sha256"] != sha256(manifest)
                or meta["backbone"]["files"]["model.safetensors"] != weight_hash):
            raise ValueError("Completed shard has invalid identity")
        return
    # Reuse only exact RGB identities from the verified November 9 run.
    reuse = config["verified_reuse"]
    old_manifest = Path(reuse["manifests"][split])
    old_rows = [json.loads(line) for line in old_manifest.read_text().splitlines()]
    old_by_id = {r["id"]: (i, r) for i, r in enumerate(old_rows)}
    if all(r["id"] in old_by_id and old_by_id[r["id"]][1] == r for r in selected):
        old_feature = Path(reuse["features"][split])
        meta = json.loads(old_feature.with_suffix(".npy.json").read_text())
        if (meta["feature_sha256"] != sha256(old_feature) or meta["manifest_sha256"] != sha256(old_manifest)
                or meta["backbone"]["files"]["model.safetensors"] != weight_hash):
            raise ValueError("Reuse source identity mismatch")
        old = np.load(old_feature, mmap_mode="r")
        array = np.stack([old[old_by_id[r["id"]][0]] for r in selected])
        temporary = path.with_suffix(".npy.reuse.partial")
        with temporary.open("xb") as stream:
            np.save(stream, array)
        temporary.replace(path)
        meta.update(manifest=str(manifest), manifest_sha256=sha256(manifest),
                    feature_file=str(path), feature_sha256=sha256(path), shape=list(array.shape),
                    ids=[r["id"] for r in selected], records=len(selected),
                    reused_from=str(old_feature), training_labels_loaded=False)
        write_json(sidecar, meta)
        return
    # An incomplete shard is retained, while a fresh attempt gets its own name.
    attempt = path
    attempt_number = 0
    while attempt.with_suffix(attempt.suffix + ".partial").exists():
        attempt_number += 1
        attempt = path.with_name(path.stem + f"_attempt{attempt_number}.npy")
    progress_path = directory / "feature_progress.json"

    def progress(done, total):
        if done % 16 == 0 or done == total:
            write_json(progress_path, {"split": split, "shard_start": start, "shard_end": end,
                       "completed_in_shard": done, "device": f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu",
                       "updated_utc": datetime.now(timezone.utc).isoformat()})

    extract_manifest(manifest, attempt, config["backbone"], batch_size=4 if gpu_ids else 2,
                     workers=0, device=f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu",
                     progress_callback=progress)
    if attempt != path:
        attempt.replace(path)
        meta = json.loads(attempt.with_suffix(".npy.json").read_text())
        meta["feature_file"] = str(path)
        write_json(sidecar, meta)
        attempt.with_suffix(".npy.json").unlink()


def merge(config, split):
    import numpy as np
    output = Path(config["features"][split])
    manifest = Path(config["manifests"][split])
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    shards = sorted(p for p in (manifest.parent / "shards" / split).glob("features_*.npy")
                    if re.fullmatch(r"features_\d{5}_\d{5}\.npy", p.name))
    if output.exists():
        meta = json.loads(output.with_suffix(".npy.json").read_text())
        if meta["manifest_sha256"] != sha256(manifest) or meta["feature_sha256"] != sha256(output):
            raise ValueError("Combined cache identity mismatch")
        return
    paths = []
    ids = []
    for shard in shards:
        meta = json.loads(shard.with_suffix(".npy.json").read_text())
        if sha256(shard) != meta["feature_sha256"]:
            raise ValueError("Shard hash mismatch before merge")
        paths.append({"path": str(shard), "sha256": meta["feature_sha256"]})
        ids.extend(meta["ids"])
    if ids != [r["id"] for r in records]:
        raise ValueError("Shard order differs from image manifest")
    partial = output.with_suffix(".npy.partial")
    if partial.exists():
        destination = partial.with_name(partial.name + f".interrupted_{int(time.time())}")
        partial.rename(destination)
    shape = (len(records), 1024, 28, 28)
    array = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=shape)
    position = 0
    for shard in shards:
        data = np.load(shard, mmap_mode="r")
        array[position:position + len(data)] = data
        position += len(data)
    array.flush()
    del array
    partial.replace(output)
    meta = json.loads(shards[0].with_suffix(".npy.json").read_text())
    meta.update(manifest=str(manifest), manifest_sha256=sha256(manifest), feature_file=str(output),
                feature_sha256=sha256(output), shape=list(shape), ids=ids, records=len(ids),
                merged_shards=paths)
    write_json(output.with_suffix(".npy.json"), meta)


def run(config_path):
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    output = writable(config["output"])
    output.mkdir(exist_ok=True)
    lock = (output / "execution.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = output / "execution.json"
    state = {"pid": os.getpid(), "config": str(config_path), "config_sha256": sha256(config_path),
             "started_utc": datetime.now(timezone.utc).isoformat(), "max_gpus": 2,
             "training_annotations_used": False, "test_images_opened": False,
             "compute_policy": "gpu_only", "cpu_fallback": False, "gpu_poll_seconds": 30}

    def phase(name, **extra):
        state.update(phase=name, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        write_json(state_path, state)
        print(json.dumps(state), flush=True)

    def call_worker(name, split=None, start=None, end=None):
        def waiting():
            phase("waiting_for_gpu", pending_work=name, physical_gpus=[], device=None,
                  split=split, shard_start=start, shard_end=end)

        chosen, leases = wait_for_gpus(output, limit=1 if name in {"extract", "report"} else 2,
                                      on_wait=waiting)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(g["uuid"] for g in chosen),
                   PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        gpu_ids = list(range(len(chosen)))
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--config", str(config_path),
                   "--worker", name, "--gpu-ids", ",".join(map(str, gpu_ids))]
        if split is not None:
            command += ["--split", split, "--start", str(start), "--end", str(end)]
        phase({"extract": "feature_extraction", "train": "training", "report": "final_label_free_validation"}[name],
              physical_gpus=chosen, device="cuda", pending_work=None,
              split=split, shard_start=start, shard_end=end)
        try:
            subprocess.run(command, env=env, check=True)
        finally:
            for handle in leases:
                handle.close()

    try:
        data = Path(config["manifests"]["train"]).parent
        phase("waiting_for_prepared_images")
        while not (data / "provenance.json").exists():
            time.sleep(15)
        provenance = json.loads((data / "provenance.json").read_text())
        if provenance["split_counts"] != config["expected_counts"] or provenance["panorama_overlap"] != 0:
            raise ValueError("Prepared data violates experiment protocol")
        state["provenance_sha256"] = sha256(data / "provenance.json")
        for split in ("train", "validation"):
            rows = [json.loads(line) for line in Path(config["manifests"][split]).read_text().splitlines()]
            expected = [{"id": r["id"], "image": r["source_image"], "split": split}
                        for r in provenance["records"] if r["split"] == split]
            if rows != expected:
                raise ValueError("Prepared manifest identity mismatch")
            shard_size = config["extraction"]["shard_size"]
            # Keep the original 64 validation items in their own reusable shard.
            boundaries = list(range(0, len(rows), shard_size)) + [len(rows)]
            if split == "validation" and 64 not in boundaries:
                boundaries = sorted(set(boundaries + [64]))
            for start, end in zip(boundaries, boundaries[1:]):
                # Already committed shards are verified in merge before training.
                shard = data / "shards" / split / f"features_{start:05d}_{end:05d}.npy"
                if shard.exists() and shard.with_suffix(".npy.json").exists():
                    continue
                call_worker("extract", split=split, start=start, end=end)
            phase("combining_feature_shards", split=split)
            merge(config, split)
        while True:
            progress_file = output / "progress.json"
            if progress_file.exists():
                progress = json.loads(progress_file.read_text())
                if progress.get("status") == "complete":
                    break
            call_worker("train")
        phase("final_label_free_validation")
        if not (output / "validation/metrics.json").exists():
            call_worker("report")
        metrics = json.loads((output / "validation/metrics.json").read_text())
        baseline = json.loads(Path(config["baseline_metrics"]).read_text())
        if [r["id"] for r in metrics["per_image"][:64]] != [r["id"] for r in baseline["per_image"]]:
            raise ValueError("Common-64 comparison identities differ")
        fields = list(metrics["means"])
        write_json(output / "validation/comparison_common64.json", {
            "evaluation_images": 64, "same_ids_and_order": True, "sampling_seed_base": 10000,
            "baseline_512_train": baseline["means"],
            "expanded_20000_train": {k: sum(r[k] for r in metrics["per_image"][:64]) / 64 for k in fields},
            "limit": "Training size, panorama view multiplicity and optimization budget change together; single seed, no instance annotations"})
        phase("complete", report=str(output / "validation/REPORT.md"))
    except BaseException as error:
        phase("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker", choices=["extract", "train", "report"])
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--split")
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()
    if args.worker:
        worker(args.config, args.worker, [int(i) for i in args.gpu_ids.split(",") if i],
               args.split, args.start, args.end)
    else:
        run(args.config)
