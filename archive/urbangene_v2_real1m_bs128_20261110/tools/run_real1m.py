#!/usr/bin/env python3
"""GPU-only queue: hardware smoke, sharded DINO extraction, fresh training."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from urbangene_v2.common import load_config, sha256, writable
from urbangene_v2.million import atomic_json as write_json, read_live_json
from run_real20k import lease_gpus


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def smoke(cfg, gpu_count):
    import numpy as np
    import torch
    from urbangene_v2.discovery import DiscoveryModel
    from urbangene_v2.million import accumulated_backward, ShardedFeatures, feature_batches
    if not 1 <= gpu_count <= 2 or not torch.cuda.is_available():
        raise ValueError("GPU required for model smoke")
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    for i in range(gpu_count):
        torch.cuda.set_per_process_memory_fraction(.75, i)
    source = Path(cfg["comparison_provenance"]).parent / "features.train.npy"
    index_path = Path(cfg["output"]) / "smoke.features.index.json"
    count = len(np.load(source, mmap_mode="r"))
    write_json(index_path, {"records": count, "shards": [{"path": str(source), "records": count}]})
    batches = feature_batches(ShardedFeatures(index_path), range(128), 128,
                              workers=cfg["training"]["workers"], pin_memory=True)
    features = next(batches).cuda(non_blocking=True)
    batches.close()
    model = DiscoveryModel(**cfg["model"]).cuda()
    if gpu_count == 2:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = accumulated_backward(model, features, 128, cfg["training"]["micro_batch_per_gpu"] * gpu_count)
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(grad):
            raise FloatingPointError("GPU smoke gradient is nonfinite")
        optimizer.step()
        losses.append(loss)
    write_json(Path(cfg["output"]) / "gpu_batch_smoke.json", {
        "status": "passed", "effective_batch_size": 128,
        "global_micro_batch": cfg["training"]["micro_batch_per_gpu"] * gpu_count,
        "gradient_accumulation": 128 // (cfg["training"]["micro_batch_per_gpu"] * gpu_count),
        "gpu_count": gpu_count, "losses": losses,
        "feature_loader": "threaded reads; no multiprocessing shared memory",
        "peak_gpu_memory_bytes": [torch.cuda.max_memory_allocated(i) for i in range(gpu_count)],
        "fresh_training_weights": "discarded; formal run starts from seed", "time_utc": timestamp()})


def extraction_worker(cfg, index, world):
    import numpy as np
    import torch
    from urbangene_v2.backbone import FrozenVisualBackbone
    from urbangene_v2.data import ImageManifestDataset
    from urbangene_v2.extract import _collate_rgb
    if not torch.cuda.is_available():
        raise ValueError("CPU DINO extraction is disabled")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.75, index)
    model = FrozenVisualBackbone.from_config(cfg["backbone"]).to(f"cuda:{index}").eval()
    directory, output = Path(cfg["data_root"]), Path(cfg["output"])
    progress_path = output / f"extraction_worker_{index}.json"
    batch_size = cfg["extraction"]["batch_size_per_gpu"]
    readers = ThreadPoolExecutor(max_workers=cfg["extraction"]["workers_per_gpu"])
    while True:
        preparation = directory / "preparation.json"
        if preparation.exists() and read_live_json(preparation).get("status") == "failed":
            raise RuntimeError("Data verification failed; inspect preparation log")
        work = []
        for split in ["validation", "train"]:
            for shard in sorted((directory / "rgb_shards" / split).glob("images_*.jsonl")):
                if int(shard.stem.split("_")[1]) // cfg["data_shard_size"] % world != index:
                    continue
                destination = directory / "feature_shards" / split / shard.name.replace("images_", "features_").replace(".jsonl", ".npy")
                if destination.exists() and destination.with_suffix(".npy.json").exists():
                    continue
                work.append((split, shard, destination))
        if not work:
            if (directory / "READY.json").exists():
                write_json(progress_path, {"status": "complete", "device": f"cuda:{index}", "time_utc": timestamp()})
                return
            write_json(progress_path, {"status": "waiting_for_verified_images", "device": f"cuda:{index}", "time_utc": timestamp()})
            time.sleep(15)
            continue
        for split, manifest, destination in work:
            destination.parent.mkdir(parents=True, exist_ok=True)
            lock = destination.with_suffix(".lock").open("a+")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            try:
                if destination.exists() and destination.with_suffix(".npy.json").exists():
                    continue
                dataset = ImageManifestDataset(manifest)
                partial_path = destination.with_suffix(".npy.partial")
                if partial_path.exists():
                    partial_path.rename(partial_path.with_name(partial_path.name + f".interrupted_{int(time.time())}"))
                array = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float16,
                                                  shape=(len(dataset), 1024, 28, 28))
                position = 0
                while position < len(dataset):
                    # Reading/resize on CPU; all backbone inference uses GPU.
                    rows = list(readers.map(dataset.__getitem__, range(position, min(position + batch_size, len(dataset)))))
                    images = _collate_rgb(rows, cfg["backbone"]["resolution"])["image"].to(f"cuda:{index}")
                    try:
                        with torch.inference_mode():
                            features = model(images)["features"].cpu().numpy().astype(np.float16)
                    except torch.cuda.OutOfMemoryError:
                        del images
                        torch.cuda.empty_cache()
                        if batch_size <= 1:
                            raise
                        batch_size //= 2
                        continue
                    if not np.isfinite(features).all():
                        raise FloatingPointError("Nonfinite float16 feature cache")
                    array[position:position + len(features)] = features
                    position += len(features)
                    if position % 128 == 0 or position == len(dataset):
                        write_json(progress_path, {"status": "extracting", "split": split,
                            "manifest": str(manifest), "extracted_in_shard": position,
                            "shard_images": len(dataset), "batch_size": batch_size,
                            "device": f"cuda:{index}", "time_utc": timestamp()})
                array.flush()
                del array
                partial_path.replace(destination)
                write_json(destination.with_suffix(".npy.json"), {
                    "manifest": str(manifest), "manifest_sha256": sha256(manifest),
                    "path": str(destination), "feature_sha256": sha256(destination),
                    "records": len(dataset), "ids": [r["id"] for r in dataset.records],
                    "shape": [len(dataset), 1024, 28, 28], "dtype": "float16",
                    "backbone": model.provenance, "resolution": 448, "patch_size": 16,
                    "image_input": "RGB [0,1], square bicubic resize, no crop",
                    "normalization": {"mean": [.485, .456, .406], "std": [.229, .224, .225]},
                    "training_labels_loaded": False, "gpu": index, "batch_size": batch_size})
            finally:
                lock.close()


def build_indices(cfg):
    directory = Path(cfg["data_root"])
    ready = read_live_json(directory / "READY.json")
    weight_hash = sha256(Path(cfg["backbone"]["weights"]) / "model.safetensors")
    old = json.loads(Path(cfg["comparison_provenance"]).read_text())
    protected = [r for r in old["records"] if r["split"] == "validation"]
    with sqlite3.connect(f"file:{directory / 'verification.sqlite'}?mode=ro", uri=True) as db:
        for record in protected:
            actual = db.execute("SELECT sha256,split FROM accepted WHERE id=?", (record["id"],)).fetchone()
            if actual != (record["source_sha256"], "validation"):
                raise ValueError("Common validation photograph changed")
    for split in ["train", "validation"]:
        manifests = sorted((directory / "rgb_shards" / split).glob("images_*.jsonl"))
        records, shards, seen = 0, [], set()
        for manifest in manifests:
            feature = directory / "feature_shards" / split / manifest.name.replace("images_", "features_").replace(".jsonl", ".npy")
            meta = json.loads(feature.with_suffix(".npy.json").read_text())
            ids = [json.loads(s)["id"] for s in manifest.read_text().splitlines()]
            if meta["ids"] != ids or meta["manifest_sha256"] != sha256(manifest):
                raise ValueError("Feature/image alignment mismatch")
            if (meta["backbone"]["files"]["model.safetensors"] != weight_hash
                    or meta["resolution"] != cfg["backbone"]["resolution"]
                    or meta["shape"] != [len(ids), 1024, 28, 28] or meta["dtype"] != "float16"):
                raise ValueError("Feature backbone or shape identity mismatch")
            if sha256(feature) != meta["feature_sha256"]:
                raise ValueError("Feature shard checksum mismatch")
            if seen.intersection(ids):
                raise ValueError("Repeated sample IDs in feature shards")
            seen.update(ids)
            records += len(ids)
            shards.append({"path": str(feature), "records": len(ids),
                           "feature_sha256": meta["feature_sha256"], "manifest": str(manifest)})
        if records != ready["split_counts"][split]:
            raise ValueError("Incomplete million-image feature dataset")
        write_json(directory / f"features.{split}.index.json", {
            "records": records, "shards": shards, "grid_hw": [28, 28], "channels": 1024,
            "backbone_weights_sha256": weight_hash,
            "data_ready_sha256": sha256(directory / "READY.json"), "training_annotations_used": False})


def run(config_path):
    cfg = load_config(config_path)
    output = writable(cfg["output"])
    output.mkdir(exist_ok=True)
    lock = (output / "execution.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"pid": os.getpid(), "config": str(Path(config_path).resolve()),
             "config_sha256": sha256(config_path), "compute_policy": "gpu_only", "max_gpus": 2,
             "cpu_fallback": False, "train_target": cfg["train_images"],
             "validation_target": cfg["validation_images"], "effective_batch_size": 128,
             "started_utc": timestamp()}

    def phase(name, **extra):
        state.update(phase=name, updated_utc=timestamp(), **extra)
        write_json(output / "execution.json", state)
        print(json.dumps(state), flush=True)

    try:
        # The older experiment already occupies two GPUs. Do not add a third.
        dependency = Path(cfg["dependency_execution"])
        while dependency.exists():
            previous = read_live_json(dependency)
            if previous["phase"] in {"complete", "stopped_by_user"}:
                break
            if previous["phase"] == "failed":
                raise RuntimeError("Previous 20k experiment failed; GPU ownership needs inspection")
            phase("waiting_for_previous_experiment", dependency_phase=previous["phase"])
            time.sleep(cfg["compute"]["gpu_poll_seconds"])
        # Avoid reserving GPUs while no image shard is ready.
        directory = Path(cfg["data_root"])
        while not list((directory / "rgb_shards/train").glob("*.jsonl")):
            if (directory / "preparation.json").exists() and read_live_json(directory / "preparation.json").get("status") == "failed":
                raise RuntimeError("Data preparation failed")
            phase("waiting_for_verified_images")
            time.sleep(30)
        while True:
            chosen, leases = lease_gpus(output, limit=2)
            if chosen:
                break
            phase("waiting_for_gpu", physical_gpus=[])
            time.sleep(cfg["compute"]["gpu_poll_seconds"])
        state["physical_gpus"] = chosen
        if shutil.disk_usage(cfg["data_root"]).free < cfg["extraction"]["minimum_free_bytes"]:
            for handle in leases:
                handle.close()
            raise OSError("Insufficient NAS space for the one-million-image feature cache")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(g["uuid"] for g in chosen),
                   PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--config", str(config_path)]
        try:
            if not (output / "gpu_batch_smoke.json").exists():
                phase("gpu_batch_smoke")
                subprocess.run(command + ["--worker", "smoke", "--gpu-count", str(len(chosen))], env=env, check=True)
            phase("feature_extraction")
            children = []
            try:
                children = [subprocess.Popen(command + ["--worker", "extract", "--index", str(i),
                                "--gpu-count", str(len(chosen))], env=env) for i in range(len(chosen))]
                while any(child.poll() is None for child in children):
                    for child in children:
                        if child.poll() not in (None, 0):
                            raise subprocess.CalledProcessError(child.returncode, child.args)
                    time.sleep(5)
                for child in children:
                    if child.returncode:
                        raise subprocess.CalledProcessError(child.returncode, child.args)
            finally:
                for child in children:
                    if child.poll() is None:
                        child.terminate()
                for child in children:
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
            phase("checking_feature_shards")
            build_indices(cfg)
            phase("training_new_model")
            subprocess.run(command + ["--worker", "train", "--gpu-count", str(len(chosen))], env=env, check=True)
            phase("complete")
        finally:
            for handle in leases:
                handle.close()
    except BaseException as error:
        phase("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--worker", choices=["smoke", "extract", "train"])
    p.add_argument("--gpu-count", type=int, default=0)
    p.add_argument("--index", type=int, default=0)
    a = p.parse_args()
    if a.worker:
        if not 1 <= a.gpu_count <= 2:
            raise ValueError("Model workers require one or two GPUs; CPU fallback disabled")
        config = load_config(a.config)
        if a.worker == "smoke":
            smoke(config, a.gpu_count)
        elif a.worker == "extract":
            extraction_worker(config, a.index, a.gpu_count)
        else:
            from train_real1m import train
            train(config, a.gpu_count)
    else:
        run(a.config)
