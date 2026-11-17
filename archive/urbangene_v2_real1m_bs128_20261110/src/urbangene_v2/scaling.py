"""Resumable epoch training for the real-20k data/budget experiment.

The discovery architecture and MSE objective are imported unchanged.
"""
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from .common import sha256, source_hashes, writable, write_json
from .data import FeatureDataset
from .discovery import DiscoveryModel
from .train import resolve_training_gpu_ids, _checkpoint_model_state


def atomic_checkpoint(path, state):
    path = writable(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state, gpu_ids):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    for i in gpu_ids:
        if i < len(state["cuda"]):
            torch.cuda.set_rng_state(state["cuda"][i], i)


@torch.inference_mode()
def preview(model, config, epoch, device):
    output = writable(Path(config["output"]) / "previews")
    output.mkdir(exist_ok=True)
    provenance = json.loads((Path(config["manifests"]["train"]).parent / "provenance.json").read_text())
    records = [json.loads(s) for s in Path(config["manifests"]["validation"]).read_text().splitlines()]
    features = np.load(config["features"]["validation"], mmap_mode="r")
    by_id = {r["id"]: (i, r) for i, r in enumerate(records)}
    ids = provenance["preview_ids"][::2]
    palette = np.random.default_rng(777).integers(35, 240, (32, 3), dtype=np.uint8)
    size, gap, row_height = 248, 8, 276
    canvas = Image.new("RGB", (3 * (size + gap), 42 + len(ids) * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    for col, title in enumerate(("Original real photograph", "Encoder grouping", "Decoder grouping")):
        draw.text((col * (size + gap) + 4, 7), title, fill="black")
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
        for row, identifier in enumerate(ids):
            index, record = by_id[identifier]
            torch.manual_seed(10000 + index)
            feature = torch.from_numpy(features[index].astype(np.float32))[None].to(device)
            prediction = model(feature)
            with Image.open(record["image"]) as opened:
                original = opened.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
            tiles = [original]
            for key in ("attention_masks", "soft_masks"):
                label = prediction[key][0].argmax(0).cpu().numpy()
                region = Image.fromarray(palette[label]).resize((size, size), Image.Resampling.NEAREST)
                tiles.append(Image.blend(original, region, .5))
            y = 42 + row * row_height
            draw.text((4, y - 16), f"{identifier.split('_')[0]} | epoch {epoch}", fill="black")
            for col, tile in enumerate(tiles):
                canvas.paste(tile, (col * (size + gap) + 4, y))
    canvas.save(output / f"epoch_{epoch:03d}.png")


def train_epoch(config, gpu_ids):
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["training"].get("cpu_threads", 4))
    device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")
    gpu_ids = resolve_training_gpu_ids(device, gpu_ids, torch.cuda.device_count() if gpu_ids else 0)
    if gpu_ids:
        for i in gpu_ids:
            torch.cuda.set_per_process_memory_fraction(.4, i)
    output = writable(config["output"])
    output.mkdir(exist_ok=True)
    resume = output / "checkpoint_resume.pt"
    model = DiscoveryModel(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    steps, epoch, best = 0, 0, float("inf")
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["config"] != config:
            raise ValueError("Resume configuration differs from original experiment")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for values in optimizer.state.values():
            for name, value in values.items():
                if torch.is_tensor(value):
                    values[name] = value.to(device)
        steps, epoch, best = state["steps"], state["epoch"], state["best_metric"]
        restore_rng(state["rng"], gpu_ids)
    if epoch >= config["training"]["epochs"]:
        return {"status": "complete", "epoch": epoch, "steps": steps}
    dataset = FeatureDataset(config["features"]["train"])
    val = FeatureDataset(config["features"]["validation"])
    if len(dataset) != config["expected_counts"]["train"] or len(val) != config["expected_counts"]["validation"]:
        raise ValueError("Training feature count does not match protocol")
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    validation = DataLoader(val, batch_size=config["training"]["batch_size"], shuffle=False)
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
    compute = {"device": str(device), "gpu_ids": gpu_ids,
               "gpu_names": [torch.cuda.get_device_name(i) for i in gpu_ids]}
    started, total_loss, samples = time.monotonic(), 0., 0
    model.train()
    history_path = output / "history.jsonl"
    with history_path.open("a") as history:
        for batch in loader:
            features = batch["features"].to(device)
            result = model(features)
            loss = torch.nn.functional.mse_loss(result["reconstructed_features"], features)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite feature reconstruction loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            if not torch.isfinite(grad):
                raise FloatingPointError("Nonfinite discovery gradient")
            optimizer.step()
            steps += 1
            samples += len(features)
            total_loss += float(loss.detach()) * len(features)
            if steps % 100 == 0:
                record = {"status": "training", "epoch": epoch + 1, "steps": steps,
                          "epoch_images_seen": samples, "train_reconstruction_mse": total_loss / samples,
                          "epoch_seconds_so_far": round(time.monotonic() - started, 2),
                          "compute": compute}
                history.write(json.dumps(record) + "\n")
                history.flush()
                write_json(output / "progress.json", record)
            if steps == 1280:
                atomic_checkpoint(output / "checkpoint_step_001280.pt", {
                    "model": _checkpoint_model_state(model), "config": config, "seed": seed,
                    "steps": steps, "epoch_fraction": epoch + samples / len(dataset)})
    model.eval()
    mse, count, encoder_mass, entropy, max_weight = 0., 0, 0., 0., 0.
    with torch.random.fork_rng(devices=gpu_ids), torch.inference_mode():
        torch.manual_seed(seed + 100000)
        for batch in validation:
            features = batch["features"].to(device)
            result = model(features)
            masks = result["soft_masks"]
            n = len(features)
            mse += float(torch.nn.functional.mse_loss(result["reconstructed_features"], features)) * n
            encoder_mass += float(result["attention_masks"].mean((2, 3)).amax(1).sum())
            entropy += float(-(masks.clamp_min(1e-12) * masks.clamp_min(1e-12).log()).sum(1).mean((1, 2)).sum())
            max_weight += float(masks.amax(1).mean((1, 2)).sum())
            count += n
    epoch += 1
    score = mse / count
    selected = score < best
    best = min(score, best)
    checkpoint = {"model": _checkpoint_model_state(model), "config": config, "seed": seed,
                  "optimizer": optimizer.state_dict(), "rng": capture_rng(), "epoch": epoch,
                  "steps": steps, "best_metric": best, "compute": compute,
                  "validation_reconstruction_mse": score}
    atomic_checkpoint(resume, checkpoint)
    lightweight = {k: v for k, v in checkpoint.items() if k not in ("optimizer", "rng")}
    if selected:
        atomic_checkpoint(output / "checkpoint_best_recon.pt", lightweight)
    if epoch in config["training"]["milestone_epochs"]:
        atomic_checkpoint(output / f"checkpoint_epoch_{epoch:03d}.pt", lightweight)
        preview(model.module if isinstance(model, torch.nn.DataParallel) else model,
                config, epoch, device)
    progress = {"status": "complete" if epoch == config["training"]["epochs"] else "epoch_complete",
        "epoch": epoch, "steps": steps, "train_images": len(dataset), "validation_images": count,
        "train_reconstruction_mse": total_loss / samples,
        "validation_reconstruction_mse": score, "best_validation_reconstruction_mse": best,
        "encoder_max_mean_attention_mass": encoder_mass / count,
        "decoder_normalized_entropy": entropy / count / math.log(config["model"]["max_slots"]),
        "decoder_mean_max_weight": max_weight / count,
        "epoch_seconds": round(time.monotonic() - started, 2), "compute": compute,
        "training_annotations_used": False}
    with (output / "epochs.jsonl").open("a") as stream:
        stream.write(json.dumps(progress) + "\n")
    write_json(output / "progress.json", progress)
    if epoch == config["training"]["epochs"]:
        atomic_checkpoint(output / "checkpoint_last.pt", lightweight)
        write_json(output / "run.json", {**progress, "config": config, "source_sha256": source_hashes(),
                   "best_selection": "lowest validation feature reconstruction MSE"})
    print(json.dumps(progress), flush=True)
    return progress
