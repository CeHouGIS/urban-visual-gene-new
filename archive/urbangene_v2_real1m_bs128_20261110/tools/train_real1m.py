#!/usr/bin/env python3
"""GPU-only training with a global effective batch of 128 and resumable I/O."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from urbangene_v2.common import sha256
from urbangene_v2.discovery import DiscoveryModel
from urbangene_v2.million import (ShardedFeatures, RepeatingShardOrder, accumulated_backward,
                                  scheduled_lr, feature_batches, atomic_json as write_json)
from urbangene_v2.scaling import atomic_checkpoint, capture_rng, restore_rng
from urbangene_v2.train import _checkpoint_model_state


@torch.inference_mode()
def evaluate(model, validation, cfg, step, device, micro):
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    base.eval()
    total, count, mass, entropy, weight = 0., 0, 0., 0., 0.
    common_total, common_count = 0., 0
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        # A per-image seed makes common samples comparable across batch sizes.
        for i in range(len(validation)):
            torch.manual_seed(10000 + i)
            feature = validation[i][None].to(device)
            out = base(feature)
            loss = float(torch.nn.functional.mse_loss(out["reconstructed_features"], feature))
            total += loss
            count += 1
            if i < 512:
                common_total += loss
                common_count += 1
            masks = out["soft_masks"]
            mass += float(out["attention_masks"].mean((2, 3)).max())
            entropy += float(-(masks.clamp_min(1e-12) * masks.clamp_min(1e-12).log()).sum(1).mean())
            weight += float(masks.amax(1).mean())
    result = {"step": step, "validation_images": count, "validation_feature_mse": total / count,
              "common_512_feature_mse": common_total / common_count,
              "encoder_max_mean_attention_mass": mass / count,
              "decoder_normalized_entropy": entropy / count / math.log(cfg["model"]["max_slots"]),
              "decoder_mean_max_weight": weight / count}
    base.train()
    return result


@torch.inference_mode()
def preview(model, validation, cfg, step, device):
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    base.eval()
    directory = Path(cfg["data_root"])
    records = []
    for shard in sorted((directory / "rgb_shards/validation").glob("*.jsonl")):
        records.extend(json.loads(s) for s in shard.read_text().splitlines())
    old = json.loads(Path(cfg["comparison_provenance"]).read_text())
    selected = set(old["preview_ids"][::2])
    palette = np.random.default_rng(777).integers(35, 240, (32, 3), dtype=np.uint8)
    image = Image.new("RGB", (768, 1146), "white")
    draw = ImageDraw.Draw(image)
    for col, title in enumerate(["Original real photograph", "Encoder grouping", "Decoder grouping"]):
        draw.text((4 + col * 256, 7), title, fill="black")
    paths = Path(cfg["output"]) / "previews"
    paths.mkdir(exist_ok=True)
    row = 0
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        for i, record in enumerate(records):
            if record["id"] not in selected:
                continue
            torch.manual_seed(10000 + i)
            out = base(validation[i][None].to(device))
            with Image.open(record["image"]) as opened:
                original = opened.convert("RGB").resize((248, 248), Image.Resampling.BICUBIC)
            tiles = [original]
            saved = {}
            for name in ["attention_masks", "soft_masks"]:
                soft = out[name][0].cpu().numpy()
                colored = Image.fromarray(palette[soft.argmax(0)]).resize((248, 248), Image.Resampling.NEAREST)
                tiles.append(Image.blend(original, colored, .5))
                saved[name] = soft.astype(np.float16)
            np.savez_compressed(paths / f"step_{step:07d}_sample_{i:04d}.npz", **saved)
            y = 42 + row * 276
            draw.text((4, y - 16), f"{record['id'].split('_')[0]} | update {step}", fill="black")
            for col, tile in enumerate(tiles):
                image.paste(tile, (4 + col * 256, y))
            row += 1
    image.save(paths / f"step_{step:07d}.png")
    base.train()


def train(cfg, gpu_count):
    if not 1 <= gpu_count <= 2 or not torch.cuda.is_available():
        raise ValueError("GPU-only training requires one or two GPUs")
    torch.set_num_threads(4)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    ids = list(range(gpu_count))
    for i in ids:
        torch.cuda.set_per_process_memory_fraction(.75, i)
    device = torch.device("cuda:0")
    root, out = Path(cfg["data_root"]), Path(cfg["output"])
    dataset = ShardedFeatures(root / "features.train.index.json")
    validation = ShardedFeatures(root / "features.validation.index.json")
    if len(dataset) != cfg["train_images"] or len(validation) != cfg["validation_images"]:
        raise ValueError("Million-image data are incomplete; refusing to train a partial dataset")
    data_identity = {s: sha256(root / f"features.{s}.index.json") for s in ["train", "validation"]}
    model = DiscoveryModel(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])
    settings = cfg["training"]
    effective, micro = settings["effective_batch_size"], settings["micro_batch_per_gpu"] * gpu_count
    if effective % micro:
        raise ValueError("Effective batch must be divisible by global micro batch")
    step, epoch, offset, best = 0, 0, 0, float("inf")
    resume = out / "checkpoint_resume.pt"
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["config"] != cfg:
            raise ValueError("Resume configuration mismatch")
        if state["data_identity"] != data_identity:
            raise ValueError("Resume feature dataset identity mismatch")
        if state["gpu_count"] != gpu_count:
            raise ValueError("Exact stochastic resume requires the original GPU count")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(device)
        step, epoch, offset, best = (state[k] for k in ["step", "data_epoch", "data_offset", "best_metric"])
        restore_rng(state["rng"], ids)
    if gpu_count > 1:
        model = torch.nn.DataParallel(model, device_ids=ids)
    order = RepeatingShardOrder(dataset, cfg["seed"], epoch, offset)
    loader = feature_batches(dataset, order, effective, workers=settings["workers"], pin_memory=True)
    iterator = iter(loader)
    start = time.monotonic()
    running, window = 0., 0
    model.train()
    while step < settings["max_updates"]:
        features = next(iterator).to(device, non_blocking=True)
        lr = scheduled_lr(settings, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = accumulated_backward(model, features, effective, micro)
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
        if not torch.isfinite(grad):
            raise FloatingPointError("Nonfinite training gradient")
        optimizer.step()
        step += 1
        offset += effective
        if offset >= len(dataset):
            epoch += offset // len(dataset)
            offset %= len(dataset)
        running += loss
        window += 1
        progress = {"status": "training", "step": step, "max_updates": settings["max_updates"],
                    "train_images": len(dataset), "effective_batch_size": effective,
                    "global_micro_batch": micro, "gradient_accumulation": effective // micro,
                    "images_seen": step * effective, "dataset_passes": step * effective / len(dataset),
                    "learning_rate": lr, "train_feature_mse": running / window,
                    "best_validation_feature_mse": best if math.isfinite(best) else None,
                    "elapsed_worker_seconds": round(time.monotonic() - start, 1),
                    "gpu_ids": ids, "cpu_fallback": False, "gpu_peak_allocated_bytes":
                    [torch.cuda.max_memory_allocated(i) for i in ids],
                    "updated_utc": datetime.now(timezone.utc).isoformat()}
        if step % 20 == 0:
            write_json(out / "progress.json", progress)
            with (out / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(progress) + "\n")
            running, window = 0., 0
        validate = step % settings["evaluate_every"] == 0 or step == settings["max_updates"]
        selected = False
        if validate:
            result = evaluate(model, validation, cfg, step, device, micro)
            selected = result["validation_feature_mse"] < best
            best = min(best, result["validation_feature_mse"])
            with (out / "validation_history.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            if step % settings["preview_every"] == 0 or selected and step == settings["evaluate_every"]:
                preview(model, validation, cfg, step, device)
        if step % settings["checkpoint_every"] == 0 or step == settings["max_updates"]:
            state = {"model": _checkpoint_model_state(model), "config": cfg, "optimizer": optimizer.state_dict(),
                     "rng": capture_rng(), "step": step, "data_epoch": epoch, "data_offset": offset,
                     "best_metric": best, "data_identity": data_identity, "gpu_count": gpu_count}
            atomic_checkpoint(resume, state)
            if selected:
                atomic_checkpoint(out / "checkpoint_best_recon.pt", {k: v for k, v in state.items()
                                  if k not in ["optimizer", "rng"]})
            if step % settings["preview_every"] == 0:
                atomic_checkpoint(out / f"checkpoint_step_{step:07d}.pt", {k: v for k, v in state.items()
                                  if k not in ["optimizer", "rng"]})
        if step == settings["max_updates"]:
            write_json(out / "progress.json", {**progress, "status": "complete",
                       "best_validation_feature_mse": best})
            atomic_checkpoint(out / "checkpoint_last.pt", {k: v for k, v in state.items()
                              if k not in ["optimizer", "rng"]})
            write_json(out / "run.json", {**progress, "status": "complete", "config": cfg,
                       "best_validation_feature_mse": best, "training_annotations_used": False})
    print(json.dumps({"training": "complete", "step": step, "best_metric": best}), flush=True)
