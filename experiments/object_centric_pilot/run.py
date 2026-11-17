"""Train discovery, freeze it, then fit a small RGB reconstruction probe."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from model import SlotModel, upsample_rgb


def load_data(root):
    archive = np.load(root / "dataset.npz")
    features = torch.from_numpy(archive["features"].astype(np.float32))
    rgb = torch.from_numpy(archive["rgb"].astype(np.float32)).permute(0, 3, 1, 2)
    manifest = json.loads((root / "manifest.json").read_text())
    splits = {s: torch.tensor([i for i, r in enumerate(manifest["records"]) if r["split"] == s])
              for s in ("train", "validation", "cross_city")}
    return features, rgb, manifest, splits


@torch.no_grad()
def evaluate(model, features, indices, device, rgb=None):
    model.eval()
    values, rgbs = [], []
    for batch in indices.split(4):
        x = features[batch].to(device)
        outputs = model(x)
        values.append(((outputs["features"] - x) ** 2).mean((1, 2)).cpu())
        if rgb is not None:
            pred, _ = model.render(outputs["slots"], outputs["masks"])
            rgbs.append(((upsample_rgb(pred) - rgb[batch].to(device)) ** 2).mean((1, 2, 3)).cpu())
    result = {"feature_mse": float(torch.cat(values).mean())}
    if rgbs:
        mse = float(torch.cat(rgbs).mean())
        result.update(rgb_mse=mse, rgb_psnr_db=-10 * math.log10(max(mse, 1e-12)))
    return result


def train(args):
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.04, int(args.device.split(":")[-1]))
        torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    features, rgb, manifest, splits = load_data(args.data)
    train_indices = splits["train"]
    model = SlotModel(slots=args.slots, rgb_mode=args.rgb_mode,
                      position_strength=args.position_strength).to(device)
    starting_step = 0
    if args.initial_checkpoint:
        initial_state = torch.load(args.initial_checkpoint, map_location=device, weights_only=True)
        incompatible = model.load_state_dict(initial_state["model"], strict=False)
        assert not incompatible.unexpected_keys
        assert all(name.startswith("rgb_conv.") for name in incompatible.missing_keys)
        starting_step = initial_state["discovery_step"]
    train_parameters = [p for name, p in model.named_parameters() if not name.startswith("rgb_")]
    optimizer = torch.optim.Adam(train_parameters, lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    started = time.monotonic()
    initial = evaluate(model, features, splits["validation"], device)
    print(json.dumps({"initial": initial}), flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        selected = train_indices[torch.randint(len(train_indices), (args.batch,))]
        x = features[selected].to(device)
        if torch.rand(()) < .5:
            x = x.reshape(-1, 28, 28, 64).flip(2).reshape(-1, 784, 64)
        outputs = model(x)
        loss = F.mse_loss(outputs["features"], x)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_parameters, 1.)
        optimizer.step()
        if step % 100 == 0:
            print(json.dumps({"stage": "discovery", "step": step, "train_mse": float(loss.detach()),
                              "seconds": round(time.monotonic() - started, 1)}), flush=True)
        if step % 500 == 0 or step == args.steps:
            validation = evaluate(model, features, splits["validation"], device)
            history.append({"stage": "discovery", "step": step, **validation})
            if validation["feature_mse"] < best:
                best = validation["feature_mse"]
                torch.save({"model": model.state_dict(), "slots": args.slots, "step": starting_step + step}, args.output / "discovery.pt")
            print(json.dumps({"validation": validation, "step": step}), flush=True)
    state = torch.load(args.output / "discovery.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    renderer = model.rgb_conv if args.rgb_mode == "features" else model.rgb_decoder
    for p in renderer.parameters():
        p.requires_grad_(True)
    model.eval()
    # All training slots use the same deterministic evaluation initialization.
    all_slots, all_masks = [], []
    with torch.no_grad():
        for batch in torch.arange(len(features)).split(4):
            outputs = model(features[batch].to(device))
            all_slots.append(outputs["slots"].cpu())
            all_masks.append(outputs["masks"].cpu())
    all_slots, all_masks = torch.cat(all_slots), torch.cat(all_masks)
    optimizer = torch.optim.Adam(renderer.parameters(), lr=5e-4)
    best_rgb = float("inf")
    for step in range(1, args.rgb_steps + 1):
        selected = train_indices[torch.randint(len(train_indices), (args.batch,))]
        prediction, _ = model.render(all_slots[selected].to(device), all_masks[selected].to(device))
        loss = F.mse_loss(upsample_rgb(prediction), rgb[selected].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(json.dumps({"stage": "rgb_probe", "step": step, "mse": float(loss.detach()),
                              "seconds": round(time.monotonic() - started, 1)}), flush=True)
        if step % 500 == 0 or step == args.rgb_steps:
            val = evaluate(model, features, splits["validation"], device, rgb)
            history.append({"stage": "rgb_probe", "step": step, **val})
            if val["rgb_mse"] < best_rgb:
                best_rgb = val["rgb_mse"]
                torch.save({"model": model.state_dict(), "slots": args.slots,
                            "seed": args.seed, "data": str(args.data),
                            "discovery_step": state["step"], "rgb_step": step,
                            "rgb_mode": args.rgb_mode,
                            "position_strength": args.position_strength}, args.output / "model.pt")
    state = torch.load(args.output / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    results = {split: evaluate(model, features, indices, device, rgb)
               for split, indices in splits.items()}
    mean_features = features[train_indices].mean(0)
    mean_rgb = rgb[train_indices].mean(0)
    for split in results:
        idx = splits[split]
        results[split]["mean_layout_feature_mse"] = float(((features[idx] - mean_features) ** 2).mean())
        results[split]["mean_layout_rgb_mse"] = float(((rgb[idx] - mean_rgb) ** 2).mean())
    report = {"backbone": manifest["backbone"], "slots": args.slots,
              "training_labels": False, "initial_validation": initial,
              "training_steps": starting_step + args.steps, "rgb_steps": args.rgb_steps,
              "batch_size": args.batch, "learning_rate": args.lr, "rgb_mode": args.rgb_mode,
              "encoder_position_strength": args.position_strength,
              "initial_checkpoint": str(args.initial_checkpoint) if args.initial_checkpoint else None,
              "checkpoint_discovery_step": state["discovery_step"],
              "pca_retained_variance": manifest["pca_retained_variance"],
              "split_counts": {k: len(v) for k, v in splits.items()},
              "metrics": results, "history": history,
              "elapsed_seconds": time.monotonic() - started,
              "max_gpu_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0,
              "scope": "Feature reconstruction plus a small RGB renderer; not autoregressive, not diffusion, not an instance-count guarantee."}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data/dinov3")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/k16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--rgb-steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--rgb-mode", choices=["direct", "features"], default="direct")
    parser.add_argument("--position-strength", type=float, default=1.)
    train(parser.parse_args())
