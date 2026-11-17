import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .common import fresh_dir, source_hashes, write_json, writable, sha256
from .data import FeatureDataset
from .discovery import DiscoveryModel


def resolve_training_gpu_ids(device, configured_ids, visible_gpu_count):
    """Resolve an explicit, at-most-two-GPU training selection."""
    ids = list(configured_ids or [])
    if len(ids) > 2:
        raise ValueError("UrbanGene V2 training is capped at two GPUs")
    if len(set(ids)) != len(ids) or any(not isinstance(i, int) or i < 0 for i in ids):
        raise ValueError("training.gpu_ids must contain distinct nonnegative integers")
    if device.type != "cuda":
        if ids:
            raise ValueError("training.gpu_ids can only be used with a CUDA device")
        return []
    primary = torch.cuda.current_device() if device.index is None else device.index
    if not ids:
        ids = [primary]
    if any(i >= visible_gpu_count for i in ids):
        raise ValueError(f"Requested GPU IDs {ids} but only {visible_gpu_count} are visible")
    if ids[0] != primary:
        raise ValueError("The first training.gpu_ids entry must match the primary CUDA device")
    return ids


def _checkpoint_model_state(model):
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    return base.state_dict()


def train_discovery(config):
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["training"].get("cpu_threads", 2))
    device = torch.device(config["device"])
    gpu_ids = resolve_training_gpu_ids(device, config["training"].get("gpu_ids"),
                                       torch.cuda.device_count() if device.type == "cuda" else 0)
    gpu_names = [torch.cuda.get_device_name(i) for i in gpu_ids]
    output = fresh_dir(config["output"])
    dataset = FeatureDataset(config["features"]["train"], config["features"].get("grid_hw"))
    max_items = config["features"].get("max_train_items")
    if max_items:
        dataset = Subset(dataset, range(min(int(max_items), len(dataset))))
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"],
                        shuffle=True, drop_last=False)
    validation_path = config["features"].get("validation")
    validation_loader = None
    if validation_path:
        validation_data = FeatureDataset(validation_path, config["features"].get("grid_hw"))
        validation_loader = DataLoader(validation_data, batch_size=config["training"]["batch_size"],
                                       shuffle=False, drop_last=False)
    model = DiscoveryModel(**config["model"]).to(device)
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    max_steps = config["training"].get("max_steps")
    history, step, best_metric = [], 0, float("inf")
    started = time.monotonic()
    write_json(output / "progress.json", {
        "status": "running", "epoch": 0, "steps": 0,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "compute": {"device": str(device), "gpu_ids": gpu_ids, "gpu_names": gpu_names},
        "config": config, "annotation_files_opened": False,
    })
    for epoch in range(config["training"]["epochs"]):
        epoch_start = time.monotonic()
        epoch_losses = []
        model.train()
        for batch in loader:
            features = batch["features"].to(device)
            output_dict = model(features)
            loss = torch.nn.functional.mse_loss(output_dict["reconstructed_features"], features)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite feature reconstruction loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"].get("gradient_clip", 1.0))
            if not torch.isfinite(grad):
                raise FloatingPointError("Non-finite discovery gradient")
            optimizer.step()
            step += 1
            history.append({"step": step, "epoch": epoch + 1, "loss": float(loss.detach()),
                            "grad_norm": float(grad)})
            epoch_losses.append(float(loss.detach()))
            if max_steps and step >= max_steps:
                break
        model.eval()
        validation_loss = None
        if validation_loader is not None:
            losses = []
            # Use fixed slot initializations each epoch without changing the
            # training RNG. Otherwise validation noise can select checkpoints.
            with torch.random.fork_rng(devices=gpu_ids), torch.inference_mode():
                torch.manual_seed(seed + 100000)
                for batch in validation_loader:
                    features = batch["features"].to(device)
                    prediction = model(features)["reconstructed_features"]
                    losses.append(float(torch.nn.functional.mse_loss(prediction, features)))
            validation_loss = float(np.mean(losses))
            if validation_loss < best_metric:
                best_metric = validation_loss
                torch.save({"model": _checkpoint_model_state(model), "config": config, "seed": seed,
                            "feature_file": config["features"]["train"]},
                           output / "checkpoint_best_recon.pt")
        else:
            best_metric = history[-1]["loss"] if history else float("inf")
            torch.save({"model": _checkpoint_model_state(model), "config": config, "seed": seed,
                        "feature_file": config["features"]["train"]},
                       output / "checkpoint_best_recon.pt")
        history[-1]["validation_reconstruction_mse"] = validation_loss
        (output / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
        progress = {"status": "running", "epoch": epoch + 1, "steps": step,
                    "train_reconstruction_mse": float(np.mean(epoch_losses)),
                    "validation_reconstruction_mse": validation_loss,
                    "best_validation_reconstruction_mse": best_metric,
                    "epoch_seconds": round(time.monotonic() - epoch_start, 2),
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                    "compute": {"device": str(device), "gpu_ids": gpu_ids}}
        write_json(output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        if max_steps and step >= max_steps:
            break
    model.eval()
    torch.save({"model": _checkpoint_model_state(model), "config": config, "seed": seed,
                "feature_file": config["features"]["train"]}, output / "checkpoint_last.pt")
    (output / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
    write_json(output / "run.json", {
        "seed": seed, "updates": step, "best_selection": "lowest final training reconstruction loss (CPU smoke only)"
        if validation_loader is None else "lowest val feature reconstruction MSE",
        "best_metric": best_metric,
        "final_train_reconstruction_mse": history[-1]["loss"] if history else None,
        "source_sha256": source_hashes(), "config": config,
        "compute": {"device": str(device), "gpu_ids": gpu_ids, "gpu_names": gpu_names},
        "annotation_files_opened": False, "training_object_labels_used": False,
        "checkpoint_last": str(output / "checkpoint_last.pt"),
        "elapsed_seconds": round(time.monotonic() - started, 2),
    })
    write_json(output / "progress.json", {**progress, "status": "complete"})
    return output


def prediction_from_slots(masks, exist_logits, background_slot=True):
    """Decode independently for each view; slot labels need not match across views."""
    active = exist_logits.sigmoid() >= .5
    if background_slot:
        active = active.clone()
        active[0] = False
    selected = torch.where(active)[0]
    if not selected.numel():
        return np.zeros(masks.shape[-2:], dtype=np.int16)
    winners = masks[selected].argmax(0)
    pred = (selected[winners] + 1).cpu().numpy().astype(np.int16)
    if background_slot:
        pred[masks[0].cpu().numpy() >= masks[selected].amax(0).cpu().numpy()] = 0
    return pred


@torch.inference_mode()
def evaluate_discovery(checkpoint_path, feature_path, annotation_path, grid_hw=None,
                       device="cpu", output_dir=None, background_slot=None):
    from PIL import Image
    from .metrics import instance_metrics, stability_iou, paired_bootstrap_difference

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]["model"]
    torch.set_num_threads(checkpoint["config"]["training"].get("cpu_threads", 2))
    if background_slot is None:
        background_slot = cfg.get("background_slot", True)
    model = DiscoveryModel(**cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = FeatureDataset(feature_path, grid_hw)
    annotations = np.load(annotation_path)
    ground_truth = annotations["instance_ids"]
    if len(dataset) != len(ground_truth):
        raise ValueError("Feature/annotation order or count mismatch")
    all_metrics, recon = [], []
    baseline_metrics = {name: [] for name in ("random_grouping", "one_slot", "regular_grid")}
    rng = np.random.default_rng(187)
    if output_dir:
        output_dir = writable(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
    for index in range(len(dataset)):
        feature = dataset[index]["features"][None].to(device)
        torch.manual_seed(10000 + index)
        result = model(feature)
        masks = result["soft_masks"][0]
        pred = prediction_from_slots(masks, result["exist_logits"][0], background_slot)
        gt = np.asarray(Image.fromarray(ground_truth[index]).resize(
            (pred.shape[1], pred.shape[0]), Image.Resampling.NEAREST))
        all_metrics.append(instance_metrics(gt, pred))
        all_metrics[-1]["present_slot_gates"] = int((result["exist_logits"][0].sigmoid() >= .5).sum())
        all_metrics[-1]["predicted_foreground_fraction"] = float((pred != 0).mean())
        all_metrics[-1]["gt_foreground_fraction"] = float((gt != 0).mean())
        recon.append(float(torch.nn.functional.mse_loss(result["reconstructed_features"], feature)))

        flipped = feature.flip(-1)
        torch.manual_seed(10000 + index)
        flipped_result = model(flipped)
        flipped_masks = flipped_result["soft_masks"][0].flip(-1)
        pred_flip = prediction_from_slots(flipped_masks, flipped_result["exist_logits"][0], background_slot)
        all_metrics[-1]["flip_stability_iou"] = stability_iou(pred, pred_flip)
        random_map = rng.integers(0, max(2, int(masks.shape[0]) - 1), size=gt.shape, dtype=np.uint8)
        grid_y = np.minimum(np.arange(gt.shape[0]) * 4 // gt.shape[0], 3)
        grid_x = np.minimum(np.arange(gt.shape[1]) * 4 // gt.shape[1], 3)
        grid = (grid_y[:, None] * 4 + grid_x[None, :] + 1).astype(np.uint8)
        for name, labels in {"random_grouping": random_map, "one_slot": np.ones_like(gt),
                             "regular_grid": grid}.items():
            baseline = instance_metrics(gt, labels)
            baseline_metrics[name].append(baseline)
            all_metrics[-1][name + "_foreground_ari"] = baseline["foreground_ari"]
        if output_dir:
            np.savez_compressed(output_dir / f"sample_{index:04d}.npz",
                                gt=gt, prediction=pred,
                                soft_masks=masks.cpu().numpy().astype(np.float16),
                                attention_masks=result["attention_masks"][0].cpu().numpy().astype(np.float16),
                                slot_features=result["slot_features"][0].cpu().numpy().astype(np.float16),
                                exist_probabilities=result["exist_logits"][0].sigmoid().cpu().numpy(),
                                geometry=result["estimated_geometry"][0].cpu().numpy(),
                                prediction_flip_back=pred_flip,
                                features=feature[0].cpu().numpy().astype(np.float16))
        if (index + 1) % 16 == 0:
            print(json.dumps({"event": "eval_progress", "samples": index + 1,
                              "total": len(dataset)}), flush=True)
    summary = {
        "samples": len(all_metrics),
        "mean_matched_mask_iou": float(np.mean([m["matched_mask_iou"] for m in all_metrics])),
        "mean_foreground_ari": float(np.mean([m["foreground_ari"] for m in all_metrics])),
        "mean_count_abs_error": float(np.mean([m["count_abs_error"] for m in all_metrics])),
        "mean_split_fraction": float(np.mean([m["split_fraction"] for m in all_metrics])),
        "mean_merge_fraction": float(np.mean([m["merge_fraction"] for m in all_metrics])),
        "mean_flip_stability_iou": float(np.mean([m["flip_stability_iou"] for m in all_metrics])),
        "mean_feature_reconstruction_mse": float(np.mean(recon)),
        "mean_random_ari": float(np.mean([m["random_grouping_foreground_ari"] for m in all_metrics])),
        "mean_one_slot_ari": float(np.mean([m["one_slot_foreground_ari"] for m in all_metrics])),
        "mean_regular_grid_ari": float(np.mean([m["regular_grid_foreground_ari"] for m in all_metrics])),
        "label_use": "evaluation only", "per_sample": all_metrics,
        "checkpoint": str(Path(checkpoint_path).resolve()), "checkpoint_sha256": sha256(checkpoint_path),
        "feature_file": str(Path(feature_path).resolve()), "feature_sha256": sha256(feature_path),
        "annotation_file": str(Path(annotation_path).resolve()), "annotation_sha256": sha256(annotation_path),
        "background_definition": "GT ID 0 excluded in FG-ARI; model slot 0 treated as background" if background_slot else "GT ID 0 excluded in FG-ARI; no model background slot",
        "mask_resolution": list(dataset[0]["features"].shape[-2:]),
        "existence_threshold": .5, "sampling_seed_base": 10000,
        "flip_scope": "horizontal flip of cached features; not RGB re-encoding",
        "split_merge_definition": "any nonzero overlap with more than one instance",
    }
    fields = ("foreground_ari", "matched_mask_iou", "count_abs_error", "split_fraction", "merge_fraction")
    summary["baselines"] = {
        name: {**{"mean_" + key: float(np.mean([m[key] for m in rows])) for key in fields},
               "per_sample": rows}
        for name, rows in baseline_metrics.items()}
    summary["paired_validation_comparisons"] = {
        name: {key: paired_bootstrap_difference([m[key] for m in all_metrics], [m[key] for m in rows])
               for key in ("foreground_ari", "matched_mask_iou")}
        for name, rows in baseline_metrics.items()}
    summary["scientific_gate"] = {
        "grouping_evidence_pass": all(
            comparison[key]["ci95_low"] > 0
            for comparison in summary["paired_validation_comparisons"].values()
            for key in ("foreground_ari", "matched_mask_iou")),
        "scope": "single-seed validation screen; separate density/stability diagnostics required before M2",
        "criterion": "positive paired image-bootstrap 95% lower bound in FG-ARI and matched IoU versus every predefined baseline",
    }
    if output_dir:
        (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
