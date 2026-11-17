"""Held-out controls, visual audits and explicit limits of the pilot."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
import torch

from model import SlotModel, upsample_rgb
from run import load_data


PALETTE = np.array([
    [68, 119, 170], [238, 102, 119], [34, 136, 51], [204, 187, 68],
    [102, 204, 238], [170, 51, 119], [187, 187, 187], [238, 153, 68],
    [119, 170, 221], [153, 153, 51], [221, 170, 51], [85, 102, 153],
    [153, 68, 85], [68, 170, 153], [170, 153, 204], [204, 102, 51],
    [60, 150, 205], [215, 75, 150], [115, 190, 70], [220, 165, 100],
    [95, 105, 195], [175, 190, 100], [160, 120, 65], [55, 135, 125],
], dtype=np.uint8)


def mask_overlay(image, masks):
    hard = masks.argmax(0).reshape(28, 28)
    colors = Image.fromarray(PALETTE[hard % len(PALETTE)]).resize(image.size, Image.Resampling.NEAREST)
    return Image.blend(image, colors, .55)


def as_image(rgb):
    array = rgb.detach().cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray((array.clip(0, 1) * 255).astype(np.uint8))


def sheet(rows, headers, labels, path, panel=260):
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        font, small = ImageFont.truetype(font_path, 16), ImageFont.truetype(font_path, 13)
    except OSError:
        font = small = ImageFont.load_default()
    gap, row_h, top = 12, panel + 42, 52
    canvas = Image.new("RGB", (len(headers) * (panel + gap) + gap,
                               len(rows) * row_h + top), "white")
    draw = ImageDraw.Draw(canvas)
    for c, header in enumerate(headers):
        draw.text((gap + c * (panel + gap), 15), header, fill="#222222", font=font)
    for r, images in enumerate(rows):
        for c, image in enumerate(images):
            canvas.paste(image.resize((panel, panel), Image.Resampling.BILINEAR),
                         (gap + c * (panel + gap), top + r * row_h))
        draw.text((gap, top + r * row_h + panel + 8), labels[r], fill="#333333", font=small)
    canvas.save(path)


def matched_iou(first, second, slots):
    # Ignore slots without any hard-assigned patches. Weight by reference area.
    a, b = first.argmax(0).numpy(), second.argmax(0).numpy()
    matrix = np.zeros((slots, slots))
    area = np.bincount(a, minlength=slots)
    for i in range(slots):
        for j in range(slots):
            union = ((a == i) | (b == j)).sum()
            matrix[i, j] = ((a == i) & (b == j)).sum() / max(union, 1)
    left, right = linear_sum_assignment(-matrix)
    return float((matrix[left, right] * area[left]).sum() / len(a))


@torch.no_grad()
def inspect(args):
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(.04, device.index)
    checkpoint = torch.load(args.output / "model.pt", map_location=device, weights_only=True)
    rgb_mode = checkpoint.get("rgb_mode", "direct")
    model = SlotModel(slots=checkpoint["slots"], rgb_mode=rgb_mode,
                      position_strength=checkpoint.get("position_strength", 1.)).to(device).eval()
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    assert not incompatible.unexpected_keys
    assert all(name.startswith("rgb_conv.") for name in incompatible.missing_keys)
    features, rgb, manifest, splits = load_data(args.data)
    indices = torch.cat((splits["validation"], splits["cross_city"]))
    outputs = []
    for batch in indices.split(4):
        current = model(features[batch].to(device))
        prediction, _ = model.render(current["slots"], current["masks"])
        current["rgb"] = upsample_rgb(prediction)
        outputs.append({k: v.cpu() for k, v in current.items()})
    out = {k: torch.cat([o[k] for o in outputs]) for k in outputs[0]}
    metrics = json.loads((args.output / "metrics.json").read_text())
    diagnostics = []
    for offset, index in enumerate(indices):
        record = manifest["records"][int(index)]
        masks = out["masks"][offset]
        hard = masks.argmax(0).reshape(28, 28).numpy()
        areas = np.bincount(hard.reshape(-1), minlength=model.slot_count) / 784
        occupied = np.flatnonzero(areas >= .01)
        components = []
        for slot in occupied:
            parts, count = label(hard == slot, np.ones((3, 3)))
            sizes = np.bincount(parts.reshape(-1))[1:]
            components.append(int((sizes >= 4).sum()))
        diagnostics.append({"id": record["id"], "split": record["split"],
                            "occupied_slots_1pct": len(occupied),
                            "largest_slot_fraction": float(areas.max()),
                            "decoder_mean_confidence": float(masks.max(0).values.mean()),
                            "disconnected_slots_4patch_min": int(sum(c > 1 for c in components)),
                            "feature_mse": float(((out["features"][offset] - features[index]) ** 2).mean())})
    # Different images' slots are substituted: tests reliance on the latent code.
    shuffled_features, shuffled_rgb = [], []
    for batch in out["slots"].roll(1, 0).split(4):
        reconstructed, masks = model.decode(batch.to(device))
        predicted, _ = model.render(batch.to(device), masks)
        shuffled_features.append(reconstructed.cpu())
        shuffled_rgb.append(upsample_rgb(predicted).cpu())
    shuffled_features, shuffled_rgb = torch.cat(shuffled_features), torch.cat(shuffled_rgb)
    metrics["diagnostics"] = {
        "heldout_views": len(indices),
        "mean_occupied_slots_1pct": float(np.mean([r["occupied_slots_1pct"] for r in diagnostics])),
        "mean_largest_slot_fraction": float(np.mean([r["largest_slot_fraction"] for r in diagnostics])),
        "fraction_views_with_disconnected_slot": float(np.mean([r["disconnected_slots_4patch_min"] > 0 for r in diagnostics])),
        "shuffled_image_slots_feature_mse": float(((shuffled_features.cpu() - features[indices]) ** 2).mean()),
        "shuffled_image_slots_rgb_mse": float(((shuffled_rgb - rgb[indices]) ** 2).mean()),
        "note": "Occupied slots are not object counts; topology and confidence are diagnostic only."}
    consistency_indices = indices[:16]
    original_masks = out["masks"][:16]
    flip_masks = []
    for batch in consistency_indices.split(4):
        x = features[batch].reshape(-1, 28, 28, 64).flip(2).reshape(-1, 784, 64).to(device)
        masks = model(x)["masks"].reshape(-1, model.slot_count, 28, 28).flip(3)
        flip_masks.append(masks.reshape(-1, model.slot_count, 784).cpu())
    flip_masks = torch.cat(flip_masks)
    flip_iou = [matched_iou(a, b, model.slot_count) for a, b in zip(original_masks, flip_masks)]
    old_noise = model.group.eval_noise.clone()
    generator = torch.Generator().manual_seed(71)
    model.group.eval_noise.copy_(torch.randn(old_noise.shape, generator=generator).to(device))
    seed_masks = []
    for batch in consistency_indices.split(4):
        seed_masks.append(model(features[batch].to(device))["masks"].cpu())
    model.group.eval_noise.copy_(old_noise)
    seed_iou = [matched_iou(a, b, model.slot_count)
                for a, b in zip(original_masks, torch.cat(seed_masks))]
    metrics["diagnostics"].update(
        flip_matched_area_weighted_iou=float(np.mean(flip_iou)),
        initialization_matched_area_weighted_iou=float(np.mean(seed_iou)),
        consistency_sample_count=16,
        flip_test_note="Cached patch maps are mirrored; DINOv3 is not rerun on flipped RGB.")
    # One predetermined validation image per city, not best-performing examples.
    selected, seen = [], set()
    for offset, index in enumerate(indices):
        city = manifest["records"][int(index)]["city"]
        if city not in seen:
            selected.append(offset)
            seen.add(city)
    rows, names = [], []
    for offset in selected:
        record = manifest["records"][int(indices[offset])]
        image = Image.open(args.data / record["image"]).convert("RGB")
        rows.append([image, mask_overlay(image, out["attention"][offset].numpy()),
                     mask_overlay(image, out["masks"][offset].numpy()), as_image(out["rgb"][offset])])
        names.append(f"{record['city']} | {record['split']} | first predetermined example")
    rgb_title = "RGB renderer (112 x 112)" if rgb_mode == "features" else "RGB probe (28 x 28)"
    sheet(rows, ["Input street view", "Encoder slot groups", "Decoder slot groups", rgb_title],
          names, args.output / "overview.png")
    for start in range(0, len(indices), 16):
        gallery = []
        labels = []
        for offset in range(start, min(start + 16, len(indices))):
            record = manifest["records"][int(indices[offset])]
            image = Image.open(args.data / record["image"]).convert("RGB")
            gallery.append([image, mask_overlay(image, out["attention"][offset].numpy()),
                            mask_overlay(image, out["masks"][offset].numpy()), as_image(out["rgb"][offset])])
            labels.append(f"{record['city']} | #{offset} | {record['pano']}")
        sheet(gallery, ["Input", "Encoder groups", "Decoder groups", "RGB probe"], labels,
              args.output / f"gallery_{start // 16:02d}.png", panel=200)
    # Slot replacement: choose a small occupied region, without semantic labels.
    rows, names, edits = [], [], []
    for target_offset, donor_offset in zip(selected[:4], selected[1:] + selected[:1]):
        masks = out["masks"][target_offset:target_offset + 1].to(device)
        slots = out["slots"][target_offset:target_offset + 1].to(device)
        donor_slots = out["slots"][donor_offset].to(device)
        area = masks[0].argmax(0).bincount(minlength=model.slot_count).float() / 784
        candidates = torch.where((area > .03) & (area < .2))[0]
        target_slot = int(candidates[0]) if len(candidates) else int(area.argmax())
        donor_masks = out["masks"][donor_offset]
        donor_area = donor_masks.argmax(0).bincount(minlength=model.slot_count).float() / 784
        occupied_donor = torch.where(donor_area > .01)[0].to(device)
        distances = ((donor_slots[occupied_donor] - slots[0, target_slot]) ** 2).mean(-1)
        donor_slot = int(occupied_donor[distances.argmax()])
        modified = slots.clone()
        modified[0, target_slot] = donor_slots[donor_slot]
        _, new_masks = model.decode(modified)
        free_rgb, _ = model.render(modified, new_masks)
        bound_rgb, _ = model.render(modified, masks)
        target_record = manifest["records"][int(indices[target_offset])]
        donor_record = manifest["records"][int(indices[donor_offset])]
        image = Image.open(args.data / target_record["image"]).convert("RGB")
        hard = masks[0].argmax(0).cpu().numpy().reshape(28, 28)
        highlight = np.asarray(image.resize((28, 28))).copy()
        highlight[hard != target_slot] = (highlight[hard != target_slot] * .2).astype(np.uint8)
        base = out["rgb"][target_offset]
        changed = upsample_rgb(free_rgb)[0].cpu()
        bound = upsample_rgb(bound_rgb)[0].cpu()
        region = torch.from_numpy((hard == target_slot).astype(np.float32))[None, None]
        region = torch.nn.functional.interpolate(region, (112, 112), mode="nearest")[0, 0] > .5
        free_delta = (changed - base).abs().mean(0)
        bound_delta = (bound - base).abs().mean(0)
        edits.append({"target": target_record["id"], "donor": donor_record["id"],
                      "target_slot": target_slot, "donor_slot": donor_slot,
                      "free_swap_inside_mae": float(free_delta[region].mean()),
                      "free_swap_outside_mae": float(free_delta[~region].mean()),
                      "fixed_mask_swap_inside_mae": float(bound_delta[region].mean()),
                      "fixed_mask_swap_outside_mae": float(bound_delta[~region].mean())})
        rows.append([image, Image.fromarray(highlight), as_image(base), as_image(changed), as_image(bound)])
        names.append(f"{target_record['city']} <- {donor_record['city']} | target slot {target_slot}; donor slot {donor_slot}")
    sheet(rows, ["Target image", "Selected region", "Original RGB probe", "Swap + new masks", "Swap + fixed masks"],
          names, args.output / "interventions.png", panel=220)
    metrics["interventions"] = edits
    metrics["intervention_note"] = "Regions have no assigned object category. Fixed-mask localization is imposed by architecture, not evidence of learned shape/appearance disentanglement. The donor is the most distant occupied slot for a detectable stress test, not a same-class semantic edit."
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output / "per_image.json").write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(metrics["diagnostics"], indent=2), flush=True)
    print("wrote overview, all held-out galleries, interventions and diagnostics", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data/dinov3")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/k16")
    parser.add_argument("--device", default="cuda:0")
    inspect(parser.parse_args())
