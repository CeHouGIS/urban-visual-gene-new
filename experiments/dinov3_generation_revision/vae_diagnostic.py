"""Compare the old RGB probe with an image-only pretrained VAE control.

This control is NOT DINOv3-conditioned generation or object discovery. The
VAE sees the original RGB image; the old probe sees compressed DINOv3 slots.
Their difference cannot isolate the decoder from the representation bottleneck.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from diffusers import AutoencoderKL


ROOT = Path(__file__).resolve().parent
PILOT = ROOT.parent / "object_centric_pilot"
sys.path.insert(0, str(PILOT))
from model import SlotModel, upsample_rgb  # noqa: E402
from evaluate_pilot import sheet  # noqa: E402


def to_image(tensor):
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).round().astype(np.uint8))


def metric(predicted, target):
    mse = float(F.mse_loss(predicted.float(), target.float()))
    return {"mse": mse, "psnr_db": -10 * math.log10(max(mse, 1e-12))}


@torch.inference_mode()
def run(args):
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is None:
            raise ValueError("Use an explicit GPU index on this shared host")
        torch.cuda.set_per_process_memory_fraction(.03, device.index)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PILOT / "data/dinov3/manifest.json").read_text())
    archive = np.load(PILOT / "data/dinov3/dataset.npz")
    chosen = []
    # Same first held-out sample in each city as the previous pilot figures.
    for city in ("Austin", "Bogota", "Lagos", "London", "Adelaide"):
        chosen.append(next(i for i, record in enumerate(manifest["records"])
                           if record["city"] == city and record["split"] != "train"))
    checkpoint = torch.load(PILOT / "results/k32-refined/model.pt", map_location="cpu", weights_only=True)
    model = SlotModel(slots=checkpoint["slots"], rgb_mode=checkpoint["rgb_mode"],
                      position_strength=checkpoint.get("position_strength", 1)).eval().to(device)
    model.load_state_dict(checkpoint["model"])
    vae = AutoencoderKL.from_pretrained(args.vae, local_files_only=True,
                                        use_safetensors=True, low_cpu_mem_usage=False).eval().to(device)
    if args.tiling:
        vae.enable_tiling()
    rows, labels, diagnostics = [], [], []
    for index in chosen:
        started = time.monotonic()
        record = manifest["records"][index]
        # Use original sources rather than another JPEG compression of the pilot copy.
        image = Image.open(record["source_image"]).convert("RGB").resize((448, 448), Image.Resampling.BILINEAR)
        target = torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)[None].to(device)
        encoded = vae.encode(target * 2 - 1).latent_dist.mode()
        reconstructed = ((vae.decode(encoded).sample + 1) / 2).clamp(0, 1)
        features = torch.from_numpy(archive["features"][index:index + 1].astype(np.float32)).to(device)
        grouped = model(features)
        old_rgb, _ = model.render(grouped["slots"], grouped["masks"])
        old_rgb = upsample_rgb(old_rgb)
        old_448 = F.interpolate(old_rgb, (448, 448), mode="bilinear", align_corners=False)
        # Exact same 112x112 target as the old pilot's evaluation.
        target_112 = torch.from_numpy(archive["rgb"][index:index + 1].astype(np.float32)).permute(0, 3, 1, 2).to(device)
        recon_112 = torch.from_numpy(np.array(to_image(reconstructed[0]).resize(
            (112, 112), Image.Resampling.BILINEAR), dtype=np.float32) / 255).permute(2, 0, 1)[None].to(device)
        assert torch.isfinite(reconstructed).all() and torch.isfinite(old_rgb).all()
        row = {"id": record["id"], "city": record["city"], "split": record["split"],
               "old_probe_at_112": metric(old_rgb, target_112),
               "old_probe_upsampled_at_448": metric(old_448, target),
               "vae_at_448": metric(reconstructed, target),
               "vae_downsampled_at_112": metric(recon_112, target_112),
               "seconds": time.monotonic() - started}
        diagnostics.append(row)
        print(json.dumps(row), flush=True)
        to_image(reconstructed[0]).save(args.output / f"{record['city']}_vae.png")
        rows.append([image, to_image(old_448[0]), to_image(reconstructed[0])])
        labels.append(f"{record['city']} | original RGB input only for VAE control")
    sheet(rows, ["Original 448px", "Old DINOv3 probe: 112px", "Pretrained VAE control: 448px"],
          labels, args.output / "comparison.png", panel=320)
    sheet([rows[i] for i in (0, 3)],
          ["Original 448px", "Old DINOv3 probe: 112px", "Pretrained VAE control: 448px"],
          [labels[i] for i in (0, 3)], args.output / "preview.png", panel=320)
    report = {
        "purpose": "Image-only reconstruction control; no training or instance labels used in this diagnostic.",
        "limitation": "VAE sees original RGB; old probe sees compressed semantic features. This is not a decoder-only ablation and does not verify DINOv3-conditioned generation.",
        "selection": "First held-out sample per city, predetermined in the previous pilot; five examples, not full validation statistics.",
        "vae": str(args.vae.resolve()), "posterior": "mode; no latent scaling between encoder and decoder",
        "resolution": 448, "device": str(device), "tiling": args.tiling,
        "per_image": diagnostics,
        "mean_mse_psnr": {},
    }
    for key in ("old_probe_at_112", "old_probe_upsampled_at_448", "vae_at_448", "vae_downsampled_at_112"):
        mse = float(np.mean([r[key]["mse"] for r in diagnostics]))
        report["mean_mse_psnr"][key] = {"mse": mse, "psnr_db": -10 * math.log10(mse)}
    if device.type == "cuda":
        report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    report["environment"] = {"torch": torch.__version__, "numpy": np.__version__}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["mean_mse_psnr"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae", type=Path, default=ROOT / "models/sd-vae-ft-mse")
    parser.add_argument("--output", type=Path, default=ROOT / "results/vae_control")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tiling", action="store_true")
    run(parser.parse_args())
