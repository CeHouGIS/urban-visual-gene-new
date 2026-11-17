import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def rgb(tensor):
    return Image.fromarray((tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()*255).astype("uint8"))


def reconstruction_grid(path, originals, ae_outputs, predictions, title="Original | frozen AE | Gene"):
    count = min(len(originals), 8)
    out = Image.new("RGB", (256*3, 32+count*256), "white")
    draw = ImageDraw.Draw(out); draw.text((8, 8), title, fill="black")
    for i in range(count):
        for j, batch in enumerate([originals, ae_outputs, predictions]):
            out.paste(rgb(batch[i]), (j*256, 32+i*256))
    out.save(path)


def scene_grid(path, generator, specs):
    out = Image.new("RGB", (256*4, 280*2), "white"); d = ImageDraw.Draw(out)
    for i, spec in enumerate(specs[:8]):
        im, _ = generator.sample(spec)
        x = i%4*256; y = i//4*280
        out.paste(im, (x, y))
        label = (f"shape={spec['building_shape']} facade={spec['facade']} "
                 f"veg={spec['vegetation_layout']} trees={spec['tree_count']}")
        d.text((x+3, y+258), label, fill="black")
    out.save(path)


def usage_plot(path, counts):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(np.arange(len(counts)), counts); ax.set(xlabel="Gene ID", ylabel="Enabled instances")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def instances_plot(path, images, program, cfg):
    m = cfg["model"]; out = Image.new("RGB", (256*4, 280), "white")
    d = ImageDraw.Draw(out)
    for i in range(min(4, len(images))):
        out.paste(rgb(images[i]), (i*256, 0))
        if "position_ids" not in program:
            continue
        for j in range(program["gene_ids"].shape[1]):
            if not int(program["enabled"][i, j]):
                continue
            pos = int(program["position_ids"][i, j]); grid = m["position_grid"]
            x = pos%grid/(grid-1)*256+i*256; y = pos//grid/(grid-1)*256
            r = m["gene_patch_hw"]*8*m["scale_levels"][int(program["scale_ids"][i, j])]/2
            gid = int(program["gene_ids"][i, j]); color = (255, 40+(gid*37)%190, 40)
            d.rectangle((x-r, y-r, x+r, y+r), outline=color, width=1)
            d.text((x, y), str(gid), fill=color)
    d.text((5, 260), "Latent footprint / Gene ID; patches are not object masks", fill="black")
    out.save(path)


def curves(path, records):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot([r["epoch"] for r in records], [r["train_l1"] for r in records], label="train")
    ax.plot([r["epoch"] for r in records], [r["val_l1"] for r in records], label="val_seen")
    ax.set(xlabel="Epoch", ylabel="L1"); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def aggregate_plots(root):
    root = Path(root)
    metrics = []
    for path in root.glob("*/eval/*/metrics.json"):
        r = json.loads(path.read_text())
        if r.get("kind") == "gene":
            metrics.append(r)
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in metrics:
        if r["split"] == "val_seen":
            ax.scatter(r["budget"]["bits"], r["psnr"])
            ax.annotate(r["run"], (r["budget"]["bits"], r["psnr"]), fontsize=6)
    ax.set(xlabel="Fixed program bits / scene (padding excluded)", ylabel="PSNR (dB)")
    fig.tight_layout(); fig.savefig(root / "rate_distortion.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for split in ["test_seen", "test_unseen_combo", "test_higher_complexity"]:
        rs = [r for r in metrics if r["split"] == split]
        if rs:
            ax.plot([r["run"] for r in rs], [r["latent_l1"] for r in rs], "o", label=split)
    ax.set(ylabel="Latent L1", xlabel="Run"); ax.tick_params(axis="x", labelrotation=30)
    if ax.lines:
        ax.legend()
    fig.tight_layout(); fig.savefig(root / "seen_unseen_comparison.png", dpi=130); plt.close(fig)
