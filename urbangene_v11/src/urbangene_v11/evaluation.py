import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from urbangene.data.dataset import SceneDataset
from urbangene.metrics import image_metrics
from urbangene.utils import save_json
from urbangene.visualize import instances_plot, reconstruction_grid, usage_plot

from .paired_composition import make_triplets
from .recombination import predict_recombination
from .training import load_model


def _means(rows, keys):
    return {key: sum(row[key] for row in rows)/len(rows) if rows else None for key in keys}


@torch.no_grad()
def evaluate_reconstruction(checkpoint, split, output, device="cpu", max_samples=None):
    model, ck = load_model(checkpoint, device)
    cfg = ck["config"]
    ds = SceneDataset(cfg["data"]["root"], split)
    if max_samples is not None:
        ds.specs = ds.specs[:max_samples]
    loader = DataLoader(ds, batch_size=cfg["evaluation"]["evaluation_batch_size"], shuffle=False,
                        num_workers=cfg["data"].get("workers", 0))
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows = []; gene_counts = torch.zeros(cfg["model"]["gene_count"], dtype=torch.long)
    image_rows = []; first = None
    for batch in loader:
        x = batch["image"].to(device)
        z = model.ae.encoder(x)
        result = model.forward_latent(z, decode=True)
        reconstruction = result["image_hat"]
        ae_image = model.ae.decoder(z)
        base_m = image_metrics(x, ae_image); gene_m = image_metrics(x, reconstruction)
        latent_l1 = (z-result["latent_hat"]).abs().mean((1, 2, 3))
        program = result["program"]
        active = program["enabled"].bool()
        gene_counts += torch.bincount(program["gene_ids"][active].cpu(), minlength=len(gene_counts))
        for i in range(len(x)):
            rows.append(dict(rgb_l1=float(gene_m["rgb_l1"][i]), psnr=float(gene_m["psnr"][i]),
                             ssim=float(gene_m["ssim_7x7_uniform"][i]),
                             ae_rgb_l1=float(base_m["rgb_l1"][i]), ae_psnr=float(base_m["psnr"][i]),
                             latent_l1=float(latent_l1[i]), active_instances=int(active[i].sum()),
                             failed=bool(gene_m["rgb_l1"][i] > cfg["evaluation"]["failure_rgb_l1"])))
        if first is None:
            first = (x[:8].cpu(), ae_image[:8].cpu(), reconstruction[:8].cpu(),
                     {k: v[:8].cpu() for k, v in program.items()})
    metrics = dict(kind="reconstruction", split=split, checkpoint=str(Path(checkpoint).resolve()),
                   samples=len(rows), budget_bits=cfg["model"]["budget_bits"],
                   gene_usage=gene_counts.tolist(), unique_genes=int((gene_counts > 0).sum()),
                   active_instances_mean=sum(r["active_instances"] for r in rows)/len(rows),
                   failed_samples=sum(r["failed"] for r in rows), **_means(rows,
                   ["rgb_l1", "psnr", "ssim", "ae_rgb_l1", "ae_psnr", "latent_l1"]))
    save_json(output / "metrics.json", metrics)
    with (output / "samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader(); writer.writerows(rows)
    if first is not None:
        x, ae_image, prediction, program = first
        reconstruction_grid(output / "reconstruction.png", x, ae_image, prediction)
        instances_plot(output / "instances.png", x, program, cfg)
    usage_plot(output / "gene_usage.png", gene_counts.numpy())
    return metrics


@torch.no_grad()
def evaluate_recombination(checkpoint, output, device="cpu", count=None, seed=42):
    model, ck = load_model(checkpoint, device); cfg = ck["config"]
    count = count or cfg["evaluation"]["recombination_samples"]
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    all_rows = []; panels = []
    for split in ("seen", "heldout"):
        split_dir = output / split; split_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, triplet in enumerate(make_triplets(count, split, seed)):
            a, b = triplet.sources()
            prediction = predict_recombination(model, a[None].to(device), b[None].to(device),
                                               triplet.region, cfg["evaluation"]["overlap_threshold"])
            # C is intentionally constructed only after target-free inference returns.
            target = triplet.target_for_scoring().to(device)[None]
            row = dict(split=split, index=index, overflow=bool(prediction["merge"].overflow),
                       requested_instances=prediction["merge"].requested_instances,
                       valid=prediction["prediction"] is not None)
            if prediction["prediction"] is not None:
                pred = prediction["prediction"].clamp(0, 1)
                full = image_metrics(target, pred)
                x0, y0, x1, y1 = triplet.region
                region_m = image_metrics(target[:, :, y0:y1, x0:x1], pred[:, :, y0:y1, x0:x1])
                outside = torch.ones_like(target); outside[:, :, y0:y1, x0:x1] = 0
                outside_l1 = ((target-pred).abs()*outside).sum()/outside.sum().clamp_min(1)
                reference_a = a.to(device)[None]
                preserve_l1 = ((pred-reference_a).abs()*outside).sum()/outside.sum().clamp_min(1)
                preserve_l1 = preserve_l1/3
                target_z = model.ae.encoder(target)
                target_l1 = (target_z-prediction["latent_hat"]).abs().mean()
                row.update(rgb_l1=float(full["rgb_l1"][0]), psnr=float(full["psnr"][0]),
                           region_rgb_l1=float(region_m["rgb_l1"][0]),
                           region_psnr=float(region_m["psnr"][0]),
                           outside_target_rgb_l1=float(outside_l1),
                           outside_source_a_change_l1=float(preserve_l1), latent_l1=float(target_l1))
                if index < 8:
                    panels.append((split, index, a, b, target[0].cpu(), pred[0].cpu()))
            else:
                row.update(rgb_l1=None, psnr=None, region_rgb_l1=None, region_psnr=None,
                           outside_target_rgb_l1=None, outside_source_a_change_l1=None, latent_l1=None)
            rows.append(row); all_rows.append(row)
        summary = dict(kind="recombination", split=split, attempted=len(rows),
                       valid=sum(r["valid"] for r in rows), overflow=sum(r["overflow"] for r in rows),
                       valid_rate=sum(r["valid"] for r in rows)/len(rows),
                       **_means([r for r in rows if r["valid"]],
                           ["rgb_l1", "psnr", "region_rgb_l1", "region_psnr",
                            "outside_target_rgb_l1", "outside_source_a_change_l1", "latent_l1"]))
        save_json(split_dir / "metrics.json", summary)
        with (split_dir / "samples.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save_json(output / "summary.json", dict(kind="recombination", checkpoint=str(Path(checkpoint).resolve()),
               samples_per_split=count, split_summaries={s: json.loads((output/s/"metrics.json").read_text())
                                                        for s in ("seen", "heldout")}))
    from PIL import Image, ImageDraw
    from urbangene.visualize import rgb
    sheet = Image.new("RGB", (256*4, 286*min(len(panels), 8)), "white"); draw = ImageDraw.Draw(sheet)
    for i, (split, index, a, b, target, pred) in enumerate(panels[:8]):
        y = i*286
        for col, image in enumerate((a, b, target, pred)):
            sheet.paste(rgb(image), (col*256, y+20))
        draw.text((4, y+3), f"{split} #{index}: A | B | C* target | prediction", fill="black")
    if panels:
        sheet.save(output / "recombination_examples.png")
    return all_rows
