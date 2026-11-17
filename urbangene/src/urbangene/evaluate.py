import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .budget import pack_program, unpack_program
from .metrics import image_metrics
from .models.urbangene import UrbanGene
from .train import load_ae, make_loader
from .utils import save_json, setup
from .visualize import aggregate_plots, instances_plot, reconstruction_grid, usage_plot


@torch.no_grad()
def evaluate(checkpoint, split, device="cpu", max_samples=None, probe_path=None):
    checkpoint = Path(checkpoint)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]; setup(cfg["seed"], cfg["training"]["cpu_threads"])
    is_gene = ck["kind"] == "gene"
    ae, _ = load_ae(ck["ae_path"] if is_gene else checkpoint, device)
    model = None
    if is_gene:
        if ck.get("ae_sha256"):
            assert hashlib.sha256(Path(ck["ae_path"]).read_bytes()).hexdigest() == ck["ae_sha256"]
        model = UrbanGene(cfg, ae, ck["method"]).to(device).eval()
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        assert not unexpected and all(k.startswith("image_ae.") for k in missing)
    loader = make_loader(cfg, split)
    output = checkpoint.parent / "eval" / split; output.mkdir(parents=True, exist_ok=True)
    rows = []; usages = torch.zeros(cfg["model"]["gene_count"], dtype=torch.long)
    enabled_counts = []; first = None; worst = []; wire = bytearray(); programs_json = []
    probe = None
    if probe_path:
        from .probe import load_probe
        probe, names = load_probe(probe_path, device)
    for batch in loader:
        if max_samples is not None and len(rows) >= max_samples:
            break
        limit = len(batch["image"]) if max_samples is None else min(len(batch["image"]), max_samples-len(rows))
        x = batch["image"][:limit].to(device); indices = batch["index"][:limit].tolist()
        target = ae.encoder(x); ae_y = ae.decoder(target)
        if model:
            result = model.forward_latent(target, decode=True)
            y = result["image_hat"]; zhat = result["latent_hat"]; program = result["program"]
            # Across-batch-roll controls content dependence; batch=1 falls back to zero.
            shuffled = model.reconstruct_program({k: v.roll(1, 0) for k, v in program.items()}, decode=False)["latent_hat"]
            if limit == 1:
                shuffled = torch.zeros_like(zhat)
            latent_l1 = (target-zhat).abs().mean((1, 2, 3))
            shuffled_l1 = (target-shuffled).abs().mean((1, 2, 3))
            for i in range(limit):
                packed = pack_program(program, cfg, ck["method"], i)
                recovered = {k: v.to(device) for k, v in unpack_program(packed, cfg, ck["method"]).items()}
                from_wire = model.reconstruct_program(recovered, decode=False)["latent_hat"]
                assert torch.allclose(from_wire, zhat[i:i+1], atol=2e-6, rtol=1e-5), "Wire program differs from rendered code"
                wire.extend(packed)
                enabled = program.get("enabled", torch.ones_like(program["gene_ids"]))[i].bool()
                usages += torch.bincount(program["gene_ids"][i][enabled].cpu(), minlength=len(usages))
                enabled_counts.append(int(enabled.sum()))
                if len(programs_json) < 32:
                    programs_json.append(dict(index=indices[i], **{k: v[i].tolist() for k, v in program.items()}))
        else:
            y = ae_y; latent_l1 = torch.zeros(limit); shuffled_l1 = torch.zeros(limit)
        measured = image_metrics(x, y)
        if probe:
            pred_orig = probe(x); pred_recon = probe(y)
        for i in range(limit):
            row = dict(index=indices[i], latent_l1=float(latent_l1[i]),
                       shuffled_latent_l1=float(shuffled_l1[i]),
                       zero_latent_l1=float(target[i].abs().mean()),
                       **{k: float(v[i]) for k, v in measured.items()})
            spec = loader.dataset.specs[indices[i]]
            row["tree_count"] = spec["tree_count"]
            if probe:
                from .probe import labels_for_spec
                labels = labels_for_spec(spec)
                for j, name in enumerate(names):
                    row[f"probe_original_{name}"] = int(pred_orig[j][i].argmax()) == labels[j]
                    row[f"probe_reconstructed_{name}"] = int(pred_recon[j][i].argmax()) == labels[j]
            rows.append(row)
            worst.append((row["rgb_l1"], x[i].cpu(), ae_y[i].cpu(), y[i].cpu()))
        worst = sorted(worst, key=lambda r: r[0], reverse=True)[:8]
        if first is None:
            first = (x.cpu(), ae_y.cpu(), y.cpu())
            if model:
                instances_plot(output / "instances.png", x, program, cfg)
    metric_keys = [k for k in rows[0] if k not in ("index", "tree_count")]
    metrics = {key: float(np.mean([row[key] for row in rows])) for key in metric_keys}
    if probe:
        metrics["probe_reliability"] = {name: metrics[f"probe_original_{name}"] >= 0.85 for name in names}
    metrics.update(kind=ck["kind"], run=checkpoint.parent.name, split=split, samples=len(rows),
                   checkpoint=str(checkpoint.resolve()),
                   checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   task="input-conditioned reconstruction; NOT target-free generation",
                   probe_used=bool(probe), ssim_definition="7x7 uniform window, valid border, RGB mean",
                   uncertainty="single training seed; no confidence intervals yet",
                   by_tree_count={str(n): {key: float(np.mean([r[key] for r in rows if r["tree_count"] == n]))
                                         for key in ["latent_l1", "psnr"]}
                                  for n in sorted({r["tree_count"] for r in rows})})
    if model:
        metrics.update(method=ck["method"], budget=model.budget,
                       enabled_instances_mean=float(np.mean(enabled_counts)),
                       genes_used=int((usages > 0).sum()),
                       shared_gene_parameters=model.library.templates.numel(),
                       shared_gene_fp32_bits=model.library.templates.numel()*32,
                       trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                       frozen_ae_parameters=sum(p.numel() for p in ae.parameters()),
                       measured_program_file_bytes=len(wire),
                       byte_padded_bits_per_scene=len(wire)*8/len(rows))
        (output / "programs.bin").write_bytes(wire)
        save_json(output / "program_examples.json", programs_json)
        usage_plot(output / "gene_usage.png", usages.tolist())
        # Isolated templates on a zero latent background, not named objects.
        from PIL import Image, ImageDraw
        from .visualize import rgb
        count, channels, patch, _ = model.library.templates.shape
        thumbnails = Image.new("RGB", (8*96, ((count+7)//8)*112), "white")
        draw = ImageDraw.Draw(thumbnails)
        for start in range(0, count, 16):
            canvas = torch.zeros(min(16, count-start), channels, 32, 32, device=device)
            offset = (32-patch)//2
            canvas[:, :, offset:offset+patch, offset:offset+patch] = model.library.templates[start:start+16]
            pictures = ae.decoder(canvas)
            for j, picture in enumerate(pictures):
                idx = start+j; x = idx%8*96; y0 = idx//8*112
                thumbnails.paste(rgb(picture).resize((96, 96)), (x, y0))
                draw.text((x+2, y0+96), f"Gene {idx}", fill="black")
        thumbnails.save(output / "gene_templates.png")
    save_json(output / "metrics.json", metrics)
    save_json(output / "per_image.json", rows)
    with (output / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    reconstruction_grid(output / "recon_grid.png", *first)
    reconstruction_grid(output / "failure_cases.png", *[torch.stack([r[j] for r in worst]) for j in range(1, 4)],
                        title="Highest RGB L1 | original / AE / reconstruction")
    aggregate_plots(checkpoint.parent.parent)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return metrics
