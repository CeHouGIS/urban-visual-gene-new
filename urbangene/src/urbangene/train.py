import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data.dataset import SceneDataset
from .models.image_autoencoder import ImageAutoencoder
from .models.urbangene import UrbanGene
from .utils import digest, save_checkpoint, save_json, setup, snapshot
from .visualize import curves, reconstruction_grid


def make_loader(cfg, split, shuffle=False):
    ds = SceneDataset(cfg["data"]["root"], split)
    return DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=shuffle,
                      num_workers=cfg["data"].get("workers", 0), pin_memory=False)


def new_ae(cfg):
    return ImageAutoencoder(cfg["model"]["latent_channels"], cfg["ae"]["width"])


def load_ae(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ae = new_ae(ckpt["config"])
    ae.load_state_dict(ckpt["model"])
    return ae.to(device).freeze(), ckpt


@torch.no_grad()
def ae_validation(ae, loader, device):
    ae.eval(); total = 0.; count = 0
    for batch in loader:
        x = batch["image"].to(device)
        total += F.l1_loss(ae(x), x, reduction="sum").item()
        count += x.numel()
    return total/count


def train_ae(cfg, run_dir, device="cpu", resume=None):
    setup(cfg["seed"], cfg["training"]["cpu_threads"])
    run = Path(run_dir); snapshot(run, cfg)
    ae = new_ae(cfg).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=cfg["ae"]["learning_rate"])
    loader = make_loader(cfg, "train", True); val = make_loader(cfg, "val_seen")
    history = []; start = 0; best = float("inf")
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        assert ck["config"] == cfg, "Resume must use identical config"
        ae.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        start = ck["epoch"]; history = ck["history"]; best = ck["best"]
        torch.set_rng_state(ck["rng_state"].cpu())
        if device.startswith("cuda") and ck.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in ck["cuda_rng_state"]])
    for epoch in range(start, cfg["ae"]["epochs"]):
        t0 = time.monotonic(); ae.train(); total = 0; count = 0
        for batch in loader:
            x = batch["image"].to(device); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.startswith("cuda") and cfg["training"]["amp"]):
                loss = F.l1_loss(ae(x), x)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite AE loss")
            loss.backward(); opt.step()
            total += loss.item()*len(x); count += len(x)
        v = ae_validation(ae, val, device)
        record = dict(epoch=epoch+1, train_l1=total/count, val_l1=v,
                      seconds=time.monotonic()-t0)
        history.append(record); is_best = v < best; best = min(v, best)
        ck = dict(kind="ae", model=ae.state_dict(), optimizer=opt.state_dict(), epoch=epoch+1,
                  config=cfg, history=history, best=best, rng_state=torch.get_rng_state(),
                  cuda_rng_state=torch.cuda.get_rng_state_all() if device.startswith("cuda") else None)
        save_checkpoint(run / "last.pt", ck)
        if is_best:
            save_checkpoint(run / "best.pt", ck)
        save_json(run / "history.json", history)
        save_json(run / "status.json", dict(state="training", **record))
        print(json.dumps(dict(stage="ae", **record)), flush=True)
    curves(run / "loss_curve.png", history)
    ae, _ = load_ae(run / "best.pt", device)
    x = next(iter(val))["image"].to(device)
    with torch.no_grad():
        y = ae(x)
    reconstruction_grid(run / "recon_grid.png", x, y, y, "Original | frozen AE | frozen AE")
    save_json(run / "status.json", dict(state="finished", epochs=len(history), best_val_l1=best))
    return run / "best.pt"


@torch.no_grad()
def cached_latents(cfg, ae, ae_path, split, device):
    # Only train/val are cached during training; tests are opened by explicit evaluation.
    key = digest(dict(ae_sha=__import__("hashlib").sha256(Path(ae_path).read_bytes()).hexdigest(),
                      manifest=Path(cfg["data"]["root"], f"{split}.jsonl").read_text()))
    root = Path(cfg["data"]["root"]) / "latent_cache"
    path = root / f"{split}_{key[:16]}.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    tensors = []
    for batch in make_loader(cfg, split):
        tensors.append(ae.encoder(batch["image"].to(device)).cpu().half())
    result = torch.cat(tensors)
    save_checkpoint(path, result)
    return result


def train_gene(cfg, ae_path, run_dir, method="urbangene", device="cpu", resume=None):
    setup(cfg["seed"], cfg["training"]["cpu_threads"])
    run = Path(run_dir); snapshot(run, cfg)
    ae, _ = load_ae(ae_path, device)
    model = UrbanGene(cfg, ae, method).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(parameters, lr=cfg["training"]["learning_rate"])
    train = cached_latents(cfg, ae, ae_path, "train", device)
    val = cached_latents(cfg, ae, ae_path, "val_seen", device)
    loader = DataLoader(TensorDataset(train), batch_size=cfg["training"]["batch_size"], shuffle=True)
    history = []; best = float("inf"); start = 0
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        assert ck["config"] == cfg and ck["method"] == method
        model.load_state_dict(ck["model"], strict=False); opt.load_state_dict(ck["optimizer"])
        history = ck["history"]; start = ck["epoch"]; best = ck["best"]
        torch.set_rng_state(ck["rng_state"].cpu())
        if device.startswith("cuda") and ck.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in ck["cuda_rng_state"]])
    gradient_checked = False
    for epoch in range(start, cfg["training"]["epochs"]):
        t0 = time.monotonic(); model.train(); total = 0.; count = 0
        temperature = max(0.5, 1-0.5*epoch/max(1, cfg["training"]["epochs"]-1))
        for (latent,) in loader:
            latent = latent.to(device).float(); opt.zero_grad(set_to_none=True)
            result = model.forward_latent(latent, temperature)
            loss = F.l1_loss(result["latent_hat"], latent)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite Gene loss")
            loss.backward()
            if not gradient_checked:
                grads = {"library": model.library.templates.grad.norm().item(),
                         "parser": sum(p.grad.norm().item() for p in model.parser.parameters() if p.grad is not None)}
                assert grads["library"] > 0 and grads["parser"] > 0, grads
                assert all(p.grad is None for p in ae.parameters())
                save_json(run / "gradient_check.json", grads)
                gradient_checked = True
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            opt.step(); total += loss.item()*len(latent); count += len(latent)
        model.eval(); vals = []
        with torch.no_grad():
            for (z,) in DataLoader(TensorDataset(val), batch_size=cfg["training"]["batch_size"]):
                z = z.to(device).float()
                vals.extend((model.forward_latent(z)["latent_hat"]-z).abs().mean((1, 2, 3)).tolist())
        v = sum(vals)/len(vals)
        record = dict(epoch=epoch+1, train_l1=total/count, val_l1=v, temperature=temperature,
                      seconds=time.monotonic()-t0)
        history.append(record); is_best = v < best; best = min(best, v)
        ck = dict(kind="gene", method=method,
                  model={k: v for k, v in model.state_dict().items() if not k.startswith("image_ae.")},
                  optimizer=opt.state_dict(), config=cfg, ae_path=str(Path(ae_path).resolve()),
                  ae_sha256=hashlib.sha256(Path(ae_path).read_bytes()).hexdigest(),
                  epoch=epoch+1, history=history, best=best, rng_state=torch.get_rng_state(),
                  cuda_rng_state=torch.cuda.get_rng_state_all() if device.startswith("cuda") else None)
        save_checkpoint(run / "last.pt", ck)
        if is_best:
            save_checkpoint(run / "best.pt", ck)
        save_json(run / "history.json", history)
        save_json(run / "status.json", dict(state="training", **record))
        print(json.dumps(dict(stage=method, **record)), flush=True)
    curves(run / "loss_curve.png", history)
    save_json(run / "status.json", dict(state="finished", epochs=len(history), best_val_l1=best,
              trainable_parameters=sum(p.numel() for p in parameters), budget=model.budget,
              gene_amp=False, gene_precision="float32; stable grid_sample/ST derivatives"))
    return run / "best.pt"
