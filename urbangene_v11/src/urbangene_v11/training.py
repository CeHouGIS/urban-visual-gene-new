import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from urbangene.train import load_ae, cached_latents
from urbangene.utils import save_json, save_checkpoint, setup, snapshot
from urbangene.visualize import curves
from .model import UrbanGeneV11


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gradient_norms(model):
    head = model.parser.head.weight.grad
    norms = {}
    if head is not None:
        head = head.reshape(16, 325, -1)
        for name, sl in {"gene_id": slice(0, 64), "active": slice(64, 66),
                         "position": slice(66, 322), "scale": slice(322, 325)}.items():
            norms[name] = float(head[:, sl].norm())
    for name in ["library", "composer"]:
        grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
        norms[name] = sum(float(g.square().sum()) for g in grads)**0.5
        if any(not torch.isfinite(g).all() for g in grads):
            raise RuntimeError(f"Nonfinite {name} gradient")
    if hasattr(model.composer, "relation"):
        for name in ["embedding", "relation", "spatial"]:
            grads = [p.grad for p in getattr(model.composer, name).parameters() if p.grad is not None]
            norms[name] = sum(float(g.square().sum()) for g in grads)**0.5
    assert all(p.grad is None for p in model.ae.parameters()), "AE must stay frozen"
    return norms


@torch.no_grad()
def validation(model, latents, batch_size, device):
    model.eval(); losses = []; shuffled = []; usages = torch.zeros(64, dtype=torch.long)
    active_counts = []; coverage = []
    for start in range(0, len(latents), batch_size):
        z = latents[start:start+batch_size].float().to(device)
        out = model.forward_latent(z); program = out["program"]
        losses.extend((z-out["latent_hat"]).abs().mean((1, 2, 3)).tolist())
        rolled = {k: v.roll(1, 0) for k, v in program.items()}
        zs = model.compose_program(rolled, False)["latent_hat"]
        shuffled.extend((z-zs).abs().mean((1, 2, 3)).tolist())
        active = program["enabled"].bool()
        usages += torch.bincount(program["gene_ids"][active].cpu(), minlength=64)
        active_counts.extend(active.sum(1).tolist())
        from .models.composition.geometry import ProgramGeometry
        geo = ProgramGeometry(model.cfg).to(device)
        sample = geo.sample(geo(model.library, program))
        coverage.extend(sample["coverage"].float().mean((1, 2)).tolist())
    return dict(val_l1=sum(losses)/len(losses), shuffled_l1=sum(shuffled)/len(shuffled),
                genes_used=int((usages > 0).sum()), gene_usage=usages.tolist(),
                active_instances_mean=sum(active_counts)/len(active_counts),
                coverage_mean=sum(coverage)/len(coverage))


def train(cfg, ae_path, output, device="cpu", resume=None, max_steps=None):
    setup(cfg["seed"], cfg["training"]["cpu_threads"])
    output = Path(output); snapshot(output, cfg)
    ae, _ = load_ae(ae_path, device); ae_hash = sha256(ae_path)
    model = UrbanGeneV11(cfg, ae).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg["training"]["learning_rate"])
    train_z = cached_latents(cfg, ae, ae_path, "train", device)
    val_z = cached_latents(cfg, ae, ae_path, "val_seen", device)
    # One logical batch is always 32 (tiny smoke:4). Chunking only changes memory.
    generator = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(TensorDataset(train_z), batch_size=cfg["training"]["batch_size"],
                        shuffle=True, generator=generator)
    micro = cfg["training"].get("micro_batch_size", cfg["training"]["batch_size"])
    history = []; start = 0; best = float("inf"); step = 0; gradients = []
    # Same Gumbel stream across methods, independent of architecture initialization.
    torch.manual_seed(cfg["seed"]+2026)
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        assert ck["config"] == cfg and ck["ae_sha256"] == ae_hash
        model.load_state_dict(ck["model"], strict=False); opt.load_state_dict(ck["optimizer"])
        start = ck["epoch"]; step = ck["step"]; history = ck["history"]; best = ck["best"]
        generator.set_state(ck["loader_rng"]); torch.set_rng_state(ck["rng"])
        if device.startswith("cuda") and ck["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
    max_steps = max_steps or 10**12
    for epoch in range(start, cfg["training"]["epochs"]):
        model.train(); begin = time.monotonic(); total = 0.; count = 0
        temperature = max(0.5, 1-0.5*epoch/max(1, cfg["training"]["epochs"]-1))
        for (batch_z,) in loader:
            opt.zero_grad(set_to_none=True); batch_loss = 0.
            for chunk in batch_z.split(micro):
                z = chunk.float().to(device)
                predicted = model.forward_latent(z, temperature)["latent_hat"]
                loss = F.l1_loss(predicted, z)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite latent L1")
                (loss*len(chunk)/len(batch_z)).backward()
                batch_loss += float(loss.detach())*len(chunk)/len(batch_z)
            if step < 3 or step % 100 == 0:
                grad = dict(step=step+1, **gradient_norms(model)); gradients.append(grad)
                save_json(output / "grad_norms.json", gradients)
                # Interior gradients are expected to be zero only before the first
                # zero-initialized output head update; parser remains trainable.
                if step < 3:
                    assert all(grad[name] > 0 for name in ["gene_id", "active", "position", "scale", "library"])
            torch.nn.utils.clip_grad_norm_(params, 1.)
            opt.step(); step += 1; total += batch_loss*len(batch_z); count += len(batch_z)
            if step % 50 == 0:
                status = dict(state="training", epoch=epoch+1, step=step, batch_l1=batch_loss,
                              elapsed_epoch_seconds=time.monotonic()-begin)
                save_json(output / "status.json", status); print(json.dumps(status), flush=True)
            if step >= max_steps:
                break
        record = dict(epoch=epoch+1, step=step, train_l1=total/count, temperature=temperature,
                      seconds=time.monotonic()-begin, **validation(model, val_z, micro, device))
        # Hard-ST vs deterministic argmax diagnostic on the same validation subset;
        # no updates or RNG perturbation of the training stream.
        with torch.random.fork_rng(devices=[torch.device(device).index or 0] if device.startswith("cuda") else []):
            model.train()
            with torch.no_grad():
                z = val_z[:micro].float().to(device)
                record["val_hard_st_l1"] = float((model.forward_latent(z, temperature)["latent_hat"]-z).abs().mean())
        model.eval(); history.append(record)
        improved = record["val_l1"] < best; best = min(best, record["val_l1"])
        state = dict(version="1.1", kind="gene", config=cfg,
                model={k: v for k, v in model.state_dict().items() if not k.startswith("ae.")},
                optimizer=opt.state_dict(), epoch=epoch+1, step=step, history=history, best=best,
                ae_path=str(Path(ae_path).resolve()), ae_sha256=ae_hash,
                rng=torch.get_rng_state(), loader_rng=generator.get_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.startswith("cuda") else None)
        save_checkpoint(output / "last.pt", state)
        if improved:
            save_checkpoint(output / "best.pt", state)
        save_json(output / "history.json", history)
        save_json(output / "status.json", dict(state="training", **record))
        print(json.dumps(record), flush=True)
        if step >= max_steps:
            break
    curves(output / "loss_curve.png", history)
    assert sha256(ae_path) == ae_hash
    save_json(output / "status.json", dict(state="finished", epochs=len(history), steps=step,
              parameter_counts=model.parameter_counts(), best_val_l1=best, ae_sha256=ae_hash,
              precision="float32", loss="latent_l1", target_free_generation_trained=False))
    return output / "best.pt"


def load_model(checkpoint, device="cpu"):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert sha256(ck["ae_path"]) == ck["ae_sha256"]
    ae, _ = load_ae(ck["ae_path"], device)
    model = UrbanGeneV11(ck["config"], ae).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not unexpected and all(k.startswith("ae.") for k in missing)
    return model.eval(), ck
