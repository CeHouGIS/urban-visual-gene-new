"""Strict local pretrained loading, without fallback to random weights."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .common import sha256


def coordinates(height, width, device=None, dtype=torch.float32):
    y = (torch.arange(height, device=device, dtype=dtype) + .5) * 2 / height - 1
    x = (torch.arange(width, device=device, dtype=dtype) + .5) * 2 / width - 1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], -1)


class FrozenVisualBackbone(nn.Module):
    def __init__(self, model, resolution, patch_size, backend, provenance):
        super().__init__()
        if resolution % patch_size:
            raise ValueError("Resolution must be a multiple of the patch size")
        self.model = model.requires_grad_(False).eval()
        self.resolution, self.patch_size = resolution, patch_size
        self.backend, self.provenance = backend, provenance
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    @classmethod
    def from_config(cls, cfg):
        path = Path(cfg["weights"]).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Pretrained DINO weights not found: {path}")
        family, backend = cfg["family"], cfg["backend"]
        if backend == "transformers":
            from transformers import AutoModel
            if not path.is_dir():
                raise ValueError("Transformers needs a model directory with config.json")
            model, loading = AutoModel.from_pretrained(
                str(path), local_files_only=True, output_loading_info=True)
            if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
                raise ValueError(f"Incomplete pretrained model: {loading}")
            model_type = model.config.model_type
            if family not in model_type or "convnext" in model_type:
                raise ValueError(f"Expected {family} ViT, got {model_type}")
            patch = model.config.patch_size
            provenance = {"backend": backend, "family": family, "path": str(path),
                          "model_type": model_type, "config": model.config.to_dict(),
                          "files": {p.name: sha256(p) for p in sorted(path.iterdir())
                                    if p.suffix in {".json", ".safetensors", ".bin"}}}
        elif backend == "torchhub_local":
            repo = Path(cfg["repository"]).resolve()
            if not (repo / "hubconf.py").is_file():
                raise FileNotFoundError(f"Local DINO source missing: {repo}")
            name = cfg["architecture"]
            if not name.startswith(family):
                raise ValueError("Architecture/family mismatch")
            model = torch.hub.load(str(repo), name, source="local", pretrained=False)
            state = torch.load(path, weights_only=True, map_location="cpu")
            if cfg.get("state_dict_key"):
                state = state[cfg["state_dict_key"]]
            elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            model.load_state_dict(state, strict=True)
            patch = cfg.get("patch_size", 16 if family == "dinov3" else 14)
            provenance = {"backend": backend, "family": family, "path": str(path),
                          "architecture": name, "sha256": sha256(path),
                          "hubconf_sha256": sha256(repo / "hubconf.py")}
        else:
            raise ValueError(f"Unsupported backbone backend: {backend}")
        return cls(model, cfg["resolution"], patch, backend, provenance)

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or not torch.isfinite(images).all():
            raise ValueError("Expected finite RGB [B,3,H,W]")
        if images.min() < 0 or images.max() > 1:
            raise ValueError("Images must be in [0,1]")
        x = F.interpolate(images, (self.resolution, self.resolution), mode="bicubic",
                          align_corners=False, antialias=True).clamp(0, 1)
        x = (x - self.mean) / self.std
        side = self.resolution // self.patch_size
        if self.backend == "transformers":
            # ViT prefix is CLS plus optional register/storage tokens.
            tokens = self.model(pixel_values=x).last_hidden_state[:, -side * side:]
        else:
            tokens = self.model.forward_features(x)["x_norm_patchtokens"]
        if tokens.shape[1] != side * side or not torch.isfinite(tokens).all():
            raise ValueError("Invalid spatial DINO output")
        features = tokens.transpose(1, 2).reshape(len(images), -1, side, side).float()
        return {"features": features,
                "coordinates": coordinates(side, side, features.device)}
