"""Small DINOSAUR-style slot model, not an official DINOSAUR reproduction.

Features are DINOv3 patch vectors projected with training-only PCA. The RGB
head is deliberately small: it probes recoverable appearance, not high
fidelity generation. No object categories or ground-truth masks enter here.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def coordinates(height, width):
    y, x = torch.meshgrid(torch.linspace(-1, 1, height),
                          torch.linspace(-1, 1, width), indexing="ij")
    return torch.stack((x, y, 1 - x, 1 - y), -1).reshape(height * width, 4)


class SlotAttention(nn.Module):
    def __init__(self, dim=96, slots=16, iterations=3):
        super().__init__()
        self.dim, self.slots, self.iterations = dim, slots, iterations
        self.mu = nn.Parameter(torch.randn(1, 1, dim) * .02)
        self.log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.keys = nn.Linear(dim, dim, bias=False)
        self.values = nn.Linear(dim, dim, bias=False)
        self.queries = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 192), nn.ReLU(), nn.Linear(192, dim))
        generator = torch.Generator().manual_seed(20261011)
        self.register_buffer("eval_noise", torch.randn(1, slots, dim, generator=generator))

    def forward(self, inputs):
        batch = len(inputs)
        noise = torch.randn(batch, self.slots, self.dim, device=inputs.device) if self.training else self.eval_noise.expand(batch, -1, -1)
        slots = self.mu + self.log_sigma.exp() * noise
        inputs = self.norm_inputs(inputs)
        keys, values = self.keys(inputs), self.values(inputs)
        for _ in range(self.iterations):
            old = slots
            logits = torch.einsum("bkd,bpd->bkp", self.queries(self.norm_slots(slots)), keys) / math.sqrt(self.dim)
            attention = logits.softmax(dim=1)
            weights = attention + 1e-8
            weights = weights / weights.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bkp,bpd->bkd", weights, values)
            slots = self.gru(updates.reshape(-1, self.dim), old.reshape(-1, self.dim)).reshape(batch, self.slots, self.dim)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attention


class SlotModel(nn.Module):
    def __init__(self, slots=16, feature_dim=64, dim=96, rgb_mode="direct", position_strength=1.):
        super().__init__()
        self.slot_count, self.dim = slots, dim
        self.rgb_mode = rgb_mode
        self.position_strength = position_strength
        self.register_buffer("grid", coordinates(28, 28))
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_mlp = nn.Sequential(nn.Linear(feature_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.position = nn.Linear(4, dim)
        self.group = SlotAttention(dim, slots)
        self.decoder = nn.Sequential(nn.Linear(dim + 4, 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU(),
                                     nn.Linear(128, feature_dim + 1))
        self.rgb_decoder = nn.Sequential(nn.Linear(dim + 4, 128), nn.ReLU(),
                                         nn.Linear(128, 128), nn.ReLU(),
                                         nn.Linear(128, 3), nn.Sigmoid())
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(feature_dim, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid())

    def encode(self, features):
        inputs = self.input_mlp(self.input_norm(features)) + self.position_strength * self.position(self.grid)[None]
        return self.group(inputs)

    def broadcast(self, slots):
        batch, count, _ = slots.shape
        return torch.cat((slots[:, :, None].expand(-1, -1, 784, -1),
                          self.grid[None, None].expand(batch, count, -1, -1)), -1)

    def decode(self, slots):
        output = self.decoder(self.broadcast(slots))
        masks = output[..., -1].softmax(dim=1)
        features = (output[..., :-1] * masks[..., None]).sum(dim=1)
        return features, masks

    def render(self, slots, masks):
        # Masks may be held fixed for an explicitly imposed spatial-binding probe.
        if self.rgb_mode == "features":
            components = self.decoder(self.broadcast(slots))[..., :-1]
            features = (components * masks[..., None]).sum(dim=1)
            feature_map = features.reshape(len(slots), 28, 28, -1).permute(0, 3, 1, 2)
            return self.rgb_conv(feature_map), components
        components = self.rgb_decoder(self.broadcast(slots))
        return (components * masks[..., None]).sum(dim=1), components

    def forward(self, features):
        slots, attention = self.encode(features)
        reconstructed, masks = self.decode(slots)
        return {"slots": slots, "attention": attention,
                "features": reconstructed, "masks": masks}


def upsample_rgb(rgb, size=112):
    if rgb.ndim == 4:
        return rgb
    rgb = rgb.reshape(len(rgb), 28, 28, 3).permute(0, 3, 1, 2)
    return F.interpolate(rgb, (size, size), mode="bilinear", align_corners=False)
