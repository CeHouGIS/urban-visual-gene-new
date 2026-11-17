import math

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import coordinates


def geometry_from_masks(masks, grid):
    weights = masks.flatten(2)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    xy = grid.reshape(-1, 2)
    center = torch.einsum("bmn,nd->bmd", weights, xy)
    variance = torch.einsum("bmn,bmnd->bmd", weights,
                            (xy[None, None] - center[:, :, None]).square())
    scale = (variance + 1e-4).sqrt().clamp(.06, 2)
    return torch.cat([center, scale], -1)


class AdaptiveSpatialSlotEncoder(nn.Module):
    """Competitive slot attention with a learned Gaussian spatial preference."""
    def __init__(self, feature_dim, slot_dim=128, max_slots=32, iterations=3,
                 spatial_strength=.1, background_slot=True, spatial_initial_scale=None):
        super().__init__()
        if max_slots < 1 or iterations < 1:
            raise ValueError("At least one slot and one iteration are required")
        self.max_slots, self.slot_dim, self.iterations = max_slots, slot_dim, iterations
        self.spatial_strength, self.background_slot = spatial_strength, background_slot
        if spatial_initial_scale is not None and spatial_initial_scale <= 0:
            raise ValueError("spatial_initial_scale must be positive")
        self.spatial_initial_scale = spatial_initial_scale
        self.input_net = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, slot_dim),
                                       nn.GELU(), nn.Linear(slot_dim, slot_dim))
        self.slot_norm = nn.LayerNorm(slot_dim)
        self.key, self.value, self.query = [nn.Linear(slot_dim, slot_dim, bias=False) for _ in range(3)]
        self.mu = nn.Parameter(torch.zeros(1, 1, slot_dim))
        self.log_sigma = nn.Parameter(torch.full((1, 1, slot_dim), -.5))
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.residual = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, 2 * slot_dim),
                                      nn.GELU(), nn.Linear(2 * slot_dim, slot_dim))
        self.presence = nn.Linear(slot_dim, 1)
        nn.init.zeros_(self.presence.weight)
        nn.init.constant_(self.presence.bias, 2.)

    def initial_geometry(self, grid, batch_size):
        """Farthest-point spatial seeds; masks and geometry then adapt normally.

        This initializes competition using coordinates only, without object
        classes, masks, boxes, counts, or any fixed final region assignment.
        """
        if self.spatial_initial_scale is None:
            return None
        points = grid.reshape(-1, 2)
        count = self.max_slots - int(self.background_slot)
        if count > len(points):
            raise ValueError("Spatially initialized foreground slots exceed patch count")
        centers = []
        for _ in range(batch_size):
            indices = []
            distances = torch.full((len(points),), float("inf"), device=points.device)
            selected = torch.randint(len(points), (), device=points.device)
            for _ in range(count):
                indices.append(selected)
                distances = torch.minimum(distances, (points - points[selected]).square().sum(-1))
                selected = distances.argmax()
            center = points[torch.stack(indices)] if indices else points[:0]
            if self.background_slot:
                center = torch.cat([torch.zeros_like(points[:1]), center], 0)
            centers.append(center)
        center = torch.stack(centers)
        return torch.cat([center, torch.full_like(center, self.spatial_initial_scale)], -1)

    def forward(self, features, initial_slots=None):
        b, _, h, w = features.shape
        grid = coordinates(h, w, features.device, features.dtype)
        tokens = self.input_net(features.flatten(2).transpose(1, 2))
        k, v = self.key(tokens), self.value(tokens)
        if initial_slots is None:
            noise = torch.randn(b, self.max_slots, self.slot_dim, device=features.device)
            slots = self.mu + self.log_sigma.exp() * noise
        else:
            slots = initial_slots
        if slots.shape != (b, self.max_slots, self.slot_dim):
            raise ValueError("Invalid initial slots")
        geom = self.initial_geometry(grid, b)
        for _ in range(self.iterations):
            old = slots
            logits = torch.einsum("bmd,bnd->bmn", self.query(self.slot_norm(slots)), k)
            logits = logits / math.sqrt(self.slot_dim)
            if geom is not None:
                relative = (grid.flatten(0, 1)[None, None] - geom[:, :, None, :2]) / geom[:, :, None, 2:]
                bias = -.5 * relative.square().sum(-1)
                if self.background_slot:
                    bias = bias * (torch.arange(self.max_slots, device=features.device) > 0)[None, :, None]
                logits = logits + self.spatial_strength * bias
            exist = self.presence(slots).squeeze(-1)
            if self.background_slot:
                gate_log = F.logsigmoid(exist)
                gate_log = torch.cat([torch.zeros_like(gate_log[:, :1]), gate_log[:, 1:]], 1)
            else:
                gate_log = F.logsigmoid(exist)
            attention = (logits + gate_log[..., None]).softmax(1)
            weights = attention + 1e-8
            weights = weights / weights.sum(-1, keepdim=True)
            updates = torch.einsum("bmn,bnd->bmd", weights, v)
            slots = self.gru(updates.reshape(-1, self.slot_dim), old.reshape(-1, self.slot_dim))
            slots = slots.reshape(b, self.max_slots, self.slot_dim)
            slots = slots + self.residual(slots)
            geom = geometry_from_masks(attention.reshape(b, self.max_slots, h, w), grid)
        return {"slot_features": slots, "attention_masks": attention.reshape(b, self.max_slots, h, w),
                "exist_logits": self.presence(slots).squeeze(-1), "estimated_geometry": geom}


class FeatureReconstructionDecoder(nn.Module):
    def __init__(self, feature_dim, slot_dim, hidden_dim=128, background_slot=True):
        super().__init__()
        self.background_slot = background_slot
        self.network = nn.Sequential(nn.Linear(slot_dim + 4, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, feature_dim + 1))

    def forward(self, slots, geometry, exist_logits, height, width):
        b, m, d = slots.shape
        grid = coordinates(height, width, slots.device, slots.dtype)
        xy = grid.reshape(1, 1, -1, 2).expand(b, m, -1, -1)
        relative = ((xy - geometry[:, :, None, :2]) / geometry[:, :, None, 2:]).clamp(-10, 10)
        embedded = torch.cat([slots[:, :, None].expand(-1, -1, height * width, -1), xy, relative], -1)
        hidden = self.network[:-1](embedded)
        projection = self.network[-1]
        raw_masks = F.linear(hidden, projection.weight[-1:], projection.bias[-1:]).squeeze(-1)
        mask_logits = raw_masks + F.logsigmoid(exist_logits)[..., None]
        if self.background_slot:
            mask_logits = torch.cat([raw_masks[:, :1], mask_logits[:, 1:]], 1)
        masks = mask_logits.softmax(1)
        # The shared final projection is linear, so combine hidden states first.
        # This is algebraically identical and avoids [B,M,HW,feature_dim] storage.
        combined = (hidden * masks[..., None]).sum(1)
        reconstruction = F.linear(combined, projection.weight[:-1], projection.bias[:-1])
        return reconstruction.transpose(1, 2).reshape(b, -1, height, width), masks.reshape(b, m, height, width)


class DiscoveryModel(nn.Module):
    def __init__(self, feature_dim, slot_dim=128, max_slots=32, iterations=3,
                 hidden_dim=128, spatial_strength=.1, background_slot=True,
                 spatial_initial_scale=None):
        super().__init__()
        self.encoder = AdaptiveSpatialSlotEncoder(feature_dim, slot_dim, max_slots, iterations,
                                                  spatial_strength, background_slot,
                                                  spatial_initial_scale)
        self.decoder = FeatureReconstructionDecoder(feature_dim, slot_dim, hidden_dim, background_slot)
        self.background_slot = background_slot

    def forward(self, features, initial_slots=None):
        if features.ndim != 4 or not torch.isfinite(features).all():
            raise ValueError("Expected finite [B,C,H,W] frozen features")
        out = self.encoder(features, initial_slots)
        reconstructed, masks = self.decoder(out["slot_features"], out["estimated_geometry"],
                                             out["exist_logits"], *features.shape[-2:])
        out.update(soft_masks=masks, reconstructed_features=reconstructed,
                   background_state=out["slot_features"][:, 0] if self.background_slot else None)
        return out
