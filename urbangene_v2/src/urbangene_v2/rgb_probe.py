"""Diagnostic continuous-slot -> frozen VAE bridge; not a discrete Gene renderer."""
import torch
from torch import nn
from torch.nn import functional as F

from .composition import decode_program


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, channels), nn.SiLU(),
                                 nn.Conv2d(channels, channels, 3, padding=1),
                                 nn.GroupNorm(8, channels), nn.SiLU(),
                                 nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.net(x) * .5


class SlotRGBBridge(nn.Module):
    """Only reconstructed features enter. Original pixels/latents are targets only.

    The shared discovery decoder has a final linear projection W:128->1024.
    Its output lies in bias+col(W). QR therefore compresses that output to 128
    channels without adding source information; Q is a frozen shared buffer.
    """
    def __init__(self, projection_weight, projection_bias, width=96):
        super().__init__()
        basis = torch.linalg.qr(projection_weight.detach().float(), mode='reduced').Q
        dim = basis.shape[1]
        self.register_buffer('basis', basis)
        self.register_buffer('feature_bias', projection_bias.detach().float().clone())
        self.register_buffer('input_mean', torch.zeros(1, dim, 1, 1))
        self.register_buffer('input_std', torch.ones(1, dim, 1, 1))
        self.register_buffer('latent_mean', torch.zeros(1, 4, 1, 1))
        self.stem = nn.Conv2d(dim + 2, width, 1)
        self.fine = nn.Sequential(ResidualBlock(width), ResidualBlock(width))
        self.coarse = nn.Sequential(nn.AvgPool2d(2), ResidualBlock(width),
                                    ResidualBlock(width), ResidualBlock(width))
        self.fuse = nn.Sequential(nn.Conv2d(width * 2, width, 1), ResidualBlock(width))
        self.output = nn.Sequential(nn.Conv2d(width, 64, 3, padding=1),
                                    ResidualBlock(64), ResidualBlock(64),
                                    nn.SiLU(), nn.Conv2d(64, 4, 3, padding=1))

    def project(self, features):
        if features.ndim != 4 or features.shape[1] != len(self.feature_bias):
            raise ValueError('Expected reconstructed discovery features [B,C,H,W]')
        return F.conv2d(features - self.feature_bias[None, :, None, None],
                        self.basis.T[:, :, None, None])

    def latent_from_projected(self, projected):
        if projected.ndim != 4 or projected.shape[1] != self.basis.shape[1]:
            raise ValueError('Wrong projected feature dimensions')
        b, _, h, w = projected.shape
        x = (projected - self.input_mean) / self.input_std.clamp_min(.01)
        yy = (torch.arange(h, device=x.device, dtype=x.dtype) + .5) * 2 / h - 1
        xx = (torch.arange(w, device=x.device, dtype=x.dtype) + .5) * 2 / w - 1
        y, xcoord = torch.meshgrid(yy, xx, indexing='ij')
        xy = torch.stack([xcoord, y])[None].expand(b, -1, -1, -1)
        fine = self.fine(self.stem(torch.cat([x, xy], 1)))
        coarse = F.interpolate(self.coarse(fine), (h, w), mode='bilinear', align_corners=False)
        merged = self.fuse(torch.cat([fine, coarse], 1))
        up = F.interpolate(merged, (2*h, 2*w), mode='bilinear', align_corners=False)
        return self.output(up) + self.latent_mean

    def forward(self, reconstructed_features):
        return self.latent_from_projected(self.project(reconstructed_features))


def freeze_module(module):
    return module.eval().requires_grad_(False)


def decode_vae_rgb(vae, latent):
    """Raw posterior-mode latents: no diffusion scheduler/scaling factor."""
    return ((vae.decode(latent).sample.float() + 1) / 2).clamp(0, 1)


def render_slot_program(program, feature_decoder, bridge, vae):
    """A replay needs a program plus shared weights, with no reference image."""
    reconstructed = decode_program(feature_decoder, program)['reconstructed_features']
    latent = bridge(reconstructed)
    return decode_vae_rgb(vae, latent)
