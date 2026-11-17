"""Residual CNN building blocks; all parameters initialized from scratch."""
import torch
from torch import nn
class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(8, channels),
            nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(3, 32, 3, padding=1), Residual(32)]
        for source, target in [(32,64),(64,96),(96,128),(128,192)]:
            layers.extend([nn.Conv2d(source,target,4,stride=2,padding=1),Residual(target)])
        layers.extend([Residual(192),nn.GroupNorm(8,192),nn.SiLU()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = [nn.Conv2d(channels,192,3,padding=1),Residual(192),Residual(192)]
        for source, target in [(192,128),(128,96),(96,64),(64,32)]:
            layers.extend([nn.Upsample(scale_factor=2,mode='nearest'),
                           nn.Conv2d(source,target,3,padding=1),Residual(target)])
        layers.extend([nn.GroupNorm(8,32),nn.SiLU(),nn.Conv2d(32,3,3,padding=1),nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


