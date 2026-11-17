"""RGB autoencoders with identical trunks and an explicit additive bottleneck."""
import torch
from torch import nn
from torch.nn import functional as F


def sparse_nonnegative(scores, active):
    if not 1 <= active <= scores.shape[1]:
        raise ValueError('Invalid per-location sparsity budget')
    positive = F.relu(scores)
    values, indices = positive.topk(active, dim=1)
    return torch.zeros_like(positive).scatter(1, indices, values)


def sum_contributions(activations, dictionary):
    return torch.einsum('bnhw,nc->bchw', activations, dictionary)


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


class GeneAE(nn.Module):
    def __init__(self, variant, latent_channels=64, dictionary_size=256, active_per_location=4):
        super().__init__()
        if variant not in {'dense_ae','sparse_n256_s4'}:
            raise ValueError('Unknown variant')
        self.variant = variant
        self.sparse = variant != 'dense_ae'
        self.active = active_per_location
        self.encoder = Encoder()
        self.decoder = Decoder(latent_channels)
        # Construct shared trunks before variant-specific layers for matched initialization.
        self.head = nn.Conv2d(192, dictionary_size if self.sparse else latent_channels, 1)
        if self.sparse:
            if not 1 <= active_per_location <= dictionary_size:
                raise ValueError('Invalid sparsity')
            nn.init.normal_(self.head.weight, std=0.03)
            nn.init.constant_(self.head.bias, 0.1)
            self.dictionary_raw = nn.Parameter(F.normalize(torch.randn(dictionary_size,latent_channels),dim=1))
        else:
            self.register_parameter('dictionary_raw', None)

    @property
    def dictionary(self):
        if not self.sparse:
            raise ValueError('Dense baseline has no gene dictionary')
        return F.normalize(self.dictionary_raw,dim=1)

    def encode(self, image):
        scores = self.head(self.encoder(image))
        if self.sparse:
            activations = sparse_nonnegative(scores,self.active)
            latent = sum_contributions(activations,self.dictionary)
        else:
            activations = latent = scores
        return latent, activations

    def contribution(self, activations, gene):
        return activations[:,gene:gene+1] * self.dictionary[gene][None,:,None,None]

    def forward(self, image):
        if image.shape[-1] % 16 or image.shape[-2] % 16:
            raise ValueError('Image dimensions must be divisible by 16')
        latent, activations = self.encode(image)
        # This is the only decoder input. No image/encoder skips bypass the sum.
        return {'reconstruction':self.decoder(latent),'latent':latent,'activations':activations}
