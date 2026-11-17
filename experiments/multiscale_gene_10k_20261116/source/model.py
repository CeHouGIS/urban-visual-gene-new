"""Multiscale feature-additive genes with a shared nonlinear RGB decoder.

All image-dependent decoder inputs are synthesized through gene dictionaries.
The dictionary sizes and positive coefficient budgets are matched per image.
"""
import torch
from torch import nn
from torch.nn import functional as F

from legacy_model import Encoder, Decoder


VARIANTS = ('single_vector', 'multiscale_vector', 'multiscale_conv', 'dense_ae')


def positive_topk_per_image(scores, budget):
    if budget < 1:
        raise ValueError('Positive coefficient budget is required')
    flat = F.relu(scores).flatten(1)
    values, indices = flat.topk(min(budget, flat.shape[1]), dim=1)
    return torch.zeros_like(flat).scatter(1, indices, values).view_as(scores)


class PyramidEncoder(Encoder):
    def forward(self, image):
        maps = {}
        value = image
        for index, layer in enumerate(self.net):
            value = layer(value)
            if index == 5:
                maps['fine'] = value
            elif index == 7:
                maps['medium'] = value
        maps['coarse'] = value
        return maps


class PyramidDecoder(Decoder):
    def __init__(self, channels, multiscale):
        # Build the original trunk before additional projections, preserving its
        # identical random initialization in all four variants.
        super().__init__(channels)
        self.projections = nn.ModuleDict()
        if multiscale:
            self.projections['medium'] = nn.Conv2d(channels, 128, 1, bias=False)
            self.projections['fine'] = nn.Conv2d(channels, 96, 1, bias=False)

    def forward(self, features):
        value = features['coarse']
        for index, layer in enumerate(self.net):
            value = layer(value)
            if index == 4 and 'medium' in features:
                value = value + self.projections['medium'](features['medium'])
            elif index == 7 and 'fine' in features:
                value = value + self.projections['fine'](features['fine'])
        return value


class GeneDictionary(nn.Module):
    def __init__(self, genes, channels, kernel):
        super().__init__()
        raw = torch.randn(genes, channels, kernel, kernel)
        self.raw = nn.Parameter(F.normalize(raw.flatten(1), dim=1).view_as(raw))
        self.kernel = kernel

    @property
    def atoms(self):
        return F.normalize(self.raw.flatten(1), dim=1).view_as(self.raw)

    def forward(self, activations):
        if self.kernel == 1:
            return torch.einsum('bnhw,nc->bchw', activations, self.atoms[:, :, 0, 0])
        # PyTorch uses cross-correlation. This orientation is kept consistently
        # for full synthesis, single-gene contributions and deletion edits.
        weights = self.atoms.permute(1, 0, 2, 3).contiguous()
        return F.conv2d(activations, weights, padding=self.kernel // 2)

    def contribution(self, activations, gene):
        if not 0 <= gene < self.raw.shape[0]:
            raise ValueError('Gene index out of range')
        weights = self.atoms[gene].unsqueeze(1)
        return F.conv2d(activations[:, gene:gene + 1], weights,
                        padding=self.kernel // 2)


class MultiscaleGeneAE(nn.Module):
    def __init__(self, variant, latent_channels=64, dictionary_size=256,
                 multiscale_dictionary_sizes=(96, 80, 80),
                 active_coefficients_per_image=1024,
                 multiscale_coefficient_budgets=(512, 256, 256)):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f'Unknown variant: {variant}')
        if sum(multiscale_dictionary_sizes) != dictionary_size:
            raise ValueError('Multiscale dictionaries must sum to the common gene count')
        if sum(multiscale_coefficient_budgets) != active_coefficients_per_image:
            raise ValueError('Multiscale coefficient budgets must sum to the common budget')
        self.variant = variant
        self.sparse = variant != 'dense_ae'
        self.multiscale = variant.startswith('multiscale_')
        self.scales = ('coarse', 'medium', 'fine') if self.multiscale else ('coarse',)
        self.gene_counts = dict(zip(self.scales, multiscale_dictionary_sizes if self.multiscale
                                    else [dictionary_size])) if self.sparse else {}
        self.budgets = dict(zip(self.scales, multiscale_coefficient_budgets if self.multiscale
                                else [active_coefficients_per_image])) if self.sparse else {}
        self.encoder = PyramidEncoder()
        self.decoder = PyramidDecoder(latent_channels, self.multiscale)
        encoder_channels = {'coarse': 192, 'medium': 128, 'fine': 96}
        self.heads = nn.ModuleDict()
        self.dictionaries = nn.ModuleDict()
        for scale in self.scales:
            self.heads[scale] = nn.Conv2d(encoder_channels[scale],
                self.gene_counts[scale] if self.sparse else latent_channels, 1)
            if self.sparse:
                nn.init.normal_(self.heads[scale].weight, std=0.03)
                nn.init.constant_(self.heads[scale].bias, 0.1)
                self.dictionaries[scale] = GeneDictionary(self.gene_counts[scale],
                    latent_channels, 3 if variant == 'multiscale_conv' else 1)

    def encode(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or any(s % 16 for s in image.shape[-2:]):
            raise ValueError('Expected BCHW RGB with spatial dimensions divisible by 16')
        encoded = self.encoder(image)
        features, activations = {}, {}
        area_ratio = image.shape[-2] * image.shape[-1] / (256 * 256)
        for scale in self.scales:
            scores = self.heads[scale](encoded[scale])
            if self.sparse:
                budget = max(1, round(self.budgets[scale] * area_ratio))
                act = positive_topk_per_image(scores, budget)
                features[scale] = self.dictionaries[scale](act)
                activations[scale] = act
            else:
                features[scale] = scores
        return features, activations

    def forward(self, image):
        features, activations = self.encode(image)
        return {'reconstruction': self.decoder(features), 'features': features,
                'activations': activations, 'latent': features['coarse']}

    def contribution(self, activations, scale, gene):
        if not self.sparse or scale not in self.dictionaries:
            raise ValueError('No dictionary at the requested scale')
        return self.dictionaries[scale].contribution(activations[scale], gene)

    def usage(self, activations):
        if not self.sparse:
            return torch.zeros(0, dtype=torch.int64, device=next(self.parameters()).device)
        return torch.cat([(activations[s] > 0).sum((0, 2, 3)) for s in self.scales])

    def spec(self, resolution=256):
        grids = {'coarse': resolution // 16, 'medium': resolution // 8, 'fine': resolution // 4}
        code_slots = sum(self.gene_counts[s] * grids[s] ** 2 for s in self.scales) if self.sparse else 0
        return {'variant': self.variant, 'parameters': sum(p.numel() for p in self.parameters()),
                'dictionary_parameters': sum(p.numel() for p in self.dictionaries.parameters()),
                'dictionary_genes': sum(self.gene_counts.values()),
                'scale_gene_counts': self.gene_counts, 'scale_grids': {s: grids[s] for s in self.scales},
                'scale_coefficient_budgets': self.budgets, 'coefficient_slots': code_slots,
                'maximum_nonzero_coefficients_per_image': sum(self.budgets.values()),
                'sparsity_policy': 'positive TopK across channel and space, separately per scale',
                'strict_additivity': 'per-scale synthesized features',
                'rgb_decode': 'nonlinear; isolated RGB contributions need not sum',
                'encoder_bypass': False}


def create_model(config, variant):
    torch.manual_seed(config['seed'])
    keys = ('latent_channels', 'dictionary_size', 'multiscale_dictionary_sizes',
            'active_coefficients_per_image', 'multiscale_coefficient_budgets')
    return MultiscaleGeneAE(variant, **{k: config['model'][k] for k in keys}).cuda()
