"""From-scratch additive codes inferred by bounded, nonnegative unrolled ISTA."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from backbone import Encoder, Decoder

VARIANTS = ('single_scale_ista', 'multiscale_residual_ista')


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

    def weights(self):
        return self.atoms.permute(1, 0, 2, 3).contiguous()

    def forward(self, activation):
        return F.conv2d(activation, self.weights(), padding=self.kernel // 2)

    def adjoint(self, feature):
        return F.conv_transpose2d(feature, self.weights(), padding=self.kernel // 2)

    def lipschitz_bound(self, weights=None):
        # Unfold has squared operator norm <= kernel area. A row-sum bound on
        # W W^T upper-bounds its largest eigenvalue, without a large spatial FFT.
        weights = self.weights() if weights is None else weights
        patch = weights.detach().float().flatten(1)
        with torch.autocast('cuda', enabled=False):
            gram = patch @ patch.T
        return (1.01 * self.kernel ** 2 * gram.abs().sum(1).max()).clamp_min(1e-5)

    def contribution(self, activation, gene):
        return F.conv2d(activation[:, gene:gene+1], self.atoms[gene].unsqueeze(1),
                        padding=self.kernel // 2)


class ISTACode(nn.Module):
    def __init__(self, iterations, step_fraction):
        super().__init__()
        self.step_logits = nn.Parameter(torch.full((iterations,), math.log(step_fraction / (1-step_fraction))))

    def forward(self, target, dictionary, penalty):
        weights = dictionary.weights()
        bound = dictionary.lipschitz_bound(weights)
        # Work in the autocast dtype, but avoid accumulating error in the bound.
        analysis = F.conv_transpose2d(target, weights, padding=dictionary.kernel // 2)
        activation = torch.zeros_like(analysis)
        for fraction in self.step_logits.sigmoid():
            step = fraction / bound
            synthesized = F.conv2d(activation, weights, padding=dictionary.kernel // 2)
            gradient = F.conv_transpose2d(synthesized, weights, padding=dictionary.kernel // 2) - analysis
            activation = F.relu(activation - step * gradient - step * penalty)
        synthesized = F.conv2d(activation, weights, padding=dictionary.kernel // 2)
        return synthesized, activation


def pyramid_targets(image):
    height, width = image.shape[-2:]
    coarse = F.interpolate(image, size=(height//4, width//4), mode='area')
    medium = F.interpolate(image, size=(height//2, width//2), mode='area')
    up = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    return {'coarse': coarse,
            'medium': medium - up(coarse, medium.shape[-2:]),
            'fine': image - up(medium, image.shape[-2:])}


class ResidualISTAGeneAE(nn.Module):
    def __init__(self, variant, settings):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant, self.settings, self.sparse = variant, settings, True
        self.multiscale = variant == 'multiscale_residual_ista'
        self.scales = ('coarse','medium','fine') if self.multiscale else ('coarse',)
        channels = settings['latent_channels']
        sizes = settings['multiscale_dictionary_sizes'] if self.multiscale else [settings['single_dictionary_size']]
        kernels = settings['multiscale_kernels'] if self.multiscale else [1]
        self.gene_counts = dict(zip(self.scales, sizes))
        # Shared trunks are constructed before scale-specific modules.
        self.encoder = PyramidEncoder()
        self.decoder = PyramidDecoder(channels, self.multiscale)
        self.heads = nn.ModuleDict()
        self.dictionaries = nn.ModuleDict()
        self.inference = nn.ModuleDict()
        self.residual_heads = nn.ModuleDict()
        inputs = {'coarse':192,'medium':128,'fine':96}
        for scale, genes, kernel in zip(self.scales, sizes, kernels):
            self.heads[scale] = nn.Conv2d(inputs[scale], channels, 1, bias=False)
            self.dictionaries[scale] = GeneDictionary(genes, channels, kernel)
            self.inference[scale] = ISTACode(settings['ista_iterations'], settings['ista_step_fraction'])
            # Signed RGB outputs are necessary for Laplacian residual targets.
            self.residual_heads[scale] = nn.Conv2d(channels, 3, 3, padding=1)
        self.set_step(0)

    def set_step(self, step):
        self.step = int(step)
        def ramp(bounds):
            return max(0., min(1., (step-bounds[0]) / (bounds[1]-bounds[0])))
        self.gates = {'coarse':1., 'medium':ramp(self.settings['medium_ramp_updates']),
                      'fine':ramp(self.settings['fine_ramp_updates'])}
        fraction = max(0., min(1., step / self.settings['lambda_ramp_updates']))
        self.penalty = self.settings['lambda_initial'] + fraction * (self.settings['lambda_final']-self.settings['lambda_initial'])

    def encode(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or any(s % 16 for s in image.shape[-2:]):
            raise ValueError('Expected RGB BCHW dimensions divisible by 16')
        encoded = self.encoder(image)
        features, activations, targets = {}, {}, {}
        for scale in self.scales:
            projected = self.heads[scale](encoded[scale])
            # No learned rescaling can evade the coefficient penalty.
            target = F.normalize(projected.float(), dim=1, eps=1e-6) * math.sqrt(projected.shape[1])
            features[scale], activations[scale] = self.inference[scale](target, self.dictionaries[scale], self.penalty)
            targets[scale] = target
        return features, activations, targets

    def forward(self, image):
        features, activations, targets = self.encode(image)
        gated = {s: features[s] * self.gates[s] for s in self.scales}
        residual_predictions = {}
        divisors = {'coarse':4, 'medium':2, 'fine':1}
        for scale in self.scales:
            prediction = self.residual_heads[scale](features[scale])
            residual_predictions[scale] = F.interpolate(prediction,
                size=tuple(v//divisors[scale] for v in image.shape[-2:]), mode='bilinear', align_corners=False)
        return {'reconstruction': self.decoder(gated), 'features':features, 'decoder_features':gated,
                'activations':activations, 'feature_targets':targets,
                'residual_predictions':residual_predictions, 'latent':features['coarse']}

    def contribution(self, activations, scale, gene):
        return self.dictionaries[scale].contribution(activations[scale], gene)

    def usage(self, activations):
        return torch.cat([(activations[s] > 0).sum((0,2,3)) for s in self.scales])

    def spec(self, resolution=256):
        return {'variant':self.variant, 'parameters':sum(p.numel() for p in self.parameters()),
                'dictionary_parameters':sum(p.numel() for p in self.dictionaries.parameters()),
                'dictionary_genes':sum(self.gene_counts.values()), 'scale_gene_counts':self.gene_counts,
                'scale_grids':{s:resolution//d for s,d in [('coarse',16),('medium',8),('fine',4)] if s in self.scales},
                'dictionary_kernels':{s:self.dictionaries[s].kernel for s in self.scales},
                'ista_iterations':self.settings['ista_iterations'],
                'sparsity_policy':'nonnegative L1 proximal inference; no hard coefficient quota',
                'inference_note':'finite unrolled iterations approximate the sparse-coding optimum',
                'step_policy':'learned sigmoid fraction of conservative operator Lipschitz bound',
                'feature_additivity':'exact linear dictionary synthesis within each scale',
                'rgb_decode':'nonlinear; isolated RGB contributions need not sum',
                'encoder_bypass':False, 'prior_checkpoint_loaded':False}


def create_model(config, variant):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; no CPU model fallback')
    torch.manual_seed(config['seed'])
    return ResidualISTAGeneAE(variant, config['model']).cuda()
