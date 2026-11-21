"""Full-channel single-scale DINO SAE; linear spatial dictionary synthesis."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from entropy import LogisticEntropy

def tensor_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Dictionary(nn.Module):
    def __init__(self, count=512, channels=64):
        super().__init__()
        if not isinstance(count, int) or count < 1:
            raise ValueError('Dictionary size must be a positive integer')
        self.raw = nn.Parameter(torch.randn(count, channels, 3, 3))

    def atoms(self):
        return self.raw.float() / self.raw.float().flatten(1).norm(dim=1).clamp_min(1e-8)[:, None, None, None]

    def forward(self, activation):
        return F.conv2d(activation, self.atoms().permute(1, 0, 2, 3), padding=1)

    @torch.no_grad()
    def normalize_(self):
        self.raw.copy_(self.atoms())


class Codec(nn.Module):
    def __init__(self, step=.125, atoms=512, channels=64):
        super().__init__()
        if step <= 0:
            raise ValueError('Quantization step must be positive')
        self.step = step
        self.dictionary = Dictionary(atoms, channels)
        self.analysis = nn.Conv2d(channels, atoms, 3, padding=1)
        self.entropy = LogisticEntropy(atoms)
        with torch.no_grad():
            self.dictionary.normalize_()
            self.analysis.weight.copy_(.1 * self.dictionary.atoms().flip(-1, -2))
            self.analysis.bias.fill_(-.05)

    def forward(self, features):
        a = F.relu(self.analysis(features.detach()))
        units = (a.float() / self.step).clamp(0, 63)
        hard = units.round()
        q = units + (hard - units).detach() if self.training else hard
        restored = self.dictionary(q * self.step)
        return restored, hard, units

    def objective(self, features, output, lambda_bpp, image_hw=512):
        restored, hard, units = output
        mse = (restored.float() - features.float()).square().mean()
        d = mse / features.float().square().mean().clamp_min(1e-8)
        rate, entropy_fit = self.entropy.rates(units, hard)
        n = hard[0].numel()
        bpp = rate * n / (image_hw * image_hw)
        loss = d + lambda_bpp * bpp
        report = dict(loss=float(loss.detach()), nmse=float(d.detach()),
                      bits_per_coefficient=float(rate.detach()), bpp=float(bpp.detach()),
                      bits_per_image=float(rate.detach()) * n,
                      nonzero_pct=float((hard > 0).float().mean()) * 100,
                      saturation_pct=float((hard == 63).float().mean()) * 100,
                      entropy_fit=float(entropy_fit.detach()))
        report['active_atoms_per_position'] = report['nonzero_pct'] * hard.shape[1] / 100
        return loss, entropy_fit, report
