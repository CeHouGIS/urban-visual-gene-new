"""Snapshot of tested per-atom discrete logistic entropy model."""
import torch
from torch import nn
from torch.nn import functional as F
SYMBOLS=64

class LogisticEntropy(nn.Module):
    """One folded discretized logistic distribution per atom, finite alphabet.

    The lower tail is assigned to symbol 0 and upper tail to symbol 63. All
    interior symbols represent integer bins. There is no free tail or escape.
    """
    def __init__(self, count):
        super().__init__()
        self.location = nn.Parameter(torch.zeros(count))
        self.log_scale = nn.Parameter(torch.zeros(count))

    def probabilities(self):
        edges = torch.arange(SYMBOLS - 1, device=self.location.device).float() + .5
        scale = F.softplus(self.log_scale.float()) + .05
        cdf = torch.sigmoid((edges[None] - self.location.float()[:, None]) / scale[:, None])
        probabilities = torch.cat([cdf[:, :1], cdf[:, 1:] - cdf[:, :-1], 1 - cdf[:, -1:]], 1)
        probabilities = probabilities.clamp_min(1e-9)
        return probabilities / probabilities.sum(1, keepdim=True)

    def rates(self, units, quantized):
        logs = -torch.log2(self.probabilities())
        channel = torch.arange(units.shape[1], device=units.device)[None, :, None, None]
        q = quantized.detach().long()
        # Train the entropy distribution by likelihood on the actual hard codes.
        entropy_fit = logs[channel, q].mean()
        # Piecewise linear surrogate for rounding: exact hard rate forward,
        # interpolated code-length gradient backward. No entropy gradient here.
        costs = logs.detach()
        low = units.detach().floor().long().clamp(0, SYMBOLS - 1)
        high = (low + 1).clamp(max=SYMBOLS - 1)
        fraction = units - low
        soft = costs[channel, low] * (1 - fraction) + costs[channel, high] * fraction
        hard = costs[channel, q]
        code_rate = (hard + soft - soft.detach()).mean()
        return code_rate, entropy_fit
