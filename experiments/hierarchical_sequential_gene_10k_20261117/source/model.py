"""Four ordered local residual dictionaries. No encoder-to-RGB bypass."""
from contextlib import nullcontext
import torch
from torch import nn
import torch.distributed as dist
from torch.nn import functional as F


def distributed():
    return dist.is_available() and dist.is_initialized()


def up(x):
    return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)


def tiles(x, grid):
    b, c, h, w = x.shape
    assert h == w and h % grid == 0
    p = h // grid
    return x.reshape(b, c, grid, p, grid, p).permute(0, 2, 4, 1, 3, 5).reshape(b*grid*grid, c, p, p)


def untile(x, grid):
    n, c, p, _ = x.shape
    return x.reshape(n//(grid*grid), grid, grid, c, p, p).permute(0, 3, 1, 4, 2, 5).reshape(n//(grid*grid), c, grid*p, grid*p)


def pyramid(x):
    values = [x]
    for _ in range(3):
        values.append(F.avg_pool2d(values[-1], 2))
    return values[::-1]


class ResidualEMA(nn.Module):
    """One shared codebook reused for four residual-quantization selections.

    Statistics from all selections and ranks are pooled before ONE EMA update
    per forward. Frozen/eval books never update, reset or initialize.
    """
    def __init__(self, size, dim, selections, decay, reset_every, dead_threshold):
        super().__init__()
        self.size, self.dim, self.selections = size, dim, selections
        self.decay, self.reset_every, self.dead_threshold = decay, reset_every, dead_threshold
        self.register_buffer('embedding', torch.randn(size, dim) / selections**0.5)
        self.register_buffer('counts', torch.ones(size))
        self.register_buffer('sums', self.embedding.clone())
        self.register_buffer('initialized', torch.tensor(False))
        self.register_buffer('updates', torch.tensor(0, dtype=torch.long))
        self.register_buffer('resets', torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def initialize(self, h):
        if not distributed() or dist.get_rank() == 0:
            ids = torch.arange(self.size, device=h.device) % len(h)
            values = h[ids] / self.selections**0.5
            # A small jitter distinguishes duplicates in tiny smoke batches.
            values = values + torch.randn_like(values) * 0.01
            self.embedding.copy_(values)
        if distributed():
            dist.broadcast(self.embedding, 0)
        self.sums.copy_(self.embedding)
        self.counts.fill_(1)
        self.initialized.fill_(True)

    @torch.no_grad()
    def update(self, counts, sums, candidates):
        if distributed():
            dist.all_reduce(counts)
            dist.all_reduce(sums)
        # Per-forward normalization makes EMA independent of tile/batch count.
        unit = counts.sum().clamp_min(1) / self.size
        counts, sums = counts / unit, sums / unit
        self.counts.lerp_(counts, 1-self.decay)
        self.sums.lerp_(sums, 1-self.decay)
        self.embedding.copy_(self.sums / self.counts.clamp_min(1e-5)[:, None])
        self.updates.add_(1)
        if self.reset_every and int(self.updates) % self.reset_every == 0:
            dead = self.counts < self.dead_threshold
            number = int(dead.sum())
            if number:
                if not distributed() or dist.get_rank() == 0:
                    ids = torch.arange(number, device=candidates.device) % len(candidates)
                    self.embedding[dead] = candidates[ids]
                if distributed():
                    dist.broadcast(self.embedding, 0)
                self.sums[dead] = self.embedding[dead]
                self.counts[dead] = 1
                self.resets.add_(number)

    def forward(self, h):
        h = h.float()
        if self.training and not bool(self.initialized):
            self.initialize(h.detach())
        residual, total = h, torch.zeros_like(h)
        ids, commitments = [], []
        counts = torch.zeros_like(self.counts)
        sums = torch.zeros_like(self.sums)
        # Snapshot is needed because EMA changes the buffer before backward.
        book = self.embedding.detach().clone()
        candidates = []
        with torch.autocast('cuda', enabled=False):
            for _ in range(self.selections):
                distance = residual.detach().square().sum(1, keepdim=True) + book.square().sum(1) - 2*residual.detach() @ book.T
                index = distance.argmin(1)
                selected = F.embedding(index, book)
                commitments.append(F.mse_loss(residual, selected))
                if self.training:
                    counts.add_(torch.bincount(index, minlength=self.size))
                    sums.index_add_(0, index, residual.detach())
                    candidates.append(residual.detach())
                total = total + selected
                residual = residual - selected
                ids.append(index)
        if self.training:
            self.update(counts, sums, torch.cat(candidates))
        straight = h + (total-h).detach()
        return straight, torch.stack(ids, 1), torch.stack(commitments).mean()

    def lookup(self, indices):
        return F.embedding(indices, self.embedding).sum(1)


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(8, cout), nn.SiLU())


class LocalLevel(nn.Module):
    def __init__(self, size, config):
        super().__init__()
        w = config['width']
        self.encoder = nn.Sequential(block(6, w, 2), block(w, w, 2), block(w, 64, 2),
                                     nn.Flatten(), nn.Linear(64*4*4, 512), nn.LayerNorm(512, elementwise_affine=False))
        self.book = ResidualEMA(size, 512, config['selections'], config['ema_decay'],
                                config['ema_reset_every'], config['ema_dead_threshold'])
        self.context = nn.Sequential(block(3, 16, 2), block(16, 24, 2), block(24, 32, 2))
        self.decode4 = block(64, 64)
        self.decode8 = block(64, w)
        self.decode16 = block(w, w)
        self.decode32 = block(w, w)
        self.rgb = nn.Conv2d(w, 3, 3, padding=1)
        nn.init.normal_(self.rgb.weight, 0, 0.001)
        nn.init.zeros_(self.rgb.bias)

    def decode(self, q, parent):
        q = q.reshape(-1, 32, 4, 4)
        v = self.decode4(torch.cat([q.to(parent.dtype), self.context(parent)], 1))
        for layer in [self.decode8, self.decode16, self.decode32]:
            v = layer(F.interpolate(v, scale_factor=2, mode='nearest'))
        return self.rgb(v).float()

    def forward(self, residual, parent):
        h = self.encoder(torch.cat([residual, parent], 1))
        q, indices, commitment = self.book(h)
        return self.decode(q, parent), indices, commitment


class SequentialGene(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.grids = config['grids']
        self.levels = nn.ModuleList([LocalLevel(k, config) for k in config['dictionary_sizes']])
        self.phase = 0

    def set_phase(self, phase):
        self.phase = phase  # 0..3=current training level, 4=joint
        for i, level in enumerate(self.levels):
            enabled = phase == 4 or i == phase
            level.requires_grad_(enabled)
            level.train(self.training and enabled)

    def train(self, mode=True):
        super().train(mode)
        self.set_phase(self.phase)
        return self

    def forward(self, x, last=None):
        if x.device.type != 'cuda':
            raise RuntimeError('GPU-only model execution')
        last = min(self.phase, 3) if last is None else last
        targets = pyramid(x)
        outputs, corrections, parents, indices, commitments = [], [], [], [], []
        previous = None
        for i in range(last+1):
            parent = torch.full_like(targets[0], 0.5) if i == 0 else up(previous)
            residual = targets[i] - parent
            delta, ids, commit = self.levels[i](tiles(residual, self.grids[i]), tiles(parent, self.grids[i]))
            delta = untile(delta, self.grids[i])
            previous = parent + delta
            outputs.append(previous)
            parents.append(parent)
            corrections.append(delta)
            indices.append(ids.reshape(len(x), self.grids[i]**2, -1))
            commitments.append(commit)
        return dict(outputs=outputs, parents=parents, corrections=corrections,
                    indices=indices, commitments=commitments, targets=targets[:last+1])

    def decode_indices(self, indices):
        previous, outputs = None, []
        b = len(indices[0])
        for i, ids in enumerate(indices):
            side = 32*self.grids[i]
            parent = self.levels[0].book.embedding.new_full((b,3,32,32), 0.5) if i == 0 else up(previous)
            assert parent.shape[-1] == side
            q = self.levels[i].book.lookup(ids.reshape(-1, ids.shape[-1]))
            delta = self.levels[i].decode(q, tiles(parent, self.grids[i]))
            previous = parent + untile(delta, self.grids[i])
            outputs.append(previous)
        return outputs


def objective(result, phase, beta):
    selected = range(4) if phase == 4 else [phase]
    recon = torch.stack([F.l1_loss(result['outputs'][i], result['targets'][i]) for i in selected]).mean()
    commit = torch.stack([result['commitments'][i] for i in selected]).mean()
    return recon + beta*commit, recon, commit
