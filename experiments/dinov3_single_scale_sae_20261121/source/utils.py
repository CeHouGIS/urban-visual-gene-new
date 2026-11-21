import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + f'.{os.getpid()}.partial')
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    part.replace(path)


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


class GridSampler:
    """Same grid-separated deterministic image stream as the original model."""
    def __init__(self, manifest, seed):
        groups = {}
        for i, line in enumerate(Path(manifest).read_text().splitlines()):
            row = json.loads(line)
            groups.setdefault(row['grid_id'], []).append(i)
        if {len(x) for x in groups.values()} != {20}:
            raise ValueError('Expected complete 5-panorama / 4-view grid bundles')
        self.grids = np.array([groups[k] for k in sorted(groups)])
        self.count = self.grids.size
        self.seed = seed
        self.cached_epoch = None

    def indices(self, offset, count):
        if count > len(self.grids):
            raise ValueError('Batch must fit unique grid cycle')
        pieces = []
        while count:
            epoch, start = divmod(offset, self.count)
            if epoch != self.cached_epoch:
                order = np.random.default_rng(self.seed).permutation(len(self.grids))
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 917]))
                self.permutation = np.stack([r[rng.permutation(len(r))]
                                             for r in self.grids[order]]).T.reshape(-1)
                self.cached_epoch = epoch
            take = min(count, self.count - start)
            pieces.append(self.permutation[start:start + take])
            offset += take
            count -= take
        return np.concatenate(pieces)


def image_tensor(pixels, indices):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; no CPU neural inference/training fallback')
    value = torch.from_numpy(np.array(pixels[indices], copy=True)).permute(0, 3, 1, 2)
    return value.to(device='cuda', dtype=torch.float32).div_(255)
