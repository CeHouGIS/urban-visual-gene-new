"""Resident uint8 subsets and an exact resumable sample stream."""
from functools import lru_cache
from pathlib import Path
import numpy as np


@lru_cache(maxsize=3)
def epoch_permutation(count, seed, epoch):
    value = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(count)
    value.flags.writeable = False
    return value


def sample_ids(offset, count, size, seed):
    if offset < 0 or count < 1 or size < 1:
        raise ValueError('Invalid sample stream arguments')
    pieces = []
    while size:
        epoch, start = divmod(offset, count)
        take = min(size, count - start)
        pieces.append(epoch_permutation(count, seed, epoch)[start:start + take])
        offset += take
        size -= take
    return np.concatenate(pieces)


def rank_ids(offset, count, size, seed, rank, world):
    if size % world or not 0 <= rank < world:
        raise ValueError('Global batch must partition equally across ranks')
    return np.split(sample_ids(offset, count, size, seed), world)[rank]


def open_pixels(config, split):
    # A 10k training subset is only 1.97 GB uint8 per process. Keeping it resident
    # avoids repeating random reads against the 196 GB mechanical-disk cache.
    value = np.load(Path(config['cache']) / f'{split}.rgb.npy', allow_pickle=False)
    side = config['data']['resolution']
    if value.shape != (config['data'][split], side, side, 3) or value.dtype != np.uint8:
        raise ValueError('Local RGB subset does not match configuration')
    value.flags.writeable = False
    return value


def image_tensor(pixels, indices, device='cuda'):
    import torch
    if not str(device).startswith('cuda'):
        raise ValueError('No CPU model training or inference fallback')
    batch = torch.from_numpy(np.array(pixels[indices], copy=True)).permute(0, 3, 1, 2)
    return batch.to(device=device, dtype=torch.float32).div_(255)


def learning_rate(seen, settings):
    import math
    warmup = settings['warmup_updates'] * settings['effective_batch_size']
    if seen < warmup:
        return settings['learning_rate'] * min(1.0, (seen + settings['effective_batch_size']) / warmup)
    fraction = min(1.0, (seen - warmup) / max(1, settings['max_training_images'] - warmup))
    return settings['minimum_learning_rate'] + 0.5 * (settings['learning_rate'] - settings['minimum_learning_rate']) * (1 + math.cos(math.pi * fraction))
