"""Deterministic batches from regenerable local RGB, without shared-memory workers."""
import numpy as np


def indices_for_step(step, count, batch_size, seed):
    start = step * batch_size
    indices = []
    while len(indices) < batch_size:
        epoch, offset = divmod(start, count)
        permutation = np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(count)
        take = min(batch_size-len(indices),count-offset)
        indices.extend(permutation[offset:offset+take].tolist())
        start += take
    return np.array(indices,dtype=np.int64)


def open_pixels(config, split):
    from pathlib import Path
    pixels = np.load(Path(config['cache']) / f'{split}.rgb.npy',mmap_mode='r')
    side = config['data']['resolution']
    if pixels.shape != (config['data'][split],side,side,3) or pixels.dtype != np.uint8:
        raise ValueError('RGB cache does not match configuration')
    return pixels


def image_tensor(pixels, indices, device):
    import torch
    # Copy only requested rows from local disk; no multiprocessing /dev/shm usage.
    batch = torch.from_numpy(np.array(pixels[indices],copy=True)).permute(0,3,1,2)
    return batch.to(device=device,dtype=torch.float32).div_(255)
