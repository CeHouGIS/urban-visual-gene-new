"""Deterministic batches from regenerable local RGB, without shared-memory workers."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np


@lru_cache(maxsize=2)
def epoch_permutation(count, seed, epoch):
    # A million-element permutation is reused, rather than rebuilt each update.
    result = np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(count)
    result.flags.writeable = False
    return result


def indices_for_step(step, count, batch_size, seed):
    start = step * batch_size
    indices = []
    while len(indices) < batch_size:
        epoch, offset = divmod(start, count)
        permutation = epoch_permutation(count, seed, epoch)
        take = min(batch_size-len(indices),count-offset)
        indices.extend(permutation[offset:offset+take].tolist())
        start += take
    return np.array(indices,dtype=np.int64)


def open_pixels(config, split):
    if split == 'train' and config['data'].get('cache_mode') == 'progressive_rgb':
        return ProgressiveRGB(config)
    pixels = np.load(Path(config['cache']) / f'{split}.rgb.npy',mmap_mode='r')
    side = config['data']['resolution']
    if pixels.shape != (config['data'][split],side,side,3) or pixels.dtype != np.uint8:
        raise ValueError('RGB cache does not match configuration')
    return pixels


def decode_rgb(path, resolution):
    from PIL import Image
    with Image.open(path) as source:
        return np.array(source.convert('RGB').resize((resolution,resolution),Image.Resampling.BICUBIC))


class ProgressiveRGB:
    """Read published cache rows, or decode the original for a cache miss.

    The background cache worker is the only writer. It sets ready[i] only after
    writing the complete RGB row. Training never consumes an unpublished row.
    """
    def __init__(self, config):
        cache = Path(config['cache'])
        self.pixels = np.load(cache/'train.rgb.npy',mmap_mode='r')
        self.ready = np.load(cache/'train.ready.npy',mmap_mode='r')
        self.resolution = config['data']['resolution']
        self.shape = (config['data']['train'],self.resolution,self.resolution,3)
        self.dtype = np.dtype(np.uint8)
        if self.pixels.shape != self.shape or self.pixels.dtype != self.dtype:
            raise ValueError('Progressive RGB cache shape/dtype mismatch')
        if self.ready.shape != (self.shape[0],) or self.ready.dtype != np.uint8:
            raise ValueError('Progressive RGB publication flags mismatch')
        with (Path(config['root'])/'data/images.train.jsonl').open() as stream:
            self.paths = [json.loads(line)['image'] for line in stream]
        if len(self.paths) != len(self):
            raise ValueError('Training manifest and progressive cache disagree')
        self.pool = ThreadPoolExecutor(max_workers=config['data']['workers'])
        self.hits = self.misses = 0
        self.read_seconds = 0.0

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, selection):
        started = time.monotonic()
        if isinstance(selection,slice):
            ids = np.arange(len(self))[selection]
        else:
            ids = np.asarray(selection,dtype=np.int64)
        scalar = ids.ndim == 0
        ids = np.atleast_1d(ids)
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(self)):
            raise IndexError('RGB row selection out of bounds')
        ready = np.array(self.ready[ids],copy=True) == 1
        batch = np.empty((len(ids),*self.shape[1:]),dtype=np.uint8)
        if ready.any():
            batch[ready] = self.pixels[ids[ready]]
        missing = np.flatnonzero(~ready)
        def load(position):
            return decode_rgb(self.paths[int(ids[position])],self.resolution)
        for position,pixels in zip(missing,self.pool.map(load,missing)):
            batch[position] = pixels
        self.hits += int(ready.sum())
        self.misses += len(missing)
        self.read_seconds += time.monotonic()-started
        return batch[0] if scalar else batch

    def stats(self):
        return {'cache_hits':self.hits,'cache_misses':self.misses,
                'rgb_read_seconds':round(self.read_seconds,3)}


def image_tensor(pixels, indices, device):
    import torch
    # Copy only requested rows from local disk; no multiprocessing /dev/shm usage.
    batch = torch.from_numpy(np.array(pixels[indices],copy=True)).permute(0,3,1,2)
    return batch.to(device=device,dtype=torch.float32).div_(255)
