"""Sharded feature I/O and accumulated batches for the million-image run."""
from bisect import bisect_right
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import json
import math
import os
from pathlib import Path
import threading
import time

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from .common import writable


def atomic_json(path, value):
    path = writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_live_json(path, attempts=10):
    # The preparer was already running with a legacy non-atomic status writer.
    for attempt in range(attempts):
        try:
            return json.loads(Path(path).read_text())
        except json.JSONDecodeError:
            if attempt == attempts - 1:
                raise
            time.sleep(.1)


class ShardedFeatures(Dataset):
    """Memory-map only a few shards; no monolithic terabyte copy."""
    def __init__(self, index, max_open=8):
        self.index_path = Path(index)
        self.index = json.loads(self.index_path.read_text())
        self.shards = self.index["shards"]
        self.ends = []
        count = 0
        for row in self.shards:
            count += row["records"]
            self.ends.append(count)
        if count != self.index["records"]:
            raise ValueError("Shard counts differ from declared dataset count")
        self.cache = OrderedDict()
        self.max_open = max_open
        self.cache_lock = threading.Lock()

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard = bisect_right(self.ends, index)
        offset = index - (self.ends[shard - 1] if shard else 0)
        with self.cache_lock:
            if shard not in self.cache:
                array = np.load(self.shards[shard]["path"], mmap_mode="r")
                if array.shape[0] != self.shards[shard]["records"]:
                    raise ValueError("Mapped feature shard has wrong record count")
                self.cache[shard] = array
                if len(self.cache) > self.max_open:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(shard)
            array = self.cache[shard]
        return torch.from_numpy(np.array(array[offset], dtype=np.float32, copy=True))


def feature_batches(dataset, order, batch_size, workers=4, pin_memory=False):
    """Threaded reads avoid the container's 64 MiB multiprocessing /dev/shm."""
    iterator = iter(order)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while indices := list(islice(iterator, batch_size)):
            batch = torch.stack(list(pool.map(dataset.__getitem__, indices)))
            yield batch.pin_memory() if pin_memory else batch


class ShuffledShardOrder(Sampler):
    """Shuffle shards and rows, with sequential shard access for NAS throughput."""
    def __init__(self, dataset, seed, epoch, skip=0):
        self.dataset, self.seed, self.epoch, self.skip = dataset, seed, epoch, skip

    def __len__(self):
        return max(0, len(self.dataset) - self.skip)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        skipped = 0
        for shard in rng.permutation(len(self.dataset.shards)):
            n = self.dataset.shards[int(shard)]["records"]
            start = self.dataset.ends[int(shard) - 1] if shard else 0
            rows = rng.permutation(n)
            if skipped + n <= self.skip:
                skipped += n
                continue
            for index in rows[max(0, self.skip - skipped):]:
                yield start + int(index)
            skipped += n


class RepeatingShardOrder(Sampler):
    def __init__(self, dataset, seed, epoch=0, skip=0):
        self.dataset, self.seed, self.epoch, self.skip = dataset, seed, epoch, skip

    def __iter__(self):
        epoch, skip = self.epoch, self.skip
        while True:
            yield from ShuffledShardOrder(self.dataset, self.seed, epoch, skip)
            epoch, skip = epoch + 1, 0


def accumulated_backward(model, features, effective_batch, micro_batch, loss_fn=None):
    """Exact sample-weighted accumulation, including the short final batch."""
    if len(features) != effective_batch:
        raise ValueError("Optimizer updates require the declared effective batch")
    criterion = loss_fn or (lambda result, target:
        torch.nn.functional.mse_loss(result["reconstructed_features"], target))
    loss_total = 0.
    for part in features.split(micro_batch):
        result = model(part)
        loss = criterion(result, part)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite reconstruction loss")
        (loss * (len(part) / effective_batch)).backward()
        loss_total += float(loss.detach()) * len(part) / effective_batch
    return loss_total


def scheduled_lr(config, update):
    base = config["learning_rate"]
    warmup = config["warmup_updates"]
    if update < warmup:
        return base * (update + 1) / max(1, warmup)
    ratio = min(1., (update - warmup) / max(1, config["max_updates"] - warmup))
    floor = config["minimum_learning_rate"]
    return floor + (base - floor) * .5 * (1 + math.cos(math.pi * ratio))
