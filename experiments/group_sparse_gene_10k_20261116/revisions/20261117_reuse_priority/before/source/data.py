"""Reuse existing RGB cache; deterministic sampling separates grid neighbors."""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np


@lru_cache(maxsize=1)
def grid_layout():
    manifest=Path(__file__).resolve().parents[1]/'data/images.train.jsonl'
    rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    groups={}
    for index,row in enumerate(rows):
        groups.setdefault(row['grid_id'],[]).append(index)
    lengths={len(indices) for indices in groups.values()}
    if lengths!={20}:
        raise ValueError('Expected 5 panoramas x 4 views in every grid')
    return np.array([groups[key] for key in sorted(groups)],dtype=np.int64)


@lru_cache(maxsize=3)
def epoch_permutation(count,seed,epoch):
    grids=grid_layout()
    if grids.size!=count:
        raise ValueError('Sample stream must match the training manifest')
    # Identical grid cycle across epochs keeps grid repeats >=500 positions
    # apart, even at epoch boundaries. Image orders change independently.
    grid_order=np.random.default_rng(seed).permutation(len(grids))
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,917]))
    selected=np.stack([row[rng.permutation(len(row))] for row in grids[grid_order]])
    result=selected.T.reshape(-1)
    result.flags.writeable=False
    return result


def sample_ids(offset,count,size,seed):
    if offset<0 or count<1 or size<1:
        raise ValueError('Invalid sample stream arguments')
    pieces=[]
    while size:
        epoch,start=divmod(offset,count)
        take=min(size,count-start)
        pieces.append(epoch_permutation(count,seed,epoch)[start:start+take])
        offset+=take
        size-=take
    return np.concatenate(pieces)


def rank_ids(offset,count,size,seed,rank,world):
    if size%world or not 0<=rank<world or size>len(grid_layout()):
        raise ValueError('Batch must partition equally and fit unique grid cycle')
    return np.split(sample_ids(offset,count,size,seed),world)[rank]


def open_pixels(config,split):
    path=Path(config['cache'])/f'{split}.rgb.npy' if split!='confirmation' else Path(json.loads((Path(config['root'])/'data/READY.json').read_text())['confirmation_cache'])
    value=np.load(path,allow_pickle=False)
    side=config['data']['resolution']
    expected=128 if split=='confirmation' else config['data'][split]
    if value.shape!=(expected,side,side,3) or value.dtype!=np.uint8:
        raise ValueError('RGB cache does not match manifest/configuration')
    value.flags.writeable=False
    return value


def image_tensor(pixels,indices,device='cuda'):
    import torch
    if not str(device).startswith('cuda'):
        raise ValueError('No CPU model training or inference')
    batch=torch.from_numpy(np.array(pixels[indices],copy=True)).permute(0,3,1,2)
    return batch.to(device=device,dtype=torch.float32).div_(255)


def learning_rate(seen,settings):
    import math
    warmup=settings['warmup_updates']*settings['effective_batch_size']
    if seen<warmup:
        return settings['learning_rate']*min(1.,(seen+settings['effective_batch_size'])/warmup)
    fraction=min(1.,(seen-warmup)/max(1,settings['max_training_images']-warmup))
    return settings['minimum_learning_rate']+.5*(settings['learning_rate']-settings['minimum_learning_rate'])*(1+math.cos(math.pi*fraction))
