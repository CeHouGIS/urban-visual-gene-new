"""Resumable GPU extraction; real source photos, train-only full-dimensional centering."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

from backbone import FrozenDINO
from utils import now, write_json


def configuration():
    return json.loads((Path(__file__).parents[1]/'configs/experiment.json').read_text())


def signature(c):
    files = {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
             for p in Path(c['weights']).iterdir() if p.suffix in ('.json', '.safetensors')}
    manifests = {k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k, v in c['manifests'].items()}
    return hashlib.sha256(json.dumps(dict(files=files, manifests=manifests, resolution=c['resolution'],
        channels=c['channels'], resize='PIL bilinear square, EXIF transpose, RGB', precision='bf16/fp16'),
        sort_keys=True).encode()).hexdigest()


class Photos(Dataset):
    def __init__(self, manifest, indices, resolution):
        self.rows = [json.loads(s) for s in Path(manifest).read_text().splitlines()]
        self.indices, self.resolution = list(indices), resolution

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = self.indices[i]
        with Image.open(self.rows[index]['image']) as image:
            image = ImageOps.exif_transpose(image).convert('RGB').resize(
                (self.resolution, self.resolution), Image.Resampling.BILINEAR)
            value = np.array(image, dtype=np.uint8, copy=True)
        return torch.from_numpy(value).permute(2, 0, 1), index


class FeatureStore:
    """Read immutable extraction shards without a second concatenated feature copy."""
    def __init__(self, c, split):
        self.c = c
        directory = Path(c['local'])/'features'
        names = ['train_0', 'train_1'] if split == 'train' else ['validation']
        self.arrays = [np.load(directory/(n+'.raw.npy'), mmap_mode='r') for n in names]
        self.ends = np.cumsum([len(a) for a in self.arrays])
        stats = np.load(directory/'normalization.npz')
        self.mean = stats['mean'].astype(np.float32)[None, :, None, None]
        self.scale = float(stats['scale'])

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, indices):
        indices = np.asarray(indices, dtype=int).reshape(-1)
        if (indices < 0).any() or (indices >= len(self)).any():
            raise IndexError('Feature index out of range')
        value = np.empty((len(indices), 1024, 32, 32), dtype=np.float32)
        start = 0
        for array, end in zip(self.arrays, self.ends):
            mask = (indices >= start) & (indices < end)
            if mask.any():
                value[mask] = array[indices[mask]-start]
            start = int(end)
        return (value-self.mean)/self.scale


def image_batches(dataset,batch,workers):
    # Threads avoid the container's 64 MiB /dev/shm multiprocessing limit.
    # Only one batch is submitted at a time, keeping prefetch memory bounded.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0,len(dataset),batch):
            items = list(pool.map(dataset.__getitem__,range(start,min(start+batch,len(dataset)))))
            yield torch.stack([p[0] for p in items]),np.array([p[1] for p in items])


def mirror(c, path):
    destination = Path(c['root'])/'features'/path.name
    if destination.exists() and destination.stat().st_size == path.stat().st_size:
        return
    partial = destination.with_suffix(destination.suffix+'.partial')
    shutil.copyfile(path, partial)
    partial.replace(destination)


def queue_mirror(c,path):
    # Disk copies run without a GPU lease, and extraction can continue immediately.
    # Keep a small bounded job set: one process per finished feature shard.
    log = Path(c['root'])/'logs'/('backup_'+path.stem+'.log')
    with log.open('a') as stream:
        child = subprocess.Popen([c['python'],str(Path(__file__).resolve()),'--mirror',str(path)],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),cwd=c['temporary_directory'],
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(Path(c['root'])/'features'/(path.stem+'.backup.json'),
        dict(status='queued',pid=child.pid,source=str(path),updated_utc=now()))


@torch.no_grad()
def extract(c, shard):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; do not fall back to CPU')
    torch.set_num_threads(4)
    root = Path(c['root'])
    local = Path(c['local'])/'features'
    sig = signature(c)
    model = FrozenDINO(c['weights'], c['resolution']).cuda().eval()
    # Benchmark actual source photos; smoke is not part of SAE training.
    sample = Photos(c['manifests']['train'], range(8), c['resolution'])
    image = torch.stack([sample[i][0] for i in range(8)]).cuda().float()/255
    benchmark = []
    for batch in c['extraction']['batch_candidates']:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        durations = []
        try:
            for _ in range(3):
                torch.cuda.synchronize()
                begin = time.monotonic()
                f = model(image[:batch])
                torch.cuda.synchronize()
                durations.append(time.monotonic()-begin)
                if not torch.isfinite(f).all():
                    raise FloatingPointError('Nonfinite DINO features')
            peak = torch.cuda.max_memory_reserved()
            benchmark.append(dict(batch=batch, seconds=float(np.mean(durations[1:])),
                images_per_second=batch/np.mean(durations[1:]), peak_gib=peak/2**30,
                fits=peak < .78*torch.cuda.get_device_properties(0).total_memory))
            del f
        except torch.OutOfMemoryError:
            benchmark.append(dict(batch=batch, fits=False))
    choice = max((r for r in benchmark if r['fits']), key=lambda r:r['images_per_second'])
    batch = choice['batch']
    write_json(root/'reports'/f'extraction_benchmark_{shard}.json', dict(candidates=benchmark, chosen=choice))
    del image
    torch.cuda.empty_cache()
    jobs = [('train_'+str(shard), 'train', shard*5000, (shard+1)*5000)]
    if shard == 1:
        jobs.append(('validation', 'validation', 0, 1000))
    # Export the requested five views immediately, before bulk extraction.
    if shard == 0:
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        fit_ids = np.linspace(0,9999,32,dtype=int)
        visual_train = Photos(c['manifests']['train'],fit_ids,c['resolution'])
        train_rgb = torch.stack([visual_train[i][0] for i in range(len(visual_train))]).cuda().float()/255
        train_f = torch.cat([model(train_rgb[i:i+batch]) for i in range(0,len(visual_train),batch)])
        tokens = train_f.permute(0,2,3,1).reshape(-1,1024)
        torch.manual_seed(c['seed'])
        _,_,basis = torch.pca_lowrank(tokens, q=3, center=True, niter=4)
        center = tokens.mean(0)
        colors = (tokens-center)@basis
        low,high = torch.quantile(colors,.01,dim=0),torch.quantile(colors,.99,dim=0)
        np.savez(root/'reports/visualization_pca.npz', mean=center.cpu().numpy(),
                 basis=basis.cpu().numpy(),low=low.cpu().numpy(),high=high.cpu().numpy(),
                 fit_train_indices=fit_ids,purpose='visualization only; not a feature projection for SAE')
        del train_rgb, train_f, tokens, colors
        data = Photos(c['manifests']['validation'], c['queries'], c['resolution'])
        rgb = torch.stack([data[i][0] for i in range(len(data))])
        raw = torch.cat([model(rgb[i:i+batch].cuda().float()/255).cpu()
                         for i in range(0,len(rgb),batch)])
        np.savez(root/'reports/dino_query_features.npz', features=raw.numpy().astype(np.float16),
                 rgb=rgb.permute(0,2,3,1).numpy(), indices=np.array(c['queries']))
        del raw, rgb
    for name, split, first, last in jobs:
        status_path = root/'features'/(name+'.progress.json')
        progress = json.loads(status_path.read_text()) if status_path.exists() else {}
        if progress and progress['signature'] != sig:
            raise ValueError('Refusing incompatible feature-cache resume')
        done = progress.get('done',0)
        path = local/(name+'.raw.npy')
        if done and not path.exists():
            raise FileNotFoundError('Progress exists without its local feature shard')
        out = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+',
            dtype=np.float16, shape=(last-first, 1024,32,32))
        dataset = Photos(c['manifests'][split], range(first+done,last), c['resolution'])
        loader = image_batches(dataset,batch,c['extraction']['workers'])
        start = time.monotonic()
        for images, indices in loader:
            if (root/'STOP_REQUESTED').exists():
                raise InterruptedError('Stop requested')
            f = model(images.cuda(non_blocking=True).float().div_(255))
            if not torch.isfinite(f).all():
                raise FloatingPointError('Nonfinite extraction output')
            out[indices-first] = f.cpu().numpy().astype(np.float16)
            done += len(images)
            if done % (batch*16) == 0 or done == last-first:
                out.flush()
                report = dict(status='extracting', signature=sig, done=done, total=last-first,
                    split=split, shard=name, seconds=time.monotonic()-start, updated_utc=now())
                write_json(status_path, report)
                print(json.dumps(report), flush=True)
        out.flush()
        del out
        queue_mirror(c,path)
        write_json(status_path, dict(status='complete',signature=sig,done=done,total=last-first,updated_utc=now()))


@torch.no_grad()
def normalize(c):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for normalization/statistics')
    local = Path(c['local'])/'features'
    arrays = [np.load(local/f'train_{i}.raw.npy', mmap_mode='r') for i in range(2)]
    total = 0
    sx = torch.zeros(1024, device='cuda', dtype=torch.float64)
    ss = torch.zeros_like(sx)
    # Exactly all train images. No validation or test data used to fit normalization.
    for a in arrays:
        for start in range(0,len(a),8):
            f = torch.from_numpy(np.array(a[start:start+8],copy=True)).cuda().double()
            sx += f.sum((0,2,3))
            ss += f.square().sum((0,2,3))
            total += len(f)*32*32
    mean = sx/total
    variance = (ss/total-mean.square()).clamp_min(0)
    scale = variance.mean().sqrt()
    if not torch.isfinite(scale) or scale < 1e-8:
        raise FloatingPointError('Degenerate DINO features')
    path = local/'normalization.npz'
    np.savez(path, mean=mean.float().cpu().numpy(), scale=float(scale), variance=variance.float().cpu().numpy(),
             train_images=10000, train_tokens=total)
    mirror(c,path)
    provenance = dict(status='ready', signature=signature(c), backbone='DINOv3 ViT-L/16',
        feature_shape=[1024,32,32], resolution=512, prefix_tokens_excluded=5, frozen=True,
        normalization=c['normalization'], train_images=10000, validation_images=1000,
        weights=c['weights'], pretrained_weights_hashed=False, rms=float(scale),
        note='No learned projection, no PCA reduction, no RGB decoder; raw fp16 features retained.',updated_utc=now())
    write_json(Path(c['root'])/'features/READY.json',provenance)
    write_json(local/'READY.json',provenance)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--shard', type=int, choices=[0,1])
    p.add_argument('--normalize', action='store_true')
    p.add_argument('--mirror')
    args = p.parse_args()
    c = configuration()
    if args.mirror:
        path=Path(args.mirror)
        mirror(c,path)
        write_json(Path(c['root'])/'features'/(path.stem+'.backup.json'),
            dict(status='complete',source=str(path),updated_utc=now()))
    elif args.normalize:
        normalize(c)
    else:
        extract(c,args.shard)
