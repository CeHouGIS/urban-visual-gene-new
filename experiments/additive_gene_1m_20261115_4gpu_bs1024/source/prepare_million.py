"""Reuse the original 1M photo manifest and publish a local cache progressively."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import time

import numpy as np

from common import load_config, now, write_json
from data import decode_rgb
from prepare_data import identify


def assert_multiview_disjoint(splits):
    """Multiple headings within train are valid; all cross-split panos are disjoint."""
    keys, ids = {}, {}
    for split,rows in splits.items():
        parsed = [identify(row) for row in rows]
        ids[split] = {r['id'] for r in parsed}
        keys[split] = {(r['city'],r['pano']) for r in parsed}
        if len(ids[split]) != len(rows):
            raise ValueError('Repeated image IDs')
        if split != 'train' and len(keys[split]) != len(rows):
            raise ValueError('Repeated holdout panoramas')
    names = list(splits)
    for i,left in enumerate(names):
        for right in names[i+1:]:
            if keys[left] & keys[right] or ids[left] & ids[right]:
                raise ValueError('Cross-split image/panorama leakage')


def ensure_link(target, source):
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError(f'Unexpected existing link: {target}')
    elif target.exists():
        raise ValueError(f'Refusing to replace existing file: {target}')
    else:
        target.symlink_to(source)


def prepare_manifest(config):
    if config['data'].get('sampling')=='ghsl_grid':
        from spatial_sampling import prepare_spatial_dataset
        return prepare_spatial_dataset(config)
    root,source,cache,reference = map(Path,[config['root'],config['source_data'],config['cache'],config['reference_10k']])
    status = root/'data/preparation.json'
    ready = root/'data/READY.json'
    if ready.exists():
        saved = json.loads(ready.read_text())
        if saved['config_data'] != config['data'] or saved['seed'] != config['seed']:
            raise ValueError('Existing million-image data configuration differs')
        for name in ['train.rgb.npy','train.ready.npy','validation.rgb.npy','test.rgb.npy']:
            if not (cache/name).exists():
                raise ValueError(f'Missing prepared cache: {name}')
        return
    write_json(status,{'status':'checking_manifest_metadata','updated_utc':now()})
    metadata = json.loads((source/'READY.json').read_text())
    if metadata['split_counts']['train'] != config['data']['train']:
        raise ValueError('Source does not contain requested number of distinct photos')
    holdouts = {}
    for split in ['validation','test']:
        path = reference/f'data/images.{split}.jsonl'
        holdouts[split] = [identify(json.loads(line)) for line in path.read_text().splitlines()]
        if len(holdouts[split]) != config['data'][split]:
            raise ValueError('Reference holdout size mismatch')
    assert_multiview_disjoint(holdouts)
    protected = set()
    with (source/'images.validation.jsonl').open() as stream:
        for line in stream:
            row = identify(json.loads(line))
            protected.add((row['city'],row['pano']))
    if any((r['city'],r['pano']) not in protected for rows in holdouts.values() for r in rows):
        raise ValueError('Reference holdout is not in the protected source pool')
    ids,panoramas,cities = set(),set(),Counter()
    with (source/'images.train.jsonl').open() as stream:
        for index,line in enumerate(stream):
            row = identify(json.loads(line))
            key = (row['city'],row['pano'])
            if row['id'] in ids or key in protected or row.get('split') != 'train':
                raise ValueError('Repeated photo, protected panorama, or wrong split in source training manifest')
            ids.add(row['id'])
            panoramas.add(key)
            cities[row['city']] += 1
            if (index+1) % 100000 == 0:
                write_json(status,{'status':'checking_manifest_metadata','rows':index+1,'updated_utc':now()})
    if len(ids) != config['data']['train']:
        raise ValueError('Actual training count differs from requested count')
    counts = {'train':len(ids),**{s:len(rows) for s,rows in holdouts.items()}}
    panorama_counts = {'train':len(panoramas),**{s:len(rows) for s,rows in holdouts.items()}}
    del ids,panoramas
    cache.mkdir(parents=True,exist_ok=True)
    resolution = config['data']['resolution']
    shape = (counts['train'],resolution,resolution,3)
    for name,dimensions in [('train.rgb.npy',shape),('train.ready.npy',(counts['train'],))]:
        path = cache/name
        if path.exists():
            array = np.load(path,mmap_mode='r')
            if array.shape != dimensions or array.dtype != np.uint8:
                raise ValueError('Existing cache shape/dtype mismatch')
        else:
            array = np.lib.format.open_memmap(path,mode='w+',shape=dimensions,dtype=np.uint8)
            if name == 'train.ready.npy':
                array[:] = 0
            array.flush()
        del array
    ensure_link(root/'data/images.train.jsonl',source/'images.train.jsonl')
    for split in ['validation','test']:
        ensure_link(root/f'data/images.{split}.jsonl',reference/f'data/images.{split}.jsonl')
        ensure_link(cache/f'{split}.rgb.npy',Path(json.loads((reference/'configs/experiment.json').read_text())['cache'])/f'{split}.rgb.npy')
    write_json(ready,{'status':'ready','seed':config['seed'],'config_data':config['data'],
        'dataset_kind':'real_google_street_view_photographs','source_dataset':str(source),
        'split_counts':counts,'panorama_counts':panorama_counts,'train_city_counts':dict(cities),
        'split_policy':'reuse original 1M training photos; original 4096 holdout panoramas protected; identical 10K experiment validation/test',
        'views_per_training_panorama':2,'all_training_photos_available':True,
        'cache_policy':'single background writer; read published local RGB or decode original NAS photo on miss',
        'local_cache_bytes_when_complete':int(np.prod(shape)),
        'cache_regenerable':True,'file_hashing_performed':False,'labels_used':False,
        'preprocessing':'RGB full-frame square bicubic resize to 256x256; no augmentation',
        'created_utc':now()})
    write_json(status,{'status':'ready','split_counts':counts,'cache_complete':False,'updated_utc':now()})


def fill_cache(config):
    root,cache = Path(config['root']),Path(config['cache'])
    lock = (cache/'.fill.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pixels = np.load(cache/'train.rgb.npy',mmap_mode='r+')
    ready = np.load(cache/'train.ready.npy',mmap_mode='r+')
    completed = int(np.count_nonzero(ready == 1))
    initial = completed
    started = time.monotonic()
    def publish(block,pool):
        nonlocal completed
        missing = [(i,row) for i,row in block if ready[i] != 1]
        if not missing:
            return
        def resize(item):
            return decode_rgb(item[1]['image'],config['data']['resolution'])
        for (index,row),rgb in zip(missing,pool.map(resize,missing)):
            pixels[index] = rgb
        # Persist RGB before publishing flags, so a crashed fill can safely resume.
        pixels.flush()
        ready[[index for index,row in missing]] = 1
        ready.flush()
        completed += len(missing)
        elapsed = time.monotonic()-started
        speed = (completed-initial)/max(elapsed,1e-6)
        write_json(root/'data/cache_progress.json',{'status':'filling','prepared':completed,'total':len(ready),
            'photos_per_second':round(speed,2),'elapsed_seconds':round(elapsed,2),
            'remaining_seconds_at_current_rate':round((len(ready)-completed)/max(speed,1e-6),1),
            'file_hashing_performed':False,'updated_utc':now()})
    with ThreadPoolExecutor(max_workers=config['data']['cache_workers']) as pool:
        with (root/'data/images.train.jsonl').open() as stream:
            block = []
            for index,line in enumerate(stream):
                if (root/'STOP_REQUESTED').exists():
                    write_json(root/'data/cache_progress.json',{'status':'stopped','prepared':completed,'total':len(ready),'updated_utc':now()})
                    return
                block.append((index,json.loads(line)))
                if len(block) == 512:
                    publish(block,pool)
                    block = []
            if block:
                publish(block,pool)
    write_json(root/'data/cache_progress.json',{'status':'complete','prepared':completed,'total':len(ready),
        'elapsed_seconds':round(time.monotonic()-started,2),'file_hashing_performed':False,'updated_utc':now()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--mode',choices=['manifest','cache'],required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        (prepare_manifest if args.mode == 'manifest' else fill_cache)(config)
    except Exception as error:
        name = 'preparation.json' if args.mode == 'manifest' else 'cache_progress.json'
        write_json(Path(config['root'])/'data'/name,{'status':'failed','error':str(error),'updated_utc':now()})
        raise
