"""Select panorama-disjoint real photos, then create disposable local RGB arrays.

Only image decoding/resizing runs on CPU. No hashes or feature-file readbacks.
The original verified million-image dataset is read only.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image

from common import load_config, now, write_json


def identify(row):
    stem = Path(row['image']).stem
    pano, heading = stem.rsplit('_', 1)
    suffix = '_' + stem
    if not row['id'].endswith(suffix):
        raise ValueError(f'Cannot derive city/panorama from {row["id"]}')
    return {**row, 'city': row['id'][:-len(suffix)], 'pano': pano, 'heading': int(heading)}


def balanced_quotas(cities, count, seed):
    cities = sorted(cities)
    random.Random(seed).shuffle(cities)
    return {city: count // len(cities) + (i < count % len(cities))
            for i, city in enumerate(cities)}


def select_train(path, cities, count, protected, seed):
    quotas = balanced_quotas(cities, count, seed)
    rng = {city: random.Random(f'{seed}/{city}') for city in cities}
    seen, pools, counts = set(), defaultdict(list), Counter()
    with Path(path).open() as stream:
        for line in stream:
            row = identify(json.loads(line))
            city = row['city']
            key = (city, row['pano'])
            if city not in quotas or key in seen or key in protected:
                continue
            seen.add(key)
            counts[city] += 1
            pool = pools[city]
            if len(pool) < quotas[city]:
                pool.append(row)
            else:
                choice = rng[city].randrange(counts[city])
                if choice < quotas[city]:
                    pool[choice] = row
    if any(len(pools[c]) != quotas[c] for c in cities):
        raise ValueError('Insufficient independent training panoramas')
    rows = [dict(row, split='train') for city in sorted(cities) for row in pools[city]]
    random.Random(seed).shuffle(rows)
    return rows


def select_holdouts(rows, cities, validation_count, test_count, seed):
    vq = balanced_quotas(cities, validation_count, seed + 1)
    tq = balanced_quotas(cities, test_count, seed + 2)
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row['city'], row['pano'])
        if key not in seen and row['city'] in vq:
            groups[row['city']].append(row)
            seen.add(key)
    result = {'validation': [], 'test': []}
    for city in sorted(cities):
        pool = groups[city]
        random.Random(f'{seed}/holdout/{city}').shuffle(pool)
        if len(pool) < vq[city] + tq[city]:
            raise ValueError(f'Insufficient protected holdout panoramas: {city}')
        result['validation'].extend(dict(r, split='validation') for r in pool[:vq[city]])
        result['test'].extend(dict(r, split='test') for r in pool[vq[city]:vq[city]+tq[city]])
    for split in result:
        random.Random(f'{seed}/{split}').shuffle(result[split])
    return result


def assert_disjoint(splits):
    keys = {s: {(r['city'], r['pano']) for r in rows} for s, rows in splits.items()}
    ids = {s: {r['id'] for r in rows} for s, rows in splits.items()}
    names = list(splits)
    for i, name in enumerate(names):
        if len(keys[name]) != len(splits[name]) or len(ids[name]) != len(splits[name]):
            raise ValueError('A split contains repeated photos or panoramas')
        for other in names[i+1:]:
            if keys[name] & keys[other] or ids[name] & ids[other]:
                raise ValueError('Train/validation/test source leakage')


def resize_row(item):
    row, resolution = item
    with Image.open(row['image']) as opened:
        return np.array(opened.convert('RGB').resize((resolution, resolution), Image.Resampling.BICUBIC))


def prepare(config):
    root, source, cache = map(Path, [config['root'], config['source_data'], config['cache']])
    status_path = root / 'data/preparation.json'
    if (root / 'data/READY.json').exists():
        ready = json.loads((root / 'data/READY.json').read_text())
        if ready['config_data'] != config['data'] or ready['seed'] != config['seed']:
            raise ValueError('Existing dataset configuration differs')
        if all((cache / f'{s}.rgb.npy').exists() for s in ['train','validation','test']):
            print('Dataset already prepared', flush=True)
            return
    write_json(status_path, {'status': 'selecting_real_photos', 'updated_utc': now()})
    manifest_paths = {s: root / f'data/images.{s}.jsonl' for s in ['train','validation','test']}
    if all(p.exists() for p in manifest_paths.values()):
        splits = {s: [json.loads(x) for x in p.read_text().splitlines()] for s,p in manifest_paths.items()}
    else:
        if any(p.exists() for p in manifest_paths.values()):
            raise ValueError('Incomplete manifests; inspect interrupted selection before continuing')
        metadata = json.loads((source / 'READY.json').read_text())
        cities = metadata['eligible_cities']
        holdouts = [identify(json.loads(x)) for x in (source / 'images.validation.jsonl').read_text().splitlines()]
        protected = {(r['city'],r['pano']) for r in holdouts}
        splits = select_holdouts(holdouts, cities, config['data']['validation'], config['data']['test'], config['seed'])
        splits['train'] = select_train(source / 'images.train.jsonl', cities, config['data']['train'], protected, config['seed'])
        assert_disjoint(splits)
        for split, rows in splits.items():
            manifest_paths[split].write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    assert_disjoint(splits)
    cache.mkdir(parents=True, exist_ok=True)
    resolution = config['data']['resolution']
    shapes = {}
    for split in ['train','validation','test']:
        rows = splits[split]
        if len(rows) != config['data'][split]:
            raise ValueError(f'Wrong split size: {split}')
        shape = (len(rows), resolution, resolution, 3)
        shapes[split] = list(shape)
        target = cache / f'{split}.rgb.npy'
        if target.exists():
            array = np.load(target, mmap_mode='r')
            if array.shape != shape or array.dtype != np.uint8:
                raise ValueError('Existing RGB cache has wrong shape/dtype')
            continue
        partial = target.with_suffix('.npy.partial')
        array = np.lib.format.open_memmap(partial, mode='w+', dtype=np.uint8, shape=shape)
        with ThreadPoolExecutor(max_workers=config['data']['workers']) as pool:
            for begin in range(0, len(rows), 128):
                items = [(r,resolution) for r in rows[begin:begin+128]]
                for offset, pixels in enumerate(pool.map(resize_row, items)):
                    array[begin+offset] = pixels
                write_json(status_path, {'status':'resizing_rgb','split':split,
                    'prepared':min(begin+128,len(rows)),'total':len(rows),'updated_utc':now()})
        array.flush()
        del array
        partial.replace(target)
    ready = {'status':'ready','seed':config['seed'],'config_data':config['data'],
        'source_dataset':str(source),'dataset_kind':'real_google_street_view_photographs',
        'split_counts':{s:len(r) for s,r in splits.items()},
        'city_counts':{s:dict(Counter(r['city'] for r in rows)) for s,rows in splits.items()},
        'panorama_counts':{s:len(rows) for s,rows in splits.items()},
        'selection':'city-balanced reservoir over unique train panoramas; first available view per pano',
        'split_policy':'entire original validation pool protected from training; holdouts panorama-disjoint',
        'preprocessing':'full image RGB, square bicubic resize, uint8 cache; no crop or augmentation',
        'cache_root':str(cache),'cache_shapes':shapes,'cache_regenerable':True,
        'file_hashing_performed':False,'prior_dataset_integrity':'reuse existing verified source dataset',
        'labels_used':False,'created_utc':now()}
    write_json(root / 'data/READY.json', ready)
    write_json(status_path, {'status':'ready','split_counts':ready['split_counts'],'updated_utc':now()})
    print(json.dumps({'status':'ready','counts':ready['split_counts']}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    try:
        prepare(load_config(args.config))
    except Exception as error:
        cfg=load_config(args.config)
        write_json(Path(cfg['root'])/'data/preparation.json',
                   {'status':'failed','error':str(error),'updated_utc':now()})
        raise
