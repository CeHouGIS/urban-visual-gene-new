"""Select complete GHSL grid bundles and stage a small local RGB subset."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import numpy as np

from common import load_config, now, write_json
from spatial_balance import hierarchical_allocation


def select_grids(rows, target, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row['continent'], row['country'], int(row['ucdb_id'])].append(row)
    quotas = hierarchical_allocation({k: len(v) for k, v in groups.items()}, target, seed)
    chosen = []
    for key in sorted(groups):
        candidates = sorted(groups[key], key=lambda x: x['grid_id'])
        random.Random(f'{seed}/{key}').shuffle(candidates)
        chosen.extend(candidates[:quotas.get(key, 0)])
    return {r['grid_id'] for r in chosen}


def validate_records(records, expected):
    if len(records) != expected:
        raise ValueError(f'Expected {expected} records, got {len(records)}')
    grids = defaultdict(list)
    for row in records:
        grids[row['grid_id']].append(row)
    for grid, rows in grids.items():
        panos = defaultdict(set)
        for row in rows:
            panos[row['pano']].add(row['heading'])
        if len(rows) != 20 or len(panos) != 5 or any(v != {0, 90, 180, 270} for v in panos.values()):
            raise ValueError(f'Incomplete grid bundle: {grid}')
    if len({r['id'] for r in records}) != expected:
        raise ValueError('Duplicate image IDs')
    return set(grids), {r['pano'] for r in records}


def prepare(config):
    root = Path(config['root'])
    source = Path(config['data']['source_root']) / 'data'
    cache = Path(config['cache'])
    if (root / 'data/READY.json').exists():
        print('Existing prepared subset retained', flush=True)
        return
    write_json(root / 'data/preparation.json', {'status': 'selecting_grids', 'updated_utc': now()})
    with (source / 'selected_grids.csv').open() as stream:
        all_grids = list(csv.DictReader(stream))
    split_sets = {}
    summaries = {}
    for split in ['train', 'validation', 'test']:
        count = config['data'][split]
        chosen = select_grids([r for r in all_grids if r['split'] == split], count // 20,
                              f'{config["seed"]}/{split}')
        records, indices = [], []
        with (source / f'images.{split}.jsonl').open() as stream:
            for i, line in enumerate(stream):
                row = json.loads(line)
                if row['grid_id'] in chosen:
                    records.append(row)
                    indices.append(i)
        split_sets[split] = validate_records(records, count)
        for other, (grids, panos) in split_sets.items():
            if other != split and (split_sets[split][0] & grids or split_sets[split][1] & panos):
                raise ValueError('Spatial grid or panorama leakage across splits')
        np.save(root / f'data/{split}.source_rows.npy', np.array(indices, dtype=np.int64))
        with (root / f'data/images.{split}.jsonl').open('w') as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        src = np.load(Path(config['data']['source_cache']) / f'{split}.rgb.npy', mmap_mode='r')
        side = config['data']['resolution']
        partial = cache / f'{split}.rgb.partial.npy'
        dest = np.lib.format.open_memmap(partial, mode='w+', dtype=np.uint8,
                                         shape=(count, side, side, 3))
        write_json(root / 'data/preparation.json', {'status': 'staging_rgb', 'split': split,
                                                   'images': count, 'updated_utc': now()})
        # Read in original sorted row order. No hashing or source-file checksum pass.
        for begin in range(0, count, 100):
            dest[begin:begin + 100] = src[indices[begin:begin + 100]]
        dest.flush()
        del dest
        partial.replace(cache / f'{split}.rgb.npy')
        summaries[split] = {'images': count, 'grids': len(chosen), 'panoramas': len(split_sets[split][1]),
            'continents': dict(Counter(r['continent'] for r in records)),
            'countries': len({r['country'] for r in records}),
            'cities': len({(r['country'], r['ucdb_id']) for r in records})}
        print(json.dumps({'split': split, **summaries[split]}), flush=True)
    write_json(root / 'data/READY.json', {'status': 'ready', 'splits': summaries,
        'source_root': config['data']['source_root'], 'sampling_seed': config['seed'],
        'grid_disjoint': True, 'panorama_disjoint': True,
        'same_cities_can_occur_across_splits': True, 'original_rgb_checksums_performed': False,
        'updated_utc': now()})
    write_json(root / 'data/preparation.json', {'status': 'ready', 'updated_utc': now()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    prepare(load_config(args.config))
