"""Experiment-local state and checkpoint helpers; never touch older runs."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import torch


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def append_json(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def checkpoint(path, state):
    path = Path(path)
    temp = path.with_name(path.name + '.partial')
    torch.save(state, temp)
    temp.replace(path)


def load_config(path):
    c = json.loads(Path(path).read_text())
    m, t = c['model'], c['training']
    assert m['grids'] == [1, 2, 4, 8]
    assert m['resolutions'] == [32, 64, 128, 256]
    assert m['dictionary_sizes'] == [64, 128, 128, 256]
    assert m['selections'] == 4 and m['latent_dim'] == 512
    assert c['compute']['gpu_only'] and c['compute']['max_gpus'] == 2
    assert c['compute']['gpus_per_model'] == 2
    assert t['effective_batch_size'] % (2 * t['micro_batch_size']) == 0
    return c
