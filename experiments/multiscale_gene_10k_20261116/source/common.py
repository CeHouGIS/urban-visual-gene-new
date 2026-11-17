"""Atomic experiment state and validated, independent configuration."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + f'.{os.getpid()}.partial')
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    partial.replace(path)


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def load_config(path):
    c = json.loads(Path(path).read_text())
    t, m, compute = c['training'], c['model'], c['compute']
    if not compute['gpu_only'] or compute['gpus_per_variant'] != 2 or compute['max_gpus'] > 4:
        raise ValueError('Only two-rank GPU training, at most four physical GPUs, is authorized')
    if t['micro_batch_size'] < 1 or t['effective_batch_size'] % compute['gpus_per_variant']:
        raise ValueError('Global batch must partition equally across ranks; positive microbatch required')
    if t['max_training_images'] % compute['gpus_per_variant']:
        raise ValueError('Exact sample budget must be divisible across ranks')
    if sum(m['multiscale_dictionary_sizes']) != m['dictionary_size']:
        raise ValueError('Total dictionary size must be matched across sparse models')
    if sum(m['multiscale_coefficient_budgets']) != m['active_coefficients_per_image']:
        raise ValueError('Sparse models must have equal per-image coefficient budgets')
    if c['data']['train'] % c['data']['photos_per_grid']:
        raise ValueError('Training subset must preserve complete grid bundles')
    return c


def resume_compatible(saved, current):
    # Changing only the microbatch changes the execution partition, not the
    # sample stream, global objective, optimizer updates or training schedule.
    import copy
    before, after = copy.deepcopy(saved), copy.deepcopy(current)
    before['training'].pop('micro_batch_size', None)
    after['training'].pop('micro_batch_size', None)
    return before == after
