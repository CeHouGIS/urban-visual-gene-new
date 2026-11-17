"""Independent immutable protocol, atomic state and strict resume compatibility."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_name(path.name+f'.{os.getpid()}.partial')
    partial.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    partial.replace(path)


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def load_config(path):
    c=json.loads(Path(path).read_text())
    t,m,compute=c['training'],c['model'],c['compute']
    if not compute['gpu_only'] or compute['gpus_per_variant']!=2 or compute['max_gpus']>4:
        raise ValueError('Only two-rank GPU models, at most four physical GPUs')
    if t['effective_batch_size']%2 or t['micro_batch_size']<1:
        raise ValueError('Invalid global/micro batch')
    if t['max_training_images']!=t['max_updates']*t['effective_batch_size']:
        raise ValueError('Exact update and image budgets disagree')
    if sum(m['multiscale_dictionary_sizes'])!=m['single_dictionary_size']:
        raise ValueError('Total gene counts must match')
    if m['ista_iterations']<1 or not 0<m['ista_step_fraction']<1:
        raise ValueError('Invalid proximal inference settings')
    if not 0<m['lambda_initial']<=m['lambda_final']:
        raise ValueError('Invalid sparse penalty schedule')
    return c


def resume_compatible(saved,current):
    import copy
    a,b=copy.deepcopy(saved),copy.deepcopy(current)
    a['training'].pop('micro_batch_size',None)
    b['training'].pop('micro_batch_size',None)
    return a==b
