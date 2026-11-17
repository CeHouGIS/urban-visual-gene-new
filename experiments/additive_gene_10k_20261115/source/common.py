"""Small state helpers. No model imports: safe for a GPU waiting process."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def load_config(path):
    config = json.loads(Path(path).read_text())
    if not config['compute']['gpu_only'] or not 1 <= config['compute']['max_gpus'] <= 2:
        raise ValueError('GPU-only execution with at most two physical GPUs is required')
    t = config['training']
    if t['effective_batch_size'] % t['micro_batch_size']:
        raise ValueError('Effective batch must be divisible by micro batch')
    return config
