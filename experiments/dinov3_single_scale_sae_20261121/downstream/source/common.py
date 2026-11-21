"""Configuration, provenance and retrieval filters shared by downstream stages."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone

import numpy as np


def config():
    path=os.environ.get('UVG_DOWNSTREAM_CONFIG')
    return json.loads((Path(path) if path else Path(__file__).parents[1]/'configs/experiment.json').read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_name(path.name+f'.{os.getpid()}.partial')
    partial.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    partial.replace(path)


def fingerprint(c):
    return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()


def rows(c,split):
    return [json.loads(s) for s in Path(c['base']['manifests'][split]).read_text().splitlines()]


def rank_scores(scores,metadata,query,top=3):
    """Exclude query panorama/grid, and return distinct gallery grids."""
    result,grids=[],set()
    q=metadata[query]
    for i in np.argsort(-np.asarray(scores),kind='stable'):
        i=int(i); row=metadata[i]
        if (i==query or row['pano']==q['pano'] or row['grid_id']==q['grid_id'] or row['grid_id'] in grids):
            continue
        grids.add(row['grid_id'])
        result.append(dict(index=i,cosine=float(scores[i]),city=row['city'],grid_id=row['grid_id'],pano=row['pano']))
        if len(result)==top:break
    return result


def phase_indices(sampler,step,batch,sources):
    return sources[step%len(sources)],sampler.indices((step//len(sources))*batch,batch)


def atomic_npy(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_name(path.name+'.partial')
    with partial.open('wb') as f:np.save(f,value)
    partial.replace(path)


def save_checkpoint(path,value):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_name(path.name+'.partial')
    torch.save(value,partial);partial.replace(path)


def mirror_file(local,destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    partial=destination.with_name(destination.name+'.partial')
    shutil.copyfile(local,partial);partial.replace(destination)


def check_upstream_config(c):
    path=Path(c['base']['root'])/'configs/experiment.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=c['parent_config_sha256']:
        raise ValueError('Upstream configuration changed; refusing incompatible continuation')
