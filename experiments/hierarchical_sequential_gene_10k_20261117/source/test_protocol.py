"""CPU-only data/index protocol checks: no neural execution or file hashes."""
import json
from pathlib import Path
import numpy as np
from data import grid_layout, sample_ids, rank_ids


def run():
    root=Path(__file__).resolve().parents[1]
    grids=grid_layout()
    assert grids.shape==(500,20)
    stream=sample_ids(0,10000,10000,42)
    assert np.array_equal(np.sort(stream),np.arange(10000))
    lookup=np.empty(10000,dtype=int)
    for i,row in enumerate(grids):
        lookup[row]=i
    for offset in [0,256,9984,10000,19968]:
        batch=sample_ids(offset,10000,256,42)
        assert len(np.unique(lookup[batch]))==256
        a=rank_ids(offset,10000,256,42,0,2)
        b=rank_ids(offset,10000,256,42,1,2)
        assert np.array_equal(np.concatenate([a,b]),batch)
    rows={s:[json.loads(v) for v in (root/f'data/images.{s}.jsonl').read_text().splitlines()]
          for s in ['train','validation','test']}
    for key in ['id','pano','grid_id']:
        sets={s:{r[key] for r in v} for s,v in rows.items()}
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            assert not sets[a]&sets[b]
    result=dict(status='passed',checks=['grid-balanced stream','rank partition','epoch uniqueness',
        'split image/panorama/grid separation'],cpu_neural_execution=False,file_hashes=False)
    (root/'reports/cpu_protocol.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    run()
