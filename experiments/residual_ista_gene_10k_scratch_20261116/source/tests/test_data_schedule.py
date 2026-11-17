import json
from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
from common import load_config,resume_compatible
from data import sample_ids,rank_ids,learning_rate,grid_layout

ROOT=Path(__file__).resolve().parents[2]


def test_sample_stream_resume_full_epochs_and_ddp_partition():
    full=sample_ids(9987,10000,512,42)
    np.testing.assert_array_equal(full,np.concatenate([sample_ids(9987,10000,31,42),sample_ids(10018,10000,481,42)]))
    np.testing.assert_array_equal(np.concatenate([rank_ids(9987,10000,256,42,r,2) for r in range(2)]),full[:256])
    assert len(set(epoch:=sample_ids(0,10000,10000,42)))==10000
    assert set(epoch)==set(range(10000))
    assert not np.array_equal(epoch,sample_ids(10000,10000,10000,42))


def test_no_repeated_grid_or_panorama_in_any_global_batch():
    layout=grid_layout()
    reverse={int(image):grid for grid,images in enumerate(layout) for image in images}
    for offset in range(0,31000,256):
        ids=sample_ids(offset,10000,256,42)
        assert len({reverse[int(i)] for i in ids})==256


def test_split_isolation_without_source_file_hashes():
    rows={s:[json.loads(x) for x in (ROOT/f'data/images.{s}.jsonl').read_text().splitlines()]
          for s in ['train','validation','test']}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        for key in ['pano','grid_id','id']:
            assert {x[key] for x in rows[a]}.isdisjoint({x[key] for x in rows[b]})


def test_exact_updates_learning_rate_and_resume_protocol(tmp_path):
    c=load_config(ROOT/'configs/experiment.json')
    t=c['training']
    assert t['max_training_images']==6000*256==1536000
    assert learning_rate(0,t)>0
    assert learning_rate(100*256,t)==pytest.approx(2e-4)
    assert learning_rate(6000*256,t)==pytest.approx(2e-5)
    changed=deepcopy(c);changed['training']['micro_batch_size']=64
    assert resume_compatible(c,changed)
    changed['model']['ista_iterations']+=1
    assert not resume_compatible(c,changed)
    c['training']['max_training_images']=1000000
    p=tmp_path/'invalid.json';p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='budgets disagree'):
        load_config(p)
