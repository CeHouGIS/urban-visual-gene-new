import json

import numpy as np
import pytest

from data import indices_for_step
from prepare_data import assert_disjoint, identify, select_holdouts, select_train


def row(city,pano,heading=0):
    return {'id':f'{city}_{pano}_{heading}', 'image':f'/photos/{city}/{pano}_{heading}.jpg','split':'train'}


def test_panorama_identity_preserves_underscores():
    result=identify(row('New_York','pano_with_underscore',90))
    assert (result['city'],result['pano'],result['heading'])==('New_York','pano_with_underscore',90)


def test_reservoir_selection_excludes_other_views_and_protected_sources(tmp_path):
    path=tmp_path/'train.jsonl'
    rows=[row(city,f'p{i}',heading) for city in ['A','B'] for i in range(12) for heading in [0,180]]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    selected=select_train(path,['A','B'],10,{('A','p0'),('B','p0')},42)
    assert len(selected)==10
    assert len({(r['city'],r['pano']) for r in selected})==10
    assert sum(r['city']=='A' for r in selected)==5
    assert all(r['pano']!='p0' for r in selected)
    assert selected==select_train(path,['A','B'],10,{('A','p0'),('B','p0')},42)


def test_holdouts_and_overlap_rejection():
    rows=[identify(row(city,f'h{i}')) for city in ['A','B'] for i in range(8)]
    splits=select_holdouts(rows,['A','B'],6,6,42)
    assert len(splits['validation'])==len(splits['test'])==6
    assert_disjoint(splits)
    splits['test'][0]=splits['validation'][0]
    with pytest.raises(ValueError,match='leakage'):
        assert_disjoint(splits)


def test_batch_indices_replay_epochs_without_losing_boundary_samples():
    batches=[indices_for_step(step,10,4,42) for step in range(5)]
    first_two_epochs=np.concatenate(batches)
    assert sorted(first_two_epochs[:10])==list(range(10))
    assert sorted(first_two_epochs[10:20])==list(range(10))
    assert np.array_equal(batches[3],indices_for_step(3,10,4,42))
