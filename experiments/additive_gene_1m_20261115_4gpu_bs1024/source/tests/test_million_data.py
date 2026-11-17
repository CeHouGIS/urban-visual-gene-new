import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from data import ProgressiveRGB, decode_rgb
from prepare_million import assert_multiview_disjoint, fill_cache


def row(city,pano,heading=0):
    return {'id':f'{city}_{pano}_{heading}','image':f'/photos/{city}/{pano}_{heading}.jpg','split':'train'}


def test_distinct_training_views_are_allowed_but_heldout_views_cannot_leak():
    splits={'train':[row('A','p0',0),row('A','p0',180)],
            'validation':[row('A','p1')],'test':[row('A','p2')]}
    assert_multiview_disjoint(splits)
    splits['validation']=[row('A','p0',90)]
    with pytest.raises(ValueError,match='leakage'):
        assert_multiview_disjoint(splits)
    splits['validation']=[row('A','p1')]
    splits['train'].append(splits['train'][0])
    with pytest.raises(ValueError,match='Repeated image'):
        assert_multiview_disjoint(splits)


def fixture_cache(tmp_path):
    (tmp_path/'data').mkdir()
    cache=tmp_path/'cache'
    cache.mkdir()
    rows=[]
    for index in range(3):
        path=tmp_path/f'photo{index}.png'
        Image.fromarray(np.random.default_rng(index).integers(0,256,(13,11,3),dtype=np.uint8)).save(path)
        rows.append({'id':f'photo{index}','image':str(path)})
    (tmp_path/'data/images.train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    rgb=np.lib.format.open_memmap(cache/'train.rgb.npy',mode='w+',dtype=np.uint8,shape=(3,8,8,3))
    flags=np.lib.format.open_memmap(cache/'train.ready.npy',mode='w+',dtype=np.uint8,shape=(3,))
    rgb[:]=255
    flags[:]=0
    rgb.flush()
    flags.flush()
    config={'root':str(tmp_path),'cache':str(cache),'data':{'train':3,'resolution':8,'workers':2,'cache_workers':2}}
    expected=np.stack([decode_rgb(r['image'],8) for r in rows])
    return config,rows,rgb,flags,expected


def test_progressive_reader_uses_published_rows_and_decodes_unpublished_rows(tmp_path):
    config,rows,rgb,flags,expected=fixture_cache(tmp_path)
    rgb[0]=expected[0]
    rgb.flush()
    flags[0]=1
    flags.flush()
    Path(rows[0]['image']).unlink()
    reader=ProgressiveRGB(config)
    try:
        # Row 1 contains poison cached pixels but is unpublished: use the photo.
        np.testing.assert_array_equal(reader[[0,1,2]],expected)
        assert (reader.hits,reader.misses)==(1,2)
        rgb[1]=expected[1]
        rgb.flush()
        flags[1]=1
        flags.flush()
        Path(rows[1]['image']).unlink()
        np.testing.assert_array_equal(reader[[1,0]],expected[[1,0]])
    finally:
        reader.pool.shutdown()


def test_cache_fill_resumes_without_rereading_published_photos(tmp_path):
    config,rows,rgb,flags,expected=fixture_cache(tmp_path)
    rgb[0]=expected[0]
    rgb.flush()
    flags[0]=1
    flags.flush()
    Path(rows[0]['image']).unlink()
    fill_cache(config)
    np.testing.assert_array_equal(flags,np.ones(3,dtype=np.uint8))
    np.testing.assert_array_equal(rgb,expected)
    for row in rows[1:]:
        Path(row['image']).unlink()
    fill_cache(config)
    status=json.loads((tmp_path/'data/cache_progress.json').read_text())
    assert status['status']=='complete' and status['prepared']==3
