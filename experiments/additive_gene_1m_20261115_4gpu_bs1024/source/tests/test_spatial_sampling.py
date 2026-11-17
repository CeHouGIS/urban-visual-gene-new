import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
import pytest
import shapely

from spatial_balance import balanced_allocation, hierarchical_allocation
from spatial_catalog import build_catalog, parse_photo
import spatial_sampling as sampling


def test_capacity_aware_balance_redistributes_shortfalls_exactly():
    quota=balanced_allocation({'small':2,'a':100,'b':100},25,42)
    assert quota['small']==2
    assert sum(quota.values())==25
    assert abs(quota['a']-quota['b'])<=1
    with pytest.raises(ValueError,match='Insufficient'):
        balanced_allocation({'only':3},4,42)
    caps={(continent,country,city):100 for continent in ['A','B']
          for country in ['x','y'] for city in [1,2]}
    quota=hierarchical_allocation(caps,64,42)
    assert set(quota.values())=={8}
    assert quota==hierarchical_allocation(caps,64,42)


def test_splits_keep_global_grids_disjoint_and_leave_training_in_every_city():
    cells=pd.DataFrame([{'continent':continent,'country':country,'ucdb_id':city,
                         'grid_id':f'{continent}-{country}-{city}-{i}'}
                        for continent in ['A','B'] for country in ['x','y']
                        for city in [1,2] for i in range(10)])
    config={'seed':42,'sampling':{'train_cells':40,'validation_cells':8,'test_cells':8}}
    chosen=sampling.select_grid_splits(cells,config)
    assert chosen['grid_id'].nunique()==56
    assert chosen['split'].value_counts().to_dict()=={'train':40,'validation':8,'test':8}
    assert chosen[chosen['split']=='train'].groupby(['continent','country','ucdb_id']).ngroups==8
    pd.testing.assert_frame_equal(chosen,sampling.select_grid_splits(cells,config))


def test_farthest_point_selection_is_distinct_and_covers_cell():
    points=pd.DataFrame({'x':[0,0,999,999,500,501], 'y':[0,999,0,999,500,501],
                         'pano':list('abcdef')})
    selected=sampling.spread_points(points,5,42)
    assert selected['pano'].nunique()==5
    assert {0,999}.issubset(set(selected['x']))
    pd.testing.assert_frame_equal(selected,sampling.spread_points(points,5,42))
    with pytest.raises(ValueError):
        sampling.spread_points(points.iloc[:4],5,42)


@pytest.mark.parametrize('pano',['-AB_ghi-xyz','a_b_c'])
def test_qc_path_parser_handles_panorama_punctuation(pano):
    line=f'google\t/host/root/mnt/nas/photos/Country/City/{pano[0]}/{pano[1]}/{pano}_270.jpg\n'
    assert parse_photo(line,'/host/mnt/nas/photos')==('Country','City',pano,270)
    assert parse_photo('google\t/photos/C/C/a/b/malformed.jpg','/photos') is None
    assert parse_photo('baidu\t/other/ab_0.jpg','/photos') is None


def test_catalog_requires_all_four_quality_passed_views_and_extends(tmp_path):
    first=tmp_path/'keep1.txt'; second=tmp_path/'keep2.txt'
    def lines(pano,headings):
        return ''.join(f'google\t/photos/A/B/{pano[0]}/{pano[1]}/{pano}_{h}.jpg\n' for h in headings)
    first.write_text(lines('ab_full',[0,90,180,270])+lines('ac_partial',[0,90,180]))
    second.write_text(lines('ad_new',[0,90,180,270]))
    config={'root':str(tmp_path),'sampling':{'quality_catalog':str(first),'image_root':'/photos'}}
    catalog=build_catalog(config)
    assert catalog['complete_quartets']==1
    config['sampling']['quality_additional_lists']=[str(second)]
    extended=build_catalog(config)
    assert extended['complete_quartets']==2
    assert set(np.load(extended['entries'][0]['panos_file']))=={'ab_full','ad_new'}
    assert build_catalog(config)['complete_quartets']==2
    second.write_text(second.read_text()+lines('ae_more',[0]))
    with pytest.raises(ValueError,match='source changed'):
        build_catalog(config)


def test_manifest_writes_five_points_and_four_headings_per_grid(tmp_path):
    (tmp_path/'data').mkdir()
    rows=[]
    for split in ['train','validation','test']:
        for i in range(5):
            rows.append(dict(pano=f'{split}{i}',lat=1,lon=1,x=i,y=i,gx=0,gy=0,year=2024,month=1,
                             grid_id=split,ucdb_id=1,continent='A',country='B',city='C',
                             source_country='B',source_city='C',split=split))
    config={'root':str(tmp_path),'sampling':{'image_root':'/photos','headings':[0,90,180,270]},
            'data':{split:20 for split in ['train','validation','test']}}
    assert sampling.write_selected_manifests(pd.DataFrame(rows),config)==config['data']
    all_ids=set()
    for split in config['data']:
        images=pd.read_json(tmp_path/f'data/images.{split}.jsonl',lines=True)
        assert len(images)==20 and images['pano'].nunique()==5
        assert images.groupby('pano')['heading'].apply(set).map(lambda x:x=={0,90,180,270}).all()
        assert all_ids.isdisjoint(images['id'])
        all_ids.update(images['id'])


def test_candidates_enforce_polygon_membership_and_refresh_extended_city(tmp_path,monkeypatch):
    (tmp_path/'data').mkdir()
    meta=tmp_path/'metadata/A/B'; meta.mkdir(parents=True)
    panos=tmp_path/'panos.npy'
    protected=tmp_path/'protected.jsonl'; protected.write_text('')
    inverse=Transformer.from_crs('ESRI:54009','EPSG:4326',always_xy=True)
    xs=[100,200,300,400,500,1500]; ys=[100]*6
    lon,lat=inverse.transform(xs,ys)
    frame=pd.DataFrame({'panoid':[f'ab{i}' for i in range(6)],'lat':lat,'lon':lon,'year':2024,'month':1})
    frame.to_parquet(meta/'panoids.parquet',index=False)
    urban=gpd.GeoDataFrame({'ucdb_id':[1],'city':['B'],'country':['A'],'continent':['Asia']},
                           geometry=[shapely.box(0,0,1000,1000)],crs='ESRI:54009')
    monkeypatch.setattr(sampling,'boundaries',lambda c:urban)
    quality={'source_files':[{'path':'fixture','bytes':1}],
             'entries':[{'source_country':'A','source_city':'B','panos_file':str(panos),'complete_quartets':4}]}
    np.save(panos,frame['panoid'].iloc[:4].to_numpy(dtype=str))
    monkeypatch.setattr(sampling,'build_catalog',lambda c:quality)
    config={'root':str(tmp_path),'sampling':{'grid_size_m':1000,'grid_crs':'ESRI:54009',
           'points_per_cell':5,'metadata_root':str(tmp_path/'metadata'),
           'protected_panoramas_manifest':str(protected)}}
    first=sampling.build_candidates(config)
    assert first['eligible_grids']==0 and first['unique_points_inside_GHSL']==4
    quality['entries'][0]['complete_quartets']=6
    np.save(panos,frame['panoid'].to_numpy(dtype=str))
    second=sampling.build_candidates(config)
    assert second['eligible_grids']==1 and second['unique_points_inside_GHSL']==5
    cells=pd.read_parquet(tmp_path/'data/eligible_grids.parquet')
    assert cells.iloc[0]['grid_id']=='MOLL1K_0_0'
