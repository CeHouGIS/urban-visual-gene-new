"""GHSL-boundary sampling: balanced continents/countries/cities, 5 points x 4 views."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import time
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
import shapely

from common import load_config, now, write_json
from data import decode_rgb
from fetch_ghsl import download
from spatial_balance import hierarchical_allocation
from spatial_catalog import build_catalog

LAYER='GHSL_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A'


def clean_name(value):
    value=str(value)
    if 'Ã' in value or 'Â' in value:
        try:
            return value.encode('latin1').decode('utf8')
        except (UnicodeEncodeError,UnicodeDecodeError):
            pass
    return value


def continent_of(region,country):
    if region in {'Oceania','Australia and New Zealand'}:
        return 'Oceania'
    if region=='Northern America':
        return 'North America'
    if region=='Latin America and the Caribbean':
        south={'Argentina','Bolivia','Brazil','Chile','Colombia','Ecuador','French Guiana','Guyana',
               'Paraguay','Peru','Suriname','Uruguay','Venezuela'}
        return 'South America' if country in south else 'North America'
    if region=='Sub-Saharan Africa':
        return 'Africa'
    if region=='Europe':
        return 'Europe'
    if region=='Northern Africa and Western Asia':
        return 'Africa' if country in {'Algeria','Egypt','Libya','Morocco','Sudan','Tunisia','Western Sahara'} else 'Asia'
    if region in {'Central and Southern Asia','Eastern and South-Eastern Asia'}:
        return 'Asia'
    raise ValueError(f'Unmapped GHSL region: {region}, {country}')


def boundaries(config):
    archive=download(config['sampling']['ghsl_zip'])
    gpkg=archive.parent/'GHS_UCDB_GLOBE_R2024A.gpkg'
    if not gpkg.exists():
        with zipfile.ZipFile(archive) as source:
            source.extract(gpkg.name,archive.parent)
    fields=['ID_UC_G0','GC_UCN_MAI_2025','GC_CNT_GAD_2025','GC_DEV_USR_2025']
    raw=gpd.read_file(gpkg,layer=LAYER,columns=fields,engine='pyogrio')
    if raw.crs.to_authority()!=('ESRI','54009'):
        raw=raw.to_crs(config['sampling']['grid_crs'])
    raw=raw.rename(columns={'ID_UC_G0':'ucdb_id','GC_UCN_MAI_2025':'city','GC_CNT_GAD_2025':'country','GC_DEV_USR_2025':'region'})
    raw['city']=raw['city'].map(clean_name)
    raw['country']=raw['country'].map(clean_name)
    raw['continent']=[continent_of(region,country) for region,country in zip(raw['region'],raw['country'])]
    invalid=~shapely.is_valid(raw.geometry.to_numpy())
    if invalid.any():
        raw.loc[invalid,'geometry']=shapely.make_valid(raw.loc[invalid,'geometry'].to_numpy())
    return raw


def build_candidates(config):
    root=Path(config['root'])
    settings=config['sampling']
    result_path=root/'data/spatial_candidate_index.json'
    quality=build_catalog(config)
    provenance={'quality_sources':quality['source_files'],
                'city_counts':[[e['source_country'],e['source_city'],e['complete_quartets']] for e in quality['entries']],
                'grid_size_m':settings['grid_size_m'],'grid_crs':settings['grid_crs'],
                'points_per_cell':settings['points_per_cell']}
    if result_path.exists():
        previous=json.loads(result_path.read_text())
        if previous.get('candidate_provenance')==provenance:
            return previous
    audit_path=root/'data/source_city_audit.json'
    prior_audit=json.loads(audit_path.read_text()) if audit_path.exists() else []
    prior_entries={(e['source_country'],e['source_city']):e for e in prior_audit}
    urban=boundaries(config)
    tree=shapely.STRtree(urban.geometry.to_numpy())
    transform=Transformer.from_crs('EPSG:4326',settings['grid_crs'],always_xy=True)
    protected=set()
    with Path(settings['protected_panoramas_manifest']).open() as stream:
        for line in stream:
            protected.add(Path(json.loads(line)['image']).stem.rsplit('_',1)[0])
    tables=[]
    reports=[]
    started=time.monotonic()
    for number,entry in enumerate(quality['entries']):
        country,city=entry['source_country'],entry['source_city']
        target=root/'data/spatial_candidates'/country/f'{city}.parquet'
        target.parent.mkdir(parents=True,exist_ok=True)
        prior=prior_entries.get((country,city),{})
        if target.exists() and prior.get('complete_quartets')==entry['complete_quartets']:
            points=pd.read_parquet(target)
            tables.append(points)
            reports.append({**prior,**entry,'inside_GHSL_points':len(points),'resumed':True})
            continue
        base=Path(settings['metadata_root'])/country/city
        candidates=[base/'panoids.parquet',base/'panoids_all.parquet',base/'panoids.csv',base/'panoids_all.csv']
        metadata=next((p for p in candidates if p.exists()),None)
        if metadata is None:
            reports.append({**entry,'excluded':'no panorama coordinates file'})
            continue
        panos=np.load(entry['panos_file'],allow_pickle=False)
        frame=pd.read_parquet(metadata) if metadata.suffix=='.parquet' else pd.read_csv(metadata,low_memory=False)
        frame=frame[frame['panoid'].isin(panos)&~frame['panoid'].isin(protected)].copy()
        for name in ['lat','lon','year','month']:
            frame[name]=pd.to_numeric(frame[name],errors='coerce') if name in frame else np.nan
        frame=frame[frame['lat'].between(-90,90)&frame['lon'].between(-180,180)]
        frame=frame.sort_values(['year','month'],ascending=False).drop_duplicates('panoid')
        x,y=transform.transform(frame['lon'].to_numpy(),frame['lat'].to_numpy())
        frame['x'],frame['y']=x,y
        query=tree.query(shapely.points(x,y),predicate='intersects')
        points=frame.iloc[query[0]][['panoid','lat','lon','year','month','x','y']].reset_index(drop=True)
        attrs=urban.iloc[query[1]][['ucdb_id','city','country','continent']].reset_index(drop=True)
        points=pd.concat([points,attrs],axis=1).sort_values('ucdb_id').drop_duplicates('panoid')
        points=points.rename(columns={'panoid':'pano'})
        points['source_country'],points['source_city']=country,city
        side=settings['grid_size_m']
        points['gx']=np.floor(points['x']/side).astype(np.int64)
        points['gy']=np.floor(points['y']/side).astype(np.int64)
        temporary=target.with_suffix('.partial')
        points.to_parquet(temporary,index=False)
        temporary.replace(target)
        tables.append(points)
        reports.append({**entry,'coordinate_file':str(metadata),'qc_panos_with_coordinates':len(frame),'inside_GHSL_points':len(points)})
        write_json(root/'data/sampling_progress.json',{'status':'matching_coordinates_to_GHSL','source_cities_completed':number+1,
            'source_cities_total':len(quality['entries']),'current_source_city':f'{country}/{city}',
            'inside_GHSL_points_so_far':sum(len(t) for t in tables),'elapsed_seconds':round(time.monotonic()-started,2),'updated_utc':now()})
    points=pd.concat(tables,ignore_index=True).sort_values(['year','month'],ascending=False).drop_duplicates('pano')
    points['location_lon']=np.rint(points['lon']*1e5).astype(np.int64)
    points['location_lat']=np.rint(points['lat']*1e5).astype(np.int64)
    # Avoid counting historical photos at effectively the same location as five points.
    points=points.drop_duplicates(['location_lon','location_lat']).reset_index(drop=True)
    columns=['gx','gy','ucdb_id','continent','country','city']
    counts=points.groupby(columns,sort=False).size().reset_index(name='candidate_points')
    valid=counts[counts['candidate_points']>=settings['points_per_cell']]
    # One owner per global square even when a square intersects adjacent urban centres.
    cells=valid.sort_values(['candidate_points','ucdb_id'],ascending=[False,True]).drop_duplicates(['gx','gy']).copy()
    cells['grid_id']=[f'MOLL1K_{int(gx)}_{int(gy)}' for gx,gy in zip(cells['gx'],cells['gy'])]
    cells=cells.sort_values(['continent','country','ucdb_id','gx','gy']).reset_index(drop=True)
    points.to_parquet(root/'data/spatial_points.parquet',index=False)
    cells.to_parquet(root/'data/eligible_grids.parquet',index=False)
    counts.to_csv(root/'data/all_grid_urban_components.csv',index=False)
    cells.groupby(['continent','country','ucdb_id','city']).agg(eligible_grids=('grid_id','size')).reset_index().to_csv(root/'data/grid_capacity_by_city.csv',index=False)
    write_json(root/'data/source_city_audit.json',reports)
    result={'status':'complete','eligible_grids':len(cells),'unique_points_inside_GHSL':len(points),
        'continents':cells.groupby('continent').size().to_dict(),'countries':int(cells['country'].nunique()),
        'GHSL_cities':int(cells['ucdb_id'].nunique()),'source_cities':len(quality['entries']),
        'grid_crs':settings['grid_crs'],'grid_size_m':settings['grid_size_m'],
        'source_panorama_QC':'all four views must be in pre-existing keep lists',
        'point_deduplication':'global pano ID, then lon/lat rounded to 1e-5 degree; newest available capture per location',
        'grid_owner':'eligible urban centre with most distinct candidate locations; ties use smallest GHSL ID',
        'candidate_provenance':provenance,
        'file_hashing_performed':False,'updated_utc':now()}
    write_json(result_path,result)
    write_json(root/'data/sampling_progress.json',result)
    return result


def select_grid_splits(cells,config):
    settings=config['sampling']
    sizes={split:settings[f'{split}_cells'] for split in ['train','validation','test']}
    grouped=cells.groupby(['continent','country','ucdb_id'],sort=True).indices
    quotas=hierarchical_allocation({key:len(indices) for key,indices in grouped.items()},sum(sizes.values()),config['seed'])
    remaining={}
    for key,quota in quotas.items():
        pool=list(grouped[key])
        random.Random(f"{config['seed']}/{key}/grids").shuffle(pool)
        remaining[key]=pool[:quota]
    labels={}
    # Leave at least one training grid for every selected city.
    for split in ['validation','test']:
        quotas=hierarchical_allocation({key:max(0,len(ids)-1) for key,ids in remaining.items()},sizes[split],f"{config['seed']}/{split}")
        for key,count in quotas.items():
            for index in remaining[key][:count]:
                labels[index]=split
            remaining[key]=remaining[key][count:]
    for ids in remaining.values():
        for index in ids:
            labels[index]='train'
    selected=cells.iloc[sorted(labels)].copy()
    selected['split']=[labels[index] for index in sorted(labels)]
    if selected['split'].value_counts().to_dict()!=sizes:
        raise AssertionError('Unexpected exact grid split counts')
    if selected['grid_id'].duplicated().any():
        raise AssertionError('A global grid was selected twice')
    return selected.reset_index(drop=True)


def spread_indices(positions,count,seed):
    if len(positions)<count:
        raise ValueError('Cannot obtain five distinct observed points')
    first=random.Random(str(seed)).randrange(len(positions))
    chosen=[first]
    distances=np.sum((positions-positions[first])**2,axis=1)
    while len(chosen)<count:
        distances[chosen]=-1
        selected=int(distances.argmax())
        chosen.append(selected)
        distances=np.minimum(distances,np.sum((positions-positions[selected])**2,axis=1))
    return np.asarray(chosen,dtype=np.int64)


def spread_points(points,count,seed):
    positions=points[['x','y']].to_numpy(dtype=np.float64)
    return points.iloc[spread_indices(positions,count,seed)].copy()


def write_selected_manifests(selected_points,config):
    root=Path(config['root'])
    settings=config['sampling']
    counts=Counter()
    if selected_points['pano'].duplicated().any():
        raise ValueError('Selected panoramas must be globally unique')
    if selected_points.groupby('grid_id')['split'].nunique().max()!=1:
        raise ValueError('A global grid cannot cross dataset splits')
    if not (selected_points.groupby('grid_id').size()==settings.get('points_per_cell',5)).all():
        raise ValueError('Every grid must contain the configured number of points')
    streams={split:(root/f'data/images.{split}.jsonl.partial').open('w') for split in ['train','validation','test']}
    try:
        for point in selected_points.to_dict('records'):
            directory=Path(settings['image_root'])/point['source_country']/point['source_city']/point['pano'][0]/point['pano'][1]
            for heading in settings['headings']:
                row={key:point[key] for key in ['pano','lat','lon','x','y','gx','gy','grid_id','ucdb_id','continent','country','city','source_country','source_city','split']}
                row.update(id=f"{point['source_city']}_{point['pano']}_{heading}",
                    image=str(directory/f"{point['pano']}_{heading}.jpg"),heading=heading,
                    year=int(point['year']) if pd.notna(point['year']) else None,
                    month=int(point['month']) if pd.notna(point['month']) else None)
                streams[point['split']].write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                counts[point['split']]+=1
    finally:
        for stream in streams.values():
            stream.close()
    for split,count in counts.items():
        if count!=config['data'][split]:
            raise AssertionError('Photo count does not match the training configuration')
        (root/f'data/images.{split}.jsonl.partial').replace(root/f'data/images.{split}.jsonl')
    return dict(counts)


def initialize_rgb_cache(config):
    root,cache=Path(config['root']),Path(config['cache'])
    side=config['data']['resolution']
    cache.mkdir(parents=True,exist_ok=True)
    for name,shape in [('train.rgb.npy',(config['data']['train'],side,side,3)),('train.ready.npy',(config['data']['train'],))]:
        path=cache/name
        if path.exists():
            existing=np.load(path,mmap_mode='r')
            if existing.shape!=shape or existing.dtype!=np.uint8:
                raise ValueError('Existing local cache disagrees with this spatial dataset')
        else:
            array=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint8,shape=shape)
            if name.endswith('ready.npy'):
                array[:]=0
            array.flush()
            del array
    for split in ['validation','test']:
        target=cache/f'{split}.rgb.npy'
        shape=(config['data'][split],side,side,3)
        if target.exists():
            existing=np.load(target,mmap_mode='r')
            if existing.shape!=shape or existing.dtype!=np.uint8:
                raise ValueError('Existing holdout RGB cache disagrees with spatial dataset')
            continue
        rows=[json.loads(line) for line in (root/f'data/images.{split}.jsonl').read_text().splitlines()]
        temporary=target.with_suffix('.partial')
        array=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.uint8,shape=shape)
        with ThreadPoolExecutor(max_workers=config['data']['cache_workers']) as pool:
            for begin in range(0,len(rows),256):
                paths=[row['image'] for row in rows[begin:begin+256]]
                for offset,rgb in enumerate(pool.map(lambda path:decode_rgb(path,side),paths)):
                    array[begin+offset]=rgb
                write_json(root/'data/preparation.json',{'status':'preparing_spatial_holdout_RGB','split':split,
                    'prepared':min(begin+256,len(rows)),'total':len(rows),'updated_utc':now()})
        array.flush()
        del array
        temporary.replace(target)


def prepare_spatial_dataset(config):
    root=Path(config['root'])
    settings=config['sampling']
    ready_path=root/'data/READY.json'
    if ready_path.exists():
        saved=json.loads(ready_path.read_text())
        if saved['config_data']!=config['data'] or saved['sampling']!=settings or saved['seed']!=config['seed']:
            raise ValueError('Existing spatial dataset protocol mismatch')
        return
    candidate_info=build_candidates(config)
    required=sum(settings[f'{split}_cells'] for split in ['train','validation','test'])
    if candidate_info['eligible_grids']<required:
        write_json(root/'data/sampling_progress.json',{**candidate_info,'status':'insufficient_eligible_grids','required_grids':required})
        raise ValueError(f"Need {required} distinct eligible grids, have {candidate_info['eligible_grids']}; no duplicates or fewer points will be substituted")
    protocol={'sampling':settings,'config_data':config['data'],'seed':config['seed']}
    receipt=root/'data/sampling_protocol.json'
    if receipt.exists() and json.loads(receipt.read_text())!=protocol:
        raise ValueError('Refusing to pair existing RGB cache with a changed sampling protocol')
    write_json(receipt,protocol)
    cells=pd.read_parquet(root/'data/eligible_grids.parquet')
    points=pd.read_parquet(root/'data/spatial_points.parquet')
    selected=select_grid_splits(cells,config)
    groups=points.groupby(['gx','gy','ucdb_id','country'],sort=False).indices
    positions=points[['x','y']].to_numpy(dtype=np.float64)
    chosen_indices=[]
    for number,grid in enumerate(selected.to_dict('records'),1):
        ids=groups[grid['gx'],grid['gy'],grid['ucdb_id'],grid['country']]
        local=spread_indices(positions[ids],settings['points_per_cell'],f"{config['seed']}/{grid['grid_id']}")
        chosen_indices.extend(ids[local])
        if number%1000==0 or number==len(selected):
            write_json(root/'data/preparation.json',{'status':'selecting_five_points_per_grid',
                'grids_completed':number,'grids_total':len(selected),'updated_utc':now()})
    chosen=points.iloc[chosen_indices].copy().reset_index(drop=True)
    chosen['grid_id']=np.repeat(selected['grid_id'].to_numpy(),settings['points_per_cell'])
    chosen['split']=np.repeat(selected['split'].to_numpy(),settings['points_per_cell'])
    if chosen['pano'].duplicated().any() or not (chosen.groupby('grid_id').size()==settings['points_per_cell']).all():
        raise AssertionError('Selected points are repeated or violate the five-per-grid constraint')
    counts=write_selected_manifests(chosen,config)
    urban=boundaries(config).set_index('ucdb_id')
    side=settings['grid_size_m']
    boxes=shapely.box(selected['gx'].to_numpy()*side,selected['gy'].to_numpy()*side,
                      (selected['gx'].to_numpy()+1)*side,(selected['gy'].to_numpy()+1)*side)
    city_geometries=urban.loc[selected['ucdb_id'],'geometry'].to_numpy()
    grid_geo=gpd.GeoDataFrame(selected,geometry=shapely.intersection(boxes,city_geometries),crs=settings['grid_crs'])
    grid_geo.to_parquet(root/'data/selected_grids.parquet',index=False)
    selected.to_csv(root/'data/selected_grids.csv',index=False)
    chosen.to_parquet(root/'data/sampled_points.parquet',index=False)
    for level in ['continent','country','city']:
        keys=['split','continent']+(['country'] if level in {'country','city'} else [])+(['ucdb_id','city'] if level=='city' else [])
        table=selected.groupby(keys).size().reset_index(name='grids')
        table['points']=table['grids']*settings['points_per_cell']
        table['photos']=table['points']*len(settings['headings'])
        table.to_csv(root/f'data/distribution_by_{level}.csv',index=False)
    write_json(root/'data/preparation.json',{'status':'sampling_complete_preparing_holdouts','split_counts':counts,'updated_utc':now()})
    initialize_rgb_cache(config)
    summary={'status':'ready','dataset_kind':'real_google_street_view_photographs','seed':config['seed'],
        'config_data':config['data'],'sampling':settings,'split_counts':counts,
        'grid_counts':selected['split'].value_counts().to_dict(),'panorama_counts':chosen['split'].value_counts().to_dict(),
        'GHSL_city_counts':selected.groupby('split')['ucdb_id'].nunique().to_dict(),
        'country_counts':selected.groupby('split')['country'].nunique().to_dict(),
        'training_continent_photos':(selected[selected['split']=='train'].groupby('continent').size()*20).to_dict(),
        'eligible_grid_census':candidate_info,'boundary_source':'official GHSL UCDB R2024A V1.2, 2025 urban-centre outlines',
        'grid_rule':'1 km Mollweide equal-area squares, clipped to assigned GHSL urban centre; one owner per global square',
        'split_policy':'entire global square is assigned to exactly one split; four views and five points never cross splits',
        'point_rule':'random seeded first point, then farthest-point selection for the remaining four; unique rounded coordinates',
        'continent_rule':'GHSL UN region/country mapping to six inhabited continents; Russia follows GHSL Europe grouping',
        'cache_policy':'published local RGB rows, NAS decode fallback while a sole writer fills the cache',
        'file_hashing_performed':False,'labels_used':False,'created_utc':now()}
    write_json(ready_path,summary)
    write_json(root/'data/preparation.json',{'status':'ready','split_counts':counts,'cache_complete':False,'updated_utc':now()})
    write_json(root/'data/sampling_progress.json',{'status':'complete','grid_counts':summary['grid_counts'],'split_counts':counts,'updated_utc':now()})
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--mode',choices=['candidates','prepare'],default='candidates')
    args=parser.parse_args()
    (build_candidates if args.mode=='candidates' else prepare_spatial_dataset)(load_config(args.config))
