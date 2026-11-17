"""Find full, previously QC-passed panorama quartets without rereading images."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from common import load_config, now, write_json


def parse_photo(line, image_root):
    parts=line.rstrip('\n').split('\t',1)
    if len(parts)!=2:
        return None
    path=parts[1].replace('/host/root/mnt/nas/','/host/mnt/nas/')
    prefix=str(image_root).rstrip('/')+'/'
    if not path.startswith(prefix):
        return None
    relative=path[len(prefix):].split('/')
    if len(relative)!=5:
        return None
    country,city,first,second,filename=relative
    if '_' not in Path(filename).stem:
        return None
    pano,angle=Path(filename).stem.rsplit('_',1)
    if angle not in {'0','90','180','270'} or len(pano)<2:
        return None
    if first!=pano[0] or second!=pano[1] or Path(filename).suffix.lower()!='.jpg':
        return None
    return country,city,pano,int(angle)


def build_catalog(config):
    root=Path(config['root'])
    settings=config['sampling']
    directory=root/'data/quality_quartets'
    directory.mkdir(parents=True,exist_ok=True)
    index_path=directory/'index.json'
    source=Path(settings['quality_catalog'])
    groups=defaultdict(dict)
    counts={'lines':0,'google_rows':0,'unparsed_or_other_dataset':0}
    sources=[source]+[Path(path) for path in settings.get('quality_additional_lists',[])]
    completed_sources=[]
    if index_path.exists():
        index=json.loads(index_path.read_text())
        if index['source']!=str(source) or index['source_bytes']!=source.stat().st_size:
            raise ValueError('Existing quartet catalog has another source')
        completed_sources=index.get('source_files',[{'path':str(source),'bytes':index['source_bytes']}])
        finished={item['path']:item['bytes'] for item in completed_sources}
        for path in sources:
            if str(path) in finished and path.stat().st_size!=finished[str(path)]:
                raise ValueError(f'Existing quality source changed: {path}')
        sources=[path for path in sources if str(path) not in finished]
        if not sources:
            return index
        for entry in index['entries']:
            groups[entry['source_country'],entry['source_city']]={str(pano):15 for pano in np.load(entry['panos_file'],allow_pickle=False)}
        counts={key:index[key] for key in counts}
    started=time.monotonic()
    for current_source in sources:
      with current_source.open() as stream:
        for line in stream:
            counts['lines']+=1
            parsed=parse_photo(line,settings['image_root'])
            if parsed is None:
                counts['unparsed_or_other_dataset']+=1
            else:
                country,city,pano,angle=parsed
                pool=groups[country,city]
                pool[pano]=pool.get(pano,0)|(1<<(angle//90))
                counts['google_rows']+=1
            if counts['lines']%250000==0:
                write_json(root/'data/sampling_progress.json',{'status':'indexing_existing_quality_lists',**counts,
                    'source_cities_seen':len(groups),'elapsed_seconds':round(time.monotonic()-started,2),'updated_utc':now()})
      completed_sources.append({'path':str(current_source),'bytes':current_source.stat().st_size})
    entries=[]
    for (country,city),pool in sorted(groups.items()):
        complete=sorted(pano for pano,mask in pool.items() if mask==15)
        if not complete:
            continue
        target=directory/country/f'{city}.panos.npy'
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix('.partial')
        with temporary.open('wb') as stream:
            np.save(stream,np.array(complete),allow_pickle=False)
        temporary.replace(target)
        entries.append({'source_country':country,'source_city':city,'panos_file':str(target),
                        'complete_quartets':len(complete),'all_seen_panoramas':len(pool)})
    result={'status':'complete','source':str(source),'source_bytes':source.stat().st_size,'source_files':completed_sources,
        'source_cities':len(entries),'source_countries':len({e['source_country'] for e in entries}),
        'complete_quartets':sum(e['complete_quartets'] for e in entries),'entries':entries,**counts,
        'quality_policy':'all four directions must be in existing QC keep list; no image/file hashing',
        'elapsed_seconds':round(time.monotonic()-started,2),'updated_utc':now()}
    write_json(index_path,result)
    write_json(root/'data/sampling_progress.json',{key:value for key,value in result.items() if key!='entries'})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    build_catalog(load_config(args.config))
