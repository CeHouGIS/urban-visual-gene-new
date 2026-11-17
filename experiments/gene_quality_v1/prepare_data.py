"""Build a fresh split and DINOv3 cache without training or evaluation labels."""
from pathlib import Path
from collections import OrderedDict, Counter
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/host/mnt/nas/huangyj/GoogleSV/urban_visual_gene')

def signature(image):
    a = np.array(image.convert('L').resize((9, 8), Image.Resampling.BILINEAR))
    return int.from_bytes(np.packbits(a[:, 1:] > a[:, :-1]).tobytes(), 'big')

def main():
    torch.set_num_threads(1)
    output = ROOT / 'data'; output.mkdir(exist_ok=True)
    if (output / 'READY.json').exists():
        print('Prepared dataset already verified; preserving locked split.', flush=True); return
    old = json.loads((ROOT.parent / 'object_centric_pilot/data/dinov3/manifest.json').read_text())
    forbidden = {(r['city'], r['pano']) for r in old['records']}
    forbidden_sha = {r['source_sha256'] for r in old['records']}
    cities = ['Austin', 'Bogota', 'Lagos', 'London', 'Adelaide']
    groups = {c: OrderedDict() for c in cities}
    for part in sorted((SOURCE / 'formal_city_qc').glob('formal_sample_images.part*.tsv')):
        for line in part.open():
            city, name = line.rstrip('\n').split('\t', 1)
            if city not in groups: continue
            pano, heading = Path(name).stem.rsplit('_', 1)
            groups[city].setdefault(pano, {})[int(heading)] = name.replace('/host/root/mnt/nas/', '/host/mnt/nas/')
    candidates, dropped, sizes = [], [], Counter()
    for city in cities:
        entries = list(groups[city].items())
        order = list(range(len(entries))); random.Random('20261020-' + city).shuffle(order)
        quota = 520 if city == 'Adelaide' else 860
        for index in order:
            pano, views = entries[index]
            if (city, pano) in forbidden or set(views) != {0, 90, 180, 270}: continue
            heading = [0, 90, 180, 270][int(hashlib.sha256(pano.encode()).hexdigest()[:8],16) % 4]
            path = Path(views[heading])
            try:
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                if sha in forbidden_sha: continue
                with Image.open(path) as im:
                    im = im.convert('RGB'); size = im.size; sig = signature(im)
                sizes[str(size)] += 1
                candidates.append({'city':city, 'pano':pano, 'heading':heading, 'source_image':str(path),
                                   'id':f'{city}_{pano}_{heading}', 'source_sha256':sha, 'dhash':sig,
                                   'source_size':size, 'cache_start':(index*4+[0,90,180,270].index(heading))*784,
                                   'cache_file':str(SOURCE/f'formal_batchtopk_307city_balanced_3000/dict_{city}_p784.f16.npy')})
            except Exception as e:
                dropped.append({'path':str(path),'reason':str(e)});continue
            if sum(r['city']==city for r in candidates)>=quota:break
        print(json.dumps({'city':city,'candidates':sum(r['city']==city for r in candidates)}),flush=True)
    # Conservative near-duplicate exclusion: keep at most one representative
    # globally before splitting. Similarity is diagnostic, not a scene identity label.
    selected, duplicate_pairs, hashes = [], [], []
    seen_sha = set()
    for r in candidates:
        duplicate = next((j for j,h in enumerate(hashes) if (h ^ r['dhash']).bit_count() <= 4),None)
        if r['source_sha256'] in seen_sha or duplicate is not None:
            duplicate_pairs.append({'removed':r['id'],'kept':selected[duplicate]['id'] if duplicate is not None else 'same_sha'})
            continue
        selected.append(r);hashes.append(r['dhash']);seen_sha.add(r['source_sha256'])
    records=[]
    for city in cities:
        available=[r for r in selected if r['city']==city]
        assert len(available)>=(400 if city=='Adelaide' else 700),(city,len(available))
        for i,r in enumerate(available[:400 if city=='Adelaide' else 700]):
            r['split']='cross_city_test' if city=='Adelaide' else 'train' if i<500 else 'validation' if i<600 else 'in_domain_test'
            r['index']=len(records);records.append(r)
    assert len(records)==3200
    assert len({r['source_sha256'] for r in records})==3200
    (output/'manifest.json').write_text(json.dumps({'seed':20261020,'records':records,'splits':dict(Counter(r['split'] for r in records)),
        'dino_source':'existing ViT-L/16 cache, 1024D; no PCA','geometry':'full-image square warp; original cached preprocessing not rerun',
        'geographic_isolation':False,'geo_limitation':'coordinates not present in ordered source manifest',
        'near_duplicate_threshold':4,'near_duplicate_metric':'64bit horizontal dHash','near_duplicate_policy':'remove before split'},indent=2))
    print('Locked manifest saved before feature cache',flush=True)
    cache=np.lib.format.open_memmap(output/'dino16.npy',mode='w+',dtype=np.float16,shape=(3200,256,1024))
    sources={city:np.load(next(r['cache_file'] for r in records if r['city']==city),mmap_mode='r') for city in cities}
    def extract(r):
        raw=np.asarray(sources[r['city']][r['cache_start']:r['cache_start']+784],dtype=np.float32).copy()
        assert raw.shape==(784,1024) and np.isfinite(raw).all()
        t=torch.from_numpy(raw).reshape(28,28,1024).permute(2,0,1)[None]
        return F.adaptive_avg_pool2d(t,(16,16))[0].permute(1,2,0).reshape(256,1024).numpy().astype(np.float16)
    ready=np.zeros(len(records),dtype=bool)
    with ThreadPoolExecutor(max_workers=8) as pool:
        # Small chunks bound prefetched results and make cache readiness explicit.
        for offset in range(0,len(records),32):
            for i,result in enumerate(pool.map(extract,records[offset:offset+32]),start=offset):
                cache[i]=result;ready[i]=True
            cache.flush();np.save(output/'dino_ready.npy',ready)
            if ready.sum()%128==0:print('feature cache',int(ready.sum()),flush=True)
    cache.flush()
    (output/'manifest.json').write_text(json.dumps({'seed':20261020,'records':records,'splits':dict(Counter(r['split'] for r in records)),
        'dino_source':'existing ViT-L/16 cache, 1024D; no PCA','geometry':'full-image square warp; original cached preprocessing not rerun',
        'geographic_isolation':False,'geo_limitation':'coordinates not present in ordered source manifest',
        'near_duplicate_threshold':4,'near_duplicate_metric':'64bit horizontal dHash','near_duplicate_policy':'remove before split'},indent=2))
    audit={'candidate_count':len(candidates),'unreadable':dropped,'near_duplicates_removed':duplicate_pairs,'size_distribution':dict(sizes),
        'previous_panoramas_quarantined':608,'full_resolution_dino_alignment_recomputed':False,
        'near_duplicate_audit_status':'automatic conservative exclusion; manual review pending',
        'cache_offset_checks':'four ordered headings x 784, no historical QC removals; original backbone not rerun'}
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    (output/'READY.json').write_text(json.dumps({'records':len(records),'dino_shape':list(cache.shape),'status':'automatic_preparation_complete_manual_duplicate_review_pending'}))
    print(json.dumps({'prepared':len(records),'splits':dict(Counter(r['split'] for r in records)),'near_duplicates_removed':len(duplicate_pairs)}),flush=True)

if __name__=='__main__':main()
