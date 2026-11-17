"""CPU-only, inference-blind selection of discovery and unseen-city confirmation."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from common import write_json, now

ROOT=Path(__file__).resolve().parents[1]

def main():
 c=json.loads((ROOT/'configs/experiment.json').read_text());rng=np.random.default_rng(20261116)
 candidates=json.loads((ROOT/'data/city_candidates.json').read_text());buckets={}
 for key,x in candidates.items():buckets.setdefault(x['continent'],[]).append(key)
 for x in buckets.values():rng.shuffle(x)
 selected=[]
 while len(selected)<128:
  for continent in sorted(buckets):
   if not buckets[continent]:continue
   city=candidates[buckets[continent].pop()]
   # One randomly chosen photograph per city: independent city clusters.
   selected.append(city['rows'][int(rng.integers(len(city['rows'])))])
   if len(selected)==128:break
  if not any(buckets.values()) and len(selected)<128:raise ValueError('Insufficient new cities')
 old=[json.loads(line) for s in ['train','validation','test'] for line in (ROOT/f'data/images.{s}.jsonl').read_text().splitlines()]
 checks={f'{field}_disjoint_from_all_old_splits':{x[field] for x in selected}.isdisjoint({x[field] for x in old}) for field in ['ucdb_id','id','pano','grid_id']}
 if not all(checks.values()):raise AssertionError(checks)
 discovery=[];panos=set()
 rows=[json.loads(x) for x in (ROOT/'data/images.validation.jsonl').read_text().splitlines()]
 for i in rng.permutation(len(rows)):
  x=rows[int(i)]
  if x['pano'] not in panos:discovery.append({'cache_index':int(i),**x});panos.add(x['pano'])
  if len(discovery)==64:break
 write_json(ROOT/'data/discovery_selection.json',discovery)
 write_json(ROOT/'data/confirmation_selection.json',{'samples':selected,'checks':checks,'count':128,'unique_cities':len({x['ucdb_id'] for x in selected}),'continents':sorted({x['continent'] for x in selected}),'seed':20261116,'within_city_sampling_frame':'First up to 80 source-manifest records per unseen city; random draw within retained frame, not all city photos.','selection_before_inference':True,'city_support_note':'New cities available in four continents only; Africa/Oceania are not covered in this confirmation set.','source_manifest':c['data']['source_root']+'/data/images.train.jsonl','no_source_checksums':True,'created_utc':now()})
 path=Path('/workspace/urban-visual-gene')/ROOT.name/'cache';path.mkdir(parents=True,exist_ok=True)
 array=np.lib.format.open_memmap(path/'confirmation.rgb.npy',mode='w+',dtype='uint8',shape=(128,256,256,3))
 for i,x in enumerate(selected):
  with Image.open(x['image']) as im:
   im=ImageOps.exif_transpose(im).convert('RGB').resize((256,256),Image.Resampling.BICUBIC)
   array[i]=np.asarray(im)
 array.flush()
 write_json(ROOT/'data/READY.json',{'status':'ready','confirmation_cache':str(path/'confirmation.rgb.npy'),'confirmation_images':128,'city_isolation_checks':checks,'reused_original_data':True,'updated_utc':now()})
 print(json.dumps({'confirmation_images':128,'city_isolation':checks,'continents':sorted({x['continent'] for x in selected})}),flush=True)
if __name__=='__main__':main()
