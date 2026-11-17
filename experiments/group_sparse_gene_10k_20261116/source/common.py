"""Atomic state files and strict validation for the isolated group-sparse trial."""
from datetime import datetime,timezone
import copy
import json
import os
from pathlib import Path

def now():return datetime.now(timezone.utc).isoformat()

def write_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 partial=path.with_name(path.name+f'.{os.getpid()}.partial')
 partial.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');partial.replace(path)

def append_json(path,value):
 with Path(path).open('a') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')

def validate_config(c):
 t,m=c['training'],c['model']
 if not c['compute']['gpu_only'] or c['compute']['gpus_per_variant']!=2 or c['compute']['max_gpus']>4:raise ValueError('Requires leased two-GPU training; no CPU fallback')
 if t['effective_batch_size']%2 or not 0<t['effective_batch_size']<=500 or not 0<t['micro_batch_size']<=t['effective_batch_size']//2:raise ValueError('Invalid batch')
 if t['max_training_images']!=t['max_updates']*t['effective_batch_size']:raise ValueError('Image/update budget mismatch')
 if len(m['budgets'])!=len(m['budget_allocations']) or len(m['dictionary_sizes'])!=3 or len(m['dictionary_kernels'])!=3:raise ValueError('Invalid scale/rate configuration')
 if sum(m['dictionary_sizes'])!=m['dictionary_size'] or m['dictionary_size']>65535:raise ValueError('Invalid dictionary size')
 for n,allocation in zip(m['budgets'],m['budget_allocations']):
  if len(allocation)!=3 or sum(allocation)!=n or any(k<1 or k>g for k,g in zip(allocation,m['dictionary_sizes'])):raise ValueError('Whole-gene quota mismatch')
 if m['quantization_bits']!=8 or m['expression_max']<=0 or m['expression_downsample']!=2:raise ValueError('Codec only supports uint8 and 2x expression downsampling')
 if m['hard_concrete_temperature']<=0 or not m['hard_concrete_gamma']<0<m['hard_concrete_zeta'] or m['hard_concrete_zeta']<=1:raise ValueError('Invalid hard-concrete settings')
 if not 0<m['initial_gate_probability']<1:raise ValueError('Invalid initial gate probability')
 if c['data']['resolution']%32:raise ValueError('Resolution must be divisible by 32')
 for key in ['group_cap_ramp_updates','gate_ramp_updates','medium_ramp_updates','fine_ramp_updates']:
  if not 0<=m[key][0]<m[key][1]:raise ValueError('Invalid ramp '+key)
 if t['gene_price']<0 or t['byte_price']<0:raise ValueError('Negative complexity price')
 schedule=m.get('rate_schedule')
 if not isinstance(schedule,list) or not schedule or any(type(i)!=int or not 0<=i<len(m['budgets']) for i in schedule) or set(schedule)!=set(range(len(m['budgets']))):raise ValueError('Invalid training rate schedule')
 if t['max_updates']%len(schedule):raise ValueError('Training updates must contain full rate cycles')
 primary=c['protocol'].get('primary_rate_index')
 if type(primary)!=int or not 0<=primary<len(m['budgets']) or m['budgets'][primary]!=c['protocol'].get('primary_gene_cap'):raise ValueError('Invalid primary evaluation rate')
 reuse=c['protocol'].get('reuse_evaluation',{})
 if reuse.get('primary_rate_index')!=primary or reuse.get('city_images')!=128 or reuse.get('deletion_batch_size',0)<1:raise ValueError('Invalid functional reuse evaluation')
 if reuse.get('material_relative_mae_increase',-1)<0 or reuse.get('material_absolute_mae_increase',-1)<0:raise ValueError('Invalid deletion materiality thresholds')
 from codec import maximum_bytes
 actual=[maximum_bytes(allocation,[(c['data']['resolution']//d//2,)*2 for d in [16,8,4]]) for allocation in m['budget_allocations']]
 if actual!=c['protocol']['max_payload_bytes_at_256']:raise ValueError('Stated payload limits disagree with codec')
 return c

def load_config(path):return validate_config(json.loads(Path(path).read_text()))

def resume_compatible(saved,current):
 a,b=copy.deepcopy(saved),copy.deepcopy(current)
 a['training'].pop('micro_batch_size',None);b['training'].pop('micro_batch_size',None)
 return a==b
