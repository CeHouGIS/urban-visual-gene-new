"""CPU data/config/codec verification only: never instantiate a neural network."""
import copy
import json
from pathlib import Path
import struct
import numpy as np
import pytest
from codec import HEADER,maximum_bytes,pack,unpack
from common import load_config,validate_config,resume_compatible
from data import sample_ids,rank_ids,grid_layout
ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('rate',[0,1,2,3])
def test_whole_gene_wire_budget_and_roundtrip(rate):
 c=load_config(ROOT/'configs/experiment.json');total=0
 rng=np.random.default_rng(49)
 for size,g,quota in zip([8,16,32],c['model']['dictionary_sizes'],c['model']['budget_allocations'][rate]):
  codes=np.zeros((g,size,size),dtype=np.float32)
  ids=rng.choice(g,quota,replace=False)
  codes[ids]=rng.integers(1,256,size=(quota,size,size)).astype(np.float32)*(np.float32(4)/255)
  blob=pack(codes);restored=unpack(blob)
  np.testing.assert_allclose(restored,codes,rtol=0,atol=0)
  assert len(blob)==32+quota*(2+size*size)
  assert (restored!=0).any((1,2)).sum()==quota
  total+=len(blob)
 assert total==c['protocol']['max_payload_bytes_at_256'][rate]


def test_empty_codes_cost_headers_only():
 z=np.zeros((256,8,8),dtype='float32');blob=pack(z)
 assert len(blob)==HEADER.size==32
 np.testing.assert_array_equal(unpack(blob),z)


def test_codec_rejects_bad_inputs_and_corrupted_payloads():
 for x in [np.array([1]),np.full((2,2,2),np.nan),np.full((2,2,2),-1.),np.full((2,2,2),5.)]:
  with pytest.raises(ValueError):pack(x)
 x=np.ones((4,2,2),dtype='float32');blob=pack(x)
 for bad in [b'',blob[:-1],b'XXXX'+blob[4:],blob[:32]+b'\x00\x00\x00\x00'+blob[36:]]:
  with pytest.raises(ValueError):unpack(bad)


def test_zero_rounded_gene_is_not_transmitted():
 x=np.zeros((256,2,2),dtype='float32');x[3]=.001
 assert len(pack(x))==32


def test_protocol_resists_invalid_quota_and_false_payload_claim():
 c=load_config(ROOT/'configs/experiment.json')
 for modify in ['quota','bytes','cpu']:
  bad=copy.deepcopy(c)
  if modify=='quota':bad['model']['budget_allocations'][0][0]+=1
  if modify=='bytes':bad['protocol']['max_payload_bytes_at_256'][0]-=1
  if modify=='cpu':bad['compute']['gpu_only']=False
  with pytest.raises(ValueError):validate_config(bad)


def test_resume_and_full_grid_stream():
 c=load_config(ROOT/'configs/experiment.json');a=copy.deepcopy(c)
 a['training']['micro_batch_size']=64;assert resume_compatible(a,c)
 a['model']['expression_max']=8;assert not resume_compatible(a,c)
 layout=grid_layout();mapping={int(k):i for i,row in enumerate(layout) for k in row}
 for offset in [0,9984,19968,222336]:
  ids=sample_ids(offset,10000,256,42)
  assert len({mapping[int(k)] for k in ids})==256
  a=rank_ids(offset,10000,256,42,0,2);b=rank_ids(offset,10000,256,42,1,2)
  np.testing.assert_array_equal(np.concatenate([a,b]),ids)
  np.testing.assert_array_equal(sample_ids(offset+128,10000,128,42),ids[128:])
