import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from model import create_model
from losses import objective
from codec import pack,unpack
pytestmark=pytest.mark.skipif(os.environ.get('RUN_GPU_TESTS')!='1',reason='Requires leased GPU')
ROOT=Path(__file__).resolve().parents[2]
torch.backends.cudnn.allow_tf32=False
torch.backends.cuda.matmul.allow_tf32=False

@pytest.mark.parametrize('method',['learned','random','dense'])
def test_rates_roundtrip_no_bypass_gradients(method):
 c=json.loads((ROOT/'configs/experiment.json').read_text());m=create_model(c,method+'_s42');m.set_step(1201)
 initial={s:d.raw.detach().clone() for s,d in m.dictionaries.items()}
 for rate in range(3):
  m.rate_index=rate;x=torch.rand(2,3,32,32,device='cuda');out=m(x)
  assert out['reconstruction'].shape==x.shape and torch.isfinite(out['reconstruction']).all()
  total=0;features={}
  for scale,a in out['activations'].items():
   k=c['model']['sparse_per_spatial'][rate]*a.shape[-1]*a.shape[-2]
   blob=pack(a[0].detach().cpu().numpy(),'sparse' if m.sparse else 'dense',k)
   total+=len(blob);reloaded=torch.from_numpy(unpack(blob)).cuda()[None]
   torch.testing.assert_close(reloaded,a[:1].float(),rtol=0,atol=0)
   if m.sparse:
    assert int((a[0]!=0).sum())<=k
    features[scale]=m.dictionaries[scale](reloaded)
   else:
    padded=F.pad(reloaded,(0,0,0,0,0,48-reloaded.shape[1]));features[scale]=m.dense_synthesis[scale](padded)
  assert total==96+6*c['model']['sparse_per_spatial'][rate]*84
  torch.testing.assert_close(m.decoder(features),out['reconstruction'][:1],rtol=2e-5,atol=2e-6)
  loss,_=objective(out,x,m,c);m.zero_grad();loss.backward()
  for name,p in m.named_parameters():
   if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all(),name
   else:assert p.grad is None
 for s,d in m.dictionaries.items():
  if method=='random':torch.testing.assert_close(d.raw,initial[s],rtol=0,atol=0)

def test_dictionary_controls_share_initialization():
 c=json.loads((ROOT/'configs/experiment.json').read_text());a=create_model(c,'learned_s42');b=create_model(c,'random_s42');d=create_model(c,'dense_s42')
 for name,p in a.state_dict().items():torch.testing.assert_close(p,b.state_dict()[name],rtol=0,atol=0)
 for trunk in ['encoder','decoder']:
  for name,p in getattr(a,trunk).state_dict().items():torch.testing.assert_close(p,getattr(d,trunk).state_dict()[name],rtol=0,atol=0)
