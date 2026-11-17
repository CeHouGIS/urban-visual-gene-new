"""Run only with a leased GPU; all model computations remain on CUDA."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from common import load_config
from model import create_model
from losses import objective
from codec import pack,unpack
ROOT=Path(__file__).resolve().parents[2]
pytestmark=pytest.mark.skipif(os.environ.get('RUN_GPU_TESTS')!='1',reason='Requires leased CUDA GPU')

def configuration():
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 return load_config(ROOT/'configs/experiment.json')

@pytest.mark.parametrize('method',['learned','random'])
def test_four_whole_gene_caps_quantization_and_codec_replay(method):
 c=configuration();m=create_model(c,method+'_s42').eval();m.set_step(2000)
 x=torch.rand(2,3,64,64,device='cuda')
 for rate in range(4):
  m.rate_index=rate;out=m(x);count,actual_bytes=m.actual_complexity(out)
  assert bool((count<=c['model']['budgets'][rate]).all())
  assert torch.isfinite(out['reconstruction']).all()
  reloaded={};nbytes=0
  for s,quota in zip(m.scales,c['model']['budget_allocations'][rate]):
   a=out['activations'][s]
   assert bool(((a!=0).any((2,3)).sum(1)<=quota).all())
   torch.testing.assert_close(a/4*255,(a/4*255).round(),atol=2e-5,rtol=0)
   assert a.shape[-2]==out['features'][s].shape[-2]//2
   blob=pack(a[0].detach().cpu().numpy());nbytes+=len(blob)
   reloaded[s]=torch.from_numpy(unpack(blob)).cuda()[None]
   torch.testing.assert_close(reloaded[s],a[:1],rtol=0,atol=0)
  assert nbytes==int(actual_bytes[0])
  replay=m.decoder(m.synthesize_codes(reloaded))
  torch.testing.assert_close(replay,out['reconstruction'][:1],rtol=2e-4,atol=3e-5)
  zero={s:torch.zeros_like(v) for s,v in reloaded.items()}
  torch.testing.assert_close(m.decoder(m.synthesize_codes(zero)),m.decoder({s:torch.zeros_like(v[:1]) for s,v in out['features'].items()}),atol=0,rtol=0)

@pytest.mark.parametrize('method',['learned','random'])
def test_end_to_end_gradients_and_frozen_dictionary(method):
 c=configuration();m=create_model(c,method+'_s42');m.set_step(2000)
 before={s:d.raw.detach().clone() for s,d in m.dictionaries.items()}
 optimizer=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001)
 x=torch.rand(3,3,64,64,device='cuda');out=m(x)
 loss,parts=objective(out,x,m,c);loss.backward()
 assert torch.isfinite(loss)
 for name,p in m.named_parameters():
  if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all(),name
  else:assert p.grad is None,name
 for s in m.scales:
  assert m.selectors[s].weight.grad.abs().sum()>0
  assert m.heads[s].weight.grad.abs().sum()>0
 optimizer.step()
 for s,d in m.dictionaries.items():
  if method=='random':torch.testing.assert_close(d.raw,before[s],rtol=0,atol=0)
  else:assert not torch.equal(d.raw,before[s])


def test_initialization_and_deterministic_eval():
 c=configuration();a=create_model(c,'learned_s42');b=create_model(c,'random_s42')
 for name,p in a.state_dict().items():torch.testing.assert_close(p,b.state_dict()[name],rtol=0,atol=0)
 a.set_step(2000);a.eval();x=torch.rand(2,3,64,64,device='cuda');y=a(x);z=a(x)
 for s in a.scales:torch.testing.assert_close(y['activations'][s],z['activations'][s],rtol=0,atol=0)


def test_analytic_hard_concrete_expectation_and_support_gradients():
 c=configuration();m=create_model(c,'learned_s42');m.set_step(2000)
 n=20000;alpha=torch.full((n,8),-2.,device='cuda',requires_grad=True)
 # No truncation: empirical pre-quantization positive rate must match analytic E[L0].
 gate,prob=m.group_gate(alpha,8)
 torch.testing.assert_close((gate.detach()>0).float().mean(),prob.mean(),rtol=0,atol=.012)
 # Upper bound also holds after a cap, up to Monte Carlo sampling error.
 gate,prob=m.group_gate(alpha,2)
 assert float((gate.detach()>0).float().sum(1).mean())<=float(prob.sum(1).mean())+.05
 weights=torch.arange(1,9,device='cuda')[None]
 (gate*weights).sum().backward()
 assert bool((alpha.grad[(gate.detach()==0)]!=0).any()),'Unselected candidate selectors must have exploration gradients'


def test_feature_additivity_and_whole_group_deletion():
 c=configuration();m=create_model(c,'learned_s42').eval();m.set_step(2000)
 out=m(torch.rand(2,3,64,64,device='cuda'))
 for s,a in out['activations'].items():
  left=a.clone();left[:,1::2]=0;right=a-left
  full=m.synthesize_scale(s,a)
  torch.testing.assert_close(m.synthesize_scale(s,left)+m.synthesize_scale(s,right),full,atol=2e-5,rtol=2e-4)
  g=int(a[0].square().sum((1,2)).argmax());removed=a[:1].clone();removed[:,g]=0
  torch.testing.assert_close(m.synthesize_scale(s,a[:1])-m.synthesize_scale(s,removed),m.contribution(out['activations'],s,g)[:1],atol=2e-5,rtol=2e-4)


def test_priority_schedule_and_true_hard_concrete_median():
 c=configuration();m=create_model(c,'learned_s42').eval()
 observed=[]
 for step in range(1,9):
  m.set_step(step);observed.append(c['model']['budgets'][m.rate_index])
 assert observed==[64,64,64,64,32,32,16,8]
 alpha=torch.tensor([[-2.,0.,2.]],device='cuda')
 gate,_=m.group_gate(alpha,3)
 expected=(torch.sigmoid(alpha/c['model']['hard_concrete_temperature'])*1.2-.1).clamp(0,1)
 torch.testing.assert_close(gate,expected)


def test_functional_reuse_audit_covers_all_active_genes(tmp_path):
 from reuse_audit import audit_reuse
 c=configuration();m=create_model(c,'learned_s42').eval().requires_grad_(False);m.set_step(2000)
 samples=json.loads((ROOT/'data/confirmation_selection.json').read_text())['samples'][:3]
 (tmp_path/'data').mkdir();(tmp_path/'data/confirmation_selection.json').write_text(json.dumps({'samples':samples}))
 c['root']=str(tmp_path);destination=tmp_path/'audit';destination.mkdir()
 pixels=np.random.default_rng(7).integers(0,256,size=(3,64,64,3),dtype=np.uint8)
 assert audit_reuse(m,pixels,c,destination,lambda:False)
 summary=json.loads((destination/'functional_reuse_summary.json').read_text())
 records=json.loads((destination/'functional_reuse_deletions.json').read_text())
 assert summary['active_gene_calls']==len(records)==sum(summary['per_gene_image_count'])
 assert 0<len(records)<=3*64
 assert len(summary['per_gene'])==512
 assert summary['city_images']==3 and summary['all_weights_frozen']
 assert {r['image_index'] for r in records}=={0,1,2}
