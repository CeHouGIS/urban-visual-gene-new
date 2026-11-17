"""Persistent final stage: await six runs, then compare frozen-code mixes on a GPU."""
import fcntl
import json
import os
from pathlib import Path
import time
import numpy as np
from common import write_json,now
from reference.supervisor import acquire
from diagnose_genes import ci,sheet
ROOT=Path(__file__).resolve().parents[1]


def wait_for_training(config):
 while True:
  if (ROOT/'STOP_REQUESTED').exists():
   write_json(ROOT/'reports/postprocess_status.json',{'status':'stopped','updated_utc':now()});return False
  complete=[]
  for variant in config['variants']:
   p=ROOT/f'runs/{variant}/progress.json'
   if p.exists() and json.loads(p.read_text()).get('status')=='complete':complete.append(variant)
  write_json(ROOT/'reports/postprocess_status.json',{'status':'waiting_for_training','complete_runs':complete,'total_runs':6,'updated_utc':now()})
  if len(complete)==6:return True
  q=ROOT/'queue_status.json'
  if q.exists() and json.loads(q.read_text()).get('status') in ['failed','stopped']:
   write_json(ROOT/'reports/postprocess_status.json',{'status':'training_queue_failed_or_stopped','updated_utc':now()});return False
  time.sleep(10)


def evaluate_model(config,variant):
 import torch
 from torch.nn import functional as F
 from model import create_model
 from data import image_tensor
 from codec import pack,unpack
 import lpips
 torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 out=ROOT/f'reports/trained_controls/{variant}';out.mkdir(parents=True,exist_ok=True)
 saved=torch.load(ROOT/f'runs/{variant}/checkpoint_best.pt',map_location='cpu',weights_only=False)
 m=create_model(config,variant).eval().requires_grad_(False)
 initial_dictionary={s:d.raw.detach().clone() for s,d in m.dictionaries.items()}
 m.load_state_dict(saved['model'],strict=True);m.set_step(saved['step'])
 if variant.startswith('random_'):
  for s,d in m.dictionaries.items():
   if not torch.equal(d.raw,initial_dictionary[s]):raise AssertionError('Frozen random dictionary changed during training')
 for key,v in m.state_dict().items():
  if not torch.equal(v.cpu(),saved['model'][key]):raise AssertionError('State mismatch: '+key)
 metric=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
 cp=np.load(json.loads((ROOT/'data/READY.json').read_text())['confirmation_cache'],mmap_mode='r')
 vp=np.load(Path(config['cache'])/'validation.rgb.npy',mmap_mode='r')
 discovery=json.loads((ROOT/'data/discovery_selection.json').read_text())
 pairs=json.loads((ROOT/'reports/frozen_diagnostics/composition_pairs.json').read_text())['pairs']
 records=[];codec_results=[]
 def synth(s,a):
  return m.dictionaries[s](a) if m.sparse else m.dense_synthesis[s](F.pad(a,(0,0,0,0,0,48-a.shape[1])))
 for rate in range(3):
  m.rate_index=rate
  a={s:[] for s in m.scales};f={s:[] for s in m.scales};recon=[]
  for i in range(0,128,8):
   result=m(image_tensor(cp,np.arange(i,min(i+8,128))));recon.append(result['reconstruction'])
   for s in m.scales:a[s].append(result['activations'][s]);f[s].append(result['decoder_features'][s])
  a={s:torch.cat(v) for s,v in a.items()};f={s:torch.cat(v) for s,v in f.items()};recon=torch.cat(recon)
  if not torch.isfinite(recon).all():raise AssertionError('Invalid reconstruction')
  # Real 256x256 payloads must decode identically to the rounded latent representation.
  total=0;decoded={}
  for s in m.scales:
   k=a[s].shape[-1]*a[s].shape[-2]*config['model']['sparse_per_spatial'][rate]
   blob=pack(a[s][0].cpu().numpy(),'sparse' if m.sparse else 'dense',k);total+=len(blob)
   (out/f'rate{rate}_{s}.bin').write_bytes(blob)
   reloaded=torch.from_numpy(unpack(blob)).cuda()[None]
   torch.testing.assert_close(reloaded,a[s][:1],rtol=0,atol=0)
   decoded[s]=synth(s,reloaded)*m.gates[s]
  expected=config['budget_protocol']['budget_bytes_including_headers'][rate]
  if total!=expected:raise AssertionError('Actual wire budget mismatch')
  replay=m.decoder(decoded);error=float((replay-recon[:1]).abs().max())
  if error>1e-4:raise AssertionError('Code replay regression '+str(error))
  codec_results.append({'rate':rate,'actual_total_bytes':total,'expected_bytes':expected,'rgb_replay_max_error':error,'checkpoint_loaded_strictly':True})
  groups={};controls={}
  if m.sparse:
   mass={s:torch.zeros(m.gene_counts[s],device='cuda') for s in ['medium','fine']}
   for i in range(0,len(discovery),8):
    _,disc,_=m.encode(image_tensor(vp,np.asarray([r['cache_index'] for r in discovery[i:i+8]])))
    for s in mass:mass[s]+=disc[s].float().square().sum((0,2,3))
   for s,energy in mass.items():
    rank=energy.argsort(descending=True);groups[s]=rank[:8].tolist()
    rng=np.random.default_rng(np.random.SeedSequence([int(variant.rsplit('_s',1)[1]),rate,0 if s=='medium' else 1]))
    controls[s]=rng.choice(rank[8:].cpu().numpy(),size=8,replace=False).tolist()
  write_json(out/f'rate{rate}_groups.json',{'selected_groups':groups,'random_control_groups':controls,'discovery_policy':'selected=discovery coefficient RMS top8; random=seeded uniform eight from remaining genes; no heldout-image selection'})
  inside=torch.zeros(256,256,device='cuda');inside[80:176,80:176]=1
  outside=torch.ones(256,256,device='cuda');outside[32:224,32:224]=0
  modes=['all_codes']+(['selected_group','random_group'] if m.sparse else [])
  examples=[]
  for pi,(receiver,donor) in enumerate(pairs):
   orig_a=image_tensor(cp,np.array([receiver]));orig_b=image_tensor(cp,np.array([donor]));base=recon[receiver:receiver+1]
   source_patch=base[:,:,80:176,80:176];donor_patch=orig_b[:,:,80:176,80:176]
   baseline_lp=float(metric(source_patch*2-1,donor_patch*2-1).mean())
   baseline_dist=float(((base-orig_b).abs().mean(1)[0]*inside).sum()/inside.sum())
   for mode in modes:
    row=[('A original',orig_a[0].cpu().numpy()),('B original',orig_b[0].cpu().numpy()),('A reconstruction',base[0].cpu().numpy())]
    for alpha in [.25,.5,1.]:
     mixed={s:f[s][receiver:receiver+1].clone() for s in m.scales}
     for s in ['medium','fine']:
      diff=torch.zeros_like(a[s][receiver:receiver+1]);h,w=diff.shape[-2:]
      genes=slice(None) if mode=='all_codes' else groups[s] if mode=='selected_group' else controls[s]
      diff[:,genes,h//4:3*h//4,w//4:3*w//4]=alpha*(a[s][donor:donor+1,genes,h//4:3*h//4,w//4:3*w//4]-a[s][receiver:receiver+1,genes,h//4:3*h//4,w//4:3*w//4])
      mixed[s]=mixed[s]+synth(s,diff)*m.gates[s]
     y=m.decoder(mixed)
     donor_distance=float(((y-orig_b).abs().mean(1)[0]*inside).sum()/inside.sum())
     outside_change=float(((y-base).abs().mean(1)[0]*outside).sum()/outside.sum())
     lp=float(metric(y[:,:,80:176,80:176]*2-1,donor_patch*2-1).mean())
     records.append({'variant':variant,'rate':rate,'bytes':total,'pair':pi,'mode':mode,'alpha':alpha,'donor_mae_reduction_fraction':(baseline_dist-donor_distance)/max(baseline_dist,1e-8),'donor_lpips_reduction':baseline_lp-lp,'receiver_outside_change_mae':outside_change,'absolute_donor_patch_mae':donor_distance,'absolute_receiver_outside_mae':float(((y-orig_a).abs().mean(1)[0]*outside).sum()/outside.sum()),'whole_image_change_mae':float((y-base).abs().mean())})
     row.append((mode+' a='+str(alpha),y[0].cpu().numpy()))
    if pi<2:examples.append(row)
  sheet(examples,out/f'rate{rate}_mix_examples.png',176)
  del a,f,recon
 write_json(out/'codec_verification.json',codec_results)
 write_json(out/'composition_per_pair.json',records)
 summary=[]
 for rate in range(3):
  for mode in sorted({r['mode'] for r in records}):
   for alpha in [.25,.5,1.]:
    subset=[r for r in records if r['rate']==rate and r['mode']==mode and r['alpha']==alpha]
    summary.append({'variant':variant,'rate':rate,'mode':mode,'alpha':alpha,'n_pairs':len(subset),**{key:ci([r[key] for r in subset]) for key in ['donor_mae_reduction_fraction','donor_lpips_reduction','receiver_outside_change_mae','whole_image_change_mae']}})
 write_json(out/'composition_summary.json',summary)
 write_json(out/'COMPLETE.json',{'status':'complete','checkpoint_step':saved['step'],'updated_utc':now()})
 del m,metric
 torch.cuda.empty_cache()


def compile_report(config):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rates=[];composition=[]
 fig,axes=plt.subplots(1,3,figsize=(13,4))
 for variant in config['variants']:
  result=json.loads((ROOT/f'runs/{variant}/city_confirmation_metrics.json').read_text())
  values=[]
  for rate,r in result['rates'].items():
   row={'variant':variant,'rate':int(rate),**r};rates.append(row);values.append(row)
  for ax,key in zip(axes,['l1','psnr_mean_per_image','lpips']):ax.plot([r['encoded_bytes_per_image']/1024 for r in values],[r[key] for r in values],'-o',label=variant)
  composition.extend(json.loads((ROOT/f'reports/trained_controls/{variant}/composition_summary.json').read_text()))
 for ax,title in zip(axes,['City-holdout L1','City-holdout PSNR','City-holdout LPIPS']):ax.set(title=title,xlabel='Encoded KiB/image',xscale='log');ax.grid(alpha=.2)
 axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(ROOT/'reports/rate_distortion_controls.png',dpi=150);plt.close(fig)
 write_json(ROOT/'reports/rate_distortion_controls.json',rates);write_json(ROOT/'reports/trained_composition_controls.json',composition)
 fig,axes=plt.subplots(1,3,figsize=(14,4))
 for variant in config['variants']:
  for rate,ax in enumerate(axes):
   rs=[r for r in composition if r['variant']==variant and r['rate']==rate and r['mode']=='all_codes']
   ax.plot([r['receiver_outside_change_mae']['mean'] for r in rs],[r['donor_lpips_reduction']['mean'] for r in rs],'-o',label=variant)
   ax.set(title=f'Rate {rate}: full latent transfer',xlabel='Outside change MAE',ylabel='Donor patch LPIPS decrease');ax.grid(alpha=.2)
 axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(ROOT/'reports/composition_controls_tradeoff.png',dpi=150);plt.close(fig)
 from summarize_results import summarize
 summarize()


def main():
 c=json.loads((ROOT/'configs/experiment.json').read_text());lock=(ROOT/'postprocess.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if not wait_for_training(c):return
 while True:
  if (ROOT/'STOP_REQUESTED').exists():return
  leases=acquire(c,1)
  if leases:break
  write_json(ROOT/'reports/postprocess_status.json',{'status':'waiting_for_gpu','updated_utc':now()});time.sleep(10)
 gpu,handle=leases[0];os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid']
 try:
  import torch
  with torch.inference_mode():
   for variant in c['variants']:
    if (ROOT/'STOP_REQUESTED').exists():return
    if (ROOT/f'reports/trained_controls/{variant}/COMPLETE.json').exists():continue
    write_json(ROOT/'reports/postprocess_status.json',{'status':'evaluating_trained_controls','variant':variant,'physical_gpu':gpu,'updated_utc':now()})
    evaluate_model(c,variant)
  compile_report(c)
  write_json(ROOT/'reports/postprocess_status.json',{'status':'complete','updated_utc':now()})
 except Exception as e:
  write_json(ROOT/'reports/postprocess_status.json',{'status':'failed','error':str(e),'updated_utc':now()});raise
 finally:handle.close()
if __name__=='__main__':main()
