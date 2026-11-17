"""Frozen-code interventions and compositional transfer; no image-quality reselection."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
from common import write_json,now
from reference.supervisor import acquire
ROOT=Path(__file__).resolve().parents[1]

def image(value):
 a=np.asarray(value)
 if a.shape[0]==3:a=a.transpose(1,2,0)
 return Image.fromarray(np.rint(np.clip(a,0,1)*255).astype('uint8'))

def sheet(rows,path,side=224):
 canvas=Image.new('RGB',(len(rows[0])*side,len(rows)*(side+28)),'white');draw=ImageDraw.Draw(canvas)
 for i,row in enumerate(rows):
  for j,(label,arr) in enumerate(row):
   canvas.paste(image(arr).resize((side,side),Image.Resampling.BICUBIC),(j*side,i*(side+28)+28))
   draw.text((j*side+4,i*(side+28)+7),label,fill='black')
 canvas.save(path)

def ci(x,seed=20261116):
 x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
 if len(x)<2:return {'mean':float(x.mean()) if len(x) else None,'ci95':None,'n':len(x)}
 rng=np.random.default_rng(seed);means=x[rng.integers(len(x),size=(2000,len(x)))].mean(1)
 return {'mean':float(x.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'n':len(x)}

def execute(config):
 import torch
 from torch.nn import functional as F
 from base_model import ResidualISTAGeneAE
 from data import image_tensor
 from metrics import ssim_per_image
 torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 out=ROOT/'reports/frozen_diagnostics';out.mkdir(exist_ok=True)
 base=Path(config['reference_root']);saved=torch.load(base/'runs/multiscale_residual_ista/checkpoint_best.pt',map_location='cpu',weights_only=False)
 m=ResidualISTAGeneAE(saved['variant'],saved['config']['model']).cuda().eval().requires_grad_(False)
 m.load_state_dict(saved['model'],strict=True);m.set_step(saved['step'])
 for name,value in m.state_dict().items():
  if not torch.equal(value.cpu(),saved['model'][name]):raise AssertionError('Checkpoint tensor mismatch: '+name)
 known=np.load(Path(config['cache'])/'test.rgb.npy',mmap_mode='r');known_tensor=torch.from_numpy(np.array(known[199:200],copy=True)).permute(0,3,1,2).cuda().float()/255
 known_pred=m(known_tensor)['reconstruction'];known_psnr=float(-10*torch.log10((known_pred-known_tensor).square().mean()))
 write_json(out/'checkpoint_verification.json',{'all_state_tensors_equal_checkpoint':True,'known_test_index':199,'known_reconstruction_psnr':known_psnr,'checkpoint_step':saved['step'],'minimum_expected_known_psnr':25.0})
 if known_psnr<25:raise AssertionError('Known-photo reconstruction regression: '+str(known_psnr))
 disc=json.loads((ROOT/'data/discovery_selection.json').read_text());confirm=json.loads((ROOT/'data/confirmation_selection.json').read_text())['samples']
 vp=np.load(Path(config['cache'])/'validation.rgb.npy',mmap_mode='r');cp=np.load(json.loads((ROOT/'data/READY.json').read_text())['confirmation_cache'],mmap_mode='r')
 def status(stage,**kw):write_json(out/'progress.json',{'status':stage,'updated_utc':now(),**kw})
 def attributes(x):
  rgb=x.mean((2,3));lum=x.mean(1);edge=((lum[:,:,1:]-lum[:,:,:-1]).abs().mean((1,2))+(lum[:,1:]-lum[:,:-1]).abs().mean((1,2)))/2
  return torch.stack([lum.mean((1,2)),rgb[:,0]-rgb[:,1],rgb[:,2]-rgb[:,1],lum.std((1,2)),edge,(x.max(1).values-x.min(1).values).mean((1,2)),lum[:,:128].mean((1,2))-lum[:,128:].mean((1,2)),lum[:,:,:128].mean((1,2))-lum[:,:,128:].mean((1,2))],1)
 def encode(pixels,ids):
  acts={s:[] for s in m.scales};features={s:[] for s in m.scales};recons=[];originals=[]
  for i in range(0,len(ids),8):
   x=image_tensor(pixels,np.asarray(ids[i:i+8]));f,a,_=m.encode(x)
   for s in m.scales:acts[s].append(a[s].float());features[s].append(f[s].float())
   recons.append(m.decoder(f).float());originals.append(x)
  return {s:torch.cat(v) for s,v in acts.items()},{s:torch.cat(v) for s,v in features.items()},torch.cat(recons),torch.cat(originals)
 status('encoding_discovery_and_confirmation')
 da,df,dr,dx=encode(vp,[x['cache_index'] for x in disc]);ca,cf,cr,cx=encode(cp,list(range(len(cp))))
 scaler=attributes(dr).std(0).clamp_min(.02)
 targets=[];rng=np.random.default_rng(20261116)
 for s in m.scales:
  energy=da[s].square().mean((0,2,3)).sqrt().cpu().numpy();rank=np.argsort(-energy)
  main=rank[:8].tolist();rest=rank[8:].tolist()
  for label,genes in [('gene_%03d'%main[0],[main[0]]),('gene_%03d'%main[1],[main[1]]),('group_top8',main)]:
   # Closest-mass unused genes, without looking at any edited image.
   control=[];available=set(rest)
   for g in genes:
    j=min(available,key=lambda j:abs(np.log(max(energy[j],1e-12)/max(energy[g],1e-12))))
    control.append(int(j));available.remove(j)
   targets.append({'name':s+'_'+label,'scale':s,'genes':genes,'control_genes':control,'discovery_rms_coefficients':[float(energy[g]) for g in genes]})
 write_json(out/'intervention_protocol.json',{'targets':targets,'discovery_images':64,'confirmation_images':128,'alpha':[0,.5,1,1.5],'attributes':['brightness','red_minus_green','blue_minus_green','contrast','edge_energy','saturation_proxy','top_minus_bottom_brightness','left_minus_right_brightness'],'attribute_scale':scaler.cpu().tolist(),'no_semantic_claim':True,'direction_estimated_on_discovery_only':True,'controls_have_their_own_discovery_direction_for_consistency':True,'random_control':'deterministic disjoint same-scale nearest-mass genes, not randomized; match per-image feature contribution L2 by scalar multiplication','source_checkpoint':str(base/'runs/multiscale_residual_ista/checkpoint_best.pt'),'source_step':saved['step'],'inference_precision':'FP32 with TF32 disabled for replay validation; no CPU model execution','created_utc':now()})
 torch.save({'activations':{s:v.cpu() for s,v in ca.items()},'features':{s:v.cpu() for s,v in cf.items()},'source_step':saved['step'],'manifest':confirm},out/'confirmation_codes.pt')
 np.savez_compressed(out/'confirmation_reconstruction.npz',originals=cx.cpu().numpy(),reconstructions=cr.cpu().numpy())
 mse=(cx-cr).square().mean((1,2,3));write_json(out/'new_city_reconstruction.json',{'count':128,'mae':ci((cx-cr).abs().mean((1,2,3)).cpu()),'psnr':ci((-10*torch.log10(mse)).cpu()),'ssim':ci(ssim_per_image(cr,cx).cpu()),'same_checkpoint_no_training':True})
 def contribution(acts,s,genes):
  weights=m.dictionaries[s].atoms[genes].permute(1,0,2,3).contiguous()
  return F.conv2d(acts[s][:,genes],weights,padding=m.dictionaries[s].kernel//2)
 def responses(acts,features,recons,t,control=False,need_pictures=False):
  s=t['scale'];own=contribution(acts,s,t['genes']);edit=contribution(acts,s,t['control_genes']) if control else own
  if control:
   ratio=own.flatten(1).norm(dim=1)/edit.flatten(1).norm(dim=1).clamp_min(1e-12);edit=edit*ratio[:,None,None,None]
  all_attrs=[];all_l1=[];all_local=[];pictures=[]
  for start in range(0,len(recons),16):
   end=min(start+16,len(recons));n=end-start
   combined={key:torch.cat([value[start:end] if key!=s else value[start:end]+(alpha-1)*edit[start:end] for alpha in [0,.5,1,1.5]],0) for key,value in features.items()}
   decoded=m.decoder(combined).reshape(4,n,3,256,256);baseline=decoded[2]
   attrs=torch.stack([attributes(v) for v in decoded]);differences=(decoded-baseline[None]).abs().mean(2)
   saliency=F.interpolate(own[start:end].square().mean(1,keepdim=True),size=(256,256),mode='bilinear',align_corners=False)[:,0]
   cutoff=saliency.flatten(1).quantile(.75,dim=1);footprint=(saliency>=cutoff[:,None,None]).float()
   local=(differences*footprint[None]).sum((2,3))/differences.sum((2,3)).clamp_min(1e-10)
   all_attrs.append(attrs.cpu().numpy());all_l1.append(differences.mean((2,3)).cpu().numpy());all_local.append(local.cpu().numpy())
   if need_pictures and start==0:pictures=decoded[:,:4].cpu().numpy()
  return np.concatenate(all_attrs,axis=1),np.concatenate(all_l1,axis=1),np.concatenate(all_local,axis=1),pictures
 records=[];summaries=[]
 for ti,t in enumerate(targets):
  status('intervention',target=ti+1,total=len(targets))
  discovery,_,_,_=responses(da,df,dr,t)
  direction=((discovery[3]-discovery[2])/scaler.cpu().numpy()).mean(0)
  direction_norm=np.linalg.norm(direction)
  for control in [False,True]:
   selected_direction=direction.copy()
   if control:
    reference,_,_,_=responses(da,df,dr,t,True)
    direction=((reference[3]-reference[2])/scaler.cpu().numpy()).mean(0)
    direction_norm=np.linalg.norm(direction)
   attr,l1,local,pics=responses(ca,cf,cr,t,control,not control)
   delta=(attr-attr[2:3])/scaler.cpu().numpy()[None,None]
   cos=lambda v:(v@direction)/(np.linalg.norm(v,axis=1)*max(direction_norm,1e-12)+1e-12)
   plus=cos(delta[3]);minus=cos(-delta[0])
   projected=delta@direction/max(direction_norm,1e-12)
   monotonic=np.all(np.diff(projected,axis=0)>=-1e-6,axis=0)
   summary={'target':t['name'],'kind':'matched_other_gene_control' if control else 'selected','genes':t['control_genes'] if control else t['genes'],'discovery_direction_norm':float(direction_norm),'alignment_with_selected_target_direction':ci((delta[3]@selected_direction)/(np.linalg.norm(delta[3],axis=1)*max(np.linalg.norm(selected_direction),1e-12)+1e-12)),'enhance_mae':ci(l1[3]),'delete_mae':ci(l1[0]),'enhance_direction_cosine':ci(plus),'delete_direction_cosine':ci(minus),'direction_monotonic_fraction':ci(monotonic.astype(float)),'enhance_locality_top_quartile':ci(local[3]),'enhance_positive_direction_fraction':ci((plus>0).astype(float)),'note':'Low-level attribute consistency only; no semantic disentanglement established.'}
   summaries.append(summary)
   for i,row in enumerate(confirm):records.append({'target':t['name'],'control':control,'city':row['city'],'ucdb_id':row['ucdb_id'],'image_id':row['id'],'enhance_mae':float(l1[3,i]),'delete_mae':float(l1[0,i]),'plus_cosine':float(plus[i]),'minus_cosine':float(minus[i]),'monotonic':bool(monotonic[i]),'attributes_at_four_doses':attr[:,i].tolist()})
   if not control:
    rows=[[('Original / '+confirm[i]['city'],cx[i].cpu().numpy()),('Full reconstruction',pics[2,i]),('Delete / alpha=0',pics[0,i]),('Enhance / alpha=1.5',pics[3,i]),('Enhance abs change x5',np.abs(pics[3,i]-pics[2,i])*5)] for i in range(4)]
    sheet(rows,out/(t['name']+'.png'))
 write_json(out/'intervention_summary.json',summaries)
 write_json(out/'intervention_per_image.json',records)
 # Pairing uses original-photo low-resolution layout descriptors before decoding mixes.
 cpu=cp.astype('float32')/255;lum=cpu.mean(3)
 descriptors=np.concatenate([lum.reshape(128,16,16,16,16).mean((2,4)).reshape(128,-1),cpu.mean(2).reshape(128,-1)],1)
 descriptors=(descriptors-descriptors.mean(0))/(descriptors.std(0)+.1)
 remaining=list(range(128));pairs=[]
 for _ in range(32):
  a=remaining.pop(0);b=min(remaining,key=lambda b:np.square(descriptors[a]-descriptors[b]).mean());remaining.remove(b);pairs.append([a,b])
 write_json(out/'composition_pairs.json',{'pairs':pairs,'matching':'nearest original-image 16x16 luminance + vertical RGB profile; greedy, distinct cities; first 32 receivers','region_normalized':[.25,.25,.75,.75],'groups':{s:next(t['genes'] for t in targets if t['scale']==s and t['name'].endswith('top8')) for s in ['medium','fine']},'selection_before_edit_decoding':True,'outside_metric_excludes_32px_halo':True})
 mix_results=[];visual=[];blind=[]
 masks={s:torch.zeros(1,1,*ca[s].shape[-2:],device='cuda') for s in m.scales}
 for s,mask in masks.items():h,w=mask.shape[-2:];mask[:,:,h//4:3*h//4,w//4:3*w//4]=1
 inside=torch.zeros(256,256,device='cuda');inside[80:176,80:176]=1
 outside=torch.ones(256,256,device='cuda');outside[32:224,32:224]=0
 pixelmask=torch.zeros(1,1,256,256,device='cuda');pixelmask[:,:,64:192,64:192]=1
 for pi,(a,b) in enumerate(pairs):
  status('composition',pair=pi+1,total=len(pairs))
  receiver={s:cf[s][a:a+1] for s in m.scales}
  outputs={};energy={}
  for mode in ['gene_group','matched_other_gene_group','all_feature_channels']:
   mixed={s:v.clone() for s,v in receiver.items()};change=0.
   for s in ['medium','fine']:
    t=next(t for t in targets if t['scale']==s and t['name'].endswith('top8'))
    if mode=='all_feature_channels':delta=(cf[s][b:b+1]-cf[s][a:a+1])*masks[s]
    else:
     genes=t['genes'] if mode=='gene_group' else t['control_genes'];diff=torch.zeros_like(ca[s][a:a+1]);diff[:,genes]=(ca[s][b:b+1,genes]-ca[s][a:a+1,genes])*masks[s]
     delta=m.dictionaries[s](diff)
    change+=float(delta.square().sum());mixed[s]=mixed[s]+delta
   outputs[mode]=m.decoder(mixed);energy[mode]=change**.5
  outputs['rgb_blend']=cr[a:a+1]*(1-.5*pixelmask)+cr[b:b+1]*(.5*pixelmask)
  # Self-swap must reproduce the receiver; group synthesis must also match direct mixed-code replay.
  null=m.decoder(receiver);null_error=float((null-cr[a:a+1]).abs().max())
  if null_error>1e-4:raise AssertionError(f'Self swap error {null_error}')
  direct={s:ca[s][a:a+1].clone() for s in m.scales}
  for s in ['medium','fine']:
   genes=next(t['genes'] for t in targets if t['scale']==s and t['name'].endswith('top8'))
   direct[s][:,genes]+=masks[s]*(ca[s][b:b+1,genes]-direct[s][:,genes])
  replay=m.decoder({s:m.dictionaries[s](direct[s]) for s in m.scales})
  replay_error=float((replay-outputs['gene_group']).abs().max())
  if replay_error>2e-4:raise AssertionError(f'Mixed code replay error {replay_error}')
  for mode,y in outputs.items():
   source=(cr[a:a+1]-cr[b:b+1]).abs().mean(1)[0]
   donor=(y-cr[b:b+1]).abs().mean(1)[0];change=(y-cr[a:a+1]).abs().mean(1)[0]
   inside_source=float((source*inside).sum()/inside.sum());inside_donor=float((donor*inside).sum()/inside.sum())
   mix_results.append({'pair':pi,'receiver_id':confirm[a]['id'],'donor_id':confirm[b]['id'],'mode':mode,'donor_distance_reduction_fraction':(inside_source-inside_donor)/max(inside_source,1e-8),'receiver_outside_mae':float((change*outside).sum()/outside.sum()),'whole_image_change_mae':float(change.mean()),'latent_change_l2':energy.get(mode),'self_swap_max_error':null_error,'mixed_code_replay_max_error':replay_error,'attributes_delta':(attributes(y)-attributes(cr[a:a+1]))[0].cpu().tolist()})
  row=[('A reconstruction',cr[a].cpu().numpy()),('B reconstruction',cr[b].cpu().numpy())]+[(mode,outputs[mode][0].cpu().numpy()) for mode in ['gene_group','matched_other_gene_group','all_feature_channels','rgb_blend']]
  if pi<6:visual.append(row)
  # Anonymous candidate strips are materials for future human review, not human scores.
  modes=list(outputs);rng.shuffle(modes)
  sheet([row[:2]+[(chr(65+j),outputs[mode][0].cpu().numpy()) for j,mode in enumerate(modes)]],out/f'blind_pair_{pi:02d}.png',160)
  blind.append({'pair':pi,'mapping':dict(zip('ABCD',modes))})
 sheet(visual,out/'composition_overview.png',208)
 write_json(out/'composition_per_pair.json',mix_results);write_json(out/'blind_answer_key.json',blind)
 comp={mode:{key:ci([r[key] for r in mix_results if r['mode']==mode]) for key in ['donor_distance_reduction_fraction','receiver_outside_mae','whole_image_change_mae']} for mode in outputs}
 write_json(out/'composition_summary.json',{'metrics':comp,'n_pairs':32,'ordinary_dense_AE_control':'pending new from-scratch control training; all_feature_channels here is the original dictionary model, not an ordinary AE','no_target_or_human_realism_scores':True,'rgb_blend_is_reference_only':True,'known_limitation':'Global GroupNorm and decoder receptive field can propagate edits beyond the mask. Pixel donor proximity does not prove semantic transfer or realism.'})
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(14,4))
 selected=[x for x in summaries if x['kind']=='selected'];controls=[x for x in summaries if x['kind']!='selected'];xx=np.arange(len(selected))
 for ax,key,title in zip(axes,['enhance_mae','enhance_direction_cosine','direction_monotonic_fraction'],['Intervention magnitude (RGB MAE)','Cross-city direction consistency','Monotonic dose response']):
  for rows,offset,label in [(selected,-.15,'Selected gene/group'),(controls,.15,'Matched other genes')]:
   means=np.array([x[key]['mean'] for x in rows]);limits=np.array([x[key]['ci95'] for x in rows]);ax.errorbar(xx+offset,means,yerr=[means-limits[:,0],limits[:,1]-means],fmt='o',label=label,capsize=2)
  ax.set_xticks(xx,[x['target'].replace('coarse','C').replace('medium','M').replace('fine','F') for x in selected],rotation=70,ha='right',fontsize=7);ax.set_title(title);ax.grid(alpha=.2)
 axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'intervention_summary.png',dpi=150);plt.close(fig)
 status('complete',intervention_targets=len(targets),confirmation_images=128,composition_pairs=32)
 return summaries,comp

def main():
 c=json.loads((ROOT/'configs/experiment.json').read_text());out=ROOT/'reports/frozen_diagnostics';out.mkdir(exist_ok=True)
 leases=acquire(c,1)
 while not leases:
  write_json(out/'progress.json',{'status':'waiting_for_gpu','updated_utc':now()});time.sleep(10);leases=acquire(c,1)
 gpu,handle=leases[0];os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid']
 write_json(out/'progress.json',{'status':'leased_gpu','physical_gpu':gpu,'pid':os.getpid(),'updated_utc':now()})
 try:
  import torch
  with torch.inference_mode():execute(c)
 except Exception as e:
  write_json(out/'progress.json',{'status':'failed','error':str(e),'updated_utc':now()});raise
 finally:handle.close()
if __name__=='__main__':main()
