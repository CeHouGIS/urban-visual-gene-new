"""Distill a shared sparse dictionary from a locked dense conditioner.

Only run after the dense generator passes its full-sampling gate.
No renderer or held-out test images are used to optimize this stage.
"""
from pathlib import Path
import argparse,json,random,time
import numpy as np
import torch
from torch.nn import functional as F
from model import DenseConditioner,GeneConditioner
from train import Data,write_json,stamp,ROOT

def main(args):
    torch.set_num_threads(4);torch.manual_seed(args.seed);random.seed(args.seed)
    device=torch.device(args.device);torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.2,device.index)
    dense=torch.load(args.teacher,map_location='cpu',weights_only=False)
    teacher_args=dense['args'];mode=teacher_args['mode'];resolution=teacher_args['resolution']
    data=Data(resolution);train=data.ids('train');validation=data.ids('validation')
    assert len(train)==2000 and len(validation)==400,'Complete training/validation caches required'
    teacher=DenseConditioner(mode,resolution=resolution).to(device).eval().requires_grad_(False)
    teacher.load_state_dict(dense['conditioner'])
    gene=GeneConditioner(4 if args.variant=='learned_s4' else 8,learned=args.variant!='random_s8',seed=17).to(device)
    optimizer=torch.optim.AdamW([p for p in gene.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    out=ROOT/'runs'/args.run;out.mkdir(parents=True,exist_ok=True)
    # Fixed teacher normalization from training split only, shared by all variants.
    with torch.inference_mode():
        total=0.;count=0
        for start in range(0,len(train),16):
            d,z=data.tensors(train[start:start+16],device);c=teacher(d,z);total+=c.square().sum().item();count+=c.numel()
        normalizer=total/count
    metadata={'stage':'dictionary_distillation','variant':args.variant,'teacher':str(Path(args.teacher).resolve()),
        'teacher_step':dense['step'],'teacher_normalizer':normalizer,'seed':args.seed,'dictionary_initialization_seed':17,
        'updates':args.updates,'effective_batch':16,'created_at':stamp(),'renderer_used':False,'test_used':False}
    write_json(out/'run.json',metadata);log=(out/'train.jsonl').open('a',buffering=1);begin=time.monotonic()
    for step in range(1,args.updates+1):
        indices=random.choices(train,k=16);d,z=data.tensors(indices,device)
        with torch.no_grad():target=teacher(d,z)
        reconstruction,ids,values=gene(target,return_code=True)
        loss=F.mse_loss(reconstruction,target)/(normalizer+1e-6)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(gene.parameters(),1.,error_if_nonfinite=True)
        lr=1e-4*min(1.,step/300)
        for g in optimizer.param_groups:g['lr']=lr
        optimizer.step();gene.normalize_bank()
        if step%10==0 or step==1:
            entry={'step':step,'loss':float(loss),'grad_norm':float(grad),'lr':lr,'elapsed':time.monotonic()-begin}
            log.write(json.dumps(entry)+'\n');write_json(out/'status.json',dict(entry,state='training',target_updates=args.updates))
            if step%100==0 or step==1:print(json.dumps(entry),flush=True)
        if step%500==0 or step==args.updates:
            with torch.inference_mode():
                errors=[];counts=torch.zeros(512,device=device);intensity=torch.zeros(512,device=device)
                for start in range(0,len(validation),16):
                    d,z=data.tensors(validation[start:start+16],device);target=teacher(d,z)
                    output,ix,coef=gene(target,return_code=True)
                    errors.append(float(F.mse_loss(output,target)/(normalizer+1e-6)))
                    counts.scatter_add_(0,ix.flatten(),torch.ones_like(coef.flatten()))
                    intensity.scatter_add_(0,ix.flatten(),coef.abs().flatten())
                diagnostics={'step':step,'validation_normalized_mse':float(np.mean(errors)),
                    'dead_genes':int((counts==0).sum()),'activation_counts':counts.cpu().tolist(),
                    'absolute_coefficient_sum':intensity.cpu().tolist(),'normalizer':normalizer}
                write_json(out/f'diagnostics_{step:06d}.json',diagnostics)
            path=out/f'checkpoint_{step:06d}.pt';temp=Path(str(path)+'.tmp')
            torch.save({'stage':'distillation','step':step,'gene':gene.state_dict(),'optimizer':optimizer.state_dict(),
                'teacher':str(Path(args.teacher).resolve()),'teacher_normalizer':normalizer,'mode':mode,
                'resolution':resolution,'variant':args.variant,'seed':args.seed},temp);temp.replace(path)
            write_json(out/'latest.json',{'step':step,'checkpoint':str(path)})
    write_json(out/'status.json',{'state':'distillation_complete','step':args.updates,'completed_at':stamp(),
        'note':'Conditional feature reconstruction only; generated-image quality not evaluated by this stage.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--teacher',required=True);p.add_argument('--run',required=True)
    p.add_argument('--variant',choices=['random_s8','learned_s8','learned_s4'],required=True)
    p.add_argument('--updates',type=int,default=2000);p.add_argument('--seed',type=int,default=17);p.add_argument('--device',default='cuda:3')
    main(p.parse_args())
