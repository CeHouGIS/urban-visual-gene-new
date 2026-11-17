"""Train the new SDXL spatial adapter; sample only from random noise.

The frozen SDXL renderer never receives source RGB or source latents directly.
Source latents may enter the evaluated V/DV conditioner only.
"""
from pathlib import Path
import argparse
import contextlib
import json
import math
import random
import time
import traceback
import subprocess
import sys
from functools import wraps

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler, EulerDiscreteScheduler, AutoencoderKL
from model import DenseConditioner, GeneConditioner, install_adapters

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/'models/sdxl-base-1.0'

def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())

def write_json(path,obj):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)

def isolated_evaluation(function):
    @wraps(function)
    def wrapped(*args,**kwargs):
        device=args[5];state=random.getstate();np_state=np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[device.index]):return function(*args,**kwargs)
        finally:random.setstate(state);np.random.set_state(np_state)
    return wrapped

class Data:
    def __init__(self,resolution=1024):
        self.resolution=resolution;suffix='' if resolution==512 else f'_{resolution}'
        self.records=json.loads((ROOT/'data/manifest.json').read_text())['records']
        self.dino=np.load(ROOT/'data/dino16.npy',mmap_mode='r')
        self.latents=np.load(ROOT/f'data/latents{suffix}.npy',mmap_mode='r')
        self.ready=np.load(ROOT/f'data/latents_ready{suffix}.npy')
        self.dino_ready=np.load(ROOT/'data/dino_ready.npy')
        self.text=torch.load(ROOT/f'data/conditioning{suffix}.pt',map_location='cpu',weights_only=True)
    def tensors(self,indices,device):
        if not self.ready[indices].all():raise RuntimeError('Required VAE targets not cached')
        if not self.dino_ready[indices].all():raise RuntimeError('Required DINO features not cached')
        d=torch.from_numpy(np.array(self.dino[indices],dtype=np.float32)).to(device)
        z=torch.from_numpy(np.array(self.latents[indices],dtype=np.float32)).to(device)
        return d,z
    def ids(self,split):return [r['index'] for r in self.records if r['split']==split and self.ready[r['index']]]

def build(args,device):
    unet=UNet2DConditionModel.from_pretrained(MODEL,subfolder='unet',variant='fp16',
        use_safetensors=True,local_files_only=True,torch_dtype=torch.bfloat16).to(device)
    unet.requires_grad_(False)
    adapters=install_adapters(unet,initial_gain=getattr(args,'initial_gain',.01),locality_bias=getattr(args,'locality_bias',0.)).to(device)
    conditioner=DenseConditioner(args.mode,resolution=args.resolution).to(device)
    if args.mode=='D':
        conditioner.appearance.requires_grad_(False);conditioner.fuse.requires_grad_(False)
    elif args.mode=='V':
        conditioner.dino.requires_grad_(False);conditioner.fuse.requires_grad_(False)
    gene=None
    if args.gene!='none':
        if not args.teacher or not args.distilled_gene:raise RuntimeError('Gene generation training requires a locked dense teacher and distilled dictionary.')
        gene=GeneConditioner(topk=4 if args.gene=='learned_s4' else 8,
            learned=args.gene!='random_s8',seed=17).to(device)
    if args.gradient_checkpointing:unet.enable_gradient_checkpointing()
    scheduler=DDPMScheduler.from_pretrained(MODEL,subfolder='scheduler',local_files_only=True)
    return unet,adapters,conditioner,gene,scheduler

def conditioning(data,device):
    text=data.text['embeddings'][0:1].to(device,dtype=torch.bfloat16)
    pooled=data.text['pooled'][0:1].to(device,dtype=torch.bfloat16)
    r=data.resolution
    times=torch.tensor([[r,r,0,0,r,r]],device=device,dtype=torch.bfloat16)
    return text,{'text_embeds':pooled,'time_ids':times}

def load_checkpoint(path,adapters,conditioner,gene=None,optimizer=None):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    adapters.load_state_dict(ckpt['adapters']);conditioner.load_state_dict(ckpt['conditioner'])
    if gene is not None and ckpt.get('gene') is not None:gene.load_state_dict(ckpt['gene'])
    if optimizer is not None and 'optimizer' in ckpt:optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt

@torch.inference_mode()
def sample_one(unet,tokens,data,device,seed,steps=30):
    scheduler=EulerDiscreteScheduler.from_pretrained(MODEL,subfolder='scheduler',local_files_only=True)
    scheduler.set_timesteps(steps,device=device)
    generator=torch.Generator(device=device).manual_seed(seed)
    # No image or encoded target is used to initialize this state.
    side=data.resolution//8
    z=torch.randn((1,4,side,side),generator=generator,device=device,dtype=torch.float32)*scheduler.init_noise_sigma
    text=data.text['embeddings'][[1,0]].to(device,dtype=torch.bfloat16)
    pooled=data.text['pooled'][[1,0]].to(device,dtype=torch.bfloat16)
    r=data.resolution
    times=torch.tensor([[r,r,0,0,r,r]]*2,device=device,dtype=torch.bfloat16)
    # Both CFG halves see the same image code; CFG acts on constant text only.
    image_tokens=tokens.expand(2,-1,-1)
    for t in scheduler.timesteps:
        latent=scheduler.scale_model_input(torch.cat([z,z]),t).to(torch.bfloat16)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            prediction=unet(latent,t,encoder_hidden_states=text,
                added_cond_kwargs={'text_embeds':pooled,'time_ids':times},
                cross_attention_kwargs={'image_tokens':image_tokens}).sample.float()
        negative,positive=prediction.chunk(2)
        prediction=negative+5.*(positive-negative)
        z=scheduler.step(prediction,t,z).prev_sample
    assert torch.isfinite(z).all(),'Nonfinite sampled latent'
    return z.cpu()

@torch.inference_mode()
@isolated_evaluation
def evaluate(args,unet,conditioner,gene,data,device,out,step,train_ids):
    unet.eval();conditioner.eval()
    if gene is not None:gene.eval()
    selected=train_ids[:2]
    validation=[];seen=set()
    for index in data.ids('validation'):
        city=data.records[index]['city']
        if city not in seen:validation.append(index);seen.add(city)
        if len(validation)==2:break
    selected+=validation
    assert len(selected)==4,'Need two train and two validation images for smoke sampling'
    d,z=data.tensors(selected,device)
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        codes=conditioner(d,z)
        if gene is not None:codes=gene(codes)
    directory=out/f'samples_{step:06d}';directory.mkdir(exist_ok=True)
    write_json(out/'sampling_status.json',{'state':'sampling','step':step,'completed_images':0,'total_images':12,'started_at':stamp()})
    write_json(out/'status.json',{'state':'sampling','step':step,'target_updates':args.updates,'updated_at':stamp()})
    jobs=[];latents=[]
    # Full correct/zero/shuffled panels use identical initial noise per row.
    for row,index in enumerate(selected):
        for kind in ['correct','zero','shuffled']:
            tokens=codes[row:row+1] if kind=='correct' else torch.zeros_like(codes[row:row+1]) if kind=='zero' else codes[(row+1)%4:(row+1)%4+1]
            latents.append(sample_one(unet,tokens,data,device,101,args.sampling_steps))
            jobs.append({'row':row,'index':index,'kind':kind,'seed':101,'file':f'{row}_{kind}.png'})
            write_json(out/'sampling_status.json',{'state':'sampling','step':step,'completed_images':len(jobs),'total_images':12,'updated_at':stamp()})
    del d,z,codes;torch.cuda.empty_cache()
    vae=AutoencoderKL.from_pretrained(MODEL,subfolder='vae',variant='fp16',use_safetensors=True,
        local_files_only=True,torch_dtype=torch.float32).eval().to(device)
    arrays=[]
    for job,latent in zip(jobs,latents):
        decoded=vae.decode(latent.to(device)/data.text['vae_scaling_factor']).sample
        image=(decoded[0].float().clamp(-1,1).permute(1,2,0).cpu().numpy()+1)*127.5
        image=Image.fromarray(image.round().astype(np.uint8));image.save(directory/job['file']);arrays.append(np.asarray(image))
    del vae;torch.cuda.empty_cache()
    canvas=Image.new('RGB',(4*256,4*280+30),'white');draw=ImageDraw.Draw(canvas)
    for col,label in enumerate(['Target','Correct code','Zero code','Shuffled code']):draw.text((col*256+8,8),label,fill='black')
    metrics=[]
    from skimage.metrics import structural_similarity
    for row,index in enumerate(selected):
        r=data.records[index]
        with Image.open(r['source_image']) as im:target=im.convert('RGB').resize((data.resolution,data.resolution),Image.Resampling.BILINEAR)
        target.save(directory/f'{row}_target.png')
        target_array=np.asarray(target,dtype=np.float32)
        canvas.paste(target.resize((256,256)),(0,30+row*280))
        draw.text((8,30+row*280+258),f"{r['split']} {r['city']} {index}",fill='black')
        for k,kind in enumerate(['correct','zero','shuffled']):
            a=arrays[row*3+k];mse=float(np.mean((a.astype(np.float32)-target_array)**2))
            canvas.paste(Image.fromarray(a).resize((256,256)),((k+1)*256,30+row*280))
            metrics.append({'index':index,'split':r['split'],'kind':kind,'seed':101,
                'psnr':float(10*math.log10(255**2/max(mse,1e-12))),
                'ssim':float(structural_similarity(a,target_array,data_range=255,channel_axis=2))})
    canvas.save(directory/'panel.png')
    # Failure to download genuine LPIPS weights is reported, never replaced with random weights.
    try:
        import lpips
        metric=lpips.LPIPS(net='alex').eval().to('cpu')
        for job,entry,a in zip(jobs,metrics,arrays):
            target=np.array(Image.open(directory/f"{job['row']}_target.png").convert('RGB'),dtype=np.float32)
            x=torch.from_numpy(a.astype(np.float32)).permute(2,0,1)[None]/127.5-1
            y=torch.from_numpy(target).permute(2,0,1)[None]/127.5-1
            entry['lpips_alex']=float(metric(x,y).item())
        lpips_status='computed with pretrained AlexNet and official LPIPS weights'
    except Exception as error:lpips_status=f'UNAVAILABLE: {type(error).__name__}: {error}'
    report={'step':step,'initialization':'pure random noise','steps':args.sampling_steps,'cfg':5,
        'image_cfg':'same image tokens in both text CFG halves','metrics':metrics,'lpips_status':lpips_status,
        'human_ratings':'not collected','test_split_used':False,'created_at':stamp()}
    write_json(directory/'metrics.json',report)
    write_json(out/'sampling_status.json',{'state':'complete','step':step,'completed_images':12,'total_images':12,'updated_at':stamp()})
    print(json.dumps({'event':'sampling_complete','step':step,'panel':str(directory/'panel.png'),'lpips_status':lpips_status}),flush=True)
    unet.train();conditioner.train()
    return report

def main(args):
    torch.set_num_threads(4);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device(args.device);torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device.index)
    torch.backends.cuda.matmul.allow_tf32=True
    out=ROOT/'runs'/args.run;out.mkdir(parents=True,exist_ok=True)
    data=Data(args.resolution);ids=data.ids('train')
    # One city block must not dominate the 32-image overfit check.
    if args.smoke_images:
        by_city={}
        for i in ids:by_city.setdefault(data.records[i]['city'],[]).append(i)
        ids=[v[j] for j in range(math.ceil(args.smoke_images/len(by_city))) for v in by_city.values() if j<len(v)][:args.smoke_images]
    if not ids:raise RuntimeError('No prepared training data')
    unet,adapters,conditioner,gene,scheduler=build(args,device)
    teacher_normalizer=None
    if args.teacher:
        teacher_checkpoint=load_checkpoint(args.teacher,adapters,conditioner)
        assert teacher_checkpoint['args']['mode']==args.mode and teacher_checkpoint['args']['resolution']==args.resolution
        assert teacher_checkpoint['args']['smoke_images']==0,'A smoke checkpoint cannot be the formal dense teacher'
        assert teacher_checkpoint['args']['gene']=='none','Teacher must be the dense model'
        conditioner.eval().requires_grad_(False)
        if gene is not None:
            distilled=torch.load(args.distilled_gene,map_location='cpu',weights_only=False)
            assert distilled['stage']=='distillation' and distilled['variant']==args.gene
            assert Path(distilled['teacher']).resolve()==Path(args.teacher).resolve(),'Dictionary and renderer must inherit the same teacher checkpoint'
            gene.load_state_dict(distilled['gene']);teacher_normalizer=distilled['teacher_normalizer']
    parameters=list(adapters.parameters())+list(conditioner.parameters())
    if gene is not None:parameters+=list(gene.parameters())
    parameters=[p for p in parameters if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=args.learning_rate,weight_decay=.01)
    start=0
    if args.resume:
        ckpt=load_checkpoint(args.resume,adapters,conditioner,gene,optimizer);start=ckpt['step']
        if 'rng' in ckpt:torch.set_rng_state(ckpt['rng'])
        if 'cuda_rng' in ckpt:torch.cuda.set_rng_state(ckpt['cuda_rng'],device)
        if 'python_rng' in ckpt:random.setstate(ckpt['python_rng'])
    elif args.initialize:load_checkpoint(args.initialize,adapters,conditioner,gene)
    text,added=conditioning(data,device)
    metadata={'args':vars(args),'training_indices':ids,'trainable_parameters':sum(p.numel() for p in parameters),
        'base_parameters_trainable':sum(p.numel() for n,p in unet.named_parameters() if p.requires_grad and '.processor.' not in n),
        'scheduler_prediction_type':scheduler.config.prediction_type,'created_at':stamp(),
        'frozen_base':True,'no_rgb_latent_skip':True,'effective_batch':args.accumulation,'device_name':torch.cuda.get_device_name(device)}
    assert metadata['base_parameters_trainable']==0,metadata
    write_json(out/'run.json',metadata);print(json.dumps(metadata),flush=True)
    def save(step):
        dest=out/f'checkpoint_{step:06d}.pt';temp=Path(str(dest)+'.tmp')
        torch.save({'step':step,'adapters':adapters.state_dict(),'conditioner':conditioner.state_dict(),
            'gene':gene.state_dict() if gene is not None else None,'optimizer':optimizer.state_dict(),
            'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(device),'python_rng':random.getstate(),
            'teacher_normalizer':teacher_normalizer,'args':vars(args)},temp)
        temp.replace(dest);write_json(out/'latest.json',{'step':step,'checkpoint':str(dest),'updated_at':stamp()})
        if args.keep_checkpoints:
            checkpoints=sorted(out.glob('checkpoint_*.pt'))
            protected=None
            if (out/'best_validation.json').exists():protected=Path(json.loads((out/'best_validation.json').read_text())['checkpoint'])
            for obsolete in checkpoints[:-args.keep_checkpoints]:
                if protected is None or obsolete.resolve()!=protected.resolve():obsolete.unlink()
    if args.eval_only:
        evaluate(args,unet,conditioner,gene,data,device,out,start,ids);return
    if args.sample_initial:evaluate(args,unet,conditioner,gene,data,device,out,start,ids)
    unet.train();conditioner.train()
    begin=time.monotonic();recent=[]
    log=(out/'train.jsonl').open('a',buffering=1)
    for step in range(start+1,args.updates+1):
        optimizer.zero_grad(set_to_none=True);loss_value=0.;step_start=time.monotonic()
        for micro in range(args.accumulation):
            index=random.choice(ids);d,z=data.tensors([index],device)
            noise=torch.randn_like(z);t=torch.randint(0,scheduler.config.num_train_timesteps,(1,),device=device).long()
            noisy=scheduler.add_noise(z,noise,t).to(torch.bfloat16)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                image_tokens=conditioner(d,z)
                condition_loss=None
                if gene is not None:
                    dense_tokens=image_tokens.detach();image_tokens=gene(dense_tokens)
                    condition_loss=F.mse_loss(image_tokens.float(),dense_tokens.float())/(teacher_normalizer+1e-6)
                prediction=unet(noisy,t,encoder_hidden_states=text,added_cond_kwargs=added,
                    cross_attention_kwargs={'image_tokens':image_tokens}).sample
                if scheduler.config.prediction_type=='epsilon':target=noise
                elif scheduler.config.prediction_type=='v_prediction':target=scheduler.get_velocity(z,noise,t)
                else:raise ValueError(scheduler.config.prediction_type)
                loss=F.mse_loss(prediction.float(),target.float())
                if condition_loss is not None:loss=loss+0.1*condition_loss
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at {step}/{micro}')
            (loss/args.accumulation).backward();loss_value+=loss.item()/args.accumulation
        missing=[n for n,p in conditioner.named_parameters() if p.requires_grad and p.grad is None and
            (n.startswith('norm') or n.startswith('fuse') and args.mode=='DV' or n.startswith('dino') and args.mode in ('D','DV') or n.startswith('appearance') and args.mode in ('V','DV'))]
        if missing:raise RuntimeError(f'Conditioner not receiving gradients: {missing}')
        clip=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        lr=args.learning_rate*min(1.,step/max(1,args.warmup))
        for group in optimizer.param_groups:group['lr']=lr
        # Report the path from image encoder to the generator on the first update.
        if step==start+1:
            checks={n:float(p.grad.float().norm()) for n,p in conditioner.named_parameters() if p.grad is not None}
            adapter_checks=[{'gain':float(p.gain.detach()),'gain_grad':float(p.gain.grad),'q_grad':float(p.q.weight.grad.norm()),'v_grad':float(p.v.weight.grad.norm())} for p in adapters]
            if not args.teacher and not any(v>0 and math.isfinite(v) for v in checks.values()):raise RuntimeError('Zero/nonfinite conditioner gradients')
            if not any(p.v.weight.grad is not None and torch.isfinite(p.v.weight.grad).all() and p.v.weight.grad.norm()>0 for p in adapters):raise RuntimeError('Zero/nonfinite adapter gradients')
            if gene is not None and not any(p.grad is not None and p.grad.norm()>0 for p in gene.coefficients.parameters()):raise RuntimeError('Gene coefficients not receiving gradients')
            write_json(out/f'gradient_check_{step:06d}.json',{'conditioner':checks,'adapters':adapter_checks})
        optimizer.step()
        if gene is not None:gene.normalize_bank()
        torch.cuda.synchronize(device);duration=time.monotonic()-step_start;recent.append(duration)
        record={'step':step,'loss':loss_value,'lr':lr,'grad_norm':float(clip),'seconds':round(duration,3),
            'elapsed_seconds':round(time.monotonic()-begin,1),'peak_allocated_GB':round(torch.cuda.max_memory_allocated(device)/1e9,3),'at':stamp()}
        log.write(json.dumps(record)+'\n')
        if step<=3 or step%5==0:print(json.dumps(record),flush=True)
        write_json(out/'status.json',dict(record,state='training',target_updates=args.updates,
            estimated_remaining_seconds=round(np.mean(recent[-20:])*(args.updates-step))))
        formal_validation=args.validation_every and (step%args.validation_every==0 or step==args.updates)
        if step%args.checkpoint_every==0 or step==args.updates or formal_validation:save(step)
        if args.evaluate_every and (step%args.evaluate_every==0 or step==args.updates):
            evaluate(args,unet,conditioner,gene,data,device,out,step,ids)
        if formal_validation:
            checkpoint=out/f'checkpoint_{step:06d}.pt'
            write_json(out/'status.json',{'state':'formal_validation','step':step,'target_updates':args.updates,'updated_at':stamp()})
            # Move train state to CPU while standalone evaluation owns the GPU.
            unet.to('cpu');conditioner.to('cpu')
            if gene is not None:gene.to('cpu')
            for state in optimizer.state.values():
                for key,value in state.items():
                    if torch.is_tensor(value):state[key]=value.cpu()
            torch.cuda.empty_cache()
            command=[sys.executable,str(ROOT/'evaluate.py'),'--checkpoint',str(checkpoint),
                '--run',args.run+f'_val_{step:06d}','--limit','64','--seeds','101','211',
                '--device',str(device),'--memory-fraction',str(args.memory_fraction)]
            with (out/f'validation_{step:06d}.log').open('a') as validation_log:
                subprocess.run(command,stdout=validation_log,stderr=subprocess.STDOUT,check=True)
            unet.to(device);conditioner.to(device)
            if gene is not None:gene.to(device)
            for state in optimizer.state.values():
                for key,value in state.items():
                    if torch.is_tensor(value) and key!='step':state[key]=value.to(device)
    log.close();write_json(out/'status.json',{'state':'training_complete','step':args.updates,'completed_at':stamp(),
        'sampling_complete':bool(args.evaluate_every),'note':'Passing generation gate requires inspection of saved samples and paired condition controls.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--device',default='cuda:2')
    p.add_argument('--mode',choices=['D','V','DV'],default='DV');p.add_argument('--gene',choices=['none','random_s8','learned_s8','learned_s4'],default='none')
    p.add_argument('--updates',type=int,default=200);p.add_argument('--accumulation',type=int,default=16)
    p.add_argument('--smoke-images',type=int,default=32);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--warmup',type=int,default=300)
    p.add_argument('--checkpoint-every',type=int,default=50);p.add_argument('--evaluate-every',type=int,default=200)
    p.add_argument('--keep-checkpoints',type=int,default=2,help='Retain this many most recent complete checkpoints; zero keeps all')
    p.add_argument('--validation-every',type=int,default=0,help='Run fixed64validation inputs and2seeds through standalone full sampling')
    p.add_argument('--sampling-steps',type=int,default=30);p.add_argument('--memory-fraction',type=float,default=.42)
    p.add_argument('--resolution',type=int,choices=[512,1024],default=1024)
    p.add_argument('--gradient-checkpointing',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--resume');p.add_argument('--initialize');p.add_argument('--eval-only',action='store_true');p.add_argument('--sample-initial',action='store_true')
    p.add_argument('--teacher',help='Common locked dense initialization; freezes the input fusion for Dense+ and all Gene variants')
    p.add_argument('--distilled-gene',help='Distillation checkpoint required when --gene is set')
    p.add_argument('--initial-gain',type=float,default=.01,help='Architecture hyperparameter; record any rescue separately')
    p.add_argument('--locality-bias',type=float,default=0.,help='Optional negative squared coordinate-distance attention bias')
    arguments=p.parse_args()
    try:main(arguments)
    except Exception as error:
        out=ROOT/'runs'/arguments.run;out.mkdir(parents=True,exist_ok=True)
        previous=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        write_json(out/'failure.json',{'exception':repr(error),'traceback':traceback.format_exc(),'previous_status':previous,'at':stamp()})
        write_json(out/'status.json',dict(previous,state='failed',exception=repr(error),failed_at=stamp()))
        raise
