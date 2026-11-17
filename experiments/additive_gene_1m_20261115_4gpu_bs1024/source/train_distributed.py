"""Two-rank DDP continuation, global batch 1024, preserving samples already seen."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from common import append_json, load_config, now, write_json
from data import image_tensor
from distributed_schedule import learning_rate, next_batch, rank_ids
from metrics import ssim_per_image
from report import plot_run
from train import (amp_context, atomic_checkpoint, create_model, perceptual_model,
                   preview, reconstruction_loss)

STOP=False


def request_stop(*_):
    global STOP
    STOP=True


def primary():
    return dist.get_rank()==0


def pixels(config,split):
    value=np.load(Path(config['cache'])/f'{split}.rgb.npy',mmap_mode='r')
    side=config['data']['resolution']
    if value.shape!=(config['data'][split],side,side,3) or value.dtype!=np.uint8:
        raise ValueError('Existing local RGB data do not match the selected dataset')
    if split=='train':
        flags=np.load(Path(config['cache'])/'train.ready.npy',mmap_mode='r')
        if flags.shape!=(len(value),) or not np.all(flags==1):
            raise ValueError('DDP continuation requires the already completed local RGB cache')
    return value


def synchronized_stop(root):
    value=torch.tensor(int(STOP or (root/'STOP_REQUESTED').exists()),device='cuda')
    dist.all_reduce(value,op=dist.ReduceOp.MAX)
    return bool(value.item())


def assert_parameters_synchronized(model):
    # Check every parameter after an optimizer update, using GPU communication.
    delta=torch.zeros((),device='cuda')
    for parameter in model.parameters():
        reference=parameter.detach().clone()
        dist.broadcast(reference,src=0)
        delta=torch.maximum(delta,(parameter.detach()-reference).abs().max())
    dist.all_reduce(delta,op=dist.ReduceOp.MAX)
    if float(delta)>1e-6:
        raise AssertionError(f'DDP parameters diverged between ranks: {float(delta)}')
    return float(delta)


def distributed_update(ddp,optimizer,perceptual,prepared,config):
    """Accumulate local microbatches; DDP averages gradients exactly once."""
    model=ddp.module
    optimizer.zero_grad(set_to_none=True)
    size=len(prepared)
    micro=config['training']['micro_batch_size']
    statistics=torch.zeros(4,device='cuda',dtype=torch.float64)
    usage=torch.zeros(config['model']['dictionary_size'],device='cuda',dtype=torch.int64)
    for begin in range(0,size,micro):
        end=min(begin+micro,size)
        context=ddp.no_sync() if end<size else nullcontext()
        with context:
            target=image_tensor(prepared,np.arange(begin,end),'cuda')
            with amp_context(config):
                output=ddp(target)
            loss,l1,lp=reconstruction_loss(output['reconstruction'],target,perceptual,
                                           config['training']['perceptual_weight'])
            weight=(end-begin)/size
            (loss*weight).backward()
        statistics+=torch.stack([loss.detach(),l1.detach(),lp.detach(),
                                output['latent'].detach().float().square().mean().sqrt()]).double()*weight
        if model.sparse:
            usage+=(output['activations'].detach()>0).sum((0,2,3))
        del output,loss,l1,lp,target
    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),config['training']['gradient_clip'])
    finite=torch.tensor(int(torch.isfinite(statistics).all() and torch.isfinite(grad)),device='cuda')
    dist.all_reduce(finite,op=dist.ReduceOp.MIN)
    if not finite.item():
        raise FloatingPointError('DDP has nonfinite loss or gradients')
    optimizer.step()
    dist.all_reduce(statistics)
    statistics/=dist.get_world_size()
    if model.sparse:
        dist.all_reduce(usage)
    return statistics.cpu().numpy(),usage,float(grad)


@torch.inference_mode()
def evaluate(model,data,config,perceptual):
    model.eval()
    rank,world=dist.get_rank(),dist.get_world_size()
    ids=np.arange(rank,len(data),world)
    names=['l1','mse','psnr_mean_per_image','ssim','lpips']
    sums=torch.zeros(7,device='cuda',dtype=torch.float64)
    usage=torch.zeros(config['model']['dictionary_size'],device='cuda',dtype=torch.int64)
    # LPIPS/SSIM evaluation uses bounded batches and disjoint rows, without padding.
    micro=min(64,config['training']['micro_batch_size'])
    for start in range(0,len(ids),micro):
        target=image_tensor(data,ids[start:start+micro],'cuda')
        with amp_context(config):
            result=model(target)
        prediction=result['reconstruction'].float()
        mse=(prediction-target).square().mean((1,2,3))
        values=[(prediction-target).abs().mean((1,2,3)),mse,
                -10*torch.log10(mse.clamp_min(1e-12)),ssim_per_image(prediction,target),
                perceptual(prediction*2-1,target*2-1).flatten()]
        sums[:5]+=torch.stack([value.sum() for value in values]).double()
        sums[5]+=len(target)
        if model.sparse:
            act=result['activations']
            usage+=(act>0).sum((0,2,3))
            sums[6]+=act.shape[0]*act.shape[2]*act.shape[3]
    dist.all_reduce(sums)
    if int(sums[5])!=len(data):
        raise AssertionError('Distributed evaluation duplicated or lost rows')
    metrics={name:float(sums[i]/sums[5]) for i,name in enumerate(names)}
    metrics['psnr_from_mean_mse']=-10*math.log10(max(metrics['mse'],1e-12))
    metrics['selection_loss']=metrics['l1']+config['training']['perceptual_weight']*metrics['lpips']
    metrics['images']=len(data)
    if model.sparse:
        dist.all_reduce(usage)
        metrics.update(gene_activation_counts=usage.cpu().tolist(),genes_used=int((usage>0).sum()),
                       inactive_genes_on_this_split=int((usage==0).sum()),
                       mean_active_per_location=float(usage.sum()/sums[6]))
    model.train()
    return metrics


def gradient_equivalence(config,variant):
    """Compare two-rank averaged gradients against the same global GPU batch."""
    model=create_model(config,variant)
    ddp=DDP(model,device_ids=[torch.cuda.current_device()],broadcast_buffers=False)
    # GroupNorm has sample-independent statistics, so splitting the batch is equivalent.
    torch.manual_seed(123)
    full=torch.rand(4,3,32,32,device='cuda')
    local=full.chunk(dist.get_world_size())[dist.get_rank()]
    ddp.zero_grad(set_to_none=True)
    for i in range(len(local)):
        context=ddp.no_sync() if i+1<len(local) else nullcontext()
        with context:
            (ddp(local[i:i+1])['reconstruction']-local[i:i+1]).square().mean().div(len(local)).backward()
    gradients={name:p.grad.detach().clone() for name,p in model.named_parameters()}
    reference=create_model(config,variant)
    reference.load_state_dict(model.state_dict())
    (reference(full)['reconstruction']-full).square().mean().backward()
    errors=[]
    for name,parameter in reference.named_parameters():
        torch.testing.assert_close(parameter.grad,gradients[name],rtol=5e-3,atol=3e-5)
        errors.append(float((parameter.grad-gradients[name]).abs().max()))
    del ddp,model,reference,gradients
    torch.cuda.empty_cache()
    return max(errors)


def smoke(config,variant):
    error=gradient_equivalence(config,variant)
    model=create_model(config,variant)
    ddp=DDP(model,device_ids=[torch.cuda.current_device()],broadcast_buffers=False)
    perceptual=perceptual_model()
    data=pixels(config,'train')
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
    rank,world=dist.get_rank(),dist.get_world_size()
    # Four global real photos, two disjoint photos per rank, twenty learning steps.
    prepared=np.array(data[np.arange(rank*2,(rank+1)*2)],copy=True)
    losses=[]
    for step in range(20):
        values,_,grad=distributed_update(ddp,optimizer,perceptual,prepared,config)
        losses.append(float(values[0]))
    if sum(losses[-5:])>=sum(losses[:5]) or grad<=0:
        raise AssertionError('DDP smoke did not demonstrate end-to-end learning')
    # Exercise the complete 1024-photo global batch, all four accumulation passes.
    ids=rank_ids(0,len(data),config['training']['effective_batch_size'],config['seed'],rank,world)
    prepared=np.array(data[ids],copy=True)
    values,usage,grad=distributed_update(ddp,optimizer,perceptual,prepared,config)
    delta=assert_parameters_synchronized(model)
    for parameter in [model.encoder.net[0].weight,model.decoder.net[0].weight]:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().sum()<=0:
            raise AssertionError('Encoder or decoder received no finite gradient')
    if model.sparse and (model.dictionary_raw.grad is None or model.dictionary_raw.grad.abs().sum()<=0):
        raise AssertionError('Dictionary received no reconstruction gradient')
    memories=[None]*world
    dist.all_gather_object(memories,torch.cuda.max_memory_allocated())
    if primary():
        write_json(Path(config['root'])/f'reports/{variant}.distributed_smoke.json',
            {'status':'passed','world_size':world,'global_batch_size':len(ids)*world,
             'per_gpu_microbatch':config['training']['micro_batch_size'],
             'accumulation_steps':len(ids)//config['training']['micro_batch_size'],
             'real_photos_small_smoke':4,'updates_small_smoke':20,'losses':losses,
             'full_global_batch_loss':float(values[0]),'gradient_norm':grad,
             'gradient_equivalence_max_abs_error':error,'rank_parameter_max_difference':delta,
             'per_rank_peak_memory_bytes':memories,'weights_discarded':True,'updated_utc':now()})
    dist.barrier()


def train(config,variant):
    root=Path(config['root']); output=root/'runs'/variant
    settings=config['training'];rank,world=dist.get_rank(),dist.get_world_size()
    data=pixels(config,'train'); validation=pixels(config,'validation')
    source=root/'origin_checkpoints'/variant/'checkpoint_resume.pt'
    resume=output/'checkpoint_resume.pt'
    saved=torch.load(resume if resume.exists() else source,map_location='cpu',weights_only=False)
    if saved['variant']!=variant or saved['config']['model']!=config['model'] or saved['config']['data']!=config['data']:
        raise ValueError('Continuation checkpoint model/data mismatch')
    model=create_model(config,variant);model.load_state_dict(saved['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    optimizer.load_state_dict(saved['optimizer'])
    # Leave the scalar Adam step on CPU; only moment buffers follow parameters.
    for state in optimizer.state.values():
        for key,value in state.items():
            if torch.is_tensor(value) and key!='step':state[key]=value.cuda()
    if resume.exists():
        if saved['config']!=config:
            raise ValueError('Distributed resume configuration mismatch')
        step,seen=saved['step'],saved['images_seen']
        base_step,base_seen=saved['base_optimizer_step'],saved['base_images_seen']
        rng=saved['rng_per_rank'][rank]
    else:
        step=0;base_step=saved['step']
        seen=base_seen=base_step*settings['previous_effective_batch_size']
        rng=saved
    torch.set_rng_state(rng['torch_rng']);torch.cuda.set_rng_state(rng['cuda_rng'])
    random.setstate(rng['python_rng']);np.random.set_state(rng['numpy_rng'])
    best=saved['best_validation_loss'];elapsed=saved['elapsed_seconds']
    usage=saved['gene_usage'].cuda() if model.sparse else None
    max_steps=math.ceil((settings['max_training_images']-base_seen)/settings['effective_batch_size'])
    ddp=DDP(model,device_ids=[torch.cuda.current_device()],broadcast_buffers=False)
    perceptual=perceptual_model();started=time.monotonic()
    if primary():
        write_json(output/'model_spec.json',{'variant':variant,'trainable_parameters':sum(p.numel() for p in model.parameters()),
            'world_size':world,'global_batch_size':settings['effective_batch_size'],
            'per_gpu_microbatch':settings['micro_batch_size'],'accumulation_steps':settings['effective_batch_size']//world//settings['micro_batch_size'],
            'continued_from':str(source),'base_optimizer_step':base_step,'base_images_seen':base_seen,
            'max_training_images_including_previous_phase':settings['max_training_images'],
            'lr_policy':'original cosine schedule indexed by cumulative samples; no LR multiplication',
            'optimizer_moments_restored':True,'labels_used':False,'updated_utc':now()})

    def checkpoint():
        rng={'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
             'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
        states=[None]*world
        dist.all_gather_object(states,rng)
        if primary():
            atomic_checkpoint(resume,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'config':config,'variant':variant,'step':step,'images_seen':seen,
                'base_optimizer_step':base_step,'base_images_seen':base_seen,
                'best_validation_loss':best,'elapsed_seconds':elapsed+time.monotonic()-started,
                'rng_per_rank':states,'gene_usage':usage.cpu() if usage is not None else None})
        dist.barrier()

    window=[]
    if primary():
        write_json(output/'progress.json',{'status':'training','variant':variant,'step':step,'max_updates':max_steps,
            'images_seen':seen,'base_images_seen':base_seen,'global_batch_size':settings['effective_batch_size'],
            'gpus_per_model':world,'updated_utc':now()})
    dist.barrier()
    while seen<settings['max_training_images']:
        if synchronized_stop(root):break
        batch=next_batch(seen,settings)
        ids=rank_ids(seen,len(data),batch,config['seed'],rank,world)
        read_started=time.monotonic();prepared=np.array(data[ids],copy=True);read_time=time.monotonic()-read_started
        lr=learning_rate(seen,settings)
        for group in optimizer.param_groups:group['lr']=lr
        values,batch_usage,_=distributed_update(ddp,optimizer,perceptual,prepared,config)
        if usage is not None:usage+=batch_usage
        step+=1;seen+=batch;window.append(values[:3])
        progress={'status':'training','variant':variant,'step':step,'max_updates':max_steps,
            'optimizer_updates_including_previous_phase':base_step+step,
            'images_seen':seen,'base_images_seen':base_seen,'dataset_passes':seen/len(data),
            'train_images':len(data),'effective_batch_size':settings['effective_batch_size'],
            'actual_global_batch_size':batch,'micro_batch_size_per_gpu':settings['micro_batch_size'],
            'gpus_per_model':world,'learning_rate':lr,'latent_rms':float(values[3]),
            'elapsed_seconds':round(elapsed+time.monotonic()-started,2),
            'continuation_elapsed_seconds':round(time.monotonic()-started,2),
            'rank0_last_RGB_read_seconds':round(read_time,3),
            'gpu_peak_memory_bytes_rank0':torch.cuda.max_memory_allocated(),
            'cpu_fallback':False,'updated_utc':now()}
        if step%settings['log_every']==0 or step==1:
            mean=np.mean(window,axis=0);window=[]
            progress.update(train_loss=float(mean[0]),train_l1=float(mean[1]),train_lpips=float(mean[2]),
                            best_validation_loss=best)
            if usage is not None:progress['genes_used_so_far']=int((usage>0).sum())
            if primary():
                append_json(output/'history.jsonl',progress);write_json(output/'progress.json',progress)
                print(json.dumps(progress),flush=True)
        due=(step%settings['evaluate_every']==0 or step==settings['first_evaluation'] or seen==settings['max_training_images'])
        if due:
            if primary():write_json(output/'progress.json',{**progress,'status':'validating'})
            result=evaluate(model,validation,config,perceptual)
            selected=result['selection_loss']<best;best=min(best,result['selection_loss'])
            result.update(step=step,images_seen=seen,variant=variant,split='validation',updated_utc=now())
            if primary():
                append_json(output/'validation_history.jsonl',result)
                if selected:
                    atomic_checkpoint(output/'checkpoint_best.pt',{'model':model.state_dict(),'config':config,
                        'variant':variant,'step':step,'images_seen':seen,'validation':result})
                plot_run(output);preview(model,validation,config,output,f'continuation_{step:06d}')
                write_json(output/'progress.json',{**progress,'status':'training','best_validation_loss':best})
            checkpoint()
        elif step%settings['checkpoint_every']==0:
            checkpoint()
    checkpoint()
    if seen<settings['max_training_images']:
        if primary():write_json(output/'progress.json',{'status':'stopped','step':step,'images_seen':seen,'updated_utc':now()})
        return
    # Preserve the previous phase best candidate if the continuation never improves it.
    best_path=output/'checkpoint_best.pt'
    if not best_path.exists():best_path=root/'origin_checkpoints'/variant/'checkpoint_best.pt'
    chosen=torch.load(best_path,map_location='cpu',weights_only=False);model.load_state_dict(chosen['model'])
    result=evaluate(model,pixels(config,'test'),config,perceptual)
    result.update(split='test',checkpoint=str(best_path),checkpoint_step=chosen['step'],variant=variant,
        checkpoint_images_seen=chosen.get('images_seen',chosen['step']*settings['previous_effective_batch_size']),
        evaluation_policy='one final test evaluation of the validation-selected checkpoint across both phases')
    if primary():
        write_json(output/'test_metrics.json',result);preview(model,pixels(config,'test'),config,output,'test_best')
        write_json(output/'progress.json',{'status':'complete','step':step,'max_updates':max_steps,
            'images_seen':seen,'best_validation_loss':best,'test':result,
            'elapsed_seconds':round(elapsed+time.monotonic()-started,2),'updated_utc':now()})
    dist.barrier()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--variant',required=True);parser.add_argument('--mode',choices=['smoke','train'],required=True)
    args=parser.parse_args();config=load_config(args.config)
    local_rank=int(os.environ['LOCAL_RANK'])
    if not torch.cuda.is_available() or torch.cuda.device_count()!=config['compute']['gpus_per_variant']:
        raise RuntimeError('DDP requires exactly two leased visible CUDA GPUs per model')
    torch.cuda.set_device(local_rank);torch.set_num_threads(4)
    dist.init_process_group('nccl',timeout=timedelta(minutes=15))
    if dist.get_world_size()!=config['compute']['gpus_per_variant']:
        raise RuntimeError('Unexpected DDP world size')
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    try:
        (smoke if args.mode=='smoke' else train)(config,args.variant)
        dist.destroy_process_group()
    except Exception as error:
        if primary():write_json(Path(config['root'])/f'reports/{args.variant}.{args.mode}_failure.json',
            {'status':'failed','error':str(error),'updated_utc':now()})
        raise
