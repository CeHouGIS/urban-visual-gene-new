"""One GPU per independent variant; RGB-supervised, resumable, no CPU fallback."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import signal
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from common import append_json, load_config, now, write_json
from data import image_tensor, indices_for_step, open_pixels
from metrics import ssim_per_image
from model import GeneAE
from report import plot_run

STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def atomic_checkpoint(path, state):
    temporary = path.with_suffix('.pt.partial')
    torch.save(state,temporary)
    temporary.replace(path)


def create_model(config, variant):
    torch.manual_seed(config['seed'])
    return GeneAE(variant,**config['model']).cuda()


def perceptual_model():
    import lpips
    # AlexNet weights are already present in the local torch cache.
    result = lpips.LPIPS(net='alex',verbose=False).cuda().eval()
    result.requires_grad_(False)
    return result


def amp_context(config):
    enabled=config['training']['precision']=='bfloat16_if_supported' and torch.cuda.is_bf16_supported()
    return torch.autocast('cuda',dtype=torch.bfloat16) if enabled else nullcontext()


def reconstruction_loss(prediction, target, perceptual, weight):
    prediction=prediction.float()
    l1=F.l1_loss(prediction,target)
    lp=perceptual(prediction*2-1,target*2-1).mean()
    return l1 + weight*lp, l1, lp


@torch.inference_mode()
def evaluate(model, pixels, config, perceptual):
    model.eval()
    totals=dict.fromkeys(['l1','mse','psnr_mean_per_image','ssim','lpips'],0.0)
    usage=torch.zeros(config['model']['dictionary_size'],dtype=torch.int64,device='cuda') if model.sparse else None
    positions=0
    micro=config['training']['micro_batch_size']
    for start in range(0,len(pixels),micro):
        target=image_tensor(pixels,np.arange(start,min(start+micro,len(pixels))),'cuda')
        with amp_context(config):
            out=model(target)
        prediction=out['reconstruction'].float()
        mse=(prediction-target).square().mean((1,2,3))
        values={'l1':(prediction-target).abs().mean((1,2,3)), 'mse':mse,
                'psnr_mean_per_image':-10*torch.log10(mse.clamp_min(1e-12)),
                'ssim':ssim_per_image(prediction,target),
                'lpips':perceptual(prediction*2-1,target*2-1).flatten()}
        for name,value in values.items():
            totals[name]+=float(value.sum())
        if usage is not None:
            act=out['activations']
            usage+=(act>0).sum((0,2,3))
            positions+=act.shape[0]*act.shape[2]*act.shape[3]
    result={k:v/len(pixels) for k,v in totals.items()}
    result['psnr_from_mean_mse']=-10*math.log10(max(result['mse'],1e-12))
    result['selection_loss']=result['l1']+config['training']['perceptual_weight']*result['lpips']
    result['images']=len(pixels)
    if usage is not None:
        result.update(gene_activation_counts=usage.cpu().tolist(),
            genes_used=int((usage>0).sum()),inactive_genes_on_this_split=int((usage==0).sum()),
            mean_active_per_location=float(usage.sum())/positions)
    model.train()
    return result


def to_pil(tensor):
    array=tensor.detach().float().cpu().clamp(0,1).permute(1,2,0).numpy()
    return Image.fromarray(np.rint(array*255).astype(np.uint8))


@torch.inference_mode()
def preview(model, pixels, config, output, label):
    model.eval()
    number=min(config['training']['preview_count'],len(pixels))
    images=image_tensor(pixels,np.arange(number),'cuda')
    out=model(images)
    side=images.shape[-1]
    sheet=Image.new('RGB',(side*3,(side+22)*number),'white')
    draw=ImageDraw.Draw(sheet)
    for i in range(number):
        y=i*(side+22)
        for col,(name,tensor) in enumerate([
            ('Original real photo',images[i]),('RGB reconstruction',out['reconstruction'][i]),
            ('Absolute error x 3',(images[i]-out['reconstruction'][i]).abs()*3)]):
            draw.text((col*side+3,y+3),name,fill='black')
            sheet.paste(to_pil(tensor),(col*side,y+22))
    directory=output/'previews'
    directory.mkdir(exist_ok=True)
    sheet.save(directory/f'{label}.png')
    if model.sparse:
        activations=out['activations']
        values,indices=activations.topk(model.active,dim=1)
        np.savez_compressed(directory/f'{label}.codes.npz',
            indices=indices.cpu().numpy().astype(np.uint16),
            values=values.cpu().numpy().astype(np.float32),
            dictionary=model.dictionary.cpu().numpy(),
            activation_shape=np.array(activations.shape),
            note=np.array('Feature-space genes; activations are not instance masks.'))
        # Fixed first two examples, strongest four genes by total activation.
        rows=min(2,number)
        edits=Image.new('RGB',(side*6,(side+22)*rows),'white')
        painter=ImageDraw.Draw(edits)
        maps=Image.new('RGB',(side*5,(side+22)*rows),'white')
        map_draw=ImageDraw.Draw(maps)
        records=[]
        for i in range(rows):
            gene_ids=activations[i].sum((1,2)).topk(4).indices.tolist()
            y=i*(side+22)
            tiles=[('Original',images[i]),('Reconstruction',out['reconstruction'][i])]
            maps.paste(to_pil(images[i]),(0,y+22))
            map_draw.text((3,y+3),'Original',fill='black')
            for col,gene in enumerate(gene_ids):
                contribution=model.contribution(activations[i:i+1],gene)
                removed=model.decoder(out['latent'][i:i+1]-contribution)[0]
                tiles.append((f'Remove gene {gene}',removed))
                heat=F.interpolate(activations[i:i+1,gene:gene+1],size=(side,side),mode='nearest')[0,0]
                heat=heat/heat.max().clamp_min(1e-8)
                color=torch.stack([heat,heat*0.2,heat*0.05])
                overlay=images[i]*0.5+color*0.5
                maps.paste(to_pil(overlay),((col+1)*side,y+22))
                map_draw.text(((col+1)*side+3,y+3),f'Gene {gene} activation',fill='black')
                records.append({'sample_index':i,'gene':gene,
                    'ablation_rgb_mae':float((removed-out['reconstruction'][i]).abs().mean())})
            for col,(title,tensor) in enumerate(tiles):
                painter.text((col*side+3,y+3),title,fill='black')
                edits.paste(to_pil(tensor),(col*side,y+22))
        edits.save(directory/f'{label}.gene_removal.png')
        maps.save(directory/f'{label}.gene_activations.png')
        write_json(directory/f'{label}.gene_removal.json',records)
    model.train()


def gpu_smoke(config, variant):
    model=create_model(config,variant)
    perceptual=perceptual_model()
    pixels=open_pixels(config,'train')
    target=image_tensor(pixels,np.arange(4),'cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
    losses=[]
    encoder_grad=dictionary_grad=decoder_grad=0.0
    for step in range(20):
        if STOP:
            return
        optimizer.zero_grad(set_to_none=True)
        with amp_context(config):
            result=model(target)
        loss,_,_=reconstruction_loss(result['reconstruction'],target,perceptual,0.1)
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(loss) or not torch.isfinite(grad):
            raise FloatingPointError('GPU smoke has nonfinite loss/gradient')
        encoder_grad=max(encoder_grad,float(model.encoder.net[0].weight.grad.norm()))
        decoder_grad=max(decoder_grad,float(model.decoder.net[0].weight.grad.norm()))
        if model.sparse:
            dictionary_grad=max(dictionary_grad,float(model.dictionary_raw.grad.norm()))
            if int((result['activations']>0).sum(1).max())>model.active:
                raise AssertionError('Sparsity constraint violated')
        optimizer.step()
        losses.append(float(loss.detach()))
    if sum(losses[-5:]) >= sum(losses[:5]) or encoder_grad<=0 or decoder_grad<=0:
        raise AssertionError('GPU smoke does not demonstrate end-to-end learning')
    if model.sparse and dictionary_grad<=0:
        raise AssertionError('Dictionary received no reconstruction gradient')
    # Check the actual configured microbatch before starting a long 1M run.
    optimizer.zero_grad(set_to_none=True)
    target=image_tensor(pixels,np.arange(config['training']['micro_batch_size']),'cuda')
    with amp_context(config):
        full=model(target)
    full_loss,_,_=reconstruction_loss(full['reconstruction'],target,perceptual,config['training']['perceptual_weight'])
    full_loss.backward()
    full_grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    if not torch.isfinite(full_loss) or not torch.isfinite(full_grad):
        raise FloatingPointError('Configured microbatch failed the GPU gradient check')
    write_json(Path(config['root'])/f'reports/{variant}.gpu_smoke.json',
        {'status':'passed','real_training_photos':4,'updates':20,'losses':losses,
         'encoder_gradient_max':encoder_grad,'decoder_gradient_max':decoder_grad,
         'dictionary_gradient_max':dictionary_grad if model.sparse else None,
         'micro_batch_size_verified':config['training']['micro_batch_size'],
         'full_microbatch_loss':float(full_loss.detach()),
         'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),
         'weights_discarded':True,'updated_utc':now()})


def train(config,variant):
    global STOP
    output=Path(config['root'])/'runs'/variant
    settings=config['training']
    train_pixels=open_pixels(config,'train')
    validation=open_pixels(config,'validation')
    model=create_model(config,variant)
    write_json(output/'model_spec.json',{'variant':variant,'trainable_parameters':sum(p.numel() for p in model.parameters()),
        'dictionary_parameters':model.dictionary_raw.numel() if model.sparse else 0,
        'latent_channels':config['model']['latent_channels'],'latent_grid':[16,16],
        'train_images':config['data']['train'],'initialization':'random seed 42, matched shared trunks',
        'labels_used':False,'updated_utc':now()})
    perceptual=perceptual_model()
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    step,best,elapsed=0,float('inf'),0.0
    gene_usage=torch.zeros(config['model']['dictionary_size'],dtype=torch.int64,device='cuda') if model.sparse else None
    resume=output/'checkpoint_resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        if saved['config']!=config or saved['variant']!=variant:
            raise ValueError('Resume configuration mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        for state in optimizer.state.values():
            for key,value in state.items():
                if torch.is_tensor(value):
                    state[key]=value.cuda()
        step,best,elapsed=saved['step'],saved['best_validation_loss'],saved['elapsed_seconds']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        if gene_usage is not None:
            gene_usage.copy_(saved['gene_usage'].cuda())
    started=time.monotonic()
    def checkpoint():
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'config':config,'variant':variant,'step':step,
            'best_validation_loss':best,'elapsed_seconds':elapsed+time.monotonic()-started,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
            'gene_usage':gene_usage.cpu() if gene_usage is not None else None}
        atomic_checkpoint(resume,state)
    write_json(output/'progress.json',{'status':'training','step':step,'max_updates':settings['max_updates'],
        'train_images':len(train_pixels),'variant':variant,'updated_utc':now()})
    if step==0:
        preview(model,validation,config,output,'step_000000')
    window=[]
    while step<settings['max_updates'] and not STOP:
        if (Path(config['root'])/'STOP_REQUESTED').exists():
            STOP=True
            break
        ids=indices_for_step(step,len(train_pixels),settings['effective_batch_size'],config['seed'])
        # Decode cache misses together using the loader thread pool, then split
        # the prepared RGB batch into GPU microbatches.
        prepared_pixels=train_pixels[ids]
        if step<settings['warmup_updates']:
            lr=settings['learning_rate']*(step+1)/settings['warmup_updates']
        else:
            fraction=(step-settings['warmup_updates'])/max(1,settings['max_updates']-settings['warmup_updates']-1)
            lr=settings['minimum_learning_rate']+(settings['learning_rate']-settings['minimum_learning_rate'])*0.5*(1+math.cos(math.pi*fraction))
        for group in optimizer.param_groups:
            group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        losses=np.zeros(3,dtype=np.float64)
        latent_rms=0.0
        for begin in range(0,len(ids),settings['micro_batch_size']):
            selected=ids[begin:begin+settings['micro_batch_size']]
            target=image_tensor(prepared_pixels,np.arange(begin,begin+len(selected)),'cuda')
            with amp_context(config):
                out=model(target)
            loss,l1,lp=reconstruction_loss(out['reconstruction'],target,perceptual,settings['perceptual_weight'])
            weight=len(selected)/len(ids)
            (loss*weight).backward()
            losses+=np.array([float(v.detach()) for v in [loss,l1,lp]])*weight
            latent_rms+=float(out['latent'].detach().float().square().mean().sqrt())*weight
            if gene_usage is not None:
                gene_usage+=(out['activations'].detach()>0).sum((0,2,3))
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'])
        if not np.isfinite(losses).all() or not torch.isfinite(grad):
            raise FloatingPointError('Training has nonfinite loss/gradient')
        optimizer.step()
        step+=1
        window.append(losses)
        progress={'status':'training','variant':variant,'step':step,'max_updates':settings['max_updates'],
            'train_images':len(train_pixels),'effective_batch_size':settings['effective_batch_size'],
            'images_seen':step*settings['effective_batch_size'],
            'dataset_passes':step*settings['effective_batch_size']/len(train_pixels),
            'learning_rate':lr,'latent_rms':latent_rms,
            'elapsed_seconds':round(elapsed+time.monotonic()-started,2),
            'gpu_peak_memory_bytes':torch.cuda.max_memory_allocated(),'cpu_fallback':False,'updated_utc':now()}
        if hasattr(train_pixels,'stats'):
            progress['data_loading']=train_pixels.stats()
        if step%settings['log_every']==0 or step==1:
            average=np.array(window).mean(0)
            progress.update(train_loss=float(average[0]),train_l1=float(average[1]),train_lpips=float(average[2]),
                best_validation_loss=best if math.isfinite(best) else None)
            if gene_usage is not None:
                progress.update(genes_used_so_far=int((gene_usage>0).sum()))
            append_json(output/'history.jsonl',progress)
            write_json(output/'progress.json',progress)
            print(json.dumps(progress),flush=True)
            window=[]
        evaluation_due=(step%settings['evaluate_every']==0 or step==settings.get('first_evaluation',settings['evaluate_every']) or step==settings['max_updates'])
        if evaluation_due:
            write_json(output/'progress.json',{**progress,'status':'validating'})
            result=evaluate(model,validation,config,perceptual)
            selected=result['selection_loss']<best
            best=min(best,result['selection_loss'])
            result.update(step=step,variant=variant,split='validation',updated_utc=now())
            append_json(output/'validation_history.jsonl',result)
            plot_run(output)
            if selected:
                atomic_checkpoint(output/'checkpoint_best.pt',{'model':model.state_dict(),
                    'config':config,'variant':variant,'step':step,'validation':result})
            checkpoint()
            preview(model,validation,config,output,f'step_{step:06d}')
            write_json(output/'progress.json',{**progress,'status':'training','best_validation_loss':best})
        elif step%settings.get('checkpoint_every',settings['evaluate_every'])==0:
            checkpoint()
    checkpoint()
    if STOP:
        write_json(output/'progress.json',{'status':'stopped','step':step,'updated_utc':now()})
        return
    # The held-out test is opened only after checkpoint selection is complete.
    chosen=torch.load(output/'checkpoint_best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(chosen['model'])
    test=open_pixels(config,'test')
    result=evaluate(model,test,config,perceptual)
    result.update(split='test',checkpoint_step=chosen['step'],variant=variant,
        evaluation_policy='one final test evaluation of the validation-selected checkpoint')
    write_json(output/'test_metrics.json',result)
    preview(model,test,config,output,'test_best')
    write_json(output/'progress.json',{'status':'complete','step':step,'max_updates':settings['max_updates'],
        'best_checkpoint_step':chosen['step'],'best_validation_loss':best,
        'test':{k:v for k,v in result.items() if k!='gene_activation_counts'},
        'elapsed_seconds':round(elapsed+time.monotonic()-started,2),'updated_utc':now()})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--variant',required=True)
    parser.add_argument('--mode',choices=['smoke','train'],default='train')
    args=parser.parse_args()
    config=load_config(args.config)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Worker requires exactly one leased, visible CUDA GPU')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    try:
        (gpu_smoke if args.mode=='smoke' else train)(config,args.variant)
    except Exception as error:
        output=Path(config['root'])/('reports' if args.mode=='smoke' else f'runs/{args.variant}')
        write_json(output/f'{args.mode}_failure.json',{'status':'failed','error':str(error),'updated_utc':now()})
        raise
