"""Two-GPU stagewise training, frozen parent EMA, optional small-LR joint phase."""
import argparse
from contextlib import nullcontext
import copy
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
from torch.nn import functional as F

from common import append_json, checkpoint, load_config, now, write_json
from data import image_tensor, open_pixels, rank_ids
from metrics import ssim_per_image
from model import SequentialGene, objective
from preview import plot_history, preview

STOP = False


def stop_signal(*_):
    global STOP
    STOP = True


def amp(config):
    enabled = config['training']['precision'] == 'bfloat16' and torch.cuda.is_bf16_supported()
    return torch.autocast('cuda', dtype=torch.bfloat16) if enabled else nullcontext()


def stopped(root):
    v = torch.tensor(int(STOP or (root/'STOP_REQUESTED').exists()), device='cuda')
    dist.all_reduce(v, op=dist.ReduceOp.MAX)
    return bool(v)


@torch.inference_mode()
def evaluate(model, pixels, config, phase, limit=None, perceptual=None, subset=None):
    was_training = model.training
    model.eval()
    last = min(phase, 3)
    n = len(pixels) if limit is None else min(len(pixels), limit)
    totals = torch.zeros((last+1, 8), device='cuda', dtype=torch.float64)
    counts = [torch.zeros(k, device='cuda', dtype=torch.long) for k in config['model']['dictionary_sizes'][:last+1]]
    image_counts = [torch.zeros_like(v) for v in counts]
    subset_totals = torch.zeros(3, device='cuda', dtype=torch.float64)
    subset = set(subset or [])
    batch = config['training']['micro_batch_size']
    ids = np.arange(dist.get_rank(), n, dist.get_world_size())
    for start in range(0, len(ids), batch):
        take = ids[start:start+batch]
        x = image_tensor(pixels, take)
        with amp(config):
            result = model(x, last=last)
        for l in range(last+1):
            p, t, parent = result['outputs'][l].float(), result['targets'][l], result['parents'][l]
            raw = (p-t).abs().mean((1,2,3))
            clipped = p.clamp(0,1)
            mse = (clipped-t).square().mean((1,2,3))
            before = (parent-t).abs().mean((1,2,3))
            vals = [raw, -10*torch.log10(mse.clamp_min(1e-12)), ssim_per_image(clipped,t), before,
                    (raw < before).float(), ((p<0)|(p>1)).float().mean((1,2,3))]
            totals[l,:6] += torch.stack([v.sum() for v in vals]).double()
            if perceptual is not None and l == last:
                totals[l,6] += perceptual(clipped*2-1, t*2-1).sum().double()
            totals[l,7] += len(t)
            selected = result['indices'][l]
            counts[l] += torch.bincount(selected.reshape(-1), minlength=len(counts[l]))
            distinct = torch.zeros((len(t), len(counts[l])), device='cuda', dtype=torch.long)
            distinct.scatter_(1, selected.reshape(len(t),-1), 1)
            image_counts[l] += distinct.sum(0)
            if subset and l == last:
                mask = torch.tensor([int(i) in subset for i in take], device='cuda')
                subset_totals += torch.stack([raw[mask].sum(), vals[1][mask].sum(), mask.sum()]).double()
    dist.all_reduce(totals)
    dist.all_reduce(subset_totals)
    for v in counts+image_counts:
        dist.all_reduce(v)
    levels = []
    for l in range(last+1):
        assert int(totals[l,7]) == n
        v = (totals[l,:7]/n).cpu().tolist()
        levels.append(dict(level=l, mae_raw=v[0], psnr_clipped=v[1], ssim_clipped=v[2],
            parent_mae_same_resolution=v[3], fraction_images_improved=v[4], out_of_range_fraction=v[5],
            lpips_clipped=v[6] if perceptual is not None and l == last else None,
            genes_used=int((counts[l]>0).sum()), calls_per_gene=counts[l].cpu().tolist(),
            images_per_gene=image_counts[l].cpu().tolist(), codebook_resets=int(model.levels[l].book.resets)))
    metric = dict(images=n, levels=levels, selection_l1=float(np.mean([l['mae_raw'] for l in levels])) if phase==4 else levels[-1]['mae_raw'])
    if subset_totals[2] > 0:
        metric['unseen_training_cities_subset'] = dict(images=int(subset_totals[2]),
            mae_raw=float(subset_totals[0]/subset_totals[2]), psnr_clipped=float(subset_totals[1]/subset_totals[2]),
            limitation='Subset of previously inspected existing test set; not a new external holdout.')
    model.train(was_training)
    return metric


def save_state(path, model, optimizer, config, phase, step, best, bad, status):
    if dist.get_rank() == 0:
        checkpoint(path, dict(model=model.state_dict(), optimizer=optimizer.state_dict() if optimizer else None,
            config=config, phase=phase, step=step, best=best, bad=bad, status=status, updated_utc=now()))
    dist.barrier()


def run(config, smoke):
    if not torch.cuda.is_available():
        raise RuntimeError('No CPU training: wait for leased GPUs')
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        raise RuntimeError('This experiment requires exactly two leased GPU ranks')
    root = Path(config['root'])
    runroot = root/'runs'/('smoke' if smoke else 'main')
    runroot.mkdir(exist_ok=True)
    settings = config['training']
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.backends.cudnn.benchmark = True
    model = SequentialGene(config['model']).cuda()
    pixels = {s: open_pixels(config,s) for s in ['train','validation','test']}
    saved = torch.load(runroot/'checkpoint_last.pt', map_location='cpu', weights_only=False) if (runroot/'checkpoint_last.pt').exists() else None
    if saved:
        if saved['config'] != config:
            raise RuntimeError('Configuration mismatch: explicit new run required')
        model.load_state_dict(saved['model'])
        if saved['status'] == 'complete':
            return
    first_phase = saved['phase'] if saved else 0
    started = time.monotonic()
    batch = settings['effective_batch_size']
    local_batch = batch//world
    micro = settings['micro_batch_size']
    if rank == 0:
        write_json(runroot/'protocol.json', dict(sequential=True, phases=[0,1,2,3,4],
            parent_condition='actual decoded prediction', loss='MAE + beta commitment',
            cpu_model_execution=False, started_utc=now(), smoke=smoke))
    for phase in range(first_phase, 5):
        name = f'L{phase}' if phase<4 else 'joint'
        model.train()
        model.set_phase(phase)
        ddp = DDP(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
            lr=settings['joint_learning_rate'] if phase==4 else settings['learning_rate'], weight_decay=settings['weight_decay'])
        step0, best, bad = 0, float('inf'), 0
        if saved and saved['phase']==phase:
            step0, best, bad = saved['step'], saved['best'], saved['bad']
            if saved['optimizer']:
                opt.load_state_dict(saved['optimizer'])
            saved = None
        max_steps = 4 if smoke else (settings['joint_updates'] if phase==4 else settings['max_stage_updates'])
        min_steps = max_steps if smoke else settings['min_stage_updates']
        if step0 == 0:
            metrics = evaluate(model,pixels['validation'],config,phase,limit=16 if smoke else None)
            best = metrics['selection_l1']
            save_state(runroot/f'best_{name}.pt',model,opt,config,phase,0,best,0,'training')
            if rank==0:
                append_json(runroot/'history.jsonl',dict(event='validation',phase=phase,step=0,metrics=metrics,updated_utc=now()))
        for step in range(step0+1,max_steps+1):
            tick = time.monotonic()
            if stopped(root):
                save_state(runroot/'checkpoint_last.pt',model,opt,config,phase,step-1,best,bad,'stopped')
                if rank==0:
                    write_json(runroot/'progress.json',dict(status='stopped',phase=name,step=step-1,updated_utc=now()))
                return
            lr0 = settings['joint_learning_rate'] if phase==4 else settings['learning_rate']
            warmup = min(1.,step/25)
            lr = lr0*warmup*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/max_steps)))
            for group in opt.param_groups:
                group['lr']=lr
            # All ranks use the same global stream; every batch has unique grids.
            offset = (phase*settings['max_stage_updates']+step-1)*batch
            ids = rank_ids(offset,len(pixels['train']),batch,config['seed'],rank,world)
            if smoke:
                ids = np.arange(rank*local_batch,(rank+1)*local_batch) % 16
            opt.zero_grad(set_to_none=True)
            statistics = torch.zeros(3,device='cuda',dtype=torch.float64)
            for start in range(0,local_batch,micro):
                take = ids[start:start+micro]
                with ddp.no_sync() if start+micro<local_batch else nullcontext():
                    x = image_tensor(pixels['train'],take)
                    with amp(config):
                        result = ddp(x)
                    loss,recon,commit = objective(result,phase,config['model']['beta'])
                    (loss*(len(take)/local_batch)).backward()
                statistics += torch.stack([loss.detach(),recon.detach(),commit.detach()]).double()*(len(take)/local_batch)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'])
            finite = torch.tensor(int(torch.isfinite(statistics).all() and torch.isfinite(grad)),device='cuda')
            dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            if not bool(finite):
                raise FloatingPointError('Nonfinite loss/gradient')
            opt.step()
            dist.all_reduce(statistics)
            statistics /= world
            seconds = time.monotonic()-tick
            if rank==0 and (step==1 or step%settings['log_every']==0 or smoke):
                row=dict(event='train',phase=phase,step=step,loss=float(statistics[0]),l1=float(statistics[1]),
                    commitment=float(statistics[2]),gradient_norm=float(grad),lr=lr,seconds_per_update=seconds,
                    global_batch=batch,images_per_second=batch/seconds,updated_utc=now())
                append_json(runroot/'history.jsonl',row)
                write_json(runroot/'progress.json',dict(status='training',phase=name,step=step,max_phase_steps=max_steps,
                    training=row,elapsed_seconds=time.monotonic()-started,updated_utc=now()))
                print(json.dumps(row),flush=True)
            do_eval = step==max_steps or step==settings['first_evaluation'] or step%settings['evaluate_every']==0
            if do_eval:
                metrics=evaluate(model,pixels['validation'],config,phase,limit=16 if smoke else None)
                improved = metrics['selection_l1'] < best-settings['minimum_improvement']
                if improved:
                    best,bad = metrics['selection_l1'],0
                    save_state(runroot/f'best_{name}.pt',model,opt,config,phase,step,best,bad,'training')
                else:
                    bad += 1
                if rank==0:
                    append_json(runroot/'history.jsonl',dict(event='validation',phase=phase,step=step,metrics=metrics,updated_utc=now()))
                    write_json(runroot/f'validation_{name}.json',metrics)
                    x=image_tensor(pixels['validation'],np.arange(settings['preview_count']))
                    with amp(config):
                        preview(model,x,runroot/f'preview_{name}.png',min(phase,3))
                    plot_history(runroot)
                    print(json.dumps(dict(event='validation',phase=name,step=step,selection_l1=metrics['selection_l1'],best=best,bad=bad)),flush=True)
                dist.barrier()
            if step%settings['checkpoint_every']==0 or step==max_steps or do_eval:
                save_state(runroot/'checkpoint_last.pt',model,opt,config,phase,step,best,bad,'training')
            if step>=min_steps and bad>=settings['plateau_patience']:
                break
        # The next level sees the best actual decoded parent, never the GT parent.
        dist.barrier()
        best_state=torch.load(runroot/f'best_{name}.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(best_state['model'])
        del ddp,opt
        if phase==3:
            save_state(runroot/'checkpoint_stagewise.pt',model,None,config,phase,step,best,bad,'stagewise_complete')
        if phase<4:
            save_state(runroot/'checkpoint_last.pt',model,None,config,phase+1,0,float('inf'),0,'training')
    perceptual=None
    if not smoke:
        import lpips
        perceptual=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
    ready=json.loads((root/'data/READY.json').read_text())
    test=evaluate(model,pixels['test'],config,4,limit=16 if smoke else None,perceptual=perceptual,
                  subset=ready['unseen_city_test_indices'])
    if rank==0:
        write_json(runroot/'test_metrics.json',test)
    save_state(runroot/'checkpoint_last.pt',model,None,config,4,step,best,bad,'complete')
    if rank==0:
        write_json(runroot/'progress.json',dict(status='complete',test_metrics=test,updated_utc=now()))
        write_json(root/'reports'/('distributed_smoke.json' if smoke else 'training_complete.json'),
            dict(status='passed' if smoke else 'complete',run=str(runroot),updated_utc=now()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--mode',choices=['smoke','train'],required=True)
    args=parser.parse_args()
    config=load_config(args.config)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    torch.set_num_threads(4)
    signal.signal(signal.SIGTERM,stop_signal)
    signal.signal(signal.SIGINT,stop_signal)
    try:
        run(config,args.mode=='smoke')
    finally:
        dist.destroy_process_group()
