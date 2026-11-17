"""Two-rank GPU training from scratch, matched samples, resumable checkpoints."""
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
from torch.nn import functional as F

from common import append_json, load_config, now, write_json, resume_compatible
from data import image_tensor, learning_rate, open_pixels, rank_ids
from metrics import ssim_per_image
from model import create_model
from losses import objective
from preview import preview
from report import plot_run


STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def primary():
    return dist.get_rank() == 0


def stop_requested(root):
    value = torch.tensor(int(STOP or (root / 'STOP_REQUESTED').exists()), device='cuda')
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def amp_context(config):
    enabled = config['training']['precision'] == 'bfloat16_if_supported' and torch.cuda.is_bf16_supported()
    return torch.autocast('cuda', dtype=torch.bfloat16) if enabled else nullcontext()


def perceptual_model():
    import lpips
    result = lpips.LPIPS(net='alex', verbose=False).cuda().eval()
    result.requires_grad_(False)
    return result


def reconstruction_loss(prediction, target, perceptual, weight):
    prediction = prediction.float()
    l1 = F.l1_loss(prediction, target)
    lp = perceptual(prediction * 2 - 1, target * 2 - 1).mean()
    return l1 + weight * lp, l1, lp


def atomic_checkpoint(path, state):
    partial = path.with_name(path.name + '.partial')
    torch.save(state, partial)
    partial.replace(path)


def distributed_update(ddp, optimizer, perceptual, prepared, config):
    optimizer.zero_grad(set_to_none=True)
    micro = config['training']['micro_batch_size']
    statistics = torch.zeros(11, device='cuda', dtype=torch.float64)
    usage = torch.zeros(config['model']['dictionary_size'], device='cuda', dtype=torch.int64)
    for begin in range(0, len(prepared), micro):
        end = min(begin + micro, len(prepared))
        context = ddp.no_sync() if end < len(prepared) else nullcontext()
        with context:
            target = image_tensor(prepared, np.arange(begin, end))
            with amp_context(config):
                output = ddp(target)
            loss, parts = objective(output, target, ddp.module, config, perceptual)
            l1, lp = parts['l1'], parts['lpips']
            weight = (end - begin) / len(prepared)
            (loss * weight).backward()
        active_genes,encoded_bytes=ddp.module.actual_complexity(output)
        expected_genes,expected_bytes=ddp.module.expected_complexity(output)
        statistics += torch.stack([loss.detach(), l1.detach(), lp.detach(),
                                   output['latent'].detach().float().square().mean().sqrt(),
                                   parts['coding_objective'].detach(), parts['pyramid_loss'].detach(),
                                   parts['rgb_loss'].detach(), active_genes.float().mean(),encoded_bytes.float().mean(),
                                   expected_genes.detach().mean(),expected_bytes.detach().mean()]).double() * weight
        if ddp.module.sparse:
            usage += ddp.module.usage(output['activations'])
        del output, target, loss, l1, lp
    grad = torch.nn.utils.clip_grad_norm_(ddp.module.parameters(), config['training']['gradient_clip'])
    finite = torch.tensor(int(torch.isfinite(statistics).all() and torch.isfinite(grad)), device='cuda')
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        raise FloatingPointError('Nonfinite training statistics or gradients')
    optimizer.step()
    dist.all_reduce(statistics)
    statistics /= dist.get_world_size()
    dist.all_reduce(usage)
    return statistics.cpu().numpy(), usage, float(grad)


@torch.inference_mode()
def evaluate_one(model,pixels,config,perceptual):
    rank,world=dist.get_rank(),dist.get_world_size()
    ids=np.arange(rank,len(pixels),world)
    totals=torch.zeros(8,device='cuda',dtype=torch.float64)
    presence=torch.zeros(model.settings['dictionary_size'],device='cuda',dtype=torch.int64)
    additive=torch.zeros(3,device='cuda');rows=[]
    micro=min(32,config['training']['micro_batch_size'])
    for begin in range(0,len(ids),micro):
        indices=ids[begin:begin+micro];target=image_tensor(pixels,indices)
        with amp_context(config):result=model(target)
        pred=result['reconstruction'].float()
        if not torch.isfinite(pred).all():raise FloatingPointError('Nonfinite evaluation')
        mse=(pred-target).square().mean((1,2,3))
        values=[(pred-target).abs().mean((1,2,3)),mse,-10*torch.log10(mse.clamp_min(1e-12)),ssim_per_image(pred,target),perceptual(pred*2-1,target*2-1).flatten()]
        genes,payload=model.actual_complexity(result)
        if bool((genes>model.settings['budgets'][model.rate_index]).any()):raise AssertionError('Whole-gene budget violated')
        totals[:5]+=torch.stack([v.sum() for v in values]).double();totals[5]+=len(target)
        totals[6]+=genes.sum();totals[7]+=payload.sum()
        presence+=torch.cat([(a!=0).any((2,3)).sum(0) for a in result['activations'].values()])
        for scale,act in result['activations'].items():
            if bool(((act!=0).any((2,3)).sum(1)>model.quota(scale)).any()):raise AssertionError('Per-scale quota violated')
        for i,index in enumerate(indices):rows.append({'index':int(index),'active_genes':int(genes[i]),'encoded_bytes':int(payload[i]),'l1':float(values[0][i]),'psnr':float(values[2][i]),'lpips':float(values[4][i])})
        if begin==0:
            for i,scale in enumerate(model.scales):
                act=result['activations'][scale][:1].float();left=act.clone();left[:,1::2]=0
                full=model.synthesize_scale(scale,act)
                error=model.synthesize_scale(scale,left)+model.synthesize_scale(scale,act-left)-full
                additive[i]=error.norm()/full.norm().clamp_min(1e-8)
    dist.all_reduce(totals);dist.all_reduce(presence);dist.all_reduce(additive,op=dist.ReduceOp.MAX)
    all_rows=[None]*world;dist.all_gather_object(all_rows,rows)
    rows=sorted([row for part in all_rows for row in part],key=lambda row:row['index'])
    if int(totals[5])!=len(pixels) or [r['index'] for r in rows]!=list(range(len(pixels))):raise AssertionError('Evaluation duplicated or omitted data')
    result={key:float(totals[i]/totals[5]) for i,key in enumerate(['l1','mse','psnr_mean_per_image','ssim','lpips'])}
    distribution=presence.double()/presence.sum().clamp_min(1)
    entropy=-(distribution*distribution.clamp_min(1e-12).log()).sum()
    result.update(images=len(pixels),selection_loss=result['l1']+config['training']['perceptual_weight']*result['lpips'],gene_cap=model.settings['budgets'][model.rate_index],mean_active_genes=float(totals[6]/totals[5]),max_active_genes=max(r['active_genes'] for r in rows),mean_encoded_bytes=float(totals[7]/totals[5]),max_encoded_bytes=max(r['encoded_bytes'] for r in rows),dataset_genes_used=int((presence>0).sum()),dataset_genes_used_in_at_least_ten_percent=int((presence>=.1*len(pixels)).sum()),effective_genes_from_usage_entropy=float(entropy.exp()) if int(presence.sum()) else 0.,per_gene_image_count=presence.cpu().tolist(),per_gene_image_fraction=(presence.float()/len(pixels)).cpu().tolist(),feature_additivity_relative_error_fp32=additive.cpu().tolist(),per_image=rows)
    return result


def evaluate(model,pixels,config,perceptual):
    prior_rate,prior_training=model.rate_index,model.training
    model.eval();rates={}
    try:
        for rate in range(len(config['model']['budgets'])):
            model.rate_index=rate;rates[str(rate)]=evaluate_one(model,pixels,config,perceptual)
    finally:
        model.rate_index=prior_rate;model.train(prior_training)
    result={key:sum(x[key] for x in rates.values())/len(rates) for key in ['l1','mse','psnr_mean_per_image','ssim','lpips','selection_loss']}
    result.update(rates=rates,count=len(pixels),selection_policy='mean validation RGB objective across four caps; actual counts/bytes reported per cap')
    return result


def gradient_equivalence(config, variant, local_count=2, accumulated_micro=1):
    # Hard TopK is discontinuous at its cutoff. Compare gradients in true FP32
    # so TF32 convolution rounding cannot change the selected sparse support
    # merely because cuDNN chooses another kernel for a different batch shape.
    old_conv = torch.backends.cudnn.allow_tf32
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return _gradient_equivalence(config, variant, local_count, accumulated_micro)
    finally:
        torch.backends.cudnn.allow_tf32 = old_conv
        torch.backends.cuda.matmul.allow_tf32 = old_matmul


def _gradient_equivalence(config, variant, local_count, accumulated_micro):
    model = create_model(config, variant)
    model.set_step(config['model']['group_cap_ramp_updates'][1])
    model.eval() # Disable gate sampling only for the gradient equivalence check.
    ddp = DDP(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
    torch.manual_seed(123)
    full = torch.rand(local_count * dist.get_world_size(), 3, 32, 32, device='cuda')
    local = full.chunk(dist.get_world_size())[dist.get_rank()]
    ddp.zero_grad(set_to_none=True)
    for i in range(0, len(local), accumulated_micro):
        end = min(i + accumulated_micro, len(local))
        with ddp.no_sync() if end < len(local) else nullcontext():
            (objective(ddp(local[i:end]), local[i:end], model, config)[0]
             * ((end-i) / len(local))).backward()
    gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    reference = create_model(config, variant)
    reference.load_state_dict(model.state_dict())
    reference.set_step(model.step)
    reference.eval()
    objective(reference(full), full, reference, config)[0].backward()
    errors = []
    for name, parameter in reference.named_parameters():
        if not parameter.requires_grad: continue
        torch.testing.assert_close(parameter.grad, gradients[name], rtol=5e-3, atol=3e-5,
                                   msg=lambda message: name + ': ' + message)
        errors.append(float((parameter.grad - gradients[name]).abs().max()))
    del ddp, model, reference, gradients
    torch.cuda.empty_cache()
    return max(errors)


def synchronized_parameters(model):
    difference = torch.zeros((), device='cuda')
    for parameter in model.parameters():
        reference = parameter.detach().clone()
        dist.broadcast(reference, src=0)
        difference = torch.maximum(difference, (parameter.detach() - reference).abs().max())
    dist.all_reduce(difference, op=dist.ReduceOp.MAX)
    if float(difference) > 1e-6:
        raise AssertionError('DDP rank parameters diverged')
    return float(difference)


@torch.inference_mode()
def profile_macs(model, side):
    count = [0]
    handles = []
    def hook(module, inputs, output):
        count[0] += output.numel() * module.in_channels // module.groups * math.prod(module.kernel_size)
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(hook))
    output = model(torch.zeros(1, 3, side, side, device='cuda'))
    for h in handles:
        h.remove()
    if model.sparse:
        for scale, activation in output['activations'].items():
            dictionary = model.dictionaries[scale]
            count[0] += (2*activation.shape[-2])*(2*activation.shape[-1])*dictionary.raw.numel()
    return count[0]


def smoke(config, variant):
    error = gradient_equivalence(config, variant)
    model = create_model(config, variant)
    model.set_step(config['model']['group_cap_ramp_updates'][1])
    ddp = DDP(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
    perceptual = perceptual_model()
    pixels = open_pixels(config, 'train')
    rank, world = dist.get_rank(), dist.get_world_size()
    prepared = np.array(pixels[np.arange(rank * 2, (rank + 1) * 2)], copy=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    losses = []
    torch.manual_seed(config['seed']+dist.get_rank()*1009)
    for _ in range(20):
        # Fixed gate noise makes the four-photo overfit check reproducible.
        torch.manual_seed(9000+dist.get_rank())
        values, _, grad = distributed_update(ddp, optimizer, perceptual, prepared, config)
        losses.append(float(values[0]))
    if sum(losses[-5:]) >= sum(losses[:5]) or grad <= 0:
        raise AssertionError('Real-photo smoke did not demonstrate learning')
    ids = rank_ids(0, len(pixels), config['training']['effective_batch_size'], config['seed'], rank, world)
    prepared = np.array(pixels[ids], copy=True)
    values, usage, grad = distributed_update(ddp, optimizer, perceptual, prepared, config)
    delta = synchronized_parameters(model)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f'Missing or invalid end-to-end gradient: {name}')
    for scale in model.scales:
        if model.selectors[scale].weight.grad.abs().sum()<=0:raise AssertionError('No gradient to gene selector '+scale)
        if model.heads[scale].weight.grad.abs().sum() <= 0:
            raise AssertionError(f'No reconstruction gradient to {scale} encoder head')
        if model.sparse and model.dictionaries[scale].raw.requires_grad and model.dictionaries[scale].raw.grad.abs().sum() <= 0:
            raise AssertionError(f'No reconstruction gradient to {scale} dictionary')
    memories = [None] * world
    dist.all_gather_object(memories, torch.cuda.max_memory_allocated())
    macs = profile_macs(model, config['data']['resolution'])
    if primary():
        write_json(Path(config['root']) / f'reports/{variant}.distributed_smoke.json',
            {'status': 'passed', 'world_size': world, 'global_batch_size': len(ids) * world,
             'per_gpu_microbatch': config['training']['micro_batch_size'],
             'accumulation_steps': math.ceil(len(ids) / config['training']['micro_batch_size']),
             'small_real_photo_count': 4, 'small_updates': 20, 'losses': losses,
             'full_global_batch_loss': float(values[0]), 'gradient_norm': grad,
             'gradient_equivalence_max_abs_error': error, 'rank_parameter_max_difference': delta,
             'per_rank_peak_memory_bytes': memories, 'forward_conv_dictionary_macs_per_image': macs,
             'weights_discarded': True, 'model_spec': model.spec(), 'updated_utc': now()})
    dist.barrier()


def train(config, variant):
    root = Path(config['root'])
    output = root / 'runs' / variant
    output.mkdir(parents=True, exist_ok=True)
    settings = config['training']
    rank, world = dist.get_rank(), dist.get_world_size()
    pixels = open_pixels(config, 'train')
    validation = open_pixels(config, 'validation')
    model = create_model(config, variant)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    torch.manual_seed(config['seed']+rank*1009) # Independent gate noise per DDP rank; saved on resume.
    step = seen = 0
    elapsed, best = 0.0, float('inf')
    usage = torch.zeros(config['model']['dictionary_size'], device='cuda', dtype=torch.int64)
    resume = output / 'checkpoint_resume.pt'
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if not resume_compatible(saved['config'], config) or saved['variant'] != variant:
            raise ValueError('Resume model/data/config mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != 'step':
                    state[key] = value.cuda()
        step, seen = saved['step'], saved['images_seen']
        model.set_step(step)
        elapsed, best = saved['elapsed_seconds'], saved['best_validation_loss']
        usage.copy_(saved['gene_usage'].cuda())
    ddp = DDP(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
    perceptual = perceptual_model()
    if resume.exists():
        # Restore AFTER constructing auxiliary networks and DDP, so their
        # initialization cannot advance a checkpoint's random stream.
        rng = saved['rng_per_rank'][rank]
        torch.set_rng_state(rng['torch_rng'])
        torch.cuda.set_rng_state(rng['cuda_rng'])
        random.setstate(rng['python_rng'])
        np.random.set_state(rng['numpy_rng'])
    started = time.monotonic()
    max_steps = math.ceil(settings['max_training_images'] / settings['effective_batch_size'])
    if primary():
        smoke_report = json.loads((root / f'reports/{variant}.distributed_smoke.json').read_text())
        write_json(output / 'model_spec.json', {**model.spec(), 'world_size': world,
            'forward_conv_dictionary_macs_per_image': smoke_report['forward_conv_dictionary_macs_per_image'],
            'initialization': f'random seed {config["seed"]}; matched CNN trunk weights; no pretrained weights',
            'trained_from_previous_checkpoint': False, 'labels_used': False, 'updated_utc': now()})

    def checkpoint():
        rng = {'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
               'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()}
        states = [None] * world
        dist.all_gather_object(states, rng)
        if primary():
            atomic_checkpoint(resume, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'config': config, 'variant': variant, 'step': step, 'images_seen': seen,
                'best_validation_loss': best, 'elapsed_seconds': elapsed + time.monotonic() - started,
                'gene_usage': usage.cpu(), 'rng_per_rank': states})
        dist.barrier()

    def validate():
        nonlocal best
        result = evaluate(model, validation, config, perceptual)
        selected = result['selection_loss'] < best
        best = min(best, result['selection_loss'])
        result.update(step=step, images_seen=seen, variant=variant, split='validation', updated_utc=now())
        if primary():
            append_json(output / 'validation_history.jsonl', result)
            if selected:
                atomic_checkpoint(output / 'checkpoint_best.pt', {'model': model.state_dict(), 'config': config,
                    'variant': variant, 'step': step, 'images_seen': seen, 'validation': result})
            preview(model, validation, config, output, f'step_{step:06d}')
            plot_run(output)
        checkpoint()

    def progress(status='training', **extra):
        return {'status': status, 'variant': variant, 'step': step, 'max_updates': max_steps,
            'images_seen': seen, 'max_training_images': settings['max_training_images'],
            'dataset_passes': seen / len(pixels), 'train_images': len(pixels),
            'effective_batch_size': settings['effective_batch_size'],
            'micro_batch_size_per_gpu': settings['micro_batch_size'], 'gpus_per_model': world,
            'best_validation_loss': best if math.isfinite(best) else None,
            'elapsed_seconds': round(elapsed + time.monotonic() - started, 2),
            'gpu_peak_memory_bytes_rank0': torch.cuda.max_memory_allocated(),
            'cpu_fallback': False, 'updated_utc': now(), **extra}

    if primary():
        write_json(output / 'progress.json', progress('initial_validation' if step == 0 else 'training'))
    if step == 0 and settings['initial_validation']:
        validate()
    window = []
    while seen < settings['max_training_images']:
        model.set_step(step+1)
        if stop_requested(root):
            break
        batch = min(settings['effective_batch_size'], settings['max_training_images'] - seen)
        ids = rank_ids(seen, len(pixels), batch, config['seed'], rank, world)
        read_started = time.monotonic()
        prepared = np.array(pixels[ids], copy=True)
        read_time = time.monotonic() - read_started
        lr = learning_rate(seen, settings)
        for group in optimizer.param_groups:
            group['lr'] = lr
        values, batch_usage, grad = distributed_update(ddp, optimizer, perceptual, prepared, config)
        usage += batch_usage
        step += 1
        seen += batch
        window.append(values)
        if step % settings['log_every'] == 0 or step == 1:
            mean = np.mean(window, axis=0)
            window = []
            entry = progress(train_loss=float(mean[0]), train_l1=float(mean[1]), train_lpips=float(mean[2]),
                latent_rms=float(values[3]), train_coding_objective=float(mean[4]),
                train_pyramid_loss=float(mean[5]), train_rgb_loss=float(mean[6]),
                train_mean_active_genes=float(mean[7]),train_mean_encoded_bytes=float(mean[8]),
                train_expected_pre_cap_genes=float(mean[9]),train_expected_pre_cap_bytes=float(mean[10]),
                training_statistics_scope='mean across recent updates and their cycled caps',
                nominal_gene_cap=config['model']['budgets'][model.rate_index],
                training_scale_caps={s:model.quota(s) for s in model.scales},scale_gates=model.gates,
                gradient_norm=grad, learning_rate=lr,
                rank0_last_rgb_read_seconds=round(read_time, 5),
                genes_used_so_far=int((usage > 0).sum()) if model.sparse else None)
            if primary():
                append_json(output / 'history.jsonl', entry)
                write_json(output / 'progress.json', entry)
                print(json.dumps(entry), flush=True)
        if step % settings['evaluate_every'] == 0 or step == settings['first_evaluation'] or seen == settings['max_training_images']:
            if primary():
                write_json(output / 'progress.json', progress('validating'))
            validate()
            if primary():
                write_json(output / 'progress.json', progress())
        elif step % settings['checkpoint_every'] == 0:
            checkpoint()
    checkpoint()
    if seen < settings['max_training_images']:
        if primary():
            write_json(output / 'progress.json', progress('stopped'))
        return
    chosen = torch.load(output / 'checkpoint_best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(chosen['model'])
    model.set_step(chosen['step'])
    test = open_pixels(config, 'test')
    result = evaluate(model, test, config, perceptual)
    result.update(split='test', variant=variant, checkpoint_step=chosen['step'],
                  checkpoint_images_seen=chosen['images_seen'],
                  evaluation_policy='one final evaluation of the validation-selected checkpoint')
    if primary():
        write_json(output / 'test_metrics.json', result)
        preview(model, test, config, output, 'test_best')
        write_json(output / 'progress.json', progress('confirming_new_cities', test=result))
    dist.barrier()
    confirmation=open_pixels(config,'confirmation')
    confirmation_metrics=evaluate(model,confirmation,config,perceptual)
    if primary():
        write_json(output/'city_confirmation_metrics.json',confirmation_metrics)
        preview(model,confirmation,config,output,'city_confirmation')
        write_json(output/'progress.json',progress('complete',test=result,city_confirmation=confirmation_metrics))
    dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--mode', choices=['smoke', 'train'], required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    config['seed']=int(args.variant.rsplit('_s',1)[1])
    if not torch.cuda.is_available() or torch.cuda.device_count() != config['compute']['gpus_per_variant']:
        raise RuntimeError('Exactly two leased visible CUDA GPUs are required; no CPU fallback')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(4)
    dist.init_process_group('nccl', timeout=timedelta(minutes=15))
    if dist.get_world_size() != config['compute']['gpus_per_variant']:
        raise RuntimeError('Unexpected distributed world size')
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        (smoke if args.mode == 'smoke' else train)(config, args.variant)
        dist.destroy_process_group()
    except Exception as error:
        if primary():
            write_json(Path(config['root']) / f'reports/{args.variant}.{args.mode}_failure.json',
                       {'status': 'failed', 'error': str(error), 'updated_utc': now()})
        raise
