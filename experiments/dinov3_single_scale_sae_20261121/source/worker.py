"""GPU-only capacity experiment: tests, real-feature memory probe, training, export."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import torch

from model import Codec, tensor_hash
from features import FeatureStore, signature
from utils import GridSampler, append_json, now, write_json

STOP = False


def stop(*_):
    global STOP
    STOP = True


def fingerprint(config, atoms):
    return hashlib.sha256(json.dumps(dict(config=config, atoms=atoms), sort_keys=True).encode()).hexdigest()


def amp():
    return torch.autocast('cuda', dtype=torch.bfloat16)


def feature_tensor(features, ids):
    return torch.from_numpy(np.array(features[ids], copy=True)).cuda().float()


def update(model, optimizer, x, micro, coefficient):
    optimizer.zero_grad(set_to_none=True)
    reports = []
    for begin in range(0, len(x), micro):
        part = x[begin:begin+micro]
        with amp():
            loss, fit, report = model.objective(part, model(part), coefficient)
        if not torch.isfinite(loss+fit):
            raise FloatingPointError('Nonfinite training objective')
        ((loss+fit)*len(part)/len(x)).backward()
        reports.append(report)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
    optimizer.step()
    model.dictionary.normalize_()
    # All configured microbatches divide the effective batch exactly.
    return dict(**{k:float(np.mean([r[k] for r in reports])) for k in reports[0]},
                gradient_norm=float(norm))


def make_model(config, atoms):
    torch.manual_seed(config['init_seed'])
    model = Codec(config['compression']['quantization_step'], atoms=atoms, channels=config['channels']).cuda()
    optimizer = torch.optim.Adam([
        dict(params=list(model.analysis.parameters())+list(model.dictionary.parameters()),
             lr=config['compression']['learning_rate']),
        dict(params=model.entropy.parameters(), lr=config['compression']['entropy_lr'])])
    return model, optimizer


def benchmark(config, atoms, features, directory):
    settings = config['compression']
    ids = GridSampler(config['manifests']['train'],config['seed']).indices(0,settings['batch_size'])
    x = feature_tensor(features, ids)
    results = []
    total_memory = torch.cuda.get_device_properties(0).total_memory
    for micro in config['benchmark_micro_batches']:
        model = optimizer = None
        gc.collect()
        torch.cuda.empty_cache()
        try:
            model, optimizer = make_model(config, atoms)
            before = tensor_hash(model)
            torch.cuda.reset_peak_memory_stats()
            durations = []
            for i in range(3):
                torch.cuda.synchronize()
                begin = time.monotonic()
                report = update(model,optimizer,x,micro,config['lambda_bpp'])
                torch.cuda.synchronize()
                durations.append(time.monotonic()-begin)
            if tensor_hash(model) == before:
                raise AssertionError('No parameter update in real-feature smoke')
            peak = torch.cuda.max_memory_reserved()
            results.append(dict(micro_batch_size=micro, effective_batch_size=len(x),
                warm_step_seconds=durations[0], mean_step_seconds=float(np.mean(durations[1:])),
                peak_reserved_gib=peak/2**30, total_gpu_gib=total_memory/2**30,
                fits_memory_budget=peak < config['benchmark_memory_fraction']*total_memory,
                status='passed', smoke_report=report))
        except torch.OutOfMemoryError:
            results.append(dict(micro_batch_size=micro,status='out_of_memory',fits_memory_budget=False))
        finally:
            del model, optimizer
            gc.collect()
            torch.cuda.empty_cache()
        write_json(directory/'benchmark.json',dict(atoms=atoms,candidates=results,updated_utc=now()))
    safe = [r for r in results if r.get('fits_memory_budget')]
    if not safe:
        raise RuntimeError('No microbatch fits the real-feature GPU memory budget')
    chosen = min(safe,key=lambda r:r['mean_step_seconds'])
    result = dict(atoms=atoms,candidates=results,chosen_micro_batch_size=chosen['micro_batch_size'],
        effective_batch_size=settings['batch_size'], real_feature_smoke_passed=True,
        main_training_restarts_from_random=True,updated_utc=now())
    write_json(directory/'benchmark.json',result)
    del x
    return chosen['micro_batch_size']


@torch.no_grad()
def evaluate(config, model, features, micro):
    model.eval()
    ids = np.linspace(0,len(features)-1,config['validation_images'],dtype=int)
    reports = []
    for begin in range(0,len(ids),micro):
        x = feature_tensor(features,ids[begin:begin+micro])
        with amp():
            _,_,report = model.objective(x,model(x),config['lambda_bpp'])
        reports.append((len(x),report))
    result = {k:float(sum(n*r[k] for n,r in reports)/len(ids)) for k in reports[0][1]}
    model.train()
    return dict(**result,images=len(ids),updated_utc=now())


def save(path, model, optimizer, config, atoms, step, encoder_hash, micro):
    part = path.with_suffix('.partial')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=config,
        atoms=atoms,step=step,stage='dinov3_sae',fingerprint=fingerprint(config,atoms),
        encoder_hash=encoder_hash,quantization_step=model.step,micro_batch_size=micro,
        rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),updated_utc=now()),part)
    part.replace(path)


@torch.no_grad()
def export(config, model, features, atoms, directory, encoder_hash, step):
    model.eval()
    before = tensor_hash(model)
    ids = np.linspace(0,len(features)-1,config['validation_images'],dtype=int)
    sums = np.zeros(atoms,dtype=np.float64)
    nonzero = np.zeros(atoms,dtype=np.int64)
    presence = np.zeros(atoms,dtype=np.int64)
    cosine_sum = 0.
    for begin in range(0,len(ids),4):
        f = feature_tensor(features,ids[begin:begin+4])
        with amp():
            restored,q,_ = model(f)
        cosine_sum += float(torch.nn.functional.cosine_similarity(restored.float(),f.float(),dim=1).sum())
        values = q.cpu().numpy().astype(np.uint8)
        sums += values.sum((0,2,3),dtype=np.float64)*model.step
        active = values > 0
        nonzero += active.sum((0,2,3))
        presence += active.reshape(len(values),atoms,-1).any(2).sum(0)
    np.savez(directory/'atom_statistics.npz',mean_weight=sums/(len(ids)*32*32),
             coverage=nonzero/(len(ids)*32*32),image_presence=presence/len(ids))
    queries = config['queries']
    codes,restorations = [],[]
    for i in range(0,len(queries),4):
        f = feature_tensor(features,queries[i:i+4])
        with amp():
            restored,q,_ = model(f)
        codes.append(q.cpu().numpy().astype(np.uint8))
        restorations.append(restored.float().cpu().numpy().astype(np.float16))
    np.savez(directory/'query_outputs.npz',codes=np.concatenate(codes),restored=np.concatenate(restorations),
             indices=np.array(queries),step=step,quantization_step=model.step)
    result=evaluate(config,model,features,4)
    result.update(atoms=atoms,feature_channels=1024,feature_grid=[32,32],step=step,
        validation_centered_feature_cosine=cosine_sum/(len(ids)*32*32),
        atoms_unseen_on_128=int((presence==0).sum()),encoder_signature=encoder_hash,
        note='Feature-space reconstruction only. No RGB decoder in this experiment.')
    write_json(directory/'evaluation.json',result)
    if tensor_hash(model) != before:
        raise AssertionError('Export changed SAE weights')
    model.train()
    # Render the saved outputs immediately, including the step-50 preview.
    with (Path(config['root'])/'logs'/f'report_{atoms}_{step}.log').open('a') as log:
        subprocess.Popen([config['python'],str(Path(config['source'])/'report.py')],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),cwd=config['temporary_directory'],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)


def main(config, atoms):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; no CPU neural inference/training')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    root = Path(config['root'])
    directory = root/'runs'/f'atoms_{atoms:04d}'
    directory.mkdir(exist_ok=True)
    import fcntl
    lock = (directory/'worker.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update(SAE_TEST_ATOMS=str(atoms),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    subprocess.run([config['python'],'-m','pytest','-q','-o','addopts=','-p','no:cacheprovider',
        str(Path(config['source']).parent/'tests/test_dino_sae.py'),
        '--junitxml='+str(directory/'gpu_tests.xml'),
        '--basetemp='+config['temporary_directory']+f'/pytest_{atoms}'],env=env,check=True)
    meta = json.loads((Path(config['root'])/'features/READY.json').read_text())
    if meta['signature'] != signature(config):
        raise ValueError('Incompatible DINO feature cache')
    features = FeatureStore(config,'train')
    validation = FeatureStore(config,'validation')
    assert len(features)==10000 and len(validation)==1000
    encoder_hash = meta['signature']
    benchmark_file = directory/'benchmark.json'
    if benchmark_file.exists() and json.loads(benchmark_file.read_text()).get('chosen_micro_batch_size'):
        micro = json.loads(benchmark_file.read_text())['chosen_micro_batch_size']
    else:
        write_json(directory/'progress.json',dict(status='benchmarking_real_features',atoms=atoms,updated_utc=now()))
        micro = benchmark(config,atoms,features,directory)
    model,optimizer = make_model(config,atoms)
    settings = config['compression']
    step = 0
    latest = directory/'checkpoint_latest.pt'
    initial_hash = tensor_hash(model)
    if latest.exists():
        saved = torch.load(latest,map_location='cuda',weights_only=False)
        if saved['fingerprint'] != fingerprint(config,atoms) or saved['encoder_hash'] != encoder_hash:
            raise ValueError('Refusing incompatible checkpoint resume')
        model.load_state_dict(saved['model'],strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'].cpu())
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        step = saved['step']
    else:
        write_json(directory/'initialization.json',dict(initialization='random',atoms=atoms,
            initial_hash=initial_hash,seed=config['seed']+20000,encoder_hash=encoder_hash,
            reused_encoder=True,backbone="DINOv3 ViT-L/16",reused_sae_weights=False,updated_utc=now()))
        validation_report = evaluate(config,model,validation,min(micro,8))
        append_json(directory/'validation.jsonl',dict(step=0,**validation_report))
    sampler = GridSampler(config['manifests']['train'],config['seed'])
    start,initial = time.monotonic(),step
    while step < settings['steps'] and not STOP and not (root/'STOP_REQUESTED').exists():
        begin = time.monotonic()
        ids = sampler.indices(step*settings['batch_size'],settings['batch_size'])
        x = feature_tensor(features,ids)
        factor = min(1.,(step+1)/settings['warmup_steps'])
        if step >= settings['warmup_steps']:
            phase = (step-settings['warmup_steps'])/max(1,settings['steps']-settings['warmup_steps'])
            factor = .2+.8*.5*(1+math.cos(math.pi*phase))
        for i,group in enumerate(optimizer.param_groups):
            group['lr'] = settings['entropy_lr' if i else 'learning_rate']*factor
        model.train()
        report = update(model,optimizer,x,micro,config['lambda_bpp'])
        step += 1
        report.update(step=step,seconds=time.monotonic()-begin,lr=optimizer.param_groups[0]['lr'])
        append_json(directory/'train.jsonl',report)
        if step == 1 or step % 25 == 0:
            elapsed = time.monotonic()-start
            progress = dict(status='training',atoms=atoms,step=step,total_steps=settings['steps'],
                effective_batch_size=settings['batch_size'],micro_batch_size=micro,
                elapsed_seconds=elapsed,remaining_seconds=elapsed/max(1,step-initial)*(settings['steps']-step),
                latest=report,updated_utc=now())
            write_json(directory/'progress.json',progress)
            print(json.dumps(progress),flush=True)
        if step == 50:
            export(config,model,validation,atoms,directory,encoder_hash,step)
        if step % settings['checkpoint_every'] == 0:
            save(latest,model,optimizer,config,atoms,step,encoder_hash,micro)
        if step % settings['validate_every'] == 0 or step == settings['steps']:
            validation_report = evaluate(config,model,validation,min(micro,8))
            append_json(directory/'validation.jsonl',dict(step=step,**validation_report))
            write_json(directory/'validation_latest.json',dict(step=step,**validation_report))
    save(latest,model,optimizer,config,atoms,step,encoder_hash,micro)
    if step < settings['steps']:
        write_json(directory/'progress.json',dict(status='stopped',step=step,atoms=atoms,updated_utc=now()))
        return
    if tensor_hash(model) == initial_hash:
        raise AssertionError('Final SAE equals random initialization')
    save(directory/'checkpoint_final.pt',model,optimizer,config,atoms,step,encoder_hash,micro)
    write_json(directory/'progress.json',dict(status='exporting_evaluation',atoms=atoms,step=step,updated_utc=now()))
    export(config,model,validation,atoms,directory,encoder_hash,step)
    write_json(directory/'progress.json',dict(status='gpu_complete',atoms=atoms,step=step,
        effective_batch_size=settings['batch_size'],micro_batch_size=micro,updated_utc=now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--atoms',type=int,required=True)
    args = parser.parse_args()
    cfg = json.loads((Path(__file__).parents[1]/'configs/experiment.json').read_text())
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        main(cfg,args.atoms)
    except Exception as error:
        write_json(Path(cfg['root'])/'runs'/f'atoms_{args.atoms:04d}'/'progress.json',
                   dict(status='failed',error=repr(error),updated_utc=now()))
        raise
