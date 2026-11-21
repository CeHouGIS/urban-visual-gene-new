"""Train one new shared RGB probe; upstream checkpoints are strictly read-only."""
import gc
import json
import math
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from torch.nn import functional as F

from common import config,now,write_json,fingerprint,phase_indices,save_checkpoint,mirror_file
from downstream_model import RGBDecoder,load_codec,source_features
from features import FeatureStore
from model import tensor_hash
from utils import GridSampler,image_tensor

STOP=False


def stop(*_):
    global STOP
    STOP=True


def amp():return torch.autocast('cuda',dtype=torch.bfloat16)


def ft(store,ids):return torch.from_numpy(store[np.asarray(ids)]).cuda()


@torch.no_grad()
def validation_input(c,f,source,codecs,ids):
    if source=='dino_reference':return f
    # Decode the SAME integer codes used for retrieval and arithmetic files.
    q=np.load(Path(c['local'])/'exports'/source/'validation.codes.npy',mmap_mode='r')
    values=torch.from_numpy(np.array(q[ids],copy=True)).cuda().float()
    return codecs[source].dictionary(values*codecs[source].step)


def make_model(c):
    torch.manual_seed(c['seed']+30000)
    model=RGBDecoder().cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=c['decoder']['learning_rate'],weight_decay=1e-4)
    return model,opt


def update(model,opt,f,target,source,codecs,micro):
    model.train();opt.zero_grad(set_to_none=True);mae=0.
    for i in range(0,len(f),micro):
        part=f[i:i+micro]
        with amp():
            restored=source_features(part,source,codecs)
            rgb=model(restored)
            loss=F.l1_loss(rgb.float(),target[i:i+len(part)])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite RGB decoder loss')
        (loss*len(part)/len(f)).backward()
        mae+=float(loss.detach())*len(part)/len(f)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
    opt.step()
    return dict(mae=mae,gradient_norm=float(norm))


def benchmark(c,store,pixels,codecs,directory):
    ids=GridSampler(c['base']['manifests']['train'],c['seed']).indices(0,c['decoder']['batch_size'])
    f,target=ft(store,ids),image_tensor(pixels,ids)
    results=[]
    for micro in c['decoder']['benchmark_micro_batches']:
        model=opt=None
        gc.collect();torch.cuda.empty_cache()
        try:
            model,opt=make_model(c)
            before=tensor_hash(model)
            torch.cuda.reset_peak_memory_stats()
            timings=[]
            for source in c['sources']:
                # Warm kernels first, then measure a real forward/backward update.
                update(model,opt,f,target,source,codecs,micro)
                torch.cuda.synchronize();start=time.monotonic()
                value=update(model,opt,f,target,source,codecs,micro)
                torch.cuda.synchronize()
                timings.append(time.monotonic()-start)
            assert before!=tensor_hash(model)
            peak=torch.cuda.max_memory_reserved()
            results.append(dict(micro_batch_size=micro,mean_step_seconds=float(np.mean(timings)),
                peak_reserved_gib=peak/2**30,passed=True,last_report=value,
                safe=peak<c['decoder']['memory_fraction']*torch.cuda.get_device_properties(0).total_memory))
        except torch.OutOfMemoryError:
            results.append(dict(micro_batch_size=micro,safe=False,passed=False,status='out_of_memory'))
        finally:
            del model,opt;gc.collect();torch.cuda.empty_cache()
    safe=[r for r in results if r.get('safe')]
    if not safe:raise RuntimeError('No decoder microbatch fits memory budget')
    chosen=min(safe,key=lambda r:r['mean_step_seconds'])['micro_batch_size']
    write_json(directory/'benchmark.json',dict(candidates=results,chosen_micro_batch_size=chosen,
        real_feature_smoke_passed=True,all_three_sources_tested=True,main_decoder_restarts_from_random=True))
    return chosen


@torch.no_grad()
def evaluate(c,decoder,codecs,store,pixels,micro):
    decoder.eval();ids=np.linspace(0,len(store)-1,c['validation_images'],dtype=int)
    result={}
    for source in c['sources']:
        maes,psnrs=[],[]
        for first in range(0,len(ids),micro):
            chosen=ids[first:first+micro];f=ft(store,chosen);target=image_tensor(pixels,chosen)
            with amp():rgb=decoder(validation_input(c,f,source,codecs,chosen)).float()
            e=rgb-target
            maes.extend(e.abs().mean((1,2,3)).cpu().tolist())
            psnrs.extend((-10*e.square().mean((1,2,3)).clamp_min(1e-12).log10()).cpu().tolist())
        result[source]=dict(mae=float(np.mean(maes)),mean_image_psnr_db=float(np.mean(psnrs)),images=len(ids))
    decoder.train()
    return result


@torch.no_grad()
def export_queries(c,decoder,codecs,store,pixels,step,directory):
    decoder.eval();ids=c['queries'];predictions={}
    f=ft(store,ids)
    for source in c['sources']:
        with amp():rgb=decoder(validation_input(c,f,source,codecs,ids)).float()
        predictions[source]=rgb.cpu().numpy().transpose(0,2,3,1)
    partial=directory/'query_rgb.npz.partial'
    with partial.open('wb') as out:
        np.savez(out,original=np.array(pixels[ids],copy=True),indices=np.array(ids),step=step,**predictions)
    partial.replace(directory/'query_rgb.npz')
    decoder.train()


def main(c):
    if not torch.cuda.is_available():raise RuntimeError('GPU required; queue instead of CPU neural training')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    root=Path(c['root']);directory=root/'runs/decoder';directory.mkdir(exist_ok=True)
    local=Path(c['local'])/'runs/decoder';local.mkdir(parents=True,exist_ok=True)
    codecs,provenance={},{}
    for m in c['dictionary_sizes']:
        name=f'atoms_{m}';codecs[name],provenance[name]=load_codec(c,m)
    original_hash={k:v['model_hash'] for k,v in provenance.items()}
    train,val=FeatureStore(c['base'],'train'),FeatureStore(c['base'],'validation')
    pixels=np.load(Path(c['rgb_cache'])/'train.rgb.npy',mmap_mode='r')
    val_pixels=np.load(Path(c['rgb_cache'])/'validation.rgb.npy',mmap_mode='r')
    if pixels.shape!=(10000,256,256,3) or val_pixels.shape!=(1000,256,256,3):
        raise ValueError('Unexpected RGB target cache')
    bench=directory/'benchmark.json'
    micro=json.loads(bench.read_text())['chosen_micro_batch_size'] if bench.exists() else benchmark(c,train,pixels,codecs,directory)
    if any(tensor_hash(v)!=original_hash[k] for k,v in codecs.items()):raise AssertionError('Smoke changed frozen SAE')
    model,opt=make_model(c);initial_hash=tensor_hash(model);step=0
    latest=local/'checkpoint_latest.pt'
    if not latest.exists() and (directory/'checkpoint_latest.pt').exists():
        mirror_file(directory/'checkpoint_latest.pt',latest)
    if latest.exists():
        saved=torch.load(latest,map_location='cuda',weights_only=False)
        if saved['fingerprint']!=fingerprint(c) or saved['codec_hashes']!=original_hash:
            raise ValueError('Incompatible downstream decoder resume')
        model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu());step=saved['step']
    else:
        write_json(directory/'initialization.json',dict(initialization='random',initial_hash=initial_hash,
            input_channels=1024,output_resolution=256,codecs=provenance,seed=c['seed']+30000,
            no_rgb_skip_connection=True,loss='RGB MAE only',sources=c['sources'],updated_utc=now()))
    def save(name):
        path=local/name
        save_checkpoint(path,dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,config=c,
            fingerprint=fingerprint(c),codec_hashes=original_hash,micro_batch_size=micro,
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),updated_utc=now()))
        mirror_file(path,directory/name)
    sampler=GridSampler(c['base']['manifests']['train'],c['seed'])
    start=time.monotonic();initial_step=step;settings=c['decoder']
    if step>settings['steps']:raise ValueError('Resume exceeds configured decoder budget')
    result=None
    while step<settings['steps'] and not STOP and not (root/'STOP_REQUESTED').exists():
        begin=time.monotonic()
        source,ids=phase_indices(sampler,step,settings['batch_size'],c['sources'])
        f,target=ft(train,ids),image_tensor(pixels,ids)
        factor=min(1.,(step+1)/settings['warmup_steps'])
        if step>=settings['warmup_steps']:
            phase=(step-settings['warmup_steps'])/(settings['steps']-settings['warmup_steps'])
            factor=.2+.8*.5*(1+math.cos(math.pi*phase))
        opt.param_groups[0]['lr']=settings['learning_rate']*factor
        report=update(model,opt,f,target,source,codecs,micro)
        step+=1;report.update(step=step,source=source,seconds=time.monotonic()-begin,lr=opt.param_groups[0]['lr'])
        with (local/'train.jsonl').open('a') as stream:stream.write(json.dumps(report)+'\n')
        if step==1 or step%25==0:
            elapsed=time.monotonic()-start
            write_json(directory/'progress.json',dict(status='training',step=step,total_steps=settings['steps'],
                effective_batch_size=settings['batch_size'],micro_batch_size=micro,sources=c['sources'],
                elapsed_seconds=elapsed,remaining_seconds=elapsed/max(1,step-initial_step)*(settings['steps']-step),
                latest=report,updated_utc=now()))
            mirror_file(local/'train.jsonl',directory/'train.jsonl');print(json.dumps(report),flush=True)
        if step==99 or step%settings['validate_every']==0 or step==settings['steps']:
            result=evaluate(c,model,codecs,val,val_pixels,min(micro,8))
            write_json(directory/'validation_latest.json',dict(step=step,results=result,updated_utc=now()))
            export_queries(c,model,codecs,val,val_pixels,step,directory)
            with (local/'validation.jsonl').open('a') as stream:stream.write(json.dumps(dict(step=step,results=result))+'\n')
            mirror_file(local/'validation.jsonl',directory/'validation.jsonl')
        if step%settings['checkpoint_every']==0:save('checkpoint_latest.pt')
    save('checkpoint_latest.pt')
    if (local/'train.jsonl').exists():mirror_file(local/'train.jsonl',directory/'train.jsonl')
    if step<settings['steps']:
        write_json(directory/'progress.json',dict(status='stopped',step=step,updated_utc=now()));return
    if tensor_hash(model)==initial_hash:raise AssertionError('Decoder did not learn')
    if any(tensor_hash(v)!=original_hash[k] for k,v in codecs.items()):raise AssertionError('Decoder changed frozen SAE')
    if result is None:
        result=evaluate(c,model,codecs,val,val_pixels,min(micro,8))
        write_json(directory/'validation_latest.json',dict(step=step,results=result,updated_utc=now()))
        export_queries(c,model,codecs,val,val_pixels,step,directory)
    save('checkpoint_final.pt')
    write_json(directory/'result.json',dict(step=step,results=result,codecs=provenance,decoder_hash=tensor_hash(model),
        upstream_unchanged=True,input_resolution=512,rgb_target_resolution=256,updated_utc=now()))
    write_json(directory/'progress.json',dict(status='complete',step=step,updated_utc=now()))


if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    main(config())
