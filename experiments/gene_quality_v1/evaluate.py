"""Evaluate a locked checkpoint using pure-noise image-conditional sampling.

Default split is validation. Test access requires an explicit locked-protocol flag.
The per-image score is averaged across seeds before the paired bootstrap.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,hashlib,random
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL
from skimage.metrics import structural_similarity
from train import Data,build,load_checkpoint,sample_one,ROOT,MODEL,write_json,stamp
from model import GeneConditioner

def evaluate(args):
    torch.set_num_threads(4);torch.manual_seed(17)
    device=torch.device(args.device);torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device.index)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=checkpoint['args'];resolution=config['resolution']
    if args.split!='validation' and not args.protocol_locked:raise RuntimeError('Test split requires --protocol-locked')
    data=Data(resolution);available=data.ids(args.split)
    expected=400
    assert len(available)==expected,f'Required complete {args.split} cache: {len(available)}/{expected}'
    # A balanced, fixed order shared by all model variants, independent of model outputs.
    by_city={}
    for i in available:by_city.setdefault(data.records[i]['city'],[]).append(i)
    selected=[v[j] for j in range(max(map(len,by_city.values()))) for v in by_city.values() if j<len(v)]
    if args.limit:selected=selected[:args.limit]
    output=ROOT/'evaluations'/args.run;output.mkdir(parents=True,exist_ok=True)
    protocol={'checkpoint':str(Path(args.checkpoint).resolve()),'checkpoint_step':checkpoint['step'],
        'split':args.split,'selected_indices':selected,'seeds':args.seeds,'sampling_steps':30,'cfg':5.,
        'resolution':resolution,'initial_state':'pure random noise','conditions':args.conditions,
        'test_access':args.split!='validation'}
    if (output/'protocol.json').exists():
        previous=json.loads((output/'protocol.json').read_text());previous.pop('created_at',None)
        assert previous==protocol,'Evaluation directory already belongs to a different protocol/checkpoint'
    else:write_json(output/'protocol.json',dict(protocol,created_at=stamp()))
    buildargs=SimpleNamespace(mode=config['mode'],resolution=resolution,gene='none',gradient_checkpointing=False,
        initial_gain=config.get('initial_gain',.01),locality_bias=config.get('locality_bias',0.))
    unet,adapters,conditioner,_,_=build(buildargs,device)
    load_checkpoint(args.checkpoint,adapters,conditioner);unet.eval();conditioner.eval()
    gene=None
    if checkpoint.get('gene') is not None:
        variant=config['gene'];gene=GeneConditioner(4 if variant=='learned_s4' else 8,variant!='random_s8',seed=17).to(device)
        gene.load_state_dict(checkpoint['gene']);gene.eval()
    vae=AutoencoderKL.from_pretrained(MODEL,subfolder='vae',variant='fp16',use_safetensors=True,
        local_files_only=True,torch_dtype=torch.float32).eval().to(device)
    import lpips
    metric=lpips.LPIPS(net='alex').eval().to('cpu')
    existing={}
    logfile=output/'scores.jsonl'
    if logfile.exists():
        for line in logfile.read_text().splitlines():
            if line.strip():
                x=json.loads(line);existing[(x['index'],x['seed'],x['condition'])]=x
    log=logfile.open('a',buffering=1)
    with torch.inference_mode():
        for row,index in enumerate(selected):
            source=data.records[index]
            # A deterministic permutation of the fixed evaluation set, no train input.
            alternate=selected[(row+max(1,len(selected)//2))%len(selected)]
            if alternate==index and 'shuffled' in args.conditions:raise ValueError('Need >=2 inputs for shuffled control')
            d,z=data.tensors([index,alternate],device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                code=conditioner(d,z)
                if gene is not None:code=gene(code)
            with Image.open(source['source_image']) as im:target=im.convert('RGB').resize((resolution,resolution),Image.Resampling.BILINEAR)
            target_array=np.array(target,dtype=np.float32);target_tensor=torch.from_numpy(target_array).permute(2,0,1)[None]/127.5-1
            for seed in args.seeds:
                for condition in args.conditions:
                    key=(index,seed,condition)
                    if key in existing:continue
                    tokens=code[0:1] if condition=='correct' else code[1:2] if condition=='shuffled' else torch.zeros_like(code[0:1])
                    latent=sample_one(unet,tokens,data,device,seed,30)
                    decoded=vae.decode(latent.to(device)/data.text['vae_scaling_factor']).sample[0]
                    a=((decoded.clamp(-1,1).float().permute(1,2,0).cpu().numpy()+1)*127.5).round().astype(np.uint8)
                    image=Image.fromarray(a)
                    if row<args.save_panel_inputs:image.save(output/f'{index}_{seed}_{condition}.png');target.save(output/f'{index}_target.png')
                    mse=float(np.mean((a.astype(np.float32)-target_array)**2))
                    x=torch.from_numpy(a.astype(np.float32)).permute(2,0,1)[None]/127.5-1
                    score={'index':index,'city':source['city'],'seed':seed,'condition':condition,
                        'lpips_alex':float(metric(x,target_tensor).item()),
                        'psnr':float(10*np.log10(255**2/max(mse,1e-12))),
                        'ssim':float(structural_similarity(a,target_array,data_range=255,channel_axis=2))}
                    existing[key]=score;log.write(json.dumps(score)+'\n')
            progress={'completed_inputs':row+1,'total_inputs':len(selected),'updated_at':stamp()}
            write_json(output/'status.json',dict(progress,state='evaluating'));print(json.dumps(progress),flush=True)
    log.close();summary={'checkpoint_step':checkpoint['step'],'split':args.split,'inputs':len(selected),'seeds':args.seeds,'condition_means':{}}
    per_condition={}
    for condition in args.conditions:
        matrix={metric_name:np.array([np.mean([existing[(i,s,condition)][metric_name] for s in args.seeds]) for i in selected]) for metric_name in ['lpips_alex','psnr','ssim']}
        per_condition[condition]=matrix
        summary['condition_means'][condition]={m:float(a.mean()) for m,a in matrix.items()}
    if 'correct' in per_condition:
        rng=np.random.default_rng(17);summary['paired_bootstrap']={}
        for condition in ['zero','shuffled']:
            if condition not in per_condition:continue
            difference=per_condition['correct']['lpips_alex']-per_condition[condition]['lpips_alex']
            estimates=np.array([rng.choice(difference,len(difference),replace=True).mean() for _ in range(5000)])
            summary['paired_bootstrap'][f'correct_minus_{condition}_lpips']={'mean':float(difference.mean()),
                '95pct_percentile_ci':np.quantile(estimates,[.025,.975]).tolist(),'unit':'input image after averaging noise seeds'}
    summary['human_ratings']='not collected';summary['note']='Numbers alone do not establish naturalness or localized editability.'
    write_json(output/'summary.json',summary);write_json(output/'status.json',{'state':'complete','updated_at':stamp()});print(json.dumps(summary),flush=True)
    if args.split=='validation' and 'correct' in summary['condition_means']:
        location=Path(args.checkpoint).resolve().parent/'best_validation.json'
        candidate={'checkpoint':str(Path(args.checkpoint).resolve()),'lpips':summary['condition_means']['correct']['lpips_alex'],
            'indices':selected,'seeds':args.seeds,'evaluation':str((output/'summary.json').resolve())}
        if location.exists():
            previous=json.loads(location.read_text())
            if previous['indices']!=selected or previous['seeds']!=args.seeds:raise RuntimeError('Do not compare best checkpoints across different validation protocols')
            if candidate['lpips']>=previous['lpips']:return
        write_json(location,candidate)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--run',required=True)
    p.add_argument('--split',choices=['validation','in_domain_test','cross_city_test'],default='validation')
    p.add_argument('--protocol-locked',action='store_true');p.add_argument('--limit',type=int,default=64)
    p.add_argument('--seeds',nargs='+',type=int,default=[101,211]);p.add_argument('--save-panel-inputs',type=int,default=4)
    p.add_argument('--conditions',nargs='+',choices=['correct','zero','shuffled'],default=['correct','zero','shuffled'])
    p.add_argument('--device',default='cuda:3');p.add_argument('--memory-fraction',type=float,default=.42)
    evaluate(p.parse_args())
