"""Encode targets with the same SDXL VAE; cache only train/validation first."""
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

ROOT=Path(__file__).resolve().parent

@torch.inference_mode()
def main(args):
    torch.set_num_threads(4)
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.35,device.index)
    model=ROOT/'models/sdxl-base-1.0';data=ROOT/'data'
    manifest=json.loads((data/'manifest.json').read_text())
    # Different length catches stale caches. Test targets remain untouched.
    records=manifest['records'];side=args.resolution//8;shape=(len(records),4,side,side)
    suffix='' if args.resolution==512 else f'_{args.resolution}'
    latent_path=data/f'latents{suffix}.npy';ready_path=data/f'latents_ready{suffix}.npy'
    if latent_path.exists():
        latents=np.load(latent_path,mmap_mode='r+');ready=np.load(ready_path)
        assert latents.shape==shape
    else:
        latents=np.lib.format.open_memmap(latent_path,mode='w+',dtype=np.float16,shape=shape)
        ready=np.zeros(len(records),dtype=bool);np.save(ready_path,ready)
    vae=AutoencoderKL.from_pretrained(model,subfolder='vae',variant='fp16',use_safetensors=True,local_files_only=True,torch_dtype=torch.float32).eval().to(device)
    scaling_factor=float(vae.config.scaling_factor)
    encoded=0
    by_city={}
    for i,r in enumerate(records):
        if r['split']=='train':by_city.setdefault(r['city'],[]).append(i)
    priority=[ids[j] for j in range(8) for ids in by_city.values()]
    for city in by_city:
        priority.append(next(i for i,r in enumerate(records) if r['city']==city and r['split']=='validation'))
    order=priority+[i for i in range(len(records)) if i not in set(priority)]
    for i in order:
        r=records[i]
        if ready[i] or (r['split'] not in ('train','validation') and not args.include_tests):continue
        with Image.open(r['source_image']) as im:
            a=np.asarray(im.convert('RGB').resize((args.resolution,args.resolution),Image.Resampling.BILINEAR),dtype=np.float32).copy()/127.5-1
        image=torch.from_numpy(a).permute(2,0,1)[None].to(device)
        z=vae.encode(image).latent_dist.mode()*vae.config.scaling_factor
        assert torch.isfinite(z).all()
        latents[i]=z[0].cpu().numpy().astype(np.float16);ready[i]=True
        encoded+=1
        if ready.sum()%50==0:
            latents.flush();np.save(ready_path,ready);print('VAE cache',int(ready.sum()),flush=True)
        if args.limit and encoded>=args.limit:break
    latents.flush();np.save(ready_path,ready)
    del vae;torch.cuda.empty_cache()
    prompts=['a street-level photograph','']
    all_embeddings=[];pooled=None
    for index,cls in enumerate([CLIPTextModel,CLIPTextModelWithProjection]):
        folder='text_encoder' if index==0 else 'text_encoder_2'
        tokfolder='tokenizer' if index==0 else 'tokenizer_2'
        tokenizer=CLIPTokenizer.from_pretrained(model/tokfolder,local_files_only=True)
        encoder=cls.from_pretrained(model/folder,variant='fp16',local_files_only=True,use_safetensors=True,torch_dtype=torch.float32).eval().to(device)
        ids=tokenizer(prompts,padding='max_length',max_length=tokenizer.model_max_length,truncation=True,return_tensors='pt').input_ids.to(device)
        out=encoder(ids,output_hidden_states=True)
        all_embeddings.append(out.hidden_states[-2].cpu())
        if index==1:pooled=out.text_embeds.cpu()
        del encoder;torch.cuda.empty_cache()
    embed=torch.cat(all_embeddings,dim=-1)
    # SDXL empty negative is zeros by model config, not an encoded empty string.
    embed[1].zero_();pooled[1].zero_()
    torch.save({'prompt':prompts[0],'embeddings':embed,'pooled':pooled,'vae_scaling_factor':scaling_factor,
                'latent_mode':'posterior_mode; float32 VAE','geometry':f'whole-image bilinear square resize{args.resolution}',
                'resolution':args.resolution},data/f'conditioning{suffix}.pt')
    print(json.dumps({'ready':int(ready.sum()),'text_shape':list(embed.shape),'pooled_shape':list(pooled.shape)}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:2');parser.add_argument('--include-tests',action='store_true')
    parser.add_argument('--limit',type=int,default=0,help='Encode this many new targets; resume safely later')
    parser.add_argument('--resolution',type=int,default=1024,choices=[512,1024])
    main(parser.parse_args())
