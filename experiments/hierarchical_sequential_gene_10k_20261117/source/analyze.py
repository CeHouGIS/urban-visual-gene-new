"""Post-training held-out diagnostics: prefixes, interventions and gene reuse."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.distributed as dist
from common import load_config, now, write_json
from data import image_tensor, open_pixels
from model import SequentialGene
from preview import preview, rgb
from train import amp, evaluate


@torch.inference_mode()
def interventions(model,pixels,config,out):
    # Known transformations, not presumed semantic ground truths.
    transforms={
        'exposure':lambda x:(x*.85),
        'white_balance':lambda x:x*torch.tensor([.85,.95,1.],device=x.device)[None,:,None,None],
        'horizontal_flip':lambda x:x.flip(-1),
        'mild_blur':lambda x:torch.nn.functional.avg_pool2d(torch.nn.functional.pad(x,(1,1,1,1),mode='reflect'),3,1),
    }
    rank,world=dist.get_rank(),dist.get_world_size()
    totals=torch.zeros(len(transforms),6,device='cuda',dtype=torch.float64)
    changes=torch.zeros(len(transforms),4,device='cuda',dtype=torch.float64)
    n=min(128,len(pixels))
    for start in range(rank*8,n,world*8):
        ids=np.arange(start,min(start+8,n))
        x=image_tensor(pixels,ids)
        with amp(config):
            original=model(x,last=3)
        p0=original['outputs'][-1]
        for t,(name,transform) in enumerate(transforms.items()):
            target=transform(x)
            with amp(config):
                other=model(target,last=3)
            totals[t,0]+=(p0-target).abs().mean((1,2,3)).sum()
            totals[t,1]+=(other['outputs'][-1]-target).abs().mean((1,2,3)).sum()
            for l in range(4):
                mixed=[v.clone() for v in original['indices']]
                mixed[l]=other['indices'][l]
                with amp(config):
                    decoded=model.decode_indices(mixed)[-1]
                totals[t,l+2]+=(decoded-target).abs().mean((1,2,3)).sum()
                changes[t,l]+=(mixed[l]!=original['indices'][l]).float().mean((1,2)).sum()
    dist.all_reduce(totals)
    dist.all_reduce(changes)
    result={name:dict(original_code_target_mae=float(totals[t,0]/n),
        transformed_code_target_mae=float(totals[t,1]/n),
        single_layer_swap_target_mae=(totals[t,2:]/n).cpu().tolist(),
        fraction_changed_indices_per_layer=(changes[t]/n).cpu().tolist()) for t,name in enumerate(transforms)}
    if rank==0:
        write_json(out/'change_response.json',dict(images=n,transforms=result,
            limitations=['Synthetic changes test their defined effects; not natural material or object ground truth.',
                'Single-layer substitutions interact with the conditional decoder; effects need not sum.',
                'Geometry comparisons use decoded outputs; raw index changes include spatial relocation.']))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        improvement=(totals[:,0:1]-totals[:,2:])/n
        values=improvement.cpu().numpy()
        v=max(float(np.abs(values).max()),1e-6)
        fig,ax=plt.subplots(figsize=(7,4))
        im=ax.imshow(values,cmap='RdBu',vmin=-v,vmax=v)
        ax.set_xticks(range(4),['L0','L1','L2','L3'])
        ax.set_yticks(range(len(transforms)),list(transforms))
        for t in range(len(transforms)):
            for l in range(4): ax.text(l,t,f'{values[t,l]:.4f}',ha='center',va='center',fontsize=9)
        ax.set_title('MAE improvement toward known transformation\nwhen replacing one layer (positive = improvement)')
        fig.colorbar(im,ax=ax)
        fig.tight_layout()
        fig.savefig(out/'change_response.png',dpi=160)
        plt.close(fig)


@torch.inference_mode()
def reuse_atlas(model,pixels,config,out):
    # Rank zero only; evaluation uses no model collectives or EMA writes.
    rows=[json.loads(v) for v in (Path(config['root'])/'data/images.test.jsonl').read_text().splitlines()]
    all_ids=[[] for _ in range(4)]
    for start in range(0,len(pixels),16):
        x=image_tensor(pixels,np.arange(start,min(start+16,len(pixels))))
        with amp(config): result=model(x,last=3)
        for l in range(4): all_ids[l].append(result['indices'][l].cpu().numpy())
    all_ids=[np.concatenate(v) for v in all_ids]
    np.savez_compressed(out/'test_gene_indices.npz',**{f'L{l}':v for l,v in enumerate(all_ids)})
    metadata=[]
    for l,codes in enumerate(all_ids):
        counts=np.bincount(codes.reshape(-1),minlength=config['model']['dictionary_sizes'][l])
        top=np.argsort(-counts)[:3]
        grid=config['model']['grids'][l]
        size=256//grid
        canvas=Image.new('RGB',(6*200,3*230),'white')
        draw=ImageDraw.Draw(canvas)
        for row,gene in enumerate(top):
            candidates=np.argwhere((codes==gene).any(-1))
            picked=[]
            seen_images=set()
            seen_cities=set()
            for pass_number in [0,1]:
                for image_id,cell_id in candidates:
                    city=(rows[image_id]['country'],rows[image_id]['city'])
                    if image_id in seen_images or (pass_number==0 and city in seen_cities): continue
                    picked.append((int(image_id),int(cell_id)))
                    seen_images.add(image_id); seen_cities.add(city)
                    if len(picked)==6: break
                if len(picked)==6: break
            for col,(image_id,cell_id) in enumerate(picked):
                gy,gx=divmod(cell_id,grid)
                crop=np.asarray(pixels[image_id])[gy*size:(gy+1)*size,gx*size:(gx+1)*size]
                pic=Image.fromarray(crop).resize((200,200),Image.Resampling.NEAREST)
                canvas.paste(pic,(col*200,row*230+30))
                draw.text((col*200+3,row*230+3),f'L{l} gene {gene}: {rows[image_id]["city"]}',fill='black')
                metadata.append(dict(level=l,gene=int(gene),image_index=image_id,cell_index=cell_id,
                    city=rows[image_id]['city'],country=rows[image_id]['country'],source_id=rows[image_id]['id']))
        canvas.save(out/f'reuse_examples_L{l}.png')
    write_json(out/'reuse_examples.json',dict(examples=metadata,
        interpretation='Actual source crops where the same gene was selected, not fixed RGB appearances of individual genes.'))


def run(config):
    root=Path(config['root']); out=root/'reports/final_diagnostics'
    out.mkdir(exist_ok=True)
    pixels=open_pixels(config,'test')
    validation=open_pixels(config,'validation')
    model=SequentialGene(config['model']).cuda()
    comparisons={}
    for label,path in [('stagewise','checkpoint_stagewise.pt'),('joint','checkpoint_last.pt')]:
        state=torch.load(root/'runs/main'/path,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);model.set_phase(4);model.eval()
        comparisons[label]=evaluate(model,validation,config,4)
    model.eval()
    if dist.get_rank()==0:
        write_json(out/'stagewise_vs_joint_validation.json',comparisons)
        x=image_tensor(pixels,np.arange(10))
        with amp(config): preview(model,x,out/'ten_test_progressive_reconstructions.png',3)
        reuse_atlas(model,pixels,config,out)
    dist.barrier()
    interventions(model,pixels,config,out)
    if dist.get_rank()==0:
        write_json(root/'reports/analysis_complete.json',dict(status='complete',updated_utc=now(),directory=str(out)))
    dist.barrier()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    torch.set_num_threads(4)
    try:run(load_config(args.config))
    finally:dist.destroy_process_group()
