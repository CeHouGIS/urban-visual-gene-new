"""Visualize actual dictionary atoms, expression maps, and fixed-code deletions."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from common import load_config, now, write_json
from reference.supervisor import acquire


def run(config, destination):
    destination.mkdir(parents=True, exist_ok=False)
    leases=acquire(config,1)
    if not leases:
        write_json(destination/'status.json',{'status':'waiting_for_gpu','updated_utc':now()})
        return False
    gpu,lease=leases[0]
    os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid']
    try:
        import torch
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from model import create_model
        from data import image_tensor

        torch.set_num_threads(4)
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        root=Path(config['root']);variant='learned_s42'
        saved=torch.load(root/f'runs/{variant}/checkpoint_best.pt',map_location='cpu',weights_only=False)
        model=create_model(config,variant).eval().requires_grad_(False)
        model.load_state_dict(saved['model'],strict=True);model.set_step(saved['step'])
        model.rate_index=config['protocol']['primary_rate_index']
        for name,value in model.state_dict().items():
            if not torch.equal(value.cpu(),saved['model'][name]):raise AssertionError('Checkpoint mismatch '+name)
        audit=json.loads((root/f'reports/gene_audit/{variant}/functional_reuse_summary.json').read_text())
        selected={s:next(g for g in audit['per_gene'] if g['gene_key'].startswith(s+':')) for s in model.scales}
        # Fixed first three cities; do not choose images based on visual quality.
        indices=np.arange(3)
        pixels=np.load(json.loads((root/'data/READY.json').read_text())['confirmation_cache'],mmap_mode='r')
        samples=json.loads((root/'data/confirmation_selection.json').read_text())['samples']
        def rgb(t):return t.detach().float().cpu().permute(1,2,0).numpy().clip(0,1)
        def plain(ax):ax.set_xticks([]);ax.set_yticks([])
        rows={};metadata={'variant':variant,'checkpoint_step':saved['step'],'primary_gene_cap':64,
            'image_indices':indices.tolist(),'selection':'first gene in each scale ranked by helpful-city count then mean deletion MAE increase; fixed first three city images',
            'difference_definition':'mean absolute RGB difference per pixel between full and gene-deleted reconstruction; fixed [0,0.25] color scale',
            'expression_definition':'quantized expression amplitude on [0,4]; fixed color scale across all scales/images; not an object mask',
            'atom_definition':'normalized dictionary weights in internal feature coordinates, not RGB colors',
            'selected_genes':selected,'images':[samples[int(i)] for i in indices],'physical_gpu':gpu,'updated_utc':now()}
        arrays={}
        with torch.inference_mode():
            x=image_tensor(pixels,indices);out=model(x)
            arrays.update(original=x.cpu().numpy(),full_reconstruction=out['reconstruction'].cpu().numpy())
            for s in model.scales:
                g=int(selected[s]['gene_key'].split(':')[1])
                acts={key:value.clone() for key,value in out['activations'].items()};acts[s][:,g]=0
                deleted=model.decoder(model.synthesize_codes(acts))
                expression=out['activations'][s][:,g].cpu().numpy()
                atom=model.dictionaries[s].atoms[g].cpu().numpy()
                diff=(out['reconstruction']-deleted).abs().mean(1).cpu().numpy()
                per_image=[]
                for i in range(len(indices)):
                    per_image.append({'index':int(indices[i]),'expression_min':float(expression[i].min()),'expression_max':float(expression[i].max()),
                        'expression_spatial_std':float(expression[i].std()),'full_mae':float((out['reconstruction'][i]-x[i]).abs().mean()),
                        'deleted_mae':float((deleted[i]-x[i]).abs().mean()),'rgb_change_mae':float(diff[i].mean()),'difference_fraction_above_display_limit':float((diff[i]>.25).mean())})
                metadata[s]={'gene':g,'atom_shape':list(atom.shape),'expression_shape':list(expression.shape[1:]),'per_image':per_image}
                rows[s]={'gene':g,'expression':expression,'atom':atom,'deleted':deleted,'diff':diff}
                arrays.update({s+'_expression':expression,s+'_atom':atom,s+'_deleted':deleted.cpu().numpy(),s+'_abs_rgb_change':diff})

            # Same photograph in every row isolates the scale comparison.
            fig,axes=plt.subplots(3,6,figsize=(17,11))
            columns=['Shared gene weights','Expression on this image','Original','Full reconstruction','Delete this gene','Deletion RGB change']
            for row,s in enumerate(model.scales):
                d=rows[s];atom=d['atom'];kernel=atom.shape[-1]
                tiled=np.full((8*(kernel+1)-1,8*(kernel+1)-1),np.nan)
                for channel in range(64):
                    yy,xx=divmod(channel,8);tiled[yy*(kernel+1):yy*(kernel+1)+kernel,xx*(kernel+1):xx*(kernel+1)+kernel]=atom[channel]
                extent=max(float(np.abs(atom).max()),1e-8)
                axes[row,0].imshow(tiled,cmap='coolwarm',vmin=-extent,vmax=extent,interpolation='nearest')
                axes[row,0].set_ylabel(f'{s} gene {d["gene"]}\nweights {tuple(atom.shape)}',fontsize=11)
                axes[row,1].imshow(d['expression'][0],vmin=0,vmax=4,cmap='magma',interpolation='nearest')
                axes[row,2].imshow(rgb(x[0]));axes[row,3].imshow(rgb(out['reconstruction'][0]))
                axes[row,4].imshow(rgb(d['deleted'][0]));axes[row,5].imshow(d['diff'][0],vmin=0,vmax=.25,cmap='inferno')
                for j in range(6):
                    plain(axes[row,j])
                    if row==0:axes[row,j].set_title(columns[j],fontsize=11)
                axes[row,1].set_xlabel(f'{d["expression"].shape[1]} x {d["expression"].shape[2]}; range 0-4',fontsize=9)
                axes[row,5].set_xlabel(f'Mean RGB change {d["diff"][0].mean():.5f}; display 0-0.25',fontsize=9)
            fig.suptitle(f'Actual coarse / medium / fine genes | {variant}, checkpoint {saved["step"]}, cap 64\nSame fixed city photo in all rows; kernel colors are feature weights, expression is not a segmentation mask',fontsize=13)
            fig.tight_layout(rect=[0,0.03,1,.94],h_pad=3.0);fig.savefig(destination/'scale_gene_overview.png',dpi=150);plt.close(fig)

            for s,d in rows.items():
                fig,axes=plt.subplots(3,5,figsize=(14,9))
                for i,index in enumerate(indices):
                    axes[i,0].imshow(rgb(x[i]));axes[i,1].imshow(rgb(out['reconstruction'][i]))
                    axes[i,2].imshow(d['expression'][i],vmin=0,vmax=4,cmap='magma',interpolation='nearest')
                    axes[i,3].imshow(rgb(d['deleted'][i]));axes[i,4].imshow(d['diff'][i],vmin=0,vmax=.25,cmap='inferno')
                    axes[i,0].set_ylabel(samples[int(index)]['city'],fontsize=10)
                    for j,label in enumerate(['Original','Full reconstruction','Expression 0-4','Delete gene','Abs RGB change 0-0.25']):
                        plain(axes[i,j])
                        if i==0:axes[i,j].set_title(label,fontsize=11)
                fig.suptitle(f'Same {s} gene {d["gene"]} on three fixed city images | checkpoint {saved["step"]}',fontsize=14)
                fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(destination/f'{s}_gene_reuse.png',dpi=150);plt.close(fig)
        np.savez_compressed(destination/'scale_gene_arrays.npz',**arrays)
        write_json(destination/'metadata.json',metadata)
        write_json(destination/'status.json',{'status':'complete','updated_utc':now()})
        print(json.dumps({'status':'complete','destination':str(destination),'selected_genes':{s:g['gene_key'] for s,g in selected.items()},'image_indices':indices.tolist()}))
        return True
    finally:lease.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not run(load_config(args.config),Path(args.output)):raise SystemExit(2)
