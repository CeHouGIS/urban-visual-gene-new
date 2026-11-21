"""Show exact quantized SAE activations over photos, without neural inference."""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from common import config,now,rows,write_json


def overlay(ax,pixels,heat,maximum):
    ax.imshow(pixels)
    # Preserve the original photo where the coefficient is exactly zero.
    ax.imshow(np.ma.masked_where(heat==0,heat),cmap='inferno',vmin=0,vmax=maximum,
        alpha=.72,extent=(-.5,255.5,255.5,-.5),interpolation='nearest')
    ax.set_xticks([]);ax.set_yticks([])


def main(c,index=666,row=16,col=16):
    root=Path(c['root']);out=root/'reports/feature_overlays';out.mkdir(exist_ok=True)
    metadata=rows(c,'validation')
    pixels=np.load(Path(c['rgb_cache'])/'validation.rgb.npy',mmap_mode='r')
    if not (0<=index<len(metadata) and 0<=row<32 and 0<=col<32):raise ValueError('Invalid image/grid index')
    delta=c['base']['compression']['quantization_step']
    models={};maximum=0.
    for m in c['dictionary_sizes']:
        name=f'atoms_{m}'
        q=np.load(Path(c['local'])/'exports'/name/'validation.codes.npy',mmap_mode='r')
        point=q[index,:,row,col].astype(np.float32)*delta
        top=np.lexsort((np.arange(m),-point))[:12]
        maximum=max(maximum,float(q[index,top].max())*delta)
        means=np.load(root/'exports'/name/'per_image_mean_weights.npy',mmap_mode='r')
        models[m]=dict(codes=q,top=top,point=point,means=means)
    maximum=max(delta,maximum)
    report={}
    for m,model in models.items():
        q,top,point=model['codes'],model['top'],model['point']
        name=f'atoms_{m}';atoms=[]
        fig,axes=plt.subplots(3,5,figsize=(17,10.8),constrained_layout=True)
        for r in range(3):
            axes[r,0].imshow(pixels[index]);axes[r,0].add_patch(Rectangle((col*8-.5,row*8-.5),8,8,
                fill=False,color='red',linewidth=1.5))
            axes[r,0].set_title(f'Original #{index} | {metadata[index]["city"]}\nRed = grid ({row},{col}), zero-based',fontsize=10)
            axes[r,0].set_xticks([]);axes[r,0].set_yticks([])
            for j in range(4):
                g=int(top[r*4+j]);heat=q[index,g].astype(np.float32)*delta
                overlay(axes[r,j+1],pixels[index],heat,maximum)
                axes[r,j+1].set_title(f'Atom {g} | weight at red grid={point[g]:.3f}\nmean={heat.mean():.3f}; active={100*(heat>0).mean():.1f}%',fontsize=10)
                atoms.append(dict(atom_id=g,point_weight=float(point[g]),image_mean_weight=float(heat.mean()),
                    active_position_fraction=float((heat>0).mean())))
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,maximum),cmap='inferno'),
            ax=axes[:,1:].ravel().tolist(),location='bottom',pad=.01,shrink=.65,label='Activation A = 0.125 Q (same color range across both SAEs)')
        fig.suptitle(f'{m}-atom SAE | 12 largest coefficients at one fixed query position\n'
            'Color = activation strength; zero regions show original photo; these are response maps, not object masks',fontsize=13)
        fig.savefig(out/f'query_{index:04d}_atoms_{m}.png',dpi=140);plt.close(fig)
        cross=[]
        fig,axes=plt.subplots(4,4,figsize=(14,14),constrained_layout=True)
        for r,g in enumerate(top[:4]):
            g=int(g);selected=[];grids={metadata[index]['grid_id']}
            for candidate in np.argsort(-model['means'][:,g],kind='stable'):
                candidate=int(candidate);grid=metadata[candidate]['grid_id']
                if grid in grids:continue
                selected.append(candidate);grids.add(grid)
                if len(selected)==3:break
            indices=[index]+selected
            # Use the actual maximum for this atom across these four photos.
            vmax=max(delta,float(q[indices,g].max())*delta)
            for j,i in enumerate(indices):
                heat=q[i,g].astype(np.float32)*delta
                overlay(axes[r,j],pixels[i],heat,vmax)
                axes[r,j].set_title(f'Atom {g} | {metadata[i]["city"]} | #{i}\n'
                    f'mean={heat.mean():.3f}; max={heat.max():.3f}',fontsize=10)
            cross.append(dict(atom_id=g,query_index=index,other_indices=selected,color_range=[0,vmax]))
        fig.suptitle(f'{m}-atom SAE | same atom across different street scenes\n'
            'Column 1 = query; columns 2-4 = highest mean response in distinct other grids; one color range per row',fontsize=13)
        fig.savefig(out/f'cross_scene_atoms_{m}.png',dpi=140);plt.close(fig)
        report[name]=dict(selected_atoms=atoms,cross_scene=cross)
    write_json(out/'selection.json',dict(validation_index=index,grid_row=row,grid_col=col,
        selection='Top 12 at the previously inspected grid position, ties broken by atom ID; no semantic selection.',
        query_color_range=[0,maximum],models=report,updated_utc=now()))
    print(str(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,default=666)
    p.add_argument('--row',type=int,default=16);p.add_argument('--col',type=int,default=16)
    args=p.parse_args();main(config(),args.index,args.row,args.col)
