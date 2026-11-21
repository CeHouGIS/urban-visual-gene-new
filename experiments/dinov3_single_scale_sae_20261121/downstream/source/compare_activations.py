"""Compare saved final SAE activations on identical photos; no neural inference."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from common import config,rows,write_json,now


def main(c):
    root=Path(c['root']);out=root/'reports/activation_comparison';out.mkdir(exist_ok=True)
    metadata=rows(c,'validation')
    pixels=np.load(Path(c['rgb_cache'])/'validation.rgb.npy',mmap_mode='r')
    models={};summary={}
    queries=[666,454,724]
    delta=c['base']['compression']['quantization_step'];maximum=63*delta
    for m in c['dictionary_sizes']:
        name=f'atoms_{m}';stats=np.load(root/'exports'/name/'atom_statistics.npz')
        top=np.argsort(-stats['mean_weight'],kind='stable')[:4]
        codes=np.load(Path(c['local'])/'exports'/name/'validation.codes.npy',mmap_mode='r')
        means=np.load(root/'exports'/name/'per_image_mean_weights.npy',mmap_mode='r')
        models[m]=dict(top=top,codes=codes)
        summary[name]=dict(top_atoms=top.tolist(),mean_active_per_position=float(stats['coverage'].sum()),
            nonzero_percent=float(stats['coverage'].mean()*100),
            mean_distinct_atoms_per_image=float((means>0).sum(1).mean()),
            unused_atoms=int((stats['image_presence']==0).sum()),queries={})
    for index in queries:
        fig,axes=plt.subplots(2,5,figsize=(15,6.6),constrained_layout=True)
        for row,(m,model) in enumerate(models.items()):
            q=np.array(model['codes'][index],copy=True)
            count=float((q>0).sum()/1024);present=int((q>0).any((1,2)).sum())
            summary[f'atoms_{m}']['queries'][str(index)]=dict(mean_active_per_position=count,
                distinct_atoms_in_image=present,displayed_atoms=model['top'].tolist())
            axes[row,0].imshow(pixels[index]);axes[row,0].set_title(
                f'{m}-atom SAE | same original #{index}\n{count:.1f} active / position; {present} used in image',fontsize=9)
            for col,g in enumerate(model['top'],1):
                heat=q[g].astype(np.float32)*delta
                axes[row,col].imshow(pixels[index])
                axes[row,col].imshow(heat,cmap='inferno',vmin=0,vmax=maximum,alpha=.68,
                    extent=(-.5,255.5,255.5,-.5),interpolation='nearest')
                axes[row,col].set_title(f'Atom {int(g)} | A = {delta} Q\nmean={heat.mean():.3f}; active={100*(heat>0).mean():.1f}%',fontsize=10)
            for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,maximum),cmap='inferno'),ax=axes[:,1:].ravel().tolist(),
            location='bottom',shrink=.7,pad=.02,label='Activation magnitude (one fixed scale for both SAEs)')
        fig.suptitle(f'Same held-out scene | {metadata[index]["city"]}\nFour largest global-mean atoms per SAE, fixed across scenes; IDs are local to each model',fontsize=12)
        fig.savefig(out/f'same_scene_{index:04d}.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    for row,(m,model) in enumerate(models.items()):
        for col,index in enumerate(queries):
            active=(np.array(model['codes'][index],copy=True)>0).sum(0)
            im=axes[row,col].imshow(active,cmap='viridis',vmin=0,vmax=600,interpolation='nearest')
            axes[row,col].set_title(f'{m} atoms | #{index} | {metadata[index]["city"]}\nmean active={active.mean():.1f}',fontsize=10)
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
    fig.colorbar(im,ax=axes.ravel().tolist(),location='bottom',shrink=.65,label='Number of active atoms at each 32 x 32 grid position')
    fig.suptitle('Activation counts on identical photos | fixed color scale across models')
    fig.savefig(out/'active_counts.png',dpi=140);plt.close(fig)
    write_json(out/'comparison.json',dict(images=1000,selection='Four largest mean-weight atoms over all 1000 validation images per model; identical held-out photos.',
        color_limits=[0,maximum],models=summary,updated_utc=now()))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main(config())
