"""CPU-only rendering of saved DINO maps and SAE activations, never neural inference."""
import json
import fcntl
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from features import configuration
from utils import now, write_json


def colors(features, basis):
    rgb = (features.transpose(0,2,3,1).astype(np.float32)-basis['mean'])@basis['basis']
    return np.clip((rgb-basis['low'])/(basis['high']-basis['low']).clip(1e-6),0,1)


def main(c):
    root = Path(c['root'])
    report = root/'reports'
    lock = (report/'render.lock').open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        return
    data_path = report/'dino_query_features.npz'
    if not data_path.exists():
        return
    data = np.load(data_path)
    basis = np.load(report/'visualization_pca.npz')
    rgb,raw,ids = data['rgb'],data['features'].astype(np.float32),data['indices']
    pca = colors(raw,basis)
    rows = [json.loads(s) for s in Path(c['manifests']['validation']).read_text().splitlines()]
    # Use one train-fitted 3D projection with one fixed color range for all images.
    # Channels do not represent tree/building/road labels.
    fig,axes = plt.subplots(len(ids),3,figsize=(12,4*len(ids)))
    for i,index in enumerate(ids):
        axes[i,0].imshow(rgb[i])
        axes[i,0].set_title(f'Original | validation {index}\n{rows[int(index)]["city"]}',fontsize=11)
        axes[i,1].imshow(pca[i],interpolation='nearest')
        axes[i,1].set_title('DINOv3 | 1024 channels, 32 x 32\n3 PCA axes shown as RGB',fontsize=11)
        # Similarity to a fixed location in the SAME image, after train mean subtraction.
        f = raw[i]-basis['mean'][:,None,None]
        f = f/np.maximum(np.linalg.norm(f,axis=0,keepdims=True),1e-8)
        y,x = 23,16
        affinity = np.einsum('chw,c->hw',f,f[:,y,x])
        axes[i,2].imshow(rgb[i])
        axes[i,2].imshow(affinity,cmap='magma',vmin=-1,vmax=1,alpha=.65,
                         extent=(-.5,511.5,511.5,-.5),interpolation='nearest')
        axes[i,2].scatter((x+.5)*16,(y+.5)*16,c='cyan',s=45,marker='+')
        axes[i,2].set_title('Feature similarity to cyan point\ncosine, train-centered; same image',fontsize=11)
        for ax in axes[i]: ax.axis('off')
    fig.suptitle('Frozen pretrained DINOv3 ViT-L/16 | held-out streets | not SAE atom maps',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.98))
    fig.savefig(report/'dinov3_features_five_scenes.png',dpi=130)
    plt.close(fig)
    completed=[]
    for m in c['dictionary_sizes']:
        directory=root/'runs'/f'atoms_{m:04d}'
        if not all((directory/name).exists() for name in ['query_outputs.npz','atom_statistics.npz','evaluation.json']): continue
        output=np.load(directory/'query_outputs.npz')
        stats=np.load(directory/'atom_statistics.npz')
        step=int(output['step'])
        metrics=json.loads((directory/'evaluation.json').read_text())
        if metrics['step'] != step: continue
        code=output['codes'].astype(np.float32)*float(output['quantization_step'])
        top=np.argsort(-stats['mean_weight'])[:4]
        fig,axes=plt.subplots(len(ids),5,figsize=(17,3.5*len(ids)))
        for i,index in enumerate(ids):
            axes[i,0].imshow(rgb[i])
            axes[i,0].set_title(f'Validation {index} | {rows[int(index)]["city"]}',fontsize=10)
            axes[i,0].axis('off')
            for j,g in enumerate(top):
                heat=code[i,g]
                vmax=max(float(np.max(code[:,g])),.125)
                axes[i,j+1].imshow(rgb[i])
                im=axes[i,j+1].imshow(heat,cmap='inferno',vmin=0,vmax=vmax,alpha=.7,
                    extent=(-.5,511.5,511.5,-.5),interpolation='nearest')
                axes[i,j+1].set_title(f'Atom {g} | mean={heat.mean():.3f}\nactive={100*(heat>0).mean():.1f}%',fontsize=10)
                axes[i,j+1].axis('off')
        fig.suptitle(f'DINOv3 + SAE {m} | checkpoint step {step}/{c["compression"]["steps"]}\n'
            'Same 4 atoms in all 5 images; fixed color scale per atom; no semantic labels',fontsize=14)
        fig.tight_layout(rect=(0,0,1,.96))
        fig.savefig(report/f'sae_{m}_same_atoms.png',dpi=120)
        plt.close(fig)
        normalization=np.load(root/'features/normalization.npz')
        restored=output['restored'].astype(np.float32)*float(normalization['scale'])+normalization['mean'][None,:,None,None]
        prediction=colors(restored,basis)
        fig,axes=plt.subplots(len(ids),3,figsize=(11,3.6*len(ids)))
        for i,index in enumerate(ids):
            axes[i,0].imshow(rgb[i]); axes[i,0].set_title(f'Original {index}')
            axes[i,1].imshow(pca[i],interpolation='nearest'); axes[i,1].set_title('DINO features (PCA color)')
            axes[i,2].imshow(prediction[i],interpolation='nearest'); axes[i,2].set_title('SAE restored features (same PCA)')
            for ax in axes[i]: ax.axis('off')
        fig.suptitle(f'{m} atoms | step {step} | feature-space reconstruction; these are not generated RGB photos')
        fig.tight_layout(rect=(0,0,1,.98))
        fig.savefig(report/f'sae_{m}_feature_reconstruction.png',dpi=120)
        plt.close(fig)
        completed.append(metrics)
    fig,ax=plt.subplots(figsize=(8,4))
    lines=0
    for m in c['dictionary_sizes']:
        file=root/'runs'/f'atoms_{m:04d}'/'train.jsonl'
        if file.exists():
            values=[json.loads(s) for s in file.read_text().splitlines() if s]
            if values:
                ax.plot([v['step'] for v in values],[v['nmse'] for v in values],label=str(m),alpha=.7)
                lines+=1
    if lines:
        ax.set(xlabel='SAE optimizer update',ylabel='Train centered-feature NMSE',yscale='log')
        ax.legend(title='Dictionary atoms'); ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(report/'training_curves.png',dpi=130)
    plt.close(fig)
    write_json(report/'results.json',dict(updated_utc=now(),runs=completed,
        dino_raw_features=str(data_path),visualization='3 PCA axes fitted on 32 TRAIN photos, used for plotting only',
        caveat='Increasing dimension/backbone strength does not enforce spatial connectivity or semantic purity.'))


if __name__=='__main__':
    main(configuration())
