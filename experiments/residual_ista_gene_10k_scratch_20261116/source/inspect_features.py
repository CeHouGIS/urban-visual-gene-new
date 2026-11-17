"""Inspect two fixed preview samples with a leased GPU; CPU renders exported maps."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import argparse
from pathlib import Path
import json
import time
import numpy as np
from common import load_config, write_json, now
from reference.supervisor import acquire


def export(config, output):
    leased=acquire(config,1)
    while not leased:
        write_json(output/'progress.json',{'status':'waiting_for_gpu','cpu_fallback':False,'updated_utc':now()})
        time.sleep(10)
        leased=acquire(config,1)
    gpu,handle=leased[0]
    try:
        os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid']
        import torch
        from model import create_model
        from data import image_tensor
        torch.set_num_threads(4)
        root=Path(config['root'])
        saved=torch.load(root/'runs/multiscale_residual_ista/checkpoint_best.pt',map_location='cpu',weights_only=False)
        model=create_model(config,'multiscale_residual_ista')
        model.load_state_dict(saved['model'])
        model.set_step(saved['step'])
        model.eval()
        indices=np.array([0,199])
        pixels=np.load(Path(config['cache'])/'test.rgb.npy',mmap_mode='r',allow_pickle=False)
        images=image_tensor(pixels,indices)
        raw={}
        def capture(module,inputs,outputs):
            raw.update({s:x.detach() for s,x in outputs.items()})
        hook=model.encoder.register_forward_hook(capture)
        enabled=torch.cuda.is_bf16_supported()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=enabled):
            result=model(images)
        hook.remove()
        arrays={'image_indices':indices,'images':images.cpu().numpy(),
                'reconstruction':result['reconstruction'].float().cpu().numpy()}
        metadata={'status':'complete','checkpoint_step':saved['step'],'indices':indices.tolist(),
                  'GPU':gpu,'BF16':enabled,'metrics_recomputed':False,'training_performed':False,
                  'note':'Two existing preview images; visualization diagnosis only. PCA colors have no assigned semantics.',
                  'scales':{},'updated_utc':now()}
        for s in model.scales:
            for group,tensor in [('raw_encoder',raw[s]),('normalized_target',result['feature_targets'][s]),
                                 ('gene_code',result['activations'][s]),('synthesized_feature',result['features'][s]),
                                 ('dictionary',model.dictionaries[s].atoms)]:
                arrays[f'{s}_{group}']=tensor.detach().float().cpu().numpy()
            z=arrays[f'{s}_synthesized_feature'];a=arrays[f'{s}_gene_code']
            metadata['scales'][s]={'raw_encoder_shape':list(raw[s].shape),'feature_target_shape':list(result['feature_targets'][s].shape),
                'gene_code_shape':list(a.shape),'synthesized_feature_shape':list(z.shape),
                'dictionary_shape':list(model.dictionaries[s].atoms.shape),
                'feature_min':float(z.min()),'feature_max':float(z.max()),'feature_rms':float(np.sqrt(np.mean(z*z))),
                'nonzero_coefficient_fraction':float(np.mean(a>0)),
                'sample0_center_feature_vector':z[0,:,z.shape[2]//2,z.shape[3]//2].tolist()}
        np.savez_compressed(output/'feature_tensors.npz',**arrays)
        write_json(output/'metadata.json',metadata)
        write_json(output/'progress.json',{'status':'export_complete','updated_utc':now()})
    finally:
        handle.close()


def render(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=np.load(output/'feature_tensors.npz',allow_pickle=False)
    metadata=json.loads((output/'metadata.json').read_text())
    scales=['coarse','medium','fine']
    pca_info={}
    colors={}
    for s in scales:
        z=data[f'{s}_synthesized_feature']
        b,c,h,w=z.shape
        values=z.transpose(0,2,3,1).reshape(-1,c).astype(np.float64)
        center=values.mean(0)
        values-=center
        eigenvalue,eigenvector=np.linalg.eigh(values.T@values/(len(values)-1))
        basis=eigenvector[:,-3:][:,::-1].copy()
        # Deterministic sign convention for exported PCA bases.
        for k in range(3):
            if basis[np.abs(basis[:,k]).argmax(),k]<0:
                basis[:,k]*=-1
        projected=values@basis
        lo,hi=np.quantile(projected,[.01,.99],axis=0)
        rgb=np.clip((projected-lo)/np.maximum(hi-lo,1e-9),0,1).reshape(b,h,w,3)
        colors[s]=rgb
        pca_info[s]={'fitted_on':'both displayed samples, shared basis and per-component display limits within scale',
            'center':center.tolist(),'basis':basis.tolist(),'display_lower_quantile_1pct':lo.tolist(),
            'display_upper_quantile_99pct':hi.tolist(),'explained_variance_ratio_3components':(eigenvalue[-3:][::-1]/max(eigenvalue.sum(),1e-12)).tolist()}
    fig,axes=plt.subplots(2,4,figsize=(12,6.2))
    for row in range(2):
        axes[row,0].imshow(data['images'][row].transpose(1,2,0))
        axes[row,0].set_title(f'Photo {row+1}: original')
        for col,s in enumerate(scales,1):
            size=colors[s].shape[1]
            axes[row,col].imshow(colors[s][row],interpolation='nearest')
            axes[row,col].set_title(f'{s}: 64 x {size} x {size}\nPCA RGB display')
        for ax in axes[row]:
            ax.axis('off')
    fig.suptitle('Actual dictionary-synthesized features Z = D * A\nColors are PCA coordinates, not semantic labels',fontsize=12)
    fig.tight_layout()
    fig.savefig(output/'synthesized_feature_pca.png',dpi=160)
    plt.close(fig)
    selected=[0,8,16,24,32,40,48,56]
    fig,axes=plt.subplots(3,8,figsize=(14,5.8),squeeze=False)
    channel_limits={}
    for row,s in enumerate(scales):
        z=data[f'{s}_synthesized_feature'][0]
        limit=max(float(np.quantile(np.abs(z),.99)),1e-5)
        channel_limits[s]=limit
        for col,ch in enumerate(selected):
            axes[row,col].imshow(z[ch],cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
            axes[row,col].set_title(f'{s}\nchannel {ch}',fontsize=10)
            axes[row,col].axis('off')
    fig.suptitle('Photo 1: selected channels from the actual 64-channel feature maps\nRed = positive; blue = negative; fixed channels; common symmetric color limit within each scale',fontsize=11)
    fig.tight_layout()
    fig.savefig(output/'selected_feature_channels.png',dpi=160)
    plt.close(fig)
    pca_info['display_note']='PCA bases differ across scales. RGB colors are not directly comparable between scales. Feature-channel numbers are not gene IDs.'
    pca_info['channel_display_limits']=channel_limits
    write_json(output/'visualization_metadata.json',pca_info)
    write_json(output/'progress.json',{'status':'complete','updated_utc':now()})
    print(json.dumps({'output':str(output),'scales':metadata['scales']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    config=load_config(args.config)
    output=Path(config['root'])/'reports/feature_inspection'
    output.mkdir(exist_ok=True)
    export(config,output)
    render(output)
