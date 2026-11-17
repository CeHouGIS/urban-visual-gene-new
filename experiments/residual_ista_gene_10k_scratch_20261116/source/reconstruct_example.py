"""Frozen model reconstruction for one preselected held-out street image."""
import argparse
import os
from pathlib import Path
import json
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from common import load_config,write_json,now
from reference.supervisor import acquire


def reconstruct(config,index):
    root=Path(config['root'])
    output=root/'reports'/f'new_image_example_test_{index:04d}'
    output.mkdir(exist_ok=False)
    row=json.loads((root/'data/images.test.jsonl').read_text().splitlines()[index])
    train=[json.loads(x) for x in (root/'data/images.train.jsonl').read_text().splitlines()]
    isolation={f'{field}_disjoint_from_training':row[field] not in {x[field] for x in train}
               for field in ['id','pano','grid_id']}
    if not all(isolation.values()):
        raise AssertionError('Example overlaps training data')
    leased=acquire(config,1)
    while not leased:
        write_json(output/'progress.json',{'status':'waiting_for_gpu','updated_utc':now()})
        time.sleep(10)
        leased=acquire(config,1)
    gpu,handle=leased[0]
    try:
        os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid']
        import torch
        from model import create_model
        from data import image_tensor
        from metrics import ssim_per_image
        torch.set_num_threads(4)
        saved=torch.load(root/'runs/multiscale_residual_ista/checkpoint_best.pt',map_location='cpu',weights_only=False)
        model=create_model(config,'multiscale_residual_ista')
        model.load_state_dict(saved['model'])
        model.set_step(saved['step'])
        model.eval().requires_grad_(False)
        pixels=np.load(Path(config['cache'])/'test.rgb.npy',mmap_mode='r',allow_pickle=False)
        target=image_tensor(pixels,np.array([index]))
        started=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
            result=model(target)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        prediction=result['reconstruction'].float()
        difference=prediction-target
        mse=difference.square().mean()
        metrics={'l1':float(difference.abs().mean()),'psnr':float(-10*torch.log10(mse.clamp_min(1e-12))),
                 'ssim':float(ssim_per_image(prediction,target).mean())}
        original=target[0].cpu().numpy()
        rebuilt=prediction[0].cpu().numpy()
        error=np.abs(rebuilt-original)
        arrays={'original':original,'reconstruction':rebuilt,'absolute_error':error}
        scale_stats={}
        for scale in model.scales:
            a=result['activations'][scale].detach().float().cpu().numpy()
            z=result['features'][scale].detach().float().cpu().numpy()
            arrays[f'{scale}_activations']=a
            arrays[f'{scale}_features']=z
            scale_stats[scale]={'genes_used_on_this_image':int((a>0).any((0,2,3)).sum()),
                'nonzero_coefficient_fraction':float(np.mean(a>0)),'gene_code_shape':list(a.shape[1:]),
                'synthesized_feature_shape':list(z.shape[1:])}
        np.savez_compressed(output/'codes_and_reconstruction.npz',**arrays)
        metadata={'test_index':index,'random_selection_seed':20261116,'image_id':row['id'],
                  'city':row['city'],'country':row['country'],'checkpoint_step':saved['step'],
                  'training_overlap_checks':isolation,'weights_updated':False,'coefficients_updated':'inferred by encoder and 6 unrolled ISTA steps; no test-time optimization',
                  'dictionary_shared_with_training':True,'scales':scale_stats,'single_image_metrics':metrics,
                  'evaluation_scope':'One randomly preselected held-out image for visualization; not a new aggregate benchmark.',
                  'GPU':gpu,'first_forward_seconds_including_CUDA_kernel_warmup':elapsed,'updated_utc':now()}
        write_json(output/'metadata.json',metadata)
    finally:
        handle.close()
    def rgb(value):
        return Image.fromarray(np.rint(np.clip(value.transpose(1,2,0),0,1)*255).astype(np.uint8))
    images=[rgb(original),rgb(rebuilt),rgb(error*3)]
    for name,im in zip(['original.png','reconstruction.png','absolute_error_x3.png'],images):
        im.save(output/name)
    side=256;header=66;footer=54
    panel=Image.new('RGB',(3*side,side+header+footer),'#101820')
    draw=ImageDraw.Draw(panel)
    path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(path,18);small=ImageFont.truetype(path,14)
    draw.text((10,9),f'Held-out street image: {row["city"]}, {row["country"]}',font=font,fill='white')
    for c,(name,im) in enumerate(zip(['Original photo','Reconstruction','Absolute error x3'],images)):
        draw.text((c*side+8,43),name,font=font,fill='#e1e7eb')
        panel.paste(im,(c*side,header))
    draw.text((10,header+side+9),f'PSNR {metrics["psnr"]:.2f} dB   |   SSIM {metrics["ssim"]:.3f}   |   MAE {metrics["l1"]:.4f}',font=small,fill='#c1cbd2')
    draw.text((10,header+side+31),'Frozen encoder, dictionaries and decoder; only this image\'s activation codes change.',font=small,fill='#c1cbd2')
    panel.save(output/'original_reconstruction_error.png')
    write_json(output/'progress.json',{'status':'complete','updated_utc':now()})
    print(json.dumps({'output':str(output),'single_image_metrics':metrics,'scales':scale_stats},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--test-index',type=int,default=312)
    args=p.parse_args()
    reconstruct(load_config(args.config),args.test_index)
