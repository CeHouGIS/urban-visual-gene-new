"""Preselect ten held-out panoramas at random; reconstruct with frozen weights."""
import argparse
import csv
import json
import os
from pathlib import Path
import time
import zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from common import load_config,write_json,now
from reference.supervisor import acquire


def select(root,seed,count,output):
    rows=[json.loads(x) for x in (root/'data/images.test.jsonl').read_text().splitlines()]
    prior=[0,199,399,599,799,999,312]
    used_panoramas={rows[i]['pano'] for i in prior}
    selected=[]
    for i in np.random.default_rng(seed).permutation(len(rows)):
        index=int(i)
        if rows[index]['pano'] in used_panoramas:
            continue
        selected.append({'order':len(selected)+1,'test_index':index,**rows[index]})
        used_panoramas.add(rows[index]['pano'])
        if len(selected)==count:
            break
    if len(selected)!=count:
        raise ValueError('Not enough distinct held-out panoramas')
    train=[json.loads(x) for x in (root/'data/images.train.jsonl').read_text().splitlines()]
    checks={f'{field}_disjoint_from_training':{x[field] for x in selected}.isdisjoint({x[field] for x in train})
            for field in ['id','pano','grid_id']}
    if not all(checks.values()):
        raise AssertionError('Random examples overlap training data')
    manifest={'seed':seed,'count':count,'selection_policy':'Fixed RNG permutation of the 1000-image test list; skip previously displayed panoramas and take the first image from each distinct panorama.',
              'excluded_prior_example_indices':prior,'selected_before_model_inference':True,
              'quality_based_filtering_or_reselection':False,'isolation_checks':checks,
              'unique_panoramas':len({x['pano'] for x in selected}),'samples':selected,'created_utc':now()}
    write_json(output/'selection.json',manifest)
    return selected


def infer(config,selected,output):
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
        root=Path(config['root'])
        checkpoint=root/'runs/multiscale_residual_ista/checkpoint_best.pt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=create_model(config,'multiscale_residual_ista')
        model.load_state_dict(saved['model'])
        model.set_step(saved['step'])
        model.eval().requires_grad_(False)
        pixels=np.load(Path(config['cache'])/'test.rgb.npy',mmap_mode='r',allow_pickle=False)
        originals=[];predictions=[];metrics=[]
        for start in range(0,len(selected),5):
            subset=selected[start:start+5]
            target=image_tensor(pixels,np.array([row['test_index'] for row in subset]))
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                result=model(target)
            prediction=result['reconstruction'].float()
            if prediction.shape!=target.shape or not torch.isfinite(prediction).all():
                raise AssertionError('Invalid reconstruction tensor')
            with torch.inference_mode():
                absolute=(prediction-target).abs()
                mse=(prediction-target).square().mean((1,2,3))
                l1=absolute.mean((1,2,3))
                psnr=-10*torch.log10(mse.clamp_min(1e-12))
                ssim=ssim_per_image(prediction,target)
                code_usage={s:(result['activations'][s]>0).any((2,3)).sum(1).cpu().tolist() for s in model.scales}
            originals.extend(target.cpu().numpy())
            predictions.extend(prediction.cpu().numpy())
            for local,row in enumerate(subset):
                metrics.append({'order':row['order'],'test_index':row['test_index'],'image_id':row['id'],
                    'city':row['city'],'country':row['country'],'continent':row['continent'],
                    'psnr':float(psnr[local]),'ssim':float(ssim[local]),'mae':float(l1[local]),
                    'genes_used':{s:int(code_usage[s][local]) for s in model.scales}})
            write_json(output/'progress.json',{'status':'inference','images_done':len(metrics),'images_total':len(selected),'updated_utc':now()})
        np.savez_compressed(output/'reconstruction_float.npz',originals=np.stack(originals),reconstructions=np.stack(predictions),
                            test_indices=np.array([x['test_index'] for x in selected]))
        summary={'images':len(metrics),'mean_psnr':float(np.mean([x['psnr'] for x in metrics])),
                 'mean_ssim':float(np.mean([x['ssim'] for x in metrics])),'mean_mae':float(np.mean([x['mae'] for x in metrics])),
                 'psnr_range':[min(x['psnr'] for x in metrics),max(x['psnr'] for x in metrics)],
                 'checkpoint':str(checkpoint),'checkpoint_step':saved['step'],'weights_updated':False,
                 'test_time_optimization':False,'GPU':gpu,
                 'scope':'Random ten-image visual diagnosis, not a replacement for the original 1000-image test benchmark.',
                 'per_image':metrics,'updated_utc':now()}
        write_json(output/'metrics.json',summary)
        with (output/'metrics.csv').open('w',newline='') as stream:
            names=['order','test_index','image_id','city','country','continent','psnr','ssim','mae']
            writer=csv.DictWriter(stream,fieldnames=names)
            writer.writeheader()
            writer.writerows([{k:r[k] for k in names} for r in metrics])
        print(json.dumps({k:v for k,v in summary.items() if k not in ['per_image']},ensure_ascii=False),flush=True)
    finally:
        handle.close()


def render(output):
    arrays=np.load(output/'reconstruction_float.npz',allow_pickle=False)
    metrics=json.loads((output/'metrics.json').read_text())
    data=metrics['per_image']
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    fonts={n:ImageFont.truetype(font_path,n) for n in [13,14,17,20]}
    side=256;width=768;row_header=44;row_footer=32;row_height=side+row_header+row_footer
    tiles=[]
    for i,record in enumerate(data):
        original=arrays['originals'][i]
        reconstruction=arrays['reconstructions'][i]
        error=np.abs(reconstruction-original)*3
        folder=output/'images'/f'{i+1:02d}_test_{record["test_index"]:04d}'
        folder.mkdir(parents=True,exist_ok=True)
        triplet=[]
        for name,value in [('original',original),('reconstruction',reconstruction),('absolute_error_x3',error)]:
            image=Image.fromarray(np.rint(np.clip(value.transpose(1,2,0),0,1)*255).astype(np.uint8))
            image.save(folder/f'{name}.png')
            triplet.append(image)
        tile=Image.new('RGB',(width,row_height),'#101820')
        draw=ImageDraw.Draw(tile)
        label=f'{i+1:02d}  {record["city"]}, {record["country"]}  |  test #{record["test_index"]}'
        draw.text((8,5),label,font=fonts[17],fill='white')
        for col,(text,image) in enumerate(zip(['Original','Reconstruction','Absolute error x3'],triplet)):
            draw.text((col*side+8,27),text,font=fonts[13],fill='#c1cbd2')
            tile.paste(image,(col*side,row_header))
        draw.text((8,row_header+side+8),f'PSNR {record["psnr"]:.2f} dB   |   SSIM {record["ssim"]:.3f}   |   MAE {record["mae"]:.4f}',font=fonts[14],fill='#c1cbd2')
        tile.save(folder/'comparison.png')
        tiles.append(tile)
    for part in range(2):
        selected=tiles[part*5:(part+1)*5]
        panel=Image.new('RGB',(width,46+row_height*len(selected)),'#101820')
        draw=ImageDraw.Draw(panel)
        draw.text((8,10),f'Random held-out street images {part*5+1:02d}-{part*5+5:02d} | Frozen multiscale model',font=fonts[17],fill='white')
        for row,tile in enumerate(selected):
            panel.paste(tile,(0,46+row*row_height))
        panel.save(output/f'random10_comparison_part{part+1}.png')
    overview=Image.new('RGB',(width*2,46+row_height*5),'#101820')
    for part in range(2):
        with Image.open(output/f'random10_comparison_part{part+1}.png') as image:
            overview.paste(image,(part*width,0))
    overview.save(output/'random10_overview.png')
    (output/'README.md').write_text('# 随机10张未见街景重建\n\n固定随机种子 20261116，从1000张测试图片中打乱抽样，排除此前展示过的全景，并保证本次10张来自不同全景。所有样本在推理前写入 selection.json，未根据重建质量筛选或重抽。图片、全景和格网均不与训练集重叠；同城可以跨训练/测试。\n\n使用6000步选优的多尺度模型，编码器、共享字典和解码器全部冻结，不做测试时优化。每张图为256×256。各行从左到右：原图、重建、RGB绝对误差乘3；显示裁剪到[0,1]。指标从未量化的浮点预测计算。\n\nrandom10_comparison_part1.png 和 part2.png 各5张；random10_overview.png 为全部10张；images/ 下保留原图、输出、误差和单张对照。metrics.json/metrics.csv 是每图和10图均值。此处为随机样本展示，不替代此前1000图的正式测试结果。\n')
    zip_path=output/'random10_reconstruction_images.zip'
    paths=[output/'selection.json',output/'metrics.json',output/'metrics.csv',output/'README.md',
           output/'random10_comparison_part1.png',output/'random10_comparison_part2.png',output/'random10_overview.png']
    paths.extend(sorted((output/'images').rglob('*.png')))
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in paths:
            archive.write(p,arcname=p.relative_to(output))
    # Inspect completed artifacts without rerunning the model.
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise AssertionError('Generated download archive is invalid')
    for p in [output/'random10_comparison_part1.png',output/'random10_comparison_part2.png',output/'random10_overview.png']:
        with Image.open(p) as image:
            image.verify()
    write_json(output/'progress.json',{'status':'complete','images':len(data),'download_zip':str(zip_path),
        'download_bytes':zip_path.stat().st_size,'updated_utc':now()})
    print(json.dumps({'output':str(output),'download':str(zip_path),'download_MB':zip_path.stat().st_size/1e6},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--seed',type=int,default=20261116)
    args=parser.parse_args()
    config=load_config(args.config)
    output=Path(config['root'])/'reports'/f'random10_unseen_reconstruction_seed_{args.seed}'
    output.mkdir(exist_ok=False)
    selected=select(Path(config['root']),args.seed,10,output)
    print(json.dumps({'selected_indices':[x['test_index'] for x in selected],
                      'cities':[x['city'] for x in selected]},ensure_ascii=False),flush=True)
    infer(config,selected,output)
    render(output)
