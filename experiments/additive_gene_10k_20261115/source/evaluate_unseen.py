"""Detailed inference on fixed held-out photos. No fitting or checkpoint selection.

The parent process leases one idle GPU using the existing project policy. Only
the worker imports torch and runs models. All source images/checkpoints are read
only. Summaries, selected images, and actual gene codes are saved on NAS.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import load_config, now, write_json


def run_worker(config, output):
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    from data import image_tensor, open_pixels
    from metrics import ssim_per_image
    from prepare_data import assert_disjoint
    from train import amp_context, create_model, perceptual_model, to_pil

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Detailed prediction requires exactly one leased GPU')
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    root = Path(config['root'])
    manifests = {s: [json.loads(line) for line in (root/f'data/images.{s}.jsonl').read_text().splitlines()]
                 for s in ['train', 'validation', 'test']}
    assert_disjoint(manifests)
    rows = manifests['test']
    pixels = open_pixels(config, 'test')
    if len(rows) != len(pixels):
        raise ValueError('Test manifest and pixels disagree')
    random_seed = 20261115
    random_indices = np.random.default_rng(random_seed).choice(len(rows), size=8, replace=False).tolist()
    write_json(output/'protocol.json', {
        'created_utc': now(), 'evaluation': 'fixed held-out test inference diagnostics',
        'test_images': len(rows), 'training_performed': False,
        'checkpoint_selection': 'reuse original validation-selected checkpoint for each model',
        'test_reuse': 'same test set as final training evaluation, now adding per-image diagnostics',
        'source_overlap': 'train/validation/test panorama and image IDs disjoint',
        'generalization_scope': 'new panoramas from the same city collection; no held-out-city claim',
        'metric_definitions': 'RGB [0,1]; per-image PSNR; original SSIM and AlexNet LPIPS implementations',
        'random_preview_seed': random_seed, 'random_preview_test_indices': random_indices,
        'ranked_previews': 'nearest images to P10/P50/P90 of sparse model PSNR, explicitly selected by performance',
        'precision': config['training']['precision'], 'gpu_count': 1,
    })
    perceptual = perceptual_model()
    predictions = {}
    per_variant = {}
    summaries = {}
    checkpoints = {}
    started = time.monotonic()
    with torch.inference_mode():
        for variant in config['variants']:
            path = root/f'runs/{variant}/checkpoint_best.pt'
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if checkpoint['config'] != config or checkpoint['variant'] != variant:
                raise ValueError('Checkpoint does not match this experiment')
            model = create_model(config, variant)
            model.load_state_dict(checkpoint['model'])
            model.eval()
            checkpoints[variant] = {'path': str(path), 'step': checkpoint['step']}
            del checkpoint
            output_pixels = np.empty(pixels.shape, dtype=np.uint8)
            values = []
            indices_parts, values_parts = [], []
            micro = config['training']['micro_batch_size']
            for begin in range(0, len(pixels), micro):
                selected = np.arange(begin, min(begin+micro, len(pixels)))
                target = image_tensor(pixels, selected, 'cuda')
                with amp_context(config):
                    result = model(target)
                prediction = result['reconstruction'].float()
                mse = (prediction-target).square().mean((1,2,3))
                metrics = {
                    'l1': (prediction-target).abs().mean((1,2,3)), 'mse': mse,
                    'psnr': -10*torch.log10(mse.clamp_min(1e-12)),
                    'ssim': ssim_per_image(prediction, target),
                    'lpips': perceptual(prediction*2-1, target*2-1).flatten(),
                }
                metrics = {key: value.cpu().numpy() for key,value in metrics.items()}
                if model.sparse:
                    activations = result['activations']
                    gene_counts = (activations>0).any(dim=(2,3)).sum(1).cpu().numpy()
                    nonzeros = (activations>0).sum((1,2,3)).cpu().numpy()
                    code_values, code_indices = activations.topk(model.active,dim=1)
                    indices_parts.append(code_indices.cpu().numpy().astype(np.uint16))
                    values_parts.append(code_values.float().cpu().numpy())
                for offset,index in enumerate(selected):
                    record = {'test_index': int(index), 'id': rows[index]['id'], 'city': rows[index]['city'],
                              'pano': rows[index]['pano'], 'variant': variant,
                              **{key: float(value[offset]) for key,value in metrics.items()}}
                    if model.sparse:
                        record.update(active_gene_types=int(gene_counts[offset]), nonzero_coefficients=int(nonzeros[offset]))
                    values.append(record)
                output_pixels[selected] = prediction.clamp(0,1).mul(255).round().to(torch.uint8).permute(0,2,3,1).cpu().numpy()
                write_json(output/'progress.json', {'status': 'inference', 'variant': variant,
                    'images_completed': int(selected[-1]+1), 'images': len(pixels), 'updated_utc': now()})
            if model.sparse:
                dictionary = model.dictionary.float().cpu().numpy()
                np.savez_compressed(output/'test_gene_codes.npz',
                    gene_indices=np.concatenate(indices_parts), activation_values=np.concatenate(values_parts),
                    dictionary=dictionary, test_ids=np.array([r['id'] for r in rows]),
                    grid_shape=np.array([16,16]), active_per_location=np.array(model.active),
                    explanation=np.array('At each location: Z=sum_k activation_k*dictionary[index_k]; no RGB bypass.'))
            predictions[variant] = output_pixels
            per_variant[variant] = values
            keys = ['psnr', 'ssim', 'lpips', 'l1', 'mse']
            if model.sparse:
                keys += ['active_gene_types', 'nonzero_coefficients']
            summary = {}
            for key in keys:
                numbers = np.array([r[key] for r in values])
                summary[key] = {'mean': float(numbers.mean()), 'median': float(np.median(numbers)),
                    'p10': float(np.quantile(numbers,0.1)), 'p90': float(np.quantile(numbers,0.9)),
                    'minimum': float(numbers.min()), 'maximum': float(numbers.max())}
            # This must reproduce the original aggregate, not create a new result.
            original = json.loads((root/f'runs/{variant}/test_metrics.json').read_text())
            differences = {'psnr': abs(summary['psnr']['mean']-original['psnr_mean_per_image']),
                           'ssim': abs(summary['ssim']['mean']-original['ssim']),
                           'lpips': abs(summary['lpips']['mean']-original['lpips'])}
            if any(delta > 1e-4 for delta in differences.values()):
                raise AssertionError(f'Unexpected aggregate evaluation difference: {differences}')
            summary['original_evaluation_absolute_differences'] = differences
            summaries[variant] = summary
            del model
            torch.cuda.empty_cache()
    for variant,values in per_variant.items():
        write_json(output/f'{variant}.per_image.json', values)
        with (output/f'{variant}.per_image.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    sparse = per_variant['sparse_n256_s4']
    scores = np.array([r['psnr'] for r in sparse])
    order = np.argsort(scores,kind='stable')
    ranked_indices = [int(order[round(q*(len(order)-1))]) for q in [0.1,0.5,0.9]]
    write_json(output/'display_samples.json', {
        'random_indices': random_indices, 'random_selection_before_inference': True,
        'ranked_P10_P50_P90_indices': ranked_indices,
        'ranked_selection_uses': 'sparse model PSNR, ascending; these are not a random sample',
        'samples': {str(i): {'source': rows[i], 'dense': per_variant['dense_ae'][i], 'gene': sparse[i]}
                    for i in set(random_indices+ranked_indices)},
    })

    def sheet(selected, name, labels=None):
        side=256
        header,row_header=34,38
        canvas=Image.new('RGB',(side*4,header+len(selected)*(side+row_header)),'white')
        draw=ImageDraw.Draw(canvas)
        for col,label in enumerate(['Original (unseen panorama)','Ordinary RGB AE','Predicted genes -> RGB','Gene absolute error x3']):
            draw.text((side*col+5,10),label,fill='black')
        for row_number,index in enumerate(selected):
            top=header+row_number*(side+row_header)
            if labels:
                draw.text((5,top+3),labels[row_number],fill='black')
                draw.text((5,top+18),f'test #{index} | {rows[index]["city"]}',fill='black')
            else:
                draw.text((5,top+3),f'test #{index} | {rows[index]["city"]}',fill='black')
            dense_metrics=per_variant['dense_ae'][index]
            gene_metrics=sparse[index]
            draw.text((side+5,top+3),f'PSNR {dense_metrics["psnr"]:.2f} | LPIPS {dense_metrics["lpips"]:.3f}',fill='black')
            draw.text((side*2+5,top+3),f'PSNR {gene_metrics["psnr"]:.2f} | LPIPS {gene_metrics["lpips"]:.3f}',fill='black')
            draw.text((side*2+5,top+18),f'{gene_metrics["active_gene_types"]} gene types in this image',fill='black')
            original=np.array(pixels[index])
            gene=predictions['sparse_n256_s4'][index]
            error=np.minimum(np.abs(original.astype(np.int16)-gene.astype(np.int16))*3,255).astype(np.uint8)
            tiles=[original,predictions['dense_ae'][index],gene,error]
            for col,tile in enumerate(tiles):
                canvas.paste(Image.fromarray(tile),(col*side,top+row_header))
        canvas.save(output/name)

    sheet(random_indices[:4],'unseen_random_first4.png')
    sheet(random_indices[4:],'unseen_random_next4.png')
    sheet(ranked_indices,'unseen_psnr_P10_P50_P90.png',['Lower PSNR (P10)',
          'Median PSNR (P50)','Higher PSNR (P90)'])
    images_dir=output/'samples'
    images_dir.mkdir(exist_ok=True)
    for index in set(random_indices+ranked_indices):
        Image.fromarray(np.array(pixels[index])).save(images_dir/f'{index:04d}.original.png')
        for variant in config['variants']:
            Image.fromarray(predictions[variant][index]).save(images_dir/f'{index:04d}.{variant}.png')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for variant,label in [('dense_ae','Ordinary AE'),('sparse_n256_s4','Additive gene AE')]:
        for ax,key in zip(axes,['psnr','ssim','lpips']):
            numbers=[r[key] for r in per_variant[variant]]
            ax.hist(numbers,bins=30,alpha=0.5,label=label)
    for ax,title in zip(axes,['PSNR (higher is better)','SSIM (higher is better)','LPIPS (lower is better)']):
        ax.set(title=title,ylabel='Test images')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/'test_metric_distributions.png',dpi=160)
    plt.close(fig)
    summaries['protocol']={'test_images':len(rows),'checkpoints':checkpoints,
                          'train_test_panorama_overlap':0,'updated_utc':now()}
    write_json(output/'summary.json',summaries)
    gene_summary=summaries['sparse_n256_s4']
    dense_summary=summaries['dense_ae']
    report=f'''# 未见街景的编码与重建：固定模型预测诊断

日期：2026-11-15（北京时间）。使用原来训练结束后、由验证集选出的第 6000 步模型。本次仅作推理诊断，没有训练、优化、重新选检查点或新增参数。

## 数据与可支持的结论

完整评价 1024 张测试照片，图片和全景 ID 与 10000 张训练照片及验证照片互不重叠。测试城市来自相同的 296 城集合；没有在此证明未见城市泛化，也未额外声称相邻拍摄地点已按距离去重。测试集是之前最终评价用过的同一组，本次细化到逐图指标，不能称为全新的第二个独立测试集。

输入完整街景，经编码器预测空间激活，使用共享字典相加后重建 RGB。它检验新图编码与重建，未检验缺失内容预测、未来街景预测或随机生成新场景。

## 整体指标

| 模型 | 逐图平均 PSNR ↑ | 平均 SSIM ↑ | 平均 LPIPS ↓ |
|---|---:|---:|---:|
| 普通 AE | {dense_summary['psnr']['mean']:.2f} | {dense_summary['ssim']['mean']:.4f} | {dense_summary['lpips']['mean']:.4f} |
| 基因 AE | {gene_summary['psnr']['mean']:.2f} | {gene_summary['ssim']['mean']:.4f} | {gene_summary['lpips']['mean']:.4f} |

基因模型 PSNR 的 P10/中位数/P90 为 {gene_summary['psnr']['p10']:.2f}/{gene_summary['psnr']['median']:.2f}/{gene_summary['psnr']['p90']:.2f} dB。分位数反映图像重建难度差异，也受照片纹理、构图等因素影响，不能仅凭这些数值比较城市。

单图实际使用的不同基因种类：平均 {gene_summary['active_gene_types']['mean']:.1f}，中位数 {gene_summary['active_gene_types']['median']:.0f}，范围 {gene_summary['active_gene_types']['minimum']:.0f}–{gene_summary['active_gene_types']['maximum']:.0f}。这与每个空间位置最多 4 个激活是两个不同统计。

## 可视化选择规则

随机样例使用种子 20261115，在计算预测结果前从测试集固定抽取 8 张。另展示按基因模型 PSNR 排序的 P10/P50/P90 对应照片，明确为按性能选取。误差图是保存的 8-bit 预测与原图的绝对差乘 3，不是新语义标签。

![固定随机样例 1](unseen_random_first4.png)

![固定随机样例 2](unseen_random_next4.png)

![P10/P50/P90 样例](unseen_psnr_P10_P50_P90.png)

![指标分布](test_metric_distributions.png)

## 可复查文件

- summary.json：总体指标与分位数、检查点来源、原总体评价复现差异。
- *.per_image.csv / *.per_image.json：每张测试图的来源、指标及单图使用的基因数量。
- test_gene_codes.npz：1024 张图的实际激活索引、幅值、共享字典与测试 ID。可用该编码在特征空间加和并解码。
- display_samples.json：随机选择与按分位数选择的图片 ID、城市与逐图指标。
- samples/：展示样例的原图与两个模型重建。
- protocol.json / execution.json：本次诊断的范围、GPU、进程与选择规则。

模型权重和原实验结果保持原样。当前证据支持在新全景上进行编码与近似重建；可加瓶颈下细节保真仍弱于普通 AE。尚不足以认定变量独立或任意组合生成有效。
'''
    (output/'REPORT.md').write_text(report)
    write_json(output/'progress.json',{'status':'complete','test_images':len(rows),
        'variants':config['variants'],'inference_elapsed_seconds':round(time.monotonic()-started,2),'updated_utc':now()})
    print(json.dumps({'status':'complete','output':str(output),'summaries':summaries}),flush=True)


def supervise(config_path,output):
    from supervisor import acquire
    config=load_config(config_path)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'execution.json',{'status':'waiting_for_gpu','pid':os.getpid(),'max_gpus':1,'updated_utc':now()})
    while True:
        leases=acquire(config,1)
        if leases:
            break
        time.sleep(config['compute']['poll_seconds'])
    gpu,handle=leases[0]
    try:
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],PYTHONDONTWRITEBYTECODE='1',
            TMPDIR=config['temporary_directory'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
            PYTHONUNBUFFERED='1',MPLCONFIGDIR=str(Path(config['temporary_directory'])/'matplotlib'))
        command=[config['python'],str(Path(__file__).resolve()),'--config',str(config_path),
                 '--output',str(output),'--worker']
        with (output/'worker.log').open('w') as log:
            proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            write_json(output/'execution.json',{'status':'inference','pid':os.getpid(),
                'worker_pid':proc.pid,'physical_gpu':gpu,'updated_utc':now()})
            returncode=proc.wait()
        write_json(output/'execution.json',{'status':'complete' if returncode==0 else 'failed',
            'returncode':returncode,'physical_gpu':gpu,'updated_utc':now()})
        if returncode:
            raise SystemExit(returncode)
    finally:
        handle.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:
        run_worker(load_config(args.config),args.output)
    else:
        supervise(args.config,args.output)
