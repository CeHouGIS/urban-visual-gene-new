"""Compare the original trained additive dictionary on a common unseen subset.

This is evaluation of frozen checkpoints, not a matched training ablation.
All model inference requires one cooperatively leased CUDA GPU.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from common import now, write_json


LABELS = {
    'original_additive_10k': 'Original additive (10k)',
    'multiscale_vector': 'Multiscale vector',
    'multiscale_conv': 'Multiscale convolutional',
    'single_vector': 'Single-scale, global TopK',
    'dense_ae': 'Dense AE',
}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def common_subset(root, original):
    rows = read_rows(root / 'data/images.validation.jsonl')
    old_train = read_rows(original / 'data/images.train.jsonl')
    new_train = read_rows(root / 'data/images.train.jsonl')
    old_panos = {row['pano'] for row in old_train}
    new_panos = {row['pano'] for row in new_train}
    excluded = [i for i, row in enumerate(rows) if row['pano'] in old_panos | new_panos]
    indices = [i for i, row in enumerate(rows) if row['pano'] not in old_panos | new_panos]
    if not indices or len({rows[i]['id'] for i in indices}) != len(indices):
        raise ValueError('Empty or duplicate evaluation manifest')
    return rows, indices, {
        'source_split': 'current experiment validation',
        'source_count': len(rows),
        'excluded_rows': excluded,
        'excluded_panoramas': sorted({rows[i]['pano'] for i in excluded}),
        'evaluation_count': len(indices),
        'training_panorama_overlap_after_exclusion': 0,
        'training_exact_image_overlap_between_experiments': len(
            {r['id'] for r in old_train} & {r['id'] for r in new_train}),
        'evaluation_continents': dict(Counter(rows[i]['continent'] for i in indices)),
        'scope': 'Known panorama isolation; not unseen-city or geographic-buffer evaluation',
        'source_file_hashes_computed': False,
    }


def load_model(root, original, variant):
    import torch
    if variant == 'original_additive_10k':
        source = original / 'source/model.py'
        spec = importlib.util.spec_from_file_location('frozen_original_additive_model', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = json.loads((original / 'configs/experiment.json').read_text())
        checkpoint = original / 'runs/sparse_n256_s4/checkpoint_best.pt'
        model = module.GeneAE('sparse_n256_s4', **config['model']).cuda()
    else:
        from model import create_model
        source = root / 'source/model.py'
        config = json.loads((root / 'configs/experiment.json').read_text())
        checkpoint = root / 'runs' / variant / 'checkpoint_best.pt'
        model = create_model(config, variant)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    model.eval().requires_grad_(False)
    settings = config['training']
    seen = state.get('images_seen', state['step'] * settings['effective_batch_size'])
    metadata = {
        'checkpoint': str(checkpoint), 'model_source': str(source),
        'checkpoint_step': state['step'], 'training_images_seen': seen,
        'training_unique_images': config['data']['train'],
        'effective_training_batch': settings['effective_batch_size'],
        'training_dataset_passes': seen / config['data']['train'],
        'sparsity': ('at most 4 positive coefficients per spatial position'
                     if variant == 'original_additive_10k' else
                     config['model']['sparsity'] if model.sparse else 'dense'),
        'trainable_parameters_in_original_model': sum(p.numel() for p in model.parameters()),
    }
    del state
    return model, metadata


def save_rgb(value, path):
    from PIL import Image
    value = value.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    Image.fromarray(np.rint(value * 255).astype(np.uint8)).save(path)


def evaluate(root, original, output, config):
    import torch
    import lpips
    from data import image_tensor, open_pixels
    from metrics import ssim_per_image

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('One leased visible GPU is required; no CPU inference')
    torch.set_num_threads(4)
    rows, indices, audit = common_subset(root, original)
    write_json(output / 'data_audit.json', audit)
    (output / 'images.validation.common.jsonl').write_text(''.join(
        json.dumps(dict(rows[i], source_validation_row=i), ensure_ascii=False) + '\n' for i in indices))
    pixels = open_pixels(config, 'validation')
    originals = output / 'originals'
    originals.mkdir()
    previews = indices[:4]
    for i in previews:
        save_rgb(image_tensor(pixels, [i])[0], originals / f'{i:04d}.png')
    perceptual = lpips.LPIPS(net='alex', verbose=False).cuda().eval().requires_grad_(False)
    use_bf16 = torch.cuda.is_bf16_supported()
    summaries = {}
    for variant in LABELS:
        model, metadata = load_model(root, original, variant)
        directory = output / variant
        directory.mkdir()
        totals = torch.zeros(5, device='cuda', dtype=torch.float64)
        counts, replay_errors, serialized = {}, {}, {}
        original_model = variant == 'original_additive_10k'
        scales = ('coarse',) if original_model else model.scales
        n = 0
        with torch.inference_mode():
            for begin in range(0, len(indices), 32):
                ids = indices[begin:begin + 32]
                target = image_tensor(pixels, ids)
                with torch.autocast('cuda', dtype=torch.bfloat16) if use_bf16 else nullcontext():
                    result = model(target)
                prediction = result['reconstruction'].float()
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError(f'Nonfinite reconstruction: {variant}')
                mse = (prediction - target).square().mean((1, 2, 3))
                values = [(prediction - target).abs().mean((1, 2, 3)), mse,
                          -10 * torch.log10(mse.clamp_min(1e-12)),
                          ssim_per_image(prediction, target),
                          perceptual(prediction * 2 - 1, target * 2 - 1).flatten()]
                totals += torch.stack([v.double().sum() for v in values])
                n += len(ids)
                if model.sparse:
                    acts = {'coarse': result['activations']} if original_model else result['activations']
                    for scale in scales:
                        a = acts[scale]
                        usage = (a > 0).sum((0, 2, 3))
                        counts[scale] = counts.get(scale, torch.zeros_like(usage)) + usage
                        if begin == 0:
                            first = a[:1].float()
                            left = first.clone()
                            left[:, 1::2] = 0
                            right = first - left
                            if original_model:
                                dictionary = lambda x: torch.einsum('bnhw,nc->bchw', x, model.dictionary)
                            else:
                                dictionary = model.dictionaries[scale]
                            full = dictionary(first)
                            error = dictionary(left) + dictionary(right) - full
                            replay_errors[scale] = float(error.norm() / full.norm().clamp_min(1e-8))
                            count = len(previews)
                            flat = a[:count].float().flatten(1)
                            budget = int((flat > 0).sum(1).max())
                            values_a, locations = flat.topk(budget, dim=1)
                            serialized[scale + '_values'] = values_a.cpu().numpy()
                            serialized[scale + '_indices'] = locations.cpu().numpy().astype(np.int32)
                            serialized[scale + '_shape'] = np.array(a[:count].shape)
                            atoms = model.dictionary if original_model else dictionary.atoms
                            serialized[scale + '_dictionary'] = atoms.detach().float().cpu().numpy()
                for j, i in enumerate(ids):
                    if i in previews:
                        save_rgb(prediction[j], directory / f'{i:04d}.png')
        if n != len(indices):
            raise AssertionError('Evaluation count mismatch')
        metrics = {name: float(totals[j] / n) for j, name in enumerate(
            ['l1', 'mse', 'psnr_mean_per_image', 'ssim', 'lpips'])}
        metrics.update(selection_loss=metrics['l1'] + 0.1 * metrics['lpips'],
                       psnr_from_mean_mse=-10 * math.log10(metrics['mse']),
                       images=n, inference_precision='BF16' if use_bf16 else 'FP32',
                       evaluation_batch_size=32, **metadata)
        if counts:
            metrics['scales'] = {}
            for scale, count in counts.items():
                sorted_counts = count.sort(descending=True).values
                total = int(count.sum())
                metrics['scales'][scale] = {
                    'genes_used': int((count > 0).sum()), 'gene_count': len(count),
                    'gene_activation_counts': count.cpu().tolist(),
                    'mean_nonzero_coefficients_per_image': total / n,
                    'most_used_gene_fraction_of_nonzero_coefficients': int(sorted_counts[0]) / max(total, 1),
                    'top10_gene_fraction_of_nonzero_coefficients': int(sorted_counts[:10].sum()) / max(total, 1),
                    'feature_additive_replay_relative_error_fp32': replay_errors[scale],
                }
            metrics['genes_used'] = sum(v['genes_used'] for v in metrics['scales'].values())
            metrics['mean_nonzero_coefficients_per_image'] = sum(
                v['mean_nonzero_coefficients_per_image'] for v in metrics['scales'].values())
            np.savez_compressed(directory / 'preview_gene_codes.npz',
                                source_validation_rows=np.array(previews), **serialized)
        summaries[variant] = metrics
        write_json(output / 'metrics.json', summaries)
        write_json(output / 'progress.json', {'status': 'evaluating', 'completed': list(summaries),
                   'latest': variant, 'updated_utc': now()})
        print(json.dumps({'variant': variant, **{k: metrics[k] for k in
              ['images', 'l1', 'psnr_mean_per_image', 'ssim', 'lpips']}}), flush=True)
        del model, result, prediction, target, counts, serialized
        torch.cuda.empty_cache()
    render_report(root, original, output, summaries, audit, previews)
    write_json(output / 'progress.json', {'status': 'complete', 'completed': list(summaries),
                                         'updated_utc': now()})


def render_errors(output, previews):
    """Render comparable RGB MAE maps from the saved 8-bit preview snapshots."""
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import colormaps

    variants = ['original_additive_10k', 'multiscale_vector', 'multiscale_conv']
    side, header, row_height, footer = 256, 72, 286, 94
    canvas = Image.new('RGB', (side * 7, header + row_height * len(previews) + footer), 'white')
    draw = ImageDraw.Draw(canvas)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 17)
    small = ImageFont.truetype(font_path, 14)
    titles = ['Ground truth']
    for variant in variants:
        titles.extend([LABELS[variant].replace(' convolutional', '\nconvolutional'), 'Absolute error vs GT'])
    for j, title in enumerate(titles):
        draw.multiline_text((j * side + 8, 14), title, font=font, fill='black')

    upper = 0.30
    cmap = colormaps['inferno']
    arrays, statistics = {}, []
    for i, row in enumerate(previews):
        y = header + i * row_height
        with Image.open(output / 'originals' / f'{row:04d}.png') as image:
            ground_truth = np.array(image.convert('RGB'), dtype=np.float32) / 255
        canvas.paste(Image.fromarray(np.rint(ground_truth * 255).astype(np.uint8)), (0, y))
        draw.text((8, y + side + 5), f'Validation image {row:04d}', font=small, fill='black')
        for j, variant in enumerate(variants):
            with Image.open(output / variant / f'{row:04d}.png') as image:
                prediction = np.array(image.convert('RGB'), dtype=np.float32) / 255
            if ground_truth.shape != prediction.shape or ground_truth.shape != (side, side, 3):
                raise ValueError('Preview RGB dimensions do not match')
            error = np.abs(prediction - ground_truth).mean(axis=2)
            heatmap = np.rint(cmap(np.clip(error / upper, 0, 1))[..., :3] * 255).astype(np.uint8)
            canvas.paste(Image.fromarray(np.rint(prediction * 255).astype(np.uint8)), ((1 + 2 * j) * side, y))
            canvas.paste(Image.fromarray(heatmap), ((2 + 2 * j) * side, y))
            mae = float(error.mean())
            draw.text(((2 + 2 * j) * side + 8, y + side + 5), f'MAE = {mae:.4f}', font=small, fill='black')
            arrays[f'{variant}_row_{row:04d}'] = error
            statistics.append({'variant': variant, 'source_validation_row': row,
                               'rgb_mae': mae, 'fraction_above_color_limit': float((error > upper).mean())})

    y = header + row_height * len(previews) + 9
    draw.text((24, y), 'Error = mean over RGB of |prediction - ground truth|; pixel values in [0, 1]',
              font=font, fill='black')
    x, width = 925, 820
    gradient = np.tile(np.linspace(0, 1, width, dtype=np.float32), (16, 1))
    bar = np.rint(cmap(gradient)[..., :3] * 255).astype(np.uint8)
    canvas.paste(Image.fromarray(bar), (x, y + 4))
    for tick in range(7):
        tx = x + round((width - 1) * tick / 6)
        draw.line((tx, y + 21, tx, y + 25), fill='black')
        label = f'{upper * tick / 6:.2f}' if tick < 6 else '>=0.30'
        draw.text((min(tx - 15, canvas.width - 62), y + 29), label, font=small, fill='black')
    draw.text((24, y + 35), 'Shared color scale for all images and models. Brighter means larger error.',
              font=font, fill='black')
    destination = output / 'original_additive_multiscale_with_errors.png'
    canvas.save(destination)
    np.savez_compressed(output / 'preview_absolute_error_maps.npz', **arrays)
    write_json(output / 'preview_error_statistics.json', {
        'definition': 'mean_RGB(abs(prediction - ground_truth)) at each pixel',
        'source': 'saved 8-bit RGB preview PNG snapshots; not the unquantized full-set metrics',
        'color_map': 'inferno', 'shared_vmin': 0, 'shared_vmax': upper,
        'values_above_vmax': 'saturated at the highest color; raw values retained in NPZ and MAE',
        'per_image_statistics': statistics,
    })
    return destination


def render_report(root, original, output, summaries, audit, previews):
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    side, header = 256, 56
    for filename, variants in [
        ('original_additive_multiscale_comparison.png', ['original_additive_10k', 'multiscale_vector', 'multiscale_conv']),
        ('all_models_comparison.png', list(LABELS)),
    ]:
        canvas = Image.new('RGB', (side * (1 + len(variants)), header + side * len(previews)), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, name in enumerate(['Original photo'] + [LABELS[v] for v in variants]):
            draw.multiline_text((j * side + 6, 8), name.replace(' convolutional', '\nconvolutional'), font=font, fill='black')
        for i, row in enumerate(previews):
            folders = ['originals'] + variants
            for j, folder in enumerate(folders):
                with Image.open(output / folder / f'{row:04d}.png') as im:
                    canvas.paste(im, (j * side, header + i * side))
        canvas.save(output / filename)

    render_errors(output, previews)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.7))
    for ax, (variant, metrics) in zip(axes, [(k, v) for k, v in summaries.items() if 'scales' in v]):
        for scale, data in metrics['scales'].items():
            y = np.sort(data['gene_activation_counts'])[::-1].astype(float)
            ax.plot(np.arange(1, len(y) + 1), y / max(y.sum(), 1), label=scale)
        ax.set(title=LABELS[variant], xlabel='Gene rank (sorted within scale)', ylabel='Fraction of nonzero coefficients')
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'gene_usage_comparison.png', dpi=150)
    plt.close(fig)

    lines = [
        '# 最早加性视觉基因与多尺度模型的共同评价', '',
        '日期：2026-11-16（北京时间）。所有模型均使用冻结的验证最优检查点，未进行额外训练。', '',
        f'共同评价集合：当前验证集 {audit["source_count"]} 张中，剔除旧训练全景对应的 '
        f'{len(audit["excluded_rows"])} 张，剩下 {audit["evaluation_count"]} 张。对各模型的训练全景均无重叠。', '',
        '同一批 RGB 缓存、256×256 输入、batch 32、相同 BF16 设置、相同 PSNR/SSIM/LPIPS 实现。'
        '预览固定为共同清单的前四张，是同一全景的四个朝向，未按重建效果筛选。', '',
        '| 模型 | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 使用基因数 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for variant, m in summaries.items():
        lines.append(f'| {LABELS[variant]} | {m["l1"]:.5f} | {m["psnr_mean_per_image"]:.2f} | '
                     f'{m["ssim"]:.4f} | {m["lpips"]:.4f} | {m.get("genes_used", "—")} |')
    lines += ['', '![原图、最早加性模型、多尺度向量、多尺度卷积](original_additive_multiscale_comparison.png)', '',
              '![与原图的绝对误差](original_additive_multiscale_with_errors.png)', '',
              '误差热图为每个像素的 RGB 通道平均绝对误差，所有图片和模型统一采用 0–0.30 色标，'
              '超过 0.30 的值用最高亮度显示。图中 MAE 使用未截断的误差。'
              '热图来自已保存的8位PNG预览，完整评价集的浮点指标仍见 `metrics.json`。', '',
              '![全部模型](all_models_comparison.png)', '',
              '## 这些比较能说明什么', '',
              '最早模型对应 `additive_gene_10k_20261115/sparse_n256_s4`，256个64维向量、16×16空间系数，'
              '每位置至多4个非负激活；潜在特征由基因贡献相加后，经非线性RGB解码器重建。'
              '当前 `single_vector` 是每图全局TopK的重新训练对照，与最早模型的竞争规则不同。', '',
              '这是历史已训练模型的输出对照。旧版使用另一批10,000张图片、6,000次更新、有效batch64、'
              '384,000张次；当前各组使用共同的10,000张图片、977次更新、有效batch1024、1,000,000张次。'
              '训练图片、更新次数和batch均不同，因此不能把历史模型与新模型的差距直接归因于多尺度结构。'
              '当前四组之间的数据与样本曝光预算保持对齐；没有重新选择检查点，也没有重复评估测试集。', '',
              '## 基因使用', '',
              '“使用基因数”指在本次988张图片中至少一次正激活的字典条目数量，'
              '不是每张图片的数量，也不是整个训练历史的累计数量。分尺度激活集中程度见下图。', '',
              '![各尺度内基因激活频率](gene_usage_comparison.png)', '',
              '`metrics.json` 包含逐尺度频数、特征相加回放误差和实际权重路径。'
              '各模型目录的 `preview_gene_codes.npz` 保存相同四张图片的真实稀疏系数与字典。'
              '不同模型的基因编号独立，不能将相同编号理解为同一成分。'
              '特征的可加性与RGB重建均不构成跨图像任意组合生成已有效的证明。', '',
              '## 位置', '', f'- 当前实验：`{root}`', f'- 最早加性实验：`{original}`',
              f'- 可复现脚本：`{root / "source/compare_legacy_additive.py"}`', '']
    (output / 'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--original-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--render-errors-only', action='store_true',
                        help='Plot saved preview errors on CPU without running models again')
    args = parser.parse_args()
    root = Path(json.loads(Path(args.config).read_text())['root'])
    config = json.loads(Path(args.config).read_text())
    original, output = Path(args.original_root), Path(args.output)
    if args.render_errors_only:
        manifest = read_rows(output / 'images.validation.common.jsonl')
        previews = [row['source_validation_row'] for row in manifest[:4]]
        destination = render_errors(output, previews)
        report = output / 'README.md'
        link = '![与原图的绝对误差](original_additive_multiscale_with_errors.png)'
        if report.exists() and link not in report.read_text():
            text = report.read_text()
            anchor = '![全部模型](all_models_comparison.png)'
            text = text.replace(anchor, link + '\n\n'
                '误差热图为每像素RGB通道平均绝对误差，统一色标0–0.30；超过0.30显示为最高亮度。'
                '图中MAE与保存的NPZ使用未截断误差。基于已保存的8位PNG预览计算，'
                '完整评价集的浮点指标仍见 `metrics.json`。\n\n' + anchor)
            report.write_text(text)
        print(destination)
        return
    if args.worker:
        evaluate(root, original, output, config)
        return
    from reference.supervisor import acquire
    output.mkdir(parents=True, exist_ok=False)
    while True:
        leases = acquire(config, 1)
        if leases:
            break
        write_json(output / 'progress.json', {'status': 'waiting_for_gpu', 'updated_utc': now()})
        time.sleep(10)
    gpu, lease = leases[0]
    write_json(output / 'gpu.json', {'physical_gpu': gpu, 'gpu_only': True, 'updated_utc': now()})
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TMPDIR=config['temporary_directory'],
               MPLCONFIGDIR=str(Path(config['temporary_directory']) / 'matplotlib'))
    command = [config['python'], str(Path(__file__).resolve()), '--config', args.config,
               '--original-root', str(original), '--output', str(output), '--worker']
    try:
        with (output / 'evaluation.log').open('w') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(output / 'progress.json', {'status': 'failed', 'returncode': result.returncode,
                                                 'updated_utc': now()})
            raise SystemExit(result.returncode)
    finally:
        lease.close()


if __name__ == '__main__':
    main()
