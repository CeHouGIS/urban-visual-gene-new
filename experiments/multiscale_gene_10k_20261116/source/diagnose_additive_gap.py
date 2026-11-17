"""Frozen GPU diagnostics for training-protocol and sparse spatial coverage gaps."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from common import now, write_json


def worker(config, original, comparison, output):
    import torch
    import lpips
    from data import open_pixels, image_tensor
    from metrics import ssim_per_image
    from compare_legacy_additive import load_model

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Requires one leased CUDA GPU; CPU inference is prohibited')
    torch.set_num_threads(4)
    rows = [json.loads(line) for line in (comparison / 'images.validation.common.jsonl').read_text().splitlines()]
    indices = [row['source_validation_row'] for row in rows]
    pixels = open_pixels(config, 'validation')
    source = original / 'source/model.py'
    spec = importlib.util.spec_from_file_location('original_dense_diagnostic', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old_config = json.loads((original / 'configs/experiment.json').read_text())
    checkpoint_path = original / 'runs/dense_ae/checkpoint_best.pt'
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = module.GeneAE('dense_ae', **old_config['model']).cuda()
    model.load_state_dict(state['model'], strict=True)
    model.eval().requires_grad_(False)
    perceptual = lpips.LPIPS(net='alex', verbose=False).cuda().eval().requires_grad_(False)
    totals = torch.zeros(5, device='cuda', dtype=torch.float64)
    use_bf16 = torch.cuda.is_bf16_supported()
    count = 0
    with torch.inference_mode():
        for start in range(0, len(indices), 32):
            target = image_tensor(pixels, indices[start:start + 32])
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                prediction = model(target)['reconstruction'].float()
            mse = (prediction - target).square().mean((1, 2, 3))
            values = [(prediction - target).abs().mean((1, 2, 3)), mse,
                      -10 * torch.log10(mse.clamp_min(1e-12)), ssim_per_image(prediction, target),
                      perceptual(prediction * 2 - 1, target * 2 - 1).flatten()]
            totals += torch.stack([v.double().sum() for v in values])
            count += len(target)
    if count != len(indices) or not torch.isfinite(totals).all():
        raise ValueError('Invalid evaluation totals')
    metrics = {name: float(totals[i] / count) for i, name in enumerate(
        ['l1', 'mse', 'psnr_mean_per_image', 'ssim', 'lpips'])}
    metrics.update(images=count, checkpoint=str(checkpoint_path), checkpoint_step=state['step'],
                   training_batch=old_config['training']['effective_batch_size'],
                   selection_loss=metrics['l1'] + .1 * metrics['lpips'],
                   inference_precision='BF16' if use_bf16 else 'FP32', evaluation_batch_size=32)
    write_json(output / 'original_dense_common_metrics.json', metrics)
    print(json.dumps({'original_dense_10k': metrics}), flush=True)
    del model, state, prediction, target, perceptual
    torch.cuda.empty_cache()

    all_stats = {}
    for variant in ['original_additive_10k', 'multiscale_vector', 'multiscale_conv', 'single_vector']:
        model, _ = load_model(Path(config['root']), original, variant)
        old = variant == 'original_additive_10k'
        scales = ('coarse',) if old else model.scales
        totals_by_scale, counts, masses = {}, {}, {}
        with torch.inference_mode():
            for start in range(0, len(indices), 32):
                target = image_tensor(pixels, indices[start:start + 32])
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                    encoded, activations = model.encode(target)
                activations = {'coarse': activations} if old else activations
                for scale in scales:
                    a = activations[scale].float()
                    mask = a > 0
                    occupied = mask.any(1)
                    if scale not in totals_by_scale:
                        totals_by_scale[scale] = torch.zeros(5, dtype=torch.float64, device='cuda')
                        counts[scale] = torch.zeros(a.shape[1], dtype=torch.int64, device='cuda')
                        masses[scale] = torch.zeros(a.shape[1], dtype=torch.float64, device='cuda')
                    totals_by_scale[scale] += torch.stack([
                        occupied.sum(), torch.tensor(occupied.numel(), device='cuda'), mask.sum(),
                        mask.any(3).any(2).sum(), torch.tensor(len(a), device='cuda')]).double()
                    counts[scale] += mask.sum((0, 2, 3))
                    masses[scale] += a.double().sum((0, 2, 3))
        stats = {}
        for scale, values in totals_by_scale.items():
            n = float(values[4])
            if n != len(indices):
                raise AssertionError('Sparse diagnostics omitted images')
            stats[scale] = {
                'mean_occupied_activation_positions_per_image': float(values[0] / n),
                'positions_per_image': int(values[1] / n),
                'occupied_activation_position_fraction': float(values[0] / values[1]),
                'mean_positive_coefficients_per_image': float(values[2] / n),
                'mean_distinct_genes_per_image': float(values[3] / n),
                'most_used_gene_by_count': int(counts[scale].argmax()),
                'largest_gene_fraction_of_activation_counts': float(counts[scale].max() / counts[scale].sum()),
                'strongest_gene_by_l1_coefficient_mass': int(masses[scale].argmax()),
                'largest_gene_fraction_of_l1_coefficient_mass': float(masses[scale].max() / masses[scale].sum()),
                'genes_used': int((counts[scale] > 0).sum()),
                'gene_count': len(counts[scale]),
            }
        all_stats[variant] = stats
        write_json(output / 'spatial_coverage.json', {
            'images': len(indices), 'models': all_stats,
            'note': 'Activation positions, not the spatial extent of synthesized features or RGB; convolution spreads to neighboring positions.'})
        print(json.dumps({variant: stats}), flush=True)
        del model, encoded, activations, a, mask, occupied, target
        torch.cuda.empty_cache()
    write_json(output / 'progress.json', {'status': 'complete', 'updated_utc': now()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--original', required=True)
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    original, comparison, output = Path(args.original), Path(args.comparison), Path(args.output)
    if args.worker:
        worker(config, original, comparison, output)
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
    write_json(output / 'progress.json', {'status': 'evaluating', 'updated_utc': now()})
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], PYTHONDONTWRITEBYTECODE='1',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               TMPDIR=config['temporary_directory'])
    command = [config['python'], str(Path(__file__).resolve()), '--config', args.config,
               '--original', str(original), '--comparison', str(comparison), '--output', str(output), '--worker']
    try:
        with (output / 'evaluation.log').open('w') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(output / 'progress.json', {'status': 'failed', 'returncode': result.returncode, 'updated_utc': now()})
            raise SystemExit(result.returncode)
    finally:
        lease.close()


if __name__ == '__main__':
    main()
