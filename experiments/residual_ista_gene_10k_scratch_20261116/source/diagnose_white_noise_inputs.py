"""Frozen, checkpoint-verified ISTA probes with white and fixed-seed RGB noise inputs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from common import load_config, write_json
from reference.supervisor import acquire


def main(seed):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / 'configs/experiment.json')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    destination = root / 'reports' / f'white_noise_input_probes_{stamp}'
    destination.mkdir(parents=True, exist_ok=False)
    leases = acquire(config, 1)
    if not leases:
        write_json(destination / 'status.json', {'status': 'waiting_for_gpu'})
        raise RuntimeError('No idle leased GPU; CPU inference is disabled.')
    gpu, handle = leases[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
    try:
        import torch
        from model import create_model

        torch.set_num_threads(4)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        checkpoint = root / 'runs/multiscale_residual_ista/checkpoint_best.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['step'] == 6000, 'Expected the original step-6000 checkpoint.'
        model = create_model(config, 'multiscale_residual_ista')
        model.load_state_dict(saved['model'], strict=True)
        model.set_step(saved['step'])
        model.eval().requires_grad_(False)
        for key, value in model.state_dict().items():
            assert torch.equal(value.cpu(), saved['model'][key]), key

        # Confirm the complete loaded model reproduces a known held-out street.
        cache = np.load(Path(config['cache']) / 'test.rgb.npy', mmap_mode='r')
        known_pixels = np.array(cache[199:200], copy=True)
        assert known_pixels.shape == (1, 256, 256, 3), known_pixels.shape
        known_image = torch.from_numpy(known_pixels).permute(0, 3, 1, 2).to('cuda', dtype=torch.float32) / 255.
        with torch.inference_mode():
            known_output = model(known_image)['reconstruction']
            known_psnr = float(-10 * torch.log10((known_output - known_image).square().mean().clamp_min(1e-12)))
        assert known_psnr > 25., f'Known-image checkpoint sanity check failed: {known_psnr}'

        rng = np.random.default_rng(seed)
        noise = rng.integers(0, 256, size=(1, 3, 256, 256), dtype=np.uint8).astype(np.float32) / 255.
        inputs = np.concatenate([np.ones_like(noise), noise], axis=0)
        image = torch.from_numpy(inputs).cuda()
        with torch.inference_mode():
            result = model(image)
        predictions = result['reconstruction'].float().cpu().numpy()
        assert predictions.shape == inputs.shape and np.isfinite(predictions).all()
        expression_maps = {s: a.float().cpu().numpy() for s, a in result['activations'].items()}
        verification = {'all_state_tensors_equal_checkpoint': True, 'checkpoint_step': saved['step'],
                        'known_test_index': 199, 'known_reconstruction_psnr': known_psnr}
        model_settings = {'ista_iterations': model.settings['ista_iterations'],
                          'sparse_penalty': model.penalty, 'scale_gates': model.gates,
                          'gene_counts': model.gene_counts, 'hard_gene_cap': None}
    finally:
        handle.close()

    # CPU operations below only store arrays and draw figures.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    measurements = []
    keys = ['white', 'random_noise']
    for index, key in enumerate(keys):
        delta = predictions[index].astype(np.float64) - inputs[index].astype(np.float64)
        mse = float(np.square(delta).mean())
        stats = {'input_kind': key, 'mae': float(np.abs(delta).mean()), 'mse': mse,
                 'psnr': float(-10 * np.log10(max(mse, 1e-12))),
                 'input_min': float(inputs[index].min()), 'input_max': float(inputs[index].max()),
                 'output_min': float(predictions[index].min()), 'output_max': float(predictions[index].max()),
                 'output_mean_rgb': predictions[index].mean((1, 2)).tolist(),
                 'output_spatial_std_rgb': predictions[index].std((1, 2)).tolist(), 'scales': {}}
        for scale, expression in expression_maps.items():
            activation = expression[index]
            positive = activation > 0
            stats['scales'][scale] = {
                'expression_shape': list(activation.shape),
                'distinct_active_genes': int(positive.any((1, 2)).sum()),
                'mean_active_genes_per_expression_location': float(positive.sum(0).mean()),
                'nonzero_coefficient_fraction': float(positive.mean()),
                'max_expression': float(activation.max())}
        stats['distinct_active_genes'] = sum(s['distinct_active_genes'] for s in stats['scales'].values())
        measurements.append(stats)
        for label, array in [('input', inputs[index]), ('output', predictions[index])]:
            pixels = np.rint(array.transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
            Image.fromarray(pixels).save(destination / f'{key}_{label}.png')
    np.savez_compressed(destination / 'inputs_outputs_and_codes.npz', inputs=inputs, outputs=predictions,
                        **{f'{s}_expressions': a for s, a in expression_maps.items()})

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 8.2))
    fig.subplots_adjust(left=.025, right=.89, bottom=.12, top=.86, hspace=.30, wspace=.07)
    error = np.abs(predictions - inputs).mean(1)
    input_titles = ['Input: all RGB values = 1', f'Input: random RGB noise (seed {seed})']
    for index, stats in enumerate(measurements):
        axes[index, 0].imshow(inputs[index].transpose(1, 2, 0), interpolation='nearest')
        axes[index, 1].imshow(predictions[index].transpose(1, 2, 0).clip(0, 1), interpolation='nearest')
        heat = axes[index, 2].imshow(error[index], vmin=0, vmax=.5, cmap='inferno', interpolation='nearest')
        axes[index, 0].set_title(input_titles[index], fontsize=11)
        axes[index, 1].set_title('Output: multiscale ISTA AE', fontsize=11)
        axes[index, 1].set_xlabel(f'MAE {stats["mae"]:.5f} | PSNR {stats["psnr"]:.2f} dB\n'
                                  f'{stats["distinct_active_genes"]} distinct genes across image', fontsize=10)
        axes[index, 2].set_title('Absolute RGB error', fontsize=11)
        for ax in axes[index]:
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#9ca3af')
    colorbar = fig.colorbar(heat, cax=fig.add_axes([.925, .27, .016, .44]))
    colorbar.set_label('Mean absolute RGB error per pixel')
    fig.suptitle('White image and random noise through the original ISTA model', fontsize=15, y=.965)
    fig.text(.025, .905, 'Step 6000 | 6 ISTA iterations | frozen full model | FP32 | no hard gene-count cap', fontsize=10)
    fig.text(.025, .045, 'Inputs: 256 x 256 RGB in [0,1]. Noise: independent uniform uint8 values / 255. Error scale: 0-0.5.', fontsize=9)
    fig.text(.025, .025, 'These are input probes outside the street-scene training distribution; no model weights or expression codes were edited.', fontsize=8)
    fig.savefig(destination / 'ista_white_noise_input_output_error.png', dpi=180, facecolor='white')
    fig.savefig(destination / 'ista_white_noise_input_output_error.pdf', facecolor='white')
    plt.close(fig)
    metadata = {'status': 'complete', 'variant': 'multiscale_residual_ista', 'checkpoint': str(checkpoint),
                'checkpoint_step': saved['step'], 'precision': 'float32, no autocast, TF32 disabled',
                'physical_gpu': gpu, 'checkpoint_verification': verification, 'settings': model_settings,
                'input_shape_bchw': list(inputs.shape), 'noise_seed': seed,
                'noise_definition': 'NumPy default_rng(seed), independent uniform uint8 RGB [0,255] divided by 255',
                'intervention': 'Complete Encoder -> six ISTA steps -> dictionary synthesis -> Decoder; no latent edits or retraining',
                'measurements': measurements, 'updated_utc': datetime.now(timezone.utc).isoformat()}
    write_json(destination / 'metadata.json', metadata)
    write_json(destination / 'status.json', {'status': 'complete'})
    print(json.dumps({'destination': str(destination), 'checkpoint_verification': verification,
                      'measurements': measurements}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args.seed)
