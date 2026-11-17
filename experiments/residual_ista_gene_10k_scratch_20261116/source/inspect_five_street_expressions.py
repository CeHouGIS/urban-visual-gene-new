"""Export and visualize actual ISTA expression maps for the five displayed streets."""
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from common import load_config, write_json
from reference.supervisor import acquire


SCALES = ('coarse', 'medium', 'fine')


def infer(root, destination, selection):
    config = load_config(root / 'configs/experiment.json')
    leases = acquire(config, 1)
    if not leases:
        write_json(destination / 'status.json', {'status': 'waiting_for_gpu'})
        raise RuntimeError('No idle leased GPU; no CPU model fallback.')
    gpu, handle = leases[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
    try:
        import torch
        from model import create_model
        from data import image_tensor

        torch.set_num_threads(4)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        checkpoint = root / 'runs/multiscale_residual_ista/checkpoint_best.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['step'] == 6000
        model = create_model(config, 'multiscale_residual_ista')
        model.load_state_dict(saved['model'], strict=True)
        model.set_step(saved['step'])
        model.eval().requires_grad_(False)
        for key, value in model.state_dict().items():
            assert torch.equal(value.cpu(), saved['model'][key]), key
        pixels = np.load(Path(config['cache']) / 'test.rgb.npy', mmap_mode='r', allow_pickle=False)
        indices = np.array([row['test_index'] for row in selection])
        images = image_tensor(pixels, indices)
        with torch.inference_mode():
            outputs = model(images)
            known = image_tensor(pixels, np.array([199]))
            known_output = model(known)['reconstruction']
            known_psnr = float(-10 * torch.log10((known_output - known).square().mean().clamp_min(1e-12)))
        assert known_psnr > 25.
        arrays = {'originals': images.cpu().numpy(),
                  'reconstructions': outputs['reconstruction'].float().cpu().numpy(),
                  'test_indices': indices}
        for scale in SCALES:
            arrays[f'{scale}_expressions'] = outputs['activations'][scale].float().cpu().numpy()
            arrays[f'{scale}_dictionary'] = model.dictionaries[scale].atoms.detach().float().cpu().numpy()
            assert np.isfinite(arrays[f'{scale}_expressions']).all()
            assert (arrays[f'{scale}_expressions'] >= 0).all()
        np.savez_compressed(destination / 'five_street_expression_arrays.npz', **arrays)
        verification = {'all_state_tensors_equal_checkpoint': True, 'known_test_index': 199,
                        'known_reconstruction_psnr': known_psnr}
        return arrays, {'variant': 'multiscale_residual_ista', 'checkpoint': str(checkpoint),
                        'checkpoint_step': saved['step'], 'checkpoint_verification': verification,
                        'precision': 'FP32, no autocast, TF32 disabled', 'physical_gpu': gpu,
                        'ista_iterations': 6, 'sparse_penalty': model.penalty, 'hard_gene_cap': None,
                        'weights_updated': False, 'selected_images': selection}
    finally:
        handle.close()


def summarize(arrays, metadata, destination):
    rows = []
    gene_rows = []
    summary = {}
    for scale in SCALES:
        activation = arrays[f'{scale}_expressions']
        _, genes, height, width = activation.shape
        positive = activation > 0
        # Fixed gene identities across all images, selected by mean coefficient mass.
        gene_mass = activation.mean((0, 2, 3))
        ranking = np.argsort(-gene_mass, kind='stable')
        selected = int(ranking[0])
        sorted_mass = np.sort(activation, axis=1)[:, ::-1]
        total = sorted_mass.sum(1)
        cumulative = np.cumsum(sorted_mass, axis=1)
        n90 = (cumulative < .9 * total[:, None]).sum(1) + 1
        n90 = np.where(total > 0, n90, 0)
        for i, sample in enumerate(metadata['selected_images']):
            local_counts = positive[i].sum(0)
            record = {'city': sample['city'], 'test_index': sample['test_index'], 'scale': scale,
                      'candidate_genes': genes, 'expression_height': height, 'expression_width': width,
                      'distinct_active_genes': int(positive[i].any((1, 2)).sum()),
                      'mean_active_genes_per_location': float(local_counts.mean()),
                      'min_active_genes_per_location': int(local_counts.min()),
                      'max_active_genes_per_location': int(local_counts.max()),
                      'nonzero_coefficient_fraction': float(positive[i].mean()),
                      'mean_genes_for_90pct_coefficient_mass_per_location': float(n90[i].mean()),
                      'selected_display_gene_id': selected,
                      'selected_display_gene_mean_expression': float(activation[i, selected].mean())}
            rows.append(record)
            for gene in range(genes):
                gene_rows.append({'city': sample['city'], 'test_index': sample['test_index'],
                                  'scale': scale, 'gene_id': gene,
                                  'mean_expression': float(activation[i, gene].mean()),
                                  'max_expression': float(activation[i, gene].max()),
                                  'active_location_fraction': float(positive[i, gene].mean())})
        values = [r for r in rows if r['scale'] == scale]
        summary[scale] = {'candidate_genes': genes, 'expression_grid': [height, width],
                          'selected_gene_id': selected, 'top_8_by_mean_expression': ranking[:8].tolist(),
                          'display_max': float(activation[:, selected].max()),
                          'distinct_active_genes_per_image': [r['distinct_active_genes'] for r in values],
                          'mean_active_genes_per_location': [r['mean_active_genes_per_location'] for r in values],
                          'mean_genes_for_90pct_coefficient_mass_per_location': [r['mean_genes_for_90pct_coefficient_mass_per_location'] for r in values]}
    for filename, records in [('expression_statistics.csv', rows), ('per_gene_expression_statistics.csv', gene_rows)]:
        with (destination / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    metadata.update({'summary': summary, 'per_image_per_scale': rows,
                     'nonzero_definition': 'Expression coefficient > 0, no positive-amplitude cutoff.',
                     'n90_definition': 'Minimum number of largest nonnegative coefficients carrying 90% of coefficient L1 mass at each location; not RGB contribution.',
                     'display_gene_selection': 'One gene per scale ranked first by mean expression over all five images and all locations; same identity displayed for all images.',
                     'display_interpretation': 'Actual A_s,g values, not PCA, segmentation, or measured RGB importance. Gene IDs are zero based and scale specific.'})
    return metadata


def plot(arrays, metadata, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    samples = metadata['selected_images']
    for mode in ['shared_gene_expression', 'local_active_gene_counts']:
        fig, axes = plt.subplots(5, 4, figsize=(12, 14.6))
        fig.subplots_adjust(left=.13, right=.985, bottom=.085, top=.875, wspace=.10, hspace=.24)
        for i, sample in enumerate(samples):
            axes[i, 0].imshow(arrays['originals'][i].transpose(1, 2, 0), interpolation='nearest')
            axes[i, 0].set_ylabel(sample['city'] + '\n' + sample['country'], fontsize=10)
            for j, scale in enumerate(SCALES, 1):
                activation = arrays[f'{scale}_expressions']
                details = metadata['summary'][scale]
                if mode == 'shared_gene_expression':
                    gene = details['selected_gene_id']
                    value = activation[i, gene]
                    im = axes[i, j].imshow(value, cmap='magma', vmin=0, vmax=details['display_max'], interpolation='nearest')
                    axes[i, j].set_xlabel(f'Mean expression {value.mean():.4f}', fontsize=9)
                    if i == 0:
                        axes[i, j].set_title(f'{scale}: gene #{gene}\n{value.shape[0]} x {value.shape[1]} expression map', fontsize=10)
                else:
                    value = (activation[i] > 0).sum(0)
                    im = axes[i, j].imshow(value, cmap='viridis', vmin=0, vmax=details['candidate_genes'], interpolation='nearest')
                    distinct = details['distinct_active_genes_per_image'][i]
                    axes[i, j].set_xlabel(f'Mean/location {value.mean():.1f}; whole image {distinct}', fontsize=9)
                    if i == 0:
                        axes[i, j].set_title(f'{scale}: {details["candidate_genes"]} candidates\nActive genes at each grid location', fontsize=10)
                if i == 4:
                    colorbar = fig.colorbar(im, ax=list(axes[:, j]), orientation='horizontal',
                                           fraction=.015, pad=.04, aspect=30)
                    colorbar.ax.tick_params(labelsize=8)
            for ax in axes[i]:
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color('#d1d5db')
        axes[0, 0].set_title('Input street photo', fontsize=11)
        if mode == 'shared_gene_expression':
            title = 'The same dictionary gene, expressed in five street images'
            subtitle = 'One gene per scale, selected by largest mean coefficient; fixed color limits down each column.'
            footer = 'Brighter = stronger expression. These maps are feature coefficients, with no assigned object semantics.'
        else:
            title = 'How many genes are active at each feature-grid location?'
            subtitle = 'Active means A > 0; all genes counted, including weak positive coefficients.'
            footer = 'Mean/location counts simultaneous genes per expression location. Whole image counts distinct identities used anywhere.'
        fig.suptitle(title, fontsize=16, y=.975)
        fig.text(.13, .94, 'Original multiscale ISTA | checkpoint 6000 | 6 iterations | frozen weights', fontsize=10)
        fig.text(.13, .913, subtitle, fontsize=9)
        fig.text(.13, .027, footer, fontsize=8)
        fig.savefig(destination / f'{mode}.png', dpi=160, facecolor='white')
        fig.savefig(destination / f'{mode}.pdf', facecolor='white')
        plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'reports/random10_unseen_reconstruction_seed_20261116'
    selection_data = json.loads((source / 'selection.json').read_text())
    assert all(selection_data['isolation_checks'].values())
    selection = selection_data['samples'][:5]
    assert [s['test_index'] for s in selection] == [347, 384, 809, 613, 561]
    destination = root / 'reports' / ('five_street_expressions_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    destination.mkdir(exist_ok=False)
    arrays, metadata = infer(root, destination, selection)
    with np.load(source / 'reconstruction_float.npz') as original_run:
        assert np.array_equal(arrays['originals'], original_run['originals'][:5])
    metadata['source_selection'] = str(source / 'selection.json')
    metadata['same_input_pixels_as_displayed_reconstructions'] = True
    metadata = summarize(arrays, metadata, destination)
    plot(arrays, metadata, destination)
    metadata.update({'status': 'complete', 'updated_utc': datetime.now(timezone.utc).isoformat()})
    write_json(destination / 'metadata.json', metadata)
    write_json(destination / 'status.json', {'status': 'complete'})
    print(json.dumps({'destination': str(destination), 'summary': metadata['summary'],
                      'verification': metadata['checkpoint_verification']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
