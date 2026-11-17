"""Plot all 512 genes' image counts, location counts and expression weights.

Uses the previously exported FP32 expression maps. CPU statistics and plotting only.
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCALES = ('coarse', 'medium', 'fine')
COLORS = {'coarse': '#c17b35', 'medium': '#287f96', 'fine': '#8261a8'}


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'reports/five_street_expressions_20261117_062418'
    source_metadata = json.loads((source / 'metadata.json').read_text())
    assert source_metadata['checkpoint_step'] == 6000
    assert source_metadata['checkpoint_verification']['all_state_tensors_equal_checkpoint']
    samples = source_metadata['selected_images']
    assert [s['test_index'] for s in samples] == [347, 384, 809, 613, 561]
    with np.load(source / 'five_street_expression_arrays.npz') as saved:
        originals = saved['originals']
        expressions = {scale: saved[f'{scale}_expressions'] for scale in SCALES}
    assert originals.shape == (5, 3, 256, 256)
    assert sum(a.shape[1] for a in expressions.values()) == 512

    destination = root / 'reports' / ('all_512_gene_responses_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    destination.mkdir(exist_ok=False)
    statistics = {}
    rows = []
    exported = {}
    global_offset = 0
    for scale, activation in expressions.items():
        assert np.isfinite(activation).all() and (activation >= 0).all()
        positive = activation > 0
        location_counts = positive.sum((2, 3))
        responded = location_counts > 0
        image_counts = responded.sum(0)
        mean_weights = activation.astype(np.float64).mean((2, 3))
        genes, height, width = activation.shape[1:]
        assert np.array_equal(responded.sum(1), source_metadata['summary'][scale]['distinct_active_genes_per_image'])
        assert np.all((image_counts >= 0) & (image_counts <= 5))
        assert np.all(location_counts <= height * width)
        assert np.array_equal(mean_weights == 0, location_counts == 0)
        statistics[scale] = {'candidate_genes': genes, 'expression_grid': [height, width],
                             'locations_per_gene_per_image': height * width,
                             'global_gene_offset': global_offset,
                             'response_count_histogram_0_to_5': np.bincount(image_counts, minlength=6).tolist(),
                             'distinct_genes_responded_in_any_image': int((image_counts > 0).sum()),
                             'distinct_genes_responded_in_all_five_images': int((image_counts == 5).sum()),
                             'per_image_active_gene_counts': responded.sum(1).tolist(),
                             'mean_weight_display_max': float(mean_weights.max())}
        exported[f'{scale}_responded_image_counts'] = image_counts
        exported[f'{scale}_positive_location_counts'] = location_counts
        exported[f'{scale}_mean_expression_weights'] = mean_weights
        for gene in range(genes):
            record = {'global_gene_id': global_offset + gene, 'scale': scale, 'scale_gene_id': gene,
                      'responded_image_count': int(image_counts[gene])}
            for i, sample in enumerate(samples):
                record[f'image_{i+1}_test_index'] = sample['test_index']
                record[f'image_{i+1}_positive_location_count'] = int(location_counts[i, gene])
                record[f'image_{i+1}_mean_expression_weight'] = float(mean_weights[i, gene])
            rows.append(record)
        global_offset += genes
    assert len(rows) == 512 and global_offset == 512
    totals = np.sum([statistics[s]['per_image_active_gene_counts'] for s in SCALES], axis=0)
    any_count = sum(statistics[s]['distinct_genes_responded_in_any_image'] for s in SCALES)
    all_count = sum(statistics[s]['distinct_genes_responded_in_all_five_images'] for s in SCALES)
    np.savez_compressed(destination / 'all_gene_response_statistics.npz', **exported)
    with (destination / 'all_512_gene_responses.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(19.5, 12.3))
    outer = fig.add_gridspec(2, 1, left=.085, right=.975, top=.875, bottom=.12,
                            height_ratios=[1, 3.3], hspace=.28)
    photos = outer[0].subgridspec(1, 5, wspace=.24)
    for i, sample in enumerate(samples):
        ax = fig.add_subplot(photos[0, i])
        ax.imshow(originals[i].transpose(1, 2, 0), interpolation='nearest')
        ax.set_title(f'#{i+1}  {sample["city"]}', fontsize=11)
        ax.set_xlabel(f'{totals[i]} / 512 genes active', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#d1d5db')
    columns = outer[1].subgridspec(1, 3, width_ratios=[2, 1, 1], wspace=.18)
    heatmap_row_labels = [f'#{i+1} {s["city"]}' for i, s in enumerate(samples)]
    for column, scale in enumerate(SCALES):
        details = statistics[scale]
        inner = columns[0, column].subgridspec(3, 2, width_ratios=[1, .035],
                                              height_ratios=[.85, 1.1, 1.1], hspace=.58, wspace=.07)
        genes = details['candidate_genes']
        counts = exported[f'{scale}_responded_image_counts']
        locations = exported[f'{scale}_positive_location_counts']
        weights = exported[f'{scale}_mean_expression_weights']
        ticks = np.unique(np.r_[np.arange(0, genes, 64 if genes == 256 else 32), genes - 1])
        ax_count = fig.add_subplot(inner[0, 0])
        ax_count.bar(np.arange(genes), counts, color=COLORS[scale], width=1, linewidth=0)
        ax_count.set_xlim(-.5, genes - .5)
        ax_count.set_ylim(0, 5.35)
        ax_count.set_yticks(range(6))
        ax_count.set_xticks(ticks)
        ax_count.tick_params(axis='x', labelsize=8)
        ax_count.grid(axis='y', color='#d1d5db', alpha=.55, linewidth=.6)
        ax_count.set_axisbelow(True)
        ax_count.set_title(f'{scale.capitalize()}: {genes} genes | {details["expression_grid"][0]} x {details["expression_grid"][1]} grid\n'
                           f'Number of images responding to each gene (0-5)', fontsize=11)
        if column == 0:
            ax_count.set_ylabel('Responding images')
        for row_index, (matrix, title, maximum, cmap) in enumerate([
            (locations, 'Responding locations in each image', details['locations_per_gene_per_image'], 'viridis'),
            (weights, 'Expression weight: mean A over all locations', details['mean_weight_display_max'], 'magma'),
        ], 1):
            ax = fig.add_subplot(inner[row_index, 0])
            image = ax.imshow(matrix, vmin=0, vmax=maximum, cmap=cmap, interpolation='nearest',
                              aspect='auto', extent=(-.5, genes - .5, 4.5, -.5))
            ax.set_title(title, fontsize=10)
            ax.set_xticks(ticks)
            ax.tick_params(axis='x', labelsize=8)
            ax.set_yticks(range(5))
            ax.set_yticklabels(heatmap_row_labels if column == 0 else [f'#{i+1}' for i in range(5)], fontsize=9)
            ax.set_yticks(np.arange(.5, 4.5, 1), minor=True)
            ax.grid(which='minor', axis='y', color='white', linewidth=.5, alpha=.5)
            ax.tick_params(which='minor', length=0)
            if row_index == 2:
                ax.set_xlabel(f'{scale.capitalize()} gene ID (0-{genes-1})', fontsize=10)
            cax = fig.add_subplot(inner[row_index, 1])
            colorbar = fig.colorbar(image, cax=cax)
            colorbar.ax.tick_params(labelsize=8)
        for spine in ['top', 'right']:
            ax_count.spines[spine].set_visible(False)
    fig.suptitle('All 512 genes: response counts and expression weights in five streets', fontsize=19, weight='bold', y=.98)
    fig.text(.085, .94, 'Same multiscale ISTA checkpoint 6000 | existing FP32 expression maps | frozen model', fontsize=11, color='#374151')
    fig.text(.085, .91, f'{any_count} / 512 genes respond in at least one image; {all_count} / 512 respond in all five. All genes shown in fixed ID order.', fontsize=11)
    fig.text(.085, .072, 'Response = A > 0. Location counts use the native feature grid, rather than RGB pixels. Weak positive values are included.', fontsize=10)
    fig.text(.085, .047, 'Weight = spatial mean of the inferred expression map A, including zeros. Dictionary weights D are fixed across the images.', fontsize=10)
    fig.text(.085, .022, 'Each scale has its own linear color bars. Use their numeric values when comparing scales; expression magnitude is not measured RGB contribution.', fontsize=10)
    fig.savefig(destination / 'all_512_gene_responses_and_weights.png', dpi=180, facecolor='white')
    fig.savefig(destination / 'all_512_gene_responses_and_weights.pdf', facecolor='white')
    plt.close(fig)

    metadata = {'status': 'complete', 'source_arrays': str(source / 'five_street_expression_arrays.npz'),
                'checkpoint': source_metadata['checkpoint'], 'checkpoint_step': source_metadata['checkpoint_step'],
                'computation': 'CPU statistics and plotting from existing verified FP32 model outputs; no new inference or training.',
                'selected_images': [{k:s[k] for k in ('city','country','test_index','id')} for s in samples],
                'total_genes': 512, 'responded_in_at_least_one_image': any_count, 'responded_in_all_five_images': all_count,
                'per_image_active_gene_counts': totals.tolist(), 'scales': statistics,
                'response_definition': 'A>0 at at least one spatial location in that image.',
                'location_response_definition': 'Number of spatial expression-grid locations with A>0, separately for every image and gene.',
                'weight_definition': 'Spatial mean of inferred expression A over all grid locations including zeros; not the shared dictionary D or causal RGB importance.',
                'ordering': 'Fixed scale-local IDs: coarse 0..255, medium 0..127, fine 0..127. CSV global IDs use offsets 0,256,384.',
                'updated_utc': datetime.now(timezone.utc).isoformat()}
    (destination / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'destination': str(destination), 'responded_in_any': any_count, 'responded_in_all_five': all_count,
                      'per_image_active_gene_counts': totals.tolist(), 'scale_histograms': {s:statistics[s]['response_count_histogram_0_to_5'] for s in SCALES}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
