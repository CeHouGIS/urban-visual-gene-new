"""Show every normalized dictionary atom and contextual effects of three atoms."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from common import load_config, write_json
from reference.supervisor import acquire


SCALES = ('coarse', 'medium', 'fine')


def weight_montage(atom):
    """Tile 64 feature channels, retaining each channel's actual kernel geometry."""
    channels, kernel, width = atom.shape
    assert channels == 64 and kernel == width
    if kernel == 1:
        return atom[:, 0, 0].reshape(8, 8)
    result = np.full((8 * (kernel + 1) - 1, 8 * (kernel + 1) - 1), np.nan)
    for channel in range(64):
        row, column = divmod(channel, 8)
        result[row * (kernel + 1):row * (kernel + 1) + kernel,
               column * (kernel + 1):column * (kernel + 1) + kernel] = atom[channel]
    return result


def export_effects(root, destination, source, source_metadata, arrays):
    config = load_config(root / 'configs/experiment.json')
    leases = acquire(config, 1)
    if not leases:
        write_json(destination / 'status.json', {'status': 'waiting_for_gpu'})
        raise RuntimeError('No idle leased GPU; CPU inference disabled.')
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
        assert saved['step'] == source_metadata['checkpoint_step'] == 6000
        model = create_model(config, 'multiscale_residual_ista')
        model.load_state_dict(saved['model'], strict=True)
        model.set_step(saved['step'])
        model.eval().requires_grad_(False)
        for key, value in model.state_dict().items():
            assert torch.equal(value.cpu(), saved['model'][key]), key
        expressions = {s: torch.from_numpy(arrays[f'{s}_expressions'][:1]).cuda() for s in SCALES}
        for s in SCALES:
            np.testing.assert_array_equal(model.dictionaries[s].atoms.detach().cpu().numpy(), arrays[f'{s}_dictionary'])
        exported = {'original': arrays['originals'][0]}
        measurements = {}
        with torch.inference_mode():
            features = {s: model.dictionaries[s](expressions[s]) * model.gates[s] for s in SCALES}
            full = model.decoder(features)
            full_array = full[0].cpu().numpy()
            replay_error = float(np.abs(full_array - arrays['reconstructions'][0]).max())
            assert replay_error < 1e-5, replay_error
            exported['full_reconstruction'] = full_array
            for scale in SCALES:
                gene = source_metadata['summary'][scale]['selected_gene_id']
                masked = expressions[scale].clone()
                masked[:, gene] = 0
                deleted_features = dict(features)
                deleted_features[scale] = model.dictionaries[scale](masked) * model.gates[scale]
                deleted = model.decoder(deleted_features)
                delta = full - deleted
                exported[f'{scale}_deleted_reconstruction'] = deleted[0].cpu().numpy()
                exported[f'{scale}_rgb_delta'] = delta[0].cpu().numpy()
                image = torch.from_numpy(arrays['originals'][:1]).cuda()
                measurements[scale] = {'gene_id': gene, 'atom_shape': list(arrays[f'{scale}_dictionary'][gene].shape),
                                       'expression_shape': list(expressions[scale].shape[2:]),
                                       'rgb_change_mae': float(delta.abs().mean()),
                                       'full_reconstruction_mae': float((full - image).abs().mean()),
                                       'deleted_reconstruction_mae': float((deleted - image).abs().mean())}
        np.savez_compressed(destination / 'contextual_gene_effects.npz', **exported)
        return exported, {'checkpoint': str(checkpoint), 'checkpoint_step': saved['step'],
                           'all_state_tensors_equal_checkpoint': True, 'all_dictionary_atoms_equal_saved_export': True,
                           'physical_gpu': gpu, 'precision': 'FP32, no autocast, TF32 disabled',
                           'saved_full_reconstruction_max_abs_replay_error': replay_error,
                           'effect_measurements': measurements,
                           'source_arrays': str(source / 'five_street_expression_arrays.npz'),
                           'example': source_metadata['selected_images'][0],
                           'intervention': 'Set one saved expression map to zero, leave all other expressions and all model weights fixed, resynthesize features and decode; no re-encoding.'}
    finally:
        handle.close()


def render_atlases(arrays, destination):
    import matplotlib
    from matplotlib import colormaps
    from PIL import Image, ImageDraw, ImageFont
    matplotlib.use('Agg')
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    small = ImageFont.truetype(font_path, 12)
    title_font = ImageFont.truetype(font_path, 21)
    font = ImageFont.truetype(font_path, 14)
    cmap = colormaps['RdBu_r'].copy()
    cmap.set_bad('#dedede')
    pdf_pages = []
    limits = {}
    for scale in SCALES:
        atoms = arrays[f'{scale}_dictionary']
        limit = float(np.abs(atoms).max())
        limits[scale] = limit
        columns = 16
        cell_width, cell_height = 90, 104
        rows = (len(atoms) + columns - 1) // columns
        canvas = Image.new('RGB', (columns * cell_width + 24, rows * cell_height + 155), 'white')
        draw = ImageDraw.Draw(canvas)
        kernel = atoms.shape[-1]
        draw.text((12, 10), f'{scale}: all {len(atoms)} shared dictionary genes | checkpoint 6000', font=title_font, fill='#111827')
        draw.text((12, 44), f'One tile = one gene with 64 feature channels, {kernel}x{kernel} weights per channel.', font=font, fill='#374151')
        draw.text((12, 67), 'Channels 0-63 arranged in an 8x8 display grid. This arrangement is not image-space geometry.', font=font, fill='#374151')
        draw.text((12, 90), f'Red = positive, blue = negative; white = zero; common weight range [{-limit:.4f}, {limit:.4f}].', font=font, fill='#374151')
        for gene, atom in enumerate(atoms):
            tile = weight_montage(atom)
            rgba = cmap(np.ma.masked_invalid((tile / limit + 1) / 2))
            rgb = np.rint(rgba[:, :, :3] * 255).astype(np.uint8)
            image = Image.fromarray(rgb).resize((78, 78), Image.Resampling.NEAREST)
            row, column = divmod(gene, columns)
            x, y = 12 + column * cell_width, 125 + row * cell_height
            canvas.paste(image, (x, y + 18))
            draw.text((x, y), f'gene {gene:03d}', font=small, fill='#111827')
        canvas.save(destination / f'{scale}_all_dictionary_genes.png')
        pdf_pages.append(canvas)
    pdf_pages[0].save(destination / 'all_512_dictionary_genes.pdf', save_all=True, append_images=pdf_pages[1:], resolution=140)
    np.savez_compressed(destination / 'normalized_dictionary_weights.npz',
                        **{s: arrays[f'{s}_dictionary'] for s in SCALES})
    return limits


def render_explanation(arrays, source_metadata, effects, metadata, destination, limits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    cmap = colormaps['RdBu_r'].copy()
    cmap.set_bad('#dedede')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(13.2, 12.8))
    grid = fig.add_gridspec(4, 4, left=.095, right=.975, top=.895, bottom=.11, hspace=.40, wspace=.26,
                           height_ratios=[.82, 1, 1, 1])
    for column, (name, array) in enumerate([('Original: Sofia', effects['original']), ('Full reconstruction', effects['full_reconstruction'])]):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(array.transpose(1, 2, 0).clip(0, 1), interpolation='nearest')
        ax.set_title(name)
        ax.set_xticks([]); ax.set_yticks([])
    text_ax = fig.add_subplot(grid[0, 2:])
    text_ax.axis('off')
    text_ax.text(0, .9, 'Gene weights D: fixed across images\nExpression A: inferred for this image\nFeature contribution: D * A\n\nRGB change = full decode - decode without gene\nOther expressions stay fixed.', va='top', fontsize=11, linespacing=1.7)
    effect_limit = max(float(np.abs(effects[f'{s}_rgb_delta']).mean(0).max()) for s in SCALES)
    metadata['effect_heatmap_max'] = effect_limit
    for row, scale in enumerate(SCALES, 1):
        gene = source_metadata['summary'][scale]['selected_gene_id']
        atom = arrays[f'{scale}_dictionary'][gene]
        activation = arrays[f'{scale}_expressions'][0, gene]
        axes = [fig.add_subplot(grid[row, column]) for column in range(4)]
        weights = axes[0].imshow(weight_montage(atom), cmap=cmap, vmin=-limits[scale], vmax=limits[scale], interpolation='nearest')
        axes[0].set_ylabel(f'{scale}: gene {gene}', fontsize=11)
        expression = axes[1].imshow(activation, cmap='magma', vmin=0,
                                    vmax=source_metadata['summary'][scale]['display_max'], interpolation='nearest')
        axes[2].imshow(effects[f'{scale}_deleted_reconstruction'].transpose(1, 2, 0).clip(0, 1), interpolation='nearest')
        effect = axes[3].imshow(np.abs(effects[f'{scale}_rgb_delta']).mean(0), cmap='inferno',
                                vmin=0, vmax=effect_limit, interpolation='nearest')
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        for ax, image in [(axes[0], weights), (axes[1], expression), (axes[3], effect)]:
            colorbar = fig.colorbar(image, ax=ax, orientation='horizontal', fraction=.045, pad=.035, aspect=27)
            colorbar.ax.tick_params(labelsize=7)
        axes[0].set_title(f'Gene weights: 64 x {atom.shape[-1]} x {atom.shape[-1]}', fontsize=10)
        axes[1].set_title(f'Expression: {activation.shape[0]} x {activation.shape[1]}', fontsize=10)
        stats = metadata['effect_measurements'][scale]
        axes[2].set_title(f'Decode without this gene\nMAE {stats["deleted_reconstruction_mae"]:.5f}', fontsize=10)
        axes[3].set_title(f'Absolute RGB change\nMean {stats["rgb_change_mae"]:.5f}', fontsize=10)
    fig.suptitle('What a gene is, how it is expressed, and how it affects the image', fontsize=16, y=.975)
    fig.text(.095, .932, 'Original multiscale ISTA | checkpoint 6000 | frozen weights | test image 347', fontsize=10)
    fig.text(.095, .057, 'Weight panels tile 64 internal channels for display. Their red/blue colors are signed weights, not RGB appearance.', fontsize=9)
    fig.text(.095, .035, 'Expression brightness represents A. Deletion effects depend on the other genes and the nonlinear decoder.', fontsize=9)
    fig.text(.095, .015, 'The three genes are selected by highest mean expression over the five prior streets, not by object labels or deletion effects.', fontsize=8)
    fig.savefig(destination / 'gene_weights_expression_and_effect.png', dpi=170, facecolor='white')
    fig.savefig(destination / 'gene_weights_expression_and_effect.pdf', facecolor='white')
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'reports/five_street_expressions_20261117_062418'
    source_metadata = json.loads((source / 'metadata.json').read_text())
    with np.load(source / 'five_street_expression_arrays.npz') as archive:
        arrays = {k: archive[k] for k in archive.files}
    destination = root / 'reports' / ('gene_weights_and_effects_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    destination.mkdir(exist_ok=False)
    effects, metadata = export_effects(root, destination, source, source_metadata, arrays)
    limits = render_atlases(arrays, destination)
    render_explanation(arrays, source_metadata, effects, metadata, destination, limits)
    metadata.update({'status': 'complete', 'weights_updated': False, 'dictionary_gene_count': 512,
                     'dictionary_weight_display_limits': limits,
                     'display_note': 'All 64 channels shown for every gene. Channel montage is a display convention; dictionary atoms do not define fixed RGB objects.',
                     'updated_utc': datetime.now(timezone.utc).isoformat()})
    write_json(destination / 'metadata.json', metadata)
    write_json(destination / 'status.json', {'status': 'complete'})
    print(json.dumps({'destination': str(destination), 'effects': metadata['effect_measurements'],
                      'replay_max_abs_error': metadata['saved_full_reconstruction_max_abs_replay_error']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
