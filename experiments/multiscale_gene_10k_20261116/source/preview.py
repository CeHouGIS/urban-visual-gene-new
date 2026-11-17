"""Inspect feature addition and contextual RGB edits without claiming RGB linearity."""
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from common import write_json
from data import image_tensor


def to_pil(value):
    array = value.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.rint(array * 255).astype(np.uint8))


def sheet(rows, destination, side):
    canvas = Image.new('RGB', (side * len(rows[0]), (side + 24) * len(rows)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, tiles in enumerate(rows):
        for j, (label, value) in enumerate(tiles):
            draw.text((j * side + 3, i * (side + 24) + 3), label, fill='black')
            canvas.paste(to_pil(value), (j * side, i * (side + 24) + 24))
    canvas.save(destination)


@torch.inference_mode()
def preview(model, pixels, config, output, label):
    previous_mode = model.training
    model.eval()
    number = min(config['training']['preview_count'], len(pixels))
    target = image_tensor(pixels, np.arange(number))
    result = model(target)
    directory = output / 'previews'
    directory.mkdir(exist_ok=True)
    side = target.shape[-1]
    sheet([[('Original real street photo', target[i]), ('Reconstruction', result['reconstruction'][i]),
            ('Absolute error x3', (target[i] - result['reconstruction'][i]).abs() * 3)]
           for i in range(number)], directory / f'{label}.png', side)
    if model.sparse:
        serialized = {'note': np.array('Sparse spatial feature codes, not instance segmentation; RGB decoder is nonlinear')}
        for scale in model.scales:
            act = result['activations'][scale].float().flatten(1)
            values, indices = act.topk(min(model.budgets[scale], act.shape[1]), dim=1)
            serialized[f'{scale}_values'] = values.cpu().numpy()
            serialized[f'{scale}_indices'] = indices.cpu().numpy().astype(np.int32)
            serialized[f'{scale}_shape'] = np.array(result['activations'][scale].shape)
            serialized[f'{scale}_dictionary'] = model.dictionaries[scale].atoms.cpu().numpy()
        np.savez_compressed(directory / f'{label}.codes.npz', **serialized)
        rows, edits, info = [], [], []
        for i in range(min(number, 2)):
            row = [('Original', target[i]), ('Full reconstruction', result['reconstruction'][i])]
            edit = [('Original', target[i]), ('Full reconstruction', result['reconstruction'][i])]
            for scale in model.scales:
                features = {s: value[i:i + 1].clone() for s, value in result['features'].items()}
                features[scale].zero_()
                removed_scale = model.decoder(features)[0]
                row.append((f'Without {scale}', removed_scale))
                activation = result['activations'][scale][i:i + 1]
                gene = int(activation.sum((0, 2, 3)).argmax())
                activations = {s: value[i:i + 1] for s, value in result['activations'].items()}
                contribution = model.contribution(activations, scale, gene)
                features = {s: value[i:i + 1].clone() for s, value in result['features'].items()}
                features[scale] -= contribution
                removed = model.decoder(features)[0]
                edit.append((f'Without {scale} gene {gene}', removed))
                heat = F.interpolate(activation[:, gene:gene + 1], size=(side, side), mode='nearest')[0, 0]
                heat /= heat.max().clamp_min(1e-8)
                overlay = target[i] * 0.6 + torch.stack([heat, heat * 0.2, heat * 0.05]) * 0.4
                row.append((f'{scale} gene {gene} activation', overlay))
                info.append({'preview_row': i, 'scale': scale, 'gene': gene,
                    'contextual_gene_deletion_rgb_mae': float((removed - result['reconstruction'][i]).abs().mean()),
                    'scale_deletion_rgb_mae': float((removed_scale - result['reconstruction'][i]).abs().mean())})
            rows.append(row)
            edits.append(edit)
        sheet(rows, directory / f'{label}.scale_decomposition.png', side)
        sheet(edits, directory / f'{label}.gene_removal.png', side)
        write_json(directory / f'{label}.gene_removal.json', info)
        # Extract templates using a single positive spatial impulse. These are
        # signed FEATURE contributions, so they use a centered diverging map.
        atoms_rows = []
        for scale in model.scales:
            activation = result['activations'][scale]
            strongest = activation.sum((0, 2, 3)).topk(min(4, activation.shape[1])).indices.tolist()
            tiles = []
            for gene in strongest:
                impulse = torch.zeros_like(activation[:1])
                impulse[0, gene, impulse.shape[2] // 2, impulse.shape[3] // 2] = 1
                pattern = model.dictionaries[scale].contribution(impulse, gene)[0]
                # Show all latent channels as a 8x8 panel of kernel patches,
                # retaining a common symmetric color scale within this atom.
                radius = model.dictionaries[scale].kernel // 2
                cy, cx = pattern.shape[-2] // 2, pattern.shape[-1] // 2
                patch = pattern[:, cy-radius:cy+radius+1, cx-radius:cx+radius+1]
                k = patch.shape[-1]
                cols = 8
                grid = torch.zeros(((len(patch)+cols-1)//cols*k, cols*k), device=patch.device)
                for c, channel in enumerate(patch):
                    grid[c//cols*k:(c//cols+1)*k, c%cols*k:(c%cols+1)*k] = channel
                grid /= grid.abs().max().clamp_min(1e-8)
                rgb = torch.stack([grid.clamp_min(0), torch.zeros_like(grid), (-grid).clamp_min(0)])
                rgb = F.interpolate(rgb[None], size=(side, side), mode='nearest')[0]
                tiles.append((f'{scale} gene {gene}: +/- features', rgb))
            atoms_rows.append(tiles)
        sheet(atoms_rows, directory / f'{label}.feature_atoms.png', side)
    model.train(previous_mode)
