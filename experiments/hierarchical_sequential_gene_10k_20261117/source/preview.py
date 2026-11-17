"""Image-only reports; raw arrays and explicit residual display scales retained."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F


def rgb(v):
    return Image.fromarray((v.detach().float().cpu().permute(1, 2, 0).numpy().clip(0, 1)*255).round().astype('uint8'))


@torch.inference_mode()
def preview(model, x, path, last):
    path = Path(path)
    was_training = model.training
    model.eval()
    result = model(x, last=last)
    n, columns = len(x), last+4
    canvas = Image.new('RGB', (columns*256, n*286), 'white')
    draw = ImageDraw.Draw(canvas)
    for row in range(n):
        pics = [rgb(x[row])]
        names = ['Original']
        for i, output in enumerate(result['outputs']):
            pics.append(rgb(F.interpolate(output[row:row+1], size=256, mode='bilinear', align_corners=False)[0]))
            names.append(f'L0..L{i} ({output.shape[-1]}px)')
        pred = F.interpolate(result['outputs'][-1][row:row+1], size=256, mode='bilinear', align_corners=False)[0]
        error = (pred-x[row]).abs().mean(0).cpu().numpy()
        heat = np.stack([np.clip(error/0.2, 0, 1), np.zeros_like(error), np.zeros_like(error)], -1)
        pics.append(Image.fromarray((heat*255).astype('uint8')))
        names.append('Abs error: black=0 red>=.2')
        # Last-stage signed contribution, zero=gray, +/-0.25=black/white.
        delta = F.interpolate(result['corrections'][-1][row:row+1], size=256, mode='nearest')[0]
        pics.append(rgb(0.5 + delta/0.5))
        names.append('Last correction: gray=0')
        for col, (pic, name) in enumerate(zip(pics, names)):
            canvas.paste(pic, (col*256, row*286+26))
            draw.text((col*256+5, row*286+7), name, fill='black')
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    np.savez_compressed(path.with_suffix('.npz'), original=x.cpu().numpy(),
        **{f'output_l{i}': v.cpu().numpy() for i,v in enumerate(result['outputs'])},
        **{f'correction_l{i}': v.cpu().numpy() for i,v in enumerate(result['corrections'])},
        **{f'indices_l{i}': v.cpu().numpy() for i,v in enumerate(result['indices'])})
    model.train(was_training)


def plot_history(root):
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path = Path(root)/'history.jsonl'
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for phase in range(5):
        current = [r for r in rows if r.get('phase') == phase]
        train = [r for r in current if r['event'] == 'train']
        val = [r for r in current if r['event'] == 'validation']
        if train:
            axes[0].plot([r['step'] for r in train], [r['l1'] for r in train], label=f'phase {phase}')
        if val:
            axes[1].plot([r['step'] for r in val], [r['metrics']['selection_l1'] for r in val], label=f'phase {phase}')
    for ax, title in zip(axes, ['Train reconstruction MAE', 'Validation reconstruction MAE']):
        ax.set(title=title, xlabel='Updates within phase', ylabel='MAE')
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(Path(root)/'loss_curves.png', dpi=140)
    plt.close(fig)
