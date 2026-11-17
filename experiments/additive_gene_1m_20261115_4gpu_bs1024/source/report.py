"""Plots stored logs; no model inference or GPU allocation."""
import json
from pathlib import Path


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def plot_run(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    train=records(output/'history.jsonl')
    validation=records(output/'validation_history.jsonl')
    fig,axes=plt.subplots(1,3,figsize=(13,3.7))
    if train:
        axes[0].plot([r['step'] for r in train],[r['train_loss'] for r in train],label='Train')
    if validation:
        steps=[r['step'] for r in validation]
        axes[0].plot(steps,[r['selection_loss'] for r in validation],label='Validation')
        axes[1].plot(steps,[r['psnr_mean_per_image'] for r in validation])
        axes[2].plot(steps,[r['lpips'] for r in validation])
    for ax,title in zip(axes,['L1 + 0.1 LPIPS','Validation PSNR (dB)','Validation LPIPS']):
        ax.set(title=title,xlabel='Optimizer update')
        ax.grid(alpha=0.2)
    if train:
        axes[0].legend()
    fig.suptitle(output.name)
    fig.tight_layout()
    temporary=output/'loss_curves.partial.png'
    fig.savefig(temporary,dpi=140)
    plt.close(fig)
    temporary.replace(output/'loss_curves.png')


def comparison(root,variants):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw
    summaries={}
    fig,axes=plt.subplots(1,3,figsize=(13,3.7))
    panels=[]
    for variant in variants:
        output=root/'runs'/variant
        data=records(output/'validation_history.jsonl')
        if data:
            steps=[r['step'] for r in data]
            for ax,key in zip(axes,['selection_loss','psnr_mean_per_image','lpips']):
                ax.plot(steps,[r[key] for r in data],label=variant)
        result=output/'test_metrics.json'
        if result.exists():
            summaries[variant]=json.loads(result.read_text())
        preview=output/'previews/test_best.png'
        if preview.exists():
            panels.append((variant,Image.open(preview).convert('RGB')))
    for ax,title in zip(axes,['Validation objective','Validation PSNR (dB)','Validation LPIPS']):
        ax.set(title=title,xlabel='Optimizer update')
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(root/'reports/comparison_curves.png',dpi=140)
    plt.close(fig)
    (root/'reports/comparison_metrics.json').write_text(json.dumps(summaries,indent=2)+'\n')
    if panels:
        width=sum(im.width for _,im in panels)
        height=max(im.height for _,im in panels)+24
        canvas=Image.new('RGB',(width,height),'white')
        painter=ImageDraw.Draw(canvas)
        x=0
        for name,im in panels:
            painter.text((x+4,4),name,fill='black')
            canvas.paste(im,(x,24))
            x+=im.width
        canvas.save(root/'reports/test_reconstruction_comparison.png')
