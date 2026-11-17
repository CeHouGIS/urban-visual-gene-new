"""Summarize actual saved logs and samples without inventing completed stages."""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent

def main(args):
    path=ROOT/'runs'/args.run;logs=[json.loads(line) for line in (path/'train.jsonl').read_text().splitlines() if line.strip()]
    if not logs:raise RuntimeError('No completed optimizer updates')
    samples=[]
    for file in sorted(path.glob('samples_*/metrics.json')):
        report=json.loads(file.read_text());means={}
        for kind in ['correct','zero','shuffled']:
            entries=[r for r in report['metrics'] if r['kind']==kind]
            means[kind]={k:float(np.mean([x[k] for x in entries])) for k in ['lpips_alex','psnr','ssim'] if all(k in x for x in entries)}
        samples.append({'step':report['step'],'means':means,'panel':str(file.parent/'panel.png')})
    result={'last_completed_update':logs[-1]['step'],'complete_sample_steps':[r['step'] for r in samples],
        'last20_mean_seconds_per_update':float(np.mean([r['seconds'] for r in logs[-20:]])),
        'max_training_memory_GB':max(r['peak_allocated_GB'] for r in logs),
        'latest20_mean_training_loss':float(np.mean([r['loss'] for r in logs[-20:]])),
        'samples':samples,'human_ratings':'not collected','quality_gate':'requires visual review and larger validation; not inferred from loss'}
    (path/'summary.json').write_text(json.dumps(result,indent=2))
    # Static, directly auditable plot using the actual numeric logs.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.5))
    steps=np.array([r['step'] for r in logs]);loss=np.array([r['loss'] for r in logs])
    axes[0].plot(steps,loss,alpha=.3,label='per optimizer update')
    if len(logs)>=10:axes[0].plot(steps[9:],np.convolve(loss,np.ones(10)/10,'valid'),label='10-update mean')
    axes[0].set(title='Training denoising loss',xlabel='Optimizer updates',ylabel='MSE');axes[0].legend(fontsize=8)
    for kind in ['correct','zero','shuffled']:
        present=[s for s in samples if 'lpips_alex' in s['means'][kind]]
        axes[1].plot([s['step'] for s in present],[s['means'][kind]['lpips_alex'] for s in present],marker='o',label=kind)
    axes[1].set(title='Full sampling: LPIPS (4 diagnostic inputs)',xlabel='Optimizer updates',ylabel='LPIPS, lower is closer');axes[1].legend(fontsize=8)
    axes[2].plot(steps,[r['seconds'] for r in logs]);axes[2].set(title='Time per optimizer update',xlabel='Optimizer updates',ylabel='Seconds')
    fig.suptitle(args.run+' | diagnostic, not final performance',fontsize=11);fig.tight_layout();fig.savefig(path/'training_curves.png',dpi=150);plt.close(fig)
    if samples:
        chosen=[samples[0],samples[-1]] if len(samples)>1 else samples
        panels=[Image.open(s['panel']).convert('RGB') for s in chosen]
        canvas=Image.new('RGB',(sum(im.width for im in panels),max(im.height for im in panels)+30),'white');draw=ImageDraw.Draw(canvas);offset=0
        for sample,panel in zip(chosen,panels):
            draw.text((offset+10,8),f"Step {sample['step']} | target / correct / zero / shuffled",fill='black');canvas.paste(panel,(offset,30));offset+=panel.width
        canvas.save(path/'before_after.png')
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='smoke1024_DV_seed17');main(p.parse_args())
