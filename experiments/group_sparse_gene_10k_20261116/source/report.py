"""CPU plots use measured active groups and real code bytes, not configured caps."""
import json
from pathlib import Path
from common import write_json

def records(path):
 p=Path(path)
 return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

def plot_run(output):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 train=records(output/'history.jsonl');val=records(output/'validation_history.jsonl')
 fig,axes=plt.subplots(1,4,figsize=(16,4))
 if train:axes[0].plot([x['step'] for x in train],[x['train_loss'] for x in train],label='Train total')
 if val:
  axes[0].plot([x['step'] for x in val],[x['selection_loss'] for x in val],label='Primary cap64 validation RGB objective')
  for rate in val[-1]['rates']:
   xs=[x['step'] for x in val];r=[x['rates'][rate] for x in val];label='Cap '+str(r[-1]['gene_cap'])
   axes[1].plot(xs,[x['psnr_mean_per_image'] for x in r],label=label)
   axes[2].plot(xs,[x['mean_active_genes'] for x in r],label=label)
   axes[3].plot(xs,[x['mean_encoded_bytes']/1024 for x in r],label=label)
 for ax,title in zip(axes,['Loss','Validation PSNR','Actual active genes / image','Actual encoded KiB / image']):ax.set(title=title,xlabel='Optimizer updates');ax.grid(alpha=.2);ax.legend(fontsize=7) if ax.lines else None
 fig.suptitle(output.name);fig.tight_layout();partial=output/'loss_curves.partial.png';fig.savefig(partial,dpi=140);plt.close(fig);partial.replace(output/'loss_curves.png')

def comparison(root,variants):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 root=Path(root);all_metrics={};curves=[]
 for variant in variants:
  path=root/f'runs/{variant}/city_confirmation_metrics.json'
  if not path.exists():continue
  result=json.loads(path.read_text());all_metrics[variant]=result
  for r in result['rates'].values():curves.append({'variant':variant,**r})
 write_json(root/'reports/comparison_metrics.json',all_metrics)
 fig,axes=plt.subplots(1,3,figsize=(14,4))
 for variant in variants:
  rows=[r for r in curves if r['variant']==variant]
  if not rows:continue
  for ax,xkey,ykey in [(axes[0],'mean_active_genes','psnr_mean_per_image'),(axes[1],'mean_encoded_bytes','psnr_mean_per_image'),(axes[2],'mean_active_genes','lpips')]:ax.plot([r[xkey] for r in rows],[r[ykey] for r in rows],'-o',label=variant)
 for ax,xlab,ylab in zip(axes,['Measured active genes/image','Measured encoded bytes/image','Measured active genes/image'],['City diagnostic PSNR','City diagnostic PSNR','City diagnostic LPIPS']):ax.set(xlabel=xlab,ylabel=ylab);ax.grid(alpha=.2);ax.legend(fontsize=7) if ax.lines else None
 fig.tight_layout();fig.savefig(root/'reports/gene_quality_budget_curves.png',dpi=150);plt.close(fig)
 lines=['# 跨城市基因复用与表达容量实验结果','','优先检验固定字典的有效跨城市复用；可组合按特征线性相加定义。主档为上限64，8/16/32是次要容量实验。城市诊断集已在先前实验查看，不作为新的盲测集。下表是重建与激活统计，最终有效复用审计需单独完成。','','|模型|上限|实际平均调用数|实际字节/图|PSNR|LPIPS|跨图使用过的基因数|','|---|---:|---:|---:|---:|---:|---:|']
 for r in curves:lines.append(f'|{r["variant"]}|{r["gene_cap"]}|{r["mean_active_genes"]:.2f}|{r["mean_encoded_bytes"]:.1f}|{r["psnr_mean_per_image"]:.2f}|{r["lpips"]:.4f}|{r["dataset_genes_used"]}|')
 lines+=['','![实际调用数与码量曲线](reports/gene_quality_budget_curves.png)','','判读：有效复用要求同一固定基因在多个城市上有可测量的重建贡献，仅有激活频率不足。固定随机字典也可以复用；learned/random比较回答学习字典是否有额外价值，不能用随机字典更好直接否定可复用。每图调用数属于次要效率指标，各尺度基因身份独立。两随机种子分别报告。主档逐基因删除审计将补充在本文件后。','']
 (root/'RESULTS.md').write_text('\n'.join(lines))
