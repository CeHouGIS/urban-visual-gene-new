"""Post-training GPU audit of actual code bytes, repeated use, and gene deletions."""
import os
import json
from pathlib import Path
import time
import numpy as np
from common import write_json,now
from reference.supervisor import acquire

def evaluate_all(config,stop_requested=None):
 root=Path(config['root']);status=root/'reports/final_audit_status.json'
 def stopped():return (stop_requested is not None and stop_requested()) or (root/'STOP_REQUESTED').exists()
 leases=[]
 while not leases:
  if stopped():write_json(status,{'status':'stopped','updated_utc':now()});return
  leases=acquire(config,1)
  if not leases:write_json(status,{'status':'waiting_for_gpu','updated_utc':now()});time.sleep(10)
 gpu,handle=leases[0];os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid']
 try:
  import torch
  from model import create_model
  from codec import pack,unpack
  from data import image_tensor
  from preview import sheet
  from reuse_audit import audit_reuse
  torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
  pixels=np.load(json.loads((root/'data/READY.json').read_text())['confirmation_cache'],mmap_mode='r')
  with torch.inference_mode():
   for variant in config['variants']:
    if stopped():
     write_json(status,{'status':'stopped','updated_utc':now()});return
    out=root/f'reports/gene_audit/{variant}';out.mkdir(parents=True,exist_ok=True)
    write_json(status,{'status':'auditing','variant':variant,'physical_gpu':gpu,'updated_utc':now()})
    saved=torch.load(root/f'runs/{variant}/checkpoint_best.pt',map_location='cpu',weights_only=False)
    m=create_model(config,variant).eval().requires_grad_(False)
    initial={s:d.raw.detach().clone() for s,d in m.dictionaries.items()}
    m.load_state_dict(saved['model'],strict=True);m.set_step(saved['step'])
    for name,value in m.state_dict().items():
     if not torch.equal(value.cpu(),saved['model'][name]):raise AssertionError('Checkpoint mismatch: '+name)
    if variant.startswith('random_'):
     for s,d in m.dictionaries.items():
      if not torch.equal(initial[s],d.raw):raise AssertionError('Frozen random dictionary changed')
    audits=[];reuse=[];deletions=[];images=[]
    metrics=json.loads((root/f'runs/{variant}/city_confirmation_metrics.json').read_text())
    write_json(status,{'status':'functional_reuse_audit','variant':variant,'physical_gpu':gpu,'updated_utc':now()})
    if not audit_reuse(m,pixels,config,out,stopped):
     write_json(status,{'status':'stopped','updated_utc':now()});return
    for rate,cap in enumerate(config['model']['budgets']):
     m.rate_index=rate;x=image_tensor(pixels,np.arange(16));result=m(x)
     count,byte_count=m.actual_complexity(result);reloaded={};wire=0
     for s,a in result['activations'].items():
      blob=pack(a[0].cpu().numpy(),config['model']['expression_max']);wire+=len(blob)
      (out/f'cap{cap}.{s}.bin').write_bytes(blob)
      reloaded[s]=torch.from_numpy(unpack(blob,config['model']['expression_max'])).cuda()[None]
      torch.testing.assert_close(reloaded[s],a[:1],rtol=0,atol=0)
     if wire!=int(byte_count[0]):raise AssertionError('Actual code bytes disagree with analytical count')
     pred=m.decoder(m.synthesize_codes(reloaded));error=float((pred-result['reconstruction'][:1]).abs().max())
     if error>1e-4:raise AssertionError('Code-only reconstruction mismatch '+str(error))
     audits.append({'cap':cap,'actual_bytes_image0':wire,'actual_genes_image0':int(count[0]),'max_gene_cap_respected':bool((count<=cap).all()),'code_only_rgb_max_error':error})
     stats=metrics['rates'][str(rate)];usage=np.asarray(stats['per_gene_image_count']);ordered=np.argsort(-usage)
     reuse.append({'cap':cap,'city_images':128,'genes_used_on_at_least_two_cities':int((usage>=2).sum()),'genes_used_on_at_least_thirteen_cities':int((usage>=13).sum()),'top16_share_of_gene_calls':float(usage[ordered[:16]].sum()/max(usage.sum(),1)),'per_gene_image_count':usage.tolist(),'usage_entropy_effective_genes':stats['effective_genes_from_usage_entropy'],'note':'Repeated use is descriptive; support limits alone do not establish learned dictionary value.'})
     for i in range(min(4,len(x))):
      if rate==0:images.append([('Original',x[i])])
      images[i].append((f'Cap{cap}, actual{int(count[i])}',result['reconstruction'][i]))
     if rate==0:
      picture_rows=[]
      for i in range(16):
       if stopped():
        write_json(status,{'status':'stopped','updated_utc':now()});return
       base=result['reconstruction'][i:i+1];acts={s:a[i:i+1] for s,a in result['activations'].items()}
       losses=[]
       for s,a in acts.items():
        used=(a[0]!=0).any((1,2)).nonzero().flatten().tolist()
        for gene in used:
         edited={key:value.clone() for key,value in acts.items()};edited[s][:,gene]=0
         y=m.decoder(m.synthesize_codes(edited))
         record={'image_index':i,'scale':s,'gene':gene,'rgb_change_mae':float((y-base).abs().mean()),'reconstruction_mae_increase':float((y-x[i:i+1]).abs().mean()-(base-x[i:i+1]).abs().mean())}
         deletions.append(record);losses.append((record['rgb_change_mae'],s,gene,y[0]))
       if i<4 and losses:
        strongest=max(losses,key=lambda value:value[0]);picture_rows.append([('Original',x[i]),('Cap8 reconstruction',base[0]),(f'Delete {strongest[1]} gene{strongest[2]}',strongest[3]),('Absolute edit x5',(strongest[3]-base[0]).abs()*5)])
      if picture_rows:sheet(picture_rows,out/'cap8_gene_deletion.png')
    sheet(images,out/'four_caps_reconstruction.png')
    write_json(out/'code_replay_audit.json',{'strict_checkpoint_match':True,'checkpoint_step':saved['step'],'rates':audits})
    write_json(out/'gene_reuse.json',reuse);write_json(out/'cap8_per_gene_deletions.json',deletions)
    write_json(out/'COMPLETE.json',{'status':'complete','updated_utc':now()})
    del m,result;torch.cuda.empty_cache()
  write_json(status,{'status':'complete','updated_utc':now()})
  with (root/'RESULTS.md').open('a') as f:
   f.write('\n## 主结果：有效跨城市复用\n\n主档为调用上限64。固定全部模型权重，逐一删除128个城市图像上每个已激活基因。下表的“有益”指删除后MAE增量超过原误差的1%且超过1e-5；这是预设描述阈值，完整连续增量与正负效果均保存。\n\n|模型|反复激活的基因数|至少2城市有益的基因数|同时覆盖至少2国家的基因数|\n|---|---:|---:|---:|\n')
   for variant in config['variants']:
    reuse=json.loads((root/f'reports/gene_audit/{variant}/functional_reuse_summary.json').read_text())
    f.write(f'|{variant}|{reuse["genes_active_in_at_least_two_images"]}|{reuse["genes_with_repeated_measurable_benefit"]}|{reuse["genes_with_repeated_measurable_benefit_across_countries"]}|\n')
   f.write('\n反复激活与有效复用分别报告；随机字典也可能复用，learned/random比较检验字典学习的额外价值。有效复用不以优于random为定义条件。特征可加性由线性合成保证。\n\n严格checkpoint对齐、真实二进制码重放、固定随机字典不变性检查均已通过。gene_reuse.json记录逐档激活频率，functional_reuse_deletions.json记录主档全部逐基因删除，cap8_per_gene_deletions.json是次要小容量诊断。删除例图选每张图影响最大的基因，完整结果均保留。详见reports/gene_audit/。\n')
 except Exception as e:
  write_json(status,{'status':'failed','error':str(e),'updated_utc':now()});raise
 finally:handle.close()
