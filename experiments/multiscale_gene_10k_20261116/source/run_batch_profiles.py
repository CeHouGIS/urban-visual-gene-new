import os,json,subprocess,time
from pathlib import Path
from common import load_config,write_json,now
from supervisor import reserve_pair
c=load_config(Path(__file__).resolve().parents[1]/'configs/experiment.json')
r=Path(c['root']);active=[]
for v in ['multiscale_vector','multiscale_conv']:
 selected=reserve_pair(c)
 if len(selected)!=2:raise RuntimeError('Batch profiling cannot acquire a complete idle GPU pair')
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g,h in selected),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(r/'source'),TMPDIR=c['temporary_directory'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',NCCL_DEBUG='WARN')
 with (r/f'logs/{v}.batch_profile.log').open('a') as f:
  proc=subprocess.Popen([c['python'],'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2','--max-restarts=0',str(r/'source/profile_batch.py'),'--config',str(r/'configs/experiment.json'),'--variant',v],cwd=r/'source',env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 active.append((v,proc,selected))
results={}
while active:
 for v,p,selected in list(active):
  code=p.poll()
  if code is None:continue
  for g,h in selected:h.close()
  results[v]={'returncode':code}
  active.remove((v,p,selected))
 write_json(r/'reports/batch_tuning/processes.json',{'active':[{'variant':v,'pid':p.pid} for v,p,s in active],'results':results,'updated_utc':now()})
 if active:time.sleep(1)
if any(x['returncode'] for x in results.values()):raise SystemExit(1)
