"""Lease four idle GPUs: two NCCL ranks for each of the two existing models."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from common import load_config, now, write_json
from supervisor import acquire, idle_gpus, read_progress

STOP=False


def request_stop(*_):
    global STOP
    STOP=True


def next_phase(root,variant):
    path=root/f'reports/{variant}.distributed_smoke.json'
    if path.exists() and json.loads(path.read_text()).get('status')=='passed':
        return 'train'
    return 'smoke'


def launch(config,variant,gpus,phase):
    root=Path(config['root']);source=root/'source'
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=','.join(gpu['uuid'] for gpu in gpus),
        PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(source),
        TMPDIR=config['temporary_directory'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        MPLCONFIGDIR=str(Path(config['temporary_directory'])/'matplotlib'),
        TORCH_DISTRIBUTED_DEBUG='OFF',NCCL_DEBUG='WARN')
    command=[config['python'],'-m','torch.distributed.run','--standalone','--nnodes=1',
        f'--nproc-per-node={len(gpus)}','--max-restarts=0',str(source/'train_distributed.py'),
        '--config',str(root/'configs/experiment.json'),'--variant',variant,'--mode',phase]
    with (root/f'logs/{variant}.{phase}.log').open('a') as log:
        return subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)


def supervise(config):
    root=Path(config['root'])
    lock=(root/'queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active={};failures={};stopping=False
    per_model=config['compute']['gpus_per_variant']
    while True:
        if STOP or (root/'STOP_REQUESTED').exists():
            # Trainers check this collectively and save before exiting; do not
            # kill torchrun while it still owns distributed model workers.
            (root/'STOP_REQUESTED').touch(exist_ok=True);stopping=True
        for variant,task in list(active.items()):
            code=task['process'].poll()
            if code is None:continue
            if code!=0:failures[variant]={'phase':task['phase'],'returncode':code}
            elif not stopping and task['phase']=='smoke':
                idle={gpu['uuid'] for gpu in idle_gpus()}
                if all(gpu['uuid'] in idle for gpu in task['gpus']):
                    task['phase']='train';task['process']=launch(config,variant,task['gpus'],'train')
                    continue
            for handle in task['leases']:handle.close()
            del active[variant]
        pending=[variant for variant in config['variants'] if variant not in active and variant not in failures
                 and read_progress(root,variant)['status']!='complete']
        used=sum(len(task['gpus']) for task in active.values())
        if not stopping:
            for variant in pending:
                if used+per_model>config['compute']['max_gpus']:break
                chosen=acquire(config,per_model)
                if len(chosen)!=per_model:
                    for gpu,handle in chosen:handle.close()
                    break
                gpus=[gpu for gpu,handle in chosen];handles=[handle for gpu,handle in chosen]
                phase=next_phase(root,variant)
                active[variant]={'gpus':gpus,'leases':handles,'phase':phase,
                    'process':launch(config,variant,gpus,phase)}
                used+=per_model
        completed=[v for v in config['variants'] if read_progress(root,v)['status']=='complete']
        if stopping:status='stopping' if active else 'stopped'
        elif active:status='training' if any(t['phase']=='train' for t in active.values()) else 'gpu_checks'
        elif failures:status='failed'
        elif len(completed)==len(config['variants']):status='complete'
        else:status='waiting_for_gpu'
        write_json(root/'queue_status.json',{'status':status,'pid':os.getpid(),
            'completed':completed,'failures':failures,'max_physical_gpus':config['compute']['max_gpus'],
            'gpus_per_model':per_model,'global_batch_per_model':config['training']['effective_batch_size'],
            'gpu_only':True,'active':{variant:{'launcher_pid':task['process'].pid,'phase':task['phase'],
                'physical_gpus':task['gpus']} for variant,task in active.items()},'updated_utc':now()})
        if not active and (stopping or failures or len(completed)==len(config['variants'])):
            if len(completed)==len(config['variants']):
                from report import comparison
                comparison(root,config['variants'])
            return
        time.sleep(config['compute']['poll_seconds'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    supervise(load_config(args.config))
