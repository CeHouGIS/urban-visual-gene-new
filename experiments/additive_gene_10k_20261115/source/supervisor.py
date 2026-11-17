"""Persistently launch only this experiment, only on leased idle GPUs (max 2)."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from common import load_config, now, write_json

STOP=False


def request_stop(*_):
    global STOP
    STOP=True


def idle_gpus():
    try:
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=15)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid',
            '--format=csv,noheader'],text=True,timeout=15)
    except (OSError,subprocess.SubprocessError):
        return []
    busy={x.strip() for x in apps.splitlines()}
    result=[]
    for line in rows.splitlines():
        index,uuid,memory,utilization=[x.strip() for x in line.split(',')]
        if uuid not in busy and int(memory)<1024 and int(utilization)<10:
            result.append({'index':int(index),'uuid':uuid})
    return result


def acquire(config,limit):
    chosen=[]
    directory=Path(config['shared_gpu_leases'])
    directory.mkdir(parents=True,exist_ok=True)
    for gpu in idle_gpus():
        if len(chosen)>=limit:
            break
        handle=(directory/f'{gpu["uuid"]}.lock').open('a+')
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        # Query again after obtaining the cooperative lock.
        if gpu['uuid'] not in {x['uuid'] for x in idle_gpus()}:
            handle.close()
            continue
        chosen.append((gpu,handle))
    return chosen


def read_progress(root,variant):
    path=root/'runs'/variant/'progress.json'
    return json.loads(path.read_text()) if path.exists() else {'status':'not_started'}


def worker_command(config,variant,phase):
    source=Path(config['root'])/'source'
    if phase=='tests':
        return [config['python'],'-m','pytest','-q','-o','addopts=',
            '-p','no:cacheprovider',str(source/'tests/test_model_gpu.py'),
            f'--junitxml={config["root"]}/reports/{variant}.gpu_tests.xml']
    return [config['python'],str(source/'train.py'),'--config',
        str(Path(config['root'])/'configs/experiment.json'),'--variant',variant,
        '--mode','smoke' if phase=='smoke' else 'train']


def next_phase(root,variant):
    smoke=root/f'reports/{variant}.gpu_smoke.json'
    tests=root/f'reports/{variant}.gpu_tests_passed.json'
    if smoke.exists() and json.loads(smoke.read_text()).get('status')=='passed':
        return 'train'
    if tests.exists():
        return 'smoke'
    return 'tests'


def launch(config,variant,gpu,phase):
    root=Path(config['root'])
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(root/'source'),TMPDIR=config['temporary_directory'],OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',MPLCONFIGDIR=str(Path(config['temporary_directory'])/'matplotlib'),
        RUN_GPU_TESTS='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    log=(root/f'logs/{variant}.{phase}.log').open('a')
    proc=subprocess.Popen(worker_command(config,variant,phase),stdout=log,stderr=subprocess.STDOUT,
        env=env,cwd=root/'source',start_new_session=True)
    log.close()
    return proc


def supervise(config):
    root=Path(config['root'])
    lock=(root/'queue.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active={}
    failures={}
    stopping=False
    while True:
        if STOP or (root/'STOP_REQUESTED').exists():
            if not stopping:
                for task in active.values():
                    if task['process'].poll() is None:
                        task['process'].terminate()
                stopping=True
        for variant,task in list(active.items()):
            returncode=task['process'].poll()
            if returncode is None:
                continue
            phase=task['phase']
            if returncode!=0:
                failures[variant]={'phase':phase,'returncode':returncode}
            elif not stopping and phase in {'tests','smoke'}:
                if phase=='tests':
                    write_json(root/f'reports/{variant}.gpu_tests_passed.json',
                               {'status':'passed','updated_utc':now()})
                # An external user can occupy a card even while our cooperative
                # lease is held. Recheck between every independent GPU process.
                if task['gpu']['uuid'] in {g['uuid'] for g in idle_gpus()}:
                    phase_to_start=next_phase(root,variant)
                    task['phase']=phase_to_start
                    task['process']=launch(config,variant,task['gpu'],phase_to_start)
                    continue
            task['lease'].close()
            del active[variant]
        ready_path=root/'data/READY.json'
        data_ready=ready_path.exists()
        data_status_path=root/'data/preparation.json'
        data_status=json.loads(data_status_path.read_text()) if data_status_path.exists() else {}
        if data_status.get('status')=='failed':
            failures['data']=data_status
        pending=[v for v in config['variants'] if v not in active and v not in failures
                 and read_progress(root,v)['status']!='complete']
        if data_ready and not stopping and 'data' not in failures and pending:
            limit=config['compute']['max_gpus']-len(active)
            for variant,(gpu,lease) in zip(pending,acquire(config,min(limit,len(pending))) if limit else []):
                phase=next_phase(root,variant)
                active[variant]={'gpu':gpu,'lease':lease,'phase':phase,
                    'process':launch(config,variant,gpu,phase)}
        completed=[v for v in config['variants'] if read_progress(root,v)['status']=='complete']
        if stopping:
            status='stopping' if active else 'stopped'
        elif len(completed)==len(config['variants']):
            status='complete'
        elif not data_ready:
            status='data_failed' if 'data' in failures else 'preparing_data'
        elif active:
            status='training' if any(t['phase']=='train' for t in active.values()) else 'gpu_checks'
        elif failures:
            status='failed'
        else:
            status='waiting_for_gpu'
        write_json(root/'queue_status.json',{'status':status,'pid':os.getpid(),
            'completed':completed,'pending':pending,'failures':failures,
            'active':{v:{'pid':t['process'].pid,'phase':t['phase'],'physical_gpu':t['gpu']}
                      for v,t in active.items()},
            'gpu_only':True,'max_physical_gpus':config['compute']['max_gpus'],
            'data_status':data_status,'updated_utc':now()})
        if not active and (stopping or failures or len(completed)==len(config['variants'])):
            if len(completed)==len(config['variants']):
                from report import comparison
                comparison(root,config['variants'])
            return
        time.sleep(config['compute']['poll_seconds'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    supervise(load_config(args.config))
