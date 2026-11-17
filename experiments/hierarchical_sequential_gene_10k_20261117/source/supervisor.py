"""Persistent experiment-local queue. Lease only two idle GPUs for all gates."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from common import load_config, now, write_json

STOP=False


def stop_signal(*_):
    global STOP
    STOP=True


def idle_gpus():
    try:
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=15)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True,timeout=15)
    except (OSError,subprocess.SubprocessError):
        return []
    busy={v.strip() for v in apps.splitlines()}
    result=[]
    for line in rows.splitlines():
        i,u,m,p=[v.strip() for v in line.split(',')]
        if u not in busy and int(m)<1024 and int(p)<10:
            result.append(dict(index=int(i),uuid=u))
    return result


def acquire(config):
    selected=[]
    directory=Path(config['shared_gpu_leases'])
    directory.mkdir(parents=True,exist_ok=True)
    for gpu in idle_gpus():
        handle=(directory/f'{gpu["uuid"]}.lock').open('a+')
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        if gpu['uuid'] not in {v['uuid'] for v in idle_gpus()}:
            handle.close()
            continue
        selected.append((gpu,handle))
        if len(selected)==2:
            return selected
    for _,h in selected:
        h.close()
    return []


def next_phase(root):
    for name,path in [('tests','reports/gpu_tests.json'),('smoke','reports/distributed_smoke.json'),('train','reports/training_complete.json'),('analysis','reports/analysis_complete.json')]:
        p=root/path
        if not p.exists() or json.loads(p.read_text()).get('status') not in {'passed','complete'}:
            return name
    return 'complete'


def run(config):
    root=Path(config['root'])
    lock=(root/'queue.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    leases=[]
    process=None
    phase=None
    while True:
        if STOP or (root/'STOP_REQUESTED').exists():
            (root/'STOP_REQUESTED').touch(exist_ok=True)
            if process and process.poll() is None:
                # Worker observes the marker at update boundaries and saves.
                write_json(root/'queue_status.json',dict(status='stopping',pid=os.getpid(),updated_utc=now()))
                time.sleep(2)
                continue
            write_json(root/'queue_status.json',dict(status='stopped',pid=os.getpid(),updated_utc=now()))
            return
        if process and process.poll() is not None:
            code=process.returncode
            if code:
                write_json(root/'queue_status.json',dict(status='failed',phase=phase,returncode=code,
                    log=str(root/f'logs/{phase}.log'),updated_utc=now()))
                return
            process=None
            # Recheck physical availability between independent GPU processes.
            for _,h in leases:
                h.close()
            leases=[]
        if process is None:
            phase=next_phase(root)
            if phase=='complete':
                write_json(root/'queue_status.json',dict(status='complete',updated_utc=now()))
                return
            leases=acquire(config)
            if leases:
                env=os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g,h in leases),
                    PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'source'),
                    TMPDIR=config['temporary_directory'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                    MPLCONFIGDIR=str(Path(config['temporary_directory'])/'matplotlib'),NCCL_DEBUG='WARN')
                env.update(CUDA_CACHE_PATH=str(Path(config['temporary_directory'])/'cuda_cache'),
                    TORCHINDUCTOR_CACHE_DIR=str(Path(config['temporary_directory'])/'torchinductor'),
                    TRITON_CACHE_DIR=str(Path(config['temporary_directory'])/'triton'))
                script={'tests':'test_gpu.py','analysis':'analyze.py'}.get(phase,'train.py')
                command=[config['python'],'-m','torch.distributed.run','--standalone','--nnodes=1',
                    '--nproc-per-node=2','--max-restarts=0',str(root/'source'/script),'--config',str(root/'configs/experiment.json')]
                if phase in {'smoke','train'}:
                    command+=['--mode',phase]
                with (root/f'logs/{phase}.log').open('a') as log:
                    process=subprocess.Popen(command,cwd=root/'source',env=env,stdout=log,stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,start_new_session=True)
        write_json(root/'queue_status.json',dict(status='running' if process else 'waiting_for_gpu',
            phase=phase,pid=os.getpid(),worker_pid=process.pid if process else None,
            physical_gpus=[g for g,h in leases],max_gpus=2,updated_utc=now()))
        time.sleep(config['compute']['poll_seconds'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    signal.signal(signal.SIGINT,stop_signal)
    signal.signal(signal.SIGTERM,stop_signal)
    run(load_config(args.config))
