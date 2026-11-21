"""Independent cooperative GPU queue: two extractors, normalization, two SAEs."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from utils import now, write_json

STOP = False


def stop(*_):
    global STOP
    STOP = True


def idle_gpus():
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=15)
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid',
        '--format=csv,noheader'],text=True,timeout=15)
    busy = {s.strip() for s in apps.splitlines()}
    for row in rows.splitlines():
        index,uuid,memory,usage = [s.strip() for s in row.split(',')]
        if uuid not in busy and int(memory)<1024 and int(usage)<10:
            yield dict(index=int(index),uuid=uuid)


def main(c):
    root, source = Path(c['root']), Path(c['source'])
    own = (root/'queue.lock').open('a+')
    fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write_json(root/'reports/source_manifest.json',
        {str(p.relative_to(source.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
         for group in ['source','configs','tests'] for p in (source.parent/group).glob('*') if p.is_file()})
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONPATH=str(source),
        TMPDIR=c['temporary_directory'], MPLCONFIGDIR=c['temporary_directory']+'/matplotlib',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    stages = [
        ('extract', [('extract_0','features.py',['--shard','0']),('extract_1','features.py',['--shard','1'])]),
        ('normalize',[('normalize','features.py',['--normalize'])]),
        ('sae',[(f'atoms_{m}','worker.py',['--atoms',str(m)]) for m in c['dictionary_sizes']])]
    running = {}
    try:
        for stage, jobs in stages:
            pending, completed = list(jobs), []
            while pending or running:
                stopping = STOP or (root/'STOP_REQUESTED').exists()
                if stopping:
                    pending.clear()
                    for _,child,_,_ in running.values():
                        if child.poll() is None:
                            child.terminate()
                for name,(gpu,child,handle,log) in list(running.items()):
                    code = child.poll()
                    if code is None:
                        continue
                    handle.close()
                    log.close()
                    del running[name]
                    if code and not stopping:
                        raise RuntimeError(f'{name} failed ({code}); see logs/{name}.log')
                    completed.append(name)
                if stopping and not running:
                    write_json(root/'progress.json',dict(status='stopped',stage=stage,updated_utc=now()))
                    return
                if not stopping:
                    for gpu in idle_gpus():
                        if not pending or len(running)>=c['max_gpus']:
                            break
                        handle = (Path(c['shared_gpu_leases'])/(gpu['uuid']+'.lock')).open('a+')
                        try:
                            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:
                            handle.close()
                            continue
                        if gpu['uuid'] not in {g['uuid'] for g in idle_gpus()}:
                            handle.close()
                            continue
                        name, script, args = pending.pop(0)
                        log = (root/'logs'/(name+'.log')).open('a')
                        child = subprocess.Popen([c['python'],str(source/script)]+args,
                            env=dict(env,CUDA_VISIBLE_DEVICES=gpu['uuid']),cwd=c['temporary_directory'],
                            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                            pass_fds=(handle.fileno(),))
                        running[name] = gpu,child,handle,log
                write_json(root/'progress.json',dict(status='stopping' if stopping else
                    ('running' if running else 'waiting_for_gpu'),stage=stage,supervisor_pid=os.getpid(),
                    pending=[x[0] for x in pending],completed=completed,max_gpus=c['max_gpus'],
                    workers={n:dict(gpu=g,pid=p.pid) for n,(g,p,_,_) in running.items()},updated_utc=now()))
                if pending or running:
                    time.sleep(5)
        with (root/'logs/report.log').open('a') as log:
            subprocess.run([c['python'],str(source/'report.py')],env=dict(env,CUDA_VISIBLE_DEVICES=''),
                cwd=c['temporary_directory'],stdout=log,stderr=subprocess.STDOUT,check=True)
        write_json(root/'progress.json',dict(status='complete',dictionary_sizes=c['dictionary_sizes'],updated_utc=now()))
    except BaseException as error:
        for _,child,handle,log in running.values():
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
            handle.close()
            log.close()
        write_json(root/'progress.json',dict(status='failed',error=repr(error),updated_utc=now()))
        raise


if __name__ == '__main__':
    c = json.loads((Path(__file__).parents[1]/'configs/experiment.json').read_text())
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    main(c)
