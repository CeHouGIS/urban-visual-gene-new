"""Prepare manifest once, then supervise GPU runs and a single cache writer."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time

from common import load_config, now, write_json


def launch(config):
    root=Path(config['root'])
    lock=(root/'bootstrap.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',TMPDIR=config['temporary_directory'],
               OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MPLCONFIGDIR=str(Path(config['temporary_directory'])/'matplotlib'))
    source=root/'source'
    common=[config['python']]
    config_args=['--config',str(root/'configs/experiment.json')]
    write_json(root/'launch_status.json',{'status':'preparing_manifest','pid':os.getpid(),'updated_utc':now()})
    with (root/'logs/manifest.log').open('a') as log:
        prep=subprocess.run(common+[str(source/'prepare_million.py')]+config_args+['--mode','manifest'],
                            env=env,stdout=log,stderr=subprocess.STDOUT)
    if prep.returncode:
        write_json(root/'launch_status.json',{'status':'failed','phase':'manifest','returncode':prep.returncode,'updated_utc':now()})
        raise SystemExit(prep.returncode)
    if (root/'STOP_REQUESTED').exists():
        write_json(root/'launch_status.json',{'status':'stopped','updated_utc':now()})
        return
    workers={}
    for name,command in [('supervisor',common+[str(source/'supervisor.py')]+config_args),
                          ('cache',common+[str(source/'prepare_million.py')]+config_args+['--mode','cache'])]:
        with (root/f'logs/{name}.log').open('a') as log:
            workers[name]=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL)
    (root/'supervisor.pid').write_text(str(workers['supervisor'].pid)+'\n')
    while True:
        status={name:{'pid':proc.pid,'returncode':proc.poll()} for name,proc in workers.items()}
        done=all(item['returncode'] is not None for item in status.values())
        failed=any(item['returncode'] not in (None,0) for item in status.values())
        write_json(root/'launch_status.json',{'status':'failed' if failed else ('complete' if done else 'running'),
            'pid':os.getpid(),'workers':status,'updated_utc':now()})
        if done:
            return
        time.sleep(10)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    launch(load_config(args.config))
