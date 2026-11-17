"""Lease at most four idle GPUs, run gates, then two independent DDP models."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from common import load_config, now, write_json
from reference.supervisor import acquire, idle_gpus


STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def read_progress(root, variant):
    path = root / 'runs' / variant / 'progress.json'
    return json.loads(path.read_text()) if path.exists() else {'status': 'not_started'}


def launch(config, variant, gpus, phase):
    root = Path(config['root'])
    source = root / 'source'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in gpus),
        PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(source),
        TMPDIR=config['temporary_directory'], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        MPLCONFIGDIR=str(Path(config['temporary_directory']) / 'matplotlib'),
        TORCH_DISTRIBUTED_DEBUG='OFF', NCCL_DEBUG='WARN')
    if phase == 'gpu_tests':
        env['RUN_GPU_TESTS'] = '1'
        command = [config['python'], '-m', 'pytest', '-q', '-o', 'addopts=', '-p', 'no:cacheprovider',
            str(source / 'tests/test_model_gpu.py'),
            f'--basetemp={config["temporary_directory"]}/pytest_gpu',
            f'--junitxml={root}/reports/gpu_model_tests.xml']
    else:
        command = [config['python'], '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
            f'--nproc-per-node={len(gpus)}', '--max-restarts=0', str(source / 'train.py'),
            '--config', str(root / 'configs/experiment.json'), '--variant', variant, '--mode', phase]
    with (root / f'logs/{variant}.{phase}.log').open('a') as log:
        return subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)


def next_phase(root, variant):
    path = root / f'reports/{variant}.distributed_smoke.json'
    return 'train' if path.exists() and json.loads(path.read_text()).get('status') == 'passed' else 'smoke'


def reserve_pair(config):
    selected = acquire(config, config['compute']['gpus_per_variant'])
    if len(selected) != config['compute']['gpus_per_variant']:
        for _, handle in selected:
            handle.close()
        return []
    return selected


def supervise(config):
    root = Path(config['root'])
    lock = (root / 'queue.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gate = root / 'reports/gpu_tests_passed.json'
    active, failures = {}, {}
    stopping = False
    while True:
        if STOP or (root / 'STOP_REQUESTED').exists():
            (root / 'STOP_REQUESTED').touch(exist_ok=True)
            stopping = True
        for variant, task in list(active.items()):
            code = task['process'].poll()
            if code is None:
                continue
            if code != 0:
                failures[variant] = {'phase': task['phase'], 'returncode': code}
            elif task['phase'] == 'gpu_tests':
                write_json(gate, {'status': 'passed', 'updated_utc': now()})
            elif not stopping and task['phase'] == 'smoke':
                if all(g['uuid'] in {x['uuid'] for x in idle_gpus()} for g in task['gpus']):
                    task['phase'] = 'train'
                    task['process'] = launch(config, variant, task['gpus'], 'train')
                    continue
            for handle in task['leases']:
                handle.close()
            del active[variant]
        data_ready = (root / 'data/READY.json').exists()
        if data_ready and not stopping and not failures:
            if not gate.exists() and not active:
                chosen = reserve_pair(config)
                if chosen:
                    gpus = [g for g, _ in chosen]
                    active['_preflight'] = {'gpus': gpus, 'leases': [h for _, h in chosen],
                        'phase': 'gpu_tests', 'process': launch(config, '_preflight', gpus, 'gpu_tests')}
            elif gate.exists():
                pending = [v for v in config['variants'] if v not in active and read_progress(root, v)['status'] != 'complete']
                used = sum(len(t['gpus']) for t in active.values())
                for variant in pending:
                    if used + config['compute']['gpus_per_variant'] > config['compute']['max_gpus']:
                        break
                    chosen = reserve_pair(config)
                    if not chosen:
                        break
                    gpus = [g for g, _ in chosen]
                    phase = next_phase(root, variant)
                    active[variant] = {'gpus': gpus, 'leases': [h for _, h in chosen],
                        'phase': phase, 'process': launch(config, variant, gpus, phase)}
                    used += len(gpus)
        completed = [v for v in config['variants'] if read_progress(root, v)['status'] == 'complete']
        pending = [v for v in config['variants'] if v not in active and v not in completed]
        if stopping:
            status = 'stopping' if active else 'stopped'
        elif failures:
            status = 'failed_draining' if active else 'failed'
        elif len(completed) == len(config['variants']):
            status = 'complete'
        elif active:
            status = 'training' if any(t['phase'] == 'train' for t in active.values()) else 'gpu_checks'
        else:
            status = 'waiting_for_gpu' if data_ready else 'preparing_data'
        write_json(root / 'queue_status.json', {'status': status, 'pid': os.getpid(), 'completed': completed,
            'pending': pending, 'failures': failures, 'max_physical_gpus': config['compute']['max_gpus'],
            'gpus_per_model': config['compute']['gpus_per_variant'],
            'global_batch_per_model': config['training']['effective_batch_size'], 'gpu_only': True,
            'active': {v: {'launcher_pid': t['process'].pid, 'phase': t['phase'], 'physical_gpus': t['gpus']}
                       for v, t in active.items()}, 'updated_utc': now()})
        if not active and (stopping or failures or len(completed) == len(config['variants'])):
            if len(completed) == len(config['variants']):
                from report import comparison
                comparison(root, config['variants'])
            return
        time.sleep(config['compute']['poll_seconds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    supervise(load_config(args.config))
