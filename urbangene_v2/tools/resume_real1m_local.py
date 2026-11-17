#!/usr/bin/env python3
"""Wait for direct-copy completion, then resume the frozen trainer on local data."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from cache_real1m_local import atomic_json, digest, fingerprint


def now():
    return datetime.now(timezone.utc).isoformat()


def proc_identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': stat[19], 'pgid': os.getpgid(pid),
                'command': Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def load_cache(cache, data_root):
    cache, data_root = Path(cache).resolve(), Path(data_root).resolve()
    complete = json.loads((cache / 'CACHE_COMPLETE.json').read_text())
    if complete['status'] != 'complete' or complete['source_root'] != str(data_root):
        raise ValueError('Local cache is incomplete or belongs to another dataset')
    files = {row['source']: row for row in complete['files']}
    if len(files) != len(complete['files']):
        raise ValueError('Repeated cache source paths')
    expected = set()
    for split in ['train', 'validation']:
        index_path = data_root / f'features.{split}.index.json'
        # Only tiny JSON indices are fingerprinted; feature payloads are never hashed.
        if digest(index_path) != complete['index_sha256'][split]:
            raise ValueError('NAS index identity changed; cannot resume this experiment')
        index = json.loads(index_path.read_text())
        for shard in index['shards']:
            source = shard['path']
            expected.add(source)
            row = files[source]
            local = Path(row['local'])
            if (local.is_symlink() or not local.resolve().is_relative_to(cache)
                    or row['sha256'] != shard['feature_sha256']
                    or row['records'] != shard['records']
                    or fingerprint(local) != row['fingerprint']):
                raise ValueError(f'Local copy completion metadata changed: {local}')
    if expected != files.keys():
        raise ValueError('Cache and original indices describe different shards')
    return complete, files


def cached_dataset_class(original, mapping):
    class LocalFeatures(original):
        def __init__(self, index, max_open=8):
            super().__init__(index, max_open=max_open)
            # Change only the in-memory read paths. The original index and its
            # checkpoint identity, sampling order and experiment config stay intact.
            self.shards = [{**row, 'path': mapping[row['path']]['local']} for row in self.shards]
    return LocalFeatures


def worker(args):
    frozen = Path(args.frozen).resolve()
    sys.path[:0] = [str(frozen / 'tools'), str(frozen / 'src')]
    from urbangene_v2.common import load_config
    import train_real1m
    cfg = load_config(args.config)
    _, mapping = load_cache(args.cache, cfg['data_root'])
    train_real1m.ShardedFeatures = cached_dataset_class(train_real1m.ShardedFeatures, mapping)
    train_real1m.train(cfg, 2)


def exact_gpu_leases(gpus, output):
    if len(gpus) != 2 or len({gpu['uuid'] for gpu in gpus}) != 2:
        raise ValueError('Exactly two distinct GPUs are required')
    directory = output.parent / 'gpu_leases'
    directory.mkdir(exist_ok=True)
    handles = []
    try:
        for gpu in gpus:
            handle = (directory / f'{gpu["uuid"]}.lock').open('a+')
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        busy = {line.strip() for line in subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'],
            text=True, timeout=15).splitlines()}
        info = subprocess.check_output([
            'nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True, timeout=15)
        states = {p[0]: (int(p[1]), int(p[2])) for line in info.splitlines()
                  if (p := [s.strip() for s in line.split(',')])}
        if any(g['uuid'] in busy or states[g['uuid']][0] >= 1024
               or states[g['uuid']][1] >= 10 for g in gpus):
            raise BlockingIOError('Original two GPUs are not idle yet')
        return handles
    except BaseException:
        for handle in handles:
            handle.close()
        raise


def supervise(args):
    state_dir, cache = Path(args.state_dir), Path(args.cache)
    frozen, config = Path(args.frozen), Path(args.config)
    sys.path[:0] = [str(frozen / 'src')]
    from urbangene_v2.common import load_config
    cfg = load_config(config)
    output = Path(cfg['output'])
    with (state_dir / 'handoff.lock').open('a+') as handoff_lock:
        fcntl.flock(handoff_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identities_path = state_dir / 'old_processes.json'
        if identities_path.exists():
            identities = json.loads(identities_path.read_text())
        else:
            identities = [proc_identity(pid) for pid in [args.old_parent, args.old_worker]]
            if any(i is None for i in identities):
                raise ValueError('Original training process disappeared before registration')
            if (identities[0]['pgid'] != args.old_parent
                    or identities[1]['pgid'] != args.old_parent
                    or any(str(config) not in i['command'] for i in identities)
                    or '--worker train' not in identities[1]['command']):
                raise ValueError('Original process identities do not match this training run')
            atomic_json(identities_path, identities)
        execution = json.loads((output / 'execution.json').read_text())
        gpus = execution['physical_gpus']
        if len(gpus) != 2 or len({g['uuid'] for g in gpus}) != 2:
            raise ValueError('Resume requires the original two distinct GPUs')
        status = {'pid': os.getpid(), 'started_utc': now(), 'cache_root': str(cache),
                  'config': str(config), 'frozen_source': str(frozen), 'physical_gpus': gpus,
                  'max_gpus': 2, 'cpu_fallback': False}

        def phase(name, **extra):
            status.update(phase=name, updated_utc=now(), **extra)
            atomic_json(state_dir / 'handoff.json', status)
            print(json.dumps(status), flush=True)

        def original_alive():
            return all(proc_identity(i['pid']) == i for i in identities)

        stopped = False
        terminated = False
        leases = []
        execution_lock = None
        try:
            phase('waiting_for_copy')
            while not (cache / 'CACHE_COMPLETE.json').exists():
                progress = state_dir / 'status.json'
                if progress.exists() and json.loads(progress.read_text())['status'] == 'failed':
                    raise RuntimeError('Copy failed; original training was left running')
                time.sleep(15)
            complete, _ = load_cache(cache, cfg['data_root'])
            progress = json.loads((output / 'progress.json').read_text())
            if progress['status'] == 'complete':
                phase('cache_ready_training_already_complete')
                return
            if not original_alive():
                raise RuntimeError('Original training identity changed; refusing to signal any process')
            resume = output / 'checkpoint_resume.pt'
            previous = fingerprint(resume)
            phase('waiting_for_next_checkpoint', progress_step=progress['step'])
            while fingerprint(resume) == previous:
                if not original_alive():
                    raise RuntimeError('Original training exited while awaiting checkpoint')
                time.sleep(.2)
            # Give the best/preview checkpoint writes time to finish after resume.
            time.sleep(1)
            if not original_alive():
                raise RuntimeError('Original training changed before handoff')
            # Signal only the two registered processes, never an unchecked process group.
            stopped = True
            for identity in reversed(identities):
                os.kill(identity['pid'], signal.SIGSTOP)
            phase('saving_handoff_record')
            import torch
            checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
            if (checkpoint['config'] != cfg or checkpoint['data_identity'] != complete['index_sha256']
                    or checkpoint['gpu_count'] != 2):
                raise ValueError('Checkpoint is incompatible; original training will be continued')
            temporary_checkpoints = list(output.glob('checkpoint*.pt.tmp'))
            if temporary_checkpoints:
                # A resumed handoff can be retried later; do not stop halfway through a save.
                raise RuntimeError('A secondary checkpoint is still being written; leaving training running')
            for name in ['checkpoint_resume.pt', 'execution.json', 'progress.json']:
                target = state_dir / ('before_local_' + name)
                if not target.exists():
                    shutil.copyfile(output / name, target)
            phase('stopping_original_worker', resume_step=checkpoint['step'])
            for identity in reversed(identities):
                os.kill(identity['pid'], signal.SIGTERM)
                os.kill(identity['pid'], signal.SIGCONT)
            stopped = False
            deadline = time.monotonic() + 60
            while any(proc_identity(i['pid']) is not None for i in identities):
                if time.monotonic() >= deadline:
                    raise RuntimeError('Original processes have not exited; no second trainer launched')
                time.sleep(1)
            terminated = True
            execution_lock = (output / 'execution.lock').open('a+')
            fcntl.flock(execution_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            atomic_json(output / 'execution.json', {**execution, 'pid': os.getpid(),
                        'phase': 'waiting_for_original_two_gpus', 'updated_utc': now(),
                        'local_cache': str(cache), 'resumed_from_step': checkpoint['step']})
            phase('waiting_for_original_two_gpus')
            while not leases:
                try:
                    leases = exact_gpu_leases(gpus, output)
                except (BlockingIOError, subprocess.SubprocessError):
                    time.sleep(15)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(g['uuid'] for g in gpus),
                   'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
                   'TRANSFORMERS_OFFLINE': '1', 'OPENBLAS_NUM_THREADS': '4',
                   'OMP_NUM_THREADS': '4', 'TMPDIR': str(cache)}
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
                       '--frozen', str(frozen), '--config', str(config), '--cache', str(cache),
                       '--state-dir', str(state_dir)]
            with (state_dir / 'local_training.log').open('a') as log:
                child = subprocess.Popen(command, env=env, cwd=frozen, stdout=log,
                                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         start_new_session=True)
                atomic_json(output / 'execution.json', {**execution, 'pid': os.getpid(),
                            'worker_pid': child.pid, 'phase': 'training_from_local_cache',
                            'updated_utc': now(), 'local_cache': str(cache),
                            'io_wrapper': str(Path(__file__).resolve()),
                            'resumed_from_step': checkpoint['step']})
                (output / 'pipeline_pid.txt').write_text(str(os.getpid()) + '\n')
                phase('training_from_local_cache', worker_pid=child.pid,
                      resumed_from_step=checkpoint['step'])
                code = child.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            current = json.loads((output / 'execution.json').read_text())
            atomic_json(output / 'execution.json', {**current, 'phase': 'complete', 'updated_utc': now()})
            phase('complete')
        except BaseException as error:
            phase('failed', error_type=type(error).__name__, error=str(error),
                  original_worker_terminated=terminated)
            raise
        finally:
            if stopped:
                for identity in identities:
                    if proc_identity(identity['pid']) == identity:
                        os.kill(identity['pid'], signal.SIGCONT)
            for handle in leases:
                handle.close()
            if execution_lock:
                execution_lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--state-dir', required=True)
    parser.add_argument('--frozen', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--old-parent', type=int)
    parser.add_argument('--old-worker', type=int)
    parser.add_argument('--worker', action='store_true')
    options = parser.parse_args()
    worker(options) if options.worker else supervise(options)
