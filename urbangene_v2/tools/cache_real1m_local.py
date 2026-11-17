#!/usr/bin/env python3
"""Copy immutable indexed features to local disk without changing NAS indices."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

CHUNK = 16 * 1024 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def fingerprint(path):
    stat = Path(path).stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino}


def plan(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.is_relative_to(source):
        raise ValueError('Cache must be separate from the immutable source dataset')
    indices, files = {}, []
    for split in ['validation', 'train']:
        index_path = source / f'features.{split}.index.json'
        indices[split] = digest(index_path)
        index = json.loads(index_path.read_text())
        if sum(row['records'] for row in index['shards']) != index['records']:
            raise ValueError('Inconsistent shard record counts')
        for row in index['shards']:
            original = Path(row['path'])
            relative = original.resolve().relative_to(source)
            local = destination / relative
            if not local.resolve().is_relative_to(destination):
                raise ValueError('Cache destination escapes its root')
            files.append({'source': str(original), 'local': str(local),
                          'sha256': row['feature_sha256'], 'records': row['records'],
                          'split': split, 'bytes': original.stat().st_size})
    if len({row['source'] for row in files}) != len(files):
        raise ValueError('Repeated source shard')
    return {'source_root': str(source), 'cache_root': str(destination),
            'index_sha256': indices, 'files': files,
            'total_bytes': sum(row['bytes'] for row in files), 'created_utc': now()}


def copy_completed(row, progress=lambda n: None):
    """Direct copy: no feature hashing or payload readback, as requested."""
    source, target = Path(row['source']), Path(row['local'])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError('Refusing a symlink cache target')
    if target.exists():
        if target.stat().st_size != row['bytes']:
            raise ValueError(f'Existing cache file has incomplete length: {target}')
        return fingerprint(target)
    temporary = target.with_name(target.name + '.copying')
    if temporary.is_symlink():
        raise ValueError('Refusing a symlink partial target')
    copied = 0
    with source.open('rb') as reader, temporary.open('wb') as writer:
        os.posix_fadvise(reader.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
        for chunk in iter(lambda: reader.read(CHUNK), b''):
            writer.write(chunk)
            copied += len(chunk)
            progress(copied)
        writer.flush()
        os.fsync(writer.fileno())
    if copied != row['bytes']:
        raise ValueError(f'Source length changed during copy: {source}')
    temporary.replace(target)
    directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return fingerprint(target)


def migrate(source, destination, state_dir, reserve_bytes=256 * 2**30):
    destination, state_dir = Path(destination).resolve(), Path(state_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    with (destination / '.migration.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        migration = plan(source, destination)
        plan_path = state_dir / 'plan.json'
        if plan_path.exists():
            previous = json.loads(plan_path.read_text())
            for key in ['source_root', 'cache_root', 'index_sha256', 'files']:
                if previous[key] != migration[key]:
                    raise ValueError('Migration source or destination identity changed')
        else:
            atomic_json(plan_path, migration)
        receipts = destination / 'receipts'
        receipts.mkdir(exist_ok=True)
        needed = sum(row['bytes'] for row in migration['files'] if not Path(row['local']).exists())
        if shutil.disk_usage(destination).free < needed + reserve_bytes:
            raise OSError('Insufficient local capacity including reserved free space')
        started = time.monotonic()
        state = {'status': 'copying', 'pid': os.getpid(), 'started_utc': now(),
                 'total_shards': len(migration['files']), 'total_bytes': migration['total_bytes'],
                 'copied_shards': 0, 'copied_bytes': 0, 'active_file_bytes_copied': 0,
                 'source_root': migration['source_root'], 'cache_root': str(destination),
                 'index_sha256': migration['index_sha256'], 'source_files_deleted': False,
                 'feature_hashing': False, 'feature_readback': False}

        def publish():
            elapsed = time.monotonic() - started
            state.update(updated_utc=now(), elapsed_seconds=round(elapsed, 1))
            atomic_json(state_dir / 'status.json', state)
            atomic_json(destination / 'MIGRATION_STATUS.json', state)

        completed = []
        try:
            publish()
            for number, row in enumerate(migration['files']):
                state.update(active_source=row['source'], active_file_bytes=row['bytes'],
                             active_file_bytes_copied=0, active_phase='copying')
                last = 0.

                def progress(copied):
                    nonlocal last
                    state['active_file_bytes_copied'] = copied
                    if time.monotonic() - last >= 10:
                        publish()
                        last = time.monotonic()

                receipt_path = receipts / f'{row["split"]}_{Path(row["local"]).name}.json'
                receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
                if (receipt and Path(row['local']).is_file()
                        and receipt['expected_sha256_from_original_index'] == row['sha256']
                        and receipt['fingerprint'] == fingerprint(row['local'])):
                    actual = receipt['fingerprint']
                else:
                    for attempt in range(5):
                        try:
                            actual = copy_completed(row, progress)
                            break
                        except OSError as error:
                            state.update(active_error=str(error), retry=attempt + 1)
                            publish()
                            if attempt == 4:
                                raise
                            time.sleep(min(10 * (attempt + 1), 60))
                    atomic_json(receipt_path, {'source': row['source'], 'local': row['local'],
                                'expected_sha256_from_original_index': row['sha256'],
                                'fingerprint': actual, 'copied_utc': now(),
                                'feature_hashing': False, 'feature_readback': False})
                sidecar = Path(row['source'] + '.json')
                if sidecar.exists():
                    target_meta = Path(row['local'] + '.json')
                    if not target_meta.exists():
                        shutil.copyfile(sidecar, target_meta)
                completed.append({**row, 'fingerprint': actual})
                state.update(copied_shards=number + 1,
                             copied_bytes=state['copied_bytes'] + row['bytes'],
                             active_file_bytes_copied=row['bytes'], active_phase='copied')
                state.pop('active_error', None)
                publish()
                print(json.dumps({k: state[k] for k in ['copied_shards', 'total_shards',
                      'copied_bytes', 'total_bytes', 'elapsed_seconds']}), flush=True)
            for split, expected in migration['index_sha256'].items():
                original = Path(source) / f'features.{split}.index.json'
                if digest(original) != expected:
                    raise ValueError('Original index changed during migration')
                shutil.copyfile(original, destination / original.name)
            complete = {**migration, 'files': completed, 'status': 'complete', 'completed_utc': now(),
                        'source_files_deleted': False, 'feature_hashing': False,
                        'feature_readback': False, 'copy_policy': 'direct copy, per user instruction'}
            atomic_json(destination / 'CACHE_COMPLETE.json', complete)
            atomic_json(state_dir / 'CACHE_COMPLETE.json', complete)
            state.update(status='complete', active_phase='complete')
            publish()
        except BaseException as error:
            state.update(status='failed', error_type=type(error).__name__, error=str(error))
            publish()
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--state-dir', required=True)
    args = parser.parse_args()
    migrate(args.source, args.destination, args.state_dir)
