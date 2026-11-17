import importlib.util
import json
from pathlib import Path

import pytest


def load_tool():
    tools = Path(__file__).resolve().parents[1] / 'tools'
    spec = importlib.util.spec_from_file_location('local_feature_copier', tools / 'cache_real1m_local.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_copy_never_hashes_feature_payload_and_resumes(tmp_path, monkeypatch):
    copier = load_tool()
    source, cache, state = [tmp_path / name for name in ['source', 'cache', 'state']]
    source.mkdir()
    for split in ['train', 'validation']:
        feature = source / 'feature_shards' / split / 'features.npy'
        feature.parent.mkdir(parents=True)
        feature.write_bytes((split * 64).encode())
        (source / f'features.{split}.index.json').write_text(json.dumps({
            'records': 1, 'shards': [{'path': str(feature), 'records': 1,
                                     'feature_sha256': 'original_metadata_only'}]}))
    original_digest = copier.digest

    def metadata_digest_only(path):
        assert Path(path).suffix == '.json', 'Feature payload must never be hashed'
        return original_digest(path)

    monkeypatch.setattr(copier, 'digest', metadata_digest_only)
    copier.migrate(source, cache, state, reserve_bytes=0)
    complete = json.loads((cache / 'CACHE_COMPLETE.json').read_text())
    assert complete['feature_hashing'] is False
    for row in complete['files']:
        assert Path(row['local']).read_bytes() == Path(row['source']).read_bytes()
    # A second execution reuses completed shards rather than copying again.
    monkeypatch.setattr(copier, 'copy_completed', lambda *a: pytest.fail('Copied a completed shard'))
    copier.migrate(source, cache, state, reserve_bytes=0)
    assert json.loads((state / 'status.json').read_text())['copied_shards'] == 2


def test_interrupted_copy_is_not_published_and_can_restart(tmp_path):
    copier = load_tool()
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.write_bytes(b'payload' * 1024)
    row = {'source': str(source), 'local': str(target), 'bytes': source.stat().st_size}

    def interrupt(_):
        raise InterruptedError('test interruption')

    with pytest.raises(InterruptedError):
        copier.copy_completed(row, interrupt)
    assert not target.exists()
    copier.copy_completed(row)
    assert target.read_bytes() == source.read_bytes()
    assert not target.with_name(target.name + '.copying').exists()


def test_incomplete_source_never_becomes_trainable(tmp_path):
    copier = load_tool()
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.write_bytes(b'short')
    with pytest.raises(ValueError, match='length changed'):
        copier.copy_completed({'source': str(source), 'local': str(target), 'bytes': 100})
    assert not target.exists()


def load_resume(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / 'tools'
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location('local_feature_resume', tools / 'resume_real1m_local.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_paths_preserve_features_order_and_original_index_identity(tmp_path, monkeypatch):
    import itertools
    import numpy as np
    from urbangene_v2.million import ShardedFeatures, RepeatingShardOrder

    copier, resume = load_tool(), load_resume(monkeypatch)
    source, cache, state = [tmp_path / name for name in ['source', 'cache', 'state']]
    source.mkdir()
    original_indices = {}
    for split in ['train', 'validation']:
        shards = []
        for number in range(2):
            path = source / 'feature_shards' / split / f'{number}.npy'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.arange(number * 3, number * 3 + 3, dtype=np.float16).reshape(-1, 1, 1, 1))
            shards.append({'path': str(path), 'records': 3, 'feature_sha256': 'metadata_only'})
        index = source / f'features.{split}.index.json'
        index.write_text(json.dumps({'records': 6, 'shards': shards}))
        original_indices[split] = index.read_bytes()
    copier.migrate(source, cache, state, reserve_bytes=0)
    _, mapping = resume.load_cache(cache, source)
    original = ShardedFeatures(source / 'features.train.index.json')
    local_class = resume.cached_dataset_class(ShardedFeatures, mapping)
    local = local_class(source / 'features.train.index.json')
    assert [original[i].item() for i in range(6)] == [local[i].item() for i in range(6)]
    for epoch, offset in [(0, 0), (1, 3), (9, 5)]:
        expected = list(itertools.islice(iter(RepeatingShardOrder(original, 42, epoch, offset)), 17))
        actual = list(itertools.islice(iter(RepeatingShardOrder(local, 42, epoch, offset)), 17))
        assert expected == actual
    for split, before in original_indices.items():
        assert (source / f'features.{split}.index.json').read_bytes() == before
    # Refuse a truncated copy at handoff using metadata only, without a payload read.
    first = Path(next(iter(mapping.values()))['local'])
    first.write_bytes(b'truncated')
    with pytest.raises(ValueError, match='completion metadata changed'):
        resume.load_cache(cache, source)


def test_busy_gpu_does_not_leave_partial_reservations(tmp_path, monkeypatch):
    import fcntl
    resume = load_resume(monkeypatch)
    gpus = [{'index': 1, 'uuid': 'GPU-1'}, {'index': 2, 'uuid': 'GPU-2'}]

    def query(command, **kwargs):
        if '--query-compute-apps=gpu_uuid' in command:
            return 'GPU-2\n'
        return 'GPU-1, 0, 0\nGPU-2, 24000, 90\n'

    monkeypatch.setattr(resume.subprocess, 'check_output', query)
    output = tmp_path / 'run'
    with pytest.raises(BlockingIOError):
        resume.exact_gpu_leases(gpus, output)
    for gpu in gpus:
        with (tmp_path / 'gpu_leases' / f'{gpu["uuid"]}.lock').open('a+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with pytest.raises(ValueError, match='two distinct'):
        resume.exact_gpu_leases(gpus + [{'index': 3, 'uuid': 'GPU-3'}], output)
