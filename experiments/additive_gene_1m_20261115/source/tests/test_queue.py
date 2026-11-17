import json
from pathlib import Path

import pytest

import supervisor


def test_idle_gpu_filter_excludes_even_low_utilization_compute_processes(monkeypatch):
    def output(command,**kwargs):
        if '--query-compute-apps=gpu_uuid' in command:
            return 'GPU-busy\n'
        return '0, GPU-busy, 400, 0\n1, GPU-idle, 50, 0\n2, GPU-memory, 1500, 0\n3, GPU-util, 20, 70\n'
    monkeypatch.setattr(supervisor.subprocess,'check_output',output)
    assert supervisor.idle_gpus()==[{'index':1,'uuid':'GPU-idle'}]


def test_no_idle_gpu_queues_without_launching_worker(tmp_path,monkeypatch):
    (tmp_path/'data').mkdir()
    (tmp_path/'runs').mkdir()
    (tmp_path/'data/READY.json').write_text('{}')
    (tmp_path/'data/preparation.json').write_text('{"status":"ready"}')
    config={'root':str(tmp_path),'variants':['dense_ae','sparse_n256_s4'],
        'compute':{'max_gpus':2,'poll_seconds':30},'shared_gpu_leases':str(tmp_path/'leases')}
    monkeypatch.setattr(supervisor,'idle_gpus',lambda:[])
    def forbidden(*args,**kwargs):
        pytest.fail('Busy GPU queue must not launch any model worker')
    monkeypatch.setattr(supervisor,'launch',forbidden)
    class StopCheck(Exception):
        pass
    def end_check(_):
        raise StopCheck()
    monkeypatch.setattr(supervisor.time,'sleep',end_check)
    with pytest.raises(StopCheck):
        supervisor.supervise(config)
    result=json.loads((tmp_path/'queue_status.json').read_text())
    assert result['status']=='waiting_for_gpu'
    assert result['active']=={}
    assert result['gpu_only'] is True


def test_cooperative_gpu_leases_never_exceed_limit(tmp_path,monkeypatch):
    monkeypatch.setattr(supervisor,'idle_gpus',lambda:[{'index':i,'uuid':f'GPU-{i}'} for i in range(4)])
    config={'shared_gpu_leases':str(tmp_path/'leases')}
    first=supervisor.acquire(config,2)
    assert len(first)==2
    second=supervisor.acquire(config,2)
    assert len(second)==2
    assert {g['uuid'] for g,h in first}.isdisjoint({g['uuid'] for g,h in second})
    for gpu,handle in first+second:
        handle.close()


def test_phase_progress_preserves_passed_gpu_checks(tmp_path):
    (tmp_path/'reports').mkdir()
    assert supervisor.next_phase(tmp_path,'dense_ae')=='tests'
    (tmp_path/'reports/dense_ae.gpu_tests_passed.json').write_text('{"status":"passed"}')
    assert supervisor.next_phase(tmp_path,'dense_ae')=='smoke'
    (tmp_path/'reports/dense_ae.gpu_smoke.json').write_text('{"status":"passed"}')
    assert supervisor.next_phase(tmp_path,'dense_ae')=='train'
