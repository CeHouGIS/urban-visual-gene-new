import math
import numpy as np
import pytest

from data import indices_for_step
from distributed_schedule import learning_rate, next_batch, rank_ids, sample_ids
from common import load_config
import supervisor_distributed as supervisor


def test_larger_batches_resume_exact_original_sample_stream_across_epoch():
    # Change batch size at a non-1024-aligned old checkpoint.
    offset=9938*256
    count=1000000
    full=sample_ids(offset,count,1024,42)
    original=np.concatenate([indices_for_step(9938+i,count,256,42) for i in range(4)])
    np.testing.assert_array_equal(full,original)
    left=rank_ids(offset,count,1024,42,0,2);right=rank_ids(offset,count,1024,42,1,2)
    assert len(left)==len(right)==512
    assert set(left).isdisjoint(right)
    np.testing.assert_array_equal(np.concatenate([left,right]),full)
    np.testing.assert_array_equal(sample_ids(999900,count,1024,42),
        np.concatenate([sample_ids(999900,count,100,42),sample_ids(1000000,count,924,42)]))


def test_total_sample_budget_has_no_fourfold_increase_or_discarded_tail():
    settings={'effective_batch_size':1024,'max_training_images':39063*256}
    for base_step in [9938,10188]:
        seen=base_step*256;steps=0
        while seen<settings['max_training_images']:
            batch=next_batch(seen,settings)
            assert 0<batch<=1024 and batch%2==0
            seen+=batch;steps+=1
        assert seen==10000128
        assert steps==math.ceil((10000128-base_step*256)/1024)


def test_lr_matches_original_cosine_at_transition_and_remains_continuous():
    t={'previous_effective_batch_size':256,'previous_max_updates':39063,
       'warmup_updates':1000,'learning_rate':2e-4,'minimum_learning_rate':2e-5}
    step=9938
    expected=2e-5+(2e-4-2e-5)*0.5*(1+math.cos(math.pi*(step-1000)/(39063-1000-1)))
    assert learning_rate(step*256,t)==pytest.approx(expected)
    assert learning_rate((39063-1)*256,t)==pytest.approx(2e-5)
    assert abs(learning_rate((step+4)*256,t)-expected)<1e-7


def test_partial_gpu_pair_is_released_and_no_distributed_worker_launches(tmp_path,monkeypatch):
    config={'root':str(tmp_path),'variants':['sparse_n256_s4','dense_ae'],
        'compute':{'gpus_per_variant':2,'max_gpus':4,'poll_seconds':10},
        'training':{'effective_batch_size':1024}}
    closed=[]
    class Handle:
        def close(self):closed.append(True)
    monkeypatch.setattr(supervisor,'acquire',lambda config,n:[({'uuid':'only-one'},Handle())])
    monkeypatch.setattr(supervisor,'launch',lambda *a:pytest.fail('Must not launch an incomplete pair'))
    class StopCheck(Exception):pass
    def finish(_):raise StopCheck()
    monkeypatch.setattr(supervisor.time,'sleep',finish)
    with pytest.raises(StopCheck):supervisor.supervise(config)
    assert closed==[True]
    import json
    assert json.loads((tmp_path/'queue_status.json').read_text())['status']=='waiting_for_gpu'


def test_config_accepts_authorized_four_gpu_and_checks_global_accumulation(tmp_path):
    import json
    c={'compute':{'gpu_only':True,'max_gpus':4,'gpus_per_variant':2},
       'training':{'effective_batch_size':1024,'micro_batch_size':128}}
    path=tmp_path/'config.json';path.write_text(json.dumps(c))
    assert load_config(path)==c
    c['training']['effective_batch_size']=1000;path.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='Global batch'):load_config(path)
