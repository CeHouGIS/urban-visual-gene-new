"""CPU scheduling checks; no neural inference or real GPU calls."""
import json
import pytest
import launch


def test_upstream_gate_never_substitutes_incomplete_preview(tmp_path,monkeypatch):
    upstream=tmp_path/'upstream';upstream.mkdir()
    (upstream/'progress.json').write_text(json.dumps(dict(status='running',stage='sae')))
    queue=object.__new__(launch.Queue)
    queue.c=dict(base=dict(root=str(upstream)),dictionary_sizes=[2048,4096])
    events=[]
    queue.stopping=lambda:len(events)>0
    queue.status=lambda state,stage,**extra:events.append((state,stage))
    monkeypatch.setattr(launch.time,'sleep',lambda _:None)
    assert queue.upstream_gate() is False
    assert events==[('waiting_for_upstream','upstream_gate')]


def test_upstream_complete_requires_both_final_files(tmp_path,monkeypatch):
    upstream=tmp_path/'upstream';upstream.mkdir()
    (upstream/'progress.json').write_text(json.dumps(dict(status='complete')))
    queue=object.__new__(launch.Queue)
    queue.c=dict(base=dict(root=str(upstream)),dictionary_sizes=[2048,4096])
    queue.stopping=lambda:False
    monkeypatch.setattr(launch,'check_upstream_config',lambda _:None)
    with pytest.raises(RuntimeError,match='final 2048-atom checkpoint absent'):queue.upstream_gate()
    for m in queue.c['dictionary_sizes']:
        path=upstream/'runs'/f'atoms_{m}'/'checkpoint_final.pt';path.parent.mkdir(parents=True);path.touch()
    assert queue.upstream_gate() is True
