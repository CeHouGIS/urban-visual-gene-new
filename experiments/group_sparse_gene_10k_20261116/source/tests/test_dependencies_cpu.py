import copy
import json
from pathlib import Path
from dependencies import upstream_state

def config(tmp_path):return {'protocol':{'wait_for_upstream':True,'upstream_queue':str(tmp_path/'queue.json'),'upstream_postprocess':str(tmp_path/'post.json')}}

def test_waits_for_training_and_postprocessing(tmp_path):
 c=config(tmp_path);assert not upstream_state(c)['ready']
 Path(c['protocol']['upstream_queue']).write_text(json.dumps({'status':'complete'}))
 assert not upstream_state(c)['ready']
 Path(c['protocol']['upstream_postprocess']).write_text(json.dumps({'status':'complete'}))
 assert upstream_state(c)['ready']


def test_never_treats_failure_or_corrupt_state_as_completion(tmp_path):
 c=config(tmp_path)
 Path(c['protocol']['upstream_queue']).write_text(json.dumps({'status':'failed'}))
 assert upstream_state(c)['status']=='upstream_failed_or_stopped'
 Path(c['protocol']['upstream_queue']).write_text('{partial')
 assert not upstream_state(c)['ready']
