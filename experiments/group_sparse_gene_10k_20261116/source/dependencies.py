"""Read-only dependency checks; never modify or stop an earlier experiment."""
import json
from pathlib import Path

def upstream_state(config):
 if not config['protocol'].get('wait_for_upstream',True):return {'ready':True,'status':'disabled'}
 records={}
 for key in ['upstream_queue','upstream_postprocess']:
  path=Path(config['protocol'][key])
  if not path.exists():records[key]={'status':'missing'};continue
  try:records[key]=json.loads(path.read_text())
  except (json.JSONDecodeError,OSError):records[key]={'status':'unreadable'}
 statuses=[x.get('status','unknown') for x in records.values()]
 failed=any(x.startswith('fail') or x in ['stopped','training_queue_failed_or_stopped','data_failed'] for x in statuses)
 return {'ready':all(x=='complete' for x in statuses),'status':'upstream_failed_or_stopped' if failed else 'ready' if all(x=='complete' for x in statuses) else 'waiting_for_previous_experiment','upstream':{k:{field:v[field] for field in ['status','updated_utc'] if field in v} for k,v in records.items()}}
