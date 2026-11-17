"""Upgrade the already-running queue after its original supervisor exits.

    Does not interrupt workers, change leases, or run another model concurrently.
"""
import argparse
import json
import os
from pathlib import Path
import time


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--old-pid',type=int,required=True)
    args=p.parse_args()
    c=json.loads(Path(args.config).read_text());root=Path(c['root'])
    while alive(args.old_pid):
        if (root/'STOP_REQUESTED').exists(): raise SystemExit(0)
        time.sleep(15)
    if (root/'reports/training_complete.json').exists() and not (root/'STOP_REQUESTED').exists():
        (root/'supervisor.pid').write_text(str(os.getpid())+'\n')
        os.execv(c['python'],[c['python'],str(root/'source/supervisor.py'),'--config',args.config])
