"""Durable continuation queue, gated on final upstream completion and GPU leases."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from common import config,now,write_json,fingerprint,check_upstream_config

STOP=False


def stop(*_):
    global STOP
    STOP=True


def read_json(path):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}


def idle_gpus():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=15)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid',
        '--format=csv,noheader'],text=True,timeout=15)
    busy={s.strip() for s in apps.splitlines()}
    result=[]
    for line in output.splitlines():
        index,uuid,memory,usage=[s.strip() for s in line.split(',')]
        if uuid not in busy and float(memory)<1024 and float(usage)<10:
            result.append(dict(index=int(index),uuid=uuid))
    return result


def snapshot(c):
    """Read canonical NAS code once; workers import the pinned local snapshot."""
    root=Path(c['root']);runtime=Path(c['local'])/'runtime';runtime.mkdir(parents=True,exist_ok=True)
    files={}
    for label,source in [('downstream',Path(c['source'])),('upstream',Path(c['base']['source']))]:
        target=runtime/label;target.mkdir(exist_ok=True)
        for path in source.glob('*.py'):
            data=path.read_bytes();(target/path.name).write_bytes(data)
            files[label+'/'+path.name]=hashlib.sha256(data).hexdigest()
    tests=runtime/'tests';tests.mkdir(exist_ok=True)
    for path in (Path(c['source']).parent/'tests').glob('*.py'):
        data=path.read_bytes();(tests/path.name).write_bytes(data)
        files['tests/'+path.name]=hashlib.sha256(data).hexdigest()
    write_json(root/'reports/source_manifest.json',dict(files=files,config_fingerprint=fingerprint(c),updated_utc=now()))
    return runtime


class Queue:
    def __init__(self,c):
        self.c=c;self.root=Path(c['root']);self.running={};self.renderer=None;self.rendered=None
        self.own=(self.root/'queue.lock').open('a+')
        fcntl.flock(self.own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_upstream_config(c)
        self.runtime=snapshot(c)
        self.env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            PYTHONPATH=str(self.runtime/'downstream')+os.pathsep+str(self.runtime/'upstream'),
            UVG_DOWNSTREAM_CONFIG=str(Path(c['source']).parent/'configs/experiment.json'),
            TMPDIR=c['temporary_directory'],MPLCONFIGDIR=c['temporary_directory']+'/matplotlib',
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
            PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')

    def stopping(self):return STOP or (self.root/'STOP_REQUESTED').exists()

    def status(self,state,stage,**extra):
        write_json(self.root/'progress.json',dict(status=state,stage=stage,supervisor_pid=os.getpid(),
            max_gpus=self.c['max_gpus'],workers={n:dict(pid=v['process'].pid,gpu=v['gpu'])
            for n,v in self.running.items()},updated_utc=now(),**extra))

    def upstream_gate(self):
        while not self.stopping():
            upstream=read_json(Path(self.c['base']['root'])/'progress.json')
            if upstream.get('status')=='complete':
                check_upstream_config(self.c)
                for m in self.c['dictionary_sizes']:
                    if not (Path(self.c['base']['root'])/'runs'/f'atoms_{m}'/'checkpoint_final.pt').exists():
                        raise RuntimeError(f'Upstream complete but final {m}-atom checkpoint absent')
                return True
            details={str(m):read_json(Path(self.c['base']['root'])/'runs'/f'atoms_{m}'/'progress.json')
                     for m in self.c['dictionary_sizes']}
            state='waiting_for_upstream'
            if upstream.get('status') in ['stopped','failed']:state='waiting_for_upstream_recovery'
            self.status(state,'upstream_gate',upstream=upstream,upstream_models=details)
            time.sleep(5)
        return False

    def complete(self,name):
        receipt=read_json(self.root/'reports/jobs'/f'{name}.json')
        if receipt.get('config_fingerprint')!=fingerprint(self.c) or receipt.get('status')!='complete':return False
        if name.startswith('export_'):
            m=name.split('_')[-1]
            return (Path(self.c['local'])/'exports'/f'atoms_{m}'/'validation.codes.npy').exists()
        if name.startswith('coding_'):
            m=name.split('_')[-1]
            return all((Path(self.c['local'])/'exports'/f'atoms_{m}'/f'decoded_codes_{i:04d}.npy').exists()
                       for i in self.c['arithmetic_validation_indices'])
        return True

    def command(self,script,args):
        if script=='pytest':return [self.c['base']['python'],'-m','pytest','-q',str(self.runtime/'tests/test_gpu.py'),
            '-p','no:cacheprovider','--junitxml='+str(self.root/'reports/gpu_contracts.xml')]
        return [self.c['base']['python'],str(self.runtime/'downstream'/script)]+args

    def start(self,job,gpu=None,handle=None):
        name,script,args,needs_gpu=job
        log=(Path(self.c['local'])/'logs'/f'{name}.log').open('a')
        try:
            process=subprocess.Popen(self.command(script,args),env=dict(self.env,CUDA_VISIBLE_DEVICES=gpu['uuid'] if gpu else ''),
                cwd=self.c['temporary_directory'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                start_new_session=True,pass_fds=(handle.fileno(),) if handle else ())
        except BaseException:
            log.close()
            if handle:handle.close()
            raise
        self.running[name]=dict(process=process,gpu=gpu,handle=handle,log=log)
        print(json.dumps(dict(event='started',job=name,pid=process.pid,gpu=gpu,updated_utc=now())),flush=True)

    def mirror_log(self,name):
        src=Path(self.c['local'])/'logs'/f'{name}.log'
        if src.exists():
            destination=self.root/'logs'/src.name
            temporary=destination.with_name(destination.name+'.partial')
            shutil.copyfile(src,temporary);temporary.replace(destination)

    def maybe_render(self):
        if self.renderer:
            child,log=self.renderer
            if child.poll() is None:return
            log.close();self.mirror_log('report_progress');self.renderer=None
            if child.returncode:print('Intermediate render failed; final report remains mandatory',flush=True)
        if not all((self.root/'exports'/f'atoms_{m}'/'READY.json').exists() for m in self.c['dictionary_sizes']):return
        step=read_json(self.root/'runs/decoder/validation_latest.json').get('step',0)
        if step==self.rendered:return
        self.rendered=step
        log=(Path(self.c['local'])/'logs/report_progress.log').open('a')
        child=subprocess.Popen(self.command('report.py',[]),env=dict(self.env,CUDA_VISIBLE_DEVICES=''),
            cwd=self.c['temporary_directory'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            start_new_session=True)
        self.renderer=child,log

    def run_stage(self,stage,jobs,render=False):
        pending=[j for j in jobs if not self.complete(j[0])]
        while pending or self.running:
            if self.stopping():return False
            for name,worker in list(self.running.items()):
                code=worker['process'].poll()
                if code is None:continue
                if worker['handle']:worker['handle'].close()
                worker['log'].close();del self.running[name];self.mirror_log(name)
                if code:raise RuntimeError(f'{name} failed ({code}); see logs/{name}.log')
                write_json(self.root/'reports/jobs'/f'{name}.json',dict(status='complete',
                    config_fingerprint=fingerprint(self.c),updated_utc=now()))
            for job in list(pending):
                if not job[3]:self.start(job);pending.remove(job)
            if any(j[3] for j in pending):
                try:available=idle_gpus()
                except (subprocess.SubprocessError,OSError):available=[]
                for gpu in available:
                    gpu_count=sum(v['gpu'] is not None for v in self.running.values())
                    if gpu_count>=self.c['max_gpus'] or not pending:break
                    job=next((j for j in pending if j[3]),None)
                    if job is None:break
                    handle=(Path(self.c['base']['shared_gpu_leases'])/(gpu['uuid']+'.lock')).open('a+')
                    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:handle.close();continue
                    try:still_idle={g['uuid'] for g in idle_gpus()}
                    except (subprocess.SubprocessError,OSError):still_idle=set()
                    if gpu['uuid'] not in still_idle:handle.close();continue
                    self.start(job,gpu,handle);pending.remove(job)
            if render:self.maybe_render()
            state='running' if self.running else 'waiting_for_gpu' if pending else 'stage_complete'
            self.status(state,stage,pending=[j[0] for j in pending])
            if pending or self.running:time.sleep(5)
        return True

    def cleanup(self):
        # Kill only this queue's own process groups; GPU lock FDs close after exit.
        children=[(v['process'],v['log']) for v in self.running.values()]
        if self.renderer:children.append(self.renderer)
        for child,_ in children:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for child,log in children:
            try:child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
            log.close()
        for name,worker in self.running.items():
            if worker['handle']:worker['handle'].close()
            self.mirror_log(name)
        self.running.clear();self.renderer=None

    def run(self):
        try:
            if not self.upstream_gate():self.status('stopped','upstream_gate');return
            stages=[('gpu_preflight',[('gpu_tests','pytest',[],True)]),
                ('export',[(f'export_{m}','export.py',['--atoms',str(m)],True) for m in self.c['dictionary_sizes']]),
                ('decode_and_coding',[('decoder','decode.py',[],True)]+
                    [(f'coding_{m}','coding.py',['--atoms',str(m)],False) for m in self.c['dictionary_sizes']]),
                ('stream_decode_verification',[('verify','verify.py',[],True)])]
            for stage,jobs in stages:
                if not self.run_stage(stage,jobs,render=stage=='decode_and_coding'):
                    self.cleanup();self.status('stopped',stage);return
            if self.renderer:
                child,log=self.renderer
                while child.poll() is None and not self.stopping():time.sleep(2)
                if self.stopping():self.cleanup();self.status('stopped','report');return
                log.close();self.renderer=None;self.mirror_log('report_progress')
            if not self.run_stage('report',[('report_final','report.py',[],False)]):
                self.cleanup();self.status('stopped','report');return
            self.status('complete','complete',result=str(self.root/'reports/RESULT_zh.md'))
        except BaseException as error:
            self.cleanup();self.status('failed','exception',error=repr(error));raise


if __name__=='__main__':
    c=config()
    for path in [Path(c['local'])/'logs',Path(c['root'])/'reports/jobs',Path(c['root'])/'logs',
                 Path(c['temporary_directory'])/'matplotlib']:
        path.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    Queue(c).run()
