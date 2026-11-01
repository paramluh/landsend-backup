"""Scoped five-node benchmark/training launcher with persistent telemetry."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

PYTHON = '/home/param/vansh-video-models/.venv/bin/python'
TORCHRUN = '/home/param/vansh-video-models/.venv/bin/torchrun'
HOSTS = ['192.168.0.8', '192.168.0.4', '192.168.0.9', '192.168.0.7', '192.168.0.6']
ROOT = Path('/mnt/paramfs/param/outputs/fullgt-performance-20261020-0505')
REPO = Path('/home/param/bimodal')


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def ssh(host, command, timeout=30):
    env = dict(os.environ)
    env.pop('SSH_AUTH_SOCK', None)
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
        'param@'+host, command], env=env, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{host}: {result.stderr[-1500:]}')
    return result.stdout.strip()


def start_ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def stop_local(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    receipt_path = Path(spec['directory'])/'process.json'
    if not receipt_path.exists():
        return {'stopped': False, 'reason': 'no process receipt'}
    receipt = json.loads(receipt_path.read_text())
    pid = receipt['pid']
    try:
        if start_ticks(pid) != receipt['start_ticks']:
            return {'stopped': False, 'reason': 'PID reused'}
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if not any(args):
            return {'stopped': False, 'reason': 'process exited and is awaiting reap'}
        if spec['config'].encode() not in args or b'learn_dmd_full_gt.py' not in args:
            raise RuntimeError('Refusing to stop an unverified process')
        os.killpg(pid, signal.SIGTERM)
        return {'stopped': True, 'pid': pid}
    except ProcessLookupError:
        return {'stopped': False, 'reason': 'already exited'}
    except FileNotFoundError:
        return {'stopped': False, 'reason': 'already exited'}


def worker(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    directory = Path(spec['directory'])
    env = dict(os.environ, **spec['env'])
    started = time.monotonic()
    monitor = child = None
    code, error = 1, None
    try:
        with (directory/'gpu.csv').open('w') as gpu, (directory/'train.log').open('w') as log:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw',
                '--format=csv,noheader,nounits', '--loop-ms=1000'], stdout=gpu, stderr=subprocess.DEVNULL)
            child = subprocess.Popen(spec['command'], cwd=spec['source'], env=env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(directory/'process.json', {'pid':child.pid,'start_ticks':start_ticks(child.pid),
                  'supervisor_pid':os.getpid(),'started_at':utc(),'command':spec['command']})
            code = child.wait()
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=45)
    finally:
        if monitor is not None and monitor.poll() is None:
            monitor.terminate()
            try:
                monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
        write(directory/'exit.json', {'returncode':code,'error':error,
              'seconds':time.monotonic()-started,'ended_at':utc()})


def stop_trial(trial):
    plan = json.loads((trial/'plan.json').read_text())
    def one(item):
        command = shlex.join([PYTHON, plan['controller'], 'stop-local', item['spec']])
        try:
            return {'host':item['host'],'result':ssh(item['host'],command)}
        except Exception as exc:
            return {'host':item['host'],'error':str(exc)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(one, plan['nodes']))
    write(trial/'stop.json', {'at':utc(),'results':results})
    return results


def watch(trial):
    plan = json.loads((trial/'plan.json').read_text())
    started = time.monotonic()
    while True:
        exits = [json.loads((Path(item['directory'])/'exit.json').read_text())
                 for item in plan['nodes'] if (Path(item['directory'])/'exit.json').exists()]
        if len(exits) == len(plan['nodes']):
            write(trial/'cluster.complete.json', {'exits':exits,'all_successful':all(x['returncode']==0 for x in exits),'at':utc()})
            return
        failure = any(item['returncode'] != 0 for item in exits)
        metrics = list((trial/'runs').glob('*/metrics.jsonl'))
        first_round = any(p.stat().st_size for p in metrics)
        timeout = plan['benchmark'] and not first_round and time.monotonic()-started > plan['startup_timeout_seconds']
        if failure or timeout:
            write(trial/'watchdog.json', {'at':utc(),'reason':'worker_failure' if failure else 'cold_start_timeout'})
            stop_trial(trial)
            # Allow torchrun to reap its own workers before final status.
            time.sleep(5)
        time.sleep(3)


def launch(args):
    if not args.name or Path(args.name).name != args.name:
        raise ValueError('Name must be one path component')
    trial = ROOT/args.name
    trial.mkdir(parents=True, exist_ok=False)
    source = trial/'code'
    source.mkdir()
    names = subprocess.check_output(['rg','--files','-g','*.py','-g','*.toml','-g','*.sh',
        '-g','requirements.txt'],cwd=REPO,text=True).splitlines()
    manifest = {}
    for name in names:
        if Path(name).parts[0] in ('artifacts','outputs'):
            continue
        origin, target = REPO/name, source/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin,target)
        manifest[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    write(source/'source-manifest.json', {'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'files':manifest,'created_at':utc()})
    config = str(source/'experiments/distillation/full-gt-nine-ref.toml')
    controller = str(source/'scripts/full_gt_cluster.py')
    nodes = []
    for rank, host in enumerate(HOSTS):
        directory = trial/f'node-{rank}'
        directory.mkdir()
        command = [TORCHRUN,'--nnodes=5','--nproc_per_node=8',f'--node_rank={rank}',
                   '--master_addr='+HOSTS[0],f'--master_port={args.port}',
                   'learn_dmd_full_gt.py','--config',config,'--expected-world-size','40']
        if not args.train:
            command += ['--benchmark',str(args.steps),'--benchmark-skip',str(args.skip)]
        for override in [f'output_dir={json.dumps(str(trial/"runs"))}',*args.set]:
            command += ['--set',override]
        env = {'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4',
            'NCCL_SOCKET_IFNAME':'eth0','GLOO_SOCKET_IFNAME':'eth0','NCCL_DEBUG':'WARN',
            'PYTORCH_ALLOC_CONF':'expandable_segments:True','TORCHINDUCTOR_COMPILE_THREADS':'8',
            'TORCHINDUCTOR_CACHE_DIR':'/tmp/param-fullgt-training-20261020/inductor',
            'TRITON_CACHE_DIR':'/tmp/param-fullgt-training-20261020/triton'}
        spec_path = directory/'spec.json'
        write(spec_path,{'source':str(source),'directory':str(directory),'config':config,'command':command,'env':env})
        nodes.append({'host':host,'rank':rank,'spec':str(spec_path),'directory':str(directory)})
    plan = {'name':args.name,'source':str(source),'controller':controller,'nodes':nodes,
        'benchmark':not args.train,'steps':args.steps,'skip':args.skip,
        'startup_timeout_seconds':args.timeout,'created_at':utc(),'overrides':args.set}
    write(trial/'plan.json',plan)
    def one(item):
        command = 'nohup '+shlex.join([PYTHON,controller,'worker',item['spec']])+' > '+shlex.quote(item['directory']+'/supervisor.log')+' 2>&1 < /dev/null & echo $!'
        return {'host':item['host'],'supervisor_pid':ssh(item['host'],command)}
    with (trial/'watcher.log').open('w') as output:
        watcher = subprocess.Popen([PYTHON,controller,'watch',str(trial)],stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    launched, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(one,item):item for item in nodes}
        for future in concurrent.futures.as_completed(futures):
            try:
                launched.append(future.result())
            except Exception as exc:
                errors.append({'host':futures[future]['host'],'error':str(exc)})
    write(trial/'launched.json',{'at':utc(),'nodes':launched,'watcher_pid':watcher.pid})
    if errors:
        write(trial/'launch-errors.json',errors)
        time.sleep(3)
        stop_trial(trial)
        raise RuntimeError(f'Launch failed: {errors}')
    print(json.dumps({'trial':str(trial),'watcher_pid':watcher.pid,'nodes':launched}))


def status(trial):
    plan = json.loads((trial/'plan.json').read_text())
    result = {'trial':str(trial),'nodes':[]}
    for item in plan['nodes']:
        directory = Path(item['directory'])
        entry = {'host':item['host']}
        for name in ('process','exit'):
            if (directory/(name+'.json')).exists():
                entry[name] = json.loads((directory/(name+'.json')).read_text())
        result['nodes'].append(entry)
    for name in ('watchdog','cluster.complete'):
        if (trial/(name+'.json')).exists():result[name]=json.loads((trial/(name+'.json')).read_text())
    result['runs']=[str(p) for p in (trial/'runs').glob('*')]
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('launch'); p.add_argument('--name',required=True);p.add_argument('--port',type=int,default=29880)
    p.add_argument('--steps',type=int,default=8);p.add_argument('--skip',type=int,default=2)
    p.add_argument('--timeout',type=int,default=660);p.add_argument('--train',action='store_true');p.add_argument('--set',action='append',default=[])
    for name in ('worker','stop-local','watch','status','stop'):
        sub.add_parser(name).add_argument('path',type=Path)
    args=parser.parse_args()
    if args.action=='launch':launch(args)
    elif args.action=='worker':worker(args.path)
    elif args.action=='stop-local':print(json.dumps(stop_local(args.path)))
    elif args.action=='watch':watch(args.path)
    elif args.action=='status':status(args.path)
    elif args.action=='stop':print(json.dumps(stop_trial(args.path)))
