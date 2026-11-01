#!/usr/bin/env python3
"""Persistent five-node joint DMD jobs, with owned-unit cleanup and GPU telemetry.

Run launch from the immutable shared source snapshot. All paths must be visible
on every node. This tool does not copy source, change data, or stop other jobs.
"""
import argparse
import concurrent.futures
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
import uuid

NODES = ['192.168.0.4', '192.168.0.9', '192.168.0.7', '192.168.0.8', '192.168.0.6']
VENV = '/home/param/vansh-video-models/.venv'
GPU_FIELDS = ['timestamp', 'index', 'uuid', 'utilization.gpu', 'utilization.memory',
              'power.draw', 'power.limit', 'memory.used', 'memory.total', 'temperature.gpu']
PROPERTIES = ['LoadState', 'ActiveState', 'SubState', 'Result', 'ExecMainStatus',
              'ExecMainCode', 'MainPID', 'ControlGroup', 'Description', 'User']


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def ssh(node, command, timeout=45, accepted=(0,)):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', node):
        raise ValueError('Invalid SSH hostname')
    environment = dict(os.environ)
    environment.pop('SSH_AUTH_SOCK', None)
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=8',
        '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2', f'param@{node}',
        shlex.join(command)], text=True, capture_output=True, timeout=timeout, env=environment)
    if result.returncode not in accepted:
        raise RuntimeError(f'{node}: {result.stderr.strip() or result.stdout.strip()}')
    return result.stdout


def controller_node(nodes):
    """Choose the allocated node on which the trusted SSH client is running."""
    addresses = {socket.gethostname(), socket.getfqdn()}
    for node in nodes:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect((node, 9))
                addresses.add(probe.getsockname()[0])
        except OSError:
            pass
    for node in nodes:
        if node in addresses:
            return node
    raise ValueError('Launcher is not an allocated node; set --supervisor-node explicitly')


def controller_preflight(nodes, source, config):
    """Run from the controller with no SSH agent, before starting any workers."""
    check = ('import json,os,pathlib,subprocess,sys; '
             'paths=sys.argv[1:]; '
             'missing=[p for p in paths if not os.access(p,os.R_OK)]; '
             'assert not missing, missing; '
             f'assert os.access({VENV + "/bin/torchrun"!r},os.X_OK); '
             'subprocess.run(["sudo","-n","true"],check=True); '
             'linger=subprocess.check_output(["loginctl","show-user","param","--property=Linger","--value"],text=True).strip(); '
             'assert linger=="yes", "Persistent training requires param Linger=yes to protect shared memory across SSH logout; run sudo loginctl enable-linger param"; '
             'manager="user@"+str(os.getuid())+".service"; '
             'active=subprocess.run(["systemctl","is-active",manager],text=True,capture_output=True); '
             'assert active.returncode==0 and active.stdout.strip()=="active", "Persistent training requires active "+manager+" after enabling param lingering"; '
             'print(json.dumps({"ready":True,"uid":os.getuid(),"linger":linger,"user_manager":manager,"user_manager_active":True}))')
    entries = [{'node': node, 'unit': 'preflight'} for node in nodes]
    results = parallel(entries, lambda entry: {'node': entry['node'],
        **json.loads(ssh(entry['node'], [VENV + '/bin/python', '-c', check,
            str(Path(source) / 'scripts/joint_cluster.py'), str(Path(source) / 'run_dmd_joint.sh'), config]))})
    if any(item.get('error') or not item.get('ready') for item in results):
        raise RuntimeError('Controller cannot manage every worker without an SSH agent: ' + json.dumps(results))
    return results


def parallel(items, operation):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(items)) as pool:
        futures = [pool.submit(operation, item) for item in items]
        results = []
        for item, future in zip(items, futures):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({'node': item['node'], 'unit': item['unit'], 'error': str(exc)})
        return results


def load_job(path):
    path = Path(path).resolve()
    job = json.loads(path.read_text())
    if job.get('schema') != 'joint-cluster-v1' or job.get('user') != 'param':
        raise ValueError('Not a joint-cluster job manifest')
    if not re.fullmatch(r'[0-9a-f]{32}', job['id']):
        raise ValueError('Invalid job ownership token')
    for entry in [*job['workers'], job['supervisor']]:
        if not entry['unit'].startswith(f'param-joint-{job["id"]}-'):
            raise ValueError('Unit does not belong to this job')
        if entry['description'] != f'joint-cluster {job["id"]} {entry["role"]}':
            raise ValueError('Unit ownership description differs')
    return job


def unit_status(entry):
    output = ssh(entry['node'], ['systemctl', 'show', entry['unit'],
        *[f'--property={key}' for key in PROPERTIES]], accepted=(0, 1))
    values = dict(line.split('=', 1) for line in output.splitlines() if '=' in line)
    owned = values.get('Description') == entry['description'] and values.get('User') == 'param'
    return {'node': entry['node'], 'unit': entry['unit'], 'owned': owned, **values}


def stop_unit(entry):
    status = unit_status(entry)
    if status.get('LoadState') == 'not-found':
        return dict(status, stopped=True)
    if not status['owned']:
        raise RuntimeError(f'Refusing to stop unit with unmatched ownership: {entry["unit"]}')
    ssh(entry['node'], ['sudo', '-n', 'systemctl', 'stop', entry['unit']], timeout=150)
    return dict(status, stopped=True)


def start_unit(job, entry, command):
    log = str(Path(job['directory']) / entry['log'])
    # The system service persists without SSH, while preflight requires user
    # lingering so logind does not clean up its shared memory after logout.
    # KillMode owns the launcher, all workers, and telemetry together.
    ssh(entry['node'], ['sudo', '-n', 'systemd-run', '--quiet', f'--unit={entry["unit"]}',
        '--uid=param', '--gid=param', f'--description={entry["description"]}',
        '--property=Type=exec', '--property=RemainAfterExit=yes',
        '--property=KillMode=control-group', '--property=TimeoutStopSec=120',
        '--property=Restart=no', '--property=LimitNOFILE=1048576',
        f'--property=WorkingDirectory={job["source"]}',
        f'--property=StandardOutput=append:{log}', f'--property=StandardError=append:{log}',
        '--setenv=PYTHONUNBUFFERED=1', *command])
    status = unit_status(entry)
    if not status['owned']:
        raise RuntimeError(f'New service ownership could not be verified: {entry["unit"]}')
    return status


def launch(args):
    source, config, directory = (Path(value).resolve() for value in (args.source, args.config, args.job_dir))
    for path in (source / 'run_dmd_joint.sh', source / 'scripts/joint_cluster.py', config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if source / 'scripts/joint_cluster.py' != Path(__file__).resolve():
        raise ValueError('Invoke joint_cluster.py from the source snapshot being launched')
    if len(set(args.nodes)) != len(args.nodes) or args.slots < 1:
        raise ValueError('Nodes must be distinct and slots positive')
    if not 1 <= args.port <= 65535:
        raise ValueError('Invalid rendezvous port')
    if args.mode == 'benchmark' and not 0 <= args.benchmark_skip < args.benchmark:
        raise ValueError('Benchmark requires 0 <= skip < updates')
    environment = {'NCCL_SOCKET_IFNAME': 'eth0', 'GLOO_SOCKET_IFNAME': 'eth0',
                   'NCCL_DEBUG': 'WARN', 'TORCHINDUCTOR_COMPILE_THREADS': '2'}
    for item in args.env:
        key, value = item.split('=', 1)
        if not (key.startswith(('NCCL_', 'GLOO_', 'TORCH_', 'TRITON_', 'OMP_', 'PYTORCH_', 'UCX_', 'FI_'))
                or key in ('CUDA_MODULE_LOADING', 'CUDA_DEVICE_MAX_CONNECTIONS')):
            raise ValueError(f'Environment key is outside the runtime allowlist: {key}')
        environment[key] = value
    identity = uuid.uuid4().hex
    supervisor_node = args.supervisor_node or controller_node(args.nodes)
    if supervisor_node not in args.nodes:
        raise ValueError('The supervisor must run on one of the allocated nodes')
    def entry(role, node):
        return {'role': role, 'node': node, 'unit': f'param-joint-{identity}-{role}.service',
                'description': f'joint-cluster {identity} {role}', 'log': f'{role}.log'}
    job = {'schema': 'joint-cluster-v1', 'id': identity, 'user': 'param', 'created_at': now(),
        'source': str(source), 'config': str(config), 'directory': str(directory), 'mode': args.mode,
        'nodes': args.nodes, 'slots': args.slots, 'world_size': len(args.nodes) * args.slots,
        'master_addr': args.nodes[0], 'master_port': args.port, 'environment': environment,
        'benchmark': args.benchmark, 'benchmark_skip': args.benchmark_skip, 'overrides': args.set,
        'workers': [entry(f'node-{rank}', node) for rank, node in enumerate(args.nodes)],
        'supervisor': entry('supervisor', supervisor_node),
        'sha256': {str(path.relative_to(source)) if path.is_relative_to(source) else str(path):
                   hashlib.sha256(path.read_bytes()).hexdigest() for path in
                   (source / 'learn_dmd_joint.py', source / 'scripts/joint_cluster.py', config)}}
    if args.dry_run:
        print(json.dumps(job, indent=2))
        return
    # Use the exact host and authentication mode of the persistent supervisor.
    # Do this before starting even one torchrun process: a healthy controller is
    # required for failure cleanup, and trusted host keys are never bypassed.
    preflight = json.loads(ssh(supervisor_node, [VENV + '/bin/python',
        str(source / 'scripts/joint_cluster.py'), '_preflight', '--nodes', *args.nodes,
        '--source', str(source), '--config', str(config)], timeout=75))
    directory.mkdir(parents=True, exist_ok=False)
    manifest = directory / 'job.json'
    write_json(manifest, job)
    write_json(directory / 'preflight.json', {'at': now(), 'controller': supervisor_node, 'workers': preflight})
    script = str(source / 'scripts/joint_cluster.py')
    try:
        receipts = parallel(job['workers'], lambda worker: start_unit(job, worker,
            [VENV + '/bin/python', script, '_node', str(manifest), '--rank', str(job['workers'].index(worker))]))
        write_json(directory / 'launch.json', {'at': now(), 'workers': receipts})
        if any(item.get('error') or item.get('ActiveState') == 'failed' for item in receipts):
            raise RuntimeError('At least one node failed to start; see launch.json')
        supervisor = start_unit(job, job['supervisor'],
            [VENV + '/bin/python', script, '_supervise', str(manifest)])
        write_json(directory / 'launch.json', {'at': now(), 'workers': receipts, 'supervisor': supervisor})
    except BaseException as exc:
        write_json(directory / 'abort.json', {'job_id': job['id'], 'at': now(), 'reason': 'launch_failure'})
        cleanup = parallel([job['supervisor'], *job['workers']], stop_unit)
        write_json(directory / 'launch.failed.json', {'at': now(), 'error': str(exc), 'cleanup': cleanup})
        raise
    print(json.dumps({'launched': True, 'manifest': str(manifest), 'world_size': job['world_size'],
                      'mode': job['mode'], 'units': receipts, 'supervisor': supervisor}))


def node_run(args):
    job = load_job(args.manifest)
    rank = args.rank
    entry = job['workers'][rank]
    directory = Path(job['directory'])
    environment = dict(os.environ, **job['environment'], NNODES=str(len(job['nodes'])),
        SLOTS=str(job['slots']), NODE_RANK=str(rank), MASTER_ADDR=job['master_addr'],
        MASTER_PORT=str(job['master_port']), TORCHRUN=VENV + '/bin/torchrun')
    command = ['bash', str(Path(job['source']) / 'run_dmd_joint.sh'), job['config']]
    for value in job['overrides']:
        command.extend(['--set', value])
    if job['mode'] == 'benchmark':
        command.extend(['--set', 'wandb.enabled=false'])
        command.extend(['--benchmark', str(job['benchmark']), '--benchmark-skip', str(job['benchmark_skip'])])
    processes = []
    terminating = False
    def signal_process(process, signum):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass
    def terminate(signum, frame):
        nonlocal terminating
        terminating = True
        for process in processes:
            if process.poll() is None:
                signal_process(process, signal.SIGTERM)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    started = now()
    returncode = 1
    with (directory / f'node-{rank}.gpu.csv').open('w') as telemetry:
        csv.writer(telemetry).writerow(GPU_FIELDS)
        telemetry.flush()
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=' + ','.join(GPU_FIELDS),
            '--format=csv,noheader,nounits', '--loop=2'], stdout=telemetry, stderr=sys.stderr,
            start_new_session=True)
        processes.append(monitor)
        try:
            trainer = subprocess.Popen(command, cwd=job['source'], env=environment, start_new_session=True)
            processes.append(trainer)
            if terminating:
                signal_process(trainer, signal.SIGTERM)
            write_json(directory / f'node-{rank}.started.json', {'job_id': job['id'], 'started_at': started,
                'wrapper_pid': os.getpid(), 'trainer_pid': trainer.pid, 'telemetry_pid': monitor.pid,
                'unit': entry['unit'], 'command': command})
            while trainer.poll() is None:
                if (directory / 'abort.json').exists():
                    terminate(signal.SIGTERM, None)
                    try:
                        trainer.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        signal_process(trainer, signal.SIGKILL)
                    break
                time.sleep(1)
            returncode = trainer.wait()
        finally:
            for process in processes:
                if process.poll() is None:
                    signal_process(process, signal.SIGTERM)
            for process in processes:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    signal_process(process, signal.SIGKILL)
                    process.wait()
            if terminating and returncode == 0:
                returncode = 143
            write_json(directory / f'node-{rank}.complete.json', {'job_id': job['id'], 'started_at': started,
                'completed_at': now(), 'returncode': returncode, 'terminated': terminating,
                'telemetry_returncode': monitor.returncode})
    return returncode


def supervise(args):
    job = load_job(args.manifest)
    directory = Path(job['directory'])
    misses = {entry['unit']: 0 for entry in job['workers']}
    while True:
        statuses = parallel(job['workers'], unit_status)
        failed, complete = [], []
        for status in statuses:
            unit = status['unit']
            misses[unit] = misses[unit] + 1 if status.get('error') else 0
            if misses[unit] >= 3 or (not status.get('error') and
                    (not status.get('owned') or status.get('ActiveState') in ('failed', 'inactive'))):
                failed.append(status)
            complete.append(status.get('owned') and status.get('SubState') == 'exited'
                            and status.get('ExecMainStatus') == '0')
        write_json(directory / 'status.json', {'at': now(), 'workers': statuses})
        if failed:
            write_json(directory / 'abort.json', {'job_id': job['id'], 'at': now(), 'reason': 'node_failure'})
            cleanup = parallel(job['workers'], stop_unit)
            while any(item.get('error') for item in cleanup):
                write_json(directory / 'cleanup.pending.json', {'at': now(), 'workers': cleanup})
                time.sleep(10)
                cleanup = parallel(job['workers'], stop_unit)
            write_json(directory / 'job.complete.json', {'job_id': job['id'], 'at': now(),
                'success': False, 'failed': failed, 'cleanup': cleanup})
            return 1
        if all(complete):
            write_json(directory / 'job.complete.json', {'job_id': job['id'], 'at': now(),
                'success': True, 'workers': statuses})
            return 0
        time.sleep(10)


def summarize(args):
    job = load_job(args.manifest)
    observations = []
    for rank, entry in enumerate(job['workers']):
        path = Path(job['directory']) / f'node-{rank}.gpu.csv'
        if not path.exists():
            continue
        with path.open() as handle:
            for row in csv.DictReader(handle):
                try:
                    timestamp = datetime.strptime(row['timestamp'].strip(), '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
                    observations.append((timestamp, rank, int(row['index']),
                        float(row['utilization.gpu']), float(row['power.draw']), float(row['memory.used'])))
                except (ValueError, TypeError, KeyError, AttributeError):
                    continue
    cutoff = max((row[0] for row in observations), default=0) - args.last_seconds if args.last_seconds else float('-inf')
    if args.since:
        cutoff = max(cutoff, datetime.fromisoformat(args.since.replace('Z', '+00:00')).timestamp())
    groups = {}
    for row in observations:
        if row[0] >= cutoff:
            groups.setdefault((row[1], row[2]), []).append(row)
    gpus = [{'node': job['nodes'][rank], 'gpu': gpu, 'samples': len(rows),
        'mean_utilization_percent': statistics.mean(row[3] for row in rows),
        'mean_power_watts': statistics.mean(row[4] for row in rows),
        'peak_memory_gib': max(row[5] for row in rows) / 1024}
        for (rank, gpu), rows in sorted(groups.items())]
    result = {'job_id': job['id'], 'gpus_observed': len(gpus), 'gpus_expected': job['world_size'],
        'samples': sum(row['samples'] for row in gpus), 'gpus': gpus,
        'mean_gpu_utilization_percent': statistics.mean(row['mean_utilization_percent'] for row in gpus) if gpus else None,
        'mean_gpu_power_watts': statistics.mean(row['mean_power_watts'] for row in gpus) if gpus else None,
        'last_seconds': args.last_seconds, 'since': args.since}
    if args.output:
        write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    start = commands.add_parser('launch')
    start.add_argument('--source', '--repo', dest='source', required=True)
    for name in ('config', 'job-dir'):
        start.add_argument('--' + name, required=True)
    start.add_argument('--mode', choices=('benchmark', 'train'), required=True)
    start.add_argument('--nodes', nargs='+', default=NODES)
    start.add_argument('--supervisor-node', help='allocated controller with trusted non-agent SSH to every worker')
    start.add_argument('--slots', type=int, default=8)
    start.add_argument('--port', type=int, default=29931)
    start.add_argument('--benchmark', type=int, default=6)
    start.add_argument('--benchmark-skip', type=int, default=2)
    start.add_argument('--env', action='append', default=[])
    start.add_argument('--set', action='append', default=[])
    start.add_argument('--dry-run', action='store_true')
    preflight = commands.add_parser('_preflight')
    preflight.add_argument('--nodes', nargs='+', required=True)
    preflight.add_argument('--source', required=True)
    preflight.add_argument('--config', required=True)
    for name in ('status', 'stop', 'summarize', '_node', '_supervise'):
        command = commands.add_parser(name)
        command.add_argument('manifest')
        if name == '_node':
            command.add_argument('--rank', type=int, required=True)
        if name == 'summarize':
            command.add_argument('--last-seconds', type=float, default=0)
            command.add_argument('--since')
            command.add_argument('--output')
    args = parser.parse_args()
    if args.command == 'launch':
        launch(args)
    elif args.command == '_preflight':
        print(json.dumps(controller_preflight(args.nodes, args.source, args.config)))
    elif args.command == '_node':
        return node_run(args)
    elif args.command == '_supervise':
        return supervise(args)
    elif args.command == 'summarize':
        summarize(args)
    else:
        job = load_job(args.manifest)
        if args.command == 'status':
            print(json.dumps({'workers': parallel(job['workers'], unit_status),
                              'supervisor': unit_status(job['supervisor'])}, indent=2))
        else:
            write_json(Path(job['directory']) / 'abort.json', {'job_id': job['id'], 'at': now(), 'reason': 'requested_stop'})
            stop_unit(job['supervisor'])
            result = {'at': now(), 'stopped': parallel(job['workers'], stop_unit)}
            write_json(Path(job['directory']) / 'stop.json', result)
            print(json.dumps(result, indent=2))
            return int(any(row.get('error') for row in result['stopped']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
