"""Explicit production start after all-host lock and identity checks; default is check-only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from common import HOSTS, RUN, command, remote, validate_spec, write_json

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path)
    parser.add_argument('--check-only', action='store_true', help='Read-only checks; never start (default)')
    parser.add_argument('--start', action='store_true', help='Start the reviewed production run after all checks pass')
    parser.add_argument('--dry-command', action='store_true', help='Print commands locally without SSH')
    args = parser.parse_args()
    if args.check_only and args.start:
        parser.error('--check-only and --start are mutually exclusive')
    spec = json.loads(args.spec.read_text()); validate_spec(spec)
    if args.dry_command:
        print(json.dumps({host: command(spec, host) for host in HOSTS}, indent=2)); return
    job = Path(RUN) / 'supervisor'
    def check(host):
        staged = json.loads(remote(host, ['cat', job / 'spec.json']).stdout)
        if staged != spec:
            raise RuntimeError(host + ': staged spec differs')
        return {'host': host, 'preflight': json.loads(remote(host, ['python3', '-B', job / 'worker.py', job / 'spec.json', '--check']).stdout)}
    def capture_check(host):
        try:
            return {'ok': True, **check(host)}
        except Exception as error:
            return {'host': host, 'ok': False, 'error': str(error),
                    'stderr': getattr(error, 'stderr', None)}
    with ThreadPoolExecutor(max_workers=7) as pool:
        checks = list(pool.map(capture_check, HOSTS))
    report = {'all_host_read_only_preflight': checks, 'started': False}
    write_json(args.spec.parent / 'LATEST-PREFLIGHT.json', report)
    print(json.dumps(report, indent=2), flush=True)
    if any(not row['ok'] for row in checks):
        raise RuntimeError('Production start refused: one or more nodes failed read-only preflight')
    if args.check_only or not args.start:
        return
    marker = args.spec.parent / 'STARTED.json'
    with marker.open('x') as stream:
        json.dump({'time': time.time(), 'source_commit': spec['source_commit'], 'preflight': checks}, stream)
    script = '''import json,pathlib,subprocess,time
job=pathlib.Path(%r)
with (job/'started-launch.json').open('x') as marker:json.dump({'time':time.time()},marker)
with (job/'worker.log').open('ab',buffering=0) as log:
 p=subprocess.Popen(['python3','-B',str(job/'worker.py'),str(job/'spec.json')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 fields=pathlib.Path('/proc/%%d/stat'%%p.pid).read_text().rsplit(')',1)[1].split()
 receipt={'pid':p.pid,'start_ticks':int(fields[19]),'time':time.time()}
 (job/'detached-supervisor.json').write_text(json.dumps(receipt))
print(json.dumps(receipt))
''' % str(job)
    def launch(host):
        return {'host': host, **json.loads(remote(host, ['python3', '-c', script]).stdout)}
    with ThreadPoolExecutor(max_workers=7) as pool:
        launched = list(pool.map(launch, HOSTS))
    write_json(args.spec.parent / 'supervisors.json', launched)
    def ready(host):
        script = 'import json,pathlib,time; p=pathlib.Path(' + repr(str(job / 'READY.json')) + ')\nfor _ in range(300):\n if p.exists(): print(p.read_text()); break\n time.sleep(.2)\nelse: raise RuntimeError("Supervisor did not become ready")'
        return {'host': host, 'ready': json.loads(remote(host, ['python3', '-c', script]).stdout)}
    with ThreadPoolExecutor(max_workers=7) as pool:
        readiness = list(pool.map(ready, HOSTS))
    # Only now can a supervisor initialize CUDA: all seven own the exclusive locks.
    go = {'source_commit': spec['source_commit'], 'config_sha256': spec['config_sha256'],
          'archive_sha256': spec['archive_sha256'], 'time': time.time()}
    script = 'import json,os,pathlib; p=pathlib.Path(' + repr(str(job / 'GO.json')) + '); t=p.with_suffix(".tmp"); t.write_text(' + repr(json.dumps(go)) + '); os.replace(t,p)'
    try:
        with ThreadPoolExecutor(max_workers=7) as pool:
            list(pool.map(lambda host: remote(host, ['python3', '-c', script]), HOSTS))
    except BaseException as error:
        # A partially delivered GO must not leave a subset waiting for 56 ranks.
        # Signal only supervisors created by THIS invocation, after PID-reuse checks.
        cleanup = []
        for receipt in launched:
            stop = '''import os,pathlib,signal,json
pid=%d;expected=%d;p=pathlib.Path('/proc/%%d/stat'%%pid)
if p.exists():
 fields=p.read_text().rsplit(')',1)[1].split()
 if int(fields[19])!=expected:raise RuntimeError('PID start ticks differ; refusing signal')
 os.kill(pid,signal.SIGTERM)
print(json.dumps({'pid':pid,'checked_start_ticks':expected,'signal':'SIGTERM'}))
''' % (receipt['pid'], receipt['start_ticks'])
            try:
                result = remote(receipt['host'], ['python3', '-c', stop])
                cleanup.append({'host': receipt['host'], 'result': result.stdout.strip()})
            except Exception as cleanup_error:
                cleanup.append({'host': receipt['host'], 'error': repr(cleanup_error), 'receipt': receipt})
        write_json(args.spec.parent / 'PARTIAL-GO-FAILURE.json', {'error': repr(error), 'cleanup': cleanup})
        raise RuntimeError('Partial GO delivery failed; owned supervisors were signalled where reachable. Inspect PARTIAL-GO-FAILURE.json for unresolved hosts.') from error
    write_json(args.spec.parent / 'GO.json', {'go': go, 'ready': readiness})
    print(json.dumps({'started': launched, 'run_dir': RUN, 'logs': str(job / 'training.log')}, indent=2))

if __name__ == '__main__':
    main()
