"""Stage an immutable committed checkout and production spec; never launch GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time
from common import HELPERS, HOSTS, RUN, SSH, command, remote, sha, write_json

BASE = Path(__file__).resolve().parent
SOURCE = Path('/home/param/pose-worktrees/pose-opd-credit')
RECIPE = 'finetune/pose_opd_credit/recipe.raw.json'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True, help='Exact committed HEAD to stage')
    parser.add_argument('--local-only', action='store_true', help='Create auditable local artifacts without SSH')
    args = parser.parse_args()
    git = lambda *argv: subprocess.check_output(['git', *argv], cwd=SOURCE, text=True).strip()
    if git('status', '--porcelain=v1', '--untracked-files=all'):
        raise SystemExit('Refusing staging: source worktree must be clean and fully committed')
    commit = git('rev-parse', '--verify', args.commit + '^{commit}')
    if commit != git('rev-parse', 'HEAD'):
        raise SystemExit('Requested commit must be the current clean HEAD')
    artifact = BASE / commit
    artifact.mkdir(exist_ok=True)
    archive = artifact / 'source.tar'
    subprocess.run(['git', 'archive', '--format=tar', '--output', str(archive), commit], cwd=SOURCE, check=True)
    manifest = {}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.isdir():
                continue
            if not member.isfile() or member.name.startswith('/') or '..' in Path(member.name).parts:
                raise SystemExit('Unsupported archive member: ' + member.name)
            content = tar.extractfile(member).read()
            manifest[member.name] = hashlib.sha256(content).hexdigest()
            if member.name == RECIPE:
                config_bytes = content
    config = json.loads(config_bytes)
    expected = {'conversation_prefix_frames': 1500, 'exclude_short_rows': True, 'effective_batch_per_rank': 8,
                'rollout_batch_per_rank': 8, 'teacher_batch_per_rank': 8, 'fixed_sequence_frames': 1636,
                'checkpoint_interval_steps': 60, 'checkpoint_interval_seconds': None,
                'keep_latest_optimizer_state': True, 'epochs': 3}
    if any(config.get(key) != value for key, value in expected.items()):
        raise SystemExit('Committed recipe does not match the approved two-minute/global320 shape')
    (artifact / 'recipe.raw.json').write_bytes(config_bytes)
    write_json(artifact / 'source-files.json', manifest)
    repo = '/home/param/pose-worktrees/pose-opd-credit-production-' + commit
    spec = {'mode': 'production', 'hosts': HOSTS, 'world_size': 40,
            'master_addr': 'fdf9:c316:93ee:0:5f4:155d:dc43:d190', 'port': 29671,
            'repo': repo, 'run_dir': RUN, 'config': repo + '/' + RECIPE,
            'source_commit': commit, 'archive_sha256': sha(archive),
            'source_manifest_sha256': sha(artifact / 'source-files.json'),
            'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
            'checkpointing_enabled': True, 'normal_adam_initialization': True,
            'helper_sha256': {name: sha(BASE / name) for name in HELPERS}}
    specpath = artifact / 'production-launch.json'
    if specpath.exists() and json.loads(specpath.read_text()) != spec:
        raise SystemExit('This commit already has a different launch spec; refusing mutation')
    write_json(specpath, spec)
    write_json(artifact / 'commands.json', {host: command(spec, host) for host in HOSTS})
    if args.local_only:
        print(specpath); return
    job = Path(RUN) / 'supervisor'
    script = '''import json,os,pathlib,tarfile
from common import sha,verify_files,write_json
job=pathlib.Path(%r);spec=json.loads((job/'spec.json').read_text());root=pathlib.Path(spec['repo'])
if any((job/name).exists() for name in ('started-launch.json','supervisor.json','torchrun.json','READY.json','GO.json')):
 raise RuntimeError('Refusing to restage a started launch')
if sha(job/'source.tar')!=spec['archive_sha256'] or sha(job/'source-files.json')!=spec['source_manifest_sha256']:
 raise RuntimeError('Transferred source archive/manifest SHA mismatch')
if not root.exists():
 temp=root.with_name(root.name+'.staging-'+str(os.getpid()));temp.mkdir()
 with tarfile.open(job/'source.tar') as tar:tar.extractall(temp,filter='data')
 for path in sorted(temp.rglob('*'),reverse=True):path.chmod(path.stat().st_mode & ~0o222)
 temp.chmod(temp.stat().st_mode & ~0o222);os.rename(temp,root)
verify_files(spec,job)
write_json(job/'STAGED.json',{'source_commit':spec['source_commit'],'config_sha256':spec['config_sha256']})
print('staged '+str(root))
''' % str(job)
    def stage(host):
        remote(host, ['python3', '-c', 'from pathlib import Path; p=Path(' + repr(str(job)) + '); p.mkdir(parents=True,exist_ok=True); assert not any((p/n).exists() for n in ("started-launch.json","supervisor.json","torchrun.json","READY.json","GO.json")), "Launch already started"'])
        files = [(archive, 'source.tar'), (artifact / 'source-files.json', 'source-files.json'), (specpath, 'spec.json')]
        files += [(BASE / name, name) for name in HELPERS]
        for src, name in files:
            subprocess.run(['rsync', '-a', '-e', shlex.join(SSH), str(src), host + ':' + str(job / name)], check=True, capture_output=True)
        result = remote(host, ['bash', '-c', 'cd ' + shlex.quote(str(job)) + ' && python3 -c ' + shlex.quote(script)])
        return {'host': host, 'result': result.stdout.strip()}
    with ThreadPoolExecutor(max_workers=5) as pool:
        receipts = list(pool.map(stage, HOSTS))
    write_json(artifact / 'STAGED.json', {'time': time.time(), 'hosts': receipts})
    print(json.dumps({'spec': str(specpath), 'staged': receipts, 'started': False}, indent=2))

if __name__ == '__main__':
    main()
