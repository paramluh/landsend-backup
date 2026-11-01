"""Snapshot a clean committed source revision and stage it without launching GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from common import EXTRA_ARGS, HELPERS, HOSTS, RUN, SSH, command, remote, sha, write_json

BASE = Path(__file__).resolve().parent
SOURCE = Path('/home/param/pose-worktrees/pose-opd-credit')
RECIPE = 'finetune/pose_opd_credit/recipe.teacher_partner.json'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--student-cut', type=int, required=True)
    parser.add_argument('--partner-cut', type=int, required=True)
    parser.add_argument('--replay-backend', choices=['sequential', 'block'], required=True)
    parser.add_argument('--local-only', action='store_true')
    args = parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True).strip():
        raise RuntimeError('Commit all source changes before immutable production preparation')
    artifact = BASE / ('snapshot-' + str(time.time_ns()))
    artifact.mkdir()
    archive = artifact / 'source.tar'
    names = sorted({name.decode() for name in subprocess.check_output(
        ['git', 'ls-files', '--cached', '-z'], cwd=SOURCE).split(b'\0')
        if name and '__pycache__' not in Path(name.decode()).parts and not name.endswith((b'.pyc', b'.pyo'))})
    manifest = {}
    config_bytes = None
    with tarfile.open(archive, 'w') as tar:
        for name in names:
            path = SOURCE / name
            if not path.exists():
                continue
            if not path.is_file() or path.is_symlink():
                raise RuntimeError('Unsupported source entry: ' + name)
            data = path.read_bytes()
            info = tar.gettarinfo(str(path), arcname=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            manifest[name] = hashlib.sha256(data).hexdigest()
            if name == RECIPE:
                config_bytes = data
    # Detect concurrent source edits instead of capturing a mixed revision.
    if any(sha(SOURCE / name) != digest for name, digest in manifest.items()):
        raise RuntimeError('Source changed while snapshotting; retry after writers finish')
    config = json.loads(config_bytes)
    expected = {'conversation_prefix_frames': 1500, 'exclude_short_rows': True,
                'effective_batch_per_rank': 6, 'rollout_batch_per_rank': 6,
                'physical_microbatch': 6, 'teacher_batch_per_rank': 6,
                'fixed_sequence_frames': 1636, 'rollout_samples_per_update': 4,
                'rollout_mode': 'teacher_partner', 'offload_optimizer_during_rollout': True,
                'epochs': 3, 'checkpoint_interval_steps': 60, 'keep_latest_optimizer_state': True,
                'loss_gradient_interval_steps': 10, 'learning_rate': 3e-6, 'pose_light_learning_rate': 8e-6,
                'partner_sampling': 'website_20261031', 'partner_audio_delay_samples': 72,
                'partner_prefill_min_frames': 38, 'partner_prefill_max_frames': 62,
                'conditioning_clip_length': 'source_max', 'pose_teacher_cfg_scale': 3.0,
                'audio_teacher': '/home/param/models/imported-weights/ppm-aloe-zoom-cotrain/export-253',
                'partner_checkpoint': '/home/param/models/imported-weights/ppm-aloe-zoom-cotrain/export-253',
                'student': '/home/param/models/imported-weights/ppm-aloe-poseduplex-raw-final/export-211',
                'actor_backbone_cut': args.student_cut,
                'partner_actor_backbone_cut': args.partner_cut,
                'partner_replay_backend': args.replay_backend}
    differences = {key: [value, config.get(key)] for key, value in expected.items() if config.get(key) != value}
    if differences:
        raise RuntimeError('Recipe does not match reviewed production settings: ' + json.dumps(differences))
    (artifact / 'recipe.teacher_partner.json').write_bytes(config_bytes)
    write_json(artifact / 'source-files.json', manifest)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    repo = RUN + '/code'
    spec = {'mode': 'three-epoch-production', 'hosts': HOSTS, 'world_size': 56,
            'master_addr': 'fdf9:c316:93ee:0:5f4:1538:2b0b:11d8', 'port': 29693,
            'repo': repo, 'run_dir': RUN, 'config': repo + '/' + RECIPE,
            'source_commit': commit, 'source_kind': 'clean-committed-working-tree',
            'archive_sha256': sha(archive), 'source_manifest_sha256': sha(artifact / 'source-files.json'),
            'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
            'checkpointing_enabled': True, 'normal_adam_initialization': True,
            'global_batch': 336, 'epochs': 3, 'expected_updates': 123,
            'expected_epoch_boundaries': [41,82,123], 'checkpoint_steps': [60,120,123],
            'model_output_dir': '/home/param/models/trained-weights/' + Path(RUN).name,
            'extra_args': EXTRA_ARGS, 'helper_sha256': {name: sha(BASE / name) for name in HELPERS}}
    specpath = artifact / 'production-launch.json'
    write_json(specpath, spec)
    write_json(artifact / 'commands.json', {host: command(spec, host) for host in HOSTS})
    if args.local_only:
        print(specpath)
        return
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
write_json(job/'STAGED.json',{'source_commit':spec['source_commit'],'config_sha256':spec['config_sha256'],'archive_sha256':spec['archive_sha256']})
print('staged '+str(root))
''' % str(job)
    def stage(host):
        remote(host, ['python3', '-c', 'from pathlib import Path; p=Path(' + repr(str(job))
                      + '); p.mkdir(parents=True,exist_ok=True); assert not any((p/n).exists() for n in '
                      + '("started-launch.json","supervisor.json","torchrun.json","READY.json","GO.json")), "Launch already started"'])
        files = [(archive, 'source.tar'), (artifact / 'source-files.json', 'source-files.json'), (specpath, 'spec.json')]
        files += [(BASE / name, name) for name in HELPERS]
        for src, name in files:
            subprocess.run(['rsync', '-a', '-e', shlex.join(SSH), str(src), host + ':' + str(job / name)],
                           check=True, capture_output=True)
        result = remote(host, ['bash', '-c', 'cd ' + shlex.quote(str(job)) + ' && python3 -c ' + shlex.quote(script)])
        return {'host': host, 'result': result.stdout.strip()}
    with ThreadPoolExecutor(max_workers=7) as pool:
        receipts = list(pool.map(stage, HOSTS))
    write_json(artifact / 'STAGED.json', {'time': time.time(), 'hosts': receipts})
    print(json.dumps({'spec': str(specpath), 'staged': receipts, 'started': False}, indent=2))


if __name__ == '__main__':
    main()
