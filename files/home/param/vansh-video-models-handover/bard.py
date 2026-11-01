"""
Moves finished checkpoints of a run to the private HF repo vanshjjw/<wandb run_name>, keeping
the run_N/stepXXXX layout, and deletes the local copies. Saves are sequential, so every step dir older than the
newest one is complete; the newest is left alone until the next save exists (--final takes it too, in one pass,
once training has ended). optim_* files are never touched: the trainer keeps only the latest optimizer state and
deletes the older one itself. A local file is deleted only once the repo's size and hash match it; anything that
fails is kept and retried at the next poll (with --final, kept for a later run).

usage: .venv/bin/python bard.py CONFIG [--final]
"""
import argparse
import hashlib
import shutil
import time
from pathlib import Path

import toml
from huggingface_hub import HfApi

POLL_S = 60
MIN_FREE_GIB = 100
KEEP = 'optim_'


def token():
    for line in Path(__file__).with_name('.env').read_text().splitlines():
        if line.startswith('VANSH_HF_TOKEN='):
            return line.split('=', 1)[1].strip()
    raise RuntimeError('VANSH_HF_TOKEN not in .env')


def step_dirs(output_dir):
    def order(d):
        return int(d.parent.name.removeprefix('run_')), int(d.name.removeprefix('step'))
    return sorted((d for d in output_dir.glob('run_*/step*') if d.is_dir()), key=order)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def matches(remote, path):
    if remote is None or remote.size != path.stat().st_size:
        return False
    if remote.lfs is not None:
        return remote.lfs.sha256 == sha256(path)
    data = path.read_bytes()
    return remote.blob_id == hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def offload(api, repo, output_dir, step_dir):
    files = [p for p in step_dir.iterdir() if p.is_file() and not p.name.startswith(KEEP)]
    if files:
        prefix = step_dir.relative_to(output_dir).as_posix()
        print(f'uploading {prefix}: {sorted(p.name for p in files)}', flush=True)
        api.upload_folder(repo_id=repo, folder_path=step_dir, path_in_repo=prefix,
                          ignore_patterns=[KEEP + '*'], commit_message=prefix)
        remote = {f.path: f for f in api.get_paths_info(repo, [f'{prefix}/{p.name}' for p in files])}
        for p in files:
            if matches(remote.get(f'{prefix}/{p.name}'), p):
                p.unlink()
            else:
                print(f'!!! {prefix}/{p.name} does not match the repo; kept', flush=True)
        print(f'uploaded {prefix}', flush=True)
    if not any(step_dir.iterdir()):
        step_dir.rmdir()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--final', action='store_true', help='one pass that also takes the newest step dir')
    args = parser.parse_args()
    config = toml.load(args.config)
    output_dir = Path(config['output_dir'])
    repo = f"vanshjjw/{config['wandb']['run_name']}"
    api = HfApi(token=token())
    api.create_repo(repo, private=True, exist_ok=True)
    print(f'watching {output_dir} -> {repo}', flush=True)
    while True:
        dirs = step_dirs(output_dir) if output_dir.exists() else []
        for step_dir in dirs if args.final else dirs[:-1]:
            try:
                offload(api, repo, output_dir, step_dir)
            except Exception as error:
                print(f'!!! {step_dir}: {type(error).__name__}: {error}; retrying next poll', flush=True)
        free = shutil.disk_usage(output_dir if output_dir.exists() else '/').free / 2**30
        if free < MIN_FREE_GIB:
            print(f'!!!!! LOW DISK: {free:.0f} GiB free', flush=True)
        if args.final:
            return
        time.sleep(POLL_S)


if __name__ == '__main__':
    main()
