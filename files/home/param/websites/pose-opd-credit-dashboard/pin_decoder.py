#!/usr/bin/env python3
"""Pin only the CPU sample service to the exact prepared production archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--commit',required=True)
    parser.add_argument('--artifact-root',type=Path,default=Path('/home/param/outputs/pose-opd-credit-launch'))
    parser.add_argument('--live-root',type=Path,default=Path('/home/param/outputs/pose-opd-credit-live'))
    args=parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}',args.commit):raise ValueError('Full commit SHA required')
    artifact=args.artifact_root/args.commit
    spec=json.loads((artifact/'production-launch.json').read_text())
    archive=artifact/'source.tar'
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    if spec['source_commit']!=args.commit or spec['archive_sha256']!=sha:raise ValueError('Prepared source identity mismatch')
    checkout=Path('/home/param/pose-worktrees')/('pose-opd-credit-decoder-'+args.commit)
    if not checkout.exists():
        temporary=checkout.with_name(checkout.name+'.tmp-'+str(os.getpid()));temporary.mkdir()
        with tarfile.open(archive) as tar:tar.extractall(temporary,filter='data')
        for path in sorted(temporary.rglob('*'),reverse=True):path.chmod(path.stat().st_mode & ~0o222)
        temporary.chmod(temporary.stat().st_mode & ~0o222);os.rename(temporary,checkout)
    manifest=json.loads((artifact/'source-files.json').read_text())
    for name,expected in manifest.items():
        if hashlib.sha256((checkout/name).read_bytes()).hexdigest()!=expected:raise ValueError('Pinned source file differs: '+name)
    template=(ROOT/'ops/pose-opd-credit-samples.service').read_text()
    template=template.replace('WorkingDirectory=/home/param/pose-worktrees/pose-opd-credit\n','WorkingDirectory='+str(checkout)+'\n')
    template=template.replace('Environment=PYTHONUNBUFFERED=1','Environment=PYTHONUNBUFFERED=1\nEnvironment=PYTHONDONTWRITEBYTECODE=1')
    template=template.replace('/home/param/outputs/pose-opd-credit-live/',str(args.live_root)+'/')
    unit=Path('/home/param/.config/systemd/user/pose-opd-credit-samples.service');unit.write_text(template)
    receipt={'source_commit':args.commit,'archive_sha256':sha,'checkout':str(checkout),'cpu_only':True}
    args.live_root.mkdir(parents=True,exist_ok=True)
    (args.live_root/'decoder-source.json').write_text(json.dumps(receipt,indent=2))
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','restart','pose-opd-credit-samples.service'],check=True)
    print(json.dumps(receipt))

if __name__=='__main__':main()
