#!/usr/bin/env python3
"""Verify and restore the private bulk backup into a chosen recovery directory."""
import argparse,hashlib,json,os,subprocess,tarfile
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('download',type=Path)
    ap.add_argument('--destination',type=Path,required=True)
    ap.add_argument('--host',help='Restore only this host; otherwise each host gets a separate subdirectory')
    ap.add_argument('--include-partial',action='store_true',help='Also restore verified archives from incomplete sets, with warnings')
    args=ap.parse_args();root=args.destination.resolve();root.mkdir(parents=True,exist_ok=True)
    sets=list(args.download.rglob('COMPLETE.json'))
    if args.include_partial:sets+=list(args.download.rglob('PARTIAL.json'))
    for complete in sorted(sets):
        data=json.loads(complete.read_text())
        if 'archives' not in data:continue
        host=complete.relative_to(args.download).parts[0]
        if args.host and args.host!=host:continue
        dest=root if args.host else root/host;dest.mkdir(parents=True,exist_ok=True)
        if data.get('state')!='complete':print('WARNING: partial archive set',complete,'errors',data.get('errors'),flush=True)
        for item in data['archives']:
            archive=complete.parent/item['archive']
            with archive.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest!=item['sha256']:raise RuntimeError('SHA256 mismatch: '+str(archive))
            print('Restoring',archive,flush=True)
            process=subprocess.Popen(['zstd','-q','-d','-c',str(archive)],stdout=subprocess.PIPE)
            with tarfile.open(fileobj=process.stdout,mode='r|') as tar:
                for member in tar:
                    path=(dest/member.name).resolve()
                    if dest not in path.parents:raise RuntimeError('Unsafe archive path: '+member.name)
                    tar.extract(member,dest,filter='data')
            process.stdout.close()
            if process.wait()!=0:raise RuntimeError('zstd failed: '+str(archive))
        aliases=complete.parent/'manifests/aliases.json'
        if aliases.exists():
            for item in json.loads(aliases.read_text()):
                source=(dest/item['target']).resolve();target=(dest/item['path']).resolve()
                if dest not in source.parents or dest not in target.parents:raise RuntimeError('Unsafe alias')
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():os.link(source,target)
    print('Restore finished:',root)

if __name__=='__main__':main()
