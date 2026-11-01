#!/usr/bin/env python3
"""Verify and restore the private bulk backup into a chosen recovery directory."""
import argparse,hashlib,json,os,subprocess,tarfile
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('download',type=Path)
    ap.add_argument('--destination',type=Path,required=True)
    ap.add_argument('--host',help='Restore only this host; otherwise each host gets a separate subdirectory')
    ap.add_argument('--complete-only',action='store_true',help='Skip verified archive parts belonging to incomplete sets')
    args=ap.parse_args();root=args.destination.resolve();root.mkdir(parents=True,exist_ok=True)
    sets=list(args.download.rglob('COMPLETE.json'))
    if not args.complete_only:sets+=list(args.download.rglob('PARTIAL.json'))
    pending_aliases=[];pending_symlinks=[]
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
                pending_aliases.append((source,target))
        directories=complete.parent/'metadata/empty-directories.json'
        if directories.exists():
            for name in json.loads(directories.read_text()):
                directory=(dest/name.lstrip('/')).resolve()
                if dest not in directory.parents:raise RuntimeError('Unsafe empty directory')
                directory.mkdir(parents=True,exist_ok=True)
        symlinks=complete.parent/'metadata/symlinks.json'
        if symlinks.exists():
            pending_symlinks.extend((dest,item) for item in json.loads(symlinks.read_text()))
    # Apply after all baseline/delta sets. A surviving alias can also recover
    # the canonical pathname of a rotating snapshot pruned during backup.
    remaining=pending_aliases
    while remaining:
        deferred=[];progress=False
        for source,target in remaining:
            if target.exists() and source.exists():continue
            if source.exists():
                target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target);progress=True
            elif target.exists():
                source.parent.mkdir(parents=True,exist_ok=True);os.link(target,source);progress=True
            else:deferred.append((source,target))
        if not progress:
            if deferred:print('WARNING: unresolved hardlink aliases:',len(deferred),flush=True)
            break
        remaining=deferred
    external=0
    for dest,item in pending_symlinks:
        link=dest/item['path'].lstrip('/')
        original=item['target']
        if original.startswith('/'):
            if not original.startswith(('/home/param/','/tmp/')):external+=1;continue
            target=dest/original.lstrip('/')
        else:target=link.parent/original
        target=target.resolve()
        if dest not in link.resolve().parents or dest not in target.parents:external+=1;continue
        if link.exists() or link.is_symlink():continue
        link.parent.mkdir(parents=True,exist_ok=True)
        link.symlink_to(os.path.relpath(target,link.parent))
    if external:print('External/system symlinks left in metadata for manual review:',external)
    print('Restore finished:',root)

if __name__=='__main__':main()
