#!/usr/bin/env python3
"""Restore deduplicated remote orphan source directly from the personal Git backup."""
import argparse,hashlib,json,os,shutil,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,required=True);p.add_argument('--host');a=p.parse_args()
    repo=Path(__file__).resolve().parent;records=json.loads(subprocess.check_output(['zstd','-q','-d','-c',str(repo/'reports/remote-orphan-code.json.zst')]))
    root=a.destination.resolve();root.mkdir(parents=True,exist_ok=True);checked=set();count=0
    for row in records:
        if a.host and row['host']!=a.host:continue
        source=(repo/row['stored']).resolve()
        if repo not in source.parents:raise ValueError('Unsafe source reference')
        if row['sha256'] not in checked:
            if hashlib.file_digest(source.open('rb'),'sha256').hexdigest()!=row['sha256']:raise RuntimeError('Source checksum mismatch: '+str(source))
            checked.add(row['sha256'])
        dest=root if a.host else root/row['host'];target=(dest/row['original'].lstrip('/')).resolve()
        if dest not in target.parents:raise ValueError('Unsafe restore destination')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);os.chmod(target,row['mode']);count+=1
    print(json.dumps({'restored_source_files':count,'destination':str(root)}))

if __name__=='__main__':main()
