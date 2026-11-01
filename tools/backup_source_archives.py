#!/usr/bin/env python3
"""Preserve source tarballs after inspecting/redacting their members."""
import argparse,hashlib,io,json,os,stat,tarfile,time
from pathlib import Path,PurePosixPath
import orphan_backup as B

ap=argparse.ArgumentParser();ap.add_argument('--delta',action='store_true');ap.add_argument('--inventory-only',action='store_true');args=ap.parse_args()
audit_path=B.BASE/'source-archive-audit.json'
previous={r['source']:r for r in json.loads(audit_path.read_text())} if audit_path.exists() else {}
group='source-archives-final-'+str(int(time.time())) if args.delta else 'source-archives'
out=B.BASE/'sanitized-source-archives';out.mkdir(parents=True,exist_ok=True)
selection=[];audit=[]
for record in json.loads((B.BASE/'exclusions.json').read_text()):
    if 'source_archive' not in record['reason']:continue
    source=Path(record['path'])
    if not source.is_file():continue
    digest=hashlib.file_digest(source.open('rb'),'sha256').hexdigest()
    clean=out/(digest+'.tar');receipt=out/(digest+'.json')
    if receipt.exists():result=json.loads(receipt.read_text())
    else:
        removed=[];redactions=0;members=0
        with tarfile.open(source,'r:*') as original,tarfile.open(clean.with_suffix('.partial'),'w') as target:
            for member in original:
                name=PurePosixPath(member.name)
                if B.excluded(member.name) or name.is_absolute() or '..' in name.parts:
                    removed.append(member.name);continue
                if member.isfile():
                    f=original.extractfile(member)
                    if Path(member.name).suffix.lower() in B.TEXT_EXT:
                        data,n=B.sanitize(f.read());member.size=len(data);redactions+=n;target.addfile(member,io.BytesIO(data))
                    else:target.addfile(member,f)
                    f.close()
                else:target.addfile(member)
                members+=1
        if removed or redactions:os.replace(clean.with_suffix('.partial'),clean)
        else:clean.with_suffix('.partial').unlink()
        result={'original_sha256':digest,'source':str(source),'members':members,'removed_credential_or_unsafe_members':removed,'redactions':redactions,'repacked':bool(removed or redactions)}
        B.atomic_json(receipt,result)
    chosen=clean if result['repacked'] else source;s=chosen.stat()
    old=previous.get(str(source),{})
    old_group=old.get('archive_group','source-archives')
    old_complete=B.BASE/'archives'/old_group/'COMPLETE.json'
    old_verified=old_complete.exists() and json.loads(old_complete.read_text()).get('state')=='complete'
    changed=not args.delta or not old_verified or old.get('original_sha256')!=digest
    if changed:selection.append({'path':str(chosen),'archive_path':str(source).lstrip('/'),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'mode':stat.S_IMODE(s.st_mode),'inode':[s.st_dev,s.st_ino]})
    audit.append({**result,'source':str(source),'archived_source':str(chosen),'archive_group':group if changed else old.get('archive_group','source-archives')})
print(json.dumps({'source_archives':len(audit),'selected':len(selection),'sanitized':sum(r['repacked'] for r in audit),'GiB':sum(r['bytes'] for r in selection)/2**30,'group':group}),flush=True)
if args.inventory_only:raise SystemExit(0)
B.pack(selection,group)
B.atomic_json(audit_path,audit)
from huggingface_hub import HfApi
import socket
HfApi().upload_file(path_or_fileobj=str(B.BASE/'source-archive-audit.json'),path_in_repo=f'{socket.gethostname()}/{group}/metadata/source-archive-audit.json',repo_id=B.HF_REPO,repo_type='dataset')
