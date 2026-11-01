#!/usr/bin/env python3
"""Store small orphan source from remote nodes in personal Git, deduplicating content."""
import concurrent.futures,hashlib,io,json,shlex,subprocess,tarfile
from pathlib import Path
import orphan_backup as B

EXPORT=r'''
import sys,json,pathlib,tarfile,io
b=pathlib.Path('/home/param/outputs/landsend-backup-20261031/orphans');sys.path.insert(0,str(b));import orphan_backup as B
roots=[pathlib.Path(x) for x in json.loads(sys.argv[1])]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',compresslevel=1) as out:
 for r in json.loads((b/'selection.json').read_text()):
  p=pathlib.Path(r['path'])
  if p.suffix.lower() not in B.SMALL_CODE or r['bytes']>4*1024**2 or B.excluded(p):continue
  if any(root==p or root in p.parents for root in roots):continue
  if '/code/' in str(p) or any(x.startswith('snapshot-') for x in p.parts):continue
  try:
   if p.stat().st_size>4*1024**2:continue
   data,n=B.sanitize(p.read_bytes());member=out.gettarinfo(str(p),arcname=str(p).lstrip('/'))
   member.size=len(data);member.pax_headers={'redactions':str(n)};out.addfile(member,io.BytesIO(data))
  except OSError:continue
'''
def export(node):
    roots=[r['path'] for r in json.loads((B.BASE.parent/f'git/landsend-{node}-inventory.json').read_text())['repos']]
    proc=subprocess.run(['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes',f'landsend-{node}','/home/param/pose-repo/.venv/bin/python','-c',shlex.quote(EXPORT),shlex.quote(json.dumps(roots))],capture_output=True,timeout=300)
    if proc.returncode:raise RuntimeError(f'node{node}: '+proc.stderr.decode(errors='replace')[:300])
    return node,proc.stdout

def main():
    known={}
    for root in (B.REPO/'files',B.REPO/'remote-files'):
        if not root.exists():continue
        for p in root.rglob('*'):
            if p.is_file() and not p.is_symlink():known.setdefault(hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),str(p.relative_to(B.REPO)))
    report=[];added=0;added_bytes=0
    exports=concurrent.futures.ThreadPoolExecutor(max_workers=6).map(export,[1,2,3,4,5,12])
    for node,payload in exports:
        count=0
        with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as source:
            for m in source:
                if not m.isfile():continue
                if Path(m.name).is_absolute() or '..' in Path(m.name).parts:raise RuntimeError('Unsafe source path')
                data,n=B.sanitize(source.extractfile(m).read());digest=hashlib.sha256(data).hexdigest()
                if digest not in known:
                    dest=B.REPO/'remote-files'/f'landsend-{node}'/m.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);dest.chmod(m.mode&0o777)
                    known[digest]=str(dest.relative_to(B.REPO));added+=1;added_bytes+=len(data)
                report.append({'host':f'landsend-{node}','original':'/'+m.name,'stored':known[digest],'sha256':digest,'bytes':len(data),'mode':m.mode,'redactions':n+int(m.pax_headers.get('redactions','0'))});count+=1
        print(json.dumps({'host':f'landsend-{node}','source_files':count,'new_unique_files_so_far':added}),flush=True)
    raw=json.dumps(report,indent=2).encode();target=B.REPO/'reports/remote-orphan-code.json.zst'
    with target.open('wb') as out:subprocess.run(['zstd','-q','-1','-c'],input=raw,stdout=out,check=True)
    B.atomic_json(B.BASE/'remote-code-summary.json',{'state':'staged','files':len(report),'new_unique_files':added,'new_unique_bytes':added_bytes,'manifest':str(target.relative_to(B.REPO))})

if __name__=='__main__':main()
