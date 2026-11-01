#!/usr/bin/env python3
"""Refresh personal GitHub recovery documentation and verified backup receipts."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import orphan_backup as B

ROOT=B.BASE.parent
REPO=B.REPO
dest=REPO/'reports/cluster-backup'
selected=[]
for base,dirs,files in os.walk(ROOT):
    p=Path(base)
    dirs[:]=[d for d in dirs if d not in {'repo','archives','hf-pose-mirror','download','downloads','extracted','payload','.git','__pycache__'}]
    if '/restore-check/' in str(p) and p.name!='restore-check':dirs[:]=[];continue
    for name in files:
        f=p/name
        if B.excluded(f) or f.is_symlink() or not f.is_file():continue
        rel=f.relative_to(ROOT)
        if f.name.endswith('.process.json') or f.suffix in {'.py','.sh','.md','.sql'} or f.suffix=='.json':
            if f.name in {'selection.json','orphan-file-inventory.json','dataset-selection.json','repo-roots.json'}:continue
            if f.stat().st_size>96*1024**2:continue
            selected.append((f,rel))
        elif 'supplementary' in rel.parts and f.suffix in {'.gz','.zst','.txt'}:selected.append((f,rel))

for f,rel in selected:
    target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
    if f.suffix in {'.gz','.zst'}:shutil.copy2(f,target);continue
    data,count=B.sanitize(f.read_bytes())
    if len(data)>1024**2:
        target=target.with_name(target.name+'.zst')
        with target.open('wb') as out:subprocess.run(['zstd','-q','-1','-c'],input=data,stdout=out,check=True)
    else:target.write_bytes(data)
for script in B.BASE.glob('*.py'):shutil.copy2(script,REPO/'tools'/script.name)
shutil.copy2(B.BASE/'checkpoint-scopes.json',REPO/'tools/checkpoint-scopes.json')
statuses=[]
for p in (B.BASE/'archives').glob('*/*.uploaded.json'):
    x=json.loads(p.read_text());statuses.append({'group':p.parent.name,**x})
B.atomic_json(REPO/'reports/landsend-0-verified-archives.json',statuses)
for p in (B.BASE/'archives').glob('*/COMPLETE.json'):
    target=dest/'orphans/archive-sets'/p.parent.name/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
for p in (B.BASE/'archives').glob('*/PARTIAL.json'):
    target=dest/'orphans/archive-sets'/p.parent.name/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)

subprocess.run(['git','add','--all','.'],cwd=REPO,check=True)
changed=subprocess.run(['git','diff','--cached','--quiet'],cwd=REPO).returncode
if changed:
    subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-q','-m','Refresh cluster recovery receipts and verified archive inventory'],cwd=REPO,check=True)
subprocess.run(['git','push','origin','main'],cwd=REPO,check=True)
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=REPO,text=True).split()[0]
assert sha==remote
result={'state':'verified','github':'https://github.com/paramluh/landsend-backup','commit':sha,'published_unix':time.time(),'receipt_files':len(selected),'root_verified_archives':len(statuses)}
B.atomic_json(ROOT/'personal-github-publication.json',result)
print(json.dumps(result),flush=True)
