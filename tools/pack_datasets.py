#!/usr/bin/env python3
"""Wait for exact remote verification, then archive only local dataset derivatives."""
import json,os,stat,time
from pathlib import Path
import orphan_backup as B

while not (B.BASE/'verified-existing-datasets.json').exists():time.sleep(10)
rows=[];links=[]
for record in json.loads((B.BASE/'orphan-file-inventory.json').read_text()):
    if not record['path'].startswith('datasets/'):continue
    p=B.HOME_ROOT/record['path']
    if B.excluded(p):continue
    try:s=p.stat()
    except OSError:continue
    rows.append({'path':str(p),'archive_path':str(p).lstrip('/'),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'mode':stat.S_IMODE(s.st_mode),'inode':[s.st_dev,s.st_ino]})
B.atomic_json(B.BASE/'dataset-selection.json',rows)
B.pack(rows,'datasets')
from huggingface_hub import HfApi
api=HfApi()
for name in ('verified-existing-datasets.json','verified-existing-dataset-paths.json','dataset-selection.json'):
    api.upload_file(path_or_fileobj=str(B.BASE/name),path_in_repo='landsend-0/datasets/metadata/'+name,repo_id=B.HF_REPO,repo_type='dataset')
