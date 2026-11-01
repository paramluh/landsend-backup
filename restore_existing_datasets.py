#!/usr/bin/env python3
"""Restore files proven to exist at pinned original dataset revisions."""
import argparse,json,shutil
from pathlib import Path
from huggingface_hub import snapshot_download

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('manifest',type=Path,help='landsend-0/datasets/metadata/verified-existing-datasets.json from HF backup')
ap.add_argument('--destination',type=Path,required=True)
args=ap.parse_args();dest=args.destination.resolve();groups={}
for row in json.loads(args.manifest.read_text())['verified']:
    groups.setdefault((row['repo'],row['revision']),[]).append(row)
for (repo,revision),rows in groups.items():
    source=Path(snapshot_download(repo,repo_type='dataset',revision=revision,allow_patterns=[r['path'] for r in rows]))
    for row in rows:
        target=dest/row['local'].lstrip('/');target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/row['path'],target)
    print(repo,revision,len(rows),'restored',flush=True)
