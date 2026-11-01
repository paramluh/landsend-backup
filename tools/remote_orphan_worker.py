#!/usr/bin/env python3
"""Inventory and upload per-host unique artifacts; uses orphan_backup.py beside this file."""
import argparse,json,os,socket,stat,time
from pathlib import Path
import orphan_backup as B

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inventory-only',action='store_true');ap.add_argument('--delta',action='store_true');ap.add_argument('--include-datasets',action='store_true');args=ap.parse_args()
    roots=[B.HOME_ROOT/'outputs',B.HOME_ROOT/'runs',B.HOME_ROOT/'websites',B.HOME_ROOT/'pose-worktrees',B.HOME_ROOT/'pose-repo',B.HOME_ROOT/'vansh-video-models',B.HOME_ROOT/'vansh-video-models-handover',B.HOME_ROOT/'pose-tokenizer',B.HOME_ROOT/'ppm-training',B.HOME_ROOT/'opd-posttrain-speedups',B.HOME_ROOT/'training-registry',B.HOME_ROOT/'bin',B.HOME_ROOT/'models',B.HOME_ROOT/'.config/systemd',B.HOME_ROOT/'.codex/sessions',B.HOME_ROOT/'.codex/attachments']
    if args.include_datasets:roots.append(B.HOME_ROOT/'datasets')
    for p in Path('/tmp').iterdir():
        try:
            if p.lstat().st_uid==os.getuid():roots.append(p)
        except OSError:pass
    roots.extend(B.HOME_ROOT/p for p in ['.bashrc','.profile','.codex/config.toml','.codex/history.jsonl','.codex/session_index.jsonl'])
    files={};links=[];empty=[];skipped=[]
    for root in roots:
        if root.is_symlink():links.append({'path':str(root),'target':os.readlink(root)});continue
        if root.is_file():files[str(root)]=root;continue
        if not root.is_dir():continue
        for base,ds,fs in os.walk(root,followlinks=False):
            p=Path(base)
            if B.BASE.parent==p or B.BASE.parent in p.parents:ds[:]=[];continue
            if not ds and not fs:empty.append(str(p))
            kept=[]
            for d in ds:
                x=p/d
                if x.is_symlink():links.append({'path':str(x),'target':os.readlink(x)})
                elif not B.excluded(x) and d not in {'site-packages','packages','dist-packages'}:kept.append(d)
            ds[:]=kept
            for f in fs:files[str(p/f)]=p/f
    rows=[]
    for p in files.values():
        try:s=p.lstat()
        except OSError:continue
        if stat.S_ISLNK(s.st_mode):links.append({'path':str(p),'target':os.readlink(p)});continue
        if not stat.S_ISREG(s.st_mode) or s.st_uid!=os.getuid():continue
        if B.excluded(p):skipped.append({'path':str(p),'reason':'credentials_or_regenerable_cache'});continue
        if p.suffix.lower()=='.distcp':skipped.append({'path':str(p),'reason':'distributed_checkpoint_agent_scope'});continue
        if '/models/' in str(p) or 'checkpoints' in p.parts:
            if p.suffix.lower() in {'.pt','.pth','.bin','.safetensors','.ckpt','.onnx','.gguf'} and s.st_size>=10*1024**2:
                skipped.append({'path':str(p),'reason':'model_weight_checkpoint_agent_scope'});continue
        if p.suffix=='.tar' and 'datasets' not in p.parts:
            skipped.append({'path':str(p),'reason':'source_archive_retained_as_expanded_source; credentials_may_be_embedded'});continue
        if p.suffix in {'.sqlite','.sqlite-wal','.sqlite-shm'}:continue
        rows.append({'path':str(p),'archive_path':str(p).lstrip('/'),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'mode':stat.S_IMODE(s.st_mode),'inode':[s.st_dev,s.st_ino]})
    if args.delta:
        old={}
        for manifest in (B.BASE/'archives').glob('artifacts*/part-*.files.json'):
            receipt=manifest.with_name(manifest.name.removesuffix('.files.json')+'.uploaded.json')
            if not receipt.exists() or json.loads(receipt.read_text()).get('state')!='verified':continue
            for r in json.loads(manifest.read_text()):
                if not r.get('error'):old[r['path']]=(r['bytes'],r['mtime_ns'])
        rows=[r for r in rows if old.get(r['path'])!=(r['bytes'],r['mtime_ns'])]
    B.atomic_json(B.BASE/'selection.json',rows)
    B.atomic_json(B.BASE/'symlinks.json',links);B.atomic_json(B.BASE/'empty-directories.json',empty);B.atomic_json(B.BASE/'exclusions.json',skipped)
    print(json.dumps({'host':socket.gethostname(),'files':len(rows),'GiB':sum(x['bytes'] for x in rows)/2**30,'symlinks':len(links),'delta':args.delta}),flush=True)
    if not args.inventory_only:
        # Delta uses a fresh output/archive namespace, retaining initial immutable receipts.
        group='artifacts-final-'+str(int(time.time())) if args.delta else 'artifacts'
        B.pack(rows,group)
        from huggingface_hub import HfApi
        api=HfApi()
        for name in ('selection.json','symlinks.json','empty-directories.json','exclusions.json'):
            api.upload_file(path_or_fileobj=str(B.BASE/name),path_in_repo=f'{socket.gethostname()}/{group}/metadata/{name}',repo_id=B.HF_REPO,repo_type='dataset')

if __name__=='__main__':main()
