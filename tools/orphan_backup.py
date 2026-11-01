#!/usr/bin/env python3
"""Sanitized, deduplicated orphan backup; checkpoints and Git histories have separate owners."""
import argparse, collections, concurrent.futures, gzip, hashlib, io, json, os, re, shutil, socket, stat, subprocess, tarfile, tempfile, time
from pathlib import Path

BASE=Path('/home/param/outputs/landsend-backup-20261031/orphans')
HOME_ROOT=Path('/home/param')
REPO=BASE/'repo'
HF_REPO=os.environ.get('LANDSEND_BACKUP_HF_REPO','InternalCan/landsend-artifacts-backup-param')
TEXT_EXT={'.py','.sh','.bash','.md','.txt','.json','.jsonl','.yaml','.yml','.toml','.ini','.cfg','.conf','.js','.ts','.tsx','.jsx','.html','.css','.csv','.tsv','.log','.patch','.diff','.sbatch','.service','.socket','.timer','.sql','.cu','.cpp','.h','.hpp','.c','.xml','.ipynb','.lock'}
SMALL_CODE={'.py','.sh','.bash','.md','.yaml','.yml','.toml','.ini','.cfg','.conf','.js','.ts','.tsx','.jsx','.html','.css','.patch','.diff','.sbatch','.service','.socket','.timer','.sql','.cu','.cpp','.h','.hpp','.c','.xml'}
PRUNE={'.git','.venv','venv','node_modules','__pycache__','.cache','.nv','.triton','.ssh','.gnupg','.hf-cli','hf-upload'}
SECRET_NAMES={'auth.json','credentials','credentials.json','.netrc','.git-credentials','token','stored_tokens','hosts.yml','authorized_keys','known_hosts','known_hosts.old'}
SECRET_PATTERNS=[
 re.compile(rb'\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,})\b'),
 re.compile(rb'-----BEGIN (?:[A-Z ]*PRIVATE KEY)-----[\s\S]*?-----END (?:[A-Z ]*PRIVATE KEY)-----'),
 re.compile(rb'\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b'),
 re.compile(rb'(?i)(?:"?(?:access_token|refresh_token|id_token|api_key|secret_key|password|HF_TOKEN|GITHUB_TOKEN|GH_TOKEN)"?\s*[:=]\s*)["\x27][A-Za-z0-9_./+\-=]{16,}["\x27]'),
]

def excluded(path):
    p=Path(path); parts=p.parts; name=p.name.lower()
    return (any(x in PRUNE for x in parts) or name in SECRET_NAMES or name.startswith('.env')
            or name.startswith('id_ed25519') or name.startswith('id_rsa') or name.endswith(('.pem','.key','.p12','.pfx'))
            or 'google-chrome-for-testing' in parts or 'shell_snapshots' in parts)

def sanitize(data):
    count=0
    for pattern in SECRET_PATTERNS:
        data,n=pattern.subn(b'[REDACTED_CREDENTIAL]',data);count+=n
    return data,count

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,path)

def gather():
    records={}
    for row in json.loads((BASE/'orphan-file-inventory.json').read_text()):
        p=HOME_ROOT/row['path']; records[str(p)]=p
    inv=BASE.parent/'git/local-inventory.json'
    if inv.exists():
        for repo in json.loads(inv.read_text()):
            for item in repo.get('ignored',[]):
                if isinstance(item,dict): item=item.get('path','')
                p=Path(repo['path'])/item
                if not excluded(p):
                    if p.is_file(): records[str(p)]=p
                    elif p.is_dir():
                        for b,ds,fs in os.walk(p,followlinks=False):
                            ds[:]=[x for x in ds if not excluded(Path(b)/x) and not (Path(b)/x).is_symlink()]
                            for n in fs: records[str(Path(b)/n)]=Path(b)/n
    extra=[HOME_ROOT/'.codex/sessions',HOME_ROOT/'.codex/attachments',HOME_ROOT/'.codex/history.jsonl',HOME_ROOT/'.codex/session_index.jsonl',HOME_ROOT/'.codex/config.toml',HOME_ROOT/'.config/systemd',HOME_ROOT/'.bashrc',HOME_ROOT/'.profile']
    for p in Path('/tmp').iterdir():
        try:
            if p.lstat().st_uid==os.getuid() and not p.is_symlink(): extra.append(p)
        except OSError:pass
    for p in extra:
        if p.is_file(): records[str(p)]=p;continue
        if not p.is_dir():continue
        for b,ds,fs in os.walk(p,followlinks=False):
            if (Path(b)/'.git').exists():ds[:]=[];continue
            ds[:]=[x for x in ds if not excluded(Path(b)/x) and not (Path(b)/x).is_symlink()]
            for n in fs:records[str(Path(b)/n)]=Path(b)/n
    result=[];skipped=[]
    for p in records.values():
        reason=None
        if excluded(p):reason='credential_or_regenerable_cache'
        elif BASE.parent in p.parents:reason='backup_workspace'
        elif p.suffix=='.tar' and 'datasets' not in p.parts:reason='source_archive_with_possible_embedded_credentials; expanded source backed up separately'
        elif 'pose-mono-dcp227-export' in p.parts:reason='checkpoint_agent_scope'
        try:s=p.lstat()
        except OSError:continue
        if stat.S_ISLNK(s.st_mode):reason='symlink_recorded_separately'
        elif not stat.S_ISREG(s.st_mode):reason='not_regular_file'
        if reason:skipped.append({'path':str(p),'reason':reason});continue
        result.append({'path':str(p),'archive_path':str(p).lstrip('/'),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'mode':stat.S_IMODE(s.st_mode),'inode':[s.st_dev,s.st_ino]})
    atomic_json(BASE/'selection.json',result);atomic_json(BASE/'exclusions.json',skipped)
    print(json.dumps({'selected':len(result),'apparent_GiB':sum(x['bytes'] for x in result)/2**30,'excluded':len(skipped)}),flush=True)
    return result

def stage_code(rows):
    REPO.mkdir(parents=True,exist_ok=True)
    receipts=[]
    for r in rows:
        p=Path(r['path'])
        if p.suffix.lower() not in SMALL_CODE or r['bytes']>4*1024**2:continue
        # Immutable per-run copies are retained in bulk archives; avoid thousands of duplicated checkouts in Git.
        if '/code/' in str(p) and ('/runs/' in str(p) or '/snapshot-' in str(p)):continue
        if any(x.startswith('snapshot-') for x in p.parts):continue
        try:data,n=sanitize(p.read_bytes())
        except OSError:continue
        dst=REPO/'files'/r['archive_path'];dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
        dst.chmod(r['mode'] & 0o777)
        receipts.append({'path':r['path'],'redactions':n,'sha256':hashlib.sha256(data).hexdigest()})
    atomic_json(REPO/'reports/code-files.json',receipts)
    print(json.dumps({'code_files':len(receipts),'redacted_files':sum(x['redactions']>0 for x in receipts)}),flush=True)

def pack(rows,group):
    from huggingface_hub import HfApi
    api=HfApi();api.create_repo(HF_REPO,repo_type='dataset',private=True,exist_ok=True)
    assert api.repo_info(HF_REPO,repo_type='dataset').private
    output=BASE/'archives'/group;output.mkdir(parents=True,exist_ok=True)
    done=output/'COMPLETE.json'
    if done.exists():print(done.read_text());return
    proven=set(json.loads((BASE/'verified-existing-dataset-paths.json').read_text())) if (BASE/'verified-existing-dataset-paths.json').exists() else set()
    rows=[r for r in rows if ('/datasets/' in r['path'])==(group=='datasets') and r['path'] not in proven]
    seen={};aliases=[];unique=[]
    for r in rows:
        key=tuple(r['inode'])
        if key in seen:aliases.append({'path':r['archive_path'],'target':seen[key]})
        else:seen[key]=r['archive_path'];unique.append(r)
    atomic_json(output/'aliases.json',aliases)
    # Small text first makes the first archive a useful recovery point immediately.
    unique.sort(key=lambda r:(r['bytes']>1024**2,r['path']))
    parts=[];part=[];size=0
    for r in unique:
        if part and size+r['bytes']>2*1024**3:parts.append(part);part=[];size=0
        part.append(r);size+=r['bytes']
    if part:parts.append(part)
    print(json.dumps({'group':group,'files':len(unique),'hardlink_aliases':len(aliases),'parts':len(parts),'unique_GiB':sum(r['bytes'] for r in unique)/2**30}),flush=True)
    statuses=[]
    upload_pool=concurrent.futures.ThreadPoolExecutor(max_workers=2)
    futures=[]
    def upload(path,meta):
        start=time.time()
        for attempt in range(1,9):
            try:
                commit=api.upload_file(path_or_fileobj=str(path),path_in_repo=f'{socket.gethostname()}/{group}/{path.name}',repo_id=HF_REPO,repo_type='dataset',commit_message=f'{socket.gethostname()} {group} {path.name}')
                remote=api.get_paths_info(HF_REPO,[f'{socket.gethostname()}/{group}/{path.name}'],repo_type='dataset',revision=commit.oid)[0]
                assert remote.size==path.stat().st_size
                sha=getattr(remote.lfs,'sha256',None) if remote.lfs else None
                if isinstance(remote.lfs,dict):sha=remote.lfs.get('sha256')
                assert sha==meta['sha256'],(sha,meta['sha256'])
                receipt={**meta,'revision':commit.oid,'upload_seconds':time.time()-start,'state':'verified','hf_path':f'{socket.gethostname()}/{group}/{path.name}'}
                atomic_json(path.with_suffix(path.suffix+'.uploaded.json'),receipt)
                print(json.dumps(receipt),flush=True)
                return receipt
            except Exception as exc:
                print(json.dumps({'upload':path.name,'attempt':attempt,'error':str(exc)}),flush=True)
                time.sleep(min(60,attempt*5))
        raise RuntimeError('upload exhausted: '+str(path))
    all_records=[]
    for i,part in enumerate(parts):
        archive=output/f'part-{i:04d}.tar.zst';meta_path=archive.with_suffix(archive.suffix+'.json')
        if archive.exists() and meta_path.exists():meta=json.loads(meta_path.read_text())
        else:
            tmp=archive.with_suffix('.partial');manifest=[]
            with tmp.open('wb') as out:
                z=subprocess.Popen(['zstd','-q','-1','-T2','-c'],stdin=subprocess.PIPE,stdout=out)
                with tarfile.open(fileobj=z.stdin,mode='w|') as tar:
                    for r in part:
                        p=Path(r['path'])
                        try:
                            info=tar.gettarinfo(str(p),arcname=r['archive_path']);info.uid=info.gid=0;info.uname=info.gname=''
                            if p.suffix.lower() in TEXT_EXT or p.name in {'.bashrc','.profile','.gitattributes','Dockerfile','Makefile'}:
                                data,n=sanitize(p.read_bytes());info.size=len(data);tar.addfile(info,io.BytesIO(data))
                            elif p.suffix=='.gz' and p.name.endswith(('.json.gz','.jsonl.gz','.log.gz','.txt.gz')):
                                data,n=sanitize(gzip.decompress(p.read_bytes()));data=gzip.compress(data,compresslevel=1,mtime=0);info.size=len(data);tar.addfile(info,io.BytesIO(data))
                            else:
                                # A binary artifact may be replaced/truncated by a live worker.
                                # Snapshot before emitting its tar header so a short read cannot
                                # corrupt the remainder of the archive.
                                n=0
                                with tempfile.TemporaryFile(dir=output) as snapshot:
                                    with p.open('rb') as f:shutil.copyfileobj(f,snapshot,4*1024**2)
                                    info.size=snapshot.tell();snapshot.seek(0);tar.addfile(info,snapshot)
                            manifest.append({**r,'archived_bytes':info.size,'redactions':n})
                        except (OSError,EOFError,tarfile.TarError) as exc:
                            manifest.append({**r,'error':str(exc)})
                z.stdin.close()
                if z.wait()!=0:raise RuntimeError('zstd failed')
            os.replace(tmp,archive)
            digest=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
            meta={'archive':archive.name,'bytes':archive.stat().st_size,'sha256':digest,'files':len(manifest),'errors':sum('error' in r for r in manifest),'created_unix':time.time()}
            atomic_json(meta_path,meta);atomic_json(archive.with_suffix(archive.suffix+'.files.json'),manifest)
        all_records.append(meta)
        receipt_path=archive.with_suffix(archive.suffix+'.uploaded.json')
        if receipt_path.exists():statuses.append(json.loads(receipt_path.read_text()))
        else:futures.append(upload_pool.submit(upload,archive,meta))
        # Bound unuploaded archive accumulation while keeping compression ahead of networking.
        if len(futures)>=4:
            statuses.append(futures.pop(0).result())
    statuses.extend(f.result() for f in futures);upload_pool.shutdown()
    for p in [output/'aliases.json',*sorted(output.glob('*.files.json'))]:
        api.upload_file(path_or_fileobj=str(p),path_in_repo=f'{socket.gethostname()}/{group}/manifests/{p.name}',repo_id=HF_REPO,repo_type='dataset')
    errors=sum(r.get('errors',0) for r in all_records)
    final={'state':'complete' if not errors else 'partial','errors':errors,'hf_repo':HF_REPO,'group':group,'archives':statuses,'files':len(unique),'aliases':len(aliases),'completed_unix':time.time()}
    status_path=done if not errors else output/'PARTIAL.json'
    atomic_json(status_path,final)
    api.upload_file(path_or_fileobj=str(status_path),path_in_repo=f'{socket.gethostname()}/{group}/{status_path.name}',repo_id=HF_REPO,repo_type='dataset')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['gather','code','pack']);ap.add_argument('--group',default='artifacts',choices=['artifacts','datasets']);args=ap.parse_args()
    rows=gather() if args.mode=='gather' else json.loads((BASE/'selection.json').read_text())
    if args.mode=='code':stage_code(rows)
    elif args.mode=='pack':pack(rows,args.group)

if __name__=='__main__':main()
