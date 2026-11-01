from pathlib import Path
s=Path('/home/param/pose-repo/finetune/duplex_distill/cluster_ops.py').read_text()
s=s[s.index('from __future__'):]
s='''"""Detached black-band reproduction training on landsend-4/5/12 only.

Adapted from the tested duplex_distill/cluster_ops.py supervisor. No eviction;
rollback signals only this launch's PID/start-tick-verified process groups.
Protected training dependencies are copied only when missing; differences abort.
"""\n'''+s
s=s.replace('HOSTS = tuple(f"landsend-{i}" for i in (0, 1, 2, 3, 4, 5, 12))','HOSTS = ("landsend-4", "landsend-5", "landsend-12")')
s=s.replace("LOCK = Path('/home/param/runs/pose-runs/.duplex-distill-gpus.lock')","LOCK = Path('/home/param/runs/pose-runs/.blackband-repro-gpus.lock')")
a=s.index('FFMPEG = ');b=s.index('\n\ndef now()',a)
s=s[:a]+'''BASE = Path('/home/param/models/imported-weights/ppm-pose6-mono17850-final/export-357-dcp')
CONFIGS = Path('/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs')
'''+s[b:]
a=s.index('def permitted(');b=s.index('\n\ndef process_identity',a)
s=s[:a]+'''def permitted(hosts, mode, *, require_full=True):
    if mode != 'train' or tuple(hosts) != HOSTS:
        raise ValueError(f'Training is restricted to all24 GPUs in exact host order {HOSTS}')
'''+s[b:]
a=s.index('def preflight(');b=s.index('\n\ndef worker(',a)
s=s[:a]+'''def preflight(manifest, base, configs):
    host = local_host()
    rows, pids = query_gpu()
    assert len(rows) == 8 and all('H100' in row for row in rows), rows
    assert not pids, f'Existing GPU jobs: {pids}; this launcher never evicts jobs'
    assert all(int(row.split(',')[3].strip()) == 0 for row in rows), rows
    base, configs, manifest = Path(base), Path(configs), Path(manifest)
    cfg = json.loads((base/'train_cfg.json').read_text())
    assert cfg['step'] == 357 and cfg['run'] == 'ppm-pose6-mono17850-j31240'
    assert (base/'COMPLETE').read_text().strip() == '357'
    shards = list(base.glob('*.distcp'))
    assert len(shards) == 64 and all(x.stat().st_size > 0 for x in shards)
    assert (base/'.metadata').is_file()
    paths = [REPO/'.venv/bin/torchrun', REPO/'.venv/bin/python',
             REPO/'finetune/transcript_pose/environment/activate.sh',
             REPO/'finetune/xabi-main-17850/run.env',
             REPO/'finetune/xabi-main-17850/code/train_aloe_pose.py',
             HERE/'train.py', HERE/'data_blackbands.py', manifest,
             configs/'Qwen3-32B/tokenizer.json', configs/'Qwen3-32B/config.json',
             configs/'Qwen3-4B/config.json', configs/'Qwen3-0.6B/config.json']
    missing = [str(p) for p in paths if not p.is_file()]
    assert not missing, f'Missing inputs: {missing}'
    addresses = json.loads(subprocess.check_output(['ip','-j','-6','address','show','dev','ibs14'],text=True))
    ipv6 = [a['local'] for i in addresses for a in i['addr_info'] if a['family']=='inet6' and a['scope']=='global']
    assert ipv6, 'No global IPv6 address on ibs14'
    hcas = sorted(p.name for p in Path('/sys/class/infiniband').iterdir())
    assert len(hcas)==8 and all('ACTIVE' in (Path('/sys/class/infiniband')/h/'ports/1/state').read_text() for h in hcas)
    packages = json.loads(subprocess.check_output([str(REPO/'.venv/bin/python'),'-c',
        'import torch,transformers,json;print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),transformers=transformers.__version__)))'],text=True))
    files = [HERE/'train.py',HERE/'data_blackbands.py',HERE/'launch_cluster.py',REPO/'finetune/xabi-main-17850/run.env']
    return dict(host=host,passed=True,checked_utc=now(),gpus=rows,ib_address=ipv6[0],hcas=hcas,packages=packages,
                base=str(base),checkpoint_shards=64,manifest=str(manifest),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
'''+s[b:]
s=s.replace("        env = os.environ.copy()","        rows, pids = query_gpu()\n        assert not pids and all(int(row.split(',')[3].strip()) == 0 for row in rows), (rows,pids)\n        env = {k:v for k,v in os.environ.items() if not k.startswith('WANDB_')}")
s=s.replace("'PYTHONUNBUFFERED': '1', 'WANDB_BASE_URL': 'https://api.forge.coreweave.com',\n                    'WANDB_ENTITY': 'canopy-labs', 'S4_FFMPEG': str(FFMPEG)","'PYTHONUNBUFFERED': '1', 'WANDB_MODE': 'disabled', 'WANDB_DISABLED': 'true',\n                    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'")
s=s.replace("                   '--mode', spec['mode'], *spec['trainer_args']]","                   *spec['trainer_args']]")
s=s.replace("'duplex-distill', *command]","'blackband-repro', *command]")
s=s.replace("        # The controller also trains: launch() has already written this exact\n        # launch directory locally. Never attach a second worker to a run.\n        assert local_host() == 'landsend-0', f'Existing worker launch directory: {job}'","        # Reject duplicate launches on any worker, even if the spec matches.\n        raise RuntimeError(f'Existing worker launch directory: {job}')")
a=s.index('def sync(host):');b=s.index('\n\ndef parallel(',a)
s=s[:a]+'''def protected_files():
    roots = [REPO/'finetune/xabi-main-17850/code', CONFIGS]
    files = [REPO/'finetune/xabi-main-17850/run.env',REPO/'finetune/storage_paths.py',
             REPO/'finetune/transcript_pose/environment/activate.sh']
    for root in roots:
        files.extend(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    return sorted(set(files))


def sync(host):
    files=protected_files()
    local={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    script = 'import sys,json,hashlib;from pathlib import Path;d=json.load(sys.stdin);print(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).is_file() else None for p in d}))'
    existing=json.loads(remote(host,'python3','-c',script,input_text=json.dumps(local)).stdout)
    conflicts=[p for p,h in existing.items() if h is not None and h!=local[p]]
    if conflicts:
        raise RuntimeError('Protected dependency hashes differ; no overwrite: '+json.dumps(conflicts))
    missing=[p for p,h in existing.items() if h is None]
    if missing:
        subprocess.run(['rsync','-a','--no-owner','--no-group','--relative','--ignore-existing','-e',shlex.join(SSH),
                        *missing,host+':/'],capture_output=True,text=True,check=True,timeout=180)
    remote(host,'mkdir','-p',HERE)
    subprocess.run(['rsync','-a','--no-owner','--no-group','--exclude=__pycache__/','--exclude=*.pyc',
                    '--exclude=infra-readiness.json','-e',shlex.join(SSH),str(HERE)+'/',f'{host}:{HERE}/'],
                   capture_output=True,text=True,check=True,timeout=180)
    checked=json.loads(remote(host,'python3','-c',script,input_text=json.dumps(local)).stdout)
    assert checked==local, 'Post-sync protected dependency hash mismatch'
    return dict(host=host,protected_files=len(files),copied_missing=len(missing),differing_files_overwritten=0)
'''+s[b:]
s=s.replace("    hosts = tuple(args.hosts.split(',')) if args.hosts else HOSTS\n    permitted(hosts, args.mode)","    hosts = HOSTS\n    permitted(hosts, args.mode)")
s=s.replace("        parallel(hosts, sync)\n        results = parallel(hosts, lambda host: json.loads(remote(host, '/usr/bin/python3', __file__, '_preflight').stdout))","        write_json(job/'sync.json', parallel(hosts, sync))\n        results = parallel(hosts, lambda host: json.loads(remote(host, '/usr/bin/python3', __file__, '_preflight', args.manifest, args.base, args.configs).stdout))\n        assert len({x['manifest_sha256'] for x in results.values()}) == 1\n        assert len({json.dumps(x['source_sha256'],sort_keys=True) for x in results.values()}) == 1")
s=s.replace("        print(f'Status: {HERE}/status_cluster.sh --run-dir {shlex.quote(str(run))}')","        print(f'Status: python3 {__file__} status --run-dir {shlex.quote(str(run))}')")
s=s.replace("print(json.dumps(preflight()))","print(json.dumps(preflight(*rest)))")
s=s.replace("        parser.add_argument('--mode', choices=('smoke', 'train'), default='train')\n        parser.add_argument('--hosts', help='Comma-separated configured hosts; subsets allowed only for smoke')\n        parser.add_argument('--port', type=int, default=29573)","        parser.set_defaults(mode='train')\n        parser.add_argument('--manifest',required=True)\n        parser.add_argument('--base',default=str(BASE))\n        parser.add_argument('--configs',default=str(CONFIGS))\n        parser.add_argument('--epochs',type=int,default=1)\n        parser.add_argument('--seed',type=int,default=20261031)\n        parser.add_argument('--port', type=int, default=29631)")
s=s.replace("        args, trainer_args = parser.parse_known_args(rest)\n        launch(args, trainer_args)","        args = parser.parse_args(rest)\n        if args.epochs < 1 or not all(Path(p).is_absolute() for p in (args.manifest,args.base,args.configs)):parser.error('Absolute paths and positive epochs required')\n        trainer_args=['--manifest',args.manifest,'--base',args.base,'--configs',args.configs,'--epochs',str(args.epochs),'--seed',str(args.seed)]\n        launch(args, trainer_args)")
s=s.replace("    elif command == 'launch':","    elif command == 'sync':\n        print(json.dumps(parallel(HOSTS,sync),indent=2))\n    elif command == 'launch':")
s=s.replace("        # Existing six-node runs remain inspectable/stoppable after expansion.\n",'')
Path('/home/param/pose-repo/finetune/blackband_repro/launch_cluster.py').write_text(s)
