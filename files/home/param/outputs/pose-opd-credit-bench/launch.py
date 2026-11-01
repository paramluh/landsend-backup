import argparse,concurrent.futures,json,pathlib,subprocess,shlex,tarfile,time
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--microbatch',type=int,default=2);p.add_argument('--updates',type=int,default=2);p.add_argument('--stage-only',action='store_true');p.add_argument('--no-graphs',action='store_true');p.add_argument('--microbatch-schedule');p.add_argument('--profile-actor-graphs',action='store_true');p.add_argument('--warm-optimizer',action='store_true');p.add_argument('--profile-policy-agreement',action='store_true');p.add_argument('--benchmark-prefix-frames',type=int);p.add_argument('--benchmark-replay-microbatches');p.add_argument('--benchmark-save-rollouts');p.add_argument('--benchmark-load-rollouts');p.add_argument('--profile-replay-stages',action='store_true');p.add_argument('--effective-batch',type=int,default=4);p.add_argument('--benchmark-shape-only',action='store_true');p.add_argument('--benchmark-replay-variants');a=p.parse_args()
base=pathlib.Path('/home/param/outputs/pose-opd-credit-bench');source=pathlib.Path('/home/param/pose-worktrees/pose-opd-credit');hosts=['landsend-'+str(i) for i in [2,3,4,5,12]]
remote=pathlib.Path('/home/param/pose-worktrees')/('opd-bench-'+a.name);run=pathlib.Path('/home/param/runs/pose-runs')/('pose-opd-bench-'+a.name);job=run/'supervisor'
ssh=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes']
def call(h,args,**kw):return subprocess.run(ssh+[h,shlex.join(map(str,args))],text=True,capture_output=True,check=True,**kw)
# Copy tracked files plus new OPD files, never arbitrary untracked assets/secrets.
names=set(x.decode() for x in subprocess.check_output(['git','ls-files','-z'],cwd=source).split(b'\0') if x)
names.update(str(x.relative_to(source)) for x in (source/'finetune/pose_opd_credit').rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.suffix in ('.py','.json','.md'))
archive=base/(a.name+'.tar')
with tarfile.open(archive,'w') as tar:
 for name in sorted(names):
  if (source/name).is_file():tar.add(source/name,arcname=name,recursive=False)
config=json.loads((source/'finetune/pose_opd_credit/recipe.raw.json').read_text());config.update(physical_microbatch=a.microbatch,effective_batch_per_rank=a.effective_batch,frozen_teacher_bf16=True)
configpath=base/(a.name+'.json');configpath.write_text(json.dumps(config,indent=2))
spec={'hosts':hosts,'master_addr':'fdf9:c316:93ee:0:5f4:155d:dc43:d190','port':29670,'updates':a.updates,'repo':str(remote),'run_dir':str(run),'config':str(remote/'benchmark-config.json'),'extra_args':(['--no-graphs'] if a.no_graphs else [])+(['--benchmark-microbatches',a.microbatch_schedule] if a.microbatch_schedule else [])+(['--profile-actor-graphs'] if a.profile_actor_graphs else [])+(['--warm-optimizer'] if a.warm_optimizer else [])+(['--profile-policy-agreement'] if a.profile_policy_agreement else [])+(['--benchmark-prefix-frames',str(a.benchmark_prefix_frames)] if a.benchmark_prefix_frames else [])+(['--benchmark-replay-microbatches',a.benchmark_replay_microbatches] if a.benchmark_replay_microbatches else [])+(['--benchmark-save-rollouts',a.benchmark_save_rollouts] if a.benchmark_save_rollouts else [])+(['--benchmark-load-rollouts',a.benchmark_load_rollouts] if a.benchmark_load_rollouts else [])+(['--profile-replay-stages'] if a.profile_replay_stages else [])+(['--benchmark-shape-only'] if a.benchmark_shape_only else [])+(['--benchmark-replay-variants',a.benchmark_replay_variants] if a.benchmark_replay_variants else [])}
specpath=base/(a.name+'-launch.json');specpath.write_text(json.dumps(spec,indent=2))
def stage(h):
 call(h,['mkdir','-p',remote,job])
 subprocess.run(['rsync','-a','-e',shlex.join(ssh),str(archive),h+':'+str(job/'source.tar')],check=True,capture_output=True)
 call(h,['tar','xf',job/'source.tar','-C',remote])
 for src,dst in [(base/'worker.py',job/'worker.py'),(specpath,job/'spec.json'),(configpath,remote/'benchmark-config.json')]:
  subprocess.run(['rsync','-a','-e',shlex.join(ssh),str(src),h+':'+str(dst)],check=True,capture_output=True)
 print('staged',h,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:list(ex.map(stage,hosts))
if not a.stage_only:
 def start(h):
  # All paths are controlled; quote shell redirection separately from argv.
  cmd='nohup '+shlex.join(['python3',str(job/'worker.py'),str(job/'spec.json')])+' >'+shlex.quote(str(job/'worker.log'))+' 2>&1 </dev/null &'
  call(h,['bash','-c',cmd]);print('started',h,flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:list(ex.map(start,hosts))
print(json.dumps(spec,indent=2))
