import io, json, os, pathlib, shlex, subprocess, tarfile, time
ROOT = pathlib.Path('/home/param/outputs/pose-opd-rollout-opt-projections')
REPO = pathlib.Path('/home/param/pose-worktrees/pose-opd-credit')
RUN = '/home/param/runs/pose-runs/rollout-opt-projections-' + str(int(time.time()))
SSH = ['ssh', '-i', '/home/param/.ssh/id_ed25519_cluster', '-o', 'BatchMode=yes', 'landsend-4']
PY = '/home/param/pose-repo/.venv/bin/python'
ACT = '/home/param/pose-repo/finetune/transcript_pose/environment/activate.sh'
def remote(code):
    return subprocess.check_output(SSH + [shlex.join(['python3', '-c', code])], text=True)
archive = subprocess.check_output(['git','archive','HEAD'], cwd=REPO)
with tarfile.open(fileobj=io.BytesIO(archive)) as src, tarfile.open(ROOT/'source.tar','w') as dst:
    for item in src:
        if item.isfile():
            local=REPO/item.name
            data=local.read_bytes()
            item.size=len(data)
            dst.addfile(item,io.BytesIO(data))
        else: dst.addfile(item)
remote(f'import pathlib; pathlib.Path({RUN!r}).mkdir(parents=True)')
subprocess.run(['scp','-i','/home/param/.ssh/id_ed25519_cluster',str(ROOT/'source.tar'),f'landsend-4:{RUN}/source.tar'],check=True)
worker = '''import fcntl,json,os,pathlib,signal,subprocess,time
run=pathlib.Path(__file__).parent
children=[]
def stop(*args):
    for p in children:
        if p.poll() is None: os.killpg(p.pid,signal.SIGTERM)
signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGINT,stop)
with open('/home/param/runs/pose-runs/.duplex-distill-gpus.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if apps: raise RuntimeError('GPU processes still present: '+apps)
    spec=json.loads((run/'spec.json').read_text())
    for item in spec['jobs']:
        log=open(run/(item['name']+'.log'),'w')
        p=subprocess.Popen(item['argv'],cwd=run/'code',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p)
    (run/'children.json').write_text(json.dumps([{'pid':p.pid,'start_ticks':pathlib.Path('/proc/%d/stat'%p.pid).read_text().rsplit(')',1)[1].split()[19]} for p in children]))
    codes=[p.wait() for p in children]
    (run/'result.json').write_text(json.dumps({'codes':codes,'finished':time.time()}))
'''
jobs=[]
for name,student,partner,packed in [('baseline','0,1','2,3','none'),('packed','4,5','6,7','all')]:
    argv=[PY,'-m','finetune.pose_opd_credit.teacher_partner_benchmark','--config','finetune/pose_opd_credit/recipe.teacher_partner.json','--output',RUN+'/'+name,'--instances','24','--frames','375','--capacity-frames','1500','--repeats','2','--teacher-replay','block','--profile-calls','0','--verify-block','--student-devices',student,'--partner-devices',partner,'--pack-partner-projections',packed]
    jobs.append({'name':name,'argv':['bash','-c','source '+shlex.quote(ACT)+' && exec '+shlex.join(argv)]})
spec={'run':RUN,'jobs':jobs}
(ROOT/'spec.json').write_text(json.dumps(spec,indent=2))
code='import pathlib,tarfile,json,subprocess; r=pathlib.Path('+repr(RUN)+'); (r/"code").mkdir(); tarfile.open(r/"source.tar").extractall(r/"code"); (r/"worker.py").write_text('+repr(worker)+'); (r/"spec.json").write_text('+repr(json.dumps(spec))+'); p=subprocess.Popen(["python3",str(r/"worker.py")],stdout=open(r/"supervisor.log","w"),stderr=subprocess.STDOUT,start_new_session=True); (r/"supervisor.pid").write_text(str(p.pid)); print(json.dumps({"pid":p.pid,"run":str(r)}))'
print(remote(code))
