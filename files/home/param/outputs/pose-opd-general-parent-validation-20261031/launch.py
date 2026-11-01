"""Launch one short compatibility check on idle node 5, never node 0."""
import io
import json
import pathlib
import shlex
import subprocess
import tarfile
import time

ROOT = pathlib.Path(__file__).parent
REPO = pathlib.Path('/home/param/pose-worktrees/pose-opd-credit')
NODE = 'landsend-5'
KEY = '/home/param/.ssh/id_ed25519_cluster'
RUN = '/home/param/runs/pose-runs/conversation-trials-' + str(int(time.time()))
PY = '/home/param/pose-repo/.venv/bin/python'
ACT = '/home/param/pose-repo/finetune/transcript_pose/environment/activate.sh'
SSH = ['ssh', '-i', KEY, '-o', 'BatchMode=yes', NODE]


def remote(code):
    return subprocess.check_output(SSH + [shlex.join(['python3', '-c', code])], text=True)


archive = subprocess.check_output(['git', 'archive', 'HEAD'], cwd=REPO)
with tarfile.open(fileobj=io.BytesIO(archive)) as src, tarfile.open(ROOT / 'source.tar', 'w') as dst:
    for item in src:
        if item.isfile():
            data = (REPO / item.name).read_bytes()
            item.size = len(data)
            dst.addfile(item, io.BytesIO(data))
        else:
            dst.addfile(item)
    for name in subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=REPO, text=True).splitlines():
        if name.startswith('finetune/pose_opd_credit/') and name.endswith('.py'):
            dst.add(REPO / name, arcname=name)
remote(f'import pathlib; pathlib.Path({RUN!r}).mkdir(parents=True)')
subprocess.run(['scp', '-i', KEY, *(str(ROOT / name) for name in ('source.tar', 'prepared_rows.pkl', 'seeds.json')), f'{NODE}:{RUN}/'], check=True)
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
    if apps: raise RuntimeError('GPU processes present; not displacing them: '+apps)
    spec=json.loads((run/'spec.json').read_text())
    for item in spec['jobs']:
        p=subprocess.Popen(item['argv'],cwd=run/'code',stdout=open(run/(item['name']+'.log'),'w'),stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p)
    (run/'children.json').write_text(json.dumps([{'pid':p.pid,'start_ticks':pathlib.Path('/proc/%d/stat'%p.pid).read_text().rsplit(')',1)[1].split()[19]} for p in children]))
    codes=[p.wait() for p in children]
    (run/'result.json').write_text(json.dumps({'codes':codes,'finished':time.time()}))
'''
jobs = []
for name, student, partner, offset in [('compatibility', '0,1', '2,3', 0)]:
    argv = [PY, '-m', 'finetune.pose_opd_credit.teacher_partner_benchmark',
        '--config', 'finetune/pose_opd_credit/recipe.teacher_partner.json', '--output', RUN + '/' + name,
        '--instances', '1', '--frames', '96', '--capacity-frames', '1500', '--repeats', '1',
        '--teacher-replay', 'block', '--profile-calls', '0', '--student-devices', student,
        '--partner-devices', partner, '--mimi-placement', 'frontend', '--row-offset', str(offset),
        '--prepared-rows', RUN + '/prepared_rows.pkl', '--rollout-seeds', RUN + '/seeds.json', '--save-conversations']
    jobs.append({'name': name, 'argv': ['bash', '-c', 'source ' + shlex.quote(ACT) + ' && exec ' + shlex.join(argv)]})
spec = {'node': NODE, 'run': RUN, 'jobs': jobs}
(ROOT / 'spec.json').write_text(json.dumps(spec, indent=2))
code = 'import pathlib,tarfile,json,subprocess; r=pathlib.Path(' + repr(RUN) + '); (r/"code").mkdir(); tarfile.open(r/"source.tar").extractall(r/"code"); (r/"worker.py").write_text(' + repr(worker) + '); (r/"spec.json").write_text(' + repr(json.dumps(spec)) + '); p=subprocess.Popen(["python3",str(r/"worker.py")],stdout=open(r/"supervisor.log","w"),stderr=subprocess.STDOUT,start_new_session=True); (r/"supervisor.pid").write_text(str(p.pid)); print(json.dumps({"pid":p.pid,"run":str(r)}))'
print(remote(code))
