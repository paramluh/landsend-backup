"""Stage a full56GPU one-update smoke, launch only with explicit --launch."""
import argparse, concurrent.futures, hashlib, json, os, shlex, subprocess, tarfile, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=Path('/home/param/pose-worktrees/pose-opd-credit')
HOSTS=['landsend-0','landsend-1','landsend-2','landsend-3','landsend-4','landsend-5','landsend-12']
SSH=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10']
SCP=['scp','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10']
def remote(host,argv):
    return subprocess.run(SSH+[host,shlex.join(list(map(str,argv)))],capture_output=True,text=True,check=True,timeout=180).stdout

def main():
    p=argparse.ArgumentParser();p.add_argument('--bias',type=float,default=0);p.add_argument('--launch',action='store_true');a=p.parse_args()
    stamp=int(time.time());local=HERE/str(stamp);local.mkdir()
    name=f'pose-opd-symmetric-train-smoke-{stamp}'
    base='/home/param/runs/pose-runs/'+name
    names=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=REPO).decode().split('\0')
    with tarfile.open(local/'source.tar','w') as archive:
        for name in sorted(set(names)-{''}):
            path=REPO/name
            if path.is_file(): archive.add(path,arcname=name,recursive=False)
    cfg=json.loads((REPO/'finetune/pose_opd_credit/recipe.teacher_partner.json').read_text())
    cfg.update(partner_pad_bias=a.bias)
    if (cfg['student'].split('/')[-1]!='export-211' or cfg['partner_conversation_mode']!='symmetric_streaming' or cfg['partner_sampling']!='website_20261031' or 56*cfg['effective_batch_per_rank']!=336 or cfg['conversation_prefix_frames']!=1500):
        raise RuntimeError('Wrongsmokerecipe')
    (local/'recipe.json').write_text(json.dumps(cfg,indent=2)+'\n')
    spec={'hosts':HOSTS,'remote':base,'training_run':base+'/run','max_updates':1,'no_save':True,
        'python':'/home/param/pose-repo/.venv/bin/python',
        'activate':'/home/param/pose-repo/finetune/transcript_pose/environment/activate.sh',
        'master_addr':'fdf9:c316:93ee:0:5f4:1538:2b0b:11d8','port':29719,
        'source_sha256':hashlib.sha256((local/'source.tar').read_bytes()).hexdigest(),
        'worker_sha256':hashlib.sha256((HERE/'worker.py').read_bytes()).hexdigest(),
        'source_git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'source_git_status':subprocess.check_output(['git','status','--short'],cwd=REPO,text=True),
        'config_sha256':hashlib.sha256((local/'recipe.json').read_bytes()).hexdigest(),
        'partner_sampling':cfg['partner_sampling'],'partner_pad_bias':a.bias,'global_batch':336,'frames':1500,'local_receipts':str(local)}
    (local/'spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    (HERE/'LATEST.json').write_text(json.dumps({'local':str(local),'remote':base},indent=2)+'\n')
    def stage(host):
        remote(host,['mkdir','-p',base+'/code'])
        for src in (local/'source.tar',local/'recipe.json',local/'spec.json',HERE/'worker.py'):
            subprocess.run(SCP+[str(src),host+':'+base+'/'],check=True,capture_output=True,timeout=180)
        remote(host,['tar','-xf',base+'/source.tar','-C',base+'/code'])
        return host
    with concurrent.futures.ThreadPoolExecutor(7) as pool:
        print(json.dumps({'staged':list(pool.map(stage,HOSTS)),'spec':spec}),flush=True)
    if not a.launch: return
    def start(host):
        script='import json,subprocess;from pathlib import Path;p=Path('+repr(base)+');o=(p/"supervisor.log").open("ab",buffering=0);c=subprocess.Popen(["/home/param/pose-repo/.venv/bin/python",str(p/"worker.py"),str(p/"spec.json")],stdin=subprocess.DEVNULL,stdout=o,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({"pid":c.pid}))'
        text=remote(host,[spec['python'],'-c',script]);(local/(host+'-started.json')).write_text(text)
    with concurrent.futures.ThreadPoolExecutor(7) as pool:list(pool.map(start,HOSTS))
    deadline=time.monotonic()+120
    while True:
        def ready(host):
            try:return json.loads(remote(host,['cat',base+'/READY.json']))
            except subprocess.CalledProcessError:return None
        with concurrent.futures.ThreadPoolExecutor(7) as pool: values=list(pool.map(ready,HOSTS))
        if all(values): break
        if time.monotonic()>deadline: raise RuntimeError('All-hostreadytimeout;workerswillselfexpire')
        time.sleep(1)
    go={'time':time.time(),'source_sha256':spec['source_sha256'],'config_sha256':spec['config_sha256']}
    (local/'GO.json').write_text(json.dumps(go)+'\n')
    with concurrent.futures.ThreadPoolExecutor(7) as pool:
        list(pool.map(lambda host:subprocess.run(SCP+[str(local/'GO.json'),host+':'+base+'/GO.json'],check=True,capture_output=True,timeout=60),HOSTS))
    (local/'LAUNCHED.json').write_text(json.dumps({'hosts':HOSTS,'spec':spec,'ready':values},indent=2)+'\n')
    print(json.dumps({'launched':True,'receipts':str(local),'remote':base}),flush=True)
if __name__=='__main__':main()
