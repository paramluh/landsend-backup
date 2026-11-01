import subprocess,concurrent.futures,json,shlex,time,os
from pathlib import Path
ROOT=Path('/home/param/outputs/landsend-backup-20261031/remote-datasets');HOSTS=['landsend-1','landsend-2','landsend-3','landsend-4','landsend-5','landsend-12'];SSH=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','ConnectTimeout=10'];RSYNC_SSH='ssh -i /home/param/.ssh/id_ed25519_cluster -o BatchMode=yes -o ConnectTimeout=10'
def probe(host):
 p=subprocess.run(SSH+[host,'test -f '+str(ROOT/'SUMMARY.json')],capture_output=True,timeout=20)
 return host,p.returncode==0
pending=set(HOSTS)
while pending:
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  for host,done in pool.map(probe,sorted(pending)):
   if done:
    out=ROOT/host;out.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','-e',RSYNC_SSH,'--include=SUMMARY.json','--include=verification.json','--include=upload-required.json','--exclude=*',host+':'+str(ROOT)+'/',str(out)+'/'],check=True)
    pending.remove(host);print(json.dumps({'verification_collected':host}),flush=True)
 if pending:time.sleep(15)
known={};plans=[]
for host in HOSTS:
 out=ROOT/host;missing=json.loads((out/'upload-required.json').read_text());selected=[];aliases=[]
 for row in missing:
  key=(row['bytes'],row['algorithm'],row['digest'])
  if key in known:
   source=known[key];aliases.append({'path':row['path'],'relative':row['relative'],'bytes':row['bytes'],'algorithm':row['algorithm'],'digest':row['digest'],'owner_host':source['host'],'owner_relative':source['relative']})
  else:known[key]={'host':host,'relative':row['relative']};selected.append(row)
 (out/'upload-selected.json').write_text(json.dumps(selected,indent=2)+'\n');(out/'cross-node-aliases.json').write_text(json.dumps(aliases,indent=2)+'\n')
 summary=json.loads((out/'SUMMARY.json').read_text());plan={'host':host,'files':summary['files'],'matched':summary['matched'],'errors':summary['errors'],'upload_files':len(selected),'upload_bytes':sum(r['bytes'] for r in selected),'cross_node_aliases':len(aliases)};plans.append(plan);print(json.dumps(plan),flush=True)
(ROOT/'upload-plan.json').write_text(json.dumps(plans,indent=2)+'\n')
def launch(host):
 out=ROOT/host
 subprocess.run(['scp','-q','-i','/home/param/.ssh/id_ed25519_cluster',str(ROOT/'upload_remainder.py'),str(out/'upload-selected.json'),str(out/'cross-node-aliases.json'),host+':'+str(ROOT)+'/'],check=True)
 code="import subprocess,json,time,os;from pathlib import Path;r=Path("+repr(str(ROOT))+");log=(r/'upload.log').open('ab',buffering=0);p=subprocess.Popen(['ionice','-c','3','nice','-n','10','/home/param/pose-repo/.venv/bin/python',str(r/'upload_remainder.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'HF_HUB_DISABLE_PROGRESS_BARS':'1'});d={'pid':p.pid,'started_unix':time.time()};(r/'upload-process.json').write_text(json.dumps(d));print(json.dumps(d))"
 p=subprocess.run(SSH+[host,'python3 -c '+shlex.quote(code)],capture_output=True,text=True,check=True)
 return {'host':host,**json.loads(p.stdout)}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:receipts=list(pool.map(launch,HOSTS))
(ROOT/'upload-processes.json').write_text(json.dumps(receipts,indent=2)+'\n');print(json.dumps({'uploads_launched':receipts}),flush=True)
