from pathlib import Path
import argparse,concurrent.futures,hashlib,json,shlex,subprocess,time
p=argparse.ArgumentParser();p.add_argument('node',type=int);args=p.parse_args()
node=f'landsend-{args.node}';key='/home/param/.ssh/id_ed25519_cluster'
src=Path('/home/param/models/imported-weights/ppm-aloe-zoom-cotrain/export-253')
out=Path('/home/param/outputs/pose-opd-general-audio-stage')/f'node{args.node}';out.mkdir(exist_ok=True)
ssh=['ssh','-i',key,'-o','BatchMode=yes',node]
base='/home/param/models/imported-weights/ppm-aloe-zoom-cotrain';final=base+'/export-253';stage=base+'/.export-253-staging-'+str(int(time.time()))
files=sorted(p for p in src.iterdir() if p.is_file() and p.name!='COMPLETE')
expected={p.name:p.stat().st_size for p in files}
verified=json.loads((src.parent/'HF_SHA256_VERIFICATION.json').read_text());hashes={Path(x['path']).name:x['actual_sha256'] for x in verified['files']}
manifest={'source':str(src),'stage':stage,'final':final,'files':expected,'sha256':hashes,'complete':(src/'COMPLETE').read_text(),'started':time.time()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
remote='import pathlib,shutil; p=pathlib.Path('+repr(base)+'); p.mkdir(parents=True,exist_ok=True); assert not pathlib.Path('+repr(final)+').exists(); assert shutil.disk_usage(p).free>'+str(sum(expected.values())+5*1024**3)+'; pathlib.Path('+repr(stage)+').mkdir()'
subprocess.run(ssh+[shlex.join(['python3','-c',remote])],check=True)
parts=[[] for _ in range(4)]
for f in sorted(files,key=lambda p:p.stat().st_size,reverse=True):parts[min(range(4),key=lambda i:sum(expected[n] for n in parts[i]))].append(f.name)
def transfer(i):
 listfile=out/f'files-{i}.txt';listfile.write_text('\n'.join(parts[i])+'\n')
 cmd=['rsync','-a','--whole-file','--inplace','--partial','--files-from='+str(listfile),'-e','ssh -i '+key+' -o BatchMode=yes',str(src)+'/',node+':'+stage+'/']
 with (out/f'rsync-{i}.log').open('wb') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(transfer,range(4)))
subprocess.run(['scp','-i',key,str(out/'manifest.json'),node+':'+stage+'/TRANSFER_EXPECTED.json'],check=True)
verify=r'''
from pathlib import Path
import concurrent.futures,hashlib,json,os,sys,time
stage=Path(sys.argv[1]);d=json.loads((stage/'TRANSFER_EXPECTED.json').read_text())
assert len(list(stage.glob('rank*.pt')))==128
m=json.loads((stage/'meta.json').read_text());assert m['step']==253 and m['world']==128 and m['run_tag']=='ppm-aloe-zoom-cotrain-j26467'
def check(name):
 p=stage/name;assert p.stat().st_size==d['files'][name]
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 assert h.hexdigest()==d['sha256'][name],name
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(check,d['files']))
(stage/'COMPLETE').write_text(d['complete']);assert not Path(d['final']).exists();os.rename(stage,d['final'])
print(json.dumps({'node':os.uname().nodename,'path':d['final'],'verified_files':len(d['files']),'verified_full_sha256':True,'step':253,'seconds':time.time()-d['started']}))
'''
result=subprocess.check_output(ssh+[shlex.join(['python3','-',stage])],input=verify,text=True)
(out/'receipt.json').write_text(result);print(result,flush=True)
