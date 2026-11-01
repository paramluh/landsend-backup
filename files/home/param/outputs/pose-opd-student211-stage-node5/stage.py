from pathlib import Path
import concurrent.futures, hashlib, json, subprocess, time

src=Path('/home/param/models/imported-weights/ppm-aloe-poseduplex-raw-final/export-211')
out=Path('/home/param/outputs/pose-opd-student211-stage-node5')
key='/home/param/.ssh/id_ed25519_cluster'
ssh=['ssh','-i',key,'-o','BatchMode=yes','landsend-5']
base='/home/param/models/imported-weights/ppm-aloe-poseduplex-raw-final'
stage=base+'/export-211.staging-codex-'+str(int(time.time()))
final=base+'/export-211'
files=sorted(p for p in src.iterdir() if p.is_file() and p.name!='COMPLETE')
meta=json.loads((src/'meta.json').read_text())
assert meta['step']==211 and meta['world']==64
expected={p.name:p.stat().st_size for p in files}
small={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.stat().st_size<10_000_000}
manifest={'source':str(src),'stage':stage,'final':final,'files':expected,'sha256_small':small,'complete':(src/'COMPLETE').read_text(),'started':time.time()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
subprocess.run(ssh+['mkdir -p '+stage],check=True)
print(json.dumps({'event':'begin','stage':stage,'bytes':sum(expected.values()),'streams':4}),flush=True)
parts=[[] for _ in range(4)]
for p in sorted(files,key=lambda p:p.stat().st_size,reverse=True):
    bucket=min(range(4),key=lambda i:sum(expected[n] for n in parts[i]))
    parts[bucket].append(p.name)
def transfer(i):
    listfile=out/f'files-{i}.txt';listfile.write_text('\n'.join(parts[i])+'\n')
    cmd=['rsync','-a','--whole-file','--inplace','--partial','--info=progress2','--outbuf=L','--files-from='+str(listfile),
         '-e','ssh -i '+key+' -o BatchMode=yes',str(src)+'/', 'landsend-5:'+stage+'/']
    with (out/f'rsync-{i}.log').open('wb') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({'event':'stream_complete','stream':i,'seconds':time.time()-manifest['started']}),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(transfer,range(4)))
subprocess.run(['scp','-i',key,str(out/'manifest.json'),'landsend-5:'+stage+'/TRANSFER_EXPECTED.json'],check=True)
verify=r'''
from pathlib import Path
import hashlib,json,os,sys
stage=Path(sys.argv[1]); expected=json.loads((stage/'TRANSFER_EXPECTED.json').read_text())
assert not (stage/'COMPLETE').exists()
for name,size in expected['files'].items():
    assert (stage/name).is_file() and (stage/name).stat().st_size==size,(name,size)
assert len(list(stage.glob('rank*.pt')))==64
assert (stage/'rep.pt').is_file()
m=json.loads((stage/'meta.json').read_text());assert m['step']==211 and m['world']==64
for name,digest in expected['sha256_small'].items():
    assert hashlib.sha256((stage/name).read_bytes()).hexdigest()==digest,name
final=Path(expected['final']);assert not final.exists(),str(final)
(stage/'COMPLETE').write_text(expected['complete'])
os.rename(stage,final)
print(json.dumps({'published':str(final),'step':m['step'],'world':m['world'],'rank_shards':64,'bytes':sum(expected['files'].values()),'verified':'rsync transfer checksums + exact sizes + small-file SHA256'}))
'''
result=subprocess.check_output(ssh+['/home/param/pose-repo/.venv/bin/python - '+stage],input=verify,text=True)
(out/'receipt.json').write_text(result)
print(result,flush=True)
print(json.dumps({'event':'complete','seconds':time.time()-manifest['started']}),flush=True)
