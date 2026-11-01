import json,hashlib,concurrent.futures,time,os
from pathlib import Path
BASE=Path('/home/param/outputs/landsend-backup-20261031');SOURCE=BASE/'orphans';OUT=BASE/'remote-datasets'
proof=json.loads((SOURCE/'verified-existing-datasets.json').read_text());rows=json.loads((SOURCE/'dataset-selection.json').read_text());catalog={}
for v in proof['verified']:
 relative=str(Path(v['local']).relative_to('/home/param/datasets'))
 catalog[relative]={'relative':relative,'bytes':v['bytes'],'algorithm':'sha256' if v.get('remote_sha256') else 'git_blob_sha1','digest':v.get('remote_sha256') or v['remote_blob'],'storage':{'kind':'original_hf','repo':v['repo'],'revision':v['revision'],'path':v['path']}}
remaining=[r for r in rows if str(Path(r['path']).relative_to('/home/param/datasets')) not in catalog]
start=time.time();done=0;size=0

def one(r):
 p=Path(r['path']);before=p.stat()
 with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 after=p.stat()
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Dataset changed while hashing: '+str(p))
 return {'relative':str(p.relative_to('/home/param/datasets')),'bytes':before.st_size,'algorithm':'sha256','digest':digest,'storage':{'kind':'node0_dataset_archive','repo':'InternalCan/landsend-artifacts-backup-param','prefix':'landsend-0/datasets','source_path':str(p)},'mtime_ns':before.st_mtime_ns}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for value in pool.map(one,remaining):
  catalog[value['relative']]=value;done+=1;size+=value['bytes']
  if done%1000==0:print(json.dumps({'derived_processed':done,'derived_total':len(remaining),'GiB':size/2**30,'seconds':time.time()-start}),flush=True)
result={'format':'landsend-dataset-baseline-v1','created_unix':time.time(),'existing_hf_proof':str(SOURCE/'verified-existing-datasets.json'),'entries':list(catalog.values()),'original_hf_files':len(proof['verified']),'node0_derived_files':len(remaining),'node0_derived_bytes':size}
(OUT/'node0-catalog.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
print(json.dumps({'state':'complete','files':len(catalog),'derived_files':len(remaining),'derived_GiB':size/2**30,'seconds':time.time()-start}),flush=True)
