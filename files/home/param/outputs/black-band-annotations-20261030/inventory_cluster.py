from pathlib import Path
import concurrent.futures,datetime,json,subprocess
PROBE=r'''
from pathlib import Path
import hashlib,json,socket
import pyarrow.parquet as pq
R=Path('/home/param/datasets')
out={'host':socket.gethostname(),'datasets':{},'metadata':{}}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def add(p):
 if not p.exists():return
 d={'bytes':p.stat().st_size,'sha256':digest(p)}
 if p.suffix=='.parquet':
  f=pq.ParquetFile(p);d.update(rows=f.metadata.num_rows,columns=f.schema_arrow.names)
 out['metadata'][str(p.relative_to(R))]=d
for p in sorted(R.iterdir()):
 if p.is_dir():out['datasets'][p.name]={'children':sorted(c.name for c in p.iterdir())}
for p in sorted((R/'aloe-duplex').glob('*/stores/*/rows.parquet')):add(p)
for rel in ['aloe-duplex/pose/POSE_INDEX.json','aloe-duplex/pose/M1VOX_INDEX_samsep.json','aloe-duplex/DATASET_READY.json','aloe-duplex/s2h-m1-samsep/VERIFIED.json','aloe-duplex/SOURCE_MANIFEST.json','aloe-duplex/pose/SOURCE_MANIFEST.json','xabi-v2pure-cache/COMPLETE.json','xabi-v2pure-cache/provenance/source-cache-manifest.json','xabi-v2pure-cache/provenance/cache-validation.json','blackband-repro-20261031/source/tools/shard_index.parquet','blackband-repro-20261031/source/index/conversations.parquet']:
 add(R/rel)
for p in sorted((R/'xabi-v2pure-cache/provenance/original/ppm/aloe-v2pure-ft/stores').glob('*/rows.parquet')):add(p)
items=list((R/'xabi-v2pure-cache/cache').glob('*/items.parquet'))
if items:
 counts=[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'rows':pq.ParquetFile(p).metadata.num_rows} for p in sorted(items)]
 out['datasets']['xabi-v2pure-cache'].update(items_shards=len(counts),items_rows=sum(x['rows'] for x in counts),items_file_sizes_rows_sha256=hashlib.sha256(json.dumps(counts,sort_keys=True).encode()).hexdigest())
print(json.dumps(out))
'''
def inspect(h):
 cmd=['/home/param/pose-repo/.venv/bin/python','-'] if h=='landsend-0' else ['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8',h,'/home/param/pose-repo/.venv/bin/python -']
 try:
  r=subprocess.run(cmd,input=PROBE,text=True,capture_output=True,timeout=120)
  return json.loads(r.stdout) if r.returncode==0 else {'host':h,'error':r.stderr,'returncode':r.returncode}
 except Exception as e:return {'host':h,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:nodes=list(pool.map(inspect,[f'landsend-{i}' for i in [0,1,2,3,4,5,12]]))
out={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Read-only metadata, no media/tokens read; xabi cache full file counts and parquet row counts, content hashes for relevant row metadata and manifests','nodes':nodes}
Path('/home/param/outputs/black-band-annotations-20261030/cluster-inventory.json').write_text(json.dumps(out,indent=2)+'\n')
for n in nodes:print(json.dumps({'host':n['host'],'error':n.get('error'),'datasets':{k:{a:v for a,v in x.items() if a!='children'} for k,x in n.get('datasets',{}).items()},'metadata_files':len(n.get('metadata',{}))}))
