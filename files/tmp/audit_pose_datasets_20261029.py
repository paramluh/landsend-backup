import pathlib,json,subprocess,socket,concurrent.futures,datetime
OUT=pathlib.Path('/home/param/pose-runs/distillation-inventory-20261029')
PROBE=r'''
import pathlib,json,subprocess,socket,numpy as np
base=pathlib.Path('/home/param/datasets');out={'host':socket.gethostname(),'datasets':{}}
def du(path):return int(subprocess.run(['du','-sb',str(path)],capture_output=True,text=True,check=True).stdout.split()[0])
def sample(path):
 x=np.load(path,mmap_mode='r',allow_pickle=False);idx=tuple(slice(0,min(d,128)) for d in x.shape);s=x[idx]
 return {'path':str(path),'shape':list(x.shape),'dtype':str(x.dtype),'sample_min':int(s.min()) if s.size else None,'sample_max':int(s.max()) if s.size else None}
r=base/'xabi-v2pure-cache'
if r.exists():
 m=json.loads((r/'provenance/source-cache-manifest.json').read_text());bad=[]
 for e in m['files']:
  p=r/'cache'/e['path']
  if not p.is_file() or p.stat().st_size!=e['bytes']:bad.append(str(p))
 out['datasets']['xabi-v2pure-cache']={'present':True,'apparent_bytes':du(r),'manifest_files_size_checked':len(m['files']),'bad_files':bad,'existing_complete':json.loads((r/'COMPLETE.json').read_text()),'array_samples':[sample(r/'cache/eval/codes.npy'),sample(r/'cache/eval/pose.npy')]}
else:out['datasets']['xabi-v2pure-cache']={'present':False}
r=base/'aloe-duplex'
if r.exists():
 bad=[];checked=0;posefiles=0;posebytes=0
 for m,root in [(r/'SOURCE_MANIFEST.json',r),(r/'pose/SOURCE_MANIFEST.json',r/'pose')]:
  for e in json.loads(m.read_text())['files']:
   p=root/e['directory']/e['path'];checked+=1
   if not p.is_file() or p.stat().st_size!=e['size']:bad.append(str(p))
 for line in (r/'pose/hf/indexes/FILES.jsonl').read_text().splitlines():
  e=json.loads(line);p=r/'pose/data'/e['path'];posefiles+=1;posebytes+=e['bytes']
  if not p.is_file() or p.stat().st_size!=e['bytes']:bad.append(str(p))
 idx=json.loads((r/'pose/POSE_INDEX.json').read_text());stores=[]
 for s in idx['stores']:
  d=pathlib.Path(s);n=json.loads((d/'NODE.json').read_text())
  stores.append({'path':s,'files':len([p for p in d.iterdir() if p.is_file()]),'total_rows':n['total_rows'],'total_frames':n['total_frames'],'split_rows':n['split_rows'],'array_sample':sample(d/'rows-00000-of-00001.codes_assistant.npy')})
 p=next(e['path'] for rr in idx['rows'].values() for row in rr.values() for e in row['sides'].values() if e['ok'])
 out['datasets']['aloe-duplex']={'present':True,'apparent_bytes':du(r),'component_apparent_bytes':{p.name:du(p) for p in r.iterdir() if p.is_dir()},'source_files_size_checked':checked,'extracted_pose_files_size_checked':posefiles,'extracted_pose_bytes':posebytes,'bad_files':bad,'store_digests':idx['store_digests'],'stores':stores,'pose_array_sample':sample(r/'pose/data'/p),'existing_ready':json.loads((r/'DATASET_READY.json').read_text()),'existing_samsep_verified':json.loads((r/'s2h-m1-samsep/VERIFIED.json').read_text())}
else:out['datasets']['aloe-duplex']={'present':False}
print(json.dumps(out))
'''
def run(h):
 c=['/home/param/pose-repo/.venv/bin/python','-'] if h=='landsend-0' else ['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8',h,'/home/param/pose-repo/.venv/bin/python -']
 r=subprocess.run(c,input=PROBE,text=True,capture_output=True,timeout=90)
 if r.returncode:return {'host':h,'error':r.stderr,'returncode':r.returncode}
 return json.loads(r.stdout)
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as ex:results=list(ex.map(run,[f'landsend-{x}' for x in [0,1,2,3,4,5,12]]))
x={'audited_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Current /home/param/datasets on seven landsend nodes. Read-only size and structural audit; prior full hash verification reports preserved. Not a fresh full-content hash of all dataset payloads.','nodes':results}
(OUT/'dataset-node-audit.json').write_text(json.dumps(x,indent=2)+'\n')
for n in results:
 print(json.dumps({'host':n['host'],'error':n.get('error'),'datasets':{k:{key:d.get(key) for key in ['present','apparent_bytes','manifest_files_size_checked','source_files_size_checked','extracted_pose_files_size_checked','bad_files']} for k,d in n.get('datasets',{}).items()}}))
