import concurrent.futures,json,pathlib,shlex,subprocess,time
from huggingface_hub import HfApi
ROOT=pathlib.Path('/home/param/outputs/landsend-backup-20261031/checkpoints');REPO='InternalCan/landsend-checkpoints-backup-param'
HOSTS=['landsend-0','landsend-1','landsend-2','landsend-3','landsend-4','landsend-5','landsend-12']
PROBE=r'''
import json,pathlib,socket
r=pathlib.Path('/home/param/outputs/landsend-backup-20261031/checkpoints');host=socket.gethostname().split('.')[0]
p=r/(host+'-uploaded.jsonl');rows=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
result=[]
for row in rows:
 source=pathlib.Path(row['source']);expected={}
 for n in ['train_state.json','meta.json']:
  file=source/n
  if not file.exists():continue
  try:m=json.loads(file.read_text())
  except Exception:continue
  if not isinstance(m.get('files'),list):continue
  for f in m['files']:
   if not isinstance(f,dict) or not f.get('sha256') or f.get('host',host)!=host:continue
   if f.get('section') and f['section']!=source.name:continue
   if (source/f['file']).is_file():expected[row['prefix']+'/'+f['file']]=f['sha256']
 result.append({**row,'expected_hashes':expected})
print(json.dumps(result))
'''
def one(host):
 command=['python3','-'] if host=='landsend-0' else ['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3 -']
 p=subprocess.run(command,input=PROBE,capture_output=True,text=True,timeout=30);p.check_returncode();return json.loads(p.stdout)
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:rows=[x for group in pool.map(one,HOSTS) for x in group]
expected={k:v for row in rows for k,v in row['expected_hashes'].items()}
api=HfApi();revision=api.repo_info(REPO).sha;actual={};paths=list(expected)
for i in range(0,len(paths),100):
 for f in api.get_paths_info(REPO,paths[i:i+100],revision=revision):
  actual[f.path]=f.lfs.get('sha256') if isinstance(f.lfs,dict) else getattr(f.lfs,'sha256',None)
mismatches=[{'path':p,'expected':sha,'actual':actual.get(p)} for p,sha in expected.items() if actual.get(p)!=sha]
result={'repo':REPO,'revision':revision,'checked':time.time(),'uploaded_leaf_count':len(rows),'manifest_hash_files':len(expected),'manifest_hash_bytes':sum(row['bytes'] for row in rows if row['expected_hashes']),'mismatches':mismatches,'verified':not mismatches,'leaf_checks':[{k:v for k,v in row.items() if k!='expected_hashes'}|{'hash_files':len(row['expected_hashes'])} for row in rows]}
(ROOT/'manifest-hash-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='leaf_checks'},indent=2))
