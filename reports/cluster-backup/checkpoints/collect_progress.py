import concurrent.futures,json,pathlib,shlex,subprocess,time
ROOT=pathlib.Path('/home/param/outputs/landsend-backup-20261031/checkpoints')
HOSTS=['landsend-0','landsend-1','landsend-2','landsend-3','landsend-4','landsend-5','landsend-12']
def one(host):
 if host=='landsend-0':
  state=json.loads((ROOT/(host+'-upload-status.json')).read_text());p=ROOT/(host+'-uploaded.jsonl');records=p.read_text() if p.exists() else ''
 else:
  code='import pathlib,json; r=pathlib.Path('+repr(str(ROOT))+'); h='+repr(host)+'; p=r/(h+"-uploaded.jsonl"); print(json.dumps({"state":json.loads((r/(h+"-upload-status.json")).read_text()),"receipts":p.read_text() if p.exists() else ""}))'
  p=subprocess.run(['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=20)
  if p.returncode:return {'host':host,'state':'unreachable','error':p.stderr[-200:]}
  doc=json.loads(p.stdout);state=doc['state'];records=doc['receipts']
  (ROOT/(host+'-upload-status.json')).write_text(json.dumps(state,indent=2));(ROOT/(host+'-uploaded.jsonl')).write_text(records)
 entries=[json.loads(s) for s in records.splitlines()]
 state['completed_bytes']=sum(x['bytes'] for x in entries)
 return state
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:rows=list(pool.map(one,HOSTS))
 summary={'checked':time.time(),'states':rows,'completed_bytes':sum(r.get('completed_bytes',0) for r in rows),'all_complete':all(r.get('state')=='complete' for r in rows)}
 (ROOT/'progress.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2))
