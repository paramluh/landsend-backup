import argparse,json,os,pathlib,socket,time,traceback
from huggingface_hub import HfApi
ROOT=pathlib.Path('/home/param/outputs/landsend-backup-20261031/checkpoints')
REPO='InternalCan/landsend-checkpoints-backup-param'
def atomic(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
 plan=json.loads(pathlib.Path(a.plan).read_text());host=plan['host'];ROOT.mkdir(parents=True,exist_ok=True)
 statepath=ROOT/(host+'-upload-status.json');receipt=ROOT/(host+'-uploaded.jsonl')
 done={json.loads(x)['source'] for x in receipt.read_text().splitlines()} if receipt.exists() else set()
 api=HfApi(); assert api.repo_info(REPO).private
 started=time.time()
 for task in plan['tasks']:
  source=task['source']
  if source in done:continue
  folder=pathlib.Path(source);prefix='hosts/'+host+'/'+str(folder.relative_to('/home/param'))
  # Upload flat leaf contents, including small metadata and RNG files alongside weights.
  names=[x.name for x in folder.iterdir() if x.is_file() and not x.is_symlink()]
  expected={x.name:x.stat().st_size for x in folder.iterdir() if x.name in names}
  attempt=0
  while True:
   attempt+=1
   atomic(statepath,{'state':'uploading','host':host,'source':source,'prefix':prefix,'bytes':sum(expected.values()),'completed_tasks':len(done),'total_tasks':len(plan['tasks']),'attempt':attempt,'time':time.time(),'started':started})
   try:
    result=api.upload_folder(repo_id=REPO,folder_path=folder,path_in_repo=prefix,allow_patterns=names,ignore_patterns=['.cache/**'],commit_message=f'Backup {host}: {folder.parent.name}/{folder.name}')
    paths=[prefix+'/'+n for n in expected]
    actual={}
    for begin in range(0,len(paths),100):
     actual.update({x.path:x.size for x in api.get_paths_info(REPO,paths[begin:begin+100],revision=result.oid)})
    assert all(actual.get(prefix+'/'+name)==size for name,size in expected.items()),'Remote size verification failed'
    entry={'host':host,'source':source,'prefix':prefix,'revision':result.oid,'files':len(expected),'bytes':sum(expected.values()),'completed':time.time()}
    with receipt.open('a') as out:out.write(json.dumps(entry)+'\n');out.flush();os.fsync(out.fileno())
    done.add(source)
    print(json.dumps(entry),flush=True)
    break
   except Exception as e:
    atomic(statepath,{'state':'retrying','host':host,'source':source,'attempt':attempt,'error':f'{type(e).__name__}: {e}','completed_tasks':len(done),'total_tasks':len(plan['tasks']),'time':time.time()})
    print(traceback.format_exc(),flush=True);time.sleep(min(60,5*attempt))
 atomic(statepath,{'state':'complete','host':host,'completed_tasks':len(done),'total_tasks':len(plan['tasks']),'time':time.time(),'started':started})
if __name__=='__main__':main()
