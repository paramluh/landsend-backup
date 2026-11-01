import os,json,subprocess,concurrent.futures,hashlib,time
from pathlib import Path
from huggingface_hub import HfApi
ROOT=Path('/home/param/outputs/landsend-backup-20261031/remote-datasets');HOSTS=['landsend-1','landsend-2','landsend-3','landsend-4','landsend-5','landsend-12'];REPO='InternalCan/landsend-artifacts-backup-param';SSH=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes'];RSYNC='ssh -i /home/param/.ssh/id_ed25519_cluster -o BatchMode=yes'
def collect(host):
 p=subprocess.run(SSH+[host,'test -f '+str(ROOT/'COMPLETE.json')],capture_output=True)
 if p.returncode:return host,False
 subprocess.run(['rsync','-a','-e',RSYNC,'--include=*/','--include=*.json','--exclude=*',host+':'+str(ROOT)+'/',str(ROOT/host)+'/'],check=True)
 return host,True
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:collected=dict(pool.map(collect,HOSTS))
if not all(collected.values()):print(json.dumps({'pending':[h for h,done in collected.items() if not done]}));raise SystemExit(2)
api=HfApi();revision=api.repo_info(REPO,repo_type='dataset').sha;catalog=json.loads((ROOT/'node0-catalog.json').read_text());catalog_sha=hashlib.sha256((ROOT/'node0-catalog.json').read_bytes()).hexdigest();node0proof=json.loads((ROOT/'node0-derived-archive-proof.json').read_text());assert node0proof['state']=='complete'
archives=list(node0proof['archives']);summaries=[];metadata=[];selected_all={};aliases_all=[]
for host in HOSTS:
 base=ROOT/host;report=json.loads((base/'verification.json').read_text());complete=json.loads((base/'COMPLETE.json').read_text());archive=json.loads((base/'upload/archives/datasets/COMPLETE.json').read_text());selected=json.loads((base/'upload-selected.json').read_text());aliases=json.loads((base/'cross-node-aliases.json').read_text());summary=json.loads((base/'SUMMARY.json').read_text())
 assert not report['errors'] and report['catalog_sha256']==catalog_sha
 assert complete['state']==archive['state']=='complete' and not archive['errors']
 assert len(selected)==complete['unique_files_uploaded'] and len(selected)+len(aliases)==len(report['upload_required'])
 manifest=[]
 for p in (base/'upload/archives/datasets').glob('*.files.json'):manifest.extend(json.loads(p.read_text()))
 by_path={r['path']:r for r in manifest}
 assert len(by_path)==len(selected)
 for row in selected:
  stored=by_path[row['path']];assert not stored.get('error') and not stored.get('redactions') and not stored.get('removed_members');assert row['bytes']==stored['archived_bytes'];assert row['mtime_ns']==stored['mtime_ns'];selected_all[(host,row['relative'])]=row
 for row in aliases:aliases_all.append((host,row))
 for item in complete['metadata']:
  p=base/item['file'];assert p.stat().st_size==item['bytes'] and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==item['sha256'];metadata.append({'hf_path':host+'/datasets/verification/'+item['file'],**item})
 archives.extend(archive['archives']);summaries.append({**summary,'unique_files_uploaded':len(selected),'unique_bytes_uploaded':sum(x['bytes'] for x in selected),'cross_node_aliases':len(aliases),'archives':len(archive['archives'])})
for host,alias in aliases_all:
 owner=selected_all[(alias['owner_host'],alias['owner_relative'])];assert all(owner[k]==alias[k] for k in ['bytes','algorithm','digest'])
checks=[]
for start in range(0,len(archives)+len(metadata),100):
 expected=(archives+metadata)[start:start+100];remote={x.path:x for x in api.get_paths_info(REPO,[r['hf_path'] for r in expected],repo_type='dataset',revision=revision)}
 for record in expected:
  item=remote[record['hf_path']];assert item.size==record['bytes'];sha=getattr(item.lfs,'sha256',None) if item.lfs else None
  if isinstance(item.lfs,dict):sha=item.lfs.get('sha256')
  if sha:assert sha==record['sha256']
  else:
   host=record['hf_path'].split('/')[0];p=ROOT/host/record['file'];data=p.read_bytes();assert item.blob_id==hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
  checks.append({'hf_path':record['hf_path'],'sha256':record['sha256'],'bytes':record['bytes'],'verified':True})
result={'state':'complete','hf_repo':REPO,'verified_at_revision':revision,'remote_nodes':len(HOSTS),'remote_regular_files':sum(r['files'] for r in summaries),'remote_bytes_hashed':sum(json.loads((ROOT/h/'verification.json').read_text())['bytes'] for h in HOSTS),'files_matched_to_existing_hf':sum(r['original_hf_matches'] for r in summaries),'files_matched_to_node0_archives':sum(r['node0_archive_matches'] for r in summaries),'unique_files_uploaded':len(selected_all),'unique_bytes_uploaded':sum(r['bytes'] for r in selected_all.values()),'cross_node_aliases':len(aliases_all),'symlinks_recorded':sum(r['symlinks'] for r in summaries),'errors':[],'nodes':summaries,'verified_uploaded_objects':checks,'completed_unix':time.time()}
(ROOT/'VERIFIED_REMOTE_DATASETS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in {'nodes','verified_uploaded_objects'}},indent=2))
