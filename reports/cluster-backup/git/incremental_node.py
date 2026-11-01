"""Take temporary-index snapshots and bundle only differences from last verified backup."""
import json,sys,os,hashlib,subprocess,socket
from pathlib import Path
BASE=Path('/home/param/outputs/landsend-backup-20261031/git');run_id=sys.argv[1];baseline=json.loads(Path(sys.argv[2]).read_text());HOST=socket.gethostname().split('.')[0]
OUT=BASE/'incremental'/run_id/HOST;OUT.mkdir(parents=True,exist_ok=True)
source=(BASE/'snapshot_repositories.py').read_text().split('\npaths=json.loads(sys.argv[1])')[0]
namespace={};exec(source,namespace);namespace['OUT']=OUT
snapshot=namespace['snapshot'];git=namespace['run'];text=namespace['strgit'];REF=namespace['BRANCH']
old={x['common']:x for x in baseline};paths={w['path'] for x in baseline for w in x['worktrees']}
skip=namespace['SKIP']|{'.codex','.ssh','.gnupg','.hf-cli','.local','.nv','.triton','datasets','runs','pose-runs','outputs'}
for root in ['/home/param','/tmp']:
 for base,ds,fs in os.walk(root,followlinks=False):
  try:owned=os.stat(base).st_uid==os.getuid()
  except OSError:ds[:]=[];continue
  if not owned and base!='/tmp':ds[:]=[];continue
  if '.git' in fs or Path(base,'.git').is_dir():paths.add(base)
  ds[:]=[d for d in ds if d not in skip and not Path(base,d).is_symlink()]
rows=[];errors=[]
for p in sorted(paths):
 if not Path(p).exists():errors.append({'path':p,'error':'previous worktree missing'});continue
 try:
  row=snapshot(p);row['staged_tree']=text(p,'rev-parse',(row['staged_commit'] or row['original_head'])+'^{tree}');rows.append(row)
 except Exception as e:errors.append({'path':p,'error':str(e)})
groups={}
for row in rows:groups.setdefault(row['common'],[]).append(row)
results=[]
for common,worktrees in groups.items():
 prior=old.get(common,{});prior_trees={x['path']:x for x in prior.get('worktrees',[])};repo=worktrees[0]['path']
 refs=text(repo,'for-each-ref','refs/heads','refs/tags','--format=%(refname) %(objectname)').splitlines();refs=[x for x in refs if not x.startswith(REF+' ')]
 oldrefs=[x for x in prior.get('local_branches',[]) if not x.startswith(REF+' ')]
 changed=[]
 for row in worktrees:
  prev=prior_trees.get(row['path'])
  if prev and 'staged_tree' not in prev:prev['staged_tree']=text(repo,'rev-parse',(prev['staged_commit'] or prev['original_head'])+'^{tree}')
  if not prev or any(row[k]!=prev[k] for k in ['tree','staged_tree','original_head','branch']):changed.append(row['path'])
  else:
   # Reuse already published identities for unchanged snapshots; fresh temporary commits were never pushed.
   row['working_commit']=prev['working_commit'];row['staged_commit']=prev['staged_commit']
 change=bool(changed or sorted(refs)!=sorted(oldrefs) or len(prior_trees)!=len(worktrees))
 result={'host':HOST,'repo':repo,'common':common,'worktrees':worktrees,'local_branches':refs,'changed':change,'changed_worktrees':changed}
 if change:
  parents=[x['working_commit'] for x in worktrees]
  for ref in refs:
   commit=text(repo,'rev-parse',ref.split()[0]+'^{commit}',check=False)
   if commit:parents.append(commit)
  prior_ref=text(repo,'rev-parse','--verify',REF,check=False)
  if prior_ref:parents.append(prior_ref)
  args=[]
  for p in dict.fromkeys(parents):args.extend(['-p',p])
  primary=next((x for x in worktrees if x['path']=='/home/param/pose-repo'),worktrees[0])
  commit=text(repo,'commit-tree',primary['tree'],*args,data=('Incremental landsend snapshot '+run_id+' '+HOST+'\n').encode());ref='refs/landsend-backup/incremental/'+run_id;git(repo,'update-ref',ref,commit)
  label=hashlib.sha256(common.encode()).hexdigest()[:12];bundle=OUT/(label+'.bundle');git(repo,'bundle','create',str(bundle),ref)
  result.update(commit=commit,bundle=str(bundle),bundle_ref=ref)
 results.append(result)
report={'host':HOST,'run_id':run_id,'groups':results,'errors':errors}
(OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'host':HOST,'groups':len(results),'changed_groups':sum(x['changed'] for x in results),'errors':errors}))
