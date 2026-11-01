"""Local integration check: appended mirror ancestry, LFS conversion, multi-worktree mapping."""
import ast,os,subprocess,json,hashlib,tempfile
from pathlib import Path
BASE=Path('/home/param/outputs/landsend-backup-20261031/git');test=Path(tempfile.mkdtemp(prefix='mirror-test-',dir=BASE));source=test/'source';mirror=test/'hf-pose-mirror';remote=test/'remote.git';out=test/'out';out.mkdir()
env={**os.environ,'GIT_AUTHOR_NAME':'Backup test','GIT_AUTHOR_EMAIL':'backup-test@localhost','GIT_COMMITTER_NAME':'Backup test','GIT_COMMITTER_EMAIL':'backup-test@localhost'}
def command(*args,data=None):return subprocess.check_output(args,input=data,env=env,stderr=subprocess.DEVNULL).strip()
command('git','init',str(source));(source/'readme').write_text('baseline\n');command('git','-C',str(source),'add','.');command('git','-C',str(source),'commit','-m','baseline');old=command('git','-C',str(source),'rev-parse','HEAD').decode();command('git','clone',str(source),str(mirror));command('git','init','--bare',str(remote));ref='refs/heads/tmp/landsend-backup-param';command('git','-C',str(mirror),'push',str(remote),old+':'+ref)
data=b'large-test-payload\n'*700000;(source/'tokenizer.json').write_bytes(data);command('git','-C',str(source),'add','.');command('git','-C',str(source),'commit','-m','large tokenizer');new=command('git','-C',str(source),'rev-parse','HEAD').decode()
module=ast.parse((BASE/'incremental_backup.py').read_text());defs=ast.Module(body=[n for n in module.body if isinstance(n,(ast.FunctionDef,ast.Import,ast.ImportFrom))],type_ignores=[]);ns={'BASE':test,'OUT':out,'RUN':'test','REF':ref,'ENV':env,'HELPER':'credential.helper='};exec(compile(defs,'functions','exec'),ns)
original=ns['git']
def git(repo,*args,**kw):return original(repo,*[str(remote) if a=='https://huggingface.co/InternalCan/pose-repo' else a for a in args],**kw)
ns['git']=git
# Local bare Git has no LFS server; upload the Git pointer with hooks disabled and inspect the local LFS object separately.
def push(repo,url,commit):
 command('git','-C',str(repo),'-c','core.hooksPath=/dev/null','push',str(remote),commit+':'+ref)
 assert command('git','--git-dir='+str(remote),'rev-parse',ref).decode()==commit
 return {'url':'local-test','commit':commit,'verified':True}
ns['push']=push
rows=[{'changed':True,'worktrees':[{'working_commit':new,'host':'landsend-0','path':'/home/param/pose-repo'},{'working_commit':old,'host':'landsend-1','path':'/home/param/pose-repo'}]}]
r=ns['hf_mirror'](source,new,rows);commit=r['commit'];command('git','-C',str(mirror),'merge-base','--is-ancestor',old,commit);pointer=command('git','-C',str(mirror),'show',commit+':tokenizer.json');digest=hashlib.sha256(data).hexdigest();assert ('oid sha256:'+digest).encode() in pointer;assert ('size '+str(len(data))).encode() in pointer
mapping=json.loads(command('git','-C',str(mirror),'show',commit+':LANDSEND_HF_INCREMENTAL_MAP.json'));assert len(mapping)==2 and {m['original'] for m in mapping}=={old,new};assert (mirror/'.git/lfs/objects'/digest[:2]/digest[2:4]/digest).read_bytes()==data
report={'state':'passed','checks':['fast_forward_prior_mirror','large_blob_converted_to_lfs','lfs_object_exact_bytes','two_worktree_commit_map','git_push_ref_verified'],'test_directory':str(test)};(BASE/'incremental-mirror-test.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
