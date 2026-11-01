"""Include every held run; post-exit context is preferred, never an eligibility veto."""
import concurrent.futures,hashlib,json,math,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pyarrow.ipc as ipc
BASE=Path('/home/param/datasets/blackband-repro-20261031');ROOT=Path('/home/param/runs/pose-runs/aloe-blackband-only-repro-20261031/data');HIST=Path('/home/param/datasets/aloe-duplex/pose/data')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
manifest=json.loads((ROOT/'manifest.json').read_text());scan=json.loads((ROOT/'static-band-scan.json').read_text())
old_manifest_sha=sha(ROOT/'manifest.json');old_scan_sha=sha(ROOT/'static-band-scan.json')
sources=manifest['sources'];items=[];missing={};counts={'train':0,'eval':0};run_n={'train':0,'eval':0};noexit={'train':0,'eval':0};short={'train':0,'eval':0};rowkeys=set()
for t in scan['tracks']:
 runs=t['runs_ge7_frames'];used=t['training_n_frames'];split=t['split'];sid=f'{t["session_id"]}__{t["seg"]}__{t["speaker_id"]}';prefix=f'ALOE/s2handover-pose/{t["session_id"]}/{t["seg"]}'
 selected={}
 for a,b in runs:
  if split=='train':
   for c in range(a//7500,(b-1)//7500+1):rowkeys.add((prefix+f'/c{c:02d}',t['b_side']))
  if used<750:short[split]+=1;continue
  start=min(max(b-375,0),used-750)
  assert min(b,start+750)-max(a,start)>=7
  selected.setdefault(start,[]).append([a,b]);run_n[split]+=1
 if selected and sid not in sources:missing[sid]=t
 for start,anchors in selected.items():
  bands=[[a,b] for a,b in runs if min(b,start+750)-max(a,start)>=7]
  item={'id':f'{sid}__f{start:07d}','source_id':sid,'split':split,'row_key':prefix+f'/c{start//7500:02d}','b_side':t['b_side'],'recording_id':t['session_id']+':'+str(int(t['seg'][3:])),'chunk_index':start//7500,'crop_start_frame':start,'crop_end_frame':start+750,'band_intervals_frames':bands,'band_intervals_local_frames':[[max(a,start)-start,min(b,start+750)-start] for a,b in bands],'anchor_band_intervals_frames':anchors,'anchor_postexit_context_frames':[start+750-b for a,b in anchors],'selection_evidence':'exact_all32_constant_pose_codes','target_masking':False,'original_training_rows_overlapped':[prefix+f'/c{c:02d}' for c in range(start//7500,(start+749)//7500+1)],'original_band_containing_rows':sorted({prefix+f'/c{c:02d}' for a,b in bands for c in range(a//7500,(b-1)//7500+1)})}
  items.append(item);counts[split]+=1
  if any(n<25 for n in item['anchor_postexit_context_frames']):noexit[split]+=1

def materialize(pair):
 sid,t=pair;shard=BASE/'source'/t['source_shard'];T=t['n_frames'];ri=t['row_index'];pref=f'speaker_{int(t["speaker_id"][-1]):02d}_'
 tab=pq.read_table(shard,columns=[pref+'pose_tokens',pref+'mimi_tokens',pref+'words'])
 local=HIST/t['relative_dir']/f'w00.{t["speaker_id"]}.pose_tokens.npy'
 def array(k):return tab[pref+k][ri].values.flatten().to_numpy().reshape(32,T)
 pose=np.load(local) if local.exists() else array('pose_tokens');mimi=array('mimi_tokens');assert pose.shape==mimi.shape==(32,T)
 target=ROOT/'sources'/sid;target.mkdir(parents=True,exist_ok=True);tmp=target/'tokens.tmp.npz';npz=target/'tokens.npz';np.savez(tmp,pose=pose.T,mimi=mimi.T);tmp.replace(npz)
 write(target/'words.json',tab[pref+'words'][ri].as_py() or [])
 source={k:v for k,v in t.items() if k not in ['runs_ge7_frames','valid_frames']}
 source.update(id=sid,npz=str(npz),words_json=str(target/'words.json'),npz_sha256=sha(npz),words_sha256=sha(target/'words.json'),pose_array_sha256=hashlib.sha256(pose.T.tobytes()).hexdigest(),mimi_array_sha256=hashlib.sha256(mimi.T.tobytes()).hexdigest(),row_key_prefix=f'ALOE/s2handover-pose/{t["session_id"]}/{t["seg"]}',source_parquet_sha256=sha(shard),source_revision=manifest['source_revision'],all_constant_pose_runs_frames=t['runs_ge7_frames'],band_definition='maximal exact ALL32-code run>=7tokens')
 return sid,source
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for sid,s in pool.map(materialize,missing.items()):sources[sid]=s
persona={}
for path in Path('/home/param/datasets/aloe-duplex/original/stores').glob('m1n*/*.persona.arrow'):
 reader=ipc.open_file(str(path))
 for b in range(reader.num_record_batches):
  for row in reader.get_batch(b).to_pylist():
   part=row['row_key'].split('/');persona[(part[-3],part[-2],int(part[-1][1:]))]={'source':str(path),'row_key':row['row_key'],'sides':{side:{k[len(side)+1:]:v for k,v in row.items() if k.startswith(side+'_')} for side in ['user','assistant']}}
for sid,s in sources.items():
 s['session']=s['session_id'];s['segment']=s['seg']
 s['persona_by_chunk']={str(c):persona[(s['session_id'],s['seg'],c)] for c in range(math.ceil(s['n_frames']/7500)) if (s['session_id'],s['seg'],c) in persona}
 write(ROOT/'sources'/sid/'source.json',s)
items.sort(key=lambda i:(i['split'],i['id']))
scan['selection_completion']={'created':time.time(),'source_script_sha256':sha(__file__),'prior_scan_sha256':old_scan_sha,'all_runs_considered':run_n,'skipped_only_source_shorter60s':short,'selected_crops':counts,'selected_crops_with_lt2s_postexit':noexit,'exact_duplicate_crops_merged':{s:run_n[s]-counts[s] for s in counts},'original_train_band_containing_rows':len(rowkeys),'policy':'All runs qualify; postexit context is preferred only. End-of-track runs are included. Full-source continuous60s windows may cross historical600s store chunk boundaries; those row IDs are recorded per item.'}
scan['counts']['train_selected_crops']=counts['train'];scan['counts']['eval_selected_crops']=counts['eval'];scan['counts']['train_run_skipped_no2s_postexit']=0
write(ROOT/'static-band-scan.json',scan)
manifest.update(created=time.time(),sources=sources,items=items,scan_report_sha256=sha(ROOT/'static-band-scan.json'),counts={'train_crops':counts['train'],'eval_crops':counts['eval'],'selected_sources':len(sources)},crop_policy='One60s window per maximal ALL32 static run>=7tokens, normally30s before/after exit, clamped tovalidsource; exactduplicatewindowsdeduplicated. Postexitcontext ispreferred,neverrequired. Only source shorter750frames excluded. No bandmask targets.',selection_completion=scan['selection_completion'])
write(ROOT/'manifest.json',manifest)
write(ROOT/'preparation-status.json',{'state':'complete','manifest_sha256':sha(ROOT/'manifest.json'),'manifest':str(ROOT/'manifest.json'),'corrected_no_context_exclusion':True,'prior_manifest_sha256':old_manifest_sha,'counts':manifest['counts'],'updated':time.time()})
print(json.dumps({'state':'complete','manifest_sha256':sha(ROOT/'manifest.json'),'counts':manifest['counts'],'selection_completion':scan['selection_completion']}),flush=True)
