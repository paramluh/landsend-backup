import concurrent.futures,hashlib,json,math,time
from pathlib import Path
import numpy as np
ROOT=Path('/home/param/runs/pose-runs/aloe-blackband-only-repro-20261031/data')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((ROOT/'manifest.json').read_text());scan=json.loads((ROOT/'static-band-scan.json').read_text())
assert manifest['scan_report_sha256']==sha(ROOT/'static-band-scan.json')
assert manifest['pose_loss_masking'] is False and manifest['no_targets_masked_for_bands'] is True
items=manifest['items'];assert len({i['id'] for i in items})==len(items)
by_source={s:[] for s in manifest['sources']}
for item in items:by_source[item['source_id']].append(item)
def verify_source(pair):
 sid,source=pair
 assert sha(source['npz'])==source['npz_sha256'] and sha(source['words_json'])==source['words_sha256']
 with np.load(source['npz']) as z:pose=z['pose'];mimi=z['mimi']
 assert pose.shape==mimi.shape==(source['n_frames'],32)
 assert pose.dtype==mimi.dtype==np.int16 and np.all((pose>=0)&(pose<2048)) and np.all((mimi>=0)&(mimi<2048))
 assert hashlib.sha256(pose.tobytes()).hexdigest()==source['pose_array_sha256']
 assert hashlib.sha256(mimi.tobytes()).hexdigest()==source['mimi_array_sha256']
 words=json.loads(Path(source['words_json']).read_text());assert isinstance(words,list)
 for w in words:assert all(k in w for k in ['start','end','word']) and math.isfinite(w['start']) and math.isfinite(w['end'])
 for item in by_source[sid]:
  start,end=item['crop_start_frame'],item['crop_end_frame'];assert end-start==750 and 0<=start<end<=source['training_n_frames']
  assert item['split']==source['split'] and item['target_masking'] is False
  assert item['selection_evidence']=='exact_all32_constant_pose_codes'
  assert item['band_intervals_frames']
  for a,b in item['band_intervals_frames']:
   assert b-a>=7 and 0<=a<b<=source['training_n_frames']
   assert np.all(pose[a:b]==pose[a])
   assert a==0 or np.any(pose[a-1]!=pose[a])
   assert b==source['training_n_frames'] or np.any(pose[b]!=pose[a])
   assert min(end,b)-max(start,a)>=7
  assert all(end-b==q for (a,b),q in zip(item['anchor_band_intervals_frames'],item['anchor_postexit_context_frames']))
 return {'source_id':sid,'verified_crops':len(by_source[sid]),'source_npz_sha256':source['npz_sha256']}
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:checks=list(pool.map(verify_source,manifest['sources'].items()))
train=[i for i in items if i['split']=='train'];ev=[i for i in items if i['split']=='eval']
assert {i['source_id'].split('__')[0] for i in train}.isdisjoint({i['source_id'].split('__')[0] for i in ev})
original_rows_with_runs=set()
for t in scan['tracks']:
 if t['split']!='train':continue
 for a,b in t['runs_ge7_frames']:
  for chunk in range(a//7500,(b-1)//7500+1):original_rows_with_runs.add((t['session_id'],t['seg'],t['speaker_id'],chunk))
covered={(i['source_id'],a,b) for i in train for a,b in i['anchor_band_intervals_frames']}
expected={(sid,a,b) for sid,s in manifest['sources'].items() if s['split']=='train' for a,b in s['all_constant_pose_runs_frames']}
assert covered==expected and len(covered)==1629
out={'all1629_train_runs_covered':True,'state':'passed','verified_at':time.time(),'manifest_sha256':sha(ROOT/'manifest.json'),'scan_report_sha256':sha(ROOT/'static-band-scan.json'),'source_count':len(checks),'train_crops':len(train),'eval_crops':len(ev),'total_verified_crops':sum(c['verified_crops'] for c in checks),'train_eval_sessions_disjoint':True,'all_crops_750_frames':True,'all_selected_bands_exact_all32_static_ge7':True,'all_source_media_unchanged':True,'postexit_context_is_descriptive_not_eligibility':True,'no_band_target_masks':True,'original_train_rows_containing_ge7_run':len(original_rows_with_runs),'source_checks':checks}
(ROOT/'manifest-verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='source_checks'}),flush=True)
