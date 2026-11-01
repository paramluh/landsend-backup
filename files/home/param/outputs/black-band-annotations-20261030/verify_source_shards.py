from pathlib import Path
import concurrent.futures,collections,datetime,hashlib,json,time
import pyarrow.parquet as pq
R=Path('/home/param/datasets/blackband-repro-20261031/source')
P=Path('/home/param/outputs/black-band-annotations-20261030/package')
O=P.parent/'source-shards-verification.json'
COLS=['session_id','seg','T','speaker_00_no_detection_intervals','speaker_01_no_detection_intervals']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inspect(p):
 try:return str(p.relative_to(R)),pq.read_table(p,columns=COLS,use_threads=False).to_pylist(),None
 except Exception as e:return str(p.relative_to(R)),[],repr(e)
start=time.monotonic();indices=pq.read_table(R/'tools/shard_index.parquet').to_pylist();lookup={(x['shard'],x['row_index']):x for x in indices}; paths=sorted((R/'data').glob('*.parquet'));issues=[];total=0;all_rows=[]
if len(lookup)!=len(indices):issues.append('duplicate index row locator')
seen=set()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for shard,rows,error in pool.map(inspect,paths):
  if error:issues.append({'shard':shard,'read_error':error});continue
  total+=len(rows)
  for row_index,row in enumerate(rows):
   loc=(shard,row_index);seen.add(loc);item=lookup.get(loc)
   if item is None:issues.append({'extra_shard_row':list(loc)});continue
   for c in COLS:
    if row[c]!=item[c]:issues.append({'locator':list(loc),'column':c,'index':item[c],'shard':row[c]})
   all_rows.append(row)
missing=set(lookup)-seen
if missing:issues.append({'missing_shard_rows':sorted(missing)})
summary={'conversations':len(all_rows),'conversations_with_any_interval':0,'conversations_with_band_ge_1s_raw_float':0,'conversations_with_band_ge_25_ticks':0,'qualifying_speaker_intervals':0,'qualifying_speaker_pose_ticks':0,'speaker_missing_annotation':collections.Counter(),'all_intervals':0}
for r in all_rows:
 any_iv=False; raw_banded=False;tick_banded=False
 for sp in ['00','01']:
  vals=r[f'speaker_{sp}_no_detection_intervals'];summary['speaker_missing_annotation'][sp]+=vals is None
  for a,b in vals or []:
   summary['all_intervals']+=1;any_iv=True;raw_banded|=b-a>=1.0
   x,y=round(a*25),round(b*25)
   if y-x>=25:
    tick_banded=True;summary['qualifying_speaker_intervals']+=1;summary['qualifying_speaker_pose_ticks']+=y-x
 summary['conversations_with_any_interval']+=any_iv;summary['conversations_with_band_ge_1s_raw_float']+=raw_banded;summary['conversations_with_band_ge_25_ticks']+=tick_banded
summary['qualifying_speaker_seconds']=summary['qualifying_speaker_pose_ticks']/25
snapshots={}
for src,dst in [('tools/shard_index.parquet','source_shard_index.parquet'),('index/conversations.parquet','source_conversations.parquet')]:
 a,b=digest(R/src),digest(P/dst);snapshots[src]={'source_sha256':a,'package_sha256':b,'equal':a==b}
 if a!=b:issues.append({'source_snapshot_mismatch':src})
package=pq.read_table(P/'conversations.parquet').to_pylist(); pkeys={(r['session_id'],r['seg']):r for r in package}
if len(pkeys)!=len(package):issues.append('duplicate package conversation identity')
for r in indices:
 key=(r['session_id'],r['seg']);p=pkeys.get(key)
 if not p:issues.append({'package_missing_conversation':key});continue
 banded=any(round(b*25)-round(a*25)>=25 for sp in ['00','01'] for a,b in (r[f'speaker_{sp}_no_detection_intervals'] or []))
 if p['banded']!=banded:issues.append({'package_banded_mismatch':key})
 for col in r:
  if p[col]!=r[col]:issues.append({'package_source_column_mismatch':key,'column':col})
manifest=json.loads((P/'MANIFEST.json').read_text())
out={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ok':not issues,'elapsed_seconds':round(time.monotonic()-start,3),'read_scope':COLS,'max_workers':4,'source_shards_checked':len(paths),'source_rows_checked':total,'index_rows':len(indices),'unique_conversation_ids':len({(x['session_id'],x['seg']) for x in indices}),'source_row_locators_exact':not missing and total==len(indices),'annotation_summary':summary,'source_snapshots':snapshots,'package_conversation_rows':len(package),'package_banded_rows':pq.ParquetFile(P/'banded_conversations.parquet').metadata.num_rows,'package_manifest_source_hashes_match':manifest['source_index_sha256']==snapshots['tools/shard_index.parquet']['source_sha256'] and manifest['source_conversations_sha256']==snapshots['index/conversations.parquet']['source_sha256'],'issues':issues}
O.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if issues:raise SystemExit(1)
