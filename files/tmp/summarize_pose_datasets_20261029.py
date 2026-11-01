from huggingface_hub import HfApi,get_token
from pathlib import Path
import collections,json,requests,concurrent.futures,datetime
O=Path('/home/param/pose-runs/distillation-inventory-20261029');api=HfApi()
repos=['InternalCan/aloe-v2vox-speech-export253','InternalCan/stage2-sam-sep-v1-mimi','InternalCan/aloe-poseduplex-v5-training-pose','InternalCan/xabi-v2pure-production-cache','InternalCan/aloe-pose-speech-vad-v2']
def info(r):
 i=api.dataset_info(r,files_metadata=True);return dict(id=i.id,url='https://huggingface.co/datasets/'+i.id,revision=i.sha,last_modified=str(i.last_modified),files=len(i.siblings),bytes=sum(x.size or 0 for x in i.siblings))
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:live=list(ex.map(info,repos))
bucket='InternalCan/aloe-conversational-speech-tokens';d=collections.defaultdict(lambda:dict(files=0,bytes=0,done_paths=[],segment_paths=[]))
for x in api.list_bucket_tree(bucket,recursive=True):
 if getattr(x,'size',None) is None:continue
 p=x.path;g=d[p.split('/')[0]];g['files']+=1;g['bytes']+=x.size
 if p.endswith('/done.json'):g['done_paths'].append(p)
 if p.endswith('/segment.json'):g['segment_paths'].append(p)
for prefix,g in d.items():
 g['completed_recordings']=len(g['done_paths']);g['segment_metadata_count']=len(g.pop('segment_paths'));g['url']='https://huggingface.co/buckets/'+bucket+'/tree/main/'+prefix
 p=prefix+'/dataset.json';r=requests.get('https://huggingface.co/buckets/'+bucket+'/resolve/'+p,headers={'Authorization':'Bearer '+get_token()},timeout=30);r.raise_for_status();g['dataset_metadata']=r.json()
x=json.load(open('/home/param/pose-runs/duplex-data-download-20261028/POSE_INDEX.original.json'));dur={}
for sp in ['train','eval']:
 rows=[r for rr in x['rows'].values() for r in rr.values() if r['split']==sp]
 dur[sp]=dict(conversations=len(rows),conversation_hours_first_chunk=sum(r['n'] for r in rows)/45000,conversation_hours_full=sum(r['T'] for r in rows)/45000,valid_pose_track_hours_first_chunk=sum(r['n'] for r in rows for s in r['sides'].values() if s['ok'])/45000,valid_pose_track_hours_full=sum(s['T'] for r in rows for s in r['sides'].values() if s['ok'])/45000)
cat=json.load(open('/home/param/training-registry/source/data/datasets.json'))['datasets']
ids={'aloe-stage2-handover-20261008','aloe-pose-ft-stores','aloe-pose-ft-sam-stores','aloe-pose-speech-vad-v1-clean-rows','aloe-pose-speech-vad-release-v2','aloe-sam-cache-active-v2pure','aloe-v2pure-ft-stores','s2h-m1-samsep-stores'}
historical=[{k:v for k,v in e.items() if k in ['id','name','contents','modalities','encoding','size','locations','held_out']} for e in cat if e['id'] in ids]
report={
 'audited_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'scope':'Authenticated HF metadata and current seven landsend nodes. No datasets downloaded or altered. Local payload check = all source/pose/cache manifest sizes + representative NumPy headers/ranges, with previous full checksummed verification records retained.',
 'primary_dataset':{'name':'Exact production Aloe pose-duplex v5 training set','training_run':'30074','checkpoint':'export-498','current_nodes':['landsend-1','landsend-2','landsend-3','landsend-4'],'absent_nodes':['landsend-0','landsend-5','landsend-12'],'root':'/home/param/datasets/aloe-duplex','total_apparent_bytes_per_node_approx':295540000000,'download_bytes':276183071521,'counts':x['counts'],'not_ok_reasons':x['not_ok_reasons'],'duration_from_training_index':dur,'sampling':'Original v5 recipe uses first chunk <=7,500 frames/600 seconds, both assistant directions. Full pose source tracks are stored. Pose loss/input absent for 2,357 invalid/missing sides.','paths':{'speech_and_text_stores':'/home/param/datasets/aloe-duplex/s2h-m1-samsep/stores/m1n{0,1,2,3}','pose_index':'/home/param/datasets/aloe-duplex/pose/POSE_INDEX.json','pose_data':'/home/param/datasets/aloe-duplex/pose/data','raw_audio':'/home/param/datasets/aloe-duplex/original/flac','audio_index':'/home/param/datasets/aloe-duplex/pose/M1VOX_INDEX_samsep.json'},'components':[{'repo':'InternalCan/aloe-v2vox-speech-export253','revision':'c544cc7eab5fdf596d92d70b7fcfa3bc920e165a','role':'original row stores, timed transcripts and raw audio','download_bytes':254430569947},{'repo':'InternalCan/stage2-sam-sep-v1-mimi','revision':'ecb72b2de2347bfbbc3b71b0236e2986b0c44572','role':'SAM-separated audio codes rebuilt into exact s2h-m1-samsep stores','download_bytes':14433746736},{'repo':'InternalCan/aloe-poseduplex-v5-training-pose','revision':'99cffe4a741ee7cd1c8821c085ca1dce2ea3b17f','role':'exact original pose tracks plus training indexes','download_bytes':7318754838}], 'prepared_speech_stores_rows':11777, 'prepared_speech_stores_hours_all_chunks':1383.06},
 'other_local_dataset':{'name':'Xabi v2pure active-speaker production cache','repo':'InternalCan/xabi-v2pure-production-cache','current_nodes':['landsend-0','landsend-1','landsend-2','landsend-3','landsend-4','landsend-5','landsend-12'],'root':'/home/param/datasets/xabi-v2pure-cache','cache_bytes':12416427207,'package_bytes':12472730733,'train_clips':367496,'eval_clips':3482,'historical_ppm_eligible_train_clips':367495,'historical_ppm_eligible_eval_clips':3481,'shards_train':960,'shards_eval':1,'suitability':'Single-speaker active clips; partner lane is silence. Not ready-to-use paired pose-duplex conversations.'},
 'hf_current_dataset_metadata':live,
 'hf_paired_window_alternatives':{'bucket':bucket,'note':'Only completed recording seals establish delivered availability; large dataset.json counts are planned totals. These prefixes are not staged under current /home/param/datasets.','prefixes':dict(d)},
 'historical_catalogue_candidates':{'catalogue':'/home/param/training-registry/source/data/DATA_CATALOGUE.md','note':'Documented historical source locations and counts, not claims of present landsend payloads. Current /data/home/amartyavarma and /scratch/amartyavarma point into code-only shared-memory trees.','entries':historical},
 'related_hf_sources_note':'aloe-pose-speech-vad-v2 has 762,028 speaker-segment rows/12,675 recordings, 20.156 GB current HF package; requires paired conversation assembly. aloe-sharded-data/aloe-duplex-handheld-v1 has video/speech/text but no pose codes per catalogue, so is not a ready pose-duplex source.',
 'evidence':['/home/param/pose-runs/distillation-inventory-20261029/dataset-node-audit.json','/home/param/pose-runs/distillation-inventory-20261029/supplemental-datasets-and-model-assets.json','/home/param/pose-runs/duplex-data-download-20261028/CLUSTER_COMPLETE.json','/home/param/pose-runs/duplex-data-download-20261028/POSE_INDEX.original.json','/home/param/datasets/xabi-v2pure-cache/README.md']}
(O/'datasets.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'report':str(O/'datasets.json'),'bytes':(O/'datasets.json').stat().st_size,'paired_prefixes':{k:{z:v[z] for z in ['files','bytes','completed_recordings','segment_metadata_count']} for k,v in d.items()}}))
