import concurrent.futures,json,pathlib,time
from huggingface_hub import HfApi
ROOT=pathlib.Path('/home/param/outputs/landsend-backup-20261031/checkpoints')
weights=json.loads((ROOT/'landsend-0.json').read_text())['files']
base=pathlib.Path('/home/param/models/imported-weights')
mapping={p.name:('InternalCan/'+p.name,'','main') for p in base.iterdir() if p.is_dir()}
mapping.update({'electron-vae':('vanshjjw/electron-vae','','9f18375922780ad10fbc209af78c467a67866e69'),'transcript-pose':('InternalCan/ppm-aloe-v2pure-main-17850-final','xabi-original/','main'),'ppmvox-v5-18100':('InternalCan/ppmvox-v5-17850','','9de2b20a48a4173a8aca15b3018566e2140ec1cf'),'next-opd-v110':('InternalCan/ppm-opd-repro-slinky','','188691b40d2962892431bd765e6179e3dd471579'),'mimi':('kyutai/mimi','','main'),'wan22-vae':('Wan-AI/Wan2.2-TI2V-5B','','main')})
# Three trained-weights folders are previously downloaded exports, not local training.
for name in ['bimodal-resolver-gt-pose-v3-step2000','bimodal-stabilizer-full-gt-pose-v3-step1200','full-gt-nine-ref-endpoint-unguided-20261020']:
 mapping['../trained-weights/'+name]=('InternalCan/'+name,'','main')
def one(entry):
 name,(repo,strip,revision)=entry;folder=(base/name).resolve();selected=[x for x in weights if x['path'].startswith(str(folder)+'/')]
 if not selected:return None
 api=HfApi();checks=[]
 try:
  info=api.model_info(repo,revision=revision,files_metadata=True)
  remote={x.rfilename:x for x in info.siblings}
  for f in selected:
   relative=str(pathlib.Path(f['path']).relative_to(folder));rp=relative.removeprefix(strip)
   if name=='next-opd-v110':rp='runs/opd-nextrun-b384-v3/rolling/checkpoint-1793330493092033558-7e901371585b/'+relative
   obj=remote.get(rp);result={'local':f['path'],'remote':rp,'bytes':f['bytes'],'matched':obj is not None and obj.size==f['bytes']}
   cache=folder/'.cache/huggingface/download'/str(relative+'.metadata')
   if cache.exists() and obj and obj.lfs:
    lines=cache.read_text().splitlines();sha=obj.lfs.get('sha256') if isinstance(obj.lfs,dict) else obj.lfs.sha256
    if len(lines)>1:result.update(cached_download_sha256=lines[1],remote_sha256=sha,cached_hash_matched=lines[1]==sha)
   checks.append(result)
  out={'folder':str(folder),'repo':repo,'revision':info.sha,'files':checks,'matched':all(x['matched'] and x.get('cached_hash_matched',True) for x in checks),'time':time.time()}
 except Exception as e:out={'folder':str(folder),'repo':repo,'error':str(e),'matched':False,'time':time.time()}
 (ROOT/('existing-'+name.replace('/','_')+'.json')).write_text(json.dumps(out,indent=2))
 return {'folder':name,'repo':repo,'matched':out['matched'],'files':len(checks),'unmatched':[x['remote'] for x in checks if not x['matched']][:6],'error':out.get('error')}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=[x for x in pool.map(one,mapping.items()) if x]
(ROOT/'existing-hf-summary.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
