import hashlib,json,time
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
repo='InternalCan/ppm-aloe-zoom-rosie-ft-e2'
revision='22b0184ed4b382dbfedfa7a0b7a6f0ed19fb24ae'
parent=Path('/home/param/models/imported-weights')
final=parent/'ppm-aloe-zoom-rosie-ft-e2'
stage=parent/'.ppm-aloe-zoom-rosie-ft-e2-downloading'
started=time.time()
assert not final.exists(),str(final)
snapshot_download(repo,revision=revision,local_dir=stage,max_workers=8)
info=HfApi().model_info(repo,revision=revision,files_metadata=True)
for entry in info.siblings:
 p=stage/entry.rfilename
 assert p.is_file(),str(p)
 assert p.stat().st_size==entry.size,(str(p),p.stat().st_size,entry.size)
meta=json.loads((stage/'export-108/meta.json').read_text())
assert meta['step']==108 and meta['run_tag']=='rosie-full-cotrain-1030-1710' and meta['world']==6
assert (stage/'export-108/COMPLETE').read_text().strip()=='108'
receipt={'repo':repo,'revision':revision,'checkpoint':'export-108','run_tag':meta['run_tag'],'files':len(info.siblings),'bytes':sum(x.size for x in info.siblings),'seconds':time.time()-started,'metadata_sha256':hashlib.sha256((stage/'export-108/meta.json').read_bytes()).hexdigest(),'path':str(final)}
(stage/'HF_IMPORT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
stage.rename(final)
Path('/home/param/outputs/pose-opd-rosie-site-audit/download_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
