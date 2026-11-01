import hashlib,json,time
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
repo='InternalCan/ppm-aloe-zoom-cotrain'
revision='8cdb31a0cd08d4b7ceeb7e1cc9b9ebbdebdc8ba4'
parent=Path('/home/param/models/imported-weights')
final=parent/'ppm-aloe-zoom-cotrain'
stage=parent/'.ppm-aloe-zoom-cotrain-downloading'
started=time.time()
assert not final.exists(),str(final)
patterns=['export-253/*','_backup/MANIFEST.export-253.*.tsv','.gitattributes']
snapshot_download(repo,revision=revision,local_dir=stage,allow_patterns=patterns,max_workers=16)
info=HfApi().model_info(repo,revision=revision,files_metadata=True)
selected=[x for x in info.siblings if x.rfilename.startswith('export-253/') or x.rfilename.startswith('_backup/MANIFEST.export-253.') or x.rfilename=='.gitattributes']
for entry in selected:
 p=stage/entry.rfilename
 assert p.is_file(),str(p)
 assert p.stat().st_size==entry.size,(str(p),p.stat().st_size,entry.size)
meta=json.loads((stage/'export-253/meta.json').read_text())
assert meta['step']==253 and meta['run_tag']=='ppm-aloe-zoom-cotrain-j26467' and meta['world']==128
assert (stage/'export-253/COMPLETE').read_text().strip()=='253'
receipt={'repo':repo,'revision':revision,'checkpoint':'export-253','run_tag':meta['run_tag'],'files':len(selected),'bytes':sum(x.size for x in selected),'seconds':time.time()-started,'metadata_sha256':hashlib.sha256((stage/'export-253/meta.json').read_bytes()).hexdigest(),'path':str(final)}
(stage/'HF_IMPORT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
stage.rename(final)
Path('/home/param/outputs/pose-opd-rosie-site-audit/general_parent_download_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
