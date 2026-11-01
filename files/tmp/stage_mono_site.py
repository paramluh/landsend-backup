from pathlib import Path
import shutil,json,hashlib
root=Path('/home/param/websites/pose-mono-dcp227-deploy')
root.mkdir(exist_ok=True)
site=root/'site';site.mkdir(exist_ok=True)
src=Path('/home/param/websites/pose-playground')
for p in src.iterdir():
 if p.is_file() and not p.is_symlink() and p.suffix in {'.py','.html','.css','.js','.md'}:
  shutil.copy2(p,site/p.name)
for name in ['openings','openings-raw','words']:
 shutil.copytree(src/name,site/name,dirs_exist_ok=True)
shutil.copytree('/home/param/websites/pose-monoplex-engine',root/'engine',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copytree('/home/param/datasets/pose-demo/prompts',root/'prompts',dirs_exist_ok=True)
for p in (root/'prompts').glob('*/video_assets.json'):
 d=json.loads(p.read_text())
 for k in ('sink_latent','reference_bank','frame0_pose'):
  d[k]=d[k].replace('/home/param/datasets/pose-demo/prompts','/home/param/pose-mono-dcp227/prompts')
 p.write_text(json.dumps(d,indent=2)+'\n')
(root/'live_support').mkdir(exist_ok=True)
shutil.copy2('/home/param/websites/pose-duplex-engine/feed.py',root/'live_support/feed.py')
(site/'FEED_URL.txt').write_text('off\n')
for name in ['runs','uploads','render/jobs','render/out','state','logs','weights_live','voiceclips','people_extra','user_presets','weights']:
 (site/name).mkdir(parents=True,exist_ok=True)
manifest={'site_source':str(src),'engine_source':'/home/param/websites/pose-monoplex-engine','checkpoint_requested':'/home/param/models/trained-weights/pose-duplex-first-band-20261030/checkpoints/pose-duplex-first-band-20261030/dcp-227','index_sha256':hashlib.sha256((site/'index.html').read_bytes()).hexdigest(),'index_byte_identical':(site/'index.html').read_bytes()==(src/'index.html').read_bytes()}
(root/'SOURCE_PROVENANCE.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
