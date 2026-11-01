import concurrent.futures,hashlib,json,os,pathlib,shutil,socket,sys,time
from huggingface_hub import snapshot_download
base=pathlib.Path('/home/param/pose-runs/distillation-inventory-20261029')
host=socket.gethostname().split('.')[0]
manifest=json.loads((base/'supplemental-manifest.json').read_text())
report={'host':host,'started':time.time(),'models':[],'sidecars':[]}
for m in manifest['models']:
 root=pathlib.Path(m['local_root']);root.mkdir(parents=True,exist_ok=True)
 print(host,m['key'],'download',m['bytes'],flush=True)
 snapshot_download(repo_id=m['repo'],revision=m['revision'],local_dir=str(root),allow_patterns=[f['path'] for f in m['files']],max_workers=4)
 def check(f):
  p=root/f['path'];size=p.stat().st_size
  if size!=f['size']:raise ValueError(f"Wrong size {p}: {size} != {f['size']}")
  if 'sha256' in f:h=hashlib.sha256();expected=f['sha256']
  else:h=hashlib.sha1(f'blob {size}\0'.encode());expected=f['git_blob_sha1']
  with p.open('rb') as stream:
   for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
  if h.hexdigest()!=expected:raise ValueError(f'Wrong hash {p}')
  return f['path']
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:verified=list(pool.map(check,m['files']))
 report['models'].append({'key':m['key'],'repo':m['repo'],'revision':m['revision'],'root':str(root),'files':len(verified),'bytes':m['bytes'],'verified':'all file sizes and hashes match HF'})
 print(host,m['key'],'verified',len(verified),flush=True)
# Keep exact gains with the matching DCPs. These filenames are outside the original 67-file manifests.
audio_source=pathlib.Path('/home/param/models/aloe-ft-vox17127-nopersona/weights/vgain_theta_fp32.json')
audio_target=pathlib.Path('/home/param/models/ppm-aloe-samsep-vox17127-final/export-249-dcp/vgain_theta_fp32.json')
audio_target.parent.mkdir(parents=True,exist_ok=True)
shutil.copyfile(audio_source,audio_target)
assert audio_source.read_bytes()==audio_target.read_bytes()
report['sidecars'].append({'path':str(audio_target),'sha256':hashlib.sha256(audio_target.read_bytes()).hexdigest(),'value':json.loads(audio_target.read_text())['vgain_theta_fp32']})
pose_source=pathlib.Path('/home/param/models/ppm-aloe-poseduplex-v5-final/export-498/meta.json')
pose_meta=json.loads(pose_source.read_text())
pose_target=pathlib.Path('/home/param/models/ppm-aloe-poseduplex-v5-final/export-498-dcp/vgain_theta_fp32.json');pose_target.parent.mkdir(parents=True,exist_ok=True)
pose_gain={'vgain_theta_fp32':pose_meta['vgain_theta_fp32'],'step':pose_meta['step'],'run_tag':pose_meta['run_tag'],'source':'hf://InternalCan/ppm-aloe-poseduplex-v5-final@8cc4375d9c76c0c68ab48a9ff67da5644bb51544/export-498/meta.json','source_git_blob_sha1':next(f['git_blob_sha1'] for m in manifest['models'] if m['key']=='pose_duplex_metadata' for f in m['files']),'note':'Use exact fp32 training gain when loading the bf16 DCP. Derived from the verified original export metadata.'}
pose_target.write_text(json.dumps(pose_gain,indent=2)+'\n')
report['sidecars'].append({'path':str(pose_target),'sha256':hashlib.sha256(pose_target.read_bytes()).hexdigest(),'value':pose_gain['vgain_theta_fp32']})
report['completed']=time.time();report['pass']=True
(base/f'supplemental-status-{host}.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
