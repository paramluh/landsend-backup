#!/usr/bin/env python3
"""Package existing real held-out clips and published raw opening assets for genweb."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file,save_file
ROOT=Path('/home/param/websites/pose-playground')
DEST=Path('/home/param/datasets/pose-demo/prompts')
SOURCE=Path('/home/param/runs/pose-runs/heldout-comparison-20261028')
RAW=Path('/home/param/models/imported-weights/ppm-aloe-poseduplex-raw-final/code/voice_clips/openings-raw')

def write(p,value):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 receipt=[]
 for entry in json.loads((SOURCE/'selection.json').read_text())['clips']:
  ident='heldout_'+entry['gallery_id'];src=SOURCE/entry['id']; dst=DEST/ident;gt=dst/'gt';gt.mkdir(parents=True,exist_ok=True)
  codes=np.load(src/'reference-codes.npz');geo=json.loads((src/'geometry.json').read_text()); gs=np.array(geo['global_state0'],np.float32)
  assert codes['audio'].shape==codes['pose'].shape and codes['pose'].shape[0]==32
  frame0=load_file(str(src/'frame0_pose.safetensors'))['frame0_pose']
  save_file({'codes':codes['audio'].copy()},str(gt/'mimi_local.safetensors'))
  save_file({'codes':codes['pose'].copy(),'global_state':gs},str(gt/'pose_tokens.safetensors'))
  save_file({'frame0_pose':frame0,'global_state':gs,'prefill_audio':codes['audio'][:,:25].copy(),'prefill_pose':codes['pose'][:,:25].copy()},str(dst/'llm_inputs.safetensors'))
  opening=ROOT/'openings'/ident;opening.mkdir(parents=True,exist_ok=True)
  for name in ('mimi_local.safetensors','pose_tokens.safetensors'):shutil.copyfile(gt/name,opening/name)
  shutil.copyfile(src/'words.json',opening/'words.json');(ROOT/'words'/ident).mkdir(parents=True,exist_ok=True);shutil.copyfile(src/'words.json',ROOT/'words'/ident/'words.json')
  source_meta={'label':f"Held-out clip {entry['gallery_id']}",'person_id':ident,'kind':'raw-xabi-heldout','split':'eval','recording_id':entry['recording_id'],'source_start_seconds':entry['source_start_seconds'],'T':codes['pose'].shape[1], 'frame_hz':12.5, 'sources':{'pose_tokens':{'frame0':{'fields':geo},'path':str(src/'reference-codes.npz'),'sha256':sha(src/'reference-codes.npz')},'words':{'path':str(src/'words.json'),'sha256':sha(src/'words.json')}},'avatar_available':True}
  write(opening/'meta.json',source_meta);write(dst/'meta.json',source_meta)
  shutil.copyfile(src/'sink_latent.safetensors',dst/'sink_latent.safetensors')
  shutil.copytree(src/'reference-bank',dst/'reference-bank',dirs_exist_ok=True)
  render={'actor_id':entry['actor_id'],'clip_key':entry['clip_key'],'sink_latent':str(dst/'sink_latent.safetensors'),'reference_bank':str(dst/'reference-bank'),'frame0_pose':str(dst/'llm_inputs.safetensors'),'reference_provenance':json.loads((src/'reference-provenance.json').read_text())}
  write(dst/'video_assets.json',render)
  receipt.append({'id':ident,'source':str(src),'frames':int(codes['pose'].shape[1]),'avatar':True})
 for src in sorted(RAW.iterdir()):
  if not src.is_dir():continue
  ident=src.name; dst=ROOT/'people_extra'/ident;gt=dst/'gt';gt.mkdir(parents=True,exist_ok=True)
  meta=json.loads((src/'meta.json').read_text());fl=meta['sources']['pose_tokens']['frame0']['fields'];gs=np.array([*fl['frame0_root'],fl['frame0_width']],np.float32)
  pc=load_file(str(src/'pose_tokens.safetensors'));save_file({'codes':pc['codes'],'global_state':gs},str(gt/'pose_tokens.safetensors'))
  shutil.copyfile(src/'mimi_local.safetensors',gt/'mimi_local.safetensors')
  meta['label']=f"Raw opening {ident} (pose preview)";meta['avatar_available']=False;write(dst/'meta.json',meta);(dst/'NO_AVATAR').touch()
  for rootname in ('openings','openings-raw'):
   shutil.copytree(src,ROOT/rootname/ident,dirs_exist_ok=True)
  (ROOT/'words'/ident).mkdir(parents=True,exist_ok=True);shutil.copyfile(src/'words.json',ROOT/'words'/ident/'words.json')
  receipt.append({'id':ident,'source':str(src),'frames':int(pc['codes'].shape[1]),'avatar':False})
 write(ROOT/'conditioning-manifest.json',{'default_person':'heldout_02','people':receipt,'note':'All source data is genuine; heldout prompts preserve original frame-zero geometry and appearance references. No studio IDs were fabricated.'})
 print(json.dumps({'prompt_count':len(receipt),'avatar_prompts':8,'default':'heldout_02'}))
if __name__=='__main__':main()
