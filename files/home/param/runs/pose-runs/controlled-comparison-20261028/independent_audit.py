"""CPU-only independent audit of completed controlled PPM/video artifacts."""
import hashlib,json,subprocess,time,traceback
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
import yaml

torch.set_num_threads(2)
ROOT=Path('/home/param/pose-runs/controlled-comparison-20261028')
RECIPES=('aligned-media','compact-all','xabi-original')
PAD,SOP,EOP=151670,151680,151681
OFF=np.arange(32,dtype=np.int64)*2048
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
arrsha=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
sel=read(ROOT/'selection.json')['clips']
plan=read(ROOT/'execution-plan.json')
tok=AutoTokenizer.from_pretrained('/home/param/models/ppm-811-17850-base-bf16/configs/Qwen3-32B',local_files_only=True)
expected={}
for team in plan:
 assert team['node'] in [f'landsend-{i}' for i in range(1,5)]
 for row,c in enumerate(team['clips']):expected[c,team['recipe']]=dict(team,row=row)
assert len(plan)==16 and len(expected)==24
assert sum(t['batch_size']==2 for t in plan)==8 and sum(t['batch_size']==1 for t in plan)==8
report=dict(status='in-progress',timestamp=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),ppm=[],video=[],triplets=[],errors=[],pending=[])
prefixes={};videorefs={}

def ppm(clip,recipe,d):
 m=read(d/'ppm/meta.json');receipt=read(d/'ppm/PPM_COMPLETE.json')
 with np.load(d/'ppm/inputs.npz') as z:ins={k:z[k].copy() for k in z.files}
 with np.load(d/'ppm/tokens.npz') as z:codes={k:z[k].copy() for k in z.files}
 n,P=m['frames'],m['P'];lane=np.load(clip['lane']);voice=np.load(clip['voice_clip']);ex=expected[clip['id'],recipe]
 assert n==752 and m['seed']==20261028+int(clip['gallery_id'])-1
 assert m['checkpoint']['ck']==f'/home/param/models/transcript-pose/{recipe}/export-200' and m['checkpoint']['step']==200
 assert m['checkpoint']['pose'] and m['checkpoint']['decp_bb'] and m['checkpoint']['decp_bb_alpha']==1
 assert m['recipe']==recipe and m['clip_index']==clip['clip_index'] and m['source_span']==[0,752]
 assert m['force_text'] and m['prefill_frames']==m['f0']==0 and not m['ground_truth_generation_media']
 assert m['batch_size']==ex['batch_size'] and m['batch_row']==ex['row']
 assert m['batch_clip_indices']==[next(c['clip_index'] for c in sel if c['id']==i) for i in ex['clips']]
 assert m['voice_provenance']['source']==clip['voice_clip'] and m['voice_clip_frames']==100 and m['conditioning_P']==109
 assert m['rng_policy']=='per-instance per-modality generators; row-wise [1,vocab] multinomial; no text sampling'
 s=m['sampling']
 assert all(s[k]==1 for k in ['text_temp','cb0_temp','light_temp','pose_temp','pose_light_temp','text_topp','text_rep','audio_rep_pen','pose_rep_cb0','pose_rep_light'])
 assert all(s[k]==0 for k in ['text_topk','cb0_topk','light_topk','pose_topk','pose_light_topk','min_p','freq_pen','pose_cfg_on','adaptive_rep_on'])
 assert receipt['sampling']==s and receipt['seed']==m['seed']
 assert voice.shape==(32,100) and np.array_equal(np.load(d/'ppm/voice_clip_mimi.npy'),voice)
 assert m['voice_codes_sha256']==arrsha(voice.astype(np.int64))==clip['voice_provenance']['codes_sha256']
 ix=np.flatnonzero(ins['depth'][1,:109,0]>=0)
 assert len(ix)==100 and np.array_equal((ins['depth'][1,ix]-OFF).T,voice)
 prefix={k:ins[k][:,:109] for k in ('ids','depth','present')}
 prefix.update(pdepth=ins['pdepth'][:109],pid=ins['pid'][:109])
 prefixes[clip['id'],recipe]=prefix
 assert np.all(ins['depth'][:,P:]==-1) and np.all(ins['pdepth'][P:]==-1)
 assert np.all(ins['ids'][0]==PAD) and np.all(ins['present'][0,P:]==0)
 assert np.array_equal(ins['ids'][1],ins['text_force'])
 assert np.array_equal(ins['text_force'][P:P+n],lane) and np.all(ins['text_force'][P+n:]==PAD)
 assert int(lane[0])<151643 and codes['lane'][0]==lane[0]
 assert np.array_equal(codes['lane'],lane)
 assert m['framewise_lane_sha256']==arrsha(ins['text_force'][P:].astype(np.int64))
 if recipe=='xabi-original':assert P==109 and m['transcript_span']==[109,109]
 else:
  assert ins['ids'][1,109]==SOP and ins['ids'][1,P-1]==EOP
  if recipe=='aligned-media':assert P==863 and np.array_equal(ins['ids'][1,110:P-1],lane)
  else:assert np.array_equal(ins['ids'][1,110:P-1],tok.encode(m['transcript'],add_special_tokens=False,split_special_tokens=True))
 for name,path in [('audio_codes','mimi.safetensors'),('pose_codes','pose_tokens.safetensors')]:
  q=codes[name];assert q.shape==(32,752) and q.min()>=0 and q.max()<2048
  assert np.array_equal(q,load_file(str(d/'ppm'/path),device='cpu')['codes'].numpy())
 wav,sr=sf.read(d/'ppm/audio.wav');v,vs=sf.read(d/'ppm/voice_clip.wav')
 assert sr==vs==24000 and len(wav)==752*1920 and len(v)==192000 and np.isfinite(wav).all() and np.isfinite(v).all()
 return dict(clip=clip['id'],recipe=recipe,passed=True,prefix_positions=P,first_lane_token=int(lane[0]),
             voice_sha256=m['voice_codes_sha256'],lane_sha256=m['framewise_lane_sha256'],seed=m['seed'],batch_size=m['batch_size'],
             node=ex['node'],gpu_pair=ex['devs'],all_32_audio_and_pose_codebooks_valid=True,full_generated_lane_exact=True,
             neutral_sampling=True,no_ground_truth_generation_media=True,voice_and_transcript_prefix_valid=True)

def video(clip,recipe,d):
 rec=read(d/'VIDEO_COMPLETE.json');ex=expected[clip['id'],recipe]
 assert rec['status']=='complete' and rec['frames']==1500 and rec['seconds']==60 and rec['source_generated_frames']==1501
 assert rec['node']==ex['node'] and rec['gpu']==ex['devs'][ex['row']]
 assert sha(d/'video.mp4')==rec['sha256']
 streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(d/'video.mp4')]))['streams']
 vv=next(s for s in streams if s['codec_type']=='video');aa=next(s for s in streams if s['codec_type']=='audio')
 assert int(vv['nb_frames'])==1500 and abs(float(vv['duration'])-60)<.001 and abs(float(aa['duration'])-60)<.05
 raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(d/'video.mp4'),'-map','0:a:0','-ac','1','-ar','24000','-f','f32le','-'])
 decoded_audio=np.frombuffer(raw,np.float32);original_audio,_=sf.read(d/'ppm/audio.wav')
 assert len(decoded_audio)>=1440000 and np.isfinite(decoded_audio).all()
 audio_correlation=float(np.corrcoef(decoded_audio[:1440000],original_audio[:1440000])[0,1]);assert audio_correlation>.97
 cfg=yaml.safe_load((d/'diffusion/config.yaml').read_text());inp=read(d/'diffusion/input.json')
 assert cfg['seed']==2026 and cfg['fps']==25 and cfg['batch_size']==1 and cfg['runtime_audio_guidance_scale']==1
 assert cfg['max_latent_frames']==376 and cfg['stabilising']['checkpoint_path']==cfg['resolving']['checkpoint_path']=='/home/param/models/full-gt-nine-ref-endpoint-unguided-20261020/step-002911/student.safetensors'
 render=read(d/'pose/render.json')
 assert render['frame0']==clip['frame0_pose'] and render['pose_tokenizer']=='InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f0de89408cb054da618e309d4334596252'
 assert render['electron_encoder']=='/home/param/models/electron-vae/encoder-v2-pose.safetensors'
 joints=np.load(d/'pose/pose_joints_xyc.npy');assert joints.shape==(1504,133,3) and np.isfinite(joints).all()
 pose=load_file(str(d/'pose/pose_latents_electron_v3.safetensors'),device='cpu')
 assert pose['latents'].shape==(377,48,30,16) and int(pose['valid_frames'])==1504 and torch.isfinite(pose['latents']).all()
 staged=load_file(str(d/'diffusion/inputs/pose_latents_electron_v3.safetensors'),device='cpu')['latents']
 assert torch.equal(staged,pose['latents'][:376])
 for name in ('mimi.safetensors','pose_tokens.safetensors'):
  assert sha(d/'pose'/name)==sha(d/'ppm'/name)
 assert sha(d/'diffusion/inputs/mimi_local.safetensors')==sha(d/'ppm/mimi.safetensors')
 sink=load_file(clip['sink_latent'],device='cpu')['latents']
 videoin=load_file(str(d/'diffusion/inputs/video_latents.safetensors'),device='cpu')['latents']
 assert videoin.shape==(376,48,30,16) and torch.equal(videoin[:1],sink) and torch.count_nonzero(videoin[1:])==0
 assert torch.isfinite(videoin).all()
 refs=[]
 for prop in range(1,10):
  p=Path(clip['reference_bank'])/clip['actor_id']/f'property_{prop}'/'latents.safetensors'
  refs.append(dict(property=prop,sha256=sha(p)))
 assert [(r['property'],r['sha256']) for r in inp['reference_files']]==[(r['property'],r['sha256']) for r in refs]
 videorefs[clip['id'],recipe]=(refs,cfg['seed'],sha(clip['sink_latent']),cfg['stabilising']['checkpoint_path'])
 status=read(d/'diffusion/generation/status.json')
 assert status['status']=='complete' and status['decoded_frames']==1501 and status['seed']==2026
 assert status['reference_keep']==[True]*9
 videorefs[clip['id'],recipe]+=(status['reference_seed'],status['reference_keep'],inp['video_source'])
 finite=[]
 for f in status['latent_files']:
  path=Path(f['path']);assert sha(path)==f['sha256']
  tensors=load_file(str(path),device='cpu');assert all(torch.isfinite(v).all() for v in tensors.values())
  finite.append(path.name)
 return dict(clip=clip['id'],recipe=recipe,passed=True,frames=1500,seconds=60,sha256=rec['sha256'],
             generated_latent_files_finite=finite,audio_correlation_with_ppm=audio_correlation,video_seed=2026,reference_sha256=refs,sink_sha256=sha(clip['sink_latent']),
             future_video_inputs_all_zero=True,staged_audio_codes_exact=True,staged_pose_latents_exact=True,
             pose_decoder_input_codes_equal_ppm=True,node=rec['node'],gpu=rec['gpu'])

for clip in sel:
 for recipe in RECIPES:
  d=ROOT/'outputs'/clip['gallery_id']/recipe
  for kind,fn,sentinel in [('ppm',ppm,'ppm/PPM_COMPLETE.json'),('video',video,'VIDEO_COMPLETE.json')]:
   if not (d/sentinel).exists():report['pending'].append(f'{clip["id"]}/{recipe}/{kind}');continue
   try:report[kind].append(fn(clip,recipe,d))
   except Exception as exc:report['errors'].append(dict(clip=clip['id'],recipe=recipe,stage=kind,error=repr(exc),traceback=traceback.format_exc()))
 for category,data in [('voice_prefix',prefixes),('video_refs_seed_sink',videorefs)]:
  if all((clip['id'],r) in data for r in RECIPES):
   first=data[clip['id'],RECIPES[0]]
   try:
    if category=='voice_prefix':assert all(all(np.array_equal(first[k],data[clip['id'],r][k]) for k in first) for r in RECIPES[1:])
    else:assert all(first==data[clip['id'],r] for r in RECIPES[1:])
    report['triplets'].append(dict(clip=clip['id'],check=category,passed=True))
   except Exception as exc:report['errors'].append(dict(clip=clip['id'],stage=category,error=repr(exc)))
report['status']='failed' if report['errors'] else ('complete' if not report['pending'] else 'in-progress')
report['counts']={k:len(report[k]) for k in ('ppm','video','triplets','pending','errors')}
report['limits']=['This is a held-out PPM comparison; some shared appearance reference photos overlap the source evaluation window.',
                  'Video frame0/sink and appearance references are intentional conditioning; future timeline video slots are verified zero.',
                  'B1/B2 RNG/logit parity checked separately on two compact examples for18 media frames; not every minute under both batch sizes.']
tmp=ROOT/'independent-audit.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(ROOT/'independent-audit.json')
print(json.dumps(dict(status=report['status'],counts=report['counts'],errors=report['errors']),indent=2),flush=True)
