"""Independent CPU audit for the two approved native website requests."""
from pathlib import Path
import json,hashlib,subprocess,sys
import numpy as np
import torch
import soundfile as sf
from safetensors.torch import load_file
import yaml

torch.set_num_threads(2)
j=Path(sys.argv[1]);recipe=sys.argv[2];root=Path('/home/param/pose-runs/controlled-comparison-20261028');out=Path('/home/param/pose-runs/native-inference-audit-20261028')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ash=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
req=read(j/'request.json');assert req['recipe']==recipe and req['text_mode']=='native' and req['clip_id']=='clip-01' and req['duration_seconds']==4 and req['seed']==20261028 and req['video_seed']==2026
assert all(req[m+'_temp']==1 and req[m+'_topk']==0 for m in ('text','audio','pose'))
assert read(j/'validated_request.json')==req
assert read(j/'state.json')['status']=='complete'
meta=read(j/'metadata.json');assert meta['request']==req and meta['frames']==100 and meta['duration_seconds']==4 and meta['media_frames']==52
p=read(j/'ppm/meta.json');P=p['P'];assert p['checkpoint']['ck']==f'/home/param/models/transcript-pose/{recipe}/export-200' and p['checkpoint']['step']==200
assert p['text_mode']=='native' and not p['force_text'] and p['frames']==52 and p['seed']==20261028 and p['batch_size']==1 and p['prefill_frames']==p['f0']==0
assert not p['ground_truth_generation_media'] and p['conditioning_P']==109 and p['voice_clip_frames']==100 and p['framewise_lane_sha256'] is None
assert p['rng_policy'].endswith('native text sampling')
assert all(p['sampling'][pre+'_temp']==1 and p['sampling'][pre+'_topk']==0 for pre in ('text','cb0','light','pose','pose_light'))
assert p['sampling']['min_p']==p['sampling']['freq_pen']==p['sampling']['pose_cfg_on']==p['sampling']['adaptive_rep_on']==0
clip=read(root/'selection.json')['clips'][0];voice=np.load(clip['voice_clip'])
assert np.array_equal(np.load(j/'ppm/voice_clip_mimi.npy'),voice) and ash(voice.astype(np.int64))==p['voice_codes_sha256']==clip['voice_provenance']['codes_sha256']
with np.load(j/'ppm/inputs.npz') as z:inp={k:z[k].copy() for k in z.files}
assert 'text_force' not in inp and np.all(inp['ids'][1,P:]==151670)
assert np.all(inp['depth'][:,P:]==-1) and np.all(inp['pdepth'][P:]==-1)
ref=Path('/home/param/pose-runs/inference-requests-20261028/jobs/489344dc0d8d422db29a2c295407314c/ppm/inputs.npz')
with np.load(ref) as z:
 for k in ('ids','depth','present'):assert np.array_equal(inp[k][:,:109],z[k][:,:109])
 if recipe=='compact-all':
  assert P==132 and p['supplied_transcript']==p['source_transcript'] and p['upfront_transcript']
  for k in ('ids','depth','present'):assert np.array_equal(inp[k][:,:P],z[k][:,:P])
 else:assert recipe=='xabi-original' and P==109 and p['transcript_span']==[109,109] and p['supplied_transcript'] is None and not p['upfront_transcript']
with np.load(j/'ppm/tokens.npz') as z:
 lane=z['lane'].copy();assert lane.shape==(52,) and lane[0]==151670 and np.issubdtype(lane.dtype,np.integer)
 assert np.all((lane>=0)&((lane<151643)|np.isin(lane,[151670,151671,151645])))
 assert ash(lane.astype(np.int64))==p['generated_lane_sha256']
 for key,file in [('audio_codes','mimi.safetensors'),('pose_codes','pose_tokens.safetensors')]:
  v=z[key];assert v.shape==(32,52) and v.min()>=0 and v.max()<2048 and np.array_equal(v,load_file(str(j/'ppm'/file),device='cpu')['codes'].numpy())
words=read(j/'ppm/words.json');assert words['lane']==lane.tolist() and words['supplied_transcript']==p['supplied_transcript'] and words['source_transcript']==p['source_transcript']
audio,sr=sf.read(j/'ppm/audio.wav');assert sr==24000 and len(audio)==99840 and np.isfinite(audio).all()
assert sha(j/'video.mp4')==meta['sha256']
streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(j/'video.mp4')]))['streams'];v=next(s for s in streams if s['codec_type']=='video');a=next(s for s in streams if s['codec_type']=='audio')
assert int(v['nb_frames'])==100 and abs(float(v['duration'])-4)<.001 and abs(float(a['duration'])-4)<.05 and int(a['sample_rate'])==24000
raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(j/'video.mp4'),'-map','0:a:0','-ac','1','-ar','24000','-f','f32le','-']);y=np.frombuffer(raw,np.float32);assert len(y)>=96000 and np.isfinite(y).all()
corr=float(np.corrcoef(y[:96000],audio[:96000])[0,1]);assert corr>.97
cfg=yaml.safe_load((j/'diffusion/config.yaml').read_text());status=read(j/'diffusion/generation/status.json');assert cfg['seed']==status['seed']==2026 and status['status']=='complete' and status['decoded_frames']==101
assert sha(j/'diffusion/inputs/mimi_local.safetensors')==sha(j/'ppm/mimi.safetensors')
pose=load_file(str(j/'ppm/pose_latents_electron_v3.safetensors'),device='cpu')['latents'];staged=load_file(str(j/'diffusion/inputs/pose_latents_electron_v3.safetensors'),device='cpu')['latents'];assert pose.shape==(27,48,30,16) and torch.equal(staged,pose[:26]) and torch.isfinite(pose).all()
vi=load_file(str(j/'diffusion/inputs/video_latents.safetensors'),device='cpu')['latents'];sink=load_file(clip['sink_latent'],device='cpu')['latents'];assert torch.equal(vi[:1],sink) and torch.count_nonzero(vi[1:])==0
for item in status['latent_files']:
 path=Path(item['path']);assert sha(path)==item['sha256'] and all(torch.isfinite(v).all() for v in load_file(str(path),device='cpu').values())
report=dict(status='complete',recipe=recipe,job_id=j.name,text_mode='native',request=req,frames=100,duration_seconds=4,media_frames=52,
 voice_codes_sha256=p['voice_codes_sha256'],prefix_positions=P,conditioning_prefix_exact=True,no_GT_framewise_text=True,no_text_force_array=True,
 no_GT_generation_media=True,source_transcript=p['source_transcript'],supplied_transcript=p['supplied_transcript'],generated_transcript=words['text'],
 lexical_lane_tokens=int(np.sum(lane<151643)),generated_lane_sha256=p['generated_lane_sha256'],neutral_text_audio_pose_sampling=True,
 all_media_codebooks_valid=True,staged_generated_conditioning_exact=True,future_video_inputs_zero=True,generated_latents_finite=True,audio_correlation=corr)
(out/f'{recipe}-interactive-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
