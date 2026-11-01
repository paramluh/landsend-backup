"""CPU-only independent artifact audit for the single approved website smoke."""
from pathlib import Path
import hashlib,json,math,subprocess,sys
import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file
import yaml

torch.set_num_threads(2)
ROOT=Path('/home/param/pose-runs/controlled-comparison-20261028')
J=Path(sys.argv[1])
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
arrsha=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
expected=dict(recipe='compact-all',clip_id='clip-01',duration_seconds=4,seed=314159,video_seed=271828,audio_temp=.8,pose_temp=1.2,audio_topk=7,pose_topk=11)
req=read(J/'request.json');assert req==expected,(req,expected)
assert read(J/'validated_request.json')==expected
state=read(J/'state.json');assert state['status']==state['stage']=='complete'
meta=read(J/'metadata.json');assert meta['request']==expected and meta['node']=='landsend-1' and meta['gpus']==[6,7]
assert meta['duration_seconds']==4 and meta['frames']==100 and meta['media_frames']==52 and meta['batch_size']==1
clip=next(c for c in read(ROOT/'selection.json')['clips'] if c['id']=='clip-01')
voice=np.load(clip['voice_clip']);lane=np.load(clip['lane'])[:52]
p=read(J/'ppm/meta.json');P=p['P']
assert p['frames']==52 and p['recipe']=='compact-all' and p['seed']==314159 and p['batch_size']==1
assert p['checkpoint']['ck']=='/home/param/models/transcript-pose/compact-all/export-200' and p['checkpoint']['step']==200
assert p['voice_clip_frames']==100 and p['conditioning_P']==109 and p['force_text'] and p['f0']==p['prefill_frames']==0 and not p['ground_truth_generation_media']
assert np.array_equal(np.load(J/'ppm/voice_clip_mimi.npy'),voice)
assert arrsha(voice.astype(np.int64))==p['voice_codes_sha256']==clip['voice_provenance']['codes_sha256']
for pre,t,k in [('cb0',.8,7),('light',.8,7),('pose',1.2,11),('pose_light',1.2,11)]:
 assert p['sampling'][pre+'_temp']==t and p['sampling'][pre+'_topk']==k
assert p['sampling']['min_p']==p['sampling']['freq_pen']==p['sampling']['pose_cfg_on']==p['sampling']['adaptive_rep_on']==0
assert p['sampling']['text_topp']==p['sampling']['audio_rep_pen']==p['sampling']['pose_rep_cb0']==p['sampling']['pose_rep_light']==1
with np.load(J/'ppm/inputs.npz') as z:inp={k:z[k].copy() for k in z.files}
assert np.array_equal(inp['ids'][1],inp['text_force']) and np.array_equal(inp['text_force'][P:P+52],lane) and np.all(inp['text_force'][P+52:]==151670)
assert inp['text_force'][P]==lane[0] and lane[0]<151643
assert np.all(inp['depth'][:,P:]==-1) and np.all(inp['pdepth'][P:]==-1)
assert inp['ids'][1,109]==151680 and inp['ids'][1,P-1]==151681
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/home/param/models/ppm-811-17850-base-bf16/configs/Qwen3-32B',local_files_only=True)
assert np.array_equal(inp['ids'][1,110:P-1],tok.encode(p['transcript'],add_special_tokens=False,split_special_tokens=True))
with np.load(J/'ppm/tokens.npz') as z:
 assert np.array_equal(z['lane'],lane)
 for key,file in [('audio_codes','mimi.safetensors'),('pose_codes','pose_tokens.safetensors')]:
  a=z[key];assert a.shape==(32,52) and a.min()>=0 and a.max()<2048 and np.array_equal(a,load_file(str(J/'ppm'/file),device='cpu')['codes'].numpy())
audio,sr=sf.read(J/'ppm/audio.wav');assert sr==24000 and len(audio)==99840 and np.isfinite(audio).all()
assert sha(J/'video.mp4')==meta['sha256']
streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(J/'video.mp4')]))['streams']
v=next(s for s in streams if s['codec_type']=='video');a=next(s for s in streams if s['codec_type']=='audio')
assert int(v['nb_frames'])==100 and abs(float(v['duration'])-4)<.001 and abs(float(a['duration'])-4)<.05 and int(a['sample_rate'])==24000
raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(J/'video.mp4'),'-map','0:a:0','-ac','1','-ar','24000','-f','f32le','-'])
y=np.frombuffer(raw,np.float32);assert np.isfinite(y).all() and len(y)>=96000
corr=float(np.corrcoef(y[:96000],audio[:96000])[0,1]);assert corr>.97
config=yaml.safe_load((J/'diffusion/config.yaml').read_text());assert config['seed']==271828
status=read(J/'diffusion/generation/status.json');assert status['status']=='complete' and status['seed']==271828 and status['decoded_frames']==101
assert sha(J/'diffusion/inputs/mimi_local.safetensors')==sha(J/'ppm/mimi.safetensors')
posed=load_file(str(J/'ppm/pose_latents_electron_v3.safetensors'),device='cpu')['latents']
poses=load_file(str(J/'diffusion/inputs/pose_latents_electron_v3.safetensors'),device='cpu')['latents']
assert posed.shape==(27,48,30,16) and poses.shape==(26,48,30,16) and torch.equal(poses,posed[:26]) and torch.isfinite(posed).all()
sink=load_file(clip['sink_latent'],device='cpu')['latents'];video=load_file(str(J/'diffusion/inputs/video_latents.safetensors'),device='cpu')['latents']
assert torch.equal(video[:1],sink) and torch.count_nonzero(video[1:])==0
for item in status['latent_files']:
 path=Path(item['path']);assert sha(path)==item['sha256'] and all(torch.isfinite(x).all() for x in load_file(str(path),device='cpu').values())
report=dict(status='complete',job_id=J.name,request=expected,ppm_media_frames=52,ppm_audio_samples=99840,audio_sample_rate=24000,
            video_frames=100,duration_seconds=4,first_lane_token=int(lane[0]),voice_sha256=p['voice_codes_sha256'],
            all_32_media_codebooks_valid=True,settings_verified_cb0_and_residual=True,forced_lane_exact=True,
            prefix_matches_compact_training=True,no_ground_truth_generation_media=True,future_video_inputs_all_zero=True,
            video_seed_verified=True,staged_generated_audio_and_pose_exact=True,generated_latents_finite=True,
            final_video_sha256=meta['sha256'],final_audio_correlation_with_ppm=corr)
(ROOT/'independent-interactive-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
