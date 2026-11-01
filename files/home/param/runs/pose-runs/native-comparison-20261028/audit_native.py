"""Independent CPU-only audit of saved native PPM/video results."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file
sys.path.insert(0,'/home/param/pose-repo/infer/transcript_pose')
import request_worker as W
import request_schema as S

def audit(path, req, clip):
 path=Path(path);req=S.normalize_request(req)
 assert req['text_mode']=='native'
 n=S.media_frames(req['duration_seconds']);seconds=req['duration_seconds']
 ppm=W.validate_ppm(path,req,clip)
 words=W.read(path/'ppm/words.json')
 with np.load(path/'ppm/tokens.npz') as tokens:assert np.array_equal(words['lane'],tokens['lane'])
 source=W.read(path/'ppm/input.json')
 assert source['cache_key']==clip['cache_key'] and source['source_span']==[clip['start_frame'],clip['start_frame']+n]
 assets=W.read(W.SOURCE/clip['id']/'video_assets.json')
 receipt=W.read(path/'diffusion/generation/status.json')
 schedule=W.validate_video_receipt(receipt,n,req['video_seed'])
 for item in receipt['latent_files']:
  target=Path(item['path']);assert W.sha(target)==item['sha256'];W.validate_latents(target)
 video_input=W.read(path/'diffusion/input.json')
 assert video_input['actor_id']==assets['actor_id'] and video_input['reference_clip_key']==assets['clip_key']
 assert len(video_input['reference_files'])==9 and video_input['input_media_frames']==n
 assert W.sha(Path(video_input['sink']))==W.sha(Path(assets['sink_latent']))
 for row in video_input['reference_files']:
  ref=Path(assets['reference_bank'])/assets['actor_id']/f"property_{row['property']}"/'latents.safetensors'
  assert W.sha(ref)==row['sha256']
 assert np.array_equal(load_file(str(path/'diffusion/inputs/mimi_local.safetensors'))['codes'].numpy(),load_file(str(path/'ppm/mimi.safetensors'))['codes'].numpy())
 pose_dir=Path(video_input['source_sample']);assert pose_dir.resolve().is_relative_to(path.resolve())
 render_meta=W.read(pose_dir/'render.json');assert W.sha(Path(render_meta['frame0']))==W.sha(Path(assets['frame0_pose']))
 assert render_meta['pose_tokenizer']=='InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f0de89408cb054da618e309d4334596252'
 pose=load_file(str(pose_dir/'pose_latents_electron_v3.safetensors'))['latents']
 staged_pose=load_file(str(path/'diffusion/inputs/pose_latents_electron_v3.safetensors'))['latents']
 assert torch.isfinite(pose).all() and torch.equal(staged_pose,pose[:n//2])
 appearance=load_file(str(path/'diffusion/inputs/video_latents.safetensors'))['latents']
 sink=load_file(assets['sink_latent'])['latents']
 assert torch.equal(appearance[:1],sink) and torch.count_nonzero(appearance[1:])==0
 final=path/'video.mp4'
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(final)]))
 v=next(s for s in probe['streams'] if s['codec_type']=='video');a=next(s for s in probe['streams'] if s['codec_type']=='audio')
 assert int(v['nb_frames'])==seconds*25 and v['r_frame_rate']=='25/1' and abs(float(v['duration'])-seconds)<.001
 assert int(a['sample_rate'])==24000 and abs(float(a['duration'])-seconds)<.05
 subprocess.run(['ffmpeg','-v','error','-xerror','-threads','2','-i',str(final),'-f','null','-'],check=True,capture_output=True)
 decoded=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(final),'-map','0:a:0','-ac','1','-ar','24000','-f','f32le','pipe:1']),dtype='<f4')[:seconds*24000].astype(np.float64)
 original,sr=sf.read(path/'ppm/audio.wav');original=original[:seconds*24000]
 assert sr==24000 and len(decoded)==len(original)==seconds*24000 and np.isfinite(decoded).all()
 if np.std(original)<1e-8:
  correlation=snr=None;assert np.max(np.abs(decoded-original))<1e-4
 else:
  correlation=float(np.corrcoef(original,decoded)[0,1]);snr=float(10*np.log10(np.mean(original**2)/max(np.mean((decoded-original)**2),1e-30)))
  assert correlation>.98 and snr>15, f'Video audio mismatch: corr={correlation}, SNR={snr}'
 for receipt_name in ('VIDEO_COMPLETE.json','metadata.json'):
  if (path/receipt_name).is_file():
   saved=W.read(path/receipt_name)
   if 'sha256' in saved:assert saved['sha256']==W.sha(final)
 return {'status':'pass','completed_at':W.stamp(),'path':str(path),'recipe':req['recipe'],'clip_id':clip['id'],
         'request':req,'source_cache_key':clip['cache_key'],'conditioning':ppm,'video_schedule':schedule,
         'voice_same_original_speaker':True,'references_same_source_actor':True,'ground_truth_future_video':False,
         'mimi_codes_exactly_generated_ppm':True,'pose_latents_exactly_generated_ppm':True,
         'video_frames':seconds*25,'seconds':seconds,'full_cpu_decode_passed':True,'sha256':W.sha(final),
         'audio_vs_ppm':{'same_time_correlation':correlation,'snr_db':snr,'comparison_samples':len(decoded),'lossy_aac_expected':True}}

if __name__=='__main__':
 path=Path(sys.argv[1]);req=W.read(path/'request.json');clip,_=W.source_clip(req['clip_id'])
 result=audit(path,req,clip);W.write(path/'independent-audit.json',result)
 print(json.dumps({k:v for k,v in result.items() if k!='conditioning'},indent=2))
