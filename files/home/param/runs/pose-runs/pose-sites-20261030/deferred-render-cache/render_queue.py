#!/usr/bin/env python3
"""Durable genweb avatar renderer using local verified identity assets and step2911."""
import argparse,json,os,shutil,subprocess,sys,threading,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from safetensors.numpy import save_file
HERE=Path(__file__).resolve().parent
SITE=Path(os.environ.get('GENWEB','/home/param/websites/pose-playground'))
PEOPLE=Path(os.environ.get('GENWEB_PEOPLE','/home/param/datasets/pose-demo/prompts'))
PY=Path('/home/param/vansh-video-models/.venv/bin/python')
ACTIVE=None
ENGINE=None
CK='/home/param/models/trained-weights/full-gt-nine-ref-endpoint-unguided-20261020/step-002911/student.safetensors'

def read(p,default=None):
 try:return json.loads(Path(p).read_text())
 except (OSError,ValueError):return default

def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.tmp{os.getpid()}');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def execute(cmd,log):
 with Path(log).open('a') as f:
  child=subprocess.Popen([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT)
  while True:
   try:
    code=child.wait(timeout=15)
    if code:raise subprocess.CalledProcessError(code,cmd)
    return
   except subprocess.TimeoutExpired:
    if ACTIVE:
     (ACTIVE['claim']).touch()
     state=read(ACTIVE['state'],{})
     progress=read(ACTIVE['work']/'diffusion/generation/status.json',{})
     state.update(at=time.time(),progress_s=min(state.get('seconds',0),int(progress.get('completed_latent_frames',0))*4/25))
     write(ACTIVE['state'],state)

def heartbeat():
 while True:
  active=ACTIVE
  if active:
   try:
    active['claim'].touch()
    state=read(active['state'],{})
    progress=read(active['work']/'diffusion/generation/status.json',{})
    if state.get('state')=='rendering':
     state.update(at=time.time(),progress_s=min(state.get('seconds',0),int(progress.get('completed_latent_frames',0))*4/25))
     write(active['state'],state)
   except OSError:pass
  time.sleep(5)

def engine():
 global ENGINE
 if ENGINE is None:
  from warm_video import WarmVideo
  ENGINE=WarmVideo()
 return ENGINE

def render(rd):
 global ACTIVE
 rid=rd.name;req=read(rd/'request.json',{});summ=read(rd/'summary.json',{})
 person=summ.get('person') or req.get('person') or (req.get('prefill') or {}).get('person') or 'heldout_02'
 assets=read(PEOPLE/person/'video_assets.json')
 z=np.load(rd/'tokens.npz');pose=np.asarray(z['pose'],np.int16);audio=np.asarray(z['mimi'],np.int16)
 if pose.ndim!=2 or pose.shape[1]!=32 or pose.shape!=audio.shape:raise ValueError('Expected aligned local pose/audio [F,32]')
 n=len(pose); assert n>=2 and pose.min()>=0 and pose.max()<2048
 if not assets:
  meta=read(SITE/'openings'/person/'meta.json',{})
  fields=meta.get('sources',{}).get('pose_tokens',{}).get('frame0',{}).get('fields')
  if not fields:raise ValueError(f'No verified pose geometry for {person}')
  sys.path.insert(0,str(SITE));import stick,torch
  torch.set_num_threads(2)
  gs=[*fields['frame0_root'],fields['frame0_width']]
  stick.stick_mp4(pose,int(summ.get('pose_prefill_frames',summ.get('prefill_frames',0))),gs,str(rd/'b.wav'),str(rd/'stick.mp4'))
  receipt={'state':'done','seconds':n/12.5,'frames':2*n,'engine':'pose-tokenizer-preview','person':person,'at':time.time(),'avatar_available':False}
  write(SITE/'render/out'/f'{rid}.json',receipt)
  print('POSE_PREVIEW_COMPLETE',rid,flush=True)
  return
 # Boundary-only continuation feeds the video decoder enough frames. Original codes/audio remain unchanged.
 padding=2+(n%2);pose=np.concatenate([pose,np.repeat(pose[-1:],padding,axis=0)]);audio=np.concatenate([audio,np.repeat(audio[-1:],padding,axis=0)])
 work=rd/'avatar-render';work.mkdir(exist_ok=True);sample=work/'pose';sample.mkdir(exist_ok=True)
 save_file({'codes':pose.T.copy()},str(sample/'pose_tokens.safetensors'));save_file({'codes':audio.T.copy()},str(sample/'mimi.safetensors'))
 state=SITE/'render/out'/f'{rid}.json';claim=SITE/'render/out'/f'{rid}.claim';claim.touch()
 write(state,{'state':'rendering','seconds':n/12.5,'progress_s':0,'engine':'param-step2911','at':time.time(),'person':person})
 ACTIVE={'claim':claim,'state':state,'work':work}
 started=time.monotonic()
 video_engine=engine()
 video_engine.render_pose(sample,assets['frame0_pose'])
 pose_seconds=time.monotonic()-started
 diffusion=work/'diffusion'
 if diffusion.exists():diffusion.rename(work/f'diffusion-attempt-{time.time_ns()}')
 video_status=video_engine.render_video(SimpleNamespace(sample=sample,sink=Path(assets['sink_latent']),
  reference_bank=Path(assets['reference_bank']),actor_id=assets['actor_id'],clip_key=assets['clip_key'],
  checkpoint=Path(CK),seed=req.get('seed',2026),out=diffusion,video_repo=Path('/home/param/vansh-video-models'),
  video_python=PY,vae=Path('/home/param/models/wan22-vae/Wan2.2_VAE.pth'),mimi=Path('/home/param/models/mimi'),max_latent_frames=None))
 out=SITE/'render/out'/f'{rid}.mp4';tmp=out.with_name(out.stem+'.partial.mp4')
 execute(['ffmpeg','-nostdin','-v','error','-y','-i',diffusion/'generation/video.mp4','-i',rd/'b.wav','-map','0:v:0','-map','1:a:0','-vf',f'trim=end_frame={2*n},setpts=PTS-STARTPTS','-af',f'atrim=end={n/12.5},asetpts=PTS-STARTPTS','-frames:v',str(2*n),'-t',str(n/12.5),'-c:v','libx264','-preset','veryfast','-crf','18','-threads','2','-c:a','aac','-ar','24000','-movflags','+faststart',tmp],work/'mux.log')
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(tmp)]))
 video=next(x for x in probe['streams'] if x['codec_type']=='video');assert int(video['nb_frames'])==2*n
 tmp.replace(out)
 receipt={'state':'done','seconds':n/12.5,'progress_s':n/12.5,'frames':2*n,'engine':'param-step2911-warm-wavefront','at':time.time(),'person':person,'checkpoint':CK,'boundary_padding_code_frames':padding,'postprocessing':'Video-only boundary conditioning repeats final2or3 codeframes; final video cropped to exact original pose timeline and muxed with original generated waveform.',
  'schedule':'full-wavefront','present_frames_per_pass':4,'weights_and_graphs_reused':video_status['weights_and_graphs_reused'],
  'timing':{'total_seconds':time.monotonic()-started,'pose_seconds':pose_seconds,
   'video_seconds':video_status['elapsed_seconds'],'sampling_seconds':video_status['sampling_seconds'],
   'mode_setup_seconds':video_status['mode_setup_seconds'],'decoding_seconds':video_status['decoding_seconds']}}
 write(state,receipt);write(work/'RENDER_COMPLETE.json',receipt)
 claim.unlink(missing_ok=True)
 ACTIVE=None
 print('RENDER_COMPLETE',rid,flush=True)

def main():
 global ACTIVE
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path);a=p.parse_args()
 threading.Thread(target=heartbeat,daemon=True).start()
 if a.run:render(a.run);return
 (SITE/'render/out').mkdir(parents=True,exist_ok=True)
 print('RENDERER_READY',os.environ.get('CUDA_VISIBLE_DEVICES'),flush=True)
 while True:
  for feed in sorted((SITE/'render/jobs').glob('*.feed.jsonl')):
   rid=feed.name.removesuffix('.feed.jsonl');rd=SITE/'runs'/rid;state=SITE/'render/out'/f'{rid}.json'
   if (SITE/'render/out'/f'{rid}.mp4').exists() or (rd/'stick.mp4').exists() or read(state,{}).get('state')=='fail':continue
   if read(rd/'status.json',{}).get('state')!='done' or not (rd/'tokens.npz').is_file():continue
   try:render(rd)
   except Exception as e:
    write(state,{'state':'fail','error':str(e),'traceback':traceback.format_exc(),'at':time.time(),'engine':'param-step2911'})
    print('RENDER_FAILED',rid,traceback.format_exc(),flush=True)
   finally:
    ACTIVE=None
    (SITE/'render/out'/f'{rid}.claim').unlink(missing_ok=True)
  time.sleep(2)
if __name__=='__main__':main()
