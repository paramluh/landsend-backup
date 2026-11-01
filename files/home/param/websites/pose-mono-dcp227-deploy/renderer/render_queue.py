#!/usr/bin/env python3
"""Durable genweb avatar renderer using local verified identity assets and step2911."""
import argparse,base64,json,os,shutil,subprocess,sys,time,traceback,wave
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file
HERE=Path(__file__).resolve().parent
SITE=Path(os.environ.get('GENWEB','/home/param/websites/pose-playground'))
PEOPLE=Path(os.environ.get('GENWEB_PEOPLE','/home/param/datasets/pose-demo/prompts'))
PY=Path(os.environ.get('VIDEO_PYTHON', str(HERE / '.venv/bin/python')))
ACTIVE=None
CK=os.environ.get('VIDEO_CHECKPOINT', '/home/param/pose-mono-dcp227/models/video-step2911/student.safetensors')

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

def ground_truth_input(feed):
 """Adapt the existing worker's recorded-code feed without adding a history run."""
 messages=[json.loads(line) for line in feed.read_text().splitlines() if line.strip()]
 rid=feed.name.removesuffix('.feed.jsonl')
 if not rid.startswith('gt-') or len(messages)<4 or messages[0].get('type')!='start' or messages[-1].get('type')!='end':
  raise ValueError('Expected a complete ground-truth start/frame/end feed')
 start=messages[0];person=start.get('person');frames=messages[1:-1]
 if not person or Path(person).name!=person or not (PEOPLE/person/'video_assets.json').is_file():
  raise ValueError('Ground-truth feed needs a known persona with matching video assets')
 if float(start.get('fps',0))!=12.5 or any(m.get('session')!=rid for m in messages):
  raise ValueError('Ground-truth feed session or frame rate mismatch')
 if any(m.get('type')!='f' or m.get('f')!=i for i,m in enumerate(frames)):
  raise ValueError('Ground-truth feed frames must be contiguous')
 pose=np.asarray([m['pose'] for m in frames],np.int64)
 mimi=np.asarray([m['mimi'] for m in frames],np.int64)
 if pose.shape!=(len(frames),32) or mimi.shape!=pose.shape or min(pose.min(),mimi.min())<0 or max(pose.max(),mimi.max())>=2048:
  raise ValueError('Ground-truth feed needs aligned local32-codebook IDs')
 pcm=[base64.b64decode(m['pcm_b64'],validate=True) for m in frames]
 if any(len(chunk)!=1920*2 for chunk in pcm):
  raise ValueError('Ground-truth feed PCM must contain1920 int16 samples per frame')
 rd=SITE/'render/inputs'/rid;rd.mkdir(parents=True,exist_ok=True)
 np.savez(rd/'tokens.npz',pose=pose.astype(np.int16),mimi=mimi.astype(np.int16))
 with wave.open(str(rd/'b.wav'),'wb') as out:
  out.setnchannels(1);out.setsampwidth(2);out.setframerate(24000);out.writeframes(b''.join(pcm))
 write(rd/'request.json',{'person':person,'seed':2026,'source_kind':'ground_truth','source_feed':str(feed),'hidden':True})
 write(rd/'summary.json',{'person':person,'frames':len(frames),'prefill_frames':0,'source_kind':'ground_truth'})
 return rd

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
 execute([PY,HERE/'bridge/transcript_pose/render.py',sample,'--frame0',assets['frame0_pose']],work/'pose.log')
 diffusion=work/'diffusion'
 if diffusion.exists():diffusion.rename(work/f'diffusion-attempt-{time.time_ns()}')
 execute([PY,HERE/'bridge/transcript_pose/render_video.py',sample,'--sink',assets['sink_latent'],'--reference-bank',assets['reference_bank'],'--actor-id',assets['actor_id'],'--clip-key',assets['clip_key'],'--checkpoint',CK,'--seed',str(req.get('seed',2026)),'--out',diffusion],work/'video.log')
 out=SITE/'render/out'/f'{rid}.mp4';tmp=out.with_name(out.stem+'.partial.mp4')
 execute(['ffmpeg','-nostdin','-v','error','-y','-i',diffusion/'generation/video.mp4','-i',rd/'b.wav','-map','0:v:0','-map','1:a:0','-vf',f'trim=end_frame={2*n},setpts=PTS-STARTPTS','-af',f'atrim=end={n/12.5},asetpts=PTS-STARTPTS','-frames:v',str(2*n),'-t',str(n/12.5),'-c:v','libx264','-preset','veryfast','-crf','18','-threads','2','-c:a','aac','-ar','24000','-movflags','+faststart',tmp],work/'mux.log')
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(tmp)]))
 video=next(x for x in probe['streams'] if x['codec_type']=='video');assert int(video['nb_frames'])==2*n
 tmp.replace(out)
 receipt={'state':'done','seconds':n/12.5,'progress_s':n/12.5,'frames':2*n,'engine':'param-step2911','at':time.time(),'person':person,'checkpoint':CK,'source_kind':req.get('source_kind','generated'),'boundary_padding_code_frames':padding,'postprocessing':'Video-only boundary conditioning repeats final2or3 codeframes; final video cropped to exact original pose timeline and muxed with original generated waveform.'}
 write(state,receipt);write(work/'RENDER_COMPLETE.json',receipt)
 claim.unlink(missing_ok=True)
 ACTIVE=None
 print('RENDER_COMPLETE',rid,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path);a=p.parse_args()
 if a.run:render(a.run);return
 (SITE/'render/out').mkdir(parents=True,exist_ok=True)
 print('RENDERER_READY',os.environ.get('CUDA_VISIBLE_DEVICES'),flush=True)
 while True:
  for feed in sorted((SITE/'render/jobs').glob('*.feed.jsonl'),key=lambda p:(p.name.startswith('gt-'),p.name)):
   rid=feed.name.removesuffix('.feed.jsonl');rd=SITE/'runs'/rid;state=SITE/'render/out'/f'{rid}.json'
   if (SITE/'render/out'/f'{rid}.mp4').exists() or (rd/'stick.mp4').exists() or read(state,{}).get('state')=='fail':continue
   if not rid.startswith('gt-') and (read(rd/'status.json',{}).get('state')!='done' or not (rd/'tokens.npz').is_file()):continue
   try:
    if rid.startswith('gt-'):rd=ground_truth_input(feed)
    render(rd)
   except Exception as e:
    write(state,{'state':'fail','error':str(e),'traceback':traceback.format_exc(),'at':time.time(),'engine':'param-step2911'})
    print('RENDER_FAILED',rid,traceback.format_exc(),flush=True)
  time.sleep(2)
if __name__=='__main__':main()
