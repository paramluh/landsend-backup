import json,hashlib,subprocess,datetime
from pathlib import Path
import numpy as np
S=Path('/home/param/runs/pose-runs/pose-jump-audit-suite-20261031');jobs=json.loads((S/'v7/diagnostic-render-jobs.json').read_text());ass=json.loads((S/'hand-focused-review/visual-assessment.json').read_text())['all_assessments'];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();checks=[]
for j in jobs:
 out=Path(j['output']);done=json.loads((out/'complete.json').read_text());assert done['state']=='done' and len(done['clips'])==1
 a=next(x for x in ass if x['run_id']==j['run_id'] and x['frame']==j['frame']);run=Path(a['root'])/'runs'/j['run_id'];src=json.loads((run/'pose_overlay.json').read_text());request=json.loads((run/'request.json').read_text());receipt=done['clips'][0];p=out/receipt['name'];actual=json.loads((p/'complete.json').read_text());assert actual==receipt
 assert actual['source_tokens_sha256']==sha(run/'tokens.npz');assert a['source_pose_sha256']==sha(run/'pose_overlay.json')
 start,stop=actual['start_frame'],actual['stop_frame'];assert(start,stop)==(j['frame']-50,j['frame']+50)
 assert actual['pose_window_sha256']==hashlib.sha256(np.asarray(src['frames'][start:stop],dtype=np.float64).tobytes()).hexdigest();assert actual['video_seed']==request['seed'];assert [x['frame'] for x in actual['events']]==[j['frame']]
 assert actual['renderer_sha256']==sha('/home/param/pose-repo/tools/pose_jump_audit/render_windows.py')
 assert actual['inference_entrypoint_sha256']==sha('/home/param/vansh-video-models/inference/distilled.py')
 streams={};media={}
 for n,h in [('video.mp4',480),('overlay.mp4',528)]:
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(p/n)]));v=next(x for x in probe['streams'] if x['codec_type']=='video');audio=next(x for x in probe['streams'] if x['codec_type']=='audio');assert int(v['nb_frames'])==100 and v['r_frame_rate']=='25/1' and int(v['width'])==256 and int(v['height'])==h and abs(float(v['duration'])-4)<.001;assert audio['codec_name']=='aac' and float(audio['duration'])>3.9;streams[n]={'video':{k:v[k] for k in ['codec_name','width','height','nb_frames','duration','r_frame_rate']},'audio':{k:audio[k] for k in ['codec_name','sample_rate','channels','duration']}};media[n]=sha(p/n)
  subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(p/n),'-vf','select=between(n\\,46\\,57),tile=6x2','-frames:v','1',str(p/(n.removesuffix('.mp4')+'-boundary-contact.png'))],check=True)
 from PIL import Image
 im=Image.open(p/'overlay.jpg');im.verify();im=Image.open(p/'overlay.jpg');assert im.format=='JPEG' and im.size==(256,528);media['overlay.jpg']=sha(p/'overlay.jpg')
 row={'job':j,'source_run':str(run),'source_pose_sha256':sha(run/'pose_overlay.json'),'source_tokens_sha256':sha(run/'tokens.npz'),'source_request_sha256':sha(run/'request.json'),'window':{'start_frame':start,'stop_frame':stop,'candidate_frame':j['frame'],'candidate_clip_frame':j['frame']-start},'media_directory':str(p),'media_sha256':media,'stream_checks':streams,'complete_receipt_sha256':sha(p/'complete.json'),'pose_window_sha256':actual['pose_window_sha256'],'appearance_note':actual['appearance_note'],'passed':True};checks.append(row);print(json.dumps({'verified':j['model'],'run':j['run_id']}),flush=True)
result={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'clips_checked':4,'videos_checked':8,'checks':checks};(S/'hand-focused-review/render-verification.json').write_text(json.dumps(result,indent=2)+'\n')
