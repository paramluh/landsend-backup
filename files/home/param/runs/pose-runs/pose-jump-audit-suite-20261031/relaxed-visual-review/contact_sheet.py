#!/usr/bin/env python3
"""Extract labeled video frames around an exact source-frame boundary (CPU)."""
from pathlib import Path
import argparse,subprocess,json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--video',type=Path,required=True);p.add_argument('--source-start',type=int,required=True);p.add_argument('--frame',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate','-of','json',str(a.video)]))['streams'][0]
w,h,count=meta['width'],meta['height'],int(meta['nb_frames']);assert meta['r_frame_rate']=='25/1'
center=a.frame-a.source_start;indices=[center+d for d in [-12,-8,-4,-3,-2,-1,0,1,2,3,4,8,12,16]];assert min(indices)>=0 and max(indices)<count
start,stop=min(indices),max(indices)+1
raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(a.video),'-vf',f'trim=start_frame={start}:end_frame={stop},setpts=PTS-STARTPTS','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
x=np.frombuffer(raw,np.uint8).reshape(stop-start,h,w,3);canvas=Image.new('RGB',(7*w,2*(h+28)),(12,15,20));font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
for k,i in enumerate(indices):
 f=i+a.source_start;canvas.paste(Image.fromarray(x[i-start]),((k%7)*w,(k//7)*(h+28)));d=ImageDraw.Draw(canvas);d.text(((k%7)*w+4,(k//7)*(h+28)+h+4),f'f{f}  {f/25:.2f}s',font=font,fill='yellow' if f==a.frame else 'white')
a.out.parent.mkdir(parents=True,exist_ok=True);canvas.save(a.out)
receipt={'source_video':str(a.video),'source_video_sha256':hashlib.sha256(a.video.read_bytes()).hexdigest(),'source_start_frame':a.source_start,'boundary_frame':a.frame,'selected_source_frames':[i+a.source_start for i in indices],'contact_sheet':str(a.out),'method':'Exact decoded video frames, no interpolation; source frame = local frame + source_start.'}
a.out.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n');print(a.out)
