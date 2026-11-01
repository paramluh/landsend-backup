"""Verified HF source video + raw detector and exact rollout-token overlays."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'python'))
import numpy as np
from PIL import Image,ImageDraw,ImageFont

SRC=ROOT/'source/hf/source-speaker01-hf.mp4'
RAW=np.load(ROOT/'source/hf/poses_25fps.npy',mmap_mode='r')
SEL=list(range(13))+list(range(91,133))
DECODED=np.load(ROOT/'audit/rollout_pixel_yxc.npy')
PARENTS=[-1,0,0,1,2,-1,-1,5,6,7,8,5,6]
for root,wrist in [(13,9),(34,10)]:
    PARENTS.append(wrist)
    for finger in range(5):
        base=root+1+4*finger
        PARENTS.extend([root,base,base+1,base+2])
EDGES=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12)]
for root in [13,34]:
    for finger in range(5):
        base=root+1+4*finger
        EDGES.extend([(root,base),(base,base+1),(base+1,base+2),(base+2,base+3)])
W,H,TOP,BOTTOM=384,720,100,72
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
SMALL=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
BG=(15,19,27)

def color(c):
    c=float(np.clip(c,0,1));t=float(np.clip((c-.2)/.4,0,1));fade=float(np.clip((c-.05)/.15,0,1))
    r=255 if t<.5 else 255*(1-(t-.5)/.5)
    g=255*t/.5 if t<.5 else 255
    return (round(r*fade),round(g*fade),0)

def overlay(im,pose):
    # Verified FFmpeg crop: source1440x2560 -> (62,0,1228,2304) ->512x960.
    # Display scales that same aspect to384x720.
    xy=np.stack([(pose[:,1]-62)*W/1228,pose[:,0]*H/2304],axis=-1)
    c=pose[:,2]
    d=ImageDraw.Draw(im)
    for i,j in EDGES:
        v=min(c[i],c[j])
        if v>=.05:d.line([tuple(xy[i]),tuple(xy[j])],fill=color(v),width=2)
    for i,(x,y) in enumerate(xy):
        if c[i]>=.05:d.ellipse((x-2,y-2,x+2,y+2),fill=color(c[i]))
    return im

def render(seconds,name,three=True):
    ncols=3 if three else 2
    out=ROOT/name
    dec=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-threads','2','-i',str(SRC),
          '-t',str(seconds),'-vf',f'scale={W}:{H}:flags=lanczos','-pix_fmt','rgb24','-f','rawvideo','pipe:1'],stdout=subprocess.PIPE)
    enc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
          '-s',f'{ncols*W}x{TOP+H+BOTTOM}','-r','25','-i','pipe:0','-i',str(SRC),'-map','0:v','-map','1:a:0',
          '-t',str(seconds),'-c:v','libx264','-threads','2','-preset','fast','-crf','21','-pix_fmt','yuv420p',
          '-c:a','aac','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
    for f in range(int(seconds*25)):
        buf=dec.stdout.read(W*H*3)
        if len(buf)!=W*H*3:raise RuntimeError(f'source truncated at frame{f}')
        source=Image.frombytes('RGB',(W,H),buf)
        panels=[overlay(source.copy(),RAW[f,SEL]),overlay(source.copy(),DECODED[f])]
        labels=['Original extracted pose','Decoded saved rollout pose']
        if three:panels.insert(0,source);labels.insert(0,'Source video')
        canvas=Image.new('RGB',(ncols*W,TOP+H+BOTTOM),BG);d=ImageDraw.Draw(canvas)
        d.text((14,9),f'48f6d2d3 / SPEAKER_01 / update 8, sample 4 / {f/25:05.2f}s',font=FONT,fill='white')
        phase='0-2s: DATASET PREFILL (all pose codes match source)' if f<50 else 'After 2s: STUDENT POSE (not expected to follow source video)'
        d.text((14,36),phase,font=SMALL,fill=(110,240,170) if f<50 else (255,174,100))
        for k,(im,label) in enumerate(zip(panels,labels)):
            canvas.paste(im,(k*W,TOP));d.text((k*W+10,75),label,font=SMALL,fill='white')
        d.text((14,TOP+H+9),'Native confidence colors: red 0.2 / yellow 0.4 / green 0.6+; fade below 0.2',font=SMALL,fill='white')
        d.text((14,TOP+H+34),'HF source video and source audio; verified crop transform and 0s start. No pose smoothing.',font=SMALL,fill='white')
        enc.stdin.write(canvas.tobytes())
        if f in [0,25,49] and three:canvas.save(ROOT/f'video-overlay-frame-{f:03d}.png')
    dec.stdout.close();enc.stdin.close()
    if dec.wait() or enc.wait():raise RuntimeError('ffmpeg failed')
    print(out,flush=True)

if __name__=='__main__':
    render(10,'video-overlay-first-10s.mp4')
    render(120,'video-overlay-full-120s.mp4',False)
