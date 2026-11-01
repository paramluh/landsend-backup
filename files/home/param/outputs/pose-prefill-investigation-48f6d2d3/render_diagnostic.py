"""Render exact saved rollout and source-token geometry; CPU-only, no model edits."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'python'))
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ASSETS = Path('/home/param/outputs/pose-opd-credit-live-attempt2/samples/assets/update-000008-a80fc2d2e65013c0')
DOC = json.loads((ASSETS / 'u8-r0-s3-pose.json').read_text())
META = json.loads((ROOT / 'source/pose-metadata.SPEAKER_01.json').read_text())
CANON = np.asarray(DOC['frames'])
CONF = np.asarray(DOC['confidence'])
PIXEL = np.load(ROOT / 'audit/rollout_pixel_yxc.npy')[..., [1, 0]]
SOURCE = np.load(ROOT / 'audit/source_decoded_first10s.npz')
PARENTS = DOC['parents']
EDGES = [(i, p) for i, p in enumerate(PARENTS) if p >= 0] + [(5, 6), (11, 12)]
FONTFILE = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT = ImageFont.truetype(FONTFILE, 18)
SMALL = ImageFont.truetype(FONTFILE, 14)
BIG = ImageFont.truetype(FONTFILE, 23)
BG = (16, 20, 28)
WHITE = (235, 238, 245)
GREEN = (88, 232, 168)
GRAY = (116, 123, 139)
PINK = (255, 130, 182)
VW, VH, HEAD, FOOT = 360, 640, 100, 72


def skeleton(canvas, xy, conf, origin=(0, 0), color=GREEN, confidence=True, scale=1.0):
    d = ImageDraw.Draw(canvas)
    xy = xy * scale + np.asarray(origin)
    for i, p in EDGES:
        col = color if not confidence or min(conf[i], conf[p]) >= .3 else GRAY
        d.line([tuple(xy[i]), tuple(xy[p])], fill=col, width=2)
    for i, pt in enumerate(xy):
        col = color if not confidence or conf[i] >= .3 else GRAY
        x, y = pt
        d.ellipse((x-2, y-2, x+2, y+2), fill=col)


def camera_panel(xy, conf, label, color):
    im = Image.new('RGB', (VW, VH), (24, 29, 39))
    skeleton(im, xy, conf, color=color, scale=VW/1440)
    return im


def frame(t, full=False):
    i = int(round(t*25))
    if full:
        width = 2*VW
        panels = [camera_panel(PIXEL[i], CONF[i], '', GREEN)]
        # Show all joints with camera aspect preserved, including off-camera detections.
        fullxy = PIXEL[i]
        lo, hi = PIXEL.min((0,1)), PIXEL.max((0,1))
        scale = min((VW-30)/(hi[0]-lo[0]), (VH-30)/(hi[1]-lo[1]))
        origin = np.asarray([VW/2,VH/2]) - (lo+hi)/2*scale
        panel = Image.new('RGB', (VW,VH), BG)
        dd=ImageDraw.Draw(panel)
        dd.rectangle((*origin, *(origin+np.asarray([1440,2560])*scale)), outline=(69,77,95))
        skeleton(panel, fullxy, CONF[i], origin=origin, scale=scale)
        panels.append(panel)
        labels = ['Original camera proportions', 'All joints; box = camera frame']
    else:
        width = 3*VW
        # Reproduce existing dashboard projection with its fixed full-rollout bounds.
        lo, hi = CANON[...,:2].min((0,1)), CANON[...,:2].max((0,1))
        scale = min((VW-60)/max(.2,hi[0]-lo[0]), (VH-50)/max(.2,hi[1]-lo[1]))
        xy = np.stack([VW/2+(CANON[i,:,0]-(lo[0]+hi[0])/2)*scale,
                       VH-20-(CANON[i,:,1]-lo[1])*scale],axis=-1)
        panel = Image.new('RGB',(VW,VH),BG)
        # Exact existing parent edges: no added shoulder/hip edges.
        d=ImageDraw.Draw(panel)
        for j,p in enumerate(PARENTS):
            if p>=0: d.line([tuple(xy[j]),tuple(xy[p])],fill=GREEN,width=2)
        for x,y in xy: d.ellipse((x-2,y-2,x+2,y+2),fill=GREEN)
        panels = [panel, camera_panel(PIXEL[i],CONF[i],'',GREEN),
                  camera_panel(SOURCE['pixel_xy'][i],SOURCE['confidence'][i],'',PINK)]
        labels = ['Current dashboard geometry', 'Saved rollout; camera proportions', 'Original dataset tokens decoded']
    im=Image.new('RGB',(width,HEAD+VH+FOOT),BG)
    d=ImageDraw.Draw(im)
    state='PREFILL - DATASET TOKENS' if i<50 else 'STUDENT GENERATED (source panel remains dataset)'
    d.text((16,10),f'Update 8 / sample 4 / SPEAKER_01 (user) / {t:05.2f}s',font=BIG,fill=WHITE)
    d.text((16,42),state,font=FONT,fill=GREEN if i<50 else PINK)
    for j,(p,l) in enumerate(zip(panels,labels)):
        im.paste(p,(j*VW,HEAD))
        d.text((j*VW+10,76),l,font=SMALL,fill=WHITE)
        if j: d.line((j*VW,HEAD,j*VW,HEAD+VH),fill=(60,68,85))
    d.text((16,HEAD+VH+10),'Bright: confidence >= 0.3 | Gray: confidence < 0.3 | No smoothing or pose edits',font=SMALL,fill=WHITE)
    d.text((16,HEAD+VH+34),'First 2 seconds: unscored prefill. Pose is 2D; no source video attached.',font=SMALL,fill=WHITE)
    return im


def encode(path, seconds, full=False):
    first=frame(0,full)
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
         '-s',f'{first.width}x{first.height}','-r','25','-i','pipe:0',
         '-i',str(ASSETS/'u8-r0-s3-user.wav'),'-i',str(ASSETS/'u8-r0-s3-generated.wav'),
         '-filter_complex','[1:a][2:a]amix=inputs=2:normalize=0,volume=0.5[a]',
         '-map','0:v','-map','[a]','-t',str(seconds),'-c:v','libx264','-threads','2',
         '-preset','fast','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(path)]
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for i in range(int(seconds*25)):
            p.stdin.write(frame(i/25,full).tobytes())
    finally: p.stdin.close()
    if p.wait(): raise RuntimeError('ffmpeg failed')
    print(path,flush=True)


if __name__=='__main__':
    encode(ROOT/'pose-comparison-first-10s.mp4',10)
    encode(ROOT/'pose-rollout-full-120s.mp4',120,True)
    times=[0,.48,1,1.96]
    ims=[frame(t) for t in times]
    # A compact two-by-two view of the body/hands in original camera space.
    sheet=Image.new('RGB',(2*2*VW,2*(HEAD+VH+FOOT)),BG)
    for k,im in enumerate(ims):
        im=im.crop((VW,0,3*VW,im.height))
        sheet.paste(im,((k%2)*2*VW,(k//2)*im.height))
    sheet.save(ROOT/'prefill-contact-sheet.png')
    frame(0).save(ROOT/'frame-zero-comparison.png')
    print(ROOT/'prefill-contact-sheet.png',flush=True)
