"""Export v1's 19 <= t < 21 second interval, holding every original frame for 1s."""
from pathlib import Path
import hashlib
import json
import math
import subprocess

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source.mp4'
POSE = ROOT / 'pose_overlay.json'
OUT = ROOT / 'v1-19-21s-pose-25x-slow.mp4'
pose = json.loads(POSE.read_text())
assert pose['fps'] == 25 and len(pose['frames']) == 632
assert (pose['width'], pose['height']) == (256, 480)
first, stop = 475, 525
raw = subprocess.check_output([
    'ffmpeg', '-v', 'error', '-i', str(SOURCE), '-an',
    '-vf', f'select=between(n\\,{first}\\,{stop-1})', '-vsync', '0',
    '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1',
])
w, h = 256, 480
assert len(raw) == (stop-first)*w*h*3, len(raw)
W, H = 512, 960
footer = 100
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 19)
small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 17)
threshold = pose['confidence']
minimum, low, high = threshold['min'], threshold['low'], threshold['high']
def color(conf, opacity):
    t = max(0, min(1, (conf-low)/(high-low)))
    alpha = max(0, min(1, (conf-minimum)/(low-minimum)))
    return (round(255*min(1, 2*(1-t))), round(255*min(1, 2*t)), 70,
            round(255*opacity*alpha))

enc = subprocess.Popen([
    'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
    '-s', f'{W}x{H+footer}', '-framerate', '1', '-i', 'pipe:0', '-an',
    '-vf', 'fps=25', '-c:v', 'libx264', '-preset', 'medium', '-crf', '16',
    '-threads', '4', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    '-metadata', 'title=v1: 19-21 seconds with pose, 25x slow motion',
    '-metadata', 'comment=Every original 25fps frame is held for one second. No temporal interpolation. Source frame numbers are zero-based.',
    str(OUT),
], stdin=subprocess.PIPE)
for local, source_index in enumerate(range(first, stop)):
    start = local*w*h*3
    base = Image.frombytes('RGB', (w, h), raw[start:start+w*h*3]).resize((W, H), Image.Resampling.LANCZOS).convert('RGBA')
    points = pose['frames'][source_index]
    valid = lambda p: len(p) >= 3 and all(math.isfinite(x) for x in p) and p[2] > minimum
    xy = lambda p: (p[0]*W, p[1]*H)
    width = max(1.5, W/160)
    # Composite each primitive so crossings retain the browser overlay's opacity.
    for a, b in pose['edges']:
        pa, pb = points[a], points[b]
        if not (valid(pa) and valid(pb)):
            continue
        conf = min(pa[2], pb[2])
        layer = Image.new('RGBA', base.size)
        d = ImageDraw.Draw(layer)
        alpha = color(conf, .85)[3]
        for stroke, rgba in ((round(width+1.5), (0, 0, 0, round(alpha*.65))),
                             (round(width), color(conf, .85))):
            d.line([xy(pa), xy(pb)], fill=rgba, width=stroke)
            r = stroke/2
            for x, y in (xy(pa), xy(pb)):
                d.ellipse((x-r, y-r, x+r, y+r), fill=rgba)
        base = Image.alpha_composite(base, layer)
    for p in points:
        if valid(p):
            x, y = xy(p)
            r = max(1.4, W/145)
            layer = Image.new('RGBA', base.size)
            ImageDraw.Draw(layer).ellipse((x-r, y-r, x+r, y+r), fill=color(p[2], .9))
            base = Image.alpha_composite(base, layer)
    frame = Image.new('RGB', (W, H+footer), (15, 18, 24))
    frame.paste(base.convert('RGB'))
    d = ImageDraw.Draw(frame)
    d.text((14, H+10), f'v1 | source {source_index/25:06.3f}s | {local+1:02d}/50', font=font, fill=(245, 246, 248))
    d.text((14, H+40), f'Frame {source_index} | pose overlay', font=small, fill=(180, 191, 208))
    d.text((14, H+67), '25x slower | 1 source frame / second', font=small, fill=(180, 191, 208))
    enc.stdin.write(frame.tobytes())
    if local in (0, 24, 49):
        frame.save(ROOT / f'preview-{source_index}.png')
enc.stdin.close()
assert enc.wait() == 0
receipt = {
    'run_id': '20261030-235733-0277', 'video_id': 'v1',
    'source_interval_seconds': [19, 21], 'interval_end_exclusive': True,
    'source_frames_zero_based': [first, stop-1], 'source_frame_count': stop-first,
    'original_fps': 25, 'seconds_per_source_frame': 1, 'slowdown_factor': 25,
    'output_duration_seconds': 50, 'output_fps': 25,
    'temporal_interpolation': False, 'audio': 'omitted for frame inspection',
    'pose_source': 'Cached generated pose, exact renderer coordinates and frame indices',
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'pose_sha256': hashlib.sha256(POSE.read_bytes()).hexdigest(),
    'output_sha256': hashlib.sha256(OUT.read_bytes()).hexdigest(),
    'output': str(OUT),
}
(ROOT/'export.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
