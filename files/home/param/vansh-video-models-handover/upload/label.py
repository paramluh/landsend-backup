"""
Labelled GT | gen side-by-sides of a sink_universal.py run: a 128 px band above (seen/unseen,
kind, the person's training hours, uuid, seed, set index; GT and gen captions), NN_<hours>h_<seen|unseen>_<kind>_
<actor>.mp4, plus manifest.csv. hours = sum of `duration` over the person's (uuid) rows in TRAIN_CSV; seen = hours > 0.
With REFS_ROOT (refs runs), a panel below the gen window: RUN_DIR/sample_<n>_refs.png (the 9 refs the model gets,
3 x 3) scaled to the video window, each ref 1/3 of it, refs from inside this chunk (frame_index in
[chunk * CHUNK_FRAMES, (chunk + 1) * CHUNK_FRAMES)) outlined red; refs_in_chunk goes in the manifest. If REFS_ROOT is a pick.py dir (has manifest.json; banks of
<uuid>/property_<N>/ that keep no frame index), the grid is REFS_ROOT/sample_<n>_refs.png (image.py, a missing
property black) and the panel is instead a strip across the full width: refs 1-5 on top, 6-9 centred below, 102 x 192
tiles, slot number on each tile, a missing property a black 'N missing' tile (512 x 992 in all); refs_present
(9 x 0/1, property order) goes in the manifest.
With --stabilised STAB_LABEL (a bimodal/sampler.py run, no refs): three panels GT | stabilised | gen (768 px wide),
RUN_DIR/sample_<n>_stabilised.mp4 between GT and RUN_DIR/sample_<n>.mp4, audio from the latter.
Usage: label.py CSV TRAIN_CSV ROWS RUN_DIR OUT_DIR GEN_LABEL SET_NAME SEED [REFS_ROOT] [--stabilised STAB_LABEL]
  (ROWS = start,end, or a list of > 2 csv rows, e.g. the yaml's rows)
"""
import json
import collections
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from safetensors import safe_open

args = sys.argv[1:]
STAB = None
if '--stabilised' in args:
    i = args.index('--stabilised')
    STAB = args[i + 1]
    del args[i:i + 2]
src, train, ROWS, RUN, OUT, GEN, SET, SEED = args[0], args[1], args[2], Path(args[3]), Path(args[4]), *args[5:8]
REFS_ROOT = args[8] if len(args) > 8 else None
assert not (STAB and REFS_ROOT), 'three panels take no refs'
GENS = [STAB, GEN] if STAB else [GEN]
PICKS = json.load(open(os.path.join(REFS_ROOT, 'manifest.json'))) if REFS_ROOT and os.path.exists(
    os.path.join(REFS_ROOT, 'manifest.json')) else None
W, HV, BAND = 256, 480, 128
CHUNK_FRAMES = 7500
STABILISED_MP4 = 'sample_{}_stabilised.mp4'
FONTS = '/usr/share/fonts/truetype/dejavu/'


def font(name, size):
    return ImageFont.truetype(FONTS + name, size)


def fit(name, size, text, width):
    while font(name, size).getlength(text) > width:
        size -= 1
    return font(name, size)


def band(row, path):
    width = W * (1 + len(GENS))
    img = Image.new('RGB', (width, BAND), (0, 0, 0))
    d = ImageDraw.Draw(img)
    lines = [
        (f"{row['seen']}  ·  {row['kind']}  ·  {float(row['train_hours']):.1f} h train", 'DejaVuSans-Bold.ttf', 22, 6),
        (row['uuid'], 'DejaVuSansMono.ttf', 18, 37),
        (f"seed {row['seed']}  ·  {SET} #{int(row['index']):02d}", 'DejaVuSans.ttf', 17, 63),
    ]
    for text, name, size, y in lines:
        d.text((width // 2, y), text, font=fit(name, size, text, width - 8), fill=(255, 255, 255), anchor='mt')
    for i, text in enumerate(['GT', *GENS]):
        d.text((W * i + W // 2, 96), text, font=fit('DejaVuSans-Bold.ttf', 21, text, W - 8), fill=(255, 255, 255),
               anchor='mt')
    img.save(path)


def in_chunk(row):
    with safe_open(os.path.join(REFS_ROOT, *row['paths'].split('/')[1:3], 'refs.safetensors'), framework='pt') as f:
        frames = f.get_tensor('frame_index').tolist()
    chunk = int(row['paths'].rstrip('/').split('/')[-1])
    return [chunk * CHUNK_FRAMES <= x < (chunk + 1) * CHUNK_FRAMES for x in frames]


def strip(row, grid_png, path):
    """The 9 refs across the full width: 1-5 on top, 6-9 centred below, ref aspect, slot number on each tile."""
    grid = Image.open(grid_png).convert('RGB')
    cw, ch = grid.width // 3, grid.height // 3
    tw = 2 * W // 5
    th = int(round(tw * ch / cw / 2)) * 2
    img = Image.new('RGB', (2 * W, 2 * th), (0, 0, 0))
    d = ImageDraw.Draw(img)
    f = font('DejaVuSans-Bold.ttf', 14)
    for slot, present in enumerate(row['refs_present']):
        r, c = divmod(slot, 5)
        x, y = (2 * W - 5 * tw) // 2 + c * tw + (tw // 2 if r else 0), r * th
        if present == '1':
            tile = grid.crop(((slot % 3) * cw, (slot // 3) * ch, (slot % 3 + 1) * cw, (slot // 3 + 1) * ch))
            img.paste(tile.resize((tw, th), Image.LANCZOS), (x, y))
        label = f'{slot + 1}' if present == '1' else f'{slot + 1} missing'
        box = d.textbbox((x + 3, y + 2), label, font=f)
        d.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill=(0, 0, 0))
        d.text((x + 3, y + 2), label, font=f, fill=(255, 255, 255))
    img.save(path)


def panel(row, grid_png, path):
    if PICKS:
        return strip(row, grid_png, path)
    grid = Image.open(grid_png).convert('RGB')
    cw, ch = grid.width // 3, grid.height // 3
    d = ImageDraw.Draw(grid)
    for r, leak in enumerate(row['leak']):
        if leak:
            x, y = (r % 3) * cw, (r // 3) * ch
            d.rectangle([x, y, x + cw - 1, y + ch - 1], outline=(255, 0, 0), width=6)
    img = Image.new('RGB', (2 * W, HV), (0, 0, 0))
    img.paste(grid.resize((W, HV), Image.LANCZOS), (W, 0))
    d = ImageDraw.Draw(img)
    lines = [('9 refs (model input)', font('DejaVuSans-Bold.ttf', 20)),
             (f"{row['refs_present'].count('1')}/9 present" if PICKS else f"{row['refs_in_chunk']}/9 from this chunk",
              font('DejaVuSans.ttf', 18)),
             ('(missing = black)' if PICKS else '(outlined red)', font('DejaVuSans.ttf', 16))]
    for i, (text, f) in enumerate(lines):
        d.text((W // 2, HV // 2 - 40 + 30 * i), text, font=f, fill=(255, 255, 255), anchor='mt')
    img.save(path)


def ffmpeg(gt, gen, band_png, panel_png, out, gen2=None):
    if gen2:
        filt = (f'[0:v][1:v][2:v]hstack=inputs=3:shortest=1[s];[s]pad=iw:ih+{BAND}:0:{BAND}:black[p];'
                f'[p][3:v]overlay=0:0[v]')
        return ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(gt), '-i', str(gen), '-i', str(gen2), '-i', str(band_png),
                '-filter_complex', filt, '-map', '[v]', '-map', '2:a?', '-c:v', 'libx264', '-crf', '18', '-preset',
                'medium', '-pix_fmt', 'yuv420p', '-threads', '8', '-c:a', 'copy', '-shortest', str(out)]
    inputs = ['-i', str(gt), '-i', str(gen), '-i', str(band_png)]
    panel_h = Image.open(panel_png).height if panel_png else 0
    filt = f'[0:v][1:v]hstack=inputs=2:shortest=1[s];[s]pad=iw:ih+{BAND + panel_h}:0:{BAND}:black[p];'
    if panel_png:
        inputs += ['-i', str(panel_png)]
        filt += f'[p][2:v]overlay=0:0[q];[q][3:v]overlay=0:{BAND + HV}[v]'
    else:
        filt += '[p][2:v]overlay=0:0[v]'
    return ['ffmpeg', '-y', '-loglevel', 'error', *inputs, '-filter_complex', filt, '-map', '[v]', '-map', '1:a?',
            '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-threads', '8', '-c:a', 'copy',
            '-shortest', str(out)]


def name(row):
    return f"{int(row['index']):02d}_{float(row['train_hours']):g}h_{row['seen']}_{row['kind']}_{row['actor']}.mp4"


def one(row):
    n = int(row['index'])
    gen = RUN / f'sample_{n}.mp4'
    stab = RUN / STABILISED_MP4.format(n) if STAB else None
    if not gen.exists() or (stab and not stab.exists()):
        return name(row), 'missing'
    band(row, TMP / f'band_{n:02d}.png')
    panel_png = None
    if REFS_ROOT:
        panel_png = TMP / f'panel_{n:02d}.png'
        panel(row, Path(REFS_ROOT if PICKS else RUN) / f'sample_{n}_refs.png', panel_png)
    left, right = (stab, gen) if STAB else (gen, None)
    r = subprocess.run(ffmpeg(RUN / f'samples_{n}.mp4', left, TMP / f'band_{n:02d}.png', panel_png, OUT / name(row), right),
                       capture_output=True, text=True)
    return name(row), r.returncode or None


seconds = collections.defaultdict(float)
for r in csv.DictReader(open(train)):
    seconds[r['uuid']] += float(r['duration'])
nums = list(map(int, ROWS.split(',')))
keep = range(*nums) if len(nums) == 2 else nums
rows = []
for n, r in enumerate(csv.DictReader(open(src))):
    if n in keep:
        hours = round(seconds[r['uuid']] / 3600, 2)
        row = {'index': n, 'actor': r['uuid'][:8], 'train_hours': hours, 'seen': 'seen' if hours else 'unseen',
               'seed': int(SEED) + n, **r}
        if PICKS:
            assert PICKS[str(n)]['paths'] == r['paths']
            row['leak'] = [False] * 9
            row['refs_present'] = ''.join(map(str, PICKS[str(n)]['present']))
        elif REFS_ROOT:
            row['leak'] = in_chunk(r)
            row['refs_in_chunk'] = sum(row['leak'])
        rows.append(row)
TMP = Path(tempfile.mkdtemp())
OUT.mkdir(parents=True, exist_ok=True)
with ThreadPoolExecutor(4) as pool:
    results = list(pool.map(one, rows))
failed = [(nm, e) for nm, e in results if e]
fields = ['file', 'generator', *[k for k in rows[0] if k != 'leak']]
with open(OUT / 'manifest.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fields, extrasaction='ignore')
    w.writeheader()
    w.writerows({'file': name(r), 'generator': ' | '.join(GENS), **r} for r in rows)
shutil.rmtree(TMP)
print('LABEL_DONE', len(rows) - len(failed), 'ok', 'failed:', failed,
      *(['refs in chunk', sum(r['refs_in_chunk'] for r in rows), '/', 9 * len(rows)] if REFS_ROOT and not PICKS else []),
      flush=True)
