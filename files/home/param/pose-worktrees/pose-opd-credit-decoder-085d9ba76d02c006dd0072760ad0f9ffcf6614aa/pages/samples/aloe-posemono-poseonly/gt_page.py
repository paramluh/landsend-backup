"""Page with Generated (existing rv/S-NN-gen-flash.mp4, untouched) next to the rendered GROUND TRUTH (pose + audio) per card + checks sheet.
   python gt_page.py      (compute node, CPU)"""
import html
import json
import os
import subprocess

import numpy as np
from PIL import Image

W = "/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly"
K = "/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_posemono_poseonly"
GD = "/data/home/amartyavarma/video/studio-posemono/gt"
F = "/data/home/amartyavarma/tools/ffmpeg-static/ffmpeg-7.0.2-amd64-static/ffmpeg"
O = f"{W}/eval5/proof-gt"
os.makedirs(O, exist_ok=True)
gt = json.load(open(f"{W}/gt-studio/gt.json"))
ROWS = [(5, "S-05", "10_27286bdd"), (13, "S-13", "06_74807fe0"), (15, "S-15", "10_27286bdd"), (20, "S-20", "19_47473a46")]
for idn in gt:
    assert os.path.exists(f"{GD}/{idn}/DONE"), idn
    subprocess.run([F, "-y", "-v", "error", "-i", f"{GD}/{idn}/video-flash.mp4", "-c", "copy", "-movflags", "+faststart", f"{K}/rv/.GT-{idn}.tmp.mp4"], check=True)
    os.replace(f"{K}/rv/.GT-{idn}.tmp.mp4", f"{K}/rv/GT-{idn}-flash.mp4")
sheet = Image.new("RGB", (256 * 6, 480 * 4))
for i, (card, row, idn) in enumerate(ROWS):
    dur = float(gt[idn]["seconds"])
    for j, (src, t) in enumerate([(f"{K}/rv/{row}-gen-flash.mp4", 0.5), (f"{K}/rv/{row}-gen-flash.mp4", 2.5), (f"{K}/rv/{row}-gen-flash.mp4", 30),
                                  (f"{K}/rv/GT-{idn}-flash.mp4", 0.5), (f"{K}/rv/GT-{idn}-flash.mp4", 2.5), (f"{K}/rv/GT-{idn}-flash.mp4", min(30, dur - 0.5))]):
        p = f"{O}/{row}-{j}.png"
        subprocess.run([F, "-v", "error", "-y", "-ss", str(t), "-i", src, "-frames:v", "1", p], check=True)
        sheet.paste(Image.open(p).convert("RGB").resize((256, 480)), (256 * j, 480 * i))
sheet.resize((768, 960)).save(f"{O}/sheet.png")
E = html.escape
cards = []
for card, row, idn in ROWS:
    s = float(gt[idn]["seconds"])
    note = "" if s >= 59.9 else f" Ground truth stops at {s:.1f} s: our store row's continuous segment for this speaker ends there."
    cards.append(f'''<div class="c"><h3>Card {card}</h3><div class="p"><figure><video controls preload="metadata" playsinline src="rv/{row}-gen-flash.mp4"></video><figcaption>Generated</figcaption></figure>
<figure><video controls preload="metadata" playsinline src="rv/GT-{idn}-flash.mp4"></video><figcaption>Ground truth pose + audio (rendered)</figcaption></figure></div>
<div class="m">{E(f"card {card} · prefill = 2 s of pose + audio from your reference video, starting image + reference images from it, then LLM and DiT autoregressed · right: the real pose + real audio of the same recording, rendered by the same DiT (same starting image, references, seed). White flash = end of the 2 s prefill." + note)}</div></div>''')
page = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pose Eval Videos</title>
<style>:root{{--bg:#f4f5f7;--fg:#151b23;--mu:#5b6673;--card:#fff;--line:#d8dde3}}@media (prefers-color-scheme:dark){{:root{{--bg:#10141a;--fg:#e4e8ee;--mu:#9aa4b1;--card:#171c24;--line:#2a313c}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}.w{{max-width:1100px;margin:0 auto;padding:20px 16px}}
.g{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}}.c{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}}
.p{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}figure{{margin:0}}figcaption{{font-size:13px;color:var(--mu);text-align:center}}
video{{width:100%;border-radius:6px;background:#000}}h3{{margin:0 0 6px}}.m{{color:var(--mu);font-size:13.5px;margin-top:6px}}
@media (max-width:480px){{.g{{grid-template-columns:1fr}}}}</style></head><body><div class="w">
<h2 style="margin:0 0 4px">Pose eval: your 4 picks</h2>
<div class="g">
{chr(10).join(cards)}
</div></div></body></html>'''
open(f"{K}/index.html.tmp", "w").write(page)
os.replace(f"{K}/index.html.tmp", f"{K}/index.html")
print("GT PAGE DONE", json.dumps({k: v["seconds"] for k, v in gt.items()}))
