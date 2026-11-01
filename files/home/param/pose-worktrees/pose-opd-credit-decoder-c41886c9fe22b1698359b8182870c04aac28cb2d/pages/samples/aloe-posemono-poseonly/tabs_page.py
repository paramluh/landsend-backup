"""Pose eval page, one TAB PER CARD (5, 13, 15, 20), 4 videos each (2 x 2), all with the white flash at 2.0-2.2 s:
  Full AR (ours)                           rv/S-NN-gen-flash.mp4 (existing, untouched)
  Ground truth                             rv/GT-<id>-flash.mp4  (real pose + real speech of the same recording, same DiT / sink / refs / seed)
  Stick figure (ours, generated pose)      rv/STICK-S-NN.mp4     (video 1's generated pose + its generated audio)
  SAM 8B: own speech + pose, from our generated text (first 60 s)   rv/SAM8B-OWN-S-NN-flash.mp4
+ a checks sheet (0.5 / 2.5 / 30 s of every video).   python tabs_page.py      (compute node, CPU)"""
import html
import json
import os
import subprocess

from PIL import Image

W = "/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly"
K = "/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_posemono_poseonly"
V = "/data/home/amartyavarma/video/studio-posemono"
F = "/data/home/amartyavarma/tools/ffmpeg-static/ffmpeg-7.0.2-amd64-static/ffmpeg"
O = f"{W}/eval5/proof-tabs"
os.makedirs(O, exist_ok=True)
gt = json.load(open(f"{W}/gt-studio/gt.json"))
ROWS = [(5, "S-05", "10_27286bdd"), (13, "S-13", "06_74807fe0"), (15, "S-15", "10_27286bdd"), (20, "S-20", "19_47473a46")]
cp = lambda s, d: (subprocess.run([F, "-y", "-v", "error", "-i", s, "-c", "copy", "-movflags", "+faststart", f"{K}/rv/.tmp-{os.path.basename(d)}"], check=True),  # noqa: E731
                   os.replace(f"{K}/rv/.tmp-{os.path.basename(d)}", d))
for idn in gt:
    cp(f"{V}/gt/{idn}/video-flash.mp4", f"{K}/rv/GT-{idn}-flash.mp4")
for card, row, idn in ROWS:
    assert os.path.exists(f"{V}/sam8b-own/{row}/DONE") and os.path.exists(f"{W}/stick/{row}.mp4"), row
    cp(f"{W}/stick/{row}.mp4", f"{K}/rv/STICK-{row}.mp4")
    cp(f"{V}/sam8b-own/{row}/video-flash.mp4", f"{K}/rv/SAM8B-OWN-{row}-flash.mp4")
LAB = ["Full AR (ours)", "Ground truth", "Stick figure (ours, generated pose)", "SAM 8B: own speech + pose, from our generated text (first 60 s)"]
sheet = Image.new("RGB", (256 * 12, 480 * 4))
E = html.escape
tabs, panes = [], []
for i, (card, row, idn) in enumerate(ROWS):
    vids = [f"rv/{row}-gen-flash.mp4", f"rv/GT-{idn}-flash.mp4", f"rv/STICK-{row}.mp4", f"rv/SAM8B-OWN-{row}-flash.mp4"]
    gs = float(gt[idn]["seconds"])
    for j, v in enumerate(vids):
        for q, t in enumerate((0.5, 2.5, 30.0)):
            tt = min(t, gs - 0.5) if j == 1 else t
            p = f"{O}/{row}-{j}-{q}.png"
            subprocess.run([F, "-v", "error", "-y", "-ss", str(tt), "-i", f"{K}/{v}", "-frames:v", "1", p], check=True)
            sheet.paste(Image.open(p).convert("RGB").resize((256, 480)), (256 * (3 * j + q), 480 * i))
    note = {5: "Cards 5 and 15 share the reference video 10_27286bdd; card 15 is a second, independent sample.",
            15: "Cards 5 and 15 share the reference video 10_27286bdd; this is the second, independent sample (its ground truth is the same as card 5's)."}.get(card, "")
    gtn = "" if gs >= 59.9 else f" Ground truth stops at {gs:.1f} s (our store row's continuous segment for this speaker ends there)."
    cells = "".join(f'<figure><figcaption>{E(LAB[j])}</figcaption><video controls preload="metadata" playsinline src="{v}"></video></figure>' for j, v in enumerate(vids))
    tabs.append(f'<button role="tab" aria-selected="{str(i == 0).lower()}" data-t="{i}">Card {card}</button>')
    panes.append(f'''<section class="pane" data-t="{i}"{"" if i == 0 else " hidden"}><p class="m">Reference video {E(idn)}. 2 s prefill from it (pose + audio, starting image + reference images), then our model generates words, speech and pose and the DiT renders it. White flash = end of the 2 s prefill. {E(note)}{E(gtn)}</p><div class="q">{cells}</div></section>''')
sheet.resize((1536, 480)).save(f"{O}/sheet.png")
page = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pose Eval Videos</title>
<style>:root{{--bg:#f4f5f7;--fg:#151b23;--mu:#5b6673;--card:#fff;--line:#d8dde3;--ac:#2563c9}}@media (prefers-color-scheme:dark){{:root{{--bg:#10141a;--fg:#e4e8ee;--mu:#9aa4b1;--card:#171c24;--line:#2a313c;--ac:#6b9cf0}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}.w{{max-width:1100px;margin:0 auto;padding:20px 16px}}
.tabs{{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}}.tabs button{{font:inherit;padding:6px 14px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);cursor:pointer}}
.tabs button[aria-selected="true"]{{background:var(--ac);border-color:var(--ac);color:#fff}}
.q{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}}figure{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px}}
figcaption{{font-size:13.5px;font-weight:600;margin-bottom:6px}}video{{width:100%;border-radius:6px;background:#000}}.m{{color:var(--mu);font-size:13.5px}}</style></head><body><div class="w">
<h2 style="margin:0">Pose eval: your 4 picks</h2>
<div class="tabs" role="tablist">{"".join(tabs)}</div>
{"".join(panes)}
</div><script>
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{{document.querySelectorAll('.tabs button').forEach(x=>x.setAttribute('aria-selected',String(x===b)));
document.querySelectorAll('.pane').forEach(p=>{{p.hidden=p.dataset.t!==b.dataset.t;if(p.hidden)p.querySelectorAll('video').forEach(v=>v.pause())}})}});
</script></body></html>'''
open(f"{K}/index.html.tmp", "w").write(page)
os.replace(f"{K}/index.html.tmp", f"{K}/index.html")
print("TABS PAGE DONE")
