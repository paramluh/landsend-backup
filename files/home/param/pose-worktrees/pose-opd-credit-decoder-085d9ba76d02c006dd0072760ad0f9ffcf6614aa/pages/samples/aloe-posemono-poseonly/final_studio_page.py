"""STUDIO final eval page + checks (10-19). Copies studio/<row>/video.mp4 -> klweb/ppm_aloe_posemono_poseonly/rv/<row>-gen.mp4 (faststart) and writes
the simple page (only the 4 generated videos, one label each). Checks: frames at 0.5 / 2.5 / 30 s of each video next to the card image (sheet),
and the video's audio 0-2 s vs the Mimi decode of the package's prefill_audio (correlation, max abs diff).
   python final_studio_page.py      (qwen-s2 venv; compute node, CPU)"""
import html
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/data/home/amartyavarma/ppm/code")
import infer_ppm as I  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

W = "/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly"
K = "/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_posemono_poseonly"
M = "/data/home/amartyavarma/video/studio-posemono/studio"
C = "/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_posemono_candidates/r"
PR = "/data/home/amartyavarma/video/demo-data/prompts"
F = "/data/home/amartyavarma/tools/ffmpeg-static/ffmpeg-7.0.2-amd64-static/ffmpeg"
O = f"{W}/eval5/proof-studio"
os.makedirs(O, exist_ok=True)
os.makedirs(f"{K}/rv", exist_ok=True)
ROWS = [(5, "S-05", "10_27286bdd"), (13, "S-13", "06_74807fe0"), (15, "S-15", "10_27286bdd"), (20, "S-20", "19_47473a46")]
mimi = I.Mimi("cpu")
sheet = Image.new("RGB", (256 * 4, 480 * 4))
res = []
for i, (card, row, idn) in enumerate(ROWS):
    src = f"{M}/{row}/video.mp4"
    assert os.path.exists(f"{M}/{row}/DONE"), row
    subprocess.run([F, "-y", "-v", "error", "-i", src, "-c", "copy", "-movflags", "+faststart", f"{K}/rv/.{row}-gen.tmp.mp4"], check=True)
    os.replace(f"{K}/rv/.{row}-gen.tmp.mp4", f"{K}/rv/{row}-gen.mp4")
    ims = [Image.open(f"{C}/c{card:02d}.jpg").convert("RGB")]
    for t in (0.5, 2.5, 30):
        p = f"{O}/{row}-{t}.png"
        subprocess.run([F, "-v", "error", "-y", "-ss", str(t), "-i", src, "-frames:v", "1", p], check=True)
        ims.append(Image.open(p).convert("RGB"))
    for j, im in enumerate(ims):
        sheet.paste(im.resize((256, 480)), (256 * j, 480 * i))
    pa = load_file(f"{PR}/{idn}/llm_inputs.safetensors")["prefill_audio"].astype(np.int64)
    ref = mimi.decode(pa)
    raw = subprocess.run([F, "-v", "error", "-i", src, "-t", "2.0", "-ac", "1", "-ar", "24000", "-f", "f32le", "-"], check=True, capture_output=True).stdout
    x = np.frombuffer(raw, np.float32)
    n = min(len(x), len(ref)) - 2400
    r = {"card": card, "row": row, "identity": idn, "audio_0_1p9s_corr_vs_package_prefill": round(float(np.corrcoef(x[:n], ref[:n])[0, 1]), 4),
         "audio_0_1p9s_max_absdiff": round(float(np.abs(x[:n] - ref[:n]).max()), 4),
         "duration": float(subprocess.run([F[:-6] + "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src],
                                          check=True, capture_output=True, text=True).stdout)}
    res.append(r)
    print(json.dumps(r), flush=True)
sheet.resize((768, 1440)).save(f"{O}/sheet.png")
json.dump(res, open(f"{O}/checks.json", "w"), indent=1)
E = html.escape
cards = "\n".join(f'<div class="c"><h3>Card {card}</h3><video controls preload="metadata" playsinline src="rv/{row}-gen.mp4"></video>'
                  f'<div class="m">{E(f"card {card} · prefill = 2 s of pose + audio from your reference video, starting image + reference images from it, then LLM and DiT autoregressed")}</div></div>'
                  for card, row, _ in ROWS)
page = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pose Eval Videos</title>
<style>:root{{--bg:#f4f5f7;--fg:#151b23;--mu:#5b6673;--card:#fff;--line:#d8dde3}}@media (prefers-color-scheme:dark){{:root{{--bg:#10141a;--fg:#e4e8ee;--mu:#9aa4b1;--card:#171c24;--line:#2a313c}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}.w{{max-width:1100px;margin:0 auto;padding:20px 16px}}
.g{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}}.c{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}}
video{{width:100%;border-radius:6px;background:#000}}h3{{margin:0 0 6px}}.m{{color:var(--mu);font-size:13.5px}}</style></head><body><div class="w">
<h2 style="margin:0 0 4px">Pose eval: your 4 picks</h2>
<div class="g">
{cards}
</div></div></body></html>'''
open(f"{K}/index.html.tmp", "w").write(page)
os.replace(f"{K}/index.html.tmp", f"{K}/index.html")
print("PAGE DONE")
