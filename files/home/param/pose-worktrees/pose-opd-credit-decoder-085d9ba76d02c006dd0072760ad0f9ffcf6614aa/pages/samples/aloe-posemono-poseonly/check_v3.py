"""(v3) frames at 0.5, 2.5 and 20 s of every page video + the card image -> one sheet; audio of each generated mp4 vs the generated (s1) and real (gt_s1)
waveforms after 2 s (correlation of 24 kHz samples). python check_allgen.py <page dir> <out dir>"""
import json
import subprocess
import sys

import numpy as np
import soundfile as sf
from PIL import Image

K, O = sys.argv[1], sys.argv[2]
W = "/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly"
F = "/data/home/amartyavarma/tools/ffmpeg-static/ffmpeg-7.0.2-amd64-static/ffmpeg"
C = "/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_posemono_candidates/r"
rows = [("R-004", 5), ("R-012", 13), ("R-014", 15), ("R-019", 20)]
sheet = Image.new("RGB", (256 * 7, 480 * 4))
for i, (r, c) in enumerate(rows):
    ims = [Image.open(f"{C}/c{c:02d}.jpg").convert("RGB")]
    for k in ("gen", "real"):
        for t in (0.5, 2.5, 20):
            p = f"{O}/{r}-{k}-{t}.png"
            subprocess.run([F, "-v", "error", "-y", "-ss", str(t), "-i", f"{K}/rv/{r}-{k}.mp4", "-frames:v", "1", p], check=True)
            ims.append(Image.open(p).convert("RGB"))
    for j, im in enumerate(ims):
        sheet.paste(im.resize((256, 480)), (256 * j, 480 * i))
    a = subprocess.run([F, "-v", "error", "-i", f"{K}/rv/{r}-gen.mp4", "-ac", "1", "-ar", "24000", "-f", "f32le", "-"], check=True, capture_output=True).stdout
    x = np.frombuffer(a, np.float32)
    g, _ = sf.read(f"{W}/runs-allgen/{r}/s1.wav", dtype="float32")
    t, _ = sf.read(f"{W}/runs-allgen/{r}/gt_s1.wav", dtype="float32")
    n = min(len(x), len(g), len(t))
    s0 = 2 * 24000
    cg = float(np.corrcoef(x[s0:n], g[s0:n])[0, 1])
    ct = float(np.corrcoef(x[s0:n], t[s0:n])[0, 1])
    print(json.dumps({"row": r, "card": c, "gen_mp4_audio_vs_generated_corr": round(cg, 3), "vs_real_corr": round(ct, 3)}), flush=True)
sheet.resize((896, 960)).save(f"{O}/sheet.png")
print("sheet", f"{O}/sheet.png")
