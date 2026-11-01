"""Replaces the inference-parameters panel of klweb/ppm_aloe_posemono_poseonly/index.html with ONE minimal table. Values READ from the runs:
runs-studio/<row>/meta.json (ours), runs-sam8b-own/<row>/meta.json (SAM 8B), video/demo-v3-1019/settings.yaml + studio / sam8b-own SAMPLE_DIR
meta.json (video model; seeds, reference slots).      python add_params_min.py      (compute node; qwen-s2 venv)"""
import html
import json
import os
import re

import yaml

W = "/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly"
K = "/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_posemono_poseonly"
V = "/data/home/amartyavarma/video/studio-posemono"
ROWS = [(5, "S-05"), (13, "S-13"), (15, "S-15"), (20, "S-20")]
fa = {r: json.load(open(f"{W}/runs-studio/{r}/meta.json")) for _, r in ROWS}
sm = {r: json.load(open(f"{W}/runs-sam8b-own/{r}/meta.json")) for _, r in ROWS}
st = {r: json.load(open(f"{V}/studio/{r}/meta.json")) for _, r in ROWS}
SET = yaml.safe_load(open("/data/home/amartyavarma/video/demo-v3-1019/settings.yaml"))["video"]
a, s = fa["S-05"], fa["S-05"]["sampling"]
ss = sm["S-05"]["sampling"]
assert all(m["sampling"] == s and m["step"] == a["step"] for m in fa.values()) and all(m["sampling"] == ss for m in sm.values())
seed_ours = sorted({m["seed"] for m in fa.values()})
seed_vid = sorted({m["seed"] for m in st.values()})
slots = sorted({len(m["ref_slots_present"]) for m in st.values()}, reverse=True)
none_rep = s["text_rep"] == s["audio_rep_pen"] == s["pose_rep"] == 1.0 and s["freq_pen"] == 0.0 and ss.get("rep_pen") is None
T = [("checkpoint", f"PPM 32B pose FT, step {a['step']}", "SAM 8B Aloe pose FT, step 636", "9-sinks, non-distilled"),
     ("prefill", f"{25 / 12.5:.0f} s real pose + audio + words", f"{sm['S-05']['prefill_pose_frames'] / 12.5:.0f} s real pose + audio", "starting image + references"),
     ("length", f"{a['n_frames'] / 12.5:.0f} s", "60 s", "59.9 s"),
     ("text sampling", f"temp {s['text_temp']:g}, no top-k", "our generated words (fixed)", "—"),
     ("audio sampling", f"{s['cb0_temp']:g} / {s['cb0_topk']}", f"{ss['temp']:g} / {ss['topk']}", "—"),
     ("pose sampling", f"{s['pose_temp']:g} / {s['pose_topk']}", f"{ss['pose_temp']:g} / {ss['pose_topk']}", "—"),
     ("CFG", f"pose {s['pose_cfg_s']:g}, silent-speech uncond", f"pose {ss['pose_cfg']:g} (all 32), silent-speech uncond", "—"),
     ("rep-pen", "none" if none_rep else "?", "none" if ss.get("rep_pen") is None else "?", "—"),
     ("prompt", "voice clip, no persona", "text upfront", "—"),
     ("seed", f"{seed_ours[0]} (card 15: {seed_ours[-1]})", "card number", f"{seed_vid[0]} (card 15: {seed_vid[-1]})"),
     ("denoising steps", "—", "—", f"{4 * SET['stabiliser']['substeps']} / {SET['resolver']['steps']}"),
     ("audio guidance", "—", "—", f"{SET['audio_guidance']['scale']:g}"),
     ("reference images", "—", "—", f"{slots[0]} (card 20: {slots[-1]})")]
E = html.escape
VG = ' class="vg"'
rows = "".join(f'<tr{VG if i == 10 else ""}><th>{E(p)}</th><td>{E(x)}</td><td>{E(y)}</td><td>{E(z)}</td></tr>' for i, (p, x, y, z) in enumerate(T))
panel = (f'<details class="ip"><summary>Inference parameters</summary><table class="pt"><thead><tr><th>Parameter</th><th>Ours (full AR)</th>'
         f'<th>SAM 8B</th><th>Video model</th></tr></thead><tbody>{rows}</tbody></table>'
         f'<p class="fn">Sampling = temp / top-k. Denoising = stabiliser / resolver steps per frame.</p></details>')
css = (".ip{margin:8px 0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 12px}.ip summary{cursor:pointer;font-weight:600}"
       ".pt{border-collapse:collapse;font-size:13px;margin-top:8px;width:100%}.pt th,.pt td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:left}"
       ".pt thead th{color:var(--mu);font-weight:600}.pt tbody th{font-weight:600;white-space:nowrap}.pt tr.vg th,.pt tr.vg td{border-top:2px solid var(--line)}"
       ".fn{margin:6px 0 0;font-size:12px;color:var(--mu)}")
page = open(f"{K}/index.html").read()
page = re.sub(r'<details class="ip">.*?</details>\n?', "", page, flags=re.S)
page = re.sub(r"\.ip\{.*?(?=</style>)", "", page, flags=re.S)
page = page.replace("</style>", css + "</style>", 1)
i = page.index('<div class="tabs"')
page = page[:i] + panel + "\n" + page[i:]
open(f"{K}/index.html.tmp", "w").write(page)
os.replace(f"{K}/index.html.tmp", f"{K}/index.html")
print("PARAMS TABLE", json.dumps(T))
