"""Adds a compact "Inference parameters" panel to klweb/ppm_aloe_posemono_poseonly/index.html. Every value is READ from the runs that made the
videos: runs-studio/<row>/meta.json (Full AR), rows-studio/rows.json (voice clip), runs-sam8b/<row>/meta.json (SAM 8B), video/demo-v3-1019
settings.yaml + sampler/weights.py (DiT), the DiT render logs ([loader] window line, [video] refs / seed / frames lines), studio / gt / sam8b
SAMPLE_DIR meta.json (sink, refs dir, seeds), ffprobe of the published videos (fps, resolution), stick_gen.py (stick figure).
   python add_params.py      (compute node; qwen-s2 venv)"""
import html
import json
import re
import subprocess

import yaml

W = "/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly"
K = "/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_posemono_poseonly"
V = "/data/home/amartyavarma/video/studio-posemono"
D3 = "/data/home/amartyavarma/video/demo-v3-1019"
FP = "/data/home/amartyavarma/tools/ffmpeg-static/ffmpeg-7.0.2-amd64-static/ffprobe"
ROWS = [(5, "S-05", "10_27286bdd"), (13, "S-13", "06_74807fe0"), (15, "S-15", "10_27286bdd"), (20, "S-20", "19_47473a46")]
E = html.escape
rs = {r["row"]: r for r in json.load(open(f"{W}/rows-studio/rows.json"))}
fa = {row: json.load(open(f"{W}/runs-studio/{row}/meta.json")) for _, row, _ in ROWS}
sm = {row: json.load(open(f"{W}/runs-sam8b-own/{row}/meta.json")) for _, row, _ in ROWS}
st = {row: json.load(open(f"{V}/studio/{row}/meta.json")) for _, row, _ in ROWS}
sa = {row: json.load(open(f"{V}/sam8b-own/{row}/meta.json")) for _, row, _ in ROWS}
gt = {idn: json.load(open(f"{V}/gt/{idn}/meta.json")) for idn in {i for _, _, i in ROWS}}
SET = yaml.safe_load(open(f"{D3}/settings.yaml"))["video"]
wsrc = open(f"{D3}/sampler/weights.py").read()
local = dict(re.findall(r"'([\w./-]+)': Path\('([^']+)'\)", wsrc))
log = open(f"{V}/logs/studio-render-S-05.log").read()
win = re.search(r"\[loader\] (window [^\n]+)", log).group(1)
vlines = {row: re.search(r"\[video\] [^\n]+", open(f"{V}/logs/studio-render-{row}.log").read()).group(0) for _, row, _ in ROWS}
probe = subprocess.run([FP, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate", "-of", "csv=p=0",
                        f"{K}/rv/S-05-gen-flash.mp4"], check=True, capture_output=True, text=True).stdout.strip()
a = fa["S-05"]
s = a["sampling"]
for m in fa.values():
    assert m["sampling"] == s and m["ckpt"] == a["ckpt"] and m["step"] == a["step"] and m["dtype"] == a["dtype"]
ss = sm["S-05"]["sampling"]
assert all(m["sampling"] == ss and m["ckpt"] == sm["S-05"]["ckpt"] for m in sm.values())
pf = 25
li = lambda k, v: f"<li><b>{E(k)}:</b> {E(str(v))}</li>"  # noqa: E731
clip = "; ".join(f"card {c}: {(rs[r]['clip']['frames'][1] - rs[r]['clip']['frames'][0]) / 12.5:.1f} s from {rs[r]['clip']['src_key'].split('/')[-1]} "
                 f"(same speaker, later chunk; trained voice_clips rule)" for c, r, _ in ROWS)
seeds = ", ".join(f"card {c}: {fa[r]['seed']}" for c, r, _ in ROWS)
full = [li("checkpoint", f"{a['ckpt']} (step {a['step']}; PPM 32B monoplex pose FT aloe-pose-mono-8020)"),
        li("prefill", f"{pf} frames = {pf / 12.5:.1f} s: the package's prefill_audio + prefill_pose; lane (words) frames 0-26 from our store row"),
        li("generation", f"frames {pf}-{a['n_frames'] - 1} ({(a['n_frames'] - pf) / 12.5:.1f} s): words, audio and pose all sampled, nothing forced"),
        li("text / words", f"temp {s['text_temp']}, top-k {s['text_topk'] or 'off'}, top-p {s['text_topp']}, min-p {s['min_p'] or 'off'}, lane grammar"),
        li("audio (32 codebooks)", f"temp {s['cb0_temp']} / {s['light_temp']}, top-k {s['cb0_topk']} / {s['light_topk']} (cb0 / cb1-31), no top-p, no CFG"),
        li("pose (32 codebooks)", f"temp {s['pose_temp']} / {s['pose_light_temp']}, top-k {s['pose_topk']} / {s['pose_light_topk']} (cb0 / cb1-31), no top-p"),
        li("CFG", f"pose only, scale {s['pose_cfg_s']} on cb0 {'+ cb1-31' if s['pose_cfg_light'] else ''}; unconditional = every Mimi input replaced by digital-silence codes"),
        li("rep-pen", f"none (text {s['text_rep']}, audio {s['audio_rep_pen']}, pose {s['pose_rep']}, freq {s['freq_pen']})"),
        li("voice clip", clip), li("persona", "none"), li("seeds", seeds + " (card 15: independent second sample)"), li("dtype", a["dtype"])]
sam = [li("checkpoint", f"{sm['S-05']['ckpt']} = {sm['S-05']['hf']}"),
       li("inputs", f"text = Full AR's generated words, first 60 s only, upfront (SAM has no word-timing input); prefill {sm['S-05']['prefill_pose_frames']} frames = the package's real audio + pose"),
       li("generation", "frames 25-749 (58 s): SAM's OWN speech (Mimi) and pose, END masked; rendered with SAM's audio"),
       li("sampling", f"audio temp {ss['temp']}, top-k {ss['topk']}; pose temp {ss['pose_temp']}, top-k {ss['pose_topk']}; no top-p"),
       li("CFG", f"pose, scale {ss['pose_cfg']} on {ss['scope']} 32 pose codebooks; unconditional = {ss['uncond']} (digital-silence Mimi); no audio CFG"),
       li("rep-pen", "none"), li("seeds", ", ".join(f"card {c}: {sm[r]['seed']}" for c, r, _ in ROWS)), li("dtype", "bf16")]
w = SET["weights"]
dit = [li("weights", f"non-distilled 9-sinks EMA {w['checkpoint']['repo']} {w['checkpoint']['file']} @ {w['checkpoint']['revision'][:8]} "
                     f"(local {local.get(w['checkpoint']['repo'])}); one checkpoint as stabiliser and resolver"),
       li("stabiliser", f"{4 * SET['stabiliser']['substeps']} steps / frame (4 bands x {SET['stabiliser']['substeps']}), commit at t {SET['stabiliser']['history_t']}"),
       li("resolver", f"renoise to t {SET['resolver']['t_start']}, {SET['resolver']['steps']} steps"),
       li("audio guidance", f"scale {SET['audio_guidance']['scale']} on each frame's first {SET['audio_guidance']['steps']} stabiliser steps"),
       li("seeds", ", ".join(f"card {c}: Full AR {st[r]['seed']}, SAM 8B {sa[r]['seed']}" for c, r, _ in ROWS) + "; ground truth " +
          ", ".join(f"{i}: {g['seed']}" for i, g in gt.items())),
       li("window", win), li("refs", "; ".join(f"card {c} ({i}): slots {st[r]['ref_slots_present']}" for c, r, i in ROWS) +
                              " (load_refs, rng [1234, crc32(ref_seed_key)], missing = pad)"),
       li("sink", "the package's sink_latent.safetensors (its chunk's first video latent)"),
       li("video", f"{probe.split(',')[0]} x {probe.split(',')[1]}, {probe.split(',')[2].split('/')[0]} fps; 375 latent frames = 59.88 s; no video prefill (from the sink)"),
       li("code", "pose-demo-website demo-v3 (video/demo-v3-1019), settings.yaml unchanged")]
stick = [li("pose", "Full AR's generated pose tokens (runs-studio pose_s1), decoded with the package's frame-0 state (global_state)"),
         li("renderer", "PPM pose pages' skeleton (samples_fast.render draw: infer_ppm POSE_EDGES / POSE_COL), 270 x 480, 25 fps; audio = Full AR's generated audio")]
panel = (f'<details class="ip"><summary>Inference parameters</summary><div class="ipg">'
         f'<div><h4>Full AR (ours)</h4><ul>{"".join(full)}</ul></div><div><h4>SAM 8B (own speech + pose)</h4><ul>{"".join(sam)}</ul></div>'
         f'<div><h4>Video model (DiT)</h4><ul>{"".join(dit)}</ul></div><div><h4>Stick figure</h4><ul>{"".join(stick)}</ul></div></div></details>')
css = (".ip{margin:8px 0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 12px}.ip summary{cursor:pointer;font-weight:600}"
       ".ipg{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:10px;margin-top:8px}.ip h4{margin:0 0 4px;font-size:13.5px}"
       ".ip ul{margin:0;padding-left:16px;font-size:12.5px;color:var(--mu)}.ip li{margin:2px 0;overflow-wrap:anywhere}.ip b{color:var(--fg);font-weight:600}")
page = open(f"{K}/index.html").read()
page = re.sub(r'<details class="ip">.*?</details>', "", page, flags=re.S)
if ".ip{" not in page:
    page = page.replace("</style>", css + "</style>", 1)
i = page.index('<div class="tabs"')
page = page[:i] + panel + "\n" + page[i:]
open(f"{K}/index.html.tmp", "w").write(page)
import os  # noqa: E402
os.replace(f"{K}/index.html.tmp", f"{K}/index.html")
print("PARAMS ADDED", vlines["S-05"][:160])
