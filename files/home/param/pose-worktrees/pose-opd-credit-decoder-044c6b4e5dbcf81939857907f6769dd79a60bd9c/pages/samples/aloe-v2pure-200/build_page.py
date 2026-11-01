"""aloe-v2pure-200 SAMPLES PAGE = the monoplex pose samples page format (klweb/ppm_aloe_pose_mono_cfg3k250: index.html + data.json, hand-edited;
copied, never a new layout) with 3 side-by-side versions per clip: 'all streams generated', 'text forced', 'Real recording' (reference).
Per clip and version: own skeleton video with own audio (t/), the page's hidden main-panel media (a/ mp3, v/ skeleton mp4), own waveform
peaks and own transcript (model word lane; text forced = the real lane), compact metric line (wrist speed % of real, gestures / 45 s), and the
Motion metrics modal with this run's numbers (analyze_v2pure.py metrics.json).
   python build_page.py <work root> <klweb out dir> <metrics.json>"""
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import imageio_ffmpeg
import numpy as np
import soundfile as sf

W, OUT, MJ = sys.argv[1:4]
SRCP = "/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_pose_mono_cfg3k250"
NB = 2400
FF = imageio_ffmpeg.get_ffmpeg_exe()
M = json.load(open(MJ))
R, AGG = M["res"], M["agg"]
rows = json.load(open(f"{W}/rows/rows.json"))
for d in ("a", "v", "t"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
KINDS = [("gen", "all streams generated", "runs-gen", ""), ("force", "text forced", "runs-force", ""), ("gt", "Real recording", "runs-gen", "gt_")]
ROLE = "B (active speaker)"


def peaks(y):                                                     # = build_page.py / add_nocfg_k140.py
    n = max(len(y) // NB * NB, NB)
    y2 = np.zeros(n, np.float32)
    y2[:min(n, len(y))] = y[:n]
    pk = np.abs(y2).reshape(NB, -1).max(1)
    return [round(float(v), 3) for v in pk / max(1e-6, float(pk.max()))]


def media(job):
    vd, pre, nm, kind = job
    fa, fv, ft = f"{OUT}/a/{nm}-{kind}.mp3", f"{OUT}/v/{nm}-{kind}.mp4", f"{OUT}/t/{nm}-{kind}.mp4"
    x, sr = sf.read(f"{vd}/{pre}s1.wav", dtype="float32")
    sf.write(f"{fa}.tmp.mp3", np.clip(x, -1, 1), sr, format="MP3", subtype="MPEG_LAYER_III")
    os.replace(f"{fa}.tmp.mp3", fa)
    shutil.copyfile(f"{vd}/{pre}pose_s1.mp4", f"{fv}.tmp")
    os.replace(f"{fv}.tmp", fv)
    subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-i", f"{vd}/{pre}pose_s1.mp4", "-i", f"{vd}/{pre}s1.wav", "-map", "0:v:0",
                    "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", f"{ft}.tmp.mp4"], check=True)
    os.replace(f"{ft}.tmp.mp4", ft)
    return x, sr


files = {"data.json": "data.json", "index.html": "index.html"}
convs = []
jobs = [(f"{W}/{d}/{r['row']}", pre, r["row"], kind) for r in rows for kind, _, d, pre in KINDS]
with ThreadPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))) as ex:
    AUD = dict(zip([(j[2], j[3]) for j in jobs], ex.map(media, jobs)))
samp = None
for r in rows:
    nm, n = r["row"], int(r["n"])
    mg, mf = (json.load(open(f"{W}/{d}/{nm}/meta.json")) for d in ("runs-gen", "runs-force"))
    assert mg["row_key"] == mf["row_key"] == r["row_key"] and mg["mode"] == "gen" and mf["mode"] == "force", nm
    assert mf["text_force_check"]["lane_equals_real"] and mf["text_force_check"]["n_equal"] == mf["text_force_check"]["n_forced"], nm
    assert mg["sampling"] == mf["sampling"], nm
    samp = samp or mg["sampling"]
    assert samp == mg["sampling"], nm
    wj = {d: json.load(open(f"{W}/{d}/{nm}/words.json")) for d in ("runs-gen", "runs-force")}
    versions, tiles = [], []
    for kind, label, d, pre in KINDS:
        x, sr = AUD[(nm, kind)]
        ws = wj[d]["gt"]["s1"] if kind == "gt" else wj[d]["s1"]
        if kind == "force":
            assert [w["text"] for w in ws] == [w["text"] for w in wj["runs-gen"]["gt"]["s1"]], nm
        v = {"label": label, "dur": round(len(x) / sr, 3),
             "video": {"B": {"src": f"v/{nm}-{kind}.mp4", "cap": "real pose" if kind == "gt" else "real pose in the 2 s prefill, generated after"}},
             "tracks": [{"ch": "B", "role": ROLE, "src": f"a/{nm}-{kind}.mp3", "peaks": peaks(x),
                         "words": [[round(w["t"], 2), w["text"], bool(w["prompt"])] for w in ws]}]}
        if kind != "gt":
            v["sampling"] = mg["sampling"]
        versions.append(v)
        g = R["gt"]["tracks"][nm]["gen"]
        s = R[kind]["tracks"][nm]["gen"]
        t = {"kind": kind, "label": label, "src": f"t/{nm}-{kind}.mp4", "real": int(kind == "gt"), "gest": round(s["gestures_45s"], 1)}
        if kind == "gt":
            t["wrist"] = round(s["wrist_speed"], 4)
        else:
            t["wrist_pct"] = round(100 * s["wrist_speed"] / g["wrist_speed"], 1)
        if kind == "force":
            t["cut"] = "text forced (real words) · audio + pose generated"
        tiles.append(t)
        for f in (f"a/{nm}-{kind}.mp3", f"v/{nm}-{kind}.mp4", f"t/{nm}-{kind}.mp4"):
            files[f] = f
    convs.append({"name": nm, "title": f"{nm} · {n / 12.5:.1f} s", "row_key": r["row_key"], "mode": "continue", "np_s": 2.0,
                  "ng_s": round((n - 25) / 12.5, 2), "order": [],
                  "ck": {"v2pure": {"step": 200, "sampling": mg["sampling"], "versions": versions,
                                    "cards": {"B": {"role": ROLE, "text": None, "clip": None}}, "score": None, "pose": 1}},
                  "stats": {"words": {"gen": mg["gen_words"], "force": mf["gen_words"]}},
                  "timing": {"gen_wall_s": mg["gen_wall_s"], "wall_s": mg["wall_s"], "s_per_step": mg["speed_s_per_step"]},
                  "tiles": tiles})
seed = sorted({json.load(open(f"{W}/runs-gen/{r['row']}/meta.json"))["seed"] for r in rows})
D = {"meta": {"ckpts": [{"tag": "v2pure", "step": 200, "label": "PPM 32B Aloe v2pure monoplex pose FT (job 23051) · step 200"}],
              "sampling": samp, "seeds": seed}, "convs": convs}
json.dump(D, open(f"{OUT}/data.json.tmp", "w"), separators=(",", ":"))
os.replace(f"{OUT}/data.json.tmp", f"{OUT}/data.json")
json.dump(files, open(f"{OUT}/files.json", "w"))

# ---------------------------------------------------------------- index.html = the existing page with exact replacements (each asserted once)
s = open(f"{SRCP}/index.html").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:90]
    s = s.replace(a, b)


A = {k: AGG[k]["gen"] for k in ("gen", "force", "gt")}
pct = lambda k: A[k]["pct_gt"]["wrist_speed"]  # noqa: E731
mn = lambda k, q: A[k]["mean"][q]  # noqa: E731
real = A["gen"]["gt_mean"]
cap = (f"Hand motion over the generated part (2 s to each clip's end), 10 clips vs the real recordings · all streams generated: wrist speed "
       f"{pct('gen'):.0f}% of real · {mn('gen', 'gestures_45s'):.1f} gestures per 45 s (real {real['gestures_45s']:.1f}) · hands still "
       f"{100 * mn('gen', 'still_frac'):.0f}% of the time (real {100 * real['still_frac']:.0f}%) · hand–speech correlation "
       f"{mn('gen', 'r_speed_energy'):.2f} (real {real['r_speed_energy']:.2f}) · text forced: {pct('force'):.0f}% · "
       f"{mn('force', 'gestures_45s'):.1f} gestures · still {100 * mn('force', 'still_frac'):.0f}% · r {mn('force', 'r_speed_energy'):.2f}")
rep("<title>PPM Mono Pose Hands</title>", "<title>PPM v2pure Pose Samples</title>")
rep("PPM Aloe monoplex pose fine-tune (channel A empty) · pose CFG 3 + pose top-k · held-out Aloe B tracks · 15 s real, then generated (fast engine)</h1>",
    "PPM 32B Aloe v2pure monoplex pose fine-tune (channel A empty) · step 200 · held-out Aloe clips · 2 s real, then generated to the clip's end "
    "(fast engine)</h1>")
i0 = s.index('<p class="headcap">')
i1 = s.index("</p>", i0)
s = s[:i0] + f'<p class="headcap">{cap}' + s[i1:]
rep("Everything is measured over the 45 s generated window\n    (15–60 s); a real recording is measured over the same window of the real pose.",
    "Everything is measured over each clip's generated part\n    (2 s to the clip's real end, 13–27 s); a real recording is measured over the same window of the real pose.")
rep("(shown as % of the real recordings of the same tracks; brackets = 95% bootstrap interval over tracks).",
    "(shown as % of the real recordings of the same clips; brackets = 95% bootstrap interval over clips).")
rep("counts as one gesture.</li>", "counts as one gesture; the count is scaled from the clip's window to 45 s.</li>")
rep('<th scope="col">Tracks</th>', '<th scope="col">Clips</th>')


def trow(k, label, ref=False):
    a = A[k]
    ci = "100% (reference)" if ref else f'{a["pct_gt"]["wrist_speed"]:.0f}% ({a["wrist_speed_pct_ci95"][0]:.0f}–{a["wrist_speed_pct_ci95"][1]:.0f})'
    cls = ' class="ref"' if ref else ""
    return (f'      <tr{cls}><th scope="row">{label}</th><td>{a["n"]}</td><td>{a["mean"]["wrist_speed"]:.4f}</td><td>{ci}</td>'
            f'<td>{a["mean"]["gestures_45s"]:.1f}</td><td>{a["mean"]["still_frac"]:.2f}</td><td>{a["mean"]["r_speed_energy"]:.2f}</td>'
            f'<td>{a["mean"]["hand_height_range"]:.2f}</td></tr>\n')


i0 = s.index("      <tbody>\n") + len("      <tbody>\n")
i1 = s.index("      </tbody></table></div>")
s = s[:i0] + trow("gen", "All streams generated") + trow("force", "Text forced (real words + timing)") + trow("gt", "Real recordings, same 10 clips", True) + s[i1:]
i0 = s.index('    <p class="mxn">The old defaults were only run on 8 of the tracks')
i1 = s.index("</p>", i0) + len("</p>\n")
s = s[:i0] + s[i1:]
rep(".tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}\n@media (max-width:1060px){.tiles{grid-template-columns:repeat(2,minmax(0,1fr))}}\n",
    ".tiles{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}\n")
rep("m.textContent=(t.real?'real continuation':'generated')+' →'", "m.textContent=(t.cut||(t.real?'real continuation':'generated'))+' →'")
# params line: every inference parameter (data.json meta.sampling), both modes
i0 = s.index("function params(){")
i1 = s.index("const HDR=")
P_JS = r"""function params(){const c=conv(),k=ckd(),s=D.meta.sampling||{};
  const lab=((D.meta.ckpts||[]).find(x=>x.tag===CK)||{}).label||('PPM step '+k.step);
  $('params').textContent=`${lab} · continue after ${c.np_s.toFixed(1)} s real · prefill = the clip's first 2.0 s of real audio + pose (frames 0–24) + its real words (the text lane runs CD = 2 frames ahead of the codes) · one PPM instance · channel A empty the whole time (as in training) · ${c.ng_s.toFixed(1)} s generated to the clip's real end · `+
    `all streams generated: B samples words + speech + pose; text factorised: PAD / NEW_WORD / WORD at temp ${s.text_temp}, then the word at temp ${s.text_temp} with min-p ${s.min_p} (${s.min_p_rel?'relative':'absolute'}) over word pieces, ${s.text_topk?'top-k '+s.text_topk:'no top-k'}, ${s.text_topp<1?'top-p '+s.text_topp:'no top-p'} · `+
    `text forced: B's text lane teacher-forced to the real words at their real timing (no text sampling), speech + pose sampled · `+
    `audio temp ${s.cb0_temp} cb0 top-k ${s.cb0_topk} · cb1-31 temp ${s.light_temp} top-k ${s.light_topk} · no audio CFG · `+
    `pose temp ${s.pose_temp} cb0 top-k ${s.pose_topk} · cb1-31 temp ${s.pose_light_temp} top-k ${s.pose_light_topk} · `+
    `pose CFG ${s.pose_cfg_on?s.pose_cfg_s:'off'} on all 32 pose codebooks (cb0 + cb1-31; uncond = the same input with every Mimi input replaced by digital-silence codes) · `+
    `no repetition / frequency penalty anywhere (text ${s.text_rep}, audio ${s.audio_rep_pen}, pose ${s.pose_rep}) · bf16 · seed ${(D.meta.seeds||[]).join(', ')} (one per session)`}
"""
s = s[:i0] + P_JS + s[i1:]
open(f"{OUT}/index.html.tmp", "w").write(s)
os.replace(f"{OUT}/index.html.tmp", f"{OUT}/index.html")
print("page:", len(convs), "clips ->", OUT)
print(cap)
