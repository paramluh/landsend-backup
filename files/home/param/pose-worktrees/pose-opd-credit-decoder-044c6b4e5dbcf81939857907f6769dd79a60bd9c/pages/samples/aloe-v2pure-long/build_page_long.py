"""aloe-v2pure LONG SAMPLES PAGE (10-18) = a NEW page klweb/ppm_aloe_v2pure_long in EXACTLY the format of klweb/ppm_aloe_v2pure_k30 (its
current index.html copied with only titles / labels / captions / table rows / the hidden params line / the tile-grid column count changed;
the data.json schema of build_page_k30.py): per 120 s held-out window (rows/build_rows_long.py; 10-18 update: 2-minute clips) 2 side-by-side tiles:
  k140long  all streams · CFG3 top-k 140    runs-gen-k140 (2 s real prefill, then ~118 s generated with the k140 column's exact sampling)
  real      Real recording                  (the window's real codes; its rv is a distilled render of the real pose + Mimi like the generated
                                             tile's, from video/distilled-k30 prep_long.py + pworker, not copied)
Per window and tile: skeleton video with own audio (t/), hidden main-panel media (a/, v/), waveform peaks, transcript (the model's own word
lane; no ASR), metric line, Motion metrics modal from analyze_long.py over the whole generated window, plus the first-30 s vs last-30 s rows.
Both tiles' rv: distilled-k30/code/patch_page.py (VM_PAGE = this page) as renders land.
   python build_page_long.py <work root> <klweb out dir> <metrics.json>"""
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
FMT = "/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_v2pure_k30"                # page format (index.html)
NB = 2400
FF = imageio_ffmpeg.get_ffmpeg_exe()
M = json.load(open(MJ))
R, AGG = M["res"], M["agg"]
rows = json.load(open(f"{W}/rows/rows.json"))
assert not os.path.exists(f"{OUT}/index.html"), "refusing to overwrite an existing page"
for d in ("a", "v", "t", "rv"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
KINDS = [("k140long", "all streams · CFG3 top-k 140", "runs-gen-k140", ""),
         ("real", "Real recording", "runs-gen-k140", "gt_")]
SRCK = {"k140long": "k140long", "real": "gt"}                     # tile kind -> metrics source key
ROLE = "B (active speaker)"
K140 = json.load(open("/data/home/amartyavarma/ppm/samples/aloe-v2pure-200/runs-gen-k140/B-000/meta.json"))["sampling"]


def peaks(y):                                                     # = work/build_page.py
    n = max(len(y) // NB * NB, NB)
    y2 = np.zeros(n, np.float32)
    y2[:min(n, len(y))] = y[:n]
    pk = np.abs(y2).reshape(NB, -1).max(1)
    return [round(float(v), 3) for v in pk / max(1e-6, float(pk.max()))]


def media(job):                                                   # = work/build_page.py media()
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
SAMP, SEEDS = {}, {}
for r in rows:
    nm, n = r["row"], int(r["n"])
    MT = {kind: json.load(open(f"{W}/{d}/{nm}/meta.json")) for kind, _, d, _ in KINDS if kind != "real"}
    WJ = {kind: json.load(open(f"{W}/{d}/{nm}/words.json")) for kind, _, d, _ in KINDS if kind != "real"}
    for kind, mt in MT.items():
        assert mt["row_key"] == r["row_key"] and mt["mode"] == "gen" and mt["sampling"] == K140, (nm, kind)
        assert mt["n_frames"] == 1500, (nm, mt["n_frames"])
        SAMP.setdefault(kind, mt["sampling"])
        SEEDS.setdefault(kind, set()).add(mt["seed"])
    gtw = WJ["k140long"]["gt"]["s1"]
    versions, tiles = [], []
    for kind, label, d, pre in KINDS:
        x, sr = AUD[(nm, kind)]
        ws = gtw if kind == "real" else WJ[kind]["s1"]
        v = {"label": label, "dur": round(len(x) / sr, 3),
             "video": {"B": {"src": f"v/{nm}-{kind}.mp4", "cap": "real pose" if kind == "real" else "real pose in the 2 s prefill, generated after"}},
             "tracks": [{"ch": "B", "role": ROLE, "src": f"a/{nm}-{kind}.mp3", "peaks": peaks(x),
                         "words": [[round(w["t"], 2), w["text"], bool(w["prompt"])] for w in ws]}]}
        if kind != "real":
            v["sampling"] = MT[kind]["sampling"]
        versions.append(v)
        g = R["gt"]["tracks"][nm]["gen"]
        s = R[SRCK[kind]]["tracks"][nm]["gen"]
        t = {"kind": kind, "label": label, "src": f"t/{nm}-{kind}.mp4", "real": int(kind == "real"), "gest": round(s["gestures_45s"], 1)}
        if kind == "real":
            t["wrist"] = round(s["wrist_speed"], 4)
        else:
            t["wrist_pct"] = round(100 * s["wrist_speed"] / g["wrist_speed"], 1)
        tiles.append(t)
        for f in (f"a/{nm}-{kind}.mp3", f"v/{nm}-{kind}.mp4", f"t/{nm}-{kind}.mp4"):
            files[f] = f
    mg = MT["k140long"]
    convs.append({"name": nm, "title": f"{nm} · {n / 12.5:.1f} s", "row_key": r["row_key"], "mode": "continue", "np_s": 2.0,
                  "ng_s": round((n - 25) / 12.5, 2), "order": [],
                  "ck": {"v2pure": {"step": 200, "sampling": mg["sampling"], "versions": versions,
                                    "cards": {"B": {"role": ROLE, "text": None, "clip": None}}, "score": None, "pose": 1}},
                  "stats": {"words": {k: MT[k]["gen_words"] for k in MT}}, "window": r["cache_key"] + f" frames 0-{int(r['n']) - 1} of {r['n_seg']} (contiguous real segment)",
                  "timing": {"gen_wall_s": mg["gen_wall_s"], "wall_s": mg["wall_s"], "s_per_step": mg["speed_s_per_step"]},
                  "tiles": tiles})
D = {"meta": {"ckpts": [{"tag": "v2pure", "step": 200, "label": "PPM 32B Aloe v2pure monoplex pose FT (job 23051) · step 200"}],
              "sampling": SAMP["k140long"], "seeds": sorted(SEEDS["k140long"])}, "convs": convs}
json.dump(files, open(f"{OUT}/files.json", "w"))

# ---------------------------------------------------------------- index.html = the old page's current index.html with exact replacements
s = open(f"{FMT}/index.html").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:90]
    s = s.replace(a, b)


A = {k: AGG[k]["gen"] for k in ("k140long", "gt")}
E = {k: AGG[k]["early"] for k in ("k140long", "gt")}
Lt = {k: AGG[k]["late"] for k in ("k140long", "gt")}
real = A["gt"]["gt_mean"]
f1 = lambda X, k: (f"wrist speed {X[k]['pct_gt']['wrist_speed']:.0f}% of real · {X[k]['mean']['gestures_45s']:.1f} gestures / 45 s · still "  # noqa: E731
                   f"{100 * X[k]['mean']['still_frac']:.0f}% · r {X[k]['mean']['r_speed_energy']:.2f}")
cap = (f"Hand motion over the generated part (2 s to 120 s), 8 held-out 120 s windows (each one contiguous real segment) vs the real recordings (real: {real['gestures_45s']:.1f} gestures / 45 s · "
       f"still {100 * real['still_frac']:.0f}% · hand–speech r {real['r_speed_energy']:.2f}) · CFG3 top-k 140, whole window: {f1(A, 'k140long')} · first 30 s "
       f"(2–32 s): {f1(E, 'k140long')} (real still {100 * E['gt']['mean']['still_frac']:.0f}%) · last 30 s (90–120 s): {f1(Lt, 'k140long')} (real still "
       f"{100 * Lt['gt']['mean']['still_frac']:.0f}%)")
rep("<title>PPM v2pure Top-k 30</title>", "<title>PPM v2pure Long Clips</title>")
i0 = s.index("<h1>")
i1 = s.index("</h1>", i0)
s = (s[:i0] + "<h1>PPM 32B Aloe v2pure monoplex pose fine-tune (channel A empty) · step 200 · held-out Aloe recordings · 2 s real, then generated to the "
     "window's end (fast engine) · 2-minute held-out windows · all streams generated · pose CFG 3, pose top-k 140" + s[i1:])
i0 = s.index('<p class="headcap">')
i1 = s.index("</p>", i0)
s = s[:i0] + f'<p class="headcap">{cap}' + s[i1:]


def trow(k, label, ref=False):                                    # = work/add_k140.py trow()
    a = A[k]
    ci = "100% (reference)" if ref else f'{a["pct_gt"]["wrist_speed"]:.0f}% ({a["wrist_speed_pct_ci95"][0]:.0f}–{a["wrist_speed_pct_ci95"][1]:.0f})'
    cls = ' class="ref"' if ref else ""
    return (f'      <tr{cls}><th scope="row">{label}</th><td>{a["n"]}</td><td>{a["mean"]["wrist_speed"]:.4f}</td><td>{ci}</td>'
            f'<td>{a["mean"]["gestures_45s"]:.1f}</td><td>{a["mean"]["still_frac"]:.2f}</td><td>{a["mean"]["r_speed_energy"]:.2f}</td>'
            f'<td>{a["mean"]["hand_height_range"]:.2f}</td></tr>\n')


i0 = s.index("      <tbody>\n") + len("      <tbody>\n")
i1 = s.index("      </tbody></table></div>")
A2 = dict(A)
A2["early"], A2["late"], A2["gt_early"], A2["gt_late"] = E["k140long"], Lt["k140long"], E["gt"], Lt["gt"]
A = A2
s = (s[:i0] + trow("k140long", "All streams generated · CFG3 top-k 140 · whole window (2–120 s)")
     + trow("early", "… first 30 s of the generated part (2–32 s)") + trow("late", "… last 30 s (90–120 s)")
     + trow("gt", "Real recordings, same 8 windows (2–120 s)", True) + trow("gt_early", "… real, 2–32 s", True) + trow("gt_late", "… real, 90–120 s", True)
     + s[i1:])
rep(".tiles{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px}\n@media (max-width:1060px){.tiles{grid-template-columns:repeat(3,minmax(0,1fr))}}\n",
    ".tiles{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}\n")
i0 = s.index("function params(){")
i1 = s.index("const HDR=")
P_JS = r"""function params(){const c=conv(),k=ckd(),s=D.meta.sampling||{};
  const lab=((D.meta.ckpts||[]).find(x=>x.tag===CK)||{}).label||('PPM step '+k.step);
  $('params').textContent=`${lab} · ${c.window||''} · continue after ${c.np_s.toFixed(1)} s real · prefill = the window's first 2.0 s of real audio + pose (frames 0–24) + its real words (the text lane runs CD = 2 frames ahead of the codes) · one PPM instance · channel A empty the whole time (as in training) · ${c.ng_s.toFixed(1)} s generated to the window end (120 s; the real recording runs the full window) · `+
    `all streams generated (sampling = the 'all streams · CFG3 top-k 140' column of ppm_aloe_v2pure_samples, unchanged): text factorised: PAD / NEW_WORD / WORD at temp ${s.text_temp}, then the word at temp ${s.text_temp} with min-p ${s.min_p} (${s.min_p_rel?'relative':'absolute'}) over word pieces, ${s.text_topk?'top-k '+s.text_topk:'no top-k'}, ${s.text_topp<1?'top-p '+s.text_topp:'no top-p'} · `+
    `audio temp ${s.cb0_temp} cb0 top-k ${s.cb0_topk} · cb1-31 temp ${s.light_temp} top-k ${s.light_topk} · no audio CFG · `+
    `pose temp ${s.pose_temp} cb0 top-k ${s.pose_topk} · cb1-31 temp ${s.pose_light_temp} top-k ${s.pose_light_topk} · `+
    `pose CFG ${s.pose_cfg_on?s.pose_cfg_s:'off'} on all 32 pose codebooks (cb0 + cb1-31; uncond = the same input with every Mimi input replaced by digital-silence codes) · `+
    `no repetition / frequency penalty anywhere (text ${s.text_rep}, audio ${s.audio_rep_pen}, pose ${s.pose_rep}) · bf16 · seed ${(D.meta.seeds||[]).join(', ')} (one per pair of windows)`}
"""
s = s[:i0] + P_JS + s[i1:]
assert "freq-pen" not in s and "pure temp" not in s and "top-p 0.95" not in s and "text forced" not in s.lower(), "stale label left"
open(f"{OUT}/index.html.tmp", "w").write(s)
os.replace(f"{OUT}/index.html.tmp", f"{OUT}/index.html")
json.dump(D, open(f"{OUT}/data.json.tmp", "w"), separators=(",", ":"))
os.replace(f"{OUT}/data.json.tmp", f"{OUT}/data.json")         # written last: the rv updater's go signal
print("page:", len(convs), "clips ->", OUT)
print(cap)
