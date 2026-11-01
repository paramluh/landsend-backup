"""aloe-8b-vs-32b PAGE = the existing samples page format (klweb/ppm_aloe_v2pure_samples: index.html + data.json, copied with exact asserted
replacements, never a new layout): per clip 4 side-by-side skeleton tiles, all rendered by the 8B SAM page's renderer (pose_infer.render:
confidence > 0.25, same colours / 2 px lines / 25 fps):
  8B SAM v2pure step 636 · CFG 3 · pose top-k 30 · full text upfront (the 8B page's own gen.mp4 = its pk30 column, without the burned-in flash;
  the page itself flashes + beeps at 2 s on every tile) | 32B v2pure step 200 · text forced | 32B · all streams generated | real recording
  (the 8B page's own ground-truth render). Clips = the 8B page's 8 held-out clips. Per tile: wrist speed % of real, gestures / 45 s, still %
  (analyze.py, the same maths for every tile).
   python build_page.py <work root> <klweb out dir>"""
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import imageio_ffmpeg
import numpy as np
import soundfile as sf

W, OUT = sys.argv[1:3]
SRCP = "/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_v2pure_samples"
S8 = "/data/home/amartyavarma/sam-aloe-ft-v2pure/samples"
O8 = f"{S8}/out/1014-215517"
NB = 2400
FF = imageio_ffmpeg.get_ffmpeg_exe()
M = json.load(open(f"{W}/ana/metrics.json"))
R, AGG = M["res"], M["agg"]
rows = json.load(open(f"{W}/rows/rows.json"))
assert not os.path.exists(f"{OUT}/index.html") or os.path.exists(f"{OUT}/.built_by_aloe_8b_vs_32b"), "refusing to overwrite a page not built here"
for d in ("a", "v", "t"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
open(f"{OUT}/.built_by_aloe_8b_vs_32b", "w").write("ppm/samples/aloe-8b-vs-32b/work/build_page.py\n")
ROLE = "B (active speaker)"
KINDS = [("b8", "8B · CFG 3 · pose top-k 30 · text upfront"), ("force", "32B · text forced"), ("gen", "32B · all streams generated"),
         ("gt", "Real recording")]


def peaks(y):
    n = max(len(y) // NB * NB, NB)
    y2 = np.zeros(n, np.float32)
    y2[:min(n, len(y))] = y[:n]
    pk = np.abs(y2).reshape(NB, -1).max(1)
    return [round(float(v), 3) for v in pk / max(1e-6, float(pk.max()))]


def srcs(r, kind):
    """-> (wav, skeleton mp4 with audio) of one tile."""
    tr, idx = r["row"], r["idx"]
    if kind == "b8":
        return f"{O8}/{idx}-pk30/gen.wav", f"{O8}/{idx}-pk30/gen.mp4"
    if kind == "gt":
        return f"{W}/runs-gen-k140/{tr}/gt_s1.wav", f"{S8}/inputs/{idx}-gt.mp4"
    return f"{W}/runs-{kind}-k140/{tr}/s1.wav", f"{W}/render/{kind}/{tr}.mp4"


def media(job):
    r, kind = job
    nm = r["row"]
    wav, mp4 = srcs(r, kind)
    x, sr = sf.read(wav, dtype="float32")
    fa = f"{OUT}/a/{nm}-{kind}.mp3"
    sf.write(f"{fa}.tmp.mp3", np.clip(x, -1, 1), sr, format="MP3", subtype="MPEG_LAYER_III")
    os.replace(f"{fa}.tmp.mp3", fa)
    for d in ("v", "t"):
        subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-i", mp4, "-c", "copy", "-movflags", "+faststart",
                        f"{OUT}/{d}/{nm}-{kind}.tmp.mp4"], check=True)
        os.replace(f"{OUT}/{d}/{nm}-{kind}.tmp.mp4", f"{OUT}/{d}/{nm}-{kind}.mp4")
    return x, sr


files = {"data.json": "data.json", "index.html": "index.html"}
jobs = [(r, kind) for r in rows for kind, _ in KINDS]
with ThreadPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))) as ex:
    AUD = dict(zip([(j[0]["row"], j[1]) for j in jobs], ex.map(media, jobs)))
samp = None
convs = []
for r in rows:
    nm, n, idx = r["row"], int(r["n"]), r["idx"]
    mg, mf = (json.load(open(f"{W}/runs-{k}-k140/{nm}/meta.json")) for k in ("gen", "force"))
    assert mg["row_key"] == mf["row_key"] == r["row_key"] and mg["mode"] == "gen" and mf["mode"] == "force", nm
    assert mf["text_force_check"]["lane_equals_real"], nm
    assert mg["sampling"] == mf["sampling"] and mg["sampling"]["pose_light_topk"] == 140 and mg["sampling"]["pose_topk"] == 30 and mg["sampling"]["pose_cfg_s"] == 3.0
    samp = samp or mg["sampling"]
    m8 = json.load(open(f"{O8}/{idx}-pk30/meta.json"))
    wj = {k: json.load(open(f"{W}/runs-{k}-k140/{nm}/words.json")) for k in ("gen", "force")}
    words8 = [w for w in json.load(open(f"{S8}/inputs/{idx}.json"))["words"] if w.strip()]
    versions, tiles = [], []
    for kind, label in KINDS:
        x, sr = AUD[(nm, kind)]
        if kind == "b8":
            ws = [[0.0, w, True] for w in words8]                 # full transcript upfront (no word lane / timing)
        else:
            src = wj["gen"]["gt"]["s1"] if kind == "gt" else wj[kind]["s1"]
            ws = [[round(w["t"], 2), w["text"], bool(w["prompt"])] for w in src]
        cap = {"gt": "real pose", "b8": "real pose in the 2 s prefill, generated after (full text upfront)"}.get(
            kind, "real pose in the 2 s prefill, generated after")
        v = {"label": label, "dur": round(len(x) / sr, 3), "video": {"B": {"src": f"v/{nm}-{kind}.mp4", "cap": cap}},
             "tracks": [{"ch": "B", "role": ROLE, "src": f"a/{nm}-{kind}.mp3", "peaks": peaks(x), "words": ws}]}
        versions.append(v)
        g = R["gt"]["tracks"][nm]["gen"]
        s = R[kind]["tracks"][nm]["gen"]
        t = {"kind": kind, "label": label, "src": f"t/{nm}-{kind}.mp4", "real": int(kind == "gt"), "gest": round(s["gestures_45s"], 1),
             "still": round(100 * s["still_frac"])}
        if kind == "gt":
            t["wrist"] = round(s["wrist_speed"], 4)
        else:
            t["wrist_pct"] = round(100 * s["wrist_speed"] / g["wrist_speed"], 1)
        t["cut"] = {"force": "text forced (real words) · audio + pose generated", "gen": "generated",
                    "b8": "generated", "gt": "real continuation"}[kind]
        if kind == "b8":
            t["end"] = f"{m8['total_s']:.1f} s ({'END' if m8['ended_with_END'] else '25 s cap'})"
        tiles.append(t)
        for f in (f"a/{nm}-{kind}.mp3", f"v/{nm}-{kind}.mp4", f"t/{nm}-{kind}.mp4"):
            files[f] = f
    convs.append({"name": nm, "title": f"{idx} · {n / 12.5:.1f} s", "row_key": r["row_key"], "key": r["key"], "mode": "continue", "np_s": 2.0,
                  "ng_s": round((n - 25) / 12.5, 2), "order": [],
                  "ck": {"cmp": {"step": 200, "sampling": mg["sampling"], "versions": versions,
                                 "cards": {"B": {"role": ROLE, "text": None, "clip": None}}, "score": None, "pose": 1}},
                  "stats": {"words": {"gen": mg["gen_words"], "force": mf["gen_words"]}}, "tiles": tiles})
seeds = sorted({json.load(open(f"{W}/runs-gen-k140/{r['row']}/meta.json"))["seed"] for r in rows})
D = {"meta": {"ckpts": [{"tag": "cmp", "step": 200, "label": "8B SAM v2pure step 636 vs PPM 32B v2pure step 200 (job 23051)"}],
              "sampling": samp, "seeds": seeds}, "convs": convs}
json.dump(D, open(f"{OUT}/data.json.tmp", "w"), separators=(",", ":"))
os.replace(f"{OUT}/data.json.tmp", f"{OUT}/data.json")
json.dump(files, open(f"{OUT}/files.json", "w"))

# ---------------------------------------------------------------- index.html = the existing page with exact replacements (each asserted once)
s = open(f"{SRCP}/index.html").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:90]
    s = s.replace(a, b)


A = {k: AGG[k]["gen"] for k in ("b8", "force", "gen", "gt")}
real = A["gen"]["gt_mean"]
NMS = {"b8": "8B (CFG 3, top-k 30, text upfront)", "force": "32B text forced", "gen": "32B all streams"}
cap = ("Hand motion over the generated part (2 s to each tile's end), the 8B page's 8 held-out clips, every tile rendered by the 8B page's "
       f"skeleton renderer · real: {real['gestures_45s']:.1f} gestures / 45 s · still {100 * real['still_frac']:.0f}% · "
       + " · ".join(f"{NMS[k]}: wrist speed {A[k]['pct_gt']['wrist_speed']:.0f}% of real · {A[k]['mean']['gestures_45s']:.1f} gestures / 45 s · "
                    f"still {100 * A[k]['mean']['still_frac']:.0f}%" for k in ("b8", "force", "gen")))
rep("<title>PPM v2pure Pose Samples</title>", "<title>8B vs 32B Pose</title>")
i0 = s.index("  <h1>")
i1 = s.index("</h1>", i0)
s = s[:i0] + ("  <h1>Aloe v2pure pose · 8B SAM (step 636) vs PPM 32B (step 200) · the 8B page's 8 held-out clips · 2 s real, then generated · "
              "all tiles rendered by the 8B page's skeleton renderer") + s[i1:]
i0 = s.index('<p class="headcap">')
i1 = s.index("</p>", i0)
s = s[:i0] + f'<p class="headcap">{cap}' + s[i1:]
rep("Everything is measured over each clip's generated part\n    (2 s to the clip's real end, 13–27 s); a real recording is measured over the same window of the real pose.",
    "Everything is measured over each tile's generated part\n    (2 s to the tile's end: the clip's real end, 24–25.5 s, for the 32B and real tiles; END or the 25 s cap for the 8B tile); "
    "a real recording is measured over the same window of the real pose. All four columns go through the same decode and the same maths.")


def trow(k, label, ref=False):
    a = A[k]
    ci = "100% (reference)" if ref else f'{a["pct_gt"]["wrist_speed"]:.0f}% ({a["wrist_speed_pct_ci95"][0]:.0f}–{a["wrist_speed_pct_ci95"][1]:.0f})'
    cls = ' class="ref"' if ref else ""
    return (f'      <tr{cls}><th scope="row">{label}</th><td>{a["n"]}</td><td>{a["mean"]["wrist_speed"]:.4f}</td><td>{ci}</td>'
            f'<td>{a["mean"]["gestures_45s"]:.1f}</td><td>{a["mean"]["still_frac"]:.2f}</td><td>{a["mean"]["r_speed_energy"]:.2f}</td>'
            f'<td>{a["mean"]["hand_height_range"]:.2f}</td></tr>\n')


i0 = s.index("      <tbody>\n") + len("      <tbody>\n")
i1 = s.index("      </tbody></table></div>")
s = s[:i0] + (trow("b8", "8B SAM v2pure · CFG 3 · pose top-k 30 · full text upfront") + trow("force", "32B · text forced (real words + timing) · CFG 3 · top-k 30 / 140")
              + trow("gen", "32B · all streams generated · CFG 3 · top-k 30 / 140") + trow("gt", "Real recordings, same 8 clips", True)) + s[i1:]
rep(".tiles{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px}\n@media (max-width:1060px){.tiles{grid-template-columns:repeat(3,minmax(0,1fr))}}",
    ".tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}\n@media (max-width:1060px){.tiles{grid-template-columns:repeat(2,minmax(0,1fr))}}")
rep("${t.real?`reference · wrist ${t.wrist.toFixed(4)} SW/frame · ${t.gest.toFixed(1)} gestures / 45 s`:`wrist ${t.wrist_pct.toFixed(0)}% of real · ${t.gest.toFixed(1)} gestures / 45 s`}",
    "${t.real?`reference · wrist ${t.wrist.toFixed(4)} SW/frame · ${t.gest.toFixed(1)} gestures / 45 s · still ${t.still}%`:`wrist ${t.wrist_pct.toFixed(0)}% of real · ${t.gest.toFixed(1)} gestures / 45 s · still ${t.still}%`}${t.end?` · ends ${t.end}`:''}")
i0 = s.index("function params(){")
i1 = s.index("const HDR=")
P_JS = r"""function params(){const c=conv(),s=D.meta.sampling||{};
  $('params').textContent=`${c.key} · continue after ${c.np_s.toFixed(1)} s real (the clip's first 2.0 s of real audio + pose) · `+
    `8B: SAM 8B v2pure step 636 (the 8B samples page's own CFG 3 / pose top-k 30 column): full transcript upfront, then generated until END or the 25 s cap; temp 1 audio + pose, audio top-k 30, pose top-k 30 on all 32 codebooks, pose CFG 3 (uncond = Mimi silence codes), seed 0 · `+
    `32B: PPM 32B v2pure step 200, fast engine, channel A empty, B's system block empty (no voice clip), generated to the clip's real end; text forced = B's text lane teacher-forced to the real words at their real timing; all streams = B samples words (factorised, word min-p ${s.min_p}) + speech + pose · `+
    `audio temp ${s.cb0_temp} top-k ${s.cb0_topk} (cb0 + cb1-31) · pose temp ${s.pose_temp} cb0 top-k ${s.pose_topk} · cb1-31 top-k ${s.pose_light_topk} · pose CFG ${s.pose_cfg_s} on all 32 pose codebooks (uncond = every Mimi input -> digital-silence codes) · no repetition penalty · bf16 · seeds ${(D.meta.seeds||[]).join(', ')} · `+
    `every tile rendered with the 8B page's pose_infer.render (bones only where both joints have confidence > 0.25, 2 px, 25 fps)`}
"""
s = s[:i0] + P_JS + s[i1:]
open(f"{OUT}/index.html.tmp", "w").write(s)
os.replace(f"{OUT}/index.html.tmp", f"{OUT}/index.html")
print("page:", len(convs), "clips ->", OUT)
print(cap)
