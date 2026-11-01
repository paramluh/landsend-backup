"""aloe-posemono-poseonly EVAL SET PAGE (10-19; EVAL5.txt, 4 rows) = klweb/ppm_aloe_posemono_poseonly in the format of klweb/ppm_aloe_v2pure_long
(its index.html copied with only titles / labels / captions / table rows / the hidden params line changed; build_page_long.py's data.json
schema). Per row (rows-np: voice clip only, no persona, 60 s) 2 side-by-side tiles, BOTH rendered by the non-distilled 9-sinks video model
(video/studio-posemono/eval4) with the studio-studio identity of the row's candidate card, driven by the real audio and:
  gen   generated pose, voice clip only, no persona (runs-poseonly: 2 s real pose prefill, then pose generated; words + audio forced)
  real  real pose (data)
NO stick figures anywhere: each tile's video (t.src) and the hidden main-panel video are the rendered mp4; no Pose / Real video toggle.
Per tile: the rendered video with audio, waveform peaks, the word lane (the real lane: forced; no ASR), metric line, Motion metrics modal
(analyze_eval5.py). System-prompt chip: B's voice clip (no persona text).
   python build_page_eval4.py <work root> <klweb out dir> <metrics.json>"""
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
FMT = "/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_v2pure_long"                # page format (index.html)
VM = "/data/home/amartyavarma/video/studio-posemono/eval4"
NB = 2400
FF = imageio_ffmpeg.get_ffmpeg_exe()
M = json.load(open(MJ))
R, AGG = M["res"], M["agg"]
rows = json.load(open(f"{W}/eval5/rows.json"))
STUDIO = ["03_b437950e", "05_e25c6859", "06_74807fe0", "09_d9ef3c94", "10_27286bdd", "12_72cb45d3", "13_956c09af", "15_d1e1322c",
          "18_1ffa9083", "19_47473a46"]
IDENT = {r["row"]: dict(prompt=STUDIO[(r["candidate"] - 1) % 10], category="studio-studio") for r in rows}
assert not os.path.exists(f"{OUT}/index.html"), "refusing to overwrite an existing page"
for d in ("a", "rv"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
GLAB = "generated pose, voice clip only, no persona"
KINDS = [("gen", GLAB, "", "poseonly"), ("real", "Real pose (data)", "gt_", "gt")]
ROLE = "B (speaker)"


def peaks(y):                                                     # = work/build_page.py
    n = max(len(y) // NB * NB, NB)
    y2 = np.zeros(n, np.float32)
    y2[:min(n, len(y))] = y[:n]
    pk = np.abs(y2).reshape(NB, -1).max(1)
    return [round(float(v), 3) for v in pk / max(1e-6, float(pk.max()))]


def media(job):                                                   # audio only (no skeleton media on this page)
    vd, pre, nm, kind = job
    fa = f"{OUT}/a/{nm}-{kind}.mp3"
    x, sr = sf.read(f"{vd}/{pre}s1.wav", dtype="float32")
    sf.write(f"{fa}.tmp.mp3", np.clip(x, -1, 1), sr, format="MP3", subtype="MPEG_LAYER_III")
    os.replace(f"{fa}.tmp.mp3", fa)
    return x, sr


def clip_mp3(nm):
    x, sr = sf.read(f"{W}/runs-poseonly/{nm}/clip_s1.wav", dtype="float32")
    f = f"{OUT}/a/{nm}-clip.mp3"
    sf.write(f"{f}.tmp.mp3", np.clip(x, -1, 1), sr, format="MP3", subtype="MPEG_LAYER_III")
    os.replace(f"{f}.tmp.mp3", f)
    return len(x) / sr


def faststart(src, dst):
    subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-i", src, "-c", "copy", "-movflags", "+faststart", f"{dst}.tmp.mp4"], check=True)
    os.replace(f"{dst}.tmp.mp4", dst)


files = {"data.json": "data.json", "index.html": "index.html"}
convs = []
jobs = [(f"{W}/runs-poseonly/{r['row']}", pre, r["row"], kind) for r in rows for kind, _, pre, _ in KINDS]
with ThreadPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))) as ex:
    AUD = dict(zip([(j[2], j[3]) for j in jobs], ex.map(media, jobs)))
    CLIPS = dict(zip([r["row"] for r in rows], ex.map(clip_mp3, [r["row"] for r in rows])))
SAMP, SEEDS, NRV = None, set(), 0
for r in rows:
    nm, n = r["row"], int(r["n"])
    mt = json.load(open(f"{W}/runs-poseonly/{nm}/meta.json"))
    wj = json.load(open(f"{W}/runs-poseonly/{nm}/words.json"))
    assert mt["row_key"] == r["row_key"] and mt["mode"] == "poseonly" and mt["n_frames"] == n and mt["step"] == 357, nm
    assert mt["force_check"]["lane_equals_real_positions"] == mt["force_check"]["audio_equals_real_positions"] == r["T"], (nm, mt["force_check"])
    assert mt["prompt"] == "voice clip only (no persona)" and r["with_persona"] is False, (nm, mt["prompt"])
    SAMP = SAMP or mt["sampling"]
    assert mt["sampling"] == SAMP, nm
    SEEDS.add(mt["seed"])
    idn = IDENT[nm]
    versions, tiles = [], []
    for kind, label, pre, sk in KINDS:
        x, sr = AUD[(nm, kind)]
        ws = wj["s1"]
        v = {"label": label, "dur": round(len(x) / sr, 3),
             "video": {"B": {"src": f"rv/{nm}-{kind}.mp4", "cap": "real pose" if kind == "real" else "real pose in the 2 s prefill, generated after"}},
             "tracks": [{"ch": "B", "role": ROLE, "src": f"a/{nm}-{kind}.mp3", "peaks": peaks(x),
                         "words": [[round(w["t"], 2), w["text"], bool(w["prompt"])] for w in ws]}]}
        if kind == "gen":
            v["sampling"] = mt["sampling"]
        versions.append(v)
        g = R["gt"]["tracks"][nm]["gen"]
        s = R[sk]["tracks"][nm]["gen"]
        t = {"kind": kind, "label": label, "src": f"rv/{nm}-{kind}.mp4", "real": int(kind == "real"), "gest": round(s["gestures_45s"], 1),
             "cut": "pose generated" if kind == "gen" else "real pose continues"}
        if kind == "real":
            t["wrist"] = round(s["wrist_speed"], 4)
        else:
            t["wrist_pct"] = round(100 * s["wrist_speed"] / g["wrist_speed"], 1)
        st = json.load(open(f"{VM}/done/{nm}-{kind}"))           # every tile must have its render (no fallback media)
        assert st["identity"] == idn["prompt"], (nm, kind, st)
        faststart(f"{VM}/out/{nm}-{kind}.mp4", f"{OUT}/rv/{nm}-{kind}.mp4")
        files[f"rv/{nm}-{kind}.mp4"] = f"rv/{nm}-{kind}.mp4"
        NRV += 1
        tiles.append(t)
        files[f"a/{nm}-{kind}.mp3"] = f"a/{nm}-{kind}.mp3"
    files[f"a/{nm}-clip.mp3"] = f"a/{nm}-clip.mp3"
    order = [{"kind": "clip", "ch": "B"}] + ([{"kind": "text", "ch": "B"}] if r["with_persona"] else [])
    cards = {"B": {"role": ROLE, "text": {"text": r["persona_text"]} if r["with_persona"] else None,
                   "clip": {"src": f"a/{nm}-clip.mp3", "seconds": round(CLIPS[nm], 2)}}}
    convs.append({"name": nm, "title": f"card {r['candidate']} · {nm} · {n / 12.5:.0f} s", "row_key": r["row_key"], "mode": "continue", "np_s": 2.0,
                  "ng_s": round((n - 25) / 12.5, 2), "order": order, "prompt": r["prompt"],
                  "ck": {"posemono": {"step": 357, "sampling": mt["sampling"], "versions": versions, "cards": cards, "score": None, "pose": 1}},
                  "stats": {"words": mt["words"]},
                  "window": f"held-out track {r['row_key']} ({r['side']}), first 60 s · video identity: studio prompt {idn['prompt']} ({idn['category']})",
                  "identity": idn, "timing": {"gen_wall_s": mt["gen_wall_s"], "wall_s": mt["wall_s"], "s_per_step": mt["speed_s_per_step"]},
                  "tiles": tiles})
D = {"meta": {"ckpts": [{"tag": "posemono", "step": 357, "label": "PPM 32B Aloe monoplex pose FT (aloe-pose-mono-8020, job 24272) · step 357"}],
              "sampling": SAMP, "seeds": sorted(SEEDS)}, "convs": convs}
json.dump(files, open(f"{OUT}/files.json", "w"))

# ---------------------------------------------------------------- index.html = the long page's current index.html with exact replacements
s = open(f"{FMT}/index.html").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:90]
    s = s.replace(a, b)


A = {k: AGG[k]["gen"] for k in ("poseonly", "gt")}
real = A["gt"]["gt_mean"]
f1 = lambda k: (f"wrist speed {A[k]['pct_gt']['wrist_speed']:.0f}% of real · {A[k]['mean']['gestures_45s']:.1f} gestures / 45 s · still "  # noqa: E731
                f"{100 * A[k]['mean']['still_frac']:.0f}% · r {A[k]['mean']['r_speed_energy']:.2f}")
cap = (f"Text + audio forced; pose generated; voice clip only, no persona. Hand motion over the generated part (2–60 s), the 4-row eval set "
       f"vs the real pose of the same windows (real: {real['gestures_45s']:.1f} gestures / 45 s · still {100 * real['still_frac']:.0f}% · "
       f"hand–speech r {real['r_speed_energy']:.2f}) · generated: {f1('poseonly')}. Both tiles of every row are rendered by the non-distilled "
       f"9-sinks video model with a studio identity (sink + reference bank of a studio recording), driven by the tile's pose and the real audio.")
rep("<title>PPM v2pure Long Clips</title>", "<title>PPM Pose Eval Set</title>")
i0 = s.index("<h1>")
i1 = s.index("</h1>", i0)
s = (s[:i0] + "<h1>PPM 32B Aloe monoplex pose fine-tune (aloe-pose-mono-8020) · step 357 · pose eval set (4 held-out Aloe tracks, 60 s) · text + "
     "audio forced; pose generated · voice clip only, no persona · pose CFG 3, pose top-k 30, temp 1" + s[i1:])
i0 = s.index('<p class="headcap">')
i1 = s.index("</p>", i0)
s = s[:i0] + f'<p class="headcap">{cap}' + s[i1:]
rep("(2 s to the clip's real end, 13–27 s)", "(2 s to the window's end: 58 s)")


def trow(k, label, ref=False):                                    # = build_page_long.py trow()
    a = A[k]
    ci = "100% (reference)" if ref else f'{a["pct_gt"]["wrist_speed"]:.0f}% ({a["wrist_speed_pct_ci95"][0]:.0f}–{a["wrist_speed_pct_ci95"][1]:.0f})'
    cls = ' class="ref"' if ref else ""
    return (f'      <tr{cls}><th scope="row">{label}</th><td>{a["n"]}</td><td>{a["mean"]["wrist_speed"]:.4f}</td><td>{ci}</td>'
            f'<td>{a["mean"]["gestures_45s"]:.1f}</td><td>{a["mean"]["still_frac"]:.2f}</td><td>{a["mean"]["r_speed_energy"]:.2f}</td>'
            f'<td>{a["mean"]["hand_height_range"]:.2f}</td></tr>\n')


i0 = s.index("      <tbody>\n") + len("      <tbody>\n")
i1 = s.index("      </tbody></table></div>")
s = (s[:i0] + trow("poseonly", "Generated pose, voice clip only, no persona · CFG 3 · pose top-k 30 (2–60 s)")
     + trow("gt", "Real pose, same 4 windows (2–60 s)", True) + s[i1:])
i0 = s.index("function params(){")
i1 = s.index("const HDR=")
P_JS = r"""function params(){const c=conv(),k=ckd(),s=D.meta.sampling||{};
  const lab=((D.meta.ckpts||[]).find(x=>x.tag===CK)||{}).label||('PPM step '+k.step);
  $('params').textContent=`${lab} · ${c.window||''} · system prompt: B's voice clip only, no persona text · one PPM instance · channel A empty the whole time (as in training) · `+
    `TEXT + AUDIO FORCED; POSE GENERATED: B's real words (lane) and real Mimi audio are teacher-forced for the whole window; the first 2.0 s of pose (frames 0–24) are real, then ${c.ng_s.toFixed(1)} s of pose are generated to the window end · `+
    `pose temp ${s.pose_temp} cb0 top-k ${s.pose_topk} · cb1-31 temp ${s.pose_light_temp} top-k ${s.pose_light_topk} · `+
    `pose CFG ${s.pose_cfg_on?s.pose_cfg_s:'off'} on all 32 pose codebooks (cb0 + cb1-31; uncond = the same input with every Mimi input replaced by digital-silence codes) · `+
    `no repetition / frequency penalty anywhere · bf16 · seed ${(D.meta.seeds||[]).join(', ')} (one per window group) · `+
    `video: the non-distilled 9-sinks model (pose-demo-website demo-v3, settings unchanged: seed 2026, stabiliser 40 steps / frame, resolver 10 steps, audio guidance 2.0), identity = the studio prompt's first frame + reference bank, driven by the tile's pose + the real audio`}
"""
s = s[:i0] + P_JS + s[i1:]
assert "k140" not in s and "v2pure" not in s.split("<script>")[0], "stale label left"
open(f"{OUT}/index.html.tmp", "w").write(s)
os.replace(f"{OUT}/index.html.tmp", f"{OUT}/index.html")
json.dump(D, open(f"{OUT}/data.json.tmp", "w"), separators=(",", ":"))
os.replace(f"{OUT}/data.json.tmp", f"{OUT}/data.json")
assert NRV == 2 * len(convs), (NRV, len(convs))
print("page:", len(convs), "rows,", NRV, "/", 2 * len(convs), "rendered tiles ->", OUT)
print(cap)
