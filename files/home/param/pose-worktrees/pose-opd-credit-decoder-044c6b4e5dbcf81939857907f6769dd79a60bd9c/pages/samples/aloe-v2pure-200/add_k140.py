"""IN-PLACE addition of the k140 variant (10-18: pose CFG 3, pose cb0 top-k 30, pose cb1-31 top-k 140; runs-gen-k140 / runs-force-k140) to
the served aloe-v2pure-200 samples page (klweb/ppm_aloe_v2pure_samples; never rebuilt from the builder, the page format unchanged):
  media     a/<row>-<kind>.mp3, v/<row>-<kind>.mp4, t/<row>-<kind>.mp4 (skeleton + own audio), kinds genk140 / forcek140
  data.json per clip two new versions + tiles; order all-gen k250, all-gen k140, text-forced k250, text-forced k140, Real; the k250 columns
            relabelled '... · CFG3 top-k 250' so the two settings read apart; meta.sampling_k140 + its seeds
  index.html 5-column tile grid (3 below 1060 px, 1 below 760 px), headcap + Motion metrics table from the new metrics.json (all 5 rows),
            params line with both pose top-k settings. Backs up data.json / index.html / files.json first (.bak-pre-k140).
   python add_k140.py <work root> <out dir> <metrics.json>"""
import json
import os
import shutil
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
import soundfile as sf

W, OUT, MJ = sys.argv[1:4]
NB = 2400
FF = imageio_ffmpeg.get_ffmpeg_exe()
M = json.load(open(MJ))
R, AGG = M["res"], M["agg"]
assert {"gen_k140", "force_k140"} <= set(R), list(R)
L250 = {"gen": "all streams · CFG3 top-k 250", "force": "text forced · CFG3 top-k 250"}
NEW = [("genk140", "gen_k140", "all streams · CFG3 top-k 140", "runs-gen-k140"), ("forcek140", "force_k140", "text forced · CFG3 top-k 140", "runs-force-k140")]
ORDER = ["gen", "genk140", "force", "forcek140", "gt"]


def peaks(y):                                                     # = build_page.py
    n = max(len(y) // NB * NB, NB)
    y2 = np.zeros(n, np.float32)
    y2[:min(n, len(y))] = y[:n]
    pk = np.abs(y2).reshape(NB, -1).max(1)
    return [round(float(v), 3) for v in pk / max(1e-6, float(pk.max()))]


for f in ("data.json", "index.html", "files.json"):
    if not os.path.isfile(f"{OUT}/{f}.bak-pre-k140"):
        shutil.copyfile(f"{OUT}/{f}", f"{OUT}/{f}.bak-pre-k140")
D = json.load(open(f"{OUT}/data.json"))
files = json.load(open(f"{OUT}/files.json"))
samp, seeds = None, set()
for c in D["convs"]:
    nm = c["name"]
    k = c["ck"]["v2pure"]
    for v in k["versions"]:                                       # relabel the k250 versions + tiles (tiles find versions by label)
        for kind, lab in L250.items():
            if v["label"] in ("all streams generated" if kind == "gen" else "text forced", lab):
                v["label"] = lab
    for t in c["tiles"]:
        if t["kind"] in L250:
            t["label"] = L250[t["kind"]]
    g = R["gt"]["tracks"][nm]["gen"]
    for kind, src, label, d in NEW:
        vd = f"{W}/{d}/{nm}"
        Mv, Wv = json.load(open(f"{vd}/meta.json")), json.load(open(f"{vd}/words.json"))
        assert os.path.isfile(f"{vd}/DONE") and Mv["row_key"] == c["row_key"], nm
        assert Mv["sampling"]["pose_topk"] == 30 and Mv["sampling"]["pose_light_topk"] == 140 and Mv["sampling"]["pose_cfg_s"] == 3.0, nm
        if kind == "forcek140":
            ch = Mv["text_force_check"]
            assert ch["lane_equals_real"] and ch["n_equal"] == ch["n_forced"], nm
        samp = samp or Mv["sampling"]
        assert samp == Mv["sampling"], nm
        seeds.add(Mv["seed"])
        x, sr = sf.read(f"{vd}/s1.wav", dtype="float32")
        fa, fv, ft = f"a/{nm}-{kind}.mp3", f"v/{nm}-{kind}.mp4", f"t/{nm}-{kind}.mp4"
        sf.write(f"{OUT}/{fa}.tmp.mp3", np.clip(x, -1, 1), sr, format="MP3", subtype="MPEG_LAYER_III")
        os.replace(f"{OUT}/{fa}.tmp.mp3", f"{OUT}/{fa}")
        shutil.copyfile(f"{vd}/pose_s1.mp4", f"{OUT}/{fv}.tmp")
        os.replace(f"{OUT}/{fv}.tmp", f"{OUT}/{fv}")
        subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-i", f"{vd}/pose_s1.mp4", "-i", f"{vd}/s1.wav", "-map", "0:v:0", "-map", "1:a:0",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", f"{OUT}/{ft}.tmp.mp4"], check=True)
        os.replace(f"{OUT}/{ft}.tmp.mp4", f"{OUT}/{ft}")
        for f in (fa, fv, ft):
            files[f] = f
        base = k["versions"][0]
        v = {"label": label, "dur": round(len(x) / sr, 3), "video": {"B": {"src": fv, "cap": base["video"]["B"]["cap"]}},
             "tracks": [{"ch": "B", "role": base["tracks"][0]["role"], "src": fa, "peaks": peaks(x),
                         "words": [[round(w["t"], 2), w["text"], bool(w["prompt"])] for w in Wv["s1"]]}],
             "sampling": Mv["sampling"]}
        k["versions"] = [u for u in k["versions"] if u["label"] != label] + [v]
        s = R[src]["tracks"][nm]["gen"]
        t = {"kind": kind, "label": label, "src": ft, "real": 0, "wrist_pct": round(100 * s["wrist_speed"] / g["wrist_speed"], 1),
             "gest": round(s["gestures_45s"], 1)}
        if kind == "forcek140":
            t["cut"] = "text forced (real words) · audio + pose generated"
        c["tiles"] = [u for u in c["tiles"] if u["kind"] != kind] + [t]
    c["tiles"].sort(key=lambda t: ORDER.index(t["kind"]))
    vo = {L250["gen"]: 0, NEW[0][2]: 1, L250["force"]: 2, NEW[1][2]: 3, "Real recording": 4}
    k["versions"].sort(key=lambda v: vo[v["label"]])
    assert [t["kind"] for t in c["tiles"]] == ORDER and [v["label"] for v in k["versions"]] == list(vo), nm
D["meta"]["sampling_k140"] = samp
D["meta"]["seeds_k140"] = sorted(seeds)
json.dump(D, open(f"{OUT}/data.json.tmp", "w"), separators=(",", ":"))
os.replace(f"{OUT}/data.json.tmp", f"{OUT}/data.json")
json.dump(files, open(f"{OUT}/files.json", "w"))

s = open(f"{OUT}/index.html").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:90]
    s = s.replace(a, b)


A = {k: AGG[k]["gen"] for k in ("gen", "gen_k140", "force", "force_k140", "gt")}
real = A["gen"]["gt_mean"]
f1 = lambda k: (f"wrist speed {A[k]['pct_gt']['wrist_speed']:.0f}% of real · {A[k]['mean']['gestures_45s']:.1f} gestures / 45 s · still "  # noqa: E731
                f"{100 * A[k]['mean']['still_frac']:.0f}% · r {A[k]['mean']['r_speed_energy']:.2f}")
cap = (f"Hand motion over the generated part (2 s to each clip's end), 10 clips vs the real recordings (real: {real['gestures_45s']:.1f} gestures / 45 s · "
       f"still {100 * real['still_frac']:.0f}% · hand–speech r {real['r_speed_energy']:.2f}) · all streams top-k 250: {f1('gen')} · all streams top-k 140: "
       f"{f1('gen_k140')} · text forced top-k 250: {f1('force')} · text forced top-k 140: {f1('force_k140')}")
i0 = s.index('<p class="headcap">')
i1 = s.index("</p>", i0)
s = s[:i0] + f'<p class="headcap">{cap}' + s[i1:]


def trow(k, label, ref=False):
    a = A[k]
    ci = "100% (reference)" if ref else f'{a["pct_gt"]["wrist_speed"]:.0f}% ({a["wrist_speed_pct_ci95"][0]:.0f}–{a["wrist_speed_pct_ci95"][1]:.0f})'
    cls = ' class="ref"' if ref else ""
    return (f'      <tr{cls}><th scope="row">{label}</th><td>{a["n"]}</td><td>{a["mean"]["wrist_speed"]:.4f}</td><td>{ci}</td>'
            f'<td>{a["mean"]["gestures_45s"]:.1f}</td><td>{a["mean"]["still_frac"]:.2f}</td><td>{a["mean"]["r_speed_energy"]:.2f}</td>'
            f'<td>{a["mean"]["hand_height_range"]:.2f}</td></tr>\n')


i0 = s.index("      <tbody>\n") + len("      <tbody>\n")
i1 = s.index("      </tbody></table></div>")
s = (s[:i0] + trow("gen", "All streams generated · CFG3 top-k 250") + trow("gen_k140", "All streams generated · CFG3 top-k 140")
     + trow("force", "Text forced (real words + timing) · CFG3 top-k 250") + trow("force_k140", "Text forced (real words + timing) · CFG3 top-k 140")
     + trow("gt", "Real recordings, same 10 clips", True) + s[i1:])
if ".tiles{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}\n" in s:
    rep(".tiles{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}\n",
        ".tiles{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px}\n@media (max-width:1060px){.tiles{grid-template-columns:repeat(3,minmax(0,1fr))}}\n")
if "(k250 columns)" not in s:
    rep("`pose temp ${s.pose_temp} cb0 top-k ${s.pose_topk} · cb1-31 temp ${s.pose_light_temp} top-k ${s.pose_light_topk} · `+",
        "`pose temp ${s.pose_temp} · top-k 250 columns: cb0 top-k ${s.pose_topk} · cb1-31 top-k ${s.pose_light_topk} (k250 columns) · top-k 140 columns: "
        "cb0 top-k ${(D.meta.sampling_k140||{}).pose_topk} · cb1-31 top-k ${(D.meta.sampling_k140||{}).pose_light_topk} · `+")
    rep("seed ${(D.meta.seeds||[]).join(', ')} (one per session)`",
        "seeds top-k 250 ${(D.meta.seeds||[]).join(', ')} · top-k 140 ${(D.meta.seeds_k140||[]).join(', ')} (one per session)`")
open(f"{OUT}/index.html.tmp", "w").write(s)
os.replace(f"{OUT}/index.html.tmp", f"{OUT}/index.html")
print("ok", len(D["convs"]), "clips")
print(cap)
