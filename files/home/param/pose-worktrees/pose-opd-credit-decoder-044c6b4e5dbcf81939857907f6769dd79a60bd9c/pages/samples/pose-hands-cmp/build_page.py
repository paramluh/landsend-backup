"""pose-hands-cmp PAGE: the PPM Aloe pose samples page format (klweb/ppm_aloe_pose_samples: same CSS / rail / panel / params line / transport /
skeleton video figures, blob-URL loading) with per item three skeleton videos GT | 8B | 32B, a hand-speed trace under each, and a small
metrics table on top. Reads <work>/metrics.json (analyze.py) and the rendered media in <out>/v, <out>/a.
   python build_page.py <work root> <klweb out dir>"""
import json
import os
import sys
import time

W, OUT = sys.argv[1], sys.argv[2]
M = json.load(open(f"{W}/metrics.json"))
gt = {}
for fn in sorted(os.listdir(f"{W}/gt8b")):
    if fn.endswith(".json"):
        g = json.load(open(f"{W}/gt8b/{fn}"))
        gt[g["track"]] = g
MOD = M["models"]
AGG = M["agg"]
PT = M["per_track"]
COL8 = [k for k in ("8b", "8b_cfg3") if k in AGG]
COL32 = [k for k in ("duo", "mono", "ppm", "duo_cfg3", "duo_k250", "mono_cfg3", "mono_rep12", "mono_adapt") if k in AGG]


def r(x, n=2):
    return None if x is None else round(float(x), n)


items = []
for tr in sorted(gt):
    g = gt[tr]
    cols = {}
    for k in ["gt"] + COL8 + COL32:
        if tr not in PT.get(k, {}) or not os.path.isfile(f"{OUT}/v/{tr}-{k}.mp4"):
            continue
        p = PT[k][tr]["gen"]
        pg = PT["gt"][tr]["gen"]
        cols[k] = {"v": f"v/{tr}-{k}.mp4", "a": f"a/{tr}-{k}.mp3" if os.path.isfile(f"{OUT}/a/{tr}-{k}.mp3") else None,
                   "trace": None,
                   "m": {"ws": r(100 * p["wrist_speed"]), "pct": r(100 * p["wrist_speed"] / max(pg["wrist_speed"], 1e-9), 0),
                         "still": r(100 * p["still_frac"], 0), "gest": r(p["gestures_45s"], 0), "vis": r(100 * p["hand_visible_frac"], 0),
                         "cb0": PT[k][tr]["codes_gen"]["cb0_distinct"]}}
    if "duo" not in cols and "mono" not in cols:
        continue
    items.append({"name": tr, "title": f"{tr[:5]} · {'speaker 1' if g['spk'] == 's1' else 'speaker 2'}", "row_key": g["row_key"],
                  "n_words_gen": g["n_words_gen"], "cols": cols})
TR = json.load(open(f"{W}/traces.json"))
for it in items:
    for k, c in it["cols"].items():
        c["trace"] = TR[k][it["name"]]

ROWS = []
for k in ["gt", "8b", "duo", "mono", "ppm", "8b_cfg3", "duo_cfg3", "duo_k250", "mono_cfg3", "mono_rep12", "mono_adapt"]:
    if k not in AGG:
        continue
    a = AGG[k]["gen"]
    pre = AGG[k]["pre"]
    ROWS.append({"k": k, "label": MOD[k][1], "long": MOD[k][0], "group": MOD[k][2], "samp": MOD[k][3], "n": a["n"],
                 "ws": r(100 * a["wrist_speed"]), "ws_pct": r(a.get("wrist_speed_pct_gt"), 0), "ws_ci": [r(x, 0) for x in a.get("wrist_speed_pct_gt_ci", [None, None])],
                 "p90": r(100 * a["wrist_p90"]), "path": r(a["path_45s"], 1), "path_pct": r(a.get("path_45s_pct_gt"), 0),
                 "still": r(100 * a["still_frac"], 0), "still_gt": r(100 * a["still_frac_gt"], 0),
                 "gest": r(a["gestures_45s"], 1), "gest_pct": r(a.get("gestures_45s_pct_gt"), 0),
                 "hrange": r(a["hand_height_range"]), "hrange_pct": r(a.get("hand_height_range_pct_gt"), 0),
                 "pstd_pct": r(a.get("wrist_pos_std_pct_gt"), 0),
                 "band_pct": r(a.get("pow_gesture_pct_gt"), 0), "hf_frac": r(100 * a["pow_hf_frac"], 0),
                 "finger_pct": r(a.get("finger_speed_pct_gt"), 0), "elbow_pct": r(a.get("elbow_speed_pct_gt"), 0),
                 "vis": r(100 * a["hand_visible_frac"], 0),
                 "r_se": r(a.get("r_speed_energy")), "spk_ratio": r(a["speed_speaking"] / a["speed_silent"]) if a.get("speed_speaking") and a.get("speed_silent") else None,
                 "speak": r(100 * a["speaking_frac"], 0) if a.get("speaking_frac") is not None else None,
                 "cb0": r(a["cb0_distinct"], 0), "cb0_gt": r(a["cb0_distinct_gt"], 0), "ent": r(a["cb0_entropy_bits"], 1),
                 "rep": r(100 * a["cb0_repeat"], 0), "pre_pct": r(pre.get("wrist_speed_pct_gt"), 0)})
H = {x["k"]: x for x in ROWS}


def pct(k):
    return f"{H[k]['ws_pct']:.0f}%" if k in H and H[k]["ws_pct"] is not None else "–"


headline = (f"Generated 45 s, no CFG, same prefills: wrist speed vs ground truth · 8B {pct('8b')} · 32B pose FT {pct('duo')}"
            + (f" · 32B monoplex {pct('mono')}" if "mono" in H else "") + (f" · PPM main {pct('ppm')}" if "ppm" in H else ""))
data = {"meta": {"headline": headline, "rows": ROWS, "made": time.strftime("%Y-%m-%d %H:%M PT", time.localtime()),
                 "labels": {k: MOD[k][1] for k in MOD}, "long": {k: MOD[k][0] for k in MOD}, "samp": {k: MOD[k][3] for k in MOD},
                 "col8": COL8, "col32": COL32, "thr": M["thresholds"], "data_ref": M.get("data_ref")},
        "items": items}
os.makedirs(OUT, exist_ok=True)
json.dump(data, open(f"{OUT}/data.json.tmp", "w"))
os.replace(f"{OUT}/data.json.tmp", f"{OUT}/data.json")
tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "page.html")).read()
open(f"{OUT}/index.html.tmp", "w").write(tpl)
os.replace(f"{OUT}/index.html.tmp", f"{OUT}/index.html")
print("page:", len(items), "items,", len(ROWS), "table rows ->", OUT)
print(headline)
