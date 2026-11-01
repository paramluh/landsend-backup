"""Eye-metric analysis of eyes_series.json (+ eye-band NCC to the clip's GT 0-2 s eyes) -> eyes_results.json. Node CPU.
Per frame, ref = the clip's GT render 0-2 s (identical inputs in all conditions there):
  open_rel = dark-core / ref median (iris+pupil visible; < 0.7 = lids lowered / eyes closed)
  iy_rel   = vertical iris centroid - ref median, / eye-window width (+ = down)
  oncam_geo = |ix - ref| < 0.08 & |iy - ref| < 0.08 & open_rel >= 0.7
  eye_ncc  = NCC(eye band crop, mean GT 0-2 s eye band); oncam_ncc = eye_ncc >= 10th pct of the GT 0-2 s frames' own NCC
   python eyes_an.py NPROC"""
import json
import sys
from multiprocessing import Pool

import cv2
import numpy as np

P = "/data/home/amartyavarma/ppm/gaze-drift/path"
UNSEEN = {"19_47473a46", "15_d1e1322c", "18_1ffa9083"}
ROWS = dict([("S-05", "10_27286bdd"), ("S-13", "06_74807fe0"), ("S-15", "10_27286bdd"), ("S-20", "19_47473a46"), ("E-03", "03_b437950e"),
             ("E-05", "05_e25c6859"), ("E-09", "09_d9ef3c94"), ("E-12", "12_72cb45d3"), ("E-13", "13_956c09af"), ("E-15", "15_d1e1322c"),
             ("E-18", "18_1ffa9083")])
S = json.load(open(f"{P}/eyes_series.json"))
FPS = 25


def band(key):
    r = S[key]
    cap = cv2.VideoCapture(r["path"])
    out = []
    (ly, lx), (ry, rx) = r["eyes_tmpl"]
    iod = r["iod"]
    t = 0
    ok, fr = cap.read()
    while ok:
        s, ty, tx = r["s"][t], r["ty"][t], r["tx"][t]
        if s is None or (isinstance(s, float) and np.isnan(s)):
            out.append(np.full((20, 64), np.nan, np.float32))
        else:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            cy = ty + (ly + ry) / 2 * s
            x1, x2 = tx + min(lx, rx) * s - 0.4 * iod * s, tx + max(lx, rx) * s + 0.4 * iod * s
            y1, y2 = cy - 0.25 * iod * s, cy + 0.25 * iod * s
            c = g[max(0, int(round(y1))):int(round(y2)), max(0, int(round(x1))):int(round(x2))]
            out.append(cv2.resize(c, (64, 20), interpolation=cv2.INTER_AREA))
        t += 1
        ok, fr = cap.read()
    return key, np.stack(out)


def ncc(a, b):
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)
    return float(np.nansum(a * b) / (np.sqrt(np.nansum(a * a) * np.nansum(b * b)) + 1e-6))


if __name__ == "__main__":
    with Pool(int(sys.argv[1])) as pool:
        B = dict(pool.map(band, list(S)))
    # tracking-failure mask: face box > 1.2 IOD from its frame-0 place or scale pinned at the search edge -> NaN
    OK = {}
    for key, r in S.items():
        ty, tx, sc = (np.asarray(r[k], float) for k in ("ty", "tx", "s"))
        iod = r["iod"]
        OK[key] = (np.abs(ty - ty[0]) <= 1.2 * iod) & (np.abs(tx - tx[0]) <= 1.2 * iod) & (sc > 0.801) & (sc < 1.399)
    # pooled reference per clip: 0-2 s of ALL renders of the clip (GT + gen; inputs identical there, render-length noise averaged out)
    ref = {}
    for idn in set(ROWS.values()):
        keys = [k for k in S if S[k]["clip"] == idn and k.split("/")[0] in ("GT", "fs", "d1", "d2", "d3")]
        sl = slice(0, 2 * FPS)
        stack = np.concatenate([B[k][sl][OK[k][sl]] for k in keys])
        tm = np.nanmean(stack, 0)
        own = [ncc(x, tm) for x in stack]
        cat = lambda m: np.concatenate([np.asarray(S[k][m], float)[sl][OK[k][sl]] for k in keys])
        ref[idn] = dict(ix=float(np.nanmedian(cat("ix"))), iy=float(np.nanmedian(cat("iy"))), dark=float(np.nanmedian(cat("dark"))),
                        tm=tm, ncc_thr=float(np.percentile(own, 10)), n_ref_frames=int(len(own)))
    per = {}
    for key, r in S.items():
        idn = r["clip"]
        R = ref[idn]
        ix, iy, dk = (np.asarray(r[k], float) for k in ("ix", "iy", "dark"))
        nc = np.array([ncc(B[key][t], R["tm"]) for t in range(len(ix))])
        bad = ~OK[key]
        ix[bad], iy[bad], dk[bad], nc[bad] = np.nan, np.nan, np.nan, np.nan
        open_rel = dk / max(R["dark"], 1e-3)
        iy_rel = iy - R["iy"]
        geo = ((np.abs(ix - R["ix"]) < 0.08) & (np.abs(iy_rel) < 0.08) & (open_rel >= 0.7)).astype(float)
        onc = (nc >= R["ncc_thr"]).astype(float)
        down = ((iy_rel > 0.08) | (open_rel < 0.7)).astype(float)
        geo[bad], onc[bad], down[bad] = np.nan, np.nan, np.nan
        o = {}
        for w, (a, b) in {"0-2": (0, 2), "2-12": (2, 11.8), "12-20": (12, 19.9)}.items():
            s = slice(int(a * FPS), int(b * FPS))
            if len(ix[s]) < FPS:
                continue
            o[w] = dict(open_rel=round(float(np.nanmean(open_rel[s])), 3), iy_rel=round(float(np.nanmean(iy_rel[s])), 4),
                        oncam_geo=round(float(np.nanmean(geo[s])), 3), eye_ncc=round(float(np.nanmean(nc[s])), 3),
                        oncam_ncc=round(float(np.nanmean(onc[s])), 3), down_or_closed=round(float(np.nanmean(down[s])), 3), tracked=round(float(np.mean(OK[key][s])), 3))
        o["clip"] = idn
        o["unseen"] = idn in UNSEEN
        o["T"] = len(ix)
        sec = lambda v: [None if np.all(np.isnan(v[i * FPS:(i + 1) * FPS])) else round(float(np.nanmean(v[i * FPS:(i + 1) * FPS])), 3)
                         for i in range(len(ix) // FPS)]
        o["per_sec"] = {k: sec(v) for k, v in (("open_rel", open_rel), ("iy_rel", iy_rel), ("ix_rel", ix - R["ix"]), ("eye_ncc", nc),
                                                ("oncam_ncc", onc), ("oncam_geo", geo), ("down_or_closed", down), ("tracked", OK[key].astype(float)))}
        per[key] = o
    # aggregates: GT is per row (S-05/S-15 share the GT clip)
    agg = {}
    mets = ("open_rel", "iy_rel", "oncam_geo", "eye_ncc", "oncam_ncc", "down_or_closed", "tracked")
    for grp in ("all", "unseen", "train"):
        rows = [r for r in ROWS if ROWS[r] != "10_27286bdd" and (grp == "all" or (ROWS[r] in UNSEEN) == (grp == "unseen"))]
        agg[grp] = {}
        for c in ("GT", "fs", "d1", "d2", "d3", "l500"):
            for w in ("2-12", "12-20"):
                ks = [f"GT/{ROWS[r]}" if c == "GT" else f"{c}/{r}" for r in rows]
                vals = [per[k][w] for k in ks if w in per[k]]
                if vals:
                    agg[grp][f"{c} {w}"] = {m: round(float(np.nanmean([v[m] for v in vals])), 3) for m in mets}
                    if c != "GT" and w == "2-12":
                        gtv = {r: per[f"GT/{ROWS[r]}"]["2-12"] for r in rows}
                        agg[grp][f"{c} {w}"]["rows_oncam_ncc_below_GT"] = int(sum(per[f"{c}/{r}"]["2-12"]["oncam_ncc"] < gtv[r]["oncam_ncc"] for r in rows))
                        agg[grp][f"{c} {w}"]["n_rows"] = len(rows)
    # render-agent outputs (12 s renders compare with E6 = GT pose + GT audio 12 s ctrl; E0-E3 are 20 s gen variants)
    rend = {k: {w: per[k].get(w) for w in ("2-12", "12-20")} for k in per if k.startswith("R")}
    out = {"note": __doc__, "aggregate": agg, "render_agent": rend, "per_video": per,
           "ref": {k: {kk: (round(v, 4) if isinstance(v, float) else None) for kk, v in d.items() if kk != "tm"} for k, d in ref.items()}}
    json.dump(out, open(f"{P}/eyes_results.json", "w"), indent=1)
    eps = {"note": "per sample, per 1-s bin (bin i = [i, i+1) s) eye metrics measured in the rendered video. open_rel = iris/pupil dark area rel. the clip's pooled "
                   "0-2 s median (<0.7 = lids lowered/closed); iy_rel / ix_rel = iris centroid offset from the clip's 0-2 s median, / eye-window width "
                   "(+y = DOWN, +x = image right); oncam = |ix_rel|<0.08 & |iy_rel|<0.08 & open_rel>=0.7 (fraction of frames in the bin); "
                   "down_or_closed = iy_rel>0.08 or open_rel<0.7; tracked = face-tracker OK fraction (clip 10_27286bdd = S-05/S-15 unreliable; "
                   "E-18 noisy: glasses). GT/<row> = the clip's GT render (12 s). R<E>/<row> = render-agent tests.",
           "rows": {}}
    for key, o in per.items():
        st, rest = key.split("/", 1)
        if st == "GT":
            for r, idn in ROWS.items():
                if idn == rest:
                    eps["rows"][f"GT/{r}"] = {"clip": idn, **{k: o["per_sec"][k] for k in ("open_rel", "iy_rel", "ix_rel", "oncam_geo", "down_or_closed", "tracked")}}
        else:
            eps["rows"][key] = {"clip": o["clip"], **{k: o["per_sec"][k] for k in ("open_rel", "iy_rel", "ix_rel", "oncam_geo", "down_or_closed", "tracked")}}
    json.dump(eps, open(f"{P}/eye_per_sec.json", "w"))
    print("eye_per_sec rows", len(eps["rows"]))
    for grp in agg:
        for k, v in agg[grp].items():
            print(grp, k, v)
    for k in sorted(rend):
        print(k, rend[k])
    for k in ("GT/15_d1e1322c", "fs/E-15", "l500/E-15", "d3/E-15", "GT/06_74807fe0", "d3/S-13", "l500/S-13"):
        print(k, "down", per[k]["per_sec"]["down_or_closed"], "iy", per[k]["per_sec"]["iy_rel"])
