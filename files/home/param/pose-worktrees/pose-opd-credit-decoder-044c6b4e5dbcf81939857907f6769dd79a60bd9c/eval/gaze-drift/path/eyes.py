"""Follow-up: per-frame EYE metrics in the rendered videos (eye openness, vertical iris position, eyes-on-camera fraction).
Face tracked as in gaze_video.py (frame-0 pose eyes -> face template of frames 0..9, multi-scale NCC); eye windows (0.6 x 0.42 IOD) at the
scaled frame-0 eye keypoints. Per eye window, with skin = median V (HSV value) of a cheek patch 0.5 IOD below the eye:
  dark   = fraction of window pixels with V < 0.55 skin       (iris + pupil + lashes visible)
  sclera = fraction with V > 0.80 skin and S < 0.55 skin S    (white of the eye visible; ~0 when lids are half-closed / eyes downcast)
  open_h = vertical extent (rows) of the dark+sclera region / window width    (eye opening height rel. eye width)
  iy, ix = iris (darkest-20% centroid) offset from the window centre / window width (+y = down)
Eyes-on-camera per frame: |ix - ref_x| < 0.08 and |iy - ref_y| < 0.08 and open_h >= 0.7 ref_open and sclera >= 0.5 ref_sclera (both eyes averaged),
ref = the clip's GT render 0-2 s median (identical inputs in every condition during 0-2 s).
   python eyes.py OUT.json NPROC"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from safetensors.torch import load_file

PR = "/data/home/amartyavarma/video/demo-data/prompts"
S20 = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20"
RQ = "/data/home/amartyavarma/ppm/ppm-pose-server/requests"
RO = "/data/home/amartyavarma/ppm/gaze-drift/render/out"
ROWS = dict([("S-05", "10_27286bdd"), ("S-13", "06_74807fe0"), ("S-15", "10_27286bdd"), ("S-20", "19_47473a46"), ("E-03", "03_b437950e"),
             ("E-05", "05_e25c6859"), ("E-09", "09_d9ef3c94"), ("E-12", "12_72cb45d3"), ("E-13", "13_956c09af"), ("E-15", "15_d1e1322c"),
             ("E-18", "18_1ffa9083")])


def frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    ok, fr = cap.read()
    while ok:
        out.append(fr)
        ok, fr = cap.read()
    return np.stack(out)


def eye_stats(hsv, gray, box, cheek):
    a, b, c, d = box
    H, W = gray.shape
    a, b, c, d = max(0, a), min(H, b), max(0, c), min(W, d)
    ca, cb, cc, cd = cheek
    ca, cb, cc, cd = max(0, ca), min(H, cb), max(0, cc), min(W, cd)
    if b - a < 4 or d - c < 4 or cb - ca < 2 or cd - cc < 2:
        return [np.nan] * 6
    v = hsv[a:b, c:d, 2].astype(np.float32)
    s = hsv[a:b, c:d, 1].astype(np.float32)
    sv = float(np.median(hsv[ca:cb, cc:cd, 2]))
    ss = float(np.median(hsv[ca:cb, cc:cd, 1])) + 1
    dark = v < 0.55 * sv
    scl = (v > 0.80 * sv) & (s < 0.55 * ss)
    rows = np.where((dark | scl).sum(1) >= 2)[0]
    open_h = (rows.max() - rows.min() + 1) / (d - c) if len(rows) else 0.0
    # core window (central 70% x 60%): excludes glasses frames / brows; dark = iris+pupil visible
    hh, ww = b - a, d - c
    ka, kb, kc, kd = int(0.2 * hh), int(0.8 * hh), int(0.15 * ww), int(0.85 * ww)
    vc = v[ka:kb, kc:kd]
    dark_core = float((vc < 0.5 * sv).mean())
    min_core = float(np.percentile(vc, 5) / max(sv, 1))
    g = gray[a:b, c:d].astype(np.float32)[ka:kb, kc:kd]
    thr = np.percentile(g, 20)
    w = np.clip(thr - g, 0, None) + 1e-6
    ys, xs = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    ix = ((w * xs).sum() / w.sum() - (g.shape[1] - 1) / 2) / g.shape[1]
    iy = ((w * ys).sum() / w.sum() - (g.shape[0] - 1) / 2) / g.shape[1]
    return [float(dark_core), float(scl.mean()), float(open_h), float(ix), float(iy), float(min_core)]


def measure(job):
    key, path, idn = job
    f0 = load_file(f"{PR}/{idn}/llm_inputs.safetensors")["frame0_pose"].numpy()
    le, re = f0[1, :2], f0[2, :2]
    mid = (le + re) / 2
    iod = float(np.linalg.norm(le - re))
    C = frames(path)
    V = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in C])
    T, H, W = V.shape
    tw, th = int(round(2.6 * iod)) | 1, int(round(2.8 * iod)) | 1
    cy0, cx0 = mid[0] + 0.45 * iod, mid[1]
    y0, x0 = int(round(cy0 - th / 2)), int(round(cx0 - tw / 2))
    tmpl = cv2.GaussianBlur(V[:10].mean(0)[y0:y0 + th, x0:x0 + tw], (3, 3), 0)
    ew, eh = 0.60 * iod, 0.42 * iod
    eyes = [(e[0] - y0, e[1] - x0) for e in (le, re)]
    scales = np.arange(0.80, 1.41, 0.04)
    cy, cx = cy0, cx0
    out = {k: [] for k in ("dark", "sclera", "open_h", "ix", "iy", "min_core", "ty", "tx", "s")}
    for t in range(T):
        im = cv2.GaussianBlur(V[t], (3, 3), 0)
        best = (-2, None, None, None)
        for s in scales:
            tws, ths = int(round(tw * s)), int(round(th * s))
            r = int(1.6 * iod * s) + 2
            ya, yb = int(max(0, cy - ths / 2 - r)), int(min(H, cy + ths / 2 + r))
            xa, xb = int(max(0, cx - tws / 2 - r)), int(min(W, cx + tws / 2 + r))
            reg = im[ya:yb, xa:xb]
            if reg.shape[0] <= ths or reg.shape[1] <= tws:
                continue
            res = cv2.matchTemplate(reg, cv2.resize(tmpl, (tws, ths), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            if mv > best[0]:
                best = (mv, s, ya + ml[1] + ths / 2, xa + ml[0] + tws / 2)
        mv, s, ncy, ncx = best
        if s is None:
            for k in out:
                out[k].append(np.nan)
            continue
        cy, cx = ncy, ncx
        ty, tx = cy - th * s / 2, cx - tw * s / 2
        hsv = cv2.cvtColor(C[t], cv2.COLOR_BGR2HSV)
        st = []
        for ey, ex in eyes:
            gy, gx = ty + ey * s, tx + ex * s
            box = (int(round(gy - eh * s / 2)), int(round(gy + eh * s / 2)), int(round(gx - ew * s / 2)), int(round(gx + ew * s / 2)))
            cheek = (int(round(gy + 0.45 * iod * s)), int(round(gy + 0.65 * iod * s)), int(round(gx - 0.2 * iod * s)), int(round(gx + 0.2 * iod * s)))
            st.append(eye_stats(hsv, V[t], box, cheek))
        st = np.nanmean(np.array(st, float), 0)
        for k, v in zip(("dark", "sclera", "open_h", "ix", "iy", "min_core"), st):
            out[k].append(round(float(v), 4))
        out["ty"].append(round(float(ty), 1))
        out["tx"].append(round(float(tx), 1))
        out["s"].append(round(float(s), 3))
    out.update(key=key, path=path, clip=idn, T=int(T), iod=round(iod, 2), eyes_tmpl=[[round(float(a), 1) for a in e] for e in eyes],
               ew=round(ew, 2), eh=round(eh, 2))
    print(key, T, flush=True)
    return out


if __name__ == "__main__":
    jobs = [(f"GT/{idn}", f"{S20}/vrefs/GT-{idn}/video.mp4", idn) for idn in sorted(set(ROWS.values()))]
    for tag in ("fs", "d1", "d2", "d3"):
        mp = json.load(open(f"{S20}/control/xs17641_{tag}.json"))
        for row, (rid, k) in mp.items():
            if row in ROWS:
                jobs.append((f"{tag}/{row}", f"{RQ}/{rid}/s{k}/video.mp4", ROWS[row]))
    for p in sorted(glob.glob(f"{RO}/GAZE-E*-*.mp4")):
        b = os.path.basename(p)[:-4]          # GAZE-E6-S-05
        e, row = b.split("-", 2)[1], b.split("-", 2)[2]
        if row in ROWS:
            jobs.append((f"R{e}/{row}", p, ROWS[row]))
    with Pool(int(sys.argv[2])) as pool:
        res = {r["key"]: r for r in pool.imap_unordered(measure, jobs)}
    json.dump(res, open(sys.argv[1], "w"))
    print("done", len(res), flush=True)
