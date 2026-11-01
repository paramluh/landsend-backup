"""H7: gaze / head-direction proxies measured IN THE RENDERED VIDEOS vs time (no face model on disk -> crude, pose-anchored proxies).
Per video (256x480, 25 fps): face template = mean of frames 0..9 around the frame-0 pose eyes (package frame0_pose, canvas px == video px),
tracked per frame by multi-scale NCC template matching ->
  pos (face centre shift / IOD), scale (face scale vs frame 0), ncc (appearance similarity to the 0-0.4 s face),
  iris (dark-centroid offset inside each eye window, / eye-window width; mean of both eyes; x and y),
  sym (mirror-symmetry axis of the face crop relative to the tracked face centre, / IOD: crude yaw proxy).
Also pixel |GT - gen| per frame (same clip) to show the two renders are identical while inputs are identical (0-2 s).
   python gaze_video.py OUT.json"""
import json
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from safetensors.torch import load_file

PR = "/data/home/amartyavarma/video/demo-data/prompts"
S20 = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20"
RQ = "/data/home/amartyavarma/ppm/ppm-pose-server/requests"
ROWS = dict([("S-05", "10_27286bdd"), ("S-13", "06_74807fe0"), ("S-15", "10_27286bdd"), ("S-20", "19_47473a46"), ("E-03", "03_b437950e"),
             ("E-05", "05_e25c6859"), ("E-09", "09_d9ef3c94"), ("E-12", "12_72cb45d3"), ("E-13", "13_956c09af"), ("E-15", "15_d1e1322c"),
             ("E-18", "18_1ffa9083")])


def frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    ok, fr = cap.read()
    while ok:
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32))
        ok, fr = cap.read()
    return np.stack(out)


def iris(win):
    if win.size == 0 or win.shape[0] < 3 or win.shape[1] < 3:
        return np.nan, np.nan
    thr = np.percentile(win, 20)
    w = np.clip(thr - win, 0, None) + 1e-6
    ys, xs = np.mgrid[0:win.shape[0], 0:win.shape[1]]
    cx = (w * xs).sum() / w.sum()
    cy = (w * ys).sum() / w.sum()
    return (cx - (win.shape[1] - 1) / 2) / win.shape[1], (cy - (win.shape[0] - 1) / 2) / win.shape[1]


def measure(job):
    key, path, idn = job
    f0 = load_file(f"{PR}/{idn}/llm_inputs.safetensors")["frame0_pose"].numpy()   # [133, 3] (y, x, conf) px
    le, re = f0[1, :2], f0[2, :2]
    mid = (le + re) / 2
    iod = float(np.linalg.norm(le - re))
    V = frames(path)
    T, H, W = V.shape
    tw = int(round(2.6 * iod)) | 1
    th = int(round(2.8 * iod)) | 1
    cy0, cx0 = mid[0] + 0.45 * iod, mid[1]                     # face centre (between eyes and mouth)
    y0, x0 = int(round(cy0 - th / 2)), int(round(cx0 - tw / 2))
    base = V[:10].mean(0)
    tmpl = cv2.GaussianBlur(base[y0:y0 + th, x0:x0 + tw], (3, 3), 0)
    # eye windows in template coords
    ew, eh = 0.55 * iod, 0.32 * iod
    eyes = [(e[0] - y0, e[1] - x0) for e in (le, re)]
    scales = np.arange(0.80, 1.41, 0.04)
    cy, cx = cy0, cx0
    rec = {k: [] for k in ("dx", "dy", "scale", "ncc", "iris_x", "iris_y", "sym")}
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
            ts = cv2.resize(tmpl, (tws, ths), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(reg, ts, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            if mv > best[0]:
                best = (mv, s, ya + ml[1] + ths / 2, xa + ml[0] + tws / 2)
        mv, s, ncy, ncx = best
        if s is None:
            for k in rec:
                rec[k].append(np.nan)
            continue
        cy, cx = ncy, ncx
        ty, tx = cy - th * s / 2, cx - tw * s / 2                 # top-left of matched face box
        ix, iy = [], []
        raw = V[t]
        for ey, ex in eyes:
            gy, gx = ty + ey * s, tx + ex * s
            a, b = int(round(gy - eh * s / 2)), int(round(gy + eh * s / 2))
            c, d = int(round(gx - ew * s / 2)), int(round(gx + ew * s / 2))
            x_, y_ = iris(raw[max(0, a):b, max(0, c):d])
            ix.append(x_)
            iy.append(y_)
        # symmetry axis: rows eye-level .. mouth, crop width 1.8 IOD around candidate axis
        ra, rb = int(round(ty + (eyes[0][0] - 0.2 * iod) * s)), int(round(ty + (eyes[0][0] + 1.3 * iod) * s))
        half = int(round(0.9 * iod * s))
        bs, bd = -2, 0.0
        for dd in np.arange(-0.4, 0.401, 0.04):
            ax = int(round(cx + dd * iod * s))
            crop = raw[max(0, ra):rb, max(0, ax - half):ax + half]
            if crop.shape[1] < 2 * half or crop.shape[0] < 4:
                continue
            fl = crop[:, ::-1]
            a1, a2 = crop - crop.mean(), fl - fl.mean()
            sc = float((a1 * a2).sum() / (np.sqrt((a1 ** 2).sum() * (a2 ** 2).sum()) + 1e-6))
            if sc > bs:
                bs, bd = sc, dd
        rec["dx"].append((cx - cx0) / iod)
        rec["dy"].append((cy - cy0) / iod)
        rec["scale"].append(float(s))
        rec["ncc"].append(float(mv))
        rec["iris_x"].append(float(np.nanmean(ix)))
        rec["iris_y"].append(float(np.nanmean(iy)))
        rec["sym"].append(float(bd))
    out = {k: [round(float(v), 4) for v in vals] for k, vals in rec.items()}
    out.update(key=key, path=path, clip=idn, T=int(T), iod=round(iod, 2))
    print(key, T, "iod", round(iod, 1), flush=True)
    return out


def pixdiff(args):
    gpath, gens = args
    G = frames(gpath)
    res = {}
    for key, p in gens:
        V = frames(p)
        T = min(len(G), len(V))
        res[key] = [round(float(np.abs(G[t] - V[t]).mean()), 3) for t in range(T)]
    return res


if __name__ == "__main__":
    jobs, pd = [], {}
    for idn in sorted(set(ROWS.values())):
        jobs.append((f"GT/{idn}", f"{S20}/vrefs/GT-{idn}/video.mp4", idn))
    for tag in ("fs", "d1", "d2", "d3"):
        mp = json.load(open(f"{S20}/control/xs17641_{tag}.json"))
        for row, (rid, k) in mp.items():
            if row in ROWS:
                p = f"{RQ}/{rid}/s{k}/video.mp4"
                jobs.append((f"{tag}/{row}", p, ROWS[row]))
                pd.setdefault(ROWS[row], []).append((f"{tag}/{row}", p))
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 6) as pool:
        out = {}
        for r in pool.imap_unordered(measure, jobs):
            out[r["key"]] = r
        diffs = {}
        for d in pool.imap_unordered(pixdiff, [(f"{S20}/vrefs/GT-{idn}/video.mp4", g) for idn, g in pd.items()]):
            diffs.update(d)
    json.dump({"series": out, "pixdiff_vs_gt": diffs}, open(sys.argv[1], "w"))
    print("done", len(out), flush=True)
