"""Strips + PSNR for the GAZE-* renders (render/out/). Per ROW:
  20 s (frames at 1, 4, 8, 12, 16, 19 s): <key>-strip.png (top original sample video.mp4, bottom the render; 160 px wide) and
     <key>-face.png (head crops 2x, centred on that variant's drawn nose) for E0-E3.
  12 s (frames at 1, 4, 8, 11 s): <key>-strip.png / -face.png for E4 (vs original), E5, E6 (vs the existing GT render), plus
     <ROW>-swap12-strip.png / -face.png = the 2x2 factorial in one image: gen pose+gen audio (original), gen pose+GT audio (E4),
     GT pose+gen audio (E5), GT pose+GT audio (E6), existing GT render.
  PSNR vs the reference (mean over all common frames, over the 2 s prefill, at each strip time) -> out/psnr.json.
   python strips.py ROW:RID:K:CLIP [...]"""
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

FF = "/data/home/amartyavarma/tools/ffmpeg-static/ffmpeg-7.0.2-amd64-static/ffmpeg"
REQ = "/data/home/amartyavarma/ppm/ppm-pose-server/requests"
VR = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20/vrefs"
R = "/data/home/amartyavarma/ppm/gaze-drift/render"
OUT = f"{R}/out"
T20, T12 = [1, 4, 8, 12, 16, 19], [1, 4, 8, 11]
FPS = 25
_CACHE = {}


def frames(mp4):
    if mp4 not in _CACHE:
        pr = subprocess.run([FF, "-i", mp4], capture_output=True, text=True).stderr
        w, h = map(int, re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", pr).groups())
        p = subprocess.run([FF, "-v", "error", "-i", mp4, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
        _CACHE[mp4] = np.frombuffer(p, np.uint8).reshape(-1, h, w, 3)
    return _CACHE[mp4]


def psnr(a, b):
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return float(10 * np.log10(255 ** 2 / mse)) if mse > 0 else 99.0


def thumb(fr, w=160):
    im = Image.fromarray(fr)
    return im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)


def face(fr, xy, half=60, scale=2):
    h, w = fr.shape[:2]
    x, y = int(round(xy[0])), int(round(xy[1]))
    x0, y0 = min(max(x - half, 0), w - 2 * half), min(max(y - half - 10, 0), h - 2 * half)
    return Image.fromarray(fr[y0:y0 + 2 * half, x0:x0 + 2 * half]).resize((2 * half * scale, 2 * half * scale), Image.LANCZOS)


def grid(rows, labels, ts, title, lw=150):
    tw, th = rows[0][0].size
    hh = 32
    g = Image.new("RGB", (lw + tw * len(ts), hh + th * len(rows)), "white")
    d = ImageDraw.Draw(g)
    d.text((4, 3), title, fill="black")
    for i, t in enumerate(ts):
        d.text((lw + i * tw + 4, 18), f"{t} s", fill="black")
    for r, (ims, lab) in enumerate(zip(rows, labels)):
        for j, line in enumerate(lab.split("|")):
            d.text((4, hh + r * th + th // 2 - 10 + 12 * j), line, fill="black")
        for i, im in enumerate(ims):
            g.paste(im, (lw + i * tw, hh + r * th))
    return g


def idx(ts, n):
    return [min(t * FPS, n - 1) for t in ts]


def compare(key, ref_mp4, ref_lab, ts, nose_ref, nose_new, res):
    v, o = frames(f"{OUT}/{key}.mp4"), frames(ref_mp4)
    n = min(len(v), len(o))
    ii = idx(ts, n)
    res[key] = {"ref": ref_lab, "frames": [len(o), len(v)], "psnr_all": float(np.mean([psnr(o[i], v[i]) for i in range(n)])),
                "psnr_prefill_0_2s": float(np.mean([psnr(o[i], v[i]) for i in range(min(50, n))])),
                "psnr_after_2s": float(np.mean([psnr(o[i], v[i]) for i in range(50, n)])),
                "psnr_at": {t: round(psnr(o[i], v[i]), 2) for t, i in zip(ts, ii)}}
    print(key, json.dumps(res[key]), flush=True)
    e = key.split("-")[1]
    grid([[thumb(o[i]) for i in ii], [thumb(v[i]) for i in ii]], [ref_lab, e], ts,
         f"{key}  PSNR vs {ref_lab}: all {res[key]['psnr_all']:.1f} dB, after 2 s {res[key]['psnr_after_2s']:.1f}").save(f"{OUT}/{key}-strip.png")
    grid([[face(o[i], nose_ref(i)) for i in ii], [face(v[i], nose_new(i)) for i in ii]], [ref_lab, e], ts,
         f"{key} head crops").save(f"{OUT}/{key}-face.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    pj = f"{OUT}/psnr.json"
    res = json.load(open(pj)) if os.path.exists(pj) else {}
    for spec in sys.argv[1:]:
        row, rid, k, clip = spec.split(":")
        orig, gt = f"{REQ}/{rid}/s{k}/video.mp4", f"{VR}/GT-{clip}/video.mp4"
        kp = np.load(f"{R}/vdata/kp-{row}.npz")
        F0 = int(kp["prefill_pose_frames"])
        fixed = kp["E0"][:F0, 0, :2].mean(0)

        kg = np.load(f"{R}/vdata/kpgt-{row}.npz") if os.path.exists(f"{R}/vdata/kpgt-{row}.npz") else None

        def track(e, src=None):
            z = kp if src is None else src
            return lambda i: z[e][min(i, len(z[e]) - 1), 0, :2]
        gtt = track("GT", kg) if kg is not None else (lambda i: fixed)
        done = {}
        for e in ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"):
            key = f"GAZE-{e}-{row}"
            if os.path.exists(f"{VR}/{key}/DONE"):
                shutil.copy(f"{VR}/{key}/video.mp4", f"{OUT}/{key}.mp4")
                done[e] = f"{OUT}/{key}.mp4"
        for e in ("E0", "E1", "E2", "E3"):
            if e in done:
                compare(f"GAZE-{e}-{row}", orig, "original", T20, track("E0"), track(e), res)
        for e in ("E4", "E9"):
            if e in done:
                compare(f"GAZE-{e}-{row}", orig, "original", T12, track("E0"), track("E0"), res)
        for e in ("E5", "E6"):
            if e in done:
                compare(f"GAZE-{e}-{row}", gt, "GT render", T12, gtt, gtt, res)
        for e in ("E7", "E8"):
            if e in done and kg is not None:
                compare(f"GAZE-{e}-{row}", done.get("E6", gt), "E6 GT ctrl", T12, gtt, track(e, kg), res)
        if "E10" in done and os.path.exists(f"{R}/vdata/kppp-{row}.npz"):
            kq = np.load(f"{R}/vdata/kppp-{row}.npz")
            compare(f"GAZE-E10-{row}", orig, "original (gen)", T20, track("E0"), track("E10", kq), res)
        cand = [("gen pose + gen audio|(original, 20 s run)", orig, track("E0")), ("gen pose + gen audio|(E9, 12 s run)", done.get("E9"), track("E0")),
                ("gen pose + GT audio|(E4)", done.get("E4"), track("E0")), ("GT pose + gen audio|(E5)", done.get("E5"), gtt),
                ("GT pose + GT audio|(E6)", done.get("E6"), gtt), ("GT pose, head locked|+ GT audio (E7)", done.get("E7"), track("E7", kg) if kg is not None else gtt),
                ("GT pose, motion x0.5|+ GT audio (E8)", done.get("E8"), track("E8", kg) if kg is not None else gtt), ("GT render|(existing)", gt, gtt)]
        rows = [r for r in cand if r[1]]
        if len(rows) > 2:
            ii = idx(T12, 297)
            grid([[thumb(frames(m)[i]) for i in ii] for _, m, _ in rows], [r[0] for r in rows], T12, f"{row} 12 s swaps", lw=190).save(f"{OUT}/{row}-swap12-strip.png")
            grid([[face(frames(m)[i], nz(i)) for i in ii] for _, m, nz in rows], [r[0] for r in rows], T12, f"{row} 12 s swaps, head crops", lw=190).save(f"{OUT}/{row}-swap12-face.png")
    json.dump(res, open(pj, "w"), indent=1)


if __name__ == "__main__":
    main()
