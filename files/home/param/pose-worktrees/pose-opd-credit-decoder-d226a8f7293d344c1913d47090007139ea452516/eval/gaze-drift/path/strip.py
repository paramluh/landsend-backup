"""Frame strip: same clip, GT render vs generated renders, frames at 1, 4, 8, 12, 16, 19 s. Usage: strip.py OUT.png CROP(0/1) label=video.mp4 ..."""
import sys

import cv2
import numpy as np

out, crop = sys.argv[1], int(sys.argv[2])
items = [a.split("=", 1) for a in sys.argv[3:]]
TS = [1, 4, 8, 12, 16, 19]
rows = []
for lab, path in items:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    ok, fr = cap.read()
    allf = []
    while ok:
        allf.append(fr)
        ok, fr = cap.read()
    h, w = allf[0].shape[:2]
    for t in TS:
        i = int(round(t * fps))
        if i < len(allf):
            im = allf[i]
        else:
            im = np.full_like(allf[0], 40)
            cv2.putText(im, "n/a", (w // 3, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        if crop:
            im = im[int(0.12 * h):int(0.50 * h), int(0.12 * w):int(0.88 * w)]
        im = cv2.resize(im, (im.shape[1] * 160 // im.shape[0] if crop else 120, 160 if crop else 225))
        cv2.putText(im, f"{t}s", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        frames.append(im)
    row = np.concatenate(frames, 1)
    lab_im = np.full((row.shape[0], 110, 3), 255, np.uint8)
    for j, part in enumerate(lab.split("|")):
        cv2.putText(lab_im, part, (3, 20 + 18 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    rows.append(np.concatenate([lab_im, row], 1))
    print(lab, path, "frames", len(allf), "fps", fps, "size", w, h, flush=True)
W = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
cv2.imwrite(out, np.concatenate(rows, 0))
