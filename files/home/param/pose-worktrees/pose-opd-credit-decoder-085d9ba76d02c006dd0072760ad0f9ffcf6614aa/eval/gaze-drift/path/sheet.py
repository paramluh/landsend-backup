"""Sanity sheet: eye-region crops at 1 fps with the eye metrics overlaid. sheet.py OUT.png key=path=clip ..."""
import sys

import cv2
import numpy as np

sys.path.insert(0, "/data/home/amartyavarma/ppm/gaze-drift/path")
import eyes as E  # noqa: E402

out = sys.argv[1]
rows = []
for a in sys.argv[2:]:
    key, path, idn = a.split("=")
    r = E.measure((key, path, idn))
    C = E.frames(path)
    tiles = []
    for sec in range(0, int(len(C) / 25)):
        t = sec * 25 + 12
        if t >= len(C) or np.isnan(r["s"][t]):
            continue
        s, ty, tx = r["s"][t], r["ty"][t], r["tx"][t]
        (ly, lx), (ry, rx) = r["eyes_tmpl"]
        iod = r["iod"]
        cy = ty + (ly + ry) / 2 * s
        x1, x2 = tx + min(lx, rx) * s - 0.5 * iod * s, tx + max(lx, rx) * s + 0.5 * iod * s
        y1, y2 = cy - 0.45 * iod * s, cy + 0.45 * iod * s
        crop = C[t][max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
        crop = cv2.resize(crop, (176, int(176 * crop.shape[0] / max(1, crop.shape[1]))), interpolation=cv2.INTER_CUBIC)
        crop = np.pad(crop, ((0, 34), (0, 0), (0, 0)), constant_values=255)
        h = crop.shape[0]
        cv2.putText(crop, f"{sec}s d{r['dark'][t]:.2f} m{r['min_core'][t]:.2f}", (2, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        cv2.putText(crop, f"iy{r['iy'][t]:+.2f} ix{r['ix'][t]:+.2f}", (2, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        tiles.append(crop)
    hh = max(x.shape[0] for x in tiles)
    tiles = [np.pad(x, ((0, hh - x.shape[0]), (0, 2), (0, 0)), constant_values=255) for x in tiles]
    row = np.concatenate(tiles, 1)
    lab = np.full((hh, 90, 3), 255, np.uint8)
    cv2.putText(lab, key, (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    rows.append(np.concatenate([lab, row], 1))
W = max(x.shape[1] for x in rows)
rows = [np.pad(x, ((0, 4), (0, W - x.shape[1]), (0, 0)), constant_values=255) for x in rows]
img = np.concatenate(rows, 0)
# split wide sheet into two halves stacked (0-10 s, 10-20 s) for readability
cv2.imwrite(out, img)
print(out, img.shape)
