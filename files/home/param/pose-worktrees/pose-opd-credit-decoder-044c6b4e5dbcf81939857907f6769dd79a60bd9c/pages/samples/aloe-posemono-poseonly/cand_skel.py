"""Candidate card visuals (10-19), CPU: real-pose stick-figure video of one candidate's first 20 s (the audio preview's window), drawn
exactly as the PPM pose pages draw it (fastinfer/samples_fast.render's draw: infer_ppm POSE_EDGES / POSE_COL, 270 x 480, 25 fps,
tag REAL), muxed with the card's 20 s audio (a/cNN.mp3), + its first frame as a JPEG poster.
   python cand_skel.py NUM    (qwen-s2 venv; one process per candidate)"""
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import torch

W = "/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly"
OUT = "/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_posemono_candidates"
sys.path.insert(0, "/data/home/amartyavarma/ppm/code")
import infer_ppm as I  # noqa: E402
import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402

torch.set_num_threads(1)
FF = imageio_ffmpeg.get_ffmpeg_exe()
NPREV = 250
num = int(sys.argv[1])
c = [x for x in json.load(open(f"{W}/candidates.json")) if x["num"] == num][0]
rd = f"{W}/rows" if int(c["row"][2:]) < 10 else f"{W}/rows2"
r = [x for x in json.load(open(f"{rd}/rows.json")) if x["row"] == c["row"]][0]
x = pickle.load(open(f"{rd}/{c['row']}.pkl", "rb"))
P = int(x["P"])
off = np.arange(I.DEPTH, dtype=np.int64) * I.CBS
pc = (x["pdepth"][P + I.CD:P + I.CD + NPREV] - off).T.astype(np.int64)
assert pc.shape == (32, NPREV) and pc.min() >= 0 and x["row_key"] == c["row_key"]
sys.path.insert(0, "/data/home/amartyavarma/ref/pose-tokenizer")
from pose_tokenizer import PoseTokenizer  # noqa: E402
ptok = PoseTokenizer.from_pretrained(I.POSE_TOK[0], device="cpu", revision=I.POSE_TOK[1])
pos, conf = I.decode_pose(ptok, pc, r["norm"]["s1"], "cpu")
H, Wd = 480, 270


def draw(p, cf):                                                     # = samples_fast.render.draw
    img = np.full((H, Wd, 3), 24, np.uint8)
    for g_, edges in I.POSE_EDGES.items():
        col = I.POSE_COL[g_]
        for i, j in edges:
            al = 0.15 + 0.85 * min(cf[i], cf[j])
            if al < 0.2:
                continue
            cl = tuple(int(v * al) for v in col)
            cv2.line(img, (int(p[i, 1] * Wd), int(p[i, 0] * H)), (int(p[j, 1] * Wd), int(p[j, 0] * H)), cl, 3 if g_ == "body" else 1, cv2.LINE_AA)
    cv2.putText(img, "REAL", (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    return img


frames = [draw(pos[f], conf[f]) for f in range(len(pos))]
vid = np.ascontiguousarray(np.stack(frames)[:, :, :, ::-1])
os.makedirs(f"{OUT}/s", exist_ok=True)
fo = f"{OUT}/s/c{num:02d}.mp4"
subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{Wd}x{H}", "-r", "25", "-i", "-",
                "-i", f"{OUT}/a/c{num:02d}.mp3", "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-crf", "26", "-preset", "veryfast",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-shortest", "-movflags", "+faststart", fo + ".tmp.mp4"], input=vid.tobytes(), check=True)
os.replace(fo + ".tmp.mp4", fo)
cv2.imwrite(f"{OUT}/s/c{num:02d}.tmp.jpg", frames[0][:, :, ::-1].copy())
os.replace(f"{OUT}/s/c{num:02d}.tmp.jpg", f"{OUT}/s/c{num:02d}.jpg")
print(num, "skeleton", fo, len(frames), "frames", flush=True)
