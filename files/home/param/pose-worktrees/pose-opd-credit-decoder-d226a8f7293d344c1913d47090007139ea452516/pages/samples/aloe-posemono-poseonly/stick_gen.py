"""Stick figure of the full-AR GENERATED pose (runs-studio/<row> pose_s1: package prefill 0-2 s, then our model) drawn exactly as the PPM pose
pages draw it (samples_fast.render's draw: infer_ppm POSE_EDGES / POSE_COL, 270 x 480, 25 fps), decoded with the package's frame-0 state
(global_state root + shoulder width, 256 x 480 frame), cut to the full-AR video's length (1497 frames = 59.88 s), the same white flash on frames
50..54, audio = the full-AR video's generated audio (s1.wav). -> stick/<row>.mp4
   python stick_gen.py ROW     (qwen-s2 venv, CPU)"""
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, "/data/home/amartyavarma/ppm/code")
import infer_ppm as I  # noqa: E402
import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402

torch.set_num_threads(1)
W = "/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly"
row = sys.argv[1]
FF = imageio_ffmpeg.get_ffmpeg_exe()
r = [x for x in json.load(open(f"{W}/rows-studio/rows.json")) if x["row"] == row][0]
z = np.load(f"{W}/runs-studio/{row}/codes.npz")
pc = z["pose_s1"].astype(np.int64)
sys.path.insert(0, "/data/home/amartyavarma/ref/pose-tokenizer")
from pose_tokenizer import PoseTokenizer  # noqa: E402
ptok = PoseTokenizer.from_pretrained(I.POSE_TOK[0], device="cpu", revision=I.POSE_TOK[1])
pos, conf = I.decode_pose(ptok, pc, r["norm"]["s1"], "cpu")
H, Wd, NF = 480, 270, 1497


def draw(p, cf, k):
    img = np.full((H, Wd, 3), 24, np.uint8)
    if 50 <= k <= 54:
        return np.full((H, Wd, 3), 255, np.uint8)                       # the white flash (2.0-2.2 s), as the rendered videos
    for g_, edges in I.POSE_EDGES.items():
        col = I.POSE_COL[g_]
        for i, j in edges:
            al = 0.15 + 0.85 * min(cf[i], cf[j])
            if al < 0.2:
                continue
            cl = tuple(int(v * al) for v in col)
            cv2.line(img, (int(p[i, 1] * Wd), int(p[i, 0] * H)), (int(p[j, 1] * Wd), int(p[j, 0] * H)), cl, 3 if g_ == "body" else 1, cv2.LINE_AA)
    cv2.putText(img, "PREFILL (real)" if k < 50 else "GENERATED", (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 180, 180) if k < 50 else (80, 200, 255), 1, cv2.LINE_AA)
    return img


fr = np.stack([draw(pos[k], conf[k], k) for k in range(NF)])
os.makedirs(f"{W}/stick", exist_ok=True)
out = f"{W}/stick/{row}.mp4"
subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{Wd}x{H}", "-r", "25", "-i", "-",
                "-i", f"{W}/runs-studio/{row}/s1.wav", "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-t", f"{NF / 25:.2f}", "-movflags", "+faststart", out + ".tmp.mp4"],
               input=np.ascontiguousarray(fr[:, :, :, ::-1][:, :, :, ::-1]).tobytes(), check=True)
os.replace(out + ".tmp.mp4", out)
print("stick", row, out, NF, flush=True)
