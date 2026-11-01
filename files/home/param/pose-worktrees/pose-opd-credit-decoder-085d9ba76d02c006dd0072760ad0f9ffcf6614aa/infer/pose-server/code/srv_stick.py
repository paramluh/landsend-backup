"""PPM pose server: stick figure of one sample (s<k>/codes.npz pose + gen.wav), drawn exactly as the pose-eval pages (infer_ppm POSE_EDGES /
POSE_COL, 270 x 480, 25 fps, white flash on the 5 frames after the prefill), decoded with the clip's frame-0 global_state.
   python srv_stick.py SAMPLE_DIR      (qwen-s2 venv, CPU) -> stick.mp4, STICK_DONE | STICK_FAIL"""
import os
import subprocess
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, "/data/home/amartyavarma/ppm/code")
import infer_ppm as I  # noqa: E402
import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402

torch.set_num_threads(4)
D = sys.argv[1]
FF = imageio_ffmpeg.get_ffmpeg_exe()
try:
    z = np.load(f"{D}/codes.npz")
    gs = z["global_state"].astype(float).tolist()
    norm = {"frame0_root": gs[:2], "frame0_width": gs[2], "orig_w": 256, "orig_h": 480}
    P2 = 2 * int(z["prefill_frames"])
    sys.path.insert(0, "/data/home/amartyavarma/ref/pose-tokenizer")
    from pose_tokenizer import PoseTokenizer  # noqa: E402
    ptok = PoseTokenizer.from_pretrained(I.POSE_TOK[0], device="cpu", revision=I.POSE_TOK[1])
    H, Wd = 480, 270

    def draw(p, cf, k):
        if P2 <= k < P2 + 5:
            return np.full((H, Wd, 3), 255, np.uint8)
        img = np.full((H, Wd, 3), 24, np.uint8)
        for g_, edges in I.POSE_EDGES.items():
            col = I.POSE_COL[g_]
            for i, j in edges:
                al = 0.15 + 0.85 * min(cf[i], cf[j])
                if al < 0.2:
                    continue
                cl = tuple(int(v * al) for v in col)
                cv2.line(img, (int(p[i, 1] * Wd), int(p[i, 0] * H)), (int(p[j, 1] * Wd), int(p[j, 0] * H)), cl, 3 if g_ == "body" else 1, cv2.LINE_AA)
        lab = "PREFILL (real)" if k < P2 else "GENERATED"
        cv2.putText(img, lab, (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180) if k < P2 else (80, 200, 255), 1, cv2.LINE_AA)
        return img

    pos, conf = I.decode_pose(ptok, z["pose"].astype(np.int64), norm, "cpu")
    NF = pos.shape[0]
    fr = np.stack([draw(pos[k], conf[k], k) for k in range(NF)])
    out = f"{D}/stick.mp4"
    subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{Wd}x{H}", "-r", "25", "-i", "-",
                    "-i", f"{D}/gen.wav", "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-t", f"{NF / 25:.2f}", "-movflags", "+faststart", out + ".tmp.mp4"],
                   input=np.ascontiguousarray(fr).tobytes(), check=True)
    os.replace(out + ".tmp.mp4", out)
    open(f"{D}/STICK_DONE", "w").write("ok\n")
    print("stick", D, NF, flush=True)
except Exception as e:  # noqa: BLE001
    traceback.print_exc()
    open(f"{D}/STICK_FAIL", "w").write(f"{type(e).__name__}: {e}\n")
    sys.exit(1)
