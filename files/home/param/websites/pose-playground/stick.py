"""10-27 user ("pose only ... display the pose as the stick figure render instead of video"): B's pose tokens -> a stick-figure mp4.

Same decode and drawing as the PPM pose pages / aloe-posemono-poseonly stick_gen.py: ppm/code/infer_ppm.decode_pose (pose-tokenizer
InternalCan/pose-tokenizer-32cb-200k-noshift @ 1d6ff4f0, decoded against the source's frame-0 root + shoulder width) and its POSE_EDGES /
POSE_COL (copied here so the worker never imports infer_ppm's model modules), 270 x 480, 25 fps (2 pose frames per 12.5 Hz step), audio =
the run's b.wav. Frames of the real-pose prefill are labelled PREFILL (real), the rest GENERATED.
  stick_mp4(pose [n, 32] local codes, n_prefill_steps, [root_y, root_x, width], wav path, out path)"""
import os
CANOPY_ROOT = os.environ.get("CANOPY_ROOT") or os.path.expanduser("~")   # cluster home holding ppm/, venvs/, models/ (see README: paths)
HOME_DIR = os.path.expanduser("~")
import subprocess
import sys

import numpy as np
import torch

POSE_TOK = ("InternalCan/pose-tokenizer-32cb-200k-noshift", "1d6ff4f0de89408cb054da618e309d4334596252")   # = infer_ppm.POSE_TOK
DEPTH = 32
POSE_EDGES = {                                                          # = infer_ppm.POSE_EDGES
    "face": [(0, 1), (0, 2), (1, 3), (2, 4)],
    "body": [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12)],
    "lh": [(i + 13, j + 13) for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
                                         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]] + [(9, 13)],
    "rh": [(i + 34, j + 34) for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
                                         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]] + [(10, 34)],
}
POSE_COL = {"face": (200, 200, 200), "body": (80, 170, 255), "lh": (90, 220, 120), "rh": (255, 140, 80)}
H, W = 480, 270
_TOK = None


def _tok():
    global _TOK
    if _TOK is None:
        p = os.environ.get("POSE_TOKENIZER_CODE", "/home/param/pose-repo/infer/engine/posetok")
        if p not in sys.path:
            sys.path.append(p)
        from pose_tokenizer import PoseTokenizer
        _TOK = PoseTokenizer.from_pretrained(os.environ.get("POSE_TOKENIZER_DIR", "/home/param/models/pose-tokenizer-32cb-200k-noshift"), device="cpu")
    return _TOK


def decode_pose(pc, gs):
    """= infer_ppm.decode_pose: pose codes [32, F] -> positions [2F, 55, 2] (y, x in [0, 1]) and confidences [2F, 55]."""
    tok = _tok()                                                        # (puts ref/pose-tokenizer on sys.path first)
    from pose_tokenizer.data.kinematic import NUM_JOINTS, offsets_to_positions
    with torch.no_grad():
        rec = tok.decode([torch.as_tensor(pc[k].astype(np.int64))[None] for k in range(DEPTH)]).float()[0].cpu().numpy()
    T = rec.shape[0]
    r3 = rec.reshape(T, NUM_JOINTS, 3)
    g = np.repeat(np.asarray([[gs[0], gs[1], gs[2]]], np.float32), T, 0)
    pos = offsets_to_positions(torch.from_numpy(g[:, :2]), torch.from_numpy(r3[:, :, :2] * g[:, 2:3, None])).numpy()
    return np.asarray(pos), np.clip(r3[:, :, 2], 0, 1)


def stick_mp4(pose, n_pf, gs, wav, out):
    import cv2
    import imageio_ffmpeg
    pos, conf = decode_pose(np.asarray(pose, np.int64).T, gs)
    nf = pos.shape[0]

    def draw(k):
        img = np.full((H, W, 3), 24, np.uint8)
        p, cf = pos[k], conf[k]
        for g_, edges in POSE_EDGES.items():
            col = POSE_COL[g_]
            for i, j in edges:
                al = 0.15 + 0.85 * min(cf[i], cf[j])
                if al < 0.2:
                    continue
                cv2.line(img, (int(p[i, 1] * W), int(p[i, 0] * H)), (int(p[j, 1] * W), int(p[j, 0] * H)), tuple(int(v * al) for v in col),
                         3 if g_ == "body" else 1, cv2.LINE_AA)
        pf = k < 2 * n_pf
        cv2.putText(img, "PREFILL (real)" if pf else "GENERATED", (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (180, 180, 180) if pf else (80, 200, 255), 1, cv2.LINE_AA)
        return img

    fr = np.stack([draw(k) for k in range(nf)])
    tmp = out + ".tmp.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{W}x{H}", "-r", "25", "-i", "-", "-i", wav, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-crf", "20",
                    "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-t", f"{nf / 25:.2f}", "-movflags", "+faststart", tmp],
                   input=np.ascontiguousarray(fr).tobytes(), check=True)
    os.replace(tmp, out)
    return nf
