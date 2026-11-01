"""10-27 user ("make a claude artefact with daisy videos a few as samples"): Daisy (the Aloe speaker who goes by Ada, account fQ85gun7...)
has no video on disk (the source mp4s are GCS-only), so each sample = her REAL pose (aloe-handover-full w00 pose tokens, decoded against
the conversation's frame-0 root + shoulder width) drawn as a stick figure (genweb/stick.py's decode + PPM pose-page drawing) with her REAL
voice (the same w00 lane's Mimi codes, aligned by construction: same T, same 80 ms steps), 40 s from the excerpt windows of aloe/ada.json."""
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb")
import stick  # noqa: E402
import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import soundfile as sf  # noqa: E402

D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/daisy"
A = "/data/home/amartyavarma/data/aloe-handover-full/DATA/files/none"
O = f"{D}/stickvid"
WIN, CTX = 500, 12
ex = sorted(json.load(open(f"{D}/aloe/ada.json"))["excerpts"], key=lambda e: -e["her_s"])[:4]
from transformers import MimiModel  # noqa: E402
mimi = MimiModel.from_pretrained("kyutai/mimi").eval()
out = []
for i, e in enumerate(ex):
    sid = e["conv"].split("/")[0]
    spk = "SPEAKER_" + e["lane"][1:]
    base = f"{A}/{sid}/seg00/w00.{spk}"
    pt = np.load(f"{base}.pose_tokens.npy")
    co = np.load(f"{base}.codes.npy")
    pt = pt if pt.shape[0] == 32 else pt.T
    co = co if co.shape[0] == 32 else co.T
    nm = json.load(open(f"{base}.pose.json"))["normalization"]
    gs = [nm["root0_yx_normalized"][0], nm["root0_yx_normalized"][1], nm["shoulder_width0_normalized"]]
    f0 = int(round(e["start_s"] * 12.5))
    f1 = min(f0 + WIN, pt.shape[1], co.shape[1])
    lo = max(0, f0 - CTX)
    with torch.no_grad():
        w = mimi.decode(torch.as_tensor(co[:, lo:f1].astype(np.int64))[None]).audio_values[0, 0].float().numpy()
    w = w[(f0 - lo) * 1920:]
    wav = f"{O}/daisy_{i + 1}.wav"
    sf.write(wav, np.clip(w, -1, 1), 24000, subtype="PCM_16")
    pos, conf = stick.decode_pose(pt[:, f0:f1].astype(np.int64), gs)
    H, W = stick.H, stick.W
    fr = []
    for k in range(pos.shape[0]):
        img = np.full((H, W, 3), 24, np.uint8)
        p, cf = pos[k], conf[k]
        for g_, edges in stick.POSE_EDGES.items():
            col = stick.POSE_COL[g_]
            for a, b in edges:
                al = 0.15 + 0.85 * min(cf[a], cf[b])
                if al < 0.2:
                    continue
                cv2.line(img, (int(p[a, 1] * W), int(p[a, 0] * H)), (int(p[b, 1] * W), int(p[b, 0] * H)), tuple(int(v * al) for v in col),
                         3 if g_ == "body" else 1, cv2.LINE_AA)
        cv2.putText(img, "Daisy - real pose", (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        fr.append(img)
    fr = np.stack(fr)
    mp4 = f"{O}/daisy_{i + 1}.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                    "-r", "25", "-i", "-", "-i", wav, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-t", f"{len(fr) / 25:.2f}", "-movflags", "+faststart", mp4],
                   input=np.ascontiguousarray(fr).tobytes(), check=True)
    out.append({"i": i + 1, "file": os.path.basename(mp4), "conv": e["conv"], "side": e["side"], "partner": e.get("partner_name"),
                "start_s": e["start_s"], "secs": round((f1 - f0) / 12.5, 1), "her_s_in_window": e["her_s"], "transcript": e.get("transcript", "")})
    print("done", i + 1, mp4, (f1 - f0) / 12.5, flush=True)
json.dump(out, open(f"{O}/samples.json", "w"), indent=1)
