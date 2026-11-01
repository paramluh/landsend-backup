"""10-27 user ("rosie not daisy"): Rosie (account xyvR1rjHTZbOvkfV1yDCEFr1dgx2, the Rosie voice-FT speaker): her REAL tracked pose
(aloe-handover-full w00 pose tokens, decoded against the call's frame-0 root + shoulder width) as a stick figure with her REAL voice
(the same lane's Mimi codes, aligned by construction), 40 s windows where she talks most (w00.vad.json), from 4 of her FT conversations.
Her lane = the speaker whose pose.json source_video is actor_<account>.mp4. Same drawing as daisy/stickvid/make.py."""
import json
import os
import re
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb")
import stick  # noqa: E402
import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import soundfile as sf  # noqa: E402

ACC = "xyvR1rjHTZbOvkfV1yDCEFr1dgx2"
O = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/rosie_vid"
A = "/data/home/amartyavarma/data/aloe-handover-full/DATA/files/none"
plan = open("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/data/xyvR1rjH/plan_lw16_e058dc63b1ab6735.json").read()
convs = sorted(set(re.findall(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/seg00", plan)))
WIN, CTX, N = 500, 12, 4
picks = []
for sid in convs:
    d = f"{A}/{sid}/seg00"
    if not os.path.isfile(f"{d}/w00.vad.json"):
        continue
    lane = None
    for s in ("SPEAKER_00", "SPEAKER_01"):
        try:
            if ACC in json.load(open(f"{d}/w00.{s}.pose.json")).get("source_video", ""):
                lane = s
        except (OSError, ValueError):
            pass
    if not lane:
        continue
    iv = json.load(open(f"{d}/w00.vad.json"))[lane]
    T = np.load(f"{d}/w00.{lane}.pose_tokens.npy", mmap_mode="r").shape
    T = T[1] if T[0] == 32 else T[0]
    best = (0, 0)
    for t0 in np.arange(60.0, max(61.0, T / 12.5 - 41), 5.0):
        sp = sum(max(0.0, min(b, t0 + 40) - max(a, t0)) for a, b in iv)
        best = max(best, (sp, float(t0)))
    picks.append((best[0], sid, lane, best[1]))
picks.sort(reverse=True)
sel, seen = [], set()
for p in picks:                                                        # the 4 talkiest windows, spread over 4 different calls
    if p[1] not in seen:
        sel.append(p)
        seen.add(p[1])
    if len(sel) == N:
        break
from transformers import MimiModel  # noqa: E402
mimi = MimiModel.from_pretrained("kyutai/mimi").eval()
out = []
for i, (sp, sid, lane, t0) in enumerate(sel):
    base = f"{A}/{sid}/seg00/w00.{lane}"
    pt, co = np.load(f"{base}.pose_tokens.npy"), np.load(f"{base}.codes.npy")
    pt, co = (pt if pt.shape[0] == 32 else pt.T), (co if co.shape[0] == 32 else co.T)
    nm = json.load(open(f"{base}.pose.json"))["normalization"]
    gs = [nm["root0_yx_normalized"][0], nm["root0_yx_normalized"][1], nm["shoulder_width0_normalized"]]
    f0 = int(round(t0 * 12.5))
    f1 = min(f0 + WIN, pt.shape[1], co.shape[1])
    lo = max(0, f0 - CTX)
    with torch.no_grad():
        w = mimi.decode(torch.as_tensor(co[:, lo:f1].astype(np.int64))[None]).audio_values[0, 0].float().numpy()[(f0 - lo) * 1920:]
    wav = f"{O}/rosie_{i + 1}.wav"
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
        cv2.putText(img, "Rosie - real pose", (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        fr.append(img)
    fr = np.stack(fr)
    mp4 = f"{O}/rosie_{i + 1}.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                    "-r", "25", "-i", "-", "-i", wav, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-t", f"{len(fr) / 25:.2f}", "-movflags", "+faststart", mp4],
                   input=np.ascontiguousarray(fr).tobytes(), check=True)
    out.append({"i": i + 1, "file": os.path.basename(mp4), "conv": sid, "lane": lane, "start_s": t0, "secs": round((f1 - f0) / 12.5, 1),
                "speech_s": round(sp, 1)})
    print("done", i + 1, sid, lane, t0, round(sp, 1), flush=True)
json.dump(out, open(f"{O}/samples.json", "w"), indent=1)
