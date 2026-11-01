"""Proofs for the v4 eval page (10-19): per row, frames 0..49 of the published generated video vs the published card clip (candidates page),
audio 0-2 s diff, pose tokens / Mimi codes for 0-2 s fed to the model == the card clip's, pose latents 0..12 equal, boundary strip (frames
46..53), and a person / stretch sheet (card image, generated 0.5 / 2.5 / 20 s, real recording 0.5 / 2.5 / 20 s).
   python prove_v3.py <page dir> <out dir>"""
import json
import subprocess
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

K, O = sys.argv[1], sys.argv[2]
W = "/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly"
V3 = "/data/home/amartyavarma/video/studio-posemono/v3"
C = "/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_posemono_candidates/r"
F = "/data/home/amartyavarma/tools/ffmpeg-static/ffmpeg-7.0.2-amd64-static/ffmpeg"
rows = [("R-004", 5), ("R-012", 13), ("R-014", 15), ("R-019", 20)]


def frames(p, n):
    raw = subprocess.run([F, "-v", "error", "-i", p, "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 480, 256, 3)


def audio(p, sec):
    raw = subprocess.run([F, "-v", "error", "-i", p, "-t", str(sec), "-ac", "1", "-ar", "24000", "-f", "f32le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32)


sheet = Image.new("RGB", (256 * 7, 480 * 4))
out = []
for i, (r, c) in enumerate(rows):
    g = frames(f"{K}/rv/{r}-gen.mp4", 54)
    k = frames(f"{C}/c{c:02d}.mp4", 50)
    d = np.abs(g[:50].astype(np.int16) - k.astype(np.int16))
    ga, ka = audio(f"{K}/rv/{r}-gen.mp4", 2.0), audio(f"{C}/c{c:02d}.mp4", 2.0)
    n = min(len(ga), len(ka)) - 2400                                    # 0-1.9 s (AAC frames touching 2.0 s excluded)
    z = np.load(f"{W}/runs-allgen/{r}/codes.npz")
    tok_ok = bool(np.array_equal(z["pose_s1"][:, :25], z["gt_pose_s1"][:, :25]) and np.array_equal(z["s1"][:, :25], z["gt_s1"][:, :25]))
    mc, mg = load_file(f"{V3}/{r}/card/mimi.safetensors")["codes"], load_file(f"{V3}/{r}/gen/mimi.safetensors")["codes"]
    pc, pg = load_file(f"{V3}/{r}/card/pose_latents_electron_v3.safetensors")["latents"], load_file(f"{V3}/{r}/gen/pose_latents_electron_v3.safetensors")["latents"]
    rec = {"row": r, "card": c, "frames_0_49_mean_absdiff": round(float(d.mean()), 3), "frames_0_49_max_absdiff": int(d.max()),
           "audio_0_1p9s_max_absdiff": round(float(np.abs(ga[:n] - ka[:n]).max()), 4), "audio_0_1p9s_corr": round(float(np.corrcoef(ga[:n], ka[:n])[0, 1]), 5),
           "llm_prefill_tokens_eq_card_tokens_0_24": tok_ok, "video_model_mimi_0_24_eq": bool(torch.equal(mc[:, :25], mg[:, :25])),
           "pose_latents_0_12_eq": bool(torch.equal(pc[:13], pg[:13])),
           "pose_latents_0_12_max_absdiff": round(float((pc[:13].float() - pg[:13].float()).abs().max()), 4),
           "boundary_px_jumps_47_52": [round(float(np.abs(g[t + 1].astype(np.float32) - g[t].astype(np.float32)).mean()), 2) for t in range(46, 52)]}
    out.append(rec)
    print(json.dumps(rec), flush=True)
    Image.fromarray(np.concatenate([g[t] for t in range(46, 54)], 1)).save(f"{O}/{r}-boundary.png")
    ims = [Image.open(f"{C}/c{c:02d}.jpg").convert("RGB")]
    for kind in ("gen", "real"):
        for t in (0.5, 2.5, 20):
            p = f"{O}/{r}-{kind}-{t}.png"
            subprocess.run([F, "-v", "error", "-y", "-ss", str(t), "-i", f"{K}/rv/{r}-{kind}.mp4", "-frames:v", "1", p], check=True)
            ims.append(Image.open(p).convert("RGB"))
    for j, im in enumerate(ims):
        sheet.paste(im.resize((256, 480)), (256 * j, 480 * i))
sheet.resize((896, 960)).save(f"{O}/sheet.png")
strip = np.concatenate([np.asarray(Image.open(f"{O}/{r}-boundary.png")) for r, _ in rows], 0)
Image.fromarray(strip).resize((strip.shape[1] // 2, strip.shape[0] // 2)).save(f"{O}/boundary.png")
json.dump(out, open(f"{O}/proof.json", "w"), indent=1)
print("PROOF DONE")
