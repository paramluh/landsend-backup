"""posefix 10-26: eye check of hands.py on rendered runs: contact sheet of avatar.mp4 (every 0.4 s of the generated part), frames inside a
detected any-hand gesture burst boxed red. usage: hands_sheet.py RUN_DIR OUT.png"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
FFD = "/data/home/amartyavarma/tools/ffmpeg-static/ffmpeg-7.0.2-amd64-static"
import score as S  # noqa: E402
import hands as HD  # noqa: E402

rd, out = os.path.normpath(sys.argv[1]), sys.argv[2]
summ = json.load(open(os.path.join(rd, "summary.json")))
N = int(summ.get("pose_prefill_frames") or 0)
codes = S._fix_codes(np.load(os.path.join(rd, "tokens.npz"))["pose"])
ck, sk, _ = S.prep_codes(codes, N)
kp, kp0 = S.to_keypoints(S.decode(ck), S.frame0_of(summ.get("person") or S.DEFAULT_PERSON))
lo = 2 * sk
k0 = kp0.numpy()
hs, ws = HD.series(kp.numpy()[:, S.P.SEL].astype(np.float64)[lo:], max(abs(float(k0[5, 0] - k0[6, 0])), 1e-3))
b = HD.bursts((ws > HD.MOVE_THR).any(1))
dur = float(subprocess.run([f"{FFD}/ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", os.path.join(rd, "avatar.mp4")],
                           capture_output=True, text=True).stdout.strip())
t_gen = N * 0.08 if dur > len(codes) * 0.08 - 1.0 else 0.0                  # the render starts at token 0 (prefill included) or at the gen
segs = [(round(t_gen + a / 25, 2), round(t_gen + (e + 1) / 25, 2)) for a, e in b]
en = "+".join(f"between(t,{a},{e})" for a, e in segs) or "0"
vf = (f"select='gte(t,{t_gen})*not(mod(n,10))',drawbox=x=0:y=0:w=iw:h=ih:color=red:t=24:enable='{en}',scale=160:-2,tile=10x5")
subprocess.run([f"{FFD}/ffmpeg", "-y", "-v", "error", "-i", os.path.join(rd, "avatar.mp4"), "-vf", vf, "-frames:v", "1", out], check=True)
print(json.dumps({"run": os.path.basename(rd), "video_s": dur, "t_gen": t_gen, "bursts_s": segs, "n": len(segs)}))
