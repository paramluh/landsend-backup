"""10-26 (user via main): per-run POSE STATS for the playground, in their OWN job (genweb/scorer.sbatch), never in the web server: every few
seconds, each run dir whose tokens.npz is new or changed gets posefix's scorer (posefix/score/run_stats.py: cb0 repeat %, out-of-frame %,
keypoint confidence over the generated tokens only, >= the N prefill frames; duplex = B's pose), written atomically to runs/<id>/pose_stats.json
with the tokens.npz mtime it was computed from. The web server only reads those files (history_api.pose_stats). Newest runs first."""
import json
import os
CANOPY_ROOT = os.environ.get("CANOPY_ROOT") or os.path.expanduser("~")   # cluster home holding ppm/, venvs/, models/ (see README: paths)
HOME_DIR = os.path.expanduser("~")
import re
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = f"{ROOT}/runs"
RID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")
sys.path.insert(0, CANOPY_ROOT + "/ppm/aloe-posedup-live/posefix/score")
import torch  # noqa: E402
torch.set_num_threads(2)
import run_stats as RS  # noqa: E402


HANDS = ("gestures_per_min", "gestures_per_min_left", "gestures_per_min_right", "gestures", "hands_moving_pct", "hand_speed_p50",
         "hand_speed_p95", "hand_seconds")                             # 10-26 posefix hands.py (generated frames only)


def todo():
    out = []
    for rid in sorted((r for r in os.listdir(RUNS) if RID.match(r)), reverse=True):
        d = f"{RUNS}/{rid}"
        try:
            mt = os.path.getmtime(f"{d}/tokens.npz")
        except OSError:
            continue
        if not os.path.exists(f"{d}/summary.json"):
            continue
        try:
            with open(f"{d}/pose_stats.json") as f:
                o = json.load(f)
            if o.get("tokens_mtime") == mt and ("gestures_per_min" in o or o.get("error")):   # 10-26: re-score runs from before the hand stats
                continue
        except (OSError, ValueError):
            pass
        out.append((d, mt))
    return out


def main():
    print(f"[scorer] watching {RUNS}", flush=True)
    while True:
        for d, mt in todo():
            t = time.time()
            try:
                r = RS.run_stats(d)
                o = {"cb0_repeat": r.get("cb0_repeat_pct"), "oof": r.get("out_of_frame_pct"), "conf": r.get("conf"), "lowconf": r.get("lowconf_pct"),
                     "prefill_frames": r.get("prefill_tokens"), "gen_frames": r.get("measured_tokens"), "person": r.get("person"), "tokens_mtime": mt,
                     "scorer": "posefix/score/run_stats.py"}
                o.update({k: r.get(k) for k in HANDS})
            except Exception as e:  # noqa: BLE001
                o = {"error": repr(e)[:300], "tokens_mtime": mt}
            with open(f"{d}/pose_stats.json.tmp", "w") as f:
                json.dump(o, f)
            os.replace(f"{d}/pose_stats.json.tmp", f"{d}/pose_stats.json")
            print(f"[scorer] {os.path.basename(d)} {o.get('cb0_repeat')} {o.get('oof')} {o.get('conf')} g/min {o.get('gestures_per_min')} {o.get('error', '')} {time.time() - t:.1f} s", flush=True)
        time.sleep(3)


if __name__ == "__main__":
    main()
