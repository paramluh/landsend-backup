"""PPM pose server client: python3 send.py [--url URL] [--wait video|gen] [--wav W] [--pose NPZ] [--voice W] key=value ...
   e.g. python3 send.py n_samples=2 pose_light_topk=250 pose_cfg=3 audio_cfg=3        (stdlib only)"""
import argparse
import base64
import json
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default=None)
ap.add_argument("--wait", default="video", choices=["video", "gen", "none"])
ap.add_argument("--wav")
ap.add_argument("--pose")
ap.add_argument("--voice")
ap.add_argument("kv", nargs="*")
a = ap.parse_args()
U = a.url or open("/data/home/amartyavarma/ppm/ppm-pose-server/URL").read().strip()
body = {}
for kv in a.kv:
    k, v = kv.split("=", 1)
    body[k] = v
for f, k in ((a.wav, "upload_wav_b64"), (a.pose, "upload_pose_b64"), (a.voice, "voice_wav_b64")):
    if f:
        body[k] = base64.b64encode(open(f, "rb").read()).decode()
if a.wav:
    body.setdefault("prompt", "upload")
t0 = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(f"{U}/generate", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=60))
print(json.dumps(r), flush=True)
if a.wait == "none" or "id" not in r:
    raise SystemExit
while True:
    s = json.load(urllib.request.urlopen(f"{U}/status/{r['id']}", timeout=30))
    if s["state"] in ("done", "failed") or (a.wait == "gen" and s["state"] not in ("queued", "generating")):
        break
    time.sleep(1)
print(json.dumps({"id": r["id"], "state": s["state"], "error": s.get("error"), "wall_s": round(time.time() - t0, 1),
                  "samples": [{"k": x["k"], "video": x["video"], "video_fail": x["video_fail"], "stick": x["stick"],
                               "gen": (x.get("meta") or {}).get("timing"), "words": (x.get("meta") or {}).get("gen_words"),
                               "text": ((x.get("meta") or {}).get("gen_text") or "")[:90]} for x in s["samples"]], "view": f"{U}/view/{r['id']}"}, indent=1))
