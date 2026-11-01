"""10-27 user (Rosie prefill + voice in the monoplex playground, "obviously"): a playground person 'rosie_xyvR1rjH' built from Rosie's REAL data
in 24272's own stores (the codes / pose tokens both Rosie and base models trained on), from her HELD-OUT call 88e8e0f4 (p7 row 1525, her side
= user), never seen by the Rosie FT:
  genweb/openings/rosie_xyvR1rjH/   conversation start [0, 12 s): mimi_local + pose_tokens + timed words (store word lane) + meta frame0 fields
  genweb/people_extra/rosie_xyvR1rjH/gt/  a 12 s window from the SECOND HALF of the same call (the training voice-clip rule: B's own clip from a
                                     later chunk of the same conversation): mimi_local (voice clip = its first 10 s) + pose_tokens + global_state
  NO_AVATAR (there is no avatar / reference images of Rosie: runs render as the stick figure)
frame0 normalisation = her lane's aloe-handover-full pose.json (tokenizer contract: the whole cut tokenized against conversation frame 0)."""
import json
import os

import numpy as np
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from safetensors.numpy import save_file

PID = "rosie_xyvR1rjH"
K, R, SIDE = 7, 1525, "user"
SD = "/data/home/amartyavarma/ppm/aloe-pose-ft/stores"
G = "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb"
root = f"{SD}/p{K}"
t = pq.read_table(f"{root}/rows.parquet").to_pydict()
i = t["row"].index(R)
key, n, src, lane = t["row_key"][i], int(t["n_frames"][i]), t["src_dir"][i], t[f"{SIDE}_speaker_dir"][i]
assert "88e8e0f4" in key and int(t["frame_start"][i]) == 0, key
pj = json.load(open(f"{src}/w00.{lane}.pose.json"))
assert "xyvR1rjHTZbOvkfV1yDCEFr1dgx2" in pj["source_video"], pj["source_video"]
nm = pj["normalization"]
gs = np.asarray([nm["root0_yx_normalized"][0], nm["root0_yx_normalized"][1], nm["shoulder_width0_normalized"]], np.float32)
codes = np.load(f"{root}/rows-00000-of-00001.codes_{SIDE}.npy", mmap_mode="r")[R]       # [32, 7500]
pose = np.load(f"{root}/rows-00000-of-00001.pose_{SIDE}.npy", mmap_mode="r")[R]
try:
    wt = ipc.open_file(f"{root}/rows-00000-of-00001.words.arrow").read_all()
except Exception:
    wt = ipc.open_stream(open(f"{root}/rows-00000-of-00001.words.arrow", "rb")).read_all()
wrow = wt.slice(R, 1).to_pylist()[0]
assert wrow["row_key"] == key
words = wrow[f"words_{SIDE}"]
words = json.loads(words) if isinstance(words, str) else words
W = [(float(w["start"]), float(w["end"]), str(w["text"]).strip()) for w in words if str(w.get("text", "")).strip()]


def window(f0, L=150):
    c, p = np.asarray(codes[:, f0:f0 + L]), np.asarray(pose[:, f0:f0 + L])
    assert (c >= 0).all() and (p >= 0).all(), ("missing frames in window", f0)
    ws = [[round(a - f0 / 12.5, 3), round(a - f0 / 12.5, 3), x] for a, b, x in W if f0 / 12.5 <= a < (f0 + L) / 12.5]
    return c, p, ws


oc, op, ow = window(0)
best = max(range(n // 2, n - 150, 25), key=lambda f: sum(1 for a, b, x in W if f / 12.5 <= a < (f + 150) / 12.5))
cc, cp, cw = window(best)
od = f"{G}/openings/{PID}"
os.makedirs(od, exist_ok=True)
save_file({"codes": oc.astype(np.int16)}, f"{od}/mimi_local.safetensors")
save_file({"codes": op.astype(np.int16)}, f"{od}/pose_tokens.safetensors")
json.dump(ow, open(f"{od}/words.json", "w"))
json.dump({"person_id": PID, "kind": "aloe-handover-pose store row (Rosie, held-out call)", "row_key": key, "store": f"p{K}", "row": R, "side": SIDE,
           "window_frames": [0, 150], "T": 150, "frame_hz": 12.5,
           "sources": {"pose_tokens": {"frame0": {"source": "aloe-handover-full pose.json normalization (conversation frame 0)",
                                                  "fields": {"frame0_root": [float(gs[0]), float(gs[1])], "frame0_width": float(gs[2])}}},
                       "mimi_local": {"path": f"{root}/rows-00000-of-00001.codes_{SIDE}.npy", "row": R}}}, open(f"{od}/meta.json", "w"), indent=1)
pd = f"{G}/people_extra/{PID}"
os.makedirs(f"{pd}/gt", exist_ok=True)
save_file({"codes": cc.astype(np.int16)}, f"{pd}/gt/mimi_local.safetensors")
save_file({"codes": cp.astype(np.int16), "global_state": gs}, f"{pd}/gt/pose_tokens.safetensors")
os.makedirs(f"{G}/words/{PID}", exist_ok=True)
json.dump({"words": cw, "source": f"store word lane of {key} ({SIDE}), frames {best}..{best + 150}"}, open(f"{G}/words/{PID}/words.json", "w"))
open(f"{pd}/NO_AVATAR", "w").write("no avatar video / reference images of Rosie exist on disk: runs render as the stick figure\n")
json.dump({"id": PID, "label": "Rosie (her real data · stick figure)", "row_key": key, "clip_window_frames": [best, best + 150],
           "opening_words": len(ow), "clip_words": len(cw)}, open(f"{pd}/meta.json", "w"), indent=1)
print(json.dumps({"opening_words": len(ow), "opening_first": ow[:6], "clip_window": [best, best + 150], "clip_words": len(cw), "gs": gs.tolist()}))
