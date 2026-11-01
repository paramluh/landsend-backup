"""PURE (10-22 23:0x PT): the user's "test inference but just set temp 1 and nothing else for everything text speech pose - completely unbiased
model generation" on the 17,641 Xabi FT samples. = the X76 requests (control/xs17641_fs.json: same 11 rows, clips, seeds, voice clip rule, no
persona, 2 s prefill + 18 s, video 9 refs) with EVERY sampler knob neutral: temp 1 on every stream, no top-k / top-p / min-p / freq pen / PAD bias
/ rep-pen / CFG, plain (joint) text draw."""
import json, os, sys, time
sys.path.insert(0, "/data/home/amartyavarma/ppm/ppm-pose-server/code")
import srv_common as C
E = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20"
SRC, TAG = "xs17641_fs", "xs17641_pure"
NEW = {"factorised_text": 0, "cat_temp": 1.0, "text_temp": 1.0, "text_topk": 0, "text_topp": 1.0, "min_p": 0.0, "freq_pen": 0.0, "pad_bias": 0.0,
       "text_rep": 1.0, "cb0_temp": 1.0, "cb0_topk": 0, "cb0_topp": 1.0, "light_temp": 1.0, "light_topk": 0, "light_topp": 1.0,
       "pose_temp": 1.0, "pose_topk": 0, "pose_topp": 1.0, "pose_light_temp": 1.0, "pose_light_topk": 0, "pose_light_topp": 1.0,
       "pose_rep": 1.0, "pose_light_rep": 1.0, "pose_rep_win": 0, "pose_cfg": 0.0, "audio_cfg": 0.0}
rows = json.load(open(f"{E}/control/{SRC}.json"))
by_rid = {}
for row, (rid, k) in rows.items():
    by_rid.setdefault(rid, []).append((row, k))
out = {}
for rid, rk in by_rid.items():
    q = json.load(open(f"{C.REQ}/{rid}/request.json"))
    body = {k: v for k, v in q.items() if not k.startswith("_") and k not in ("frames_prefill", "frames_gen")}
    body.update(NEW)
    nrid, r = C.normalize(body)
    for k, v in NEW.items():
        assert r[k] == v, (k, r[k], v)
    d = C.rdir(nrid)
    cached = os.path.exists(f"{d}/request.json")
    if not cached:
        os.makedirs(d, exist_ok=True)
        C.write_json(f"{d}/request.json", {**r, "_submitted": time.time()})
    for row, k in rk:
        out[row] = [nrid, k]
    print(f"{TAG}: {rid} -> {nrid} {'cached' if cached else 'new'} {[x for x, _ in rk]}", flush=True)
out = {row: out[row] for row in rows}
json.dump(out, open(f"{E}/control/{TAG}.json", "w"), indent=1)
open(f"{E}/control/{TAG}.watch", "w").write("".join(f"{rid} {k}\n" for rid, k in out.values()))
print("submitted", time.strftime("%H:%M:%S"))
