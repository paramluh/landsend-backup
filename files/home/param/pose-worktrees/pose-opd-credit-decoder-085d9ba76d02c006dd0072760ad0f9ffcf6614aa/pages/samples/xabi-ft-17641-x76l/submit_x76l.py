"""X76L (10-23): the user's "use no cfg but pose and speech light to 250" on the 17,641 Xabi FT samples. Each request = the X76 request
(s20 control/xs17641_fs.json: same 11 rows / 10 requests, clips, seeds, 2 s prefill + 18 s, voice clip by rule, no persona, temps 1, no rep-pen,
text fantastic set, speech cb0 k30, pose cb0 k30, video 9 refs 40+10 steps / seed 2026, ckpt = FT-(b) export-200 node-local) with ONLY
pose light top-k 50 -> 250 and speech light top-k 30 -> 250; CFG stays OFF (pose_cfg 0, audio_cfg 0). Writes request dirs as server.py does
(C.normalize, request.json + _submitted) and s20 control/xs17641_l250.json ({row: [rid, k]}) + .watch.  (stdlib; run on slinky-15: ckpt_ok)"""
import json, os, sys, time
sys.path.insert(0, "/data/home/amartyavarma/ppm/ppm-pose-server/code")
import srv_common as C
E = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20"
SRC, TAG = "xs17641_fs", "xs17641_l250"
NEW = {"pose_light_topk": 250, "light_topk": 250}
FS = json.load(open(C.FANTASTIC))["params"]
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
    for k in ("text_temp", "text_topk", "min_p", "freq_pen", "freq_win", "pad_bias", "factorised_text", "cat_temp", "text_rep", "cb0_temp", "cb0_topk", "light_temp"):
        assert r[k] == FS[k], (k, r[k], FS[k])
    assert r["light_topk"] == 250 and r["pose_light_topk"] == 250 and r["pose_topk"] == 30 and r["pose_temp"] == 1 and r["pose_light_temp"] == 1
    assert r["pose_cfg"] == 0 and r["audio_cfg"] == 0 and r["pose_rep"] == 1 and r["pose_light_rep"] == 1 and r["persona"] == "" and r["voice_clip"] == "rule"
    d = C.rdir(nrid)
    cached = os.path.exists(f"{d}/request.json")
    if not cached:
        os.makedirs(d, exist_ok=True)
        C.write_json(f"{d}/request.json", {**r, "_submitted": time.time()})
    diff = {k: (q.get(k), r.get(k)) for k in set(q) | set(r) if not k.startswith("_") and q.get(k) != r.get(k)}
    assert set(diff) == set(NEW), (rid, diff)
    for row, k in rk:
        out[row] = [nrid, k]
    print(f"{TAG}: {rid} -> {nrid} {'cached' if cached else 'new'} {[x for x, _ in rk]} diff {diff}", flush=True)
out = {row: out[row] for row in rows}
json.dump(out, open(f"{E}/control/{TAG}.json", "w"), indent=1)
open(f"{E}/control/{TAG}.watch", "w").write("".join(f"{rid} {k}\n" for rid, k in out.values()))
print("submitted", time.strftime("%H:%M:%S"))
