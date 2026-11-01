"""10-22 xabi-ft-17352-17641 samples: one run per export on the PPM pose-server on-disk queue, each = the page's newest run (s20 control/
cfg3_a150p50.json: same 11 rows / 10 requests, clips, seeds, 2 s prefill + 18 s, voice clip by rule, no persona, temps 1, no rep-pen, pose cb0 k30 /
pose light k50, video 9-sinks + 9 refs 40+10 steps / audio guidance 2 / seed 2026) with ONLY: pose CFG OFF (pose_cfg 0), speech light top-k 30
(fantastic set: cb0 k30 + light k30), and the export. Writes request dirs exactly as server.py POST /generate does (C.normalize, request.json
+ _submitted), s20 control/<tag>.json ({row: [rid, k]}) + <tag>.watch.   python3 submit_xs.py    (stdlib; run on slinky-15: ckpt_ok)"""
import json, os, sys, time
sys.path.insert(0, "/data/home/amartyavarma/ppm/ppm-pose-server/code")
import srv_common as C
E = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20"
SRC = "cfg3_a150p50"
RUNS = [("xs17352_fs", "/scratch/amartyavarma/xabi-samples/w17352/export-200"),
        ("xs17641_fs", "/scratch/amartyavarma/xabi-samples/w17641/export-200"),
        ("xs16610_fs", "/data/home/amartyavarma/ppm/aloe-v2pure-16610/export-200")]
NEW = {"pose_cfg": 0.0, "light_topk": 30}
FS = json.load(open(C.FANTASTIC))["params"]
rows = json.load(open(f"{E}/control/{SRC}.json"))
by_rid = {}
for row, (rid, k) in rows.items():
    by_rid.setdefault(rid, []).append((row, k))
for tag, ck in RUNS:
    out = {}
    for rid, rk in by_rid.items():
        q = json.load(open(f"{C.REQ}/{rid}/request.json"))
        body = {k: v for k, v in q.items() if not k.startswith("_") and k not in ("frames_prefill", "frames_gen")}
        body.update(NEW, ckpt=ck)
        nrid, r = C.normalize(body)
        for k in ("text_temp", "text_topk", "min_p", "freq_pen", "freq_win", "pad_bias", "factorised_text", "cat_temp", "text_rep", "cb0_temp", "cb0_topk",
                  "light_temp", "light_topk"):
            assert r[k] == FS[k], (k, r[k], FS[k])
        assert r["pose_cfg"] == 0 and r["audio_cfg"] == 0 and r["pose_rep"] == 1 and r["pose_light_rep"] == 1 and r["persona"] == "" and r["voice_clip"] == "rule"
        d = C.rdir(nrid)
        cached = os.path.exists(f"{d}/request.json")
        if not cached:
            os.makedirs(d, exist_ok=True)
            C.write_json(f"{d}/request.json", {**r, "_submitted": time.time()})
        diff = {k: (q.get(k), r.get(k)) for k in set(q) | set(r) if not k.startswith("_") and q.get(k) != r.get(k)}
        assert set(diff) <= set(NEW) | {"ckpt"}, (tag, diff)
        for row, k in rk:
            out[row] = [nrid, k]
        print(f"{tag}: {rid} -> {nrid} {'cached' if cached else 'new'} {[x for x, _ in rk]} diff {diff}", flush=True)
    out = {row: out[row] for row in rows}
    json.dump(out, open(f"{E}/control/{tag}.json", "w"), indent=1)
    open(f"{E}/control/{tag}.watch", "w").write("".join(f"{rid} {k}\n" for rid, k in out.values()))
print("submitted", time.strftime("%H:%M:%S"))
