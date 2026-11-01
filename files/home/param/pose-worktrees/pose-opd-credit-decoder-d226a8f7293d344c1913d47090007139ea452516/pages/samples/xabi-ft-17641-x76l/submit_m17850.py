"""M17850 (10-22 23:10 PT, user via main): the NEW monoplex FT from main step 17,850 (aloe-v2pure-main-17850 export-200, node-local copy
/scratch/amartyavarma/xabi-samples/w17850/export-200) on the page's 11 clips with the XJ4 requests' clips / seeds / voice clips (rule) / no persona /
2 s prefill + 18 s / video settings. Two runs:
  xs17850_j4   BEST = XJ4 exactly (judge cfg-cs4: pose cb0 T0.952 k250, light T0.966 k75, light rep 1.022 / 50, pose CFG 4.0 all cb, uncond = speech
               silenced; plain text T1; speech T1 k30 / k30), only the ckpt changes.
  xs17850_n150 NO CFG: temp 1 on every stream; cb0 top-k 50 (speech and pose); cb1-31 top-k 150 (speech and pose); plain text T1 (joint draw, no
               top-k / min-p / freq pen / PAD bias); no rep-pen; no CFG.
Writes request dirs as server.py does + s20 control/<tag>.json ({row: [rid, k]}) + .watch.  (stdlib; run on slinky-15: ckpt_ok)"""
import json, os, sys, time
sys.path.insert(0, "/data/home/amartyavarma/ppm/ppm-pose-server/code")
import srv_common as C
E = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20"
SRC = "xs17641_j4"
CK = "/scratch/amartyavarma/xabi-samples/w17850/export-200"
PLAIN = {"factorised_text": 0, "cat_temp": 1.0, "text_temp": 1.0, "text_topk": 0, "text_topp": 1.0, "min_p": 0.0, "freq_pen": 0.0, "pad_bias": 0.0,
         "text_rep": 1.0}
RUNS = {"xs17850_j4": {"ckpt": CK},
        "xs17850_n150": {"ckpt": CK, **PLAIN, "cb0_temp": 1.0, "cb0_topk": 50, "cb0_topp": 1.0, "light_temp": 1.0, "light_topk": 150, "light_topp": 1.0,
                         "pose_temp": 1.0, "pose_topk": 50, "pose_topp": 1.0, "pose_light_temp": 1.0, "pose_light_topk": 150, "pose_light_topp": 1.0,
                         "pose_rep": 1.0, "pose_light_rep": 1.0, "pose_rep_win": 0, "pose_cfg": 0.0, "audio_cfg": 0.0}}
rows = json.load(open(f"{E}/control/{SRC}.json"))
by_rid = {}
for row, (rid, k) in rows.items():
    by_rid.setdefault(rid, []).append((row, k))
for TAG, NEW in RUNS.items():
    out = {}
    for rid, rk in by_rid.items():
        q = json.load(open(f"{C.REQ}/{rid}/request.json"))
        body = {k: v for k, v in q.items() if not k.startswith("_") and k not in ("frames_prefill", "frames_gen")}
        body.update(NEW)
        nrid, r = C.normalize(body)
        for k, v in {**PLAIN, "cb0_temp": 1.0, "light_temp": 1.0, "audio_cfg": 0.0}.items():
            assert r[k] == v, (TAG, k, r[k], v)
        assert r["ckpt"] == CK and r["persona"] == "" and r["voice_clip"] == "rule" and r["seed"] == q["seed"] and r["clip"] == q["clip"]
        if TAG == "xs17850_j4":
            assert (r["pose_temp"], r["pose_topk"], r["pose_light_temp"], r["pose_light_topk"], r["pose_rep"], r["pose_light_rep"], r["pose_rep_win"],
                    r["pose_cfg"], r["pose_cfg_scope"], r["cfg_uncond"], r["cb0_topk"], r["light_topk"]) == \
                (0.952, 250, 0.966, 75, 1.0, 1.022, 50, 4.0, "all", "silence", 30, 30), TAG
        else:
            assert (r["pose_temp"], r["pose_topk"], r["pose_light_temp"], r["pose_light_topk"], r["pose_rep"], r["pose_light_rep"], r["pose_cfg"],
                    r["cb0_topk"], r["light_topk"]) == (1.0, 50, 1.0, 150, 1.0, 1.0, 0.0, 50, 150), TAG
        d = C.rdir(nrid)
        cached = os.path.exists(f"{d}/request.json")
        if not cached:
            os.makedirs(d, exist_ok=True)
            C.write_json(f"{d}/request.json", {**r, "_submitted": time.time()})
        diff = {k: (q.get(k), r.get(k)) for k in set(q) | set(r) if not k.startswith("_") and q.get(k) != r.get(k)}
        assert set(diff) <= set(NEW), (rid, diff)
        for row, k in rk:
            out[row] = [nrid, k]
        print(f"{TAG}: {rid} -> {nrid} {'cached' if cached else 'new'} {[x for x, _ in rk]} diff {sorted(diff)}", flush=True)
    out = {row: out[row] for row in rows}
    json.dump(out, open(f"{E}/control/{TAG}.json", "w"), indent=1)
    open(f"{E}/control/{TAG}.watch", "w").write("".join(f"{rid} {k}\n" for rid, k in out.values()))
print("submitted", time.strftime("%H:%M:%S"))
