"""K60L180 (10-23 20:43 PT, user via main -75: "rollouts same as before but top-k audio light 180") = xs17850_k60 EXACTLY except speech cb1-31
top-k 60 -> 180 (cb0 stays 60). Was K60 (10-23 ~20:10 PT, user via main): FREE pose rollouts from the 17,850 monoplex FT (aloe-v2pure-main-17850 export-200, node-local copy
/scratch/amartyavarma/xabi-samples/w17850/export-200) on the page's 11 clips with the XJ4 requests' clips / seeds / voice clips (rule) / no persona /
2 s prefill + 18 s / video settings = submit_m17850.py with ONLY the sampling dict changed. One run:
  xs17850_k60  temp 1 on every stream; pose cb0 top-k 60, pose cb1-31 top-k 60; speech cb0 top-k 60, cb1-31 top-k 60; text joint draw (factorised 0
               as xs17850_n150) T1 top-k 60 min-p 0.05 (absolute, word choice), no top-p / freq pen / PAD bias / rep-pen; no pose rep-pen; no CFG.
Writes request dirs as server.py does + s20 control/<tag>.json ({row: [rid, k]}) + .watch.  (stdlib; run on a compute node: ckpt_ok)"""
import json, os, sys, time
sys.path.insert(0, "/data/home/amartyavarma/ppm/ppm-pose-server/code")
import srv_common as C
E = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20"
SRC = "xs17641_j4"
CK = "/scratch/amartyavarma/xabi-samples/w17850/export-200"
PLAIN = {"factorised_text": 0, "cat_temp": 1.0, "text_temp": 1.0, "text_topk": 0, "text_topp": 1.0, "min_p": 0.0, "freq_pen": 0.0, "pad_bias": 0.0,
         "text_rep": 1.0}
TEXT = {**PLAIN, "text_topk": 60, "min_p": 0.05}
RUNS = {"xs17850_k60l180": {"ckpt": CK, **TEXT, "cb0_temp": 1.0, "cb0_topk": 60, "cb0_topp": 1.0, "light_temp": 1.0, "light_topk": 180, "light_topp": 1.0,
                        "pose_temp": 1.0, "pose_topk": 60, "pose_topp": 1.0, "pose_light_temp": 1.0, "pose_light_topk": 60, "pose_light_topp": 1.0,
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
        for k, v in {**TEXT, "cb0_temp": 1.0, "light_temp": 1.0, "audio_cfg": 0.0}.items():
            assert r[k] == v, (TAG, k, r[k], v)
        assert r["ckpt"] == CK and r["persona"] == "" and r["voice_clip"] == "rule" and r["seed"] == q["seed"] and r["clip"] == q["clip"]
        assert (r["pose_temp"], r["pose_topk"], r["pose_topp"], r["pose_light_temp"], r["pose_light_topk"], r["pose_light_topp"], r["pose_rep"],
                r["pose_light_rep"], r["pose_rep_win"], r["pose_cfg"], r["cb0_topk"], r["cb0_topp"], r["light_topk"], r["light_topp"]) == \
            (1.0, 60, 1.0, 1.0, 60, 1.0, 1.0, 1.0, 0, 0.0, 60, 1.0, 180, 1.0), TAG
        assert r["prefill_s"] == 2.0 and r["gen_s"] == 18.0 and "gt_force" not in r, TAG
        for k in ("video", "stick", "video_stab_steps", "video_res_steps", "video_seed", "voice_clip", "persona", "prefill_s", "gen_s", "n_samples"):
            assert r[k] == q[k], (TAG, k)
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
