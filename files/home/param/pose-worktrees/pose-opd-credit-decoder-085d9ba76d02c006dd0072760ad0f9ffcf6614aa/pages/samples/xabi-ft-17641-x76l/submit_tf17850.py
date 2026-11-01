"""TF17850 (10-23 ~08:50 PT, user via main): GT TEACHER-FORCED pose samples from the 17,850 monoplex FT (aloe-v2pure-main-17850 export-200,
node-local /scratch/amartyavarma/xabi-samples/w17850/export-200) on the page's 11 clips with the XJ4 requests' clips / seeds / voice clips (rule)
/ no persona / video settings; 2 s prefill + 10 s (12 s = the GT window: the clip packages hold 12 s of GT; main OK 10-23). gt_force: B's word
lane AND Mimi codes are teacher-forced to the clip's GT (rows/<id>.gt12.npz) on every generated frame; only the pose is sampled.
Text / speech sampler settings = xs17850_n150's (they never act: both are forced). Three runs:
  xs17850_tf_k150   NO CFG: temp 1; pose cb0 top-k 50, pose cb1-31 top-k 150 (= xs17850_n150's pose settings); no rep-pen; pose_cfg 0, audio_cfg 0.
  xs17850_tf_k50c2  temp 1; pose cb0 top-k 50, pose cb1-31 top-k 50; pose CFG 2.0 on all 32 pose codebooks; uncond = XJ4's rule 'silence'
                    (every Mimi input -> digital-silence codes, lane kept).
  xs17850_tf_k50c4  the same with pose CFG 4.0.
Writes request dirs as server.py does + s20 control/<tag>.json ({row: [rid, k]}) + .watch.  (stdlib; run on a compute node: ckpt_ok)"""
import json, os, sys, time
sys.path.insert(0, "/data/home/amartyavarma/ppm/ppm-pose-server/code")
import srv_common as C
E = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20"
SRC = "xs17641_j4"
CK = "/scratch/amartyavarma/xabi-samples/w17850/export-200"
PLAIN = {"factorised_text": 0, "cat_temp": 1.0, "text_temp": 1.0, "text_topk": 0, "text_topp": 1.0, "min_p": 0.0, "freq_pen": 0.0, "pad_bias": 0.0,
         "text_rep": 1.0}
POSE_N150 = {"pose_temp": 1.0, "pose_topp": 1.0, "pose_light_temp": 1.0, "pose_light_topp": 1.0, "pose_rep": 1.0, "pose_light_rep": 1.0, "pose_rep_win": 0}
SPEECH_N150 = {**PLAIN, "cb0_temp": 1.0, "cb0_topk": 50, "cb0_topp": 1.0, "light_temp": 1.0, "light_topk": 150, "light_topp": 1.0, "audio_cfg": 0.0}
TF = {"ckpt": CK, "gt_force": True, "gen_s": 10.0, **SPEECH_N150, **POSE_N150}
RUNS = {"xs17850_tf_k150": {**TF, "pose_topk": 50, "pose_light_topk": 150, "pose_cfg": 0.0},
        "xs17850_tf_k50c2": {**TF, "pose_topk": 50, "pose_light_topk": 50, "pose_cfg": 2.0, "pose_cfg_scope": "all", "cfg_uncond": "silence"},
        "xs17850_tf_k50c4": {**TF, "pose_topk": 50, "pose_light_topk": 50, "pose_cfg": 4.0, "pose_cfg_scope": "all", "cfg_uncond": "silence"}}
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
        assert r["gt_force"] is True and r["prefill_s"] == 2.0 and r["gen_s"] == 10.0 and r["frames_prefill"] == 25 and r["frames_gen"] == 125, TAG
        assert (r["cb0_topk"], r["light_topk"], r["pose_temp"], r["pose_light_temp"], r["pose_topk"], r["pose_rep"], r["pose_light_rep"],
                r["pose_topp"], r["pose_light_topp"]) == (50, 150, 1.0, 1.0, 50, 1.0, 1.0, 1.0, 1.0), TAG
        if TAG == "xs17850_tf_k150":
            assert (r["pose_light_topk"], r["pose_cfg"]) == (150, 0.0), TAG
        else:
            assert (r["pose_light_topk"], r["pose_cfg"], r["pose_cfg_scope"], r["cfg_uncond"]) == \
                (50, {"xs17850_tf_k50c2": 2.0, "xs17850_tf_k50c4": 4.0}[TAG], "all", "silence"), TAG
            assert q["cfg_uncond"] == "silence" and q["pose_cfg_scope"] == "all", "uncond rule = XJ4's"
        for k in ("video", "stick", "video_stab_steps", "video_res_steps", "video_seed", "voice_clip", "persona", "prefill_s", "n_samples"):
            assert r[k] == q[k], (TAG, k)
        d = C.rdir(nrid)
        cached = os.path.exists(f"{d}/request.json")
        if not cached:
            os.makedirs(d, exist_ok=True)
            C.write_json(f"{d}/request.json", {**r, "_submitted": time.time()})
        diff = {k: (q.get(k), r.get(k)) for k in set(q) | set(r) if not k.startswith("_") and q.get(k) != r.get(k)}
        assert set(diff) <= set(NEW) | {"frames_gen"}, (rid, diff)
        for row, k in rk:
            out[row] = [nrid, k]
        print(f"{TAG}: {rid} -> {nrid} {'cached' if cached else 'new'} {[x for x, _ in rk]} diff {sorted(diff)}", flush=True)
    out = {row: out[row] for row in rows}
    json.dump(out, open(f"{E}/control/{TAG}.json", "w"), indent=1)
    open(f"{E}/control/{TAG}.watch", "w").write("".join(f"{rid} {k}\n" for rid, k in out.values()))
print("submitted", time.strftime("%H:%M:%S"))
