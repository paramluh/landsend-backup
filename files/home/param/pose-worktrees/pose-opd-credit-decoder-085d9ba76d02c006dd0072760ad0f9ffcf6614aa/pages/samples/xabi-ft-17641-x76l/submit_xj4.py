"""XJ4 (10-22 22:15 PT, user via main): the pose-judge sweep's BEST setting cfg-cs4 on the hot 17,641 samples server. Each request = the XD3 request
(s20 control/xs17641_d3.json: same 11 rows / 10 requests, clips, seeds, 2 s prefill + 18 s, voice clip by rule, no persona, pose CFG on all 32 cb with
uncond = speech silenced, video 9 refs 40+10 steps / seed 2026, ckpt = FT-(b) export-200 node-local) with the sweep's cfg-cs4 knobs EXACTLY
(ppm/judge-sweep/settings/cfg-cs4.jsonl = the repo grid's F011cfg recipe at CFG 4.0; text/speech = the sweep engine's base: joint text draw T1, no
top-k / min-p / freq pen / PAD bias; speech T1 k30 / k30): pose cb0 T0.952 k250, pose cb1-31 T0.966 k75, pose rep 1 / light 1.022 window 50,
pose CFG 4.0. Writes request dirs as server.py does + s20 control/xs17641_j4.json ({row: [rid, k]}) + .watch.  (stdlib; run on slinky-15: ckpt_ok)"""
import json, os, sys, time
sys.path.insert(0, "/data/home/amartyavarma/ppm/ppm-pose-server/code")
import srv_common as C
E = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20"
SRC, TAG = "xs17641_d3", "xs17641_j4"
NEW = {"pose_topk": 250, "pose_light_topk": 75, "pose_temp": 0.952, "pose_light_temp": 0.966, "pose_rep": 1.0, "pose_light_rep": 1.022,
       "pose_rep_win": 50, "pose_cfg": 4.0, "factorised_text": 0, "text_topk": 0, "min_p": 0.0, "freq_pen": 0.0, "pad_bias": 0.0}
SW = json.loads(open("/data/home/amartyavarma/ppm/judge-sweep/settings/cfg-cs4.jsonl").readline())
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
    for k in ("pose_topk", "pose_light_topk", "pose_temp", "pose_light_temp", "pose_rep", "pose_light_rep", "pose_rep_win"):
        assert r[k] == SW[k], (k, r[k], SW[k])
    assert r["pose_cfg"] == SW["pose_cfg_s"] == 4.0 and r["pose_cfg_scope"] == "all" and r["cfg_uncond"] == "silence" and r["audio_cfg"] == 0
    assert r["text_temp"] == 1 and r["text_topp"] == 1 and r["text_rep"] == 1 and r["cat_temp"] == 1
    assert (r["cb0_temp"], r["cb0_topk"], r["light_temp"], r["light_topk"], r["cb0_topp"], r["light_topp"]) == (1, 30, 1, 30, 1, 1)
    assert r["persona"] == "" and r["voice_clip"] == "rule" and r["pose_topp"] == 1 and r["pose_light_topp"] == 1
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
