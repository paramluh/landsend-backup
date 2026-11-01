"""X76C1/C2/C3 (10-22 19:4x PT): the user's "do the topk 30 pose light but cfg 1, 2, 3 on pose" on the 17,641 Xabi FT samples. Each request =
the X76L request (control/xs17641_l250.json: same 11 rows, clips, seeds, voice clip, no persona, text fantastic set, speech cb0 k30 / light k250,
pose cb0 k30, video 9 refs) with ONLY pose light top-k 250 -> 30 and pose CFG 0 -> w (w = 1, 2, 3) on all 32 pose codebooks, uncond = silence
(every Mimi input of the uncond stream -> digital-silence codes, words kept = the established pose-CFG uncond of runs CF3/E3); audio CFG off."""
import json, os, sys, time
sys.path.insert(0, "/data/home/amartyavarma/ppm/ppm-pose-server/code")
import srv_common as C
E = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20"
SRC = "xs17641_l250"
FS = json.load(open(C.FANTASTIC))["params"]
rows = json.load(open(f"{E}/control/{SRC}.json"))
by_rid = {}
for row, (rid, k) in rows.items():
    by_rid.setdefault(rid, []).append((row, k))
for w in (3, 2, 1):
    TAG = f"xs17641_c{w}"
    NEW = {"pose_light_topk": 30, "pose_cfg": float(w)}
    out = {}
    for rid, rk in by_rid.items():
        q = json.load(open(f"{C.REQ}/{rid}/request.json"))
        body = {k: v for k, v in q.items() if not k.startswith("_") and k not in ("frames_prefill", "frames_gen")}
        body.update(NEW)
        body.update({"pose_cfg_scope": "all", "cfg_uncond": "silence", "audio_cfg": 0.0})
        nrid, r = C.normalize(body)
        for k in ("text_temp", "text_topk", "min_p", "freq_pen", "freq_win", "pad_bias", "factorised_text", "cat_temp", "text_rep", "cb0_temp", "cb0_topk", "light_temp"):
            assert r[k] == FS[k], (k, r[k], FS[k])
        assert r["light_topk"] == 250 and r["pose_light_topk"] == 30 and r["pose_topk"] == 30 and r["pose_temp"] == 1 and r["pose_light_temp"] == 1
        assert r["pose_cfg"] == w and r["pose_cfg_scope"] == "all" and r["cfg_uncond"] == "silence" and r["audio_cfg"] == 0
        assert r["pose_rep"] == 1 and r["pose_light_rep"] == 1 and r["persona"] == "" and r["voice_clip"] == "rule"
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
