"""GAZE (10-23): sampling test for the 'generated pose too still' finding. Clones the X76 no-CFG request set (s20 control/xs17641_fs.json:
same 11 rows, clips, seeds, prefill, text/speech params, video settings, ckpt) changing ONLY pose sampling:
  gz_l500  : pose light (cb1-31) top-k 50 -> 500 (geom: light k250 still 0.20, off 0.06, GT 0.12)
  gz_l1000 : pose light top-k 50 -> 1000
Writes request dirs as server.py does (C.normalize, request.json + _submitted); row maps go to ppm/gaze-drift/samp/<tag>.json (not the s20 page)."""
import json, os, sys, time
sys.path.insert(0, "/data/home/amartyavarma/ppm/ppm-pose-server/code")
import srv_common as C
E = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20"
G = "/data/home/amartyavarma/ppm/gaze-drift/samp"
VARIANTS = {"gz_l500": {"pose_light_topk": 500}, "gz_l1000": {"pose_light_topk": 1000}}
rows = json.load(open(f"{E}/control/xs17641_fs.json"))
by_rid = {}
for row, (rid, k) in rows.items():
    by_rid.setdefault(rid, []).append((row, k))
for tag, NEW in VARIANTS.items():
    out = {}
    for rid, rk in by_rid.items():
        q = json.load(open(f"{C.REQ}/{rid}/request.json"))
        body = {k: v for k, v in q.items() if not k.startswith("_") and k not in ("frames_prefill", "frames_gen")}
        body.update(NEW)
        nrid, r = C.normalize(body)
        diff = {k: (q.get(k), r.get(k)) for k in set(q) | set(r) if not k.startswith("_") and q.get(k) != r.get(k)}
        assert set(diff) == set(NEW), (rid, diff)
        d = C.rdir(nrid)
        cached = os.path.exists(f"{d}/request.json")
        if not cached:
            os.makedirs(d, exist_ok=True)
            C.write_json(f"{d}/request.json", {**r, "_submitted": time.time()})
        for row, k in rk:
            out[row] = [nrid, k]
        print(f"{tag}: {rid} -> {nrid} {'cached' if cached else 'new'} {[x for x, _ in rk]} diff {diff}", flush=True)
    json.dump({row: out[row] for row in rows}, open(f"{G}/{tag}.json", "w"), indent=1)
print("submitted", time.strftime("%H:%M:%S"))
