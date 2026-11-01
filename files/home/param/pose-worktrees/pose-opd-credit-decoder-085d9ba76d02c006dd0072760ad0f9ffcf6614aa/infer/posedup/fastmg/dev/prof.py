#!/usr/bin/env python3
"""PER-FRAME BREAKDOWN of a live engine on REAL weights (fastmg dev). Same per-frame loop as server.py (step + result; ms = step -> result).
usage: prof.py <engine: cur | mg> <ck> <out json> [frames] [extra json kwargs for the engine]
  cur = the LIVE engine module (aloe-posedup-live/live_ppm_pose.py, imported read-only), 2 GPUs exactly as server.py builds it.
  mg  = the staged multi-GPU engine (fastmg/live_ppm_pose_mg.py), every visible GPU.
Prefix + mic audio = a real training conversation (poseduplex samples convs/c1.npz: its system block, A = its stem 1), seed 4242.
Per graph: device time (CUDA events right around each replay) and its start / end offset within the frame (per-device frame-start event:
every stream is idle at step start, so the offsets share one time base up to launch skew)."""
import json
import os
import sys
import time

FM = "/data/home/amartyavarma/ppm/aloe-posedup-live/fastmg"
DEMO = "/data/home/amartyavarma/ppm/aloe-posedup-live"
WHICH, CK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NF = int(sys.argv[4]) if len(sys.argv) > 4 else 600
KW = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
sys.path.insert(0, FM if WHICH == "mg" else DEMO)
import numpy as np  # noqa: E402
import torch  # noqa: E402
if WHICH == "mg":
    import live_ppm_pose_mg as LP  # noqa: E402
else:
    import live_ppm_pose as LP  # noqa: E402
CONV = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs/c1.npz"
SPF = LP.SPF
log = LP.log


class TG:
    """graph wrapper: CUDA events around replay() on the current stream of the current device."""

    def __init__(self, g, name, rec):
        self.g, self.name, self.rec = g, name, rec

    def replay(self):
        if self.rec.on:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            self.g.replay()
            e1.record()
            self.rec.ev.append((self.name, torch.cuda.current_device(), e0, e1))
        else:
            self.g.replay()


class Rec:
    on = False
    ev = []


def run(eng, pre, A, params, n, f0=0, rec=None, devs=()):
    eng.start(pre, 4242, params)
    eng.write_a(0, A[:(n + f0 + 2) * SPF + 72] if f0 == 0 else np.zeros((n + f0 + 2) * SPF + 72, np.float32))
    out = {"tok": [], "mimi": [], "pose": [], "ms": [], "graphs": []}
    if f0:                                                 # late-context latency probe: skip to conversation frame f0 (KV content irrelevant)
        eng.vx.k = f0
    for f in range(f0, f0 + n):
        T = eng.P + f
        st = {}
        if rec is not None:
            rec.ev = []
            for d in devs:
                with torch.cuda.device(d):
                    e = torch.cuda.Event(enable_timing=True); e.record(); st[d.index] = e
        t1 = time.perf_counter()
        dec = eng.step(T)
        tok, pp, pn, y = eng.result(T, dec)
        out["ms"].append((time.perf_counter() - t1) * 1e3)
        out["tok"].append(tok); out["mimi"].append(list(eng.last_codes)); out["pose"].append(list(eng.last_pose))
        if rec is not None:
            for d in devs:
                torch.cuda.synchronize(d)
            out["graphs"].append([(nm, di, round(st[di].elapsed_time(e0), 3), round(st[di].elapsed_time(e1), 3), round(e0.elapsed_time(e1), 3))
                                  for nm, di, e0, e1 in rec.ev])
    return out


def lat(ms, skip=5):
    ms = np.asarray(ms[skip:])
    return {"n": int(len(ms)), "p50": round(float(np.percentile(ms, 50)), 2), "p95": round(float(np.percentile(ms, 95)), 2),
            "p99": round(float(np.percentile(ms, 99)), 2), "max": round(float(ms.max()), 2), "mean": round(float(ms.mean()), 2)}


def gsum(graphs, skip=5):
    """per graph name: median device ms, median start / end offset in the frame."""
    agg = {}
    for fr in graphs[skip:]:
        for nm, di, s, e, d in fr:
            a = agg.setdefault(f"{nm}@{di}", {"dur": [], "start": [], "end": []})
            a["dur"].append(d); a["start"].append(s); a["end"].append(e)
    return {k: {kk: round(float(np.median(v)), 3) for kk, v in a.items()} for k, a in sorted(agg.items(), key=lambda x: np.median(x[1]["start"]))}


@torch.no_grad()
def main():
    t0 = time.time()
    n = torch.cuda.device_count()
    devs = [torch.device(f"cuda:{i}") for i in range(n)]
    if WHICH == "cur":
        devs = devs[:2]
        m, ldev, cfgs, info = LP.build_model(CK, devs)
        eng = LP.LivePPM(m, ldev, cfgs, devs, max_s=600.0)
    else:
        m, ldev, cfgs, info = LP.build_model(CK, devs, **{k: v for k, v in KW.items() if k in ("tp",)})
        eng = LP.LivePPM(m, ldev, cfgs, devs, max_s=600.0, **{k: v for k, v in KW.items() if k not in ("tp",)})
    eng.capture()
    rec = Rec()
    for k in list(eng.graphs):
        eng.graphs[k] = TG(eng.graphs[k], k if isinstance(k, str) else str(k), rec)
    z = np.load(CONV)
    pre = {"P": int(z["pre_ids_0"].shape[1]), "ids": z["pre_ids_0"], "depth": z["pre_depth_0"], "present": z["pre_present_0"]}
    A = z["stem_1"].astype(np.float32)
    log(f"prefix P {pre['P']}, A {len(A) / 24000:.0f} s, engine {type(eng).__module__} graphs {list(eng.graphs)}")
    res = {"which": WHICH, "ck": CK, "info": {k: v for k, v in info.items() if k != "ck"}, "P": pre["P"], "frames": NF,
           "mem_gib": {str(d): round(torch.cuda.memory_allocated(d) / 2 ** 30, 2) for d in devs}}
    PCFG = {"pose_cfg_on": 1, "pose_cfg_s": 3.0, "pose_cfg_cb0": 1, "pose_cfg_light": 1}
    o_off = run(eng, pre, A, {}, NF)
    res["off"] = lat(o_off["ms"])
    log(f"pose CFG off: {res['off']}")
    o_on = run(eng, pre, A, PCFG, NF)
    res["on"] = lat(o_on["ms"])
    log(f"pose CFG on: {res['on']}")
    rec.on = True
    o_g = run(eng, pre, A, {}, 120, rec=rec, devs=devs)
    res["graphs_off"] = gsum(o_g["graphs"]); res["timed_off"] = lat(o_g["ms"])
    o_g2 = run(eng, pre, A, PCFG, 120, rec=rec, devs=devs)
    res["graphs_on"] = gsum(o_g2["graphs"]); res["timed_on"] = lat(o_g2["ms"])
    late = int(KW.get("late", 7300))
    o_l = run(eng, pre, A, PCFG, 60, f0=late, rec=rec, devs=devs)
    res[f"graphs_on_f{late}"] = gsum(o_l["graphs"]); res[f"timed_on_f{late}"] = lat(o_l["ms"])
    rec.on = False
    o_l2 = run(eng, pre, A, PCFG, 100, f0=late)
    res[f"on_f{late}"] = lat(o_l2["ms"])
    o_l3 = run(eng, pre, A, {}, 100, f0=late)
    res[f"off_f{late}"] = lat(o_l3["ms"])
    np.savez(OUT.replace(".json", ".npz"), tok_off=np.asarray(o_off["tok"]), mimi_off=np.asarray(o_off["mimi"]), pose_off=np.asarray(o_off["pose"]),
             tok_on=np.asarray(o_on["tok"]), mimi_on=np.asarray(o_on["mimi"]), pose_on=np.asarray(o_on["pose"]),
             ms_off=np.asarray(o_off["ms"]), ms_on=np.asarray(o_on["ms"]))
    words = " ".join(w["text"] for w in LP.IP.lane_words(np.asarray(o_off["tok"])))
    res["words_off"] = words[:600]
    res["pose_frames_off"] = int((np.asarray(o_off["pose"])[:, 0] >= 0).sum())
    res["s"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    log(f"PROF DONE -> {OUT}: off {res['off']} on {res['on']}")


if __name__ == "__main__":
    main()
