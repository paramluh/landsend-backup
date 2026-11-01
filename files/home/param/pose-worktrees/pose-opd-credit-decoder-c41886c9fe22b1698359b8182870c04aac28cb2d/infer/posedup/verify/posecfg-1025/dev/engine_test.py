#!/usr/bin/env python3
"""pose-CFG dev test (10-25), RANDOM WEIGHTS with the export-498 architecture (the checkpoint lives on slinky-10 only): one process, one model,
OLD engine (live_ppm_pose.py.bak-pre-posecfg-1025) vs NEW engine (stage-posecfg/live_ppm_pose.py, pose_cfg_rows) on the same prefix (a real
training system block: voxsamples convs/c1.npz), the same A audio and seed:
  1. new (pose CFG off) == old, byte for byte: B lane tokens, B Mimi codes, pose codes, B audio; defaults AND audio CFG on (rows 0 / 1 intact);
  2. pose CFG on (s 3, cb0 + light): runs, finite, pose codes differ; switch on mid-session at frame SW: identical before SW;
  3. graph == eager with pose CFG on; audio CFG + pose CFG both on; cb0-only scope;
  4. new module with pose_cfg_rows=0 == old;
  5. per-frame latency (step + result sync, the server's ms) for each config.
usage: python engine_test.py <ci.pkl> <out json> [frames]"""
import importlib.util
import json
import math
import pickle
import sys
import time

STAGE = "/data/home/amartyavarma/ppm/aloe-posedup-live/stage-posecfg"
OLD = "/data/home/amartyavarma/ppm/aloe-posedup-live/live_ppm_pose.py.bak-pre-posecfg-1025"
sys.path.insert(0, STAGE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import live_ppm_pose as LPn  # noqa: E402

spec = importlib.util.spec_from_file_location("live_ppm_pose_old", OLD, loader=importlib.machinery.SourceFileLoader("live_ppm_pose_old", OLD))
LPo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LPo)

CI, OUT = sys.argv[1], sys.argv[2]
NF = int(sys.argv[3]) if len(sys.argv) > 3 else 240
SW = 60
CONV = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs/c1.npz"
log = LPn.log
SPF = LPn.SPF


@torch.no_grad()
def rand_fill(m, ci, adapter):
    g = {}
    for n, p in m.named_parameters():
        d = p.device
        if d not in g:
            g[d] = torch.Generator(device=d); g[d].manual_seed(1234 + (d.index or 0))
        if n.startswith("vgain."):
            p.fill_(math.log(0.0038))
        elif "norm" in n:
            p.fill_(1.0)
        elif n.endswith(".bias"):
            p.zero_()
        else:
            t = torch.empty(p.shape, dtype=torch.float32, device=d).normal_(0.0, 0.02, generator=g[d])
            p.copy_(t.to(p.dtype))
    log(f"[rand_fill] {sum(1 for _ in m.parameters())} tensors")


def run(eng, pre, A, params, n, graphs=True, live=None):
    """the server's per-frame loop (step + result; ms = step -> result sync); live = (frame, params) applied before that frame's step."""
    eng.start(pre, 4242, params)
    eng.write_a(0, A[:(n + 2) * SPF + 72])
    out = {"tok": [], "mimi": [], "pose": [], "y": [], "ms": []}
    for f in range(n):
        if live and f == live[0]:
            eng.set_params(live[1])
        T = eng.P + f
        t1 = time.perf_counter()
        dec = eng.step(T, graphs=graphs)
        tok, pp, pn, y = eng.result(T, dec)
        out["ms"].append((time.perf_counter() - t1) * 1e3)
        assert math.isfinite(pp) and math.isfinite(pn) and (y is None or np.isfinite(y).all()), f
        out["tok"].append(tok); out["mimi"].append(list(eng.last_codes)); out["pose"].append(list(eng.last_pose))
        out["y"].append(None if y is None else y.copy())
    out["yb"] = b"".join(y.astype(np.float32).tobytes() for y in out["y"] if y is not None)
    return out


def cmp(a, b, n0=0):
    r = {}
    for k in ("tok", "mimi", "pose"):
        x, y = np.asarray(a[k][n0:]), np.asarray(b[k][n0:])
        neq = np.flatnonzero((x != y).reshape(len(x), -1).any(1)) if x.shape == y.shape else [0]
        r[k] = {"identical": bool(x.shape == y.shape and (x == y).all()), "first_diff_frame": (int(neq[0]) + n0) if len(neq) else None,
                "frames_differing": int(len(neq))}
    r["audio_identical"] = a["yb"] == b["yb"] if n0 == 0 else None
    return r


def trunc(o, n):
    r = {k: v[:n] for k, v in o.items() if k != "yb"}
    r["yb"] = b"".join(y.astype(np.float32).tobytes() for y in r["y"] if y is not None)
    return r


def lat(o, skip=5):
    ms = np.asarray(o["ms"][skip:])
    return {"p50": round(float(np.percentile(ms, 50)), 2), "p95": round(float(np.percentile(ms, 95)), 2), "max": round(float(ms.max()), 2),
            "mean": round(float(ms.mean()), 2), "rtf": round(float(ms.mean()) / 80.0, 3)}


@torch.no_grad()
def main():
    t0 = time.time()
    ci = pickle.load(open(CI, "rb"))
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    for L in (LPn, LPo):
        L.ckpt_info = lambda p, ci=ci: dict(ci)
        L.load_weights = rand_fill
    m, ldev, cfgs, info = LPn.build_model(ci["path"], devs)
    z = np.load(CONV)
    pre = {"P": int(z["pre_ids_0"].shape[1]), "ids": z["pre_ids_0"], "depth": z["pre_depth_0"], "present": z["pre_present_0"]}
    A = z["stem_1"].astype(np.float32)
    log(f"prefix P {pre['P']}, B prompt codes {(pre['depth'][1, :, 0] >= 0).sum()}, A prompt codes {(pre['depth'][0, :, 0] >= 0).sum()}, A {len(A) / 24000:.0f} s")
    res = {"info": {k: v for k, v in info.items() if k != "ck"}, "frames": NF, "switch_frame": SW}
    CFGA = {"cfg_on": 1, "cfg_s": 2.0, "cfg_cb0": 1}
    PCFG = {"pose_cfg_on": 1, "pose_cfg_s": 3.0, "pose_cfg_cb0": 1, "pose_cfg_light": 1}

    eo = LPo.LivePPM(m, ldev, cfgs, devs, max_s=600.0)
    eo.capture()
    mem_old = {str(d): round(torch.cuda.memory_allocated(d) / 2 ** 30, 2) for d in devs}
    o_def = run(eo, pre, A, {}, NF)
    o_def2 = run(eo, pre, A, {}, NF)
    o_cfga = run(eo, pre, A, CFGA, NF)
    res["old"] = {"repeat_identical": cmp(o_def, o_def2), "lat_default": lat(o_def), "lat_audio_cfg": lat(o_cfga), "mem_gib_after": mem_old}
    log(f"OLD done: {res['old']}")
    del eo
    torch.cuda.empty_cache()

    en = LPn.LivePPM(m, ldev, cfgs, devs, max_s=600.0)
    assert en.pcfg and en.NR == 3
    en.capture()
    res["new_graphs"] = list(en.graphs)
    res["new_mem_gib"] = {str(d): round(torch.cuda.memory_allocated(d) / 2 ** 30, 2) for d in devs}
    n_def = run(en, pre, A, {}, NF)
    n_cfga = run(en, pre, A, CFGA, NF)
    n_on = run(en, pre, A, PCFG, NF)
    n_on2 = run(en, pre, A, PCFG, NF)
    n_sw = run(en, pre, A, {}, NF, live=(SW, PCFG))
    n_both = run(en, pre, A, dict(CFGA, **PCFG), NF)
    n_cb0 = run(en, pre, A, dict(PCFG, pose_cfg_light=0), NF)
    n_s1 = run(en, pre, A, dict(PCFG, pose_cfg_s=1.0), NF)
    n_eag = run(en, pre, A, PCFG, min(NF, 80), graphs=False)
    n_def_after = run(en, pre, A, {}, NF)
    pd = np.asarray(n_on["pose"]) != np.asarray(n_def["pose"])
    res["new"] = {
        "off_vs_old_default": cmp(n_def, o_def), "off_vs_old_audio_cfg": cmp(n_cfga, o_cfga),
        "off_again_after_cfg_runs_vs_old": cmp(n_def_after, o_def),
        "on_repeat_identical": cmp(n_on, n_on2), "on_vs_off": cmp(n_on, n_def),
        "pose_codes_differ_on_vs_off": {"frames_any_code_differs": int(pd.any(1).sum()), "cb0_differs": int(pd[:, 0].sum()), "of": len(pd)},
        "switch_at_SW_vs_off": cmp(n_sw, n_def), "switch_vs_on_from_start": cmp(n_sw, n_on),
        "graph_vs_eager_on": cmp(trunc(n_on, len(n_eag["tok"])), n_eag),
        "both_cfg_vs_pose_only": cmp(n_both, n_on), "cb0_only_vs_off": cmp(n_cb0, n_def),
        "s1_vs_off (w=1: lu + 1 (lc - lu) is lc up to fp32 rounding)": cmp(n_s1, n_def),
        "lat_off": lat(n_def), "lat_audio_cfg": lat(n_cfga), "lat_pose_cfg_on": lat(n_on), "lat_pose_cfg_cb0_only": lat(n_cb0),
        "lat_both_cfg": lat(n_both)}
    log(f"NEW done: {json.dumps(res['new'], default=str)}")
    del en
    torch.cuda.empty_cache()

    e2 = LPn.LivePPM(m, ldev, cfgs, devs, max_s=600.0, pose_cfg_rows=False)
    assert not e2.pcfg and e2.NR == 2
    e2.capture()
    r2 = run(e2, pre, A, {}, NF)
    _, bad = e2.set_params({"pose_cfg_on": 1})
    res["rows0"] = {"vs_old_default": cmp(r2, o_def), "pose_cfg_on_problem": bad, "lat": lat(r2)}
    del e2
    ok = (res["old"]["repeat_identical"]["audio_identical"] and all(res["new"]["off_vs_old_default"][k]["identical"] for k in ("tok", "mimi", "pose"))
          and res["new"]["off_vs_old_default"]["audio_identical"] and res["new"]["off_vs_old_audio_cfg"]["audio_identical"]
          and res["new"]["off_again_after_cfg_runs_vs_old"]["audio_identical"]
          and res["new"]["on_repeat_identical"]["audio_identical"] and res["new"]["pose_codes_differ_on_vs_off"]["frames_any_code_differs"] > 0
          and res["new"]["switch_at_SW_vs_off"]["pose"]["first_diff_frame"] is not None and res["new"]["switch_at_SW_vs_off"]["pose"]["first_diff_frame"] >= SW
          and res["rows0"]["vs_old_default"]["audio_identical"])
    res["PASS"] = bool(ok)
    res["s"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    log(f"ENGINE TEST {'PASS' if ok else 'FAIL'} -> {OUT}")


if __name__ == "__main__":
    main()
