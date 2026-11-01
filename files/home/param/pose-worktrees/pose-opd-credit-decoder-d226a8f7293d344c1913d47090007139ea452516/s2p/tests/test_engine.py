#!/usr/bin/env python3
"""pose-s2p engine tests (a) exactness at init, (b) wiring, (d) latency. ONE process, the deployed genweb-worker layout: 2 GPUs,
export-357, FastPPM(n_inst=1, cfg_rows=True, max_s = 120 + 11.68 + 2, mimi_stream=False, silence = Mimi(zeros) as worker.py), the
worker's DEF params (fantastic set, pose CFG off). Engines share ONE copy of the model weights:
  orig = the ORIGINAL deployed engine module (aloe-pose-monoplex-infer/ppm/fast_ppm.py, loaded under another module name),
  off  = branch engine, flag off (the model has no s_emb: = every deployed checkpoint),
  full = branch engine, s2p 'full' (zero s_emb = the trainer's init), cb0 = branch engine, s2p 'cb0' (zero s_emb).
Every shared dependency file (model_ppm, ppm_io, light_exact_ppm, ...) is asserted byte-identical between the two dirs.
usage: test_engine.py [stages, default a,b,d]"""
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

BR = "/data/home/amartyavarma/ppm/pose-s2p/infer/ppm"
OR = "/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer/ppm"
MI = "/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer"
CK = "/data/home/amartyavarma/ppm/aloe-pose-mono-8020/exports/export-357"
G = "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb"
OUT = "/data/home/amartyavarma/ppm/pose-s2p/tests/out"
os.environ.pop("PPM_S2P", None)
sys.path.insert(0, BR)
import torch  # noqa: E402
import fast_ppm as FB  # noqa: E402

for fn in sorted(os.listdir(OR)):
    if fn.endswith(".py") and fn != "fast_ppm.py":
        a, b = (hashlib.md5(open(f"{d}/{fn}", "rb").read()).hexdigest() for d in (OR, BR))
        assert a == b, ("shared dependency differs", fn)
spec = importlib.util.spec_from_file_location("fast_ppm_orig", f"{OR}/fast_ppm.py")
ORIG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ORIG)
assert ORIG.M is FB.M and ORIG.IP is FB.IP
sys.path.insert(1, MI)
import infer as MIF  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

STAGES = (sys.argv[1] if len(sys.argv) > 1 else "a,b,d").split(",")
CD, OFF, PAD, DEPTH, BF = FB.CD, FB.OFF, FB.PAD, FB.DEPTH, FB.BF
RES = {"run1_eager_a": "tests/out/engine_results_run1_j30894.json (job 30894: (a) eager all 3 scenarios + graphs free / free_cfg PASS)",
       "t_start": time.strftime("%Y-%m-%d %H:%M:%S"), "host": os.uname().nodename}
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(f"[s2p-test {time.strftime('%H:%M:%S')}]", *a, flush=True)


def save():
    json.dump(RES, open(f"{OUT}/engine_results.json", "w"), indent=1, default=str)


DEF = dict(FB.DEFAULTS, **{k: v for k, v in MIF.TONIGHT.items() if k in FB.PARAMS})     # = genweb worker.py DEF
DEF.update(text_temp=1.0, text_topk=50, text_topp=1.0, min_p=0.05, min_p_rel=0, freq_pen=0.25, freq_win=50, cb0_temp=1.0, cb0_topk=30,
           light_temp=1.0, light_topk=30, pose_temp=1.0, pose_topk=30, pose_light_temp=1.0, pose_light_topk=151, pose_cfg_on=0)
DEF_CFG = dict(DEF, pose_cfg_on=1)

devs = [torch.device("cuda:0"), torch.device("cuda:1")]
torch.cuda.set_device(devs[-1])
RES["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
t0 = time.time()
with torch.inference_mode():
    m, ldev, cfgs, binfo = FB.build_model(CK, devs)
assert binfo["pose"] and binfo["decp_bb"] and not binfo["s2p"], binfo
assert not hasattr(m.tower_p, "s_emb")
H4 = cfgs[1].hidden_size
log(f"model loaded {time.time() - t0:.0f}s {binfo['layers_per_dev']}")


def mk_se(mode):
    se = FB.S2PEmb(H4, mode).to(device=devs[0], dtype=BF)
    with torch.no_grad():
        se.weight.zero_()
    se.weight.requires_grad_(False)
    return se


SE = {"full": mk_se("full"), "cb0": mk_se("cb0")}
MAXS = 120.0 + 11.68 + 2
NFR = int(round(MAXS * 12.5)) + 8
with torch.inference_mode():
    mimi = FB.IP.Mimi(devs[-1])
    SIL = mimi.encode(np.zeros((2048 + NFR) * 1920, np.float32))[:, :2048 + NFR]
    E = {}
    for name, mod, kw in (("orig", ORIG, {}), ("off", FB, {}), ("full", FB, {"s2p": SE["full"]}), ("cb0", FB, {"s2p": SE["cb0"]})):
        E[name] = mod.FastPPM(m, ldev, cfgs, devs, n_inst=1, cfg_rows=True, max_s=MAXS, mimi_stream=False, silence=SIL, **kw)
        E[name].capture()
        assert (getattr(E[name], "s2p", None) or None) == (kw["s2p"].mode if kw else None)
RES["mem_gib"] = {str(d): round(torch.cuda.max_memory_allocated(d) / 2 ** 30, 1) for d in devs}
log("engines", list(E), RES["mem_gib"])

# ------------------------------------------------------------------------------------------------ instances (= worker.py run())
PERSON = "01_ab63d745"
od = f"{G}/openings/{PERSON}"
APO = np.asarray(load_file(f"{od}/mimi_local.safetensors")["codes"], np.int64)
CPO = np.asarray(load_file(f"{od}/pose_tokens.safetensors")["codes"], np.int64)
TN = min(APO.shape[1], CPO.shape[1])
CLIP = APO[:, TN - 125:TN]                                           # a 10 s voice clip of the same speaker (end of the opening)
SP_ = MIF.system_block(CLIP, None)
WORDS = MIF.read_words(f"{od}/words.json")
RES["inst"] = {"person": PERSON, "opening_frames": int(TN), "clip_frames": 125}


def inst_free(n):
    x = MIF.assemble(SP_, np.full(n + CD, PAD, np.int64), np.zeros((DEPTH, 0), np.int64), np.zeros((DEPTH, 0), np.int64), n, 0)
    return x, None, True, n                                          # (inst, force fn, cold_pad, frames)


def inst_poseonly(n):
    """= worker.py pose only (no pose prefill): real speech codes + timed-word lane forced every frame, pose generated."""
    n = min(n, TN)
    pa = APO[:, :n].T.copy()
    lane_f = MIF.lane_from_words(WORDS, max(n + CD, 152) + 64)[:n + CD]
    x = MIF.assemble(SP_, np.full(n + CD, PAD, np.int64), np.zeros((DEPTH, 0), np.int64), np.zeros((DEPTH, 0), np.int64), n, 0)
    P = int(x["P"])
    lane_ids = np.asarray(MIF.assemble(SP_, lane_f, np.zeros((DEPTH, 0), np.int64), np.zeros((DEPTH, 0), np.int64), n, 0)["ids"][1], np.int64)

    def frc(f):
        g = f - CD
        d = {"lane": lane_ids[P + f:P + f + 1]}
        if 0 <= g < n:
            d["codes"] = (pa[g] + OFF)[None]
        return d
    return x, frc, False, n                                          # n <= the opening's real-speech length (TN frames)


LAT = ("hv_p", "zp0", "hv_s", "z0", "hn_prev")


def run(eng, inst, n, seed, params, graphs, extra=None, rec=True):
    """one session; extra(f, d) may edit the force dict of frame f. -> (readback, {latent: [frames, ...] cpu})."""
    x, frc, cold, n_ = inst
    n = min(n, n_)
    with torch.inference_mode():
        eng.start([dict(x, a_mode="empty")], seed, params, f0=0)
        L = {k: [] for k in LAT + ("sc",)}
        for f in range(n + CD):
            d = frc(f) if frc else None
            if extra is not None:
                d = extra(f, dict(d or {})) or None
            eng.step(f, graphs=graphs, force=d, cold_pad=cold)
            if rec:
                for k in LAT:
                    L[k].append(getattr(eng, k).clone())
                if getattr(eng, "s2p", None):
                    L["sc"].append(eng.sc_used.clone())
        S = eng.readback(0, int(x["P"]) + n + CD)
        L = {k: torch.stack(v).cpu() for k, v in L.items() if v}
    S["P"] = int(x["P"])
    return S, L


def diff(A, B):
    """-> {key: n differing elements (arrays) / max abs diff (latents)}; all zero = identical."""
    (SA, LA), (SB, LB) = A, B
    out = {}
    for k in ("lane_b", "dep_b", "pdep"):
        out[k] = int((SA[k] != SB[k]).sum())
    for k in LAT:
        out[k] = 0.0 if torch.equal(LA[k], LB[k]) else float((LA[k].float() - LB[k].float()).abs().max())
    return out


def same(dd):
    return all(v == 0 for v in dd.values())


SEED = 1234
SCEN = {"free": (inst_free, DEF), "free_cfg": (inst_free, DEF_CFG), "poseonly": (inst_poseonly, DEF)}

# ------------------------------------------------------------------------------------------------ (a) exactness at init
if "a" in STAGES:
    RES["a"] = {}
    for mode, n in (("eager", 100), ("graphs", 150)):
        if mode not in os.environ.get("A_MODES", "eager,graphs").split(","):
            continue
        g = mode == "graphs"
        for sc, (mkinst, prm) in SCEN.items():
            inst = mkinst(n)
            n = inst[3]
            ref = run(E["orig"], inst, n, SEED, prm, g)
            r = {"floor_orig_vs_orig": diff(ref, run(E["orig"], inst, n, SEED, prm, g))}
            for name in ("off", "full", "cb0"):
                r[f"orig_vs_{name}"] = diff(ref, run(E[name], inst, n, SEED, prm, g))
            r["pass"] = all(same(v) for v in r.values() if isinstance(v, dict))
            P = ref[0]["P"]
            r["frames"] = n
            n = 150 if g else 100
            r["gen_pose_frames"] = int((ref[0]["pdep"][P + CD:P + CD + n, 0] >= 0).sum())
            r["gen_words"] = int(((ref[0]["lane_b"][P:P + n] < FB.TE)).sum())
            RES["a"][f"{mode}/{sc}"] = r
            log(f"(a) {mode}/{sc}: pass {r['pass']} " + json.dumps({k: v for k, v in r.items() if isinstance(v, dict)}))
            save()
    RES["a_pass"] = all(v["pass"] for v in RES["a"].values())
    log("(a) PASS" if RES["a_pass"] else "(a) FAIL")
    save()

# ------------------------------------------------------------------------------------------------ (b) wiring (non-zero s_emb)
if "b" in STAGES:
    gen = torch.Generator(device="cpu").manual_seed(20261027)
    with torch.no_grad():
        for mode, se in SE.items():
            w = torch.randn(se.weight.shape, generator=gen) * (0.02 / DEPTH ** 0.5)       # = the tower code_emb init scale
            se.weight.copy_(w.to(se.weight.device, BF))
    RES["b"] = {}
    for mode in ("eager", "graphs"):
        g = mode == "graphs"
        n = 120 if g else 60
        # b1: pose forced to the flag-off run's pose -> every speech / text output identical, pose logits differ
        inst = inst_free(n)
        ref = run(E["off"], inst, n, SEED, DEF, g)
        Pr = ref[0]["P"]
        pref = ref[0]["pdep"]

        def fpose(f, d, pref=pref, Pr=Pr):
            if f >= CD:
                d["pose"] = pref[Pr + f:Pr + f + 1]
            return d
        for name in ("full", "cb0"):
            got = run(E[name], inst, n, SEED, DEF, g, extra=fpose)
            dd = diff(ref, got)
            zd = (got[1]["zp0"][CD:, 0] != ref[1]["zp0"][CD:, 0]).any(-1)
            r = {"diff_vs_off": dd, "speech_text_trunk_identical": all(dd[k] == 0 for k in ("lane_b", "dep_b", "hv_s", "z0", "hn_prev")),
                 "pdep_identical": dd["pdep"] == 0, "pose_logit_frames_changed": int(zd.sum()), "of": int(zd.numel()),
                 "pose_logit_maxdiff": float((got[1]["zp0"][CD:, 0] - ref[1]["zp0"][CD:, 0]).abs().max())}
            r["pass"] = r["speech_text_trunk_identical"] and r["pdep_identical"] and r["pose_logit_frames_changed"] == r["of"]
            RES["b"][f"{mode}/b1_speech_invariant/{name}"] = r
            log(f"(b1) {mode} {name}: {json.dumps(r)}")
        # b2: perturb B's speech codes at frame k (pose-only run: speech forced) -> pose logits at k change, speech at k unchanged
        inst = inst_poseonly(n)
        k = min(CD + 40, inst[3] // 2)

        def pert(which):
            def fx(f, d):
                if f == k:
                    c = np.asarray(d["codes"]).copy()
                    if which in ("all", "cb0"):
                        c[0, 0] = (c[0, 0] + 777) % 2048
                    if which in ("all", "light"):
                        c[0, 1:] = OFF[1:] + (c[0, 1:] - OFF[1:] + 555) % 2048
                    d["codes"] = c
                return d
            return fx
        for name in ("off", "full", "cb0"):
            base = run(E[name], inst, n, SEED, DEF, g)
            for which in ("all", "cb0", "light"):
                got = run(E[name], inst, n, SEED, DEF, g, extra=pert(which))
                zb, zg = base[1]["zp0"][:, 0], got[1]["zp0"][:, 0]
                before = bool(torch.equal(zb[:k], zg[:k]))
                at_k = not torch.equal(zb[k], zg[k])
                after = not torch.equal(zb[k + 1], zg[k + 1])
                sp_k = bool(torch.equal(base[1]["z0"][:k + 1], got[1]["z0"][:k + 1]) and torch.equal(base[1]["hv_s"][:k + 1], got[1]["hv_s"][:k + 1]))
                exp_at_k = {"off": False, "full": True, "cb0": which in ("all", "cb0")}[name]
                r = {"pose_logits_identical_before_k": before, "pose_logits_changed_at_k": at_k, "expected_change_at_k": exp_at_k,
                     "pose_logits_changed_at_k+1 (via trunk)": after, "speech_logits_identical_through_k": sp_k,
                     "pose_logit_maxdiff_at_k": float((zb[k] - zg[k]).abs().max())}
                r["pass"] = before and at_k == exp_at_k and sp_k and after
                RES["b"][f"{mode}/b2_perturb_speech_T/{name}/{which}"] = r
                log(f"(b2) {mode} {name} perturb {which}: {json.dumps(r)}")
        save()
    # b3: uncond (pose CFG) rows read the Mimi codes of digital silence; instance rows read B's codes at T; prefill applies s_emb
    n = 60
    inst = inst_free(n)
    for name in ("full", "cb0"):
        S, L = run(E[name], inst, n, SEED, DEF_CFG, True)
        P = S["P"]
        nc = L["sc"].shape[-1]
        cond_ok = bool(np.array_equal(L["sc"][:, 0].numpy(), S["dep_b"][P:P + n + CD, :nc]))
        exp_u = np.full((n + CD, nc), -1, np.int64)
        exp_u[CD:] = (SIL[:, :n].T + OFF)[:, :nc]
        unc_ok = bool(np.array_equal(L["sc"][:, 1].numpy(), exp_u))
        # prefill: pose-tower KV vs the flag-off engine after start(): identical before the clip's first coded position, different after
        with torch.inference_mode():
            for e_ in (E["off"], E[name]):
                e_.start([dict(inst[0], a_mode="empty")], SEED, DEF_CFG, f0=0)
            kv = [(E["off"].tpK[-1][r, :P] != E[name].tpK[-1][r, :P]).flatten(1).any(-1).cpu().numpy() for r in (0, 1)]
        c0 = int(np.flatnonzero(inst[0]["depth"][1, :P, 0] >= 0)[0])
        pre_ok = [bool(not kv_[:c0].any() and kv_[c0:].any()) for kv_ in kv]
        r = {"cond_rows_read_B_codes_at_T": cond_ok, "uncond_rows_read_silence_codes": unc_ok, "first_clip_pos": c0, "P": P,
             "prefill_kv_differs_only_from_clip_on (cond, uncond)": pre_ok}
        r["pass"] = cond_ok and unc_ok and all(pre_ok)
        RES["b"][f"graphs/b3_rows/{name}"] = r
        log(f"(b3) {name}: {json.dumps(r)}")
    RES["b_pass"] = all(v["pass"] for v in RES["b"].values())
    log("(b) PASS" if RES["b_pass"] else "(b) FAIL")
    save()

# ------------------------------------------------------------------------------------------------ (d) latency (CUDA graphs, same GPUs)
if "d" in STAGES:
    RES["d"] = {}
    n = 300
    for sc in ("free", "free_cfg", "poseonly"):
        mkinst, prm = SCEN[sc]
        inst = mkinst(n)
        x, frc, cold, n_ = inst
        nn_ = min(n, n_)
        for name in ("orig", "full", "cb0"):
            eng = E[name]
            with torch.inference_mode():
                for rep in range(2):                                  # rep 0 warm-up
                    eng.start([dict(x, a_mode="empty")], SEED, prm, f0=0)
                    ts = []
                    for d_ in devs:
                        torch.cuda.synchronize(d_)
                    t0 = time.perf_counter()
                    for f in range(nn_ + CD):
                        t1 = time.perf_counter()
                        eng.step(f, graphs=True, force=frc(f) if frc else None, cold_pad=cold)
                        for d_ in devs:
                            torch.cuda.synchronize(d_)
                        ts.append(time.perf_counter() - t1)
                    tot_sync = time.perf_counter() - t0
                    eng.start([dict(x, a_mode="empty")], SEED, prm, f0=0)
                    for d_ in devs:
                        torch.cuda.synchronize(d_)
                    t0 = time.perf_counter()
                    for f in range(nn_ + CD):
                        eng.step(f, graphs=True, force=frc(f) if frc else None, cold_pad=cold)
                    for d_ in devs:
                        torch.cuda.synchronize(d_)
                    tot_async = time.perf_counter() - t0
            ts = np.asarray(ts[CD + 2:]) * 1e3
            r = {"sync_ms_median": round(float(np.median(ts)), 2), "sync_ms_p90": round(float(np.percentile(ts, 90)), 2),
                 "sync_ms_max": round(float(ts.max()), 2), "async_ms_per_frame": round(1e3 * tot_async / (nn_ + CD), 2), "frames": nn_ + CD}
            RES["d"][f"{sc}/{name}"] = r
            log(f"(d) {sc} {name}: {json.dumps(r)}")
        save()
RES["t_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
save()
log("done")
