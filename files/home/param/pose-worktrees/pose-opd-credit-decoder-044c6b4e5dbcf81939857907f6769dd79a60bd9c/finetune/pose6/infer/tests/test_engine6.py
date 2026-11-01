#!/usr/bin/env python3
"""pose6 A1 engine proofs, ONE process in the DEPLOYED layout (genweb worker.py): 2 GPUs, export-357 loaded ONCE (37 + 27 backbone layers,
pose branch on cuda:0), every engine built as worker.py builds it: FastPPM(n_inst=1, cfg_rows=True, max_s = 120 + 11.68 + 2,
mimi_stream=False, silence = Mimi(zeros)) + capture(), worker.py's DEF params, seed 1234, Person 01's opening (voice clip = its last 10 s).
Engines share the model weights:
  orig = the DEPLOYED engine module (aloe-pose-monoplex-infer/ppm/fast_ppm.py, under another module name; shared deps asserted identical)
  off  = pose6 engine, both flags off (the model has no s_emb / inp.g0 = every deployed checkpoint)
  s2p  = pose6 engine, s2p cb0 (module SE), G0COND off
  new  = pose6 engine, s2p cb0 (SE) + G0COND (module G0); instances carry g0f
Stages (argv[1], default i,w,iii,iv):
  i    (i) flags off == orig and (ii) zero new params (SE = 0, G0 = 0, f non-zero) == orig: free / free_cfg / poseonly, eager (100 frames)
       and CUDA graphs (150), orig-vs-orig floor first; B lane, speech codes, pose codes + latents hv_p, zp0, hv_s, z0, hn (both rows).
  w    G0COND wiring with a NON-ZERO G0 (scale: ||g0(f)|| ~ a pose frame's ||inp.pose|| sum): prefill locality (trunk KV layer 0 differs
       exactly at P - 1, last layer identical before P - 1) on the cond AND the uncond row, the uncond row's SLOT input = the cond row's
       (same f), f = 0 == flag off (bias-free), f changes the output, fail-loud start() cases on the real engines.
  iii  engine vs the TRAINING forward (training model_ppm: pose6/code if B1's G0COND port is in, else pose-s2p/code + the 811 Inp.forward
       G0COND lines ported in this harness), teacher-forced, on the real Aloe eval micros of cpu_slot_train (4 prompt types), cond row and
       pose-CFG uncond row (trainer on the silence micro, same f); non-zero SE / G0 vs the zero-param floor (= today's engine-vs-trainer).
  iv   per-frame latency, today (orig) / s2p cb0 / s2p cb0 + G0COND, free / free_cfg / poseonly, CUDA graphs (300 frames) and eager (80):
       sync (both GPUs synchronised per frame) median / p90 and async (no per-frame sync) per frame; start() (prefill) time."""
import hashlib
import importlib.util
import inspect
import json
import os
import pickle
import sys
import time

import numpy as np

BR = "/data/home/amartyavarma/ppm/pose6/infer/ppm"
OR = "/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer/ppm"
CK = "/data/home/amartyavarma/ppm/aloe-pose-mono-8020/exports/export-357"
G = "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb"
OUT = "/data/home/amartyavarma/ppm/pose6/infer/tests/out"
TRAIN_B1 = "/data/home/amartyavarma/ppm/pose6/code/model_ppm.py"
TRAIN_S2P = "/data/home/amartyavarma/ppm/pose-s2p/code/model_ppm.py"
for k in ("PPM_S2P", "PPM_G0COND"):
    os.environ.pop(k, None)
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
sys.path.insert(1, os.path.dirname(BR))
import infer as MIF  # noqa: E402   (pose6/infer/infer.py = the deployed infer.py, md5 identical)
from safetensors.numpy import load_file  # noqa: E402

STAGES = (sys.argv[1] if len(sys.argv) > 1 else "i,w,iii,iv").split(",")
CD, OFF, PAD, DEPTH, CBS, BF, TE = FB.CD, FB.OFF, FB.PAD, FB.DEPTH, FB.CBS, FB.BF, FB.TE
RES = {"t_start": time.strftime("%Y-%m-%d %H:%M:%S"), "host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID"), "stages": STAGES}
os.makedirs(OUT, exist_ok=True)
RF = f"{OUT}/engine6_results_{os.environ.get('SLURM_JOB_ID', 'x')}.json"


def log(*a):
    print(f"[a1-gpu {time.strftime('%H:%M:%S')}]", *a, flush=True)


def save():
    json.dump(RES, open(RF, "w"), indent=1, default=str)


def g0_feat(ry, rx, w, H, W):
    """= pose5/code-811-s2p/data_ppm.g0_feat (float64 maths, float32 storage)."""
    f = np.array([ry, rx, np.log(w), -ry / w, (1.0 - ry) / w, -rx / w, (1.0 - rx) / w, np.log(H / W)], np.float64)
    return f.astype(np.float32)


F_A = g0_feat(0.5764973759651184, 0.530517578125, 0.81787109375, 1920.0, 1080.0)   # Person 01 opening frame 0 (root, width); 9:16 portrait
F_B = g0_feat(0.42, 0.61, 0.35, 1080.0, 1920.0)                                      # another plausible clip (landscape, smaller person)
RES["f"] = {"F_A": F_A.tolist(), "F_B": F_B.tolist(), "rule": "data_ppm.g0_feat: [r_y, r_x, log w, -r_y/w, (1-r_y)/w, -r_x/w, (1-r_x)/w, log(H/W)]"}

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
assert binfo["pose"] and binfo["decp_bb"] and not binfo["s2p"] and not binfo["g0cond"], binfo
assert not hasattr(m.tower_p, "s_emb") and not hasattr(m.inp, "g0")
c32, c4, c06 = cfgs
H, H4 = c32.hidden_size, c4.hidden_size
d0, dl = devs[0], devs[-1]
log(f"model loaded {time.time() - t0:.0f}s {binfo['layers_per_dev']}")
RES["binfo"] = {k: binfo[k] for k in ("ck", "step", "layers_per_dev", "s2p", "g0cond", "decp_bb_alpha")}

with torch.inference_mode():
    SE = FB.S2PEmb(H4, "cb0").to(device=d0, dtype=BF)
    G0 = torch.nn.Linear(FB.G0_DIM, H, bias=False).to(device=d0, dtype=BF)
    for mod in (SE, G0):
        mod.weight.zero_()
        mod.weight.requires_grad_(False)
MAXS = 120.0 + 11.68 + 2
NFR = int(round(MAXS * 12.5)) + 8
with torch.inference_mode():
    mimi = FB.IP.Mimi(dl)
    SIL = mimi.encode(np.zeros((2048 + NFR) * 1920, np.float32))[:, :2048 + NFR]
    E = {}
    for name, mod, kw in (("orig", ORIG, {}), ("off", FB, {}), ("s2p", FB, {"s2p": SE, "g0": False}), ("new", FB, {"s2p": SE, "g0": G0})):
        E[name] = mod.FastPPM(m, ldev, cfgs, devs, n_inst=1, cfg_rows=True, max_s=MAXS, mimi_stream=False, silence=SIL, **kw)
        E[name].capture()
assert E["off"].s2p is None and E["off"].g0 is None and E["s2p"].s2p == "cb0" and E["s2p"].g0 is None
assert E["new"].s2p == "cb0" and E["new"].g0 is G0
RES["mem_gib"] = {str(d): round(torch.cuda.max_memory_allocated(d) / 2 ** 30, 1) for d in devs}
log("engines", list(E), RES["mem_gib"])
save()

# ------------------------------------------------------------------------------------------------ instances (= worker.py run())
PERSON = "01_ab63d745"
od = f"{G}/openings/{PERSON}"
APO = np.asarray(load_file(f"{od}/mimi_local.safetensors")["codes"], np.int64)
CPO = np.asarray(load_file(f"{od}/pose_tokens.safetensors")["codes"], np.int64)
TN = min(APO.shape[1], CPO.shape[1])
CLIP = APO[:, TN - 125:TN]
SP_ = MIF.system_block(CLIP, None)
WORDS = MIF.read_words(f"{od}/words.json")
RES["inst"] = {"person": PERSON, "opening_frames": int(TN), "clip_frames": 125, "P": int(SP_["P"])}


def inst_free(n):
    x = MIF.assemble(SP_, np.full(n + CD, PAD, np.int64), np.zeros((DEPTH, 0), np.int64), np.zeros((DEPTH, 0), np.int64), n, 0)
    return x, None, True, n


def inst_poseonly(n):
    """= worker.py pose only: real speech codes + timed-word lane forced every frame, pose generated."""
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
    return x, frc, False, n


LAT = ("hv_p", "zp0", "hv_s", "z0", "hn_prev")


def run(eng, inst, n, seed, params, graphs, extra=None, rec=True, g0f=F_A):
    """one session (g0f goes into the instance only on a G0COND engine). -> (readback, {latent: [frames, ...] cpu})."""
    x, frc, cold, n_ = inst
    n = min(n, n_)
    it = dict(x, a_mode="empty")
    if getattr(eng, "g0", None) is not None:
        it["g0f"] = g0f
    with torch.inference_mode():
        eng.start([it], seed, params, f0=0)
        L = {k: [] for k in LAT}
        for f in range(n + CD):
            d = frc(f) if frc else None
            if extra is not None:
                d = extra(f, dict(d or {})) or None
            eng.step(f, graphs=graphs, force=d, cold_pad=cold)
            if rec:
                for k in LAT:
                    L[k].append(getattr(eng, k).clone())
        S = eng.readback(0, int(x["P"]) + n + CD)
        L = {k: torch.stack(v).cpu() for k, v in L.items() if v}
    S["P"] = int(x["P"])
    return S, L


def diff(A, B):
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


def set_new_params(zero, seed=20261027, g0_mult=1.0, se_mult=1.0):
    """SE / G0 zero, or SE ~ N(0, 0.02 / sqrt 32) (the tower code_emb init scale, = pose-s2p) and G0 ~ N(0, sigma) with sigma from RES['scale']."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.inference_mode():
        if zero:
            SE.weight.zero_(); G0.weight.zero_()
        else:
            SE.weight.copy_((torch.randn(SE.weight.shape, generator=gen) * (0.02 / DEPTH ** 0.5) * se_mult).to(d0, BF))
            G0.weight.copy_((torch.randn(G0.weight.shape, generator=gen) * RES["scale"]["sigma"] * g0_mult).to(d0, BF))


def scale_stats():
    """G0 scale: ||g0(F_A)|| ~ the median per-frame ||inp.pose(pose codes)|| of Person 01's opening (the pose input it sits next to)."""
    with torch.inference_mode():
        pc = torch.as_tensor(CPO[:, :min(200, CPO.shape[1])].T + OFF, device=d0)
        pn = m.inp.pose(pc).float().norm(dim=-1)
        rows = m.inp.pose.weight.float().norm(dim=-1)
        nl = m.inp.emb[1].weight[FB.SP_NL].float().norm()
        tgt = float(pn.median())
        sigma = tgt / (float(np.linalg.norm(F_A)) * H ** 0.5)
    RES["scale"] = {"pose_frame_input_norm_median": tgt, "pose_frame_input_norm_p10_p90": [float(pn.quantile(0.1)), float(pn.quantile(0.9))],
                    "inp_pose_row_norm_median": float(rows.median()), "emb_B_newline_row_norm": float(nl), "norm_F_A": float(np.linalg.norm(F_A)),
                    "norm_F_B": float(np.linalg.norm(F_B)), "sigma": sigma,
                    "rule": "G0.weight ~ N(0, sigma^2), sigma = median ||inp.pose(frame)|| / (||F_A|| sqrt(H)) -> E||g0(F_A)|| = that median"}
    set_new_params(False)
    with torch.inference_mode():
        RES["scale"]["g0_F_A_norm"] = float(G0(torch.as_tensor(F_A[None], device=d0).to(BF)).float().norm())
        RES["scale"]["g0_F_B_norm"] = float(G0(torch.as_tensor(F_B[None], device=d0).to(BF)).float().norm())
        RES["scale"]["s_emb_row_norm_median"] = float(SE.weight.float().norm(dim=-1).median())
    set_new_params(True)
    log(f"scale {json.dumps(RES['scale'])}")


scale_stats()
save()

# ------------------------------------------------------------------------------------------------ (i) + (ii) exactness
if "i" in STAGES:
    set_new_params(True)
    RES["i"] = {}
    for mode, n0 in (("eager", int(os.environ.get("A1_EAGER_FRAMES", "50"))), ("graphs", 150)):
        g = mode == "graphs"
        for sc, (mkinst, prm) in SCEN.items():
            inst = mkinst(n0)
            n = inst[3]
            ref = run(E["orig"], inst, n, SEED, prm, g)
            r = {"floor_orig_vs_orig": diff(ref, run(E["orig"], inst, n, SEED, prm, g))}
            r["i_orig_vs_off"] = diff(ref, run(E["off"], inst, n, SEED, prm, g))
            if g:                                                               # (eager: s2p-only = pose-s2p's proven path; new covers it)
                r["ii_orig_vs_s2p_zero"] = diff(ref, run(E["s2p"], inst, n, SEED, prm, g))
            r["ii_orig_vs_new_zero_fA"] = diff(ref, run(E["new"], inst, n, SEED, prm, g, g0f=F_A))
            r["g0_rows_new"] = [(a, b_, c.tolist()) for a, b_, c in E["new"].g0_rows]
            P = ref[0]["P"]
            r["pass"] = (all(same(v) for k, v in r.items() if isinstance(v, dict))
                         and [(a, b_) for a, b_, _ in r["g0_rows_new"]] == [(0, P - 1), (1, P - 1)])
            r["frames"] = n
            r["gen_pose_frames"] = int((ref[0]["pdep"][P + CD:P + CD + n, 0] >= 0).sum())
            r["gen_words"] = int((ref[0]["lane_b"][P:P + n] < TE).sum())
            RES["i"][f"{mode}/{sc}"] = r
            log(f"(i/ii) {mode}/{sc}: pass {r['pass']} " + json.dumps({k: v for k, v in r.items() if isinstance(v, dict)}))
            save()
    RES["i_pass"] = all(v["pass"] for v in RES["i"].values())
    log("(i/ii) PASS" if RES["i_pass"] else "(i/ii) FAIL")
    save()

# ------------------------------------------------------------------------------------------------ (w) G0COND wiring (non-zero G0, SE = 0)
if "w" in STAGES:
    set_new_params(False)
    with torch.inference_mode():
        SE.weight.zero_()                                                        # isolate G0COND
    RES["w"] = {}
    inst = inst_free(60)
    x = inst[0]
    P = int(x["P"])
    with torch.inference_mode():
        E["off"].start([dict(x, a_mode="empty")], SEED, DEF_CFG, f0=0)
        E["new"].start([dict(x, a_mode="empty", g0f=F_A)], SEED, DEF_CFG, f0=0)
        for dv in devs:
            torch.cuda.synchronize(dv)
        loc = {}
        for r_ in (0, 1):
            for nm_, li in (("layer0", 0), ("last", len(E["new"].bbK) - 1)):
                kd = (E["off"].bbK[li][r_, :P] != E["new"].bbK[li][r_, :P]).flatten(1).any(-1).cpu().numpy()
                vd = (E["off"].bbV[li][r_, :P] != E["new"].bbV[li][r_, :P]).flatten(1).any(-1).cpu().numpy()
                loc[f"row{r_}/{nm_}"] = {"K_differing_positions": np.flatnonzero(kd).tolist()[:8], "n_K_diff": int(kd.sum()),
                                         "V_differing_positions": np.flatnonzero(vd).tolist()[:8], "n_V_diff": int(vd.sum())}
        L0 = E["new"].bbK[0]
        slot_same = bool(torch.equal(L0[0, P - 1], L0[1, P - 1]) and torch.equal(E["new"].bbV[0][0, P - 1], E["new"].bbV[0][1, P - 1]))
        hn_dev = {"cond_hn_prev_maxdiff": float((E["off"].hn_prev[0].float() - E["new"].hn_prev[0].float()).abs().max()),
                  "uncond_hn_prev_maxdiff": float((E["off"].hn_prev[1].float() - E["new"].hn_prev[1].float()).abs().max())}
        rows = [(a, b_, c.tolist()) for a, b_, c in E["new"].g0_rows]
    w1 = {"P": P, "locality": loc, "uncond_row_SLOT_kv_layer0_equals_cond_row": slot_same, "g0_rows": rows, **hn_dev}
    w1["pass"] = (all(v["K_differing_positions"] == [P - 1] and v["V_differing_positions"] == [P - 1] for k, v in loc.items() if k.endswith("layer0"))
                  and all(v["n_K_diff"] >= 1 and v["K_differing_positions"][0] == P - 1 for k, v in loc.items() if k.endswith("last"))
                  and slot_same and [(a, b_) for a, b_, _ in rows] == [(0, P - 1), (1, P - 1)]
                  and all(np.allclose(c, F_A, 0, 0) for _, _, c in rows) and hn_dev["cond_hn_prev_maxdiff"] > 0 and hn_dev["uncond_hn_prev_maxdiff"] > 0)
    RES["w"]["w1_prefill_locality_and_uncond_same_f"] = w1
    log(f"(w1) {json.dumps(w1)}")
    # w2: f = 0 -> identical to flag off (bias-free inp.g0), graphs, free_cfg (uncond rows guide the pose draw)
    for mode in ("eager", "graphs"):
        g = mode == "graphs"
        nw_ = 60 if g else 30
        a_ = run(E["off"], inst_free(nw_), nw_, SEED, DEF_CFG, g)
        b_ = run(E["new"], inst_free(nw_), nw_, SEED, DEF_CFG, g, g0f=np.zeros(8, np.float32))
        dd = diff(a_, b_)
        RES["w"][f"w2_f_zero_equals_off/{mode}"] = {"diff": dd, "pass": same(dd)}
        log(f"(w2) {mode}: {json.dumps(dd)}")
    # w3: f matters (F_A vs F_B vs off), graphs free
    ra = run(E["new"], inst_free(60), 60, SEED, DEF, True, g0f=F_A)
    rb = run(E["new"], inst_free(60), 60, SEED, DEF, True, g0f=F_B)
    ro = run(E["off"], inst_free(60), 60, SEED, DEF, True)
    ra2 = run(E["new"], inst_free(60), 60, SEED, DEF, True, g0f=F_A)
    w3 = {"F_A_vs_F_B": diff(ra, rb), "F_A_vs_off": diff(ra, ro), "F_A_repeat_floor": diff(ra, ra2),
          "frame0_hn_maxdiff_F_A_vs_off": float((ra[1]["hn_prev"][0].float() - ro[1]["hn_prev"][0].float()).abs().max())}
    w3["pass"] = same(w3["F_A_repeat_floor"]) and not same(w3["F_A_vs_F_B"]) and not same(w3["F_A_vs_off"])
    RES["w"]["w3_f_changes_output"] = w3
    log(f"(w3) {json.dumps(w3)}")
    # w4: fail loud on the real engines (checked before any engine state changes)
    xi = dict(x, a_mode="empty")
    bad = dict(xi, g0f=F_A, ids=np.array(x["ids"], copy=True))
    bad["ids"][1, P - 1] = PAD

    def raises(fn, exc):
        try:
            with torch.inference_mode():
                fn()
        except exc as e:
            return f"{type(e).__name__}: {str(e)[:120]}"
        return None
    w4 = {"new_without_g0f": raises(lambda: E["new"].start([xi], SEED, DEF), ValueError),
          "off_with_g0f": raises(lambda: E["off"].start([dict(xi, g0f=F_A)], SEED, DEF), ValueError),
          "s2p_only_with_g0f": raises(lambda: E["s2p"].start([dict(xi, g0f=F_A)], SEED, DEF), ValueError),
          "orig_ignores_g0f_key (deployed engine, documented)": raises(lambda: E["orig"].start([dict(xi, g0f=F_A)], SEED, DEF), Exception),
          "new_slot_not_newline": raises(lambda: E["new"].start([bad], SEED, DEF), AssertionError)}
    w4["pass"] = all(w4[k] for k in ("new_without_g0f", "off_with_g0f", "s2p_only_with_g0f", "new_slot_not_newline"))
    RES["w"]["w4_fail_loud"] = w4
    log(f"(w4) {json.dumps(w4)}")
    RES["w_pass"] = all(v["pass"] for v in RES["w"].values())
    log("(w) PASS" if RES["w_pass"] else "(w) FAIL")
    set_new_params(True)
    save()

# ------------------------------------------------------------------------------------------------ (iii) engine vs the training forward
if "iii" in STAGES or "iiis" in STAGES or "loc2" in STAGES:
    import torch.nn as nn
    tpath = TRAIN_B1
    b1 = False
    try:
        src = open(TRAIN_B1).read()
        b1 = "g0cond" in src and "def __init__(self, c32, c4, c06" in src and "g0cond=False" in src
    except OSError:
        pass
    if not b1:
        tpath = TRAIN_S2P
    def _load_train(path_):
        sp_ = importlib.util.spec_from_file_location("model_ppm_train", path_)
        mod_ = importlib.util.module_from_spec(sp_)
        sp_.loader.exec_module(mod_)
        return mod_
    try:
        TM = _load_train(tpath)
    except Exception as e_:                                                    # B1's file mid-edit: fall back (recorded)
        log(f"(iii) training model {tpath} failed to import ({type(e_).__name__}: {e_}); using {TRAIN_S2P}")
        RES["iii_train_import_error"] = f"{tpath}: {type(e_).__name__}: {e_}"
        tpath, b1 = TRAIN_S2P, False
        TM = _load_train(tpath)
    has_g0 = ("g0cond" in inspect.signature(TM.PPM.__init__).parameters and hasattr(TM, "Inp")
              and "g0" in inspect.signature(TM.Inp.forward).parameters)          # B1's port with the 811 API (Inp.forward(.., g0=(gpos, gf)))
    with torch.device("meta"):
        tm = TM.PPM(c32, c4, c06, int(m.head.proj.G.shape[0]), int(m.tower_s.text_emb.weight.shape[0]), decp_bb=True, s2p="cb0",
                    **({"g0cond": True} if has_g0 else {}))
        if has_g0:
            assert hasattr(tm.inp, "g0") and tm.inp.g0.bias is None, "B1 port: PPM(g0cond=True) must build inp.g0 (no bias)"
        if not has_g0:                                                         # HARNESS PORT (= pose5/code-811-s2p/model_ppm.py:504)
            tm.inp.g0 = nn.Linear(FB.G0_DIM, c32.hidden_size, bias=False)
    sd = dict(m.state_dict())
    sd["tower_p.s_emb.weight"], sd["inp.g0.weight"] = SE.weight, G0.weight
    names = dict(tm.named_parameters())
    assert set(names) == set(sd), ("training model keys != engine keys + new", sorted(set(names) ^ set(sd))[:8])
    for n_, t_ in sd.items():                                                  # share the engine's tensors (no copy): module._parameters[leaf] = t
        mod_ = tm.get_submodule(n_.rsplit(".", 1)[0])
        mod_._parameters[n_.rsplit(".", 1)[1]] = t_
    for dec, dv in ((tm.dec_s, dl), (tm.dec_p, d0)):
        dec.model.rotary_emb = type(dec.model.rotary_emb)(config=c06, device=dv)
    tm.decp_bb_alpha = float(m.decp_bb_alpha)
    left = [n_ for n_, t_ in list(tm.named_parameters()) + list(tm.named_buffers()) if t_.is_meta]
    assert not left, ("meta tensors left in the training model", left[:8])
    tm.eval()
    RES["iii_train_model"] = {"file": tpath, "b1_port": b1, "g0_in_training_class": has_g0,
                              "note": None if has_g0 else "B1's pose6/code G0COND port not in yet: training Inp.forward + the 3 G0COND lines of "
                                                          "pose5/code-811-s2p/model_ppm.py:253-255 (verbatim) applied in this harness"}

    def train_inp(ids, depth, present, pdepth, g0):
        if has_g0:
            return tm.inp(ids, depth, present, pdepth, **({"g0": g0} if g0 is not None else {}))
        out = tm.inp(ids, depth, present, pdepth)
        if g0 is not None:                                                     # = code-811-s2p model_ppm.py:253-255
            gpos, gf = g0
            out = out.index_add(0, gpos, tm.inp.g0(gf.to(tm.inp.g0.weight.dtype)).to(out.dtype))
        return out

    @torch.inference_mode()
    def trainer_ref(it, L, g0pos, g0f, sil=False):
        """the training modules over positions [0, L) of the micro (packed segment = the whole prefix; pid 0..L-1) -> dict."""
        t = lambda x, d: torch.as_tensor(np.ascontiguousarray(x), device=d)  # noqa: E731
        ids, depth, present, pdepth = it["ids"][:, :L], it["depth"][:, :L], it["present"][:, :L], it["pdepth"][:L]
        if sil:
            depth = FB.silence_depth(np.asarray(depth), SIL)
        with torch.cuda.device(d0):
            g0 = None if g0f is None else (t(np.asarray(g0pos, np.int64), d0), t(np.asarray(g0f, np.float32)[None], d0))
            e = train_inp(t(ids, d0), t(depth, d0), t(present, d0), t(pdepth, d0), g0)[None]
        pid = torch.arange(L)[None]
        h, cache = e, {}
        for i, blk in enumerate(tm.layers):
            dv = ldev[i]
            if h.device != dv:
                h = h.to(dv)
            if dv not in cache:
                cache[dv] = (TM.Rot(c32, dv)(h, pid.to(dv)), TM.Ctx([L], dv))
            with torch.cuda.device(dv):
                h = blk(h, *cache[dv])
        out = {}
        with torch.cuda.device(dl):
            nrm = tm.head.norm
            hn = TM.FO.rms(h[0].to(dl), nrm.weight, float(nrm.variance_epsilon))   # = training Head.forward's hn
            out["hn"] = hn
            ctx = TM.Ctx([L], dl)
            lab = torch.where(t(depth[1, :, 0], dl) >= 0, t(depth[1, :, 0], dl), torch.full((L,), -100, device=dl))
            hv_s, *_ = tm.tower_s(t(ids[1], dl), t(depth[1], dl), hn, pid.to(dl), ctx, TM.Rot(c4, dl), lab.long(), torch.zeros(L, device=dl))
            out["hv_s"] = hv_s
            out["z0"] = (hv_s @ tm.tower_s.cb0_head.weight.t()).float()
        with torch.cuda.device(d0):
            hp = hn.to(d0)
            ctx = TM.Ctx([L], d0)
            plab = torch.where(t(pdepth[:, 0], d0) >= 0, t(pdepth[:, 0], d0), torch.full((L,), -100, device=d0))
            hv_p, *_ = tm.tower_p(t(ids[1], d0), t(pdepth, d0), hp, pid.to(d0), ctx, TM.Rot(c4, d0), plab.long(), torch.zeros(L, device=d0),
                                  sdep=t(depth[1], d0))                       # = training PPM.forward: sdep = B's speech codes at T
            out["hv_p"] = hv_p
            out["zp0"] = (hv_p @ tm.tower_p.cb0_head.weight.t()).float()
            out["zb"] = TM.shift1(tm.decp_bb(hp), ctx, 0.0) * float(tm.decp_bb_alpha)
        return out

    @torch.inference_mode()
    def dec_ref(dec, hv, depth_g, flat, zb):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x, _ = dec.inputs(hv, depth_g, flat) if zb is None else dec.inputs(hv, depth_g, flat, zb)
            hl = dec.model(inputs_embeds=x, use_cache=False).last_hidden_state
            return torch.einsum("ndh,dvh->ndv", hl[:, 1:], dec.heads.weight.view(DEPTH - 1, CBS, -1)).float()

    @torch.inference_mode()
    def fused_teacher(dec, dev, hv, codes, zb, S=8):
        out = []
        with torch.cuda.device(dev):
            fd = FB.FusedDec(dec, dev, S)
            for i in range(0, hv.shape[0], S):
                out.append(fd.teacher_logits(hv[i:i + S], codes[i:i + S], None if zb is None else zb[i:i + S]))
            del fd
        return torch.cat(out)

    def cmp_logits(ze, zr, mask_word=None):
        ze, zr = ze.float(), zr.float().to(ze.device)
        a = (ze.argmax(-1) == zr.argmax(-1)).float()
        pe, pr = torch.softmax(ze, -1), torch.softmax(zr, -1)
        tv = 0.5 * (pe - pr).abs().sum(-1)
        fin = torch.isfinite(ze) & torch.isfinite(zr)
        d = torch.where(fin, (ze - zr).abs(), torch.zeros_like(ze))
        r = {"n": int(ze.shape[0]), "top1": round(float(a.mean()), 5), "tv_mean": round(float(tv.mean()), 6), "tv_max": round(float(tv.max()), 5),
             "max_abs": round(float(d.max()), 5), "mean_abs": round(float(d.sum() / fin.sum()), 6), "ref_absmax": round(float(zr.abs().max()), 2)}
        if mask_word is not None and bool(mask_word.any()):
            w = ze[:, :TE][mask_word].argmax(-1) == zr[:, :TE][mask_word].argmax(-1)
            r["word_top1"] = round(float(w.float().mean()), 5)
        return r

    def maxabs(a, b):
        return round(float((a.float() - b.float().to(a.device)).abs().max()), 5)

    def engine_tf(eng, mic, N, g0f, graphs):
        """engine teacher-forced from conversation frame 0 (prefill = the system block), B's lane / Mimi / pose forced every frame."""
        P = int(mic["P"])
        L = P + N + CD
        it = {"ids": np.asarray(mic["ids"])[:, :L], "depth": np.asarray(mic["depth"])[:, :L], "present": np.asarray(mic["present"])[:, :L],
              "pdepth": np.asarray(mic["pdepth"])[:L], "P": P, "a_mode": "empty"}
        if eng.g0 is not None:
            it["g0f"] = g0f
        rec = {k: [] for k in ("zt", "z0", "zp0", "hn", "hv_p", "zb")}
        with torch.inference_mode():
            eng.start([it], SEED, DEF, f0=0)
            for f in range(N):
                T = P + f
                force = {"lane": [int(it["ids"][1, T])]}
                if it["depth"][1, T, 0] >= 0:
                    force["codes"] = it["depth"][1, T][None]
                if it["pdepth"][T, 0] >= 0:
                    force["pose"] = it["pdepth"][T][None]
                eng.step(f, graphs=graphs, force=force)
                rec["zt"].append(eng.zt[0].clone()); rec["z0"].append(eng.z0[0].clone()); rec["zp0"].append(eng.zp0.clone())
                rec["hn"].append(eng.hn_prev.clone()); rec["hv_p"].append(eng.hv_p.clone()); rec["zb"].append(eng.zb_p.clone())
            for dv in devs:
                torch.cuda.synchronize(dv)
        return {k: torch.stack(v) for k, v in rec.items()}

    micros = {}
    for typ in ("clip", "persona", "empty", "clip+persona"):
        fp = f"{OUT}/micros/{typ}.pkl"
        if os.path.isfile(fp):
            micros[typ] = pickle.load(open(fp, "rb"))
    assert micros, "no parity micros (run cpu.sbatch first)"
    RES["iii"] = RES.get("iii", {})
    for zero in ((True, False) if "iii" in STAGES else ()):
        set_new_params(zero)
        tag = "zero_new_params_floor" if zero else "nonzero_new_params"
        for typ, mic in micros.items():
            P = int(mic["P"])
            N = int(min(int(os.environ.get("A1_PARITY_FRAMES", "300")), mic["n_frames"]))
            L = P + N
            g0pos = list(np.asarray(mic["g0pos"]).reshape(-1)) if "g0pos" in mic else [P - 1]
            for mode in (("graphs", "eager") if (typ == "clip" and not zero) else ("graphs",)):
                if mode == "eager":
                    N = min(N, 100)
                    L = P + N
                er = engine_tf(E["new"], mic, N, F_A, mode == "graphs")
                ref = trainer_ref(mic, L, g0pos, F_A)
                refu = trainer_ref(mic, L, g0pos, F_A, sil=True)
                sel = torch.arange(P - 1, L - 1, device=dl)
                zr = E["new"].text_logits(ref["hn"][sel])
                y = torch.as_tensor(np.asarray(mic["ids"])[1, P:L], device=dl)
                pr = {"P": P, "frames": N, "g0pos": [int(v) for v in g0pos], "g0pos_src": "micro (B1 data)" if "g0pos" in mic else "P - 1 (D1)",
                      "text": cmp_logits(er["zt"], zr, y < TE),
                      "speech_cb0": cmp_logits(er["z0"], ref["z0"][P:L]),
                      "pose_cb0_cond": cmp_logits(er["zp0"][:, 0].to(dl), ref["zp0"][P:L].to(dl)),
                      "pose_cb0_uncond": cmp_logits(er["zp0"][:, 1].to(dl), refu["zp0"][P:L].to(dl)),
                      "hn_cond_max_abs": maxabs(er["hn"][:, 0], ref["hn"][P:L]), "hn_uncond_max_abs": maxabs(er["hn"][:, 1], refu["hn"][P:L]),
                      "hn_ref_absmax": round(float(ref["hn"][P:L].float().abs().max()), 1),
                      "hv_p_cond_max_abs": maxabs(er["hv_p"][:, 0], ref["hv_p"][P:L]),
                      "zb_cond_max_abs": maxabs(er["zb"][CD:, 0], ref["zb"][P + CD:L])}                # (zb is computed on frames >= CD)
                fl = torch.linspace(CD, N - 1, 24).round().long()
                pd_ = torch.as_tensor(np.asarray(mic["pdepth"])[:L], device=d0)
                codes_p = pd_[(P + fl).to(d0)]
                with torch.cuda.device(d0):
                    ref_p = dec_ref(tm.dec_p, ref["hv_p"], pd_, (P + fl).to(d0), ref["zb"])
                zp_e = fused_teacher(m.dec_p, d0, er["hv_p"][fl.to(d0), 0].to(d0), codes_p - E["new"].offp, er["zb"][fl.to(d0), 0].to(d0))
                zp_s = fused_teacher(m.dec_p, d0, ref["hv_p"][(P + fl).to(d0)], codes_p - E["new"].offp, ref["zb"][(P + fl).to(d0)])
                pr["pose_dec_e2e"] = cmp_logits(zp_e.reshape(-1, CBS).to(dl), ref_p.reshape(-1, CBS).to(dl))
                pr["pose_dec_same_input"] = cmp_logits(zp_s.reshape(-1, CBS).to(dl), ref_p.reshape(-1, CBS).to(dl))
                if not zero:                                                     # effect size: the mods vs the zero-param trainer
                    set_new_params(True)
                    ref0 = trainer_ref(mic, L, g0pos, F_A)
                    set_new_params(False)
                    pr["effect_trainer_nonzero_vs_zero"] = {"hn_max_abs": maxabs(ref["hn"][P:L], ref0["hn"][P:L]),
                                                            "text": cmp_logits(E["new"].text_logits(ref0["hn"][sel]), zr, y < TE),
                                                            "pose_cb0": cmp_logits(ref0["zp0"][P:L].to(dl), ref["zp0"][P:L].to(dl)),
                                                            "speech_cb0": cmp_logits(ref0["z0"][P:L], ref["z0"][P:L])}
                    del ref0
                RES["iii"][f"{tag}/{typ}/{mode}"] = pr
                log(f"(iii) {tag}/{typ}/{mode}: {json.dumps(pr)}")
                del er, ref, refu
                torch.cuda.empty_cache()
                save()
    set_new_params(True)
    save()

    # -------------------------------------------------------------------------------------------- (iii-strong) + (loc)
    if "iiis" in STAGES:
        def stream_stats(ze, zr, mask=None):
            r = cmp_logits(ze, zr, mask)
            return {k: r[k] for k in ("top1", "tv_mean", "max_abs", "mean_abs") if k in r}

        def hn_stats(a, b):
            d = (a.float() - b.float().to(a.device)).abs()
            rm = d.max(-1).values
            return {"max_abs": round(float(d.max()), 4), "mean_abs": round(float(d.mean()), 6), "rowmax_median": round(float(rm.median()), 4),
                    "rowmax_p99": round(float(torch.quantile(rm.float(), 0.99)), 4), "n_gt_2": int((d > 2).sum())}

        def locate(eng_h, ref_h, P, floor_d=None, k=5):
            """top-k |engine - ref| entries of hn [N, H]: frame, position, channel, ref / engine values, and the floor run's diff there."""
            d = (eng_h.float() - ref_h.float().to(eng_h.device)).abs()
            vals, idx = d.flatten().topk(k)
            out = []
            for v, i in zip(vals.tolist(), idx.tolist()):
                f_, c_ = divmod(i, d.shape[1])
                e = {"frame": f_, "position": P + f_, "channel": c_, "diff": round(v, 4), "ref": round(float(ref_h[f_, c_]), 4),
                     "engine": round(float(eng_h[f_, c_]), 4), "ref_rowabsmax": round(float(ref_h[f_].float().abs().max()), 2),
                     "channel_ref_absmax_over_frames": round(float(ref_h[:, c_].float().abs().max()), 2)}
                if floor_d is not None:
                    e["floor_diff_same_frame_channel"] = round(float(floor_d[f_, c_]), 4)
                    e["floor_rowmax_same_frame"] = round(float(floor_d[f_].max()), 4)
                out.append(e)
            return out

        RES["iiis"] = {}
        NS = int(os.environ.get("A1_STRONG_FRAMES", "150"))
        SCALES = {"realistic": (1.0, 1.0), "strong": (float(os.environ.get("A1_G0_MULT", "10")), float(os.environ.get("A1_SE_MULT", "30")))}
        RES["iiis_cfg"] = {"frames": NS, "scales (g0 sigma mult, s_emb std mult)": SCALES, "micros": ["clip", "empty"], "mode": "graphs",
                           "g0_sigma_realistic": RES["scale"]["sigma"], "s_emb_std_realistic": 0.02 / DEPTH ** 0.5}
        for typ in ("clip", "empty"):
            if typ not in micros:
                continue
            mic = micros[typ]
            P = int(mic["P"])
            N = int(min(NS, mic["n_frames"]))
            L = P + N
            g0pos = [P - 1]
            sel = torch.arange(P - 1, L - 1, device=dl)
            y = torch.as_tensor(np.asarray(mic["ids"])[1, P:L], device=dl)
            # the trainer-side uncond reference = the engine's uncond input rule (checked here, not assumed)
            dsil = FB.silence_depth(np.asarray(mic["depth"])[:, :L], SIL)
            dorig = np.asarray(mic["depth"])[:, :L]
            chg = np.flatnonzero((dsil[1, :, 0] != dorig[1, :, 0]) | (dsil[1] != dorig[1]).any(-1))
            st = E["new"].sil_t.cpu().numpy()
            conv = np.arange(P + CD, L)
            eng_rule = np.where(dorig[1, conv, :1] >= 0, st[np.clip(conv - P - CD, 0, st.shape[0] - 1)], -1)
            RES["iiis"][f"{typ}/uncond_reference"] = {
                "P": P, "silence_positions_in_system_block": int((chg < P).sum()), "silence_positions_in_conversation": int((chg >= P).sum()),
                "first_conversation_silence_position": int(chg[chg >= P][0]) if (chg >= P).any() else None,
                "voice_clip_coded_positions": int((dorig[1, :P, 0] >= 0).sum()),
                "conversation_silence_codes == engine per-frame rule sil_t[T - P - CD]": bool(np.array_equal(dsil[1, conv], eng_rule)),
                "channel_A_untouched (absent)": bool((dsil[0] == dorig[0]).all()), "ids / present / pdepth": "unchanged (only depth is replaced)",
                "g0pos": g0pos, "g0_f": "F_A (= the cond row's)"}
            # zero-param floor: engine vs trainer, cond + uncond
            set_new_params(True)
            er0 = engine_tf(E["new"], mic, N, F_A, True)
            ref0, refu0 = trainer_ref(mic, L, g0pos, F_A), trainer_ref(mic, L, g0pos, F_A, sil=True)
            floor_du = (er0["hn"][:, 1].float() - refu0["hn"][P:L].float()).abs()
            floor_dc = (er0["hn"][:, 0].float() - ref0["hn"][P:L].float()).abs()
            zr0 = E["new"].text_logits(ref0["hn"][sel])
            floor = {"text": stream_stats(er0["zt"], zr0, y < TE), "speech_cb0": stream_stats(er0["z0"], ref0["z0"][P:L]),
                     "pose_cb0_cond": stream_stats(er0["zp0"][:, 0].to(dl), ref0["zp0"][P:L].to(dl)),
                     "pose_cb0_uncond": stream_stats(er0["zp0"][:, 1].to(dl), refu0["zp0"][P:L].to(dl)),
                     "hn_cond": hn_stats(er0["hn"][:, 0], ref0["hn"][P:L]), "hn_uncond": hn_stats(er0["hn"][:, 1], refu0["hn"][P:L]),
                     "hv_p_cond_max_abs": maxabs(er0["hv_p"][:, 0], ref0["hv_p"][P:L])}
            RES["iiis"][f"{typ}/floor_zero_params"] = {"parity": floor, "loc_hn_uncond_top5": locate(er0["hn"][:, 1], refu0["hn"][P:L], P),
                                                       "loc_hn_cond_top5": locate(er0["hn"][:, 0], ref0["hn"][P:L], P)}
            log(f"(iiis) {typ} floor: {json.dumps(RES['iiis'][f'{typ}/floor_zero_params'])}")
            save()
            for sname, (gm, sm) in SCALES.items():
                set_new_params(False, g0_mult=gm, se_mult=sm)
                with torch.inference_mode():
                    gn = float(G0(torch.as_tensor(F_A[None], device=d0).to(BF)).float().norm())
                er = engine_tf(E["new"], mic, N, F_A, True)
                ref, refu = trainer_ref(mic, L, g0pos, F_A), trainer_ref(mic, L, g0pos, F_A, sil=True)
                zr = E["new"].text_logits(ref["hn"][sel])
                par = {"text": stream_stats(er["zt"], zr, y < TE), "speech_cb0": stream_stats(er["z0"], ref["z0"][P:L]),
                       "pose_cb0_cond": stream_stats(er["zp0"][:, 0].to(dl), ref["zp0"][P:L].to(dl)),
                       "pose_cb0_uncond": stream_stats(er["zp0"][:, 1].to(dl), refu["zp0"][P:L].to(dl)),
                       "hn_cond": hn_stats(er["hn"][:, 0], ref["hn"][P:L]), "hn_uncond": hn_stats(er["hn"][:, 1], refu["hn"][P:L]),
                       "hv_p_cond_max_abs": maxabs(er["hv_p"][:, 0], ref["hv_p"][P:L])}
                eff = {"text": stream_stats(zr0, zr, y < TE), "speech_cb0": stream_stats(ref0["z0"][P:L], ref["z0"][P:L]),
                       "pose_cb0_cond": stream_stats(ref0["zp0"][P:L].to(dl), ref["zp0"][P:L].to(dl)),
                       "pose_cb0_uncond": stream_stats(refu0["zp0"][P:L].to(dl), refu["zp0"][P:L].to(dl)),
                       "hn_cond": hn_stats(ref0["hn"][P:L], ref["hn"][P:L]), "hn_uncond": hn_stats(refu0["hn"][P:L], refu["hn"][P:L]),
                       "hv_p_cond_max_abs": maxabs(ref0["hv_p"][P:L], ref["hv_p"][P:L])}
                ratio = {}
                for k_ in ("text", "speech_cb0", "pose_cb0_cond", "pose_cb0_uncond"):
                    ratio[k_] = {"tv_mean effect/parity": round(eff[k_]["tv_mean"] / max(par[k_]["tv_mean"], 1e-9), 2),
                                 "mean_abs effect/parity": round(eff[k_]["mean_abs"] / max(par[k_]["mean_abs"], 1e-9), 2),
                                 "parity/floor tv_mean": round(par[k_]["tv_mean"] / max(floor[k_]["tv_mean"], 1e-9), 2)}
                for k_ in ("hn_cond", "hn_uncond"):
                    ratio[k_] = {"mean_abs effect/parity": round(eff[k_]["mean_abs"] / max(par[k_]["mean_abs"], 1e-9), 2),
                                 "parity/floor mean_abs": round(par[k_]["mean_abs"] / max(floor[k_]["mean_abs"], 1e-9), 2),
                                 "parity/floor max_abs": round(par[k_]["max_abs"] / max(floor[k_]["max_abs"], 1e-9), 2)}
                rr = {"g0_mult": gm, "se_mult": sm, "g0_F_A_norm": round(gn, 3), "parity_engine_vs_trainer": par,
                      "effect_trainer_nonzero_vs_zero": eff, "ratios": ratio,
                      "loc_hn_uncond_top5": locate(er["hn"][:, 1], refu["hn"][P:L], P, floor_du),
                      "loc_hn_cond_top5": locate(er["hn"][:, 0], ref["hn"][P:L], P, floor_dc)}
                RES["iiis"][f"{typ}/{sname}"] = rr
                log(f"(iiis) {typ}/{sname}: {json.dumps(rr)}")
                del er, ref, refu
                torch.cuda.empty_cache()
                save()
            del er0, ref0, refu0
        set_new_params(True)
        save()

    # -------------------------------------------------------------------------------------------- (loc2) the clip-micro uncond outlier
    if "loc2" in STAGES:
        """per frame 0..K-1 (clip micro, realistic + strong): uncond / cond hn row-max |diff| of (a) the engine STEP path (graphs, = served) vs
        the trainer, (b) the engine PREFILL path (its eager FA3-varlen prefill over [0, P + K), teacher-forced) vs the trainer, (c) the
        trainer's own sensitivity: f[0] moved by ONE bf16 ulp (the trunk input at the SLOT changes by ~1 ulp of g0's row) -> its hn change."""
        RES["loc2"] = {}
        K = int(os.environ.get("A1_LOC2_FRAMES", "12"))
        mic = micros["clip"]
        P = int(mic["P"])
        N = 40
        L = P + N
        g0pos = [P - 1]
        F_U = F_A.copy()
        F_U[0] = float(torch.tensor(F_A[0]).to(BF).float() * (1 + 2 ** -7))     # the next bf16 value above f[0]'s bf16 (1 ulp)
        assert torch.tensor(F_U[0]).to(BF) != torch.tensor(F_A[0]).to(BF), "1-ulp nudge must change the bf16 input"
        for sname, (gm, sm) in (("realistic", (1.0, 1.0)), ("strong", (float(os.environ.get("A1_G0_MULT", "10")), float(os.environ.get("A1_SE_MULT", "30"))))):
            set_new_params(False, g0_mult=gm, se_mult=sm)
            er = engine_tf(E["new"], mic, N, F_A, True)
            ref, refu = trainer_ref(mic, L, g0pos, F_A), trainer_ref(mic, L, g0pos, F_A, sil=True)
            refn, refun = trainer_ref(mic, L, g0pos, F_U), trainer_ref(mic, L, g0pos, F_U, sil=True)
            Np = P + K
            ids, dep, pres, pdep = (np.asarray(mic[k_]) for k_ in ("ids", "depth", "present", "pdepth"))
            with torch.inference_mode():
                hp_c = E["new"]._prefill_row(0, ids[:, :Np], dep[:, :Np], pres[:, :Np], pdep[:Np], True, g0=(P - 1, F_A))
                hp_u = E["new"]._prefill_row(1, ids[:, :Np], FB.silence_depth(dep[:, :Np], SIL), pres[:, :Np], pdep[:Np], False, g0=(P - 1, F_A))
                E["new"]._view(0)

            def rowmax(a, b):
                return [round(float(x), 4) for x in (a.float() - b.float().to(a.device)).abs().max(-1).values[:K]]
            rr = {"g0_mult": gm, "se_mult": sm, "frames": K,
                  "uncond: step(served) vs trainer": rowmax(er["hn"][:, 1], refu["hn"][P:L]),
                  "uncond: engine prefill path vs trainer": rowmax(hp_u[P:Np], refu["hn"][P:Np]),
                  "uncond: step vs engine prefill path": rowmax(er["hn"][:K, 1], hp_u[P:Np]),
                  "uncond: trainer sensitivity to a 1-ulp change of f[0]": rowmax(refun["hn"][P:L], refu["hn"][P:L]),
                  "cond: step(served) vs trainer": rowmax(er["hn"][:, 0], ref["hn"][P:L]),
                  "cond: engine prefill path vs trainer": rowmax(hp_c[P:Np], ref["hn"][P:Np]),
                  "cond: trainer sensitivity to a 1-ulp change of f[0]": rowmax(refn["hn"][P:L], ref["hn"][P:L]),
                  "uncond: hn row absmax (trainer)": [round(float(x), 2) for x in refu["hn"][P:P + K].float().abs().max(-1).values]}
            RES["loc2"][sname] = rr
            log(f"(loc2) {sname}: {json.dumps(rr)}")
            del er, ref, refu, refn, refun, hp_c, hp_u
            torch.cuda.empty_cache()
            save()
        set_new_params(True)
        save()

# ------------------------------------------------------------------------------------------------ (iv) latency
if "iv" in STAGES:
    set_new_params(False)
    RES["iv"] = {}
    for mode, n in (("graphs", 300), ("eager", int(os.environ.get("A1_EAGER_LAT_FRAMES", "30")))):
        g = mode == "graphs"
        for sc in ("free", "free_cfg", "poseonly"):
            mkinst, prm = SCEN[sc]
            x, frc, cold, n_ = mkinst(n)
            nn_ = min(n, n_)
            for name, label in (("orig", "today"), ("s2p", "s2p_cb0"), ("new", "s2p_cb0+g0")):
                eng = E[name]
                it = dict(x, a_mode="empty", **({"g0f": F_A} if getattr(eng, "g0", None) is not None else {}))
                with torch.inference_mode():
                    tst = []
                    for rep in range(2):
                        for dv in devs:
                            torch.cuda.synchronize(dv)
                        ta = time.perf_counter()
                        eng.start([it], SEED, prm, f0=0)
                        tst.append(time.perf_counter() - ta)
                        ts = []
                        for f in range(nn_ + CD if rep else min(30 if g else 6, nn_ + CD)):   # rep 0 = warm-up
                            t1 = time.perf_counter()
                            eng.step(f, graphs=g, force=frc(f) if frc else None, cold_pad=cold)
                            for dv in devs:
                                torch.cuda.synchronize(dv)
                            ts.append(time.perf_counter() - t1)
                    eng.start([it], SEED, prm, f0=0)
                    for dv in devs:
                        torch.cuda.synchronize(dv)
                    ta = time.perf_counter()
                    for f in range(nn_ + CD):
                        eng.step(f, graphs=g, force=frc(f) if frc else None, cold_pad=cold)
                    for dv in devs:
                        torch.cuda.synchronize(dv)
                    tot_async = time.perf_counter() - ta
                ts = np.asarray(ts[CD + 2:]) * 1e3
                r = {"sync_ms_median": round(float(np.median(ts)), 2), "sync_ms_p90": round(float(np.percentile(ts, 90)), 2),
                     "sync_ms_max": round(float(ts.max()), 2), "async_ms_per_frame": round(1e3 * tot_async / (nn_ + CD), 2), "frames": nn_ + CD,
                     "start_prefill_ms": round(1e3 * tst[-1], 1)}
                RES["iv"][f"{mode}/{sc}/{label}"] = r
                log(f"(iv) {mode} {sc} {label}: {json.dumps(r)}")
            save()
RES["t_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
save()
log("done")
