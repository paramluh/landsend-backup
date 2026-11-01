#!/usr/bin/env python3
"""pose6 A3: GPU end-to-end test of worker_s2pg0.py. NOT RUN YET: it needs a checkpoint with BOTH keys (tower_p.s_emb cb0 + inp.g0) and
2 GPUs from main. Runs the REAL Worker (build_model + check_mods + CUDA-graph capture, = the playground worker) on HIDDEN-style requests,
with every output path redirected to pose6/worker/tests/out/gpu: nothing goes to genweb runs/, render/jobs or the live feed.

R1 avatar person, conversation-start prefill (opening with frame-0 fields) -> kind openings
R2 avatar person, actor-clip prefill -> kind clip
R3 pose only, SAM opening -> kind openings-sam (stick figure, in the run dir)
R4 Rosie (no avatar), start prefill -> kind openings (stick figure)
R5 = R1 with the request override g0f = R1's own f -> tokens bit-identical to R1 (same seed, same f: the override path = the source path)
R6 = R1 with the override f = 0 -> tokens differ from R1 (g0f reaches the model; inp.g0 is non-zero in a trained checkpoint)
E1 eng.start without g0f -> ValueError (the engine refuses; never f = 0)
For every run: status done, summary g0f == g0src.g0_from_opening(dir, kind) of its g0_source, ms/frame reported.

TEST-ONLY OVERLAY (POSE6_TEST_OVERLAY=1; no checkpoint with both keys exists yet): GENWEB_CK = export-357 (neither key). The TEST wraps
the engine's build_model: after the REAL load, it attaches tower_p.s_emb (FP.S2PEmb cb0, N(0, 0.02 / sqrt 32)) and inp.g0
(Linear 8 -> H, no bias, N(0, sigma^2)) on the pose GPU, seeded exactly like A1's test_engine6.set_new_params (one CPU generator, seed
20261027, s_emb first). sigma follows A1's rule: median ||inp.pose(Person 01 opening frame)|| / (||F_A|| sqrt(H)). It then marks binfo
s2p cb0 + g0cond (zero_init False), so the worker's own check_mods sees both mods, and asserts both weights are non-zero. worker_s2pg0.py
itself is NOT changed: its md5 is asserted below.
usage: GENWEB_CK=<ckpt> [POSE6_TEST_OVERLAY=1] python -B test_worker_gpu.py   (CUDA_VISIBLE_DEVICES = 2 GPUs; env: run_gpu_test.sh)"""
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/out/gpu"
VAR = f"{os.path.dirname(HERE)}/worker_s2pg0.py"
WORKER_MD5 = "c5afe6e536f6a0cb3b4a658acc92bc76"                         # the worker file under test (README); never edited for the test
assert hashlib.md5(open(VAR, "rb").read()).hexdigest() == WORKER_MD5, ("worker_s2pg0.py changed since this test was written", VAR)
assert os.environ.get("GENWEB_CK"), "set GENWEB_CK"
OVERLAY = os.environ.get("POSE6_TEST_OVERLAY", "0") == "1"
assert os.environ.get("PPM_S2P") is None and os.environ.get("PPM_G0COND") is None, "PPM_S2P / PPM_G0COND must be unset (the engine follows the checkpoint)"
os.environ.pop("GENWEB_WEIGHTS", None)
for d in ("runs", "render/jobs", "render/jobs_vansh", "render/jobs_vansh50", "render/out"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
spec = importlib.util.spec_from_file_location("genweb_worker_s2pg0", VAR)
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)
W.WID = "gputest"
OVL = {}
if OVERLAY:
    import torch
    from safetensors.numpy import load_file as _lf
    _real_build = W.FP.build_model
    _cpo = np.asarray(_lf(f"{W.G}/openings/01_ab63d745/pose_tokens.safetensors")["codes"], np.int64)
    _fa = np.array([0.5764973759651184, 0.530517578125, 0.81787109375], np.float64)            # = A1's F_A (test_engine6.py line 78)
    F_A = np.array([_fa[0], _fa[1], np.log(_fa[2]), -_fa[0] / _fa[2], (1 - _fa[0]) / _fa[2], -_fa[1] / _fa[2], (1 - _fa[1]) / _fa[2],
                    np.log(1920.0 / 1080.0)], np.float64).astype(np.float32)

    def build_overlay(ck, devs, *a, **k):
        m, ldev, cfgs, binfo = _real_build(ck, devs, *a, **k)
        assert binfo.get("s2p") is None and not binfo.get("g0cond") and not hasattr(m.tower_p, "s_emb") and not hasattr(m.inp, "g0"), \
            ("TEST OVERLAY only on a checkpoint WITHOUT the new keys", binfo)
        c32, c4, _ = cfgs
        d0, BF = devs[0], W.FP.BF
        with torch.no_grad():                                           # (the model's own tensors: read under the caller's inference mode)
            pc = torch.as_tensor(_cpo[:, :min(200, _cpo.shape[1])].T + W.FP.OFF, device=d0)
            tgt = float(m.inp.pose(pc).float().norm(dim=-1).median())
        sigma = tgt / (float(np.linalg.norm(F_A)) * c32.hidden_size ** 0.5)
        with torch.inference_mode(False), torch.no_grad():              # the new modules: ordinary tensors (= A1's SE / G0)
            gen = torch.Generator(device="cpu").manual_seed(20261027)
            se = W.FP.S2PEmb(c4.hidden_size, "cb0").to(device=d0, dtype=BF)
            g0 = torch.nn.Linear(W.FP.G0_DIM, c32.hidden_size, bias=False).to(device=d0, dtype=BF)
            se.weight.copy_((torch.randn(se.weight.shape, generator=gen) * (0.02 / W.FP.DEPTH ** 0.5)).to(d0, BF))
            g0.weight.copy_((torch.randn(g0.weight.shape, generator=gen) * sigma).to(d0, BF))
            for p in list(se.parameters()) + list(g0.parameters()):
                p.requires_grad_(False)
        m.tower_p.s_emb, m.inp.g0 = se, g0
        OVL.update(sigma=sigma, pose_frame_input_norm_median=tgt, s_emb_abs_sum=float(se.weight.float().abs().sum()),
                   g0_abs_sum=float(g0.weight.float().abs().sum()), g0_F_A_norm=float(g0(torch.as_tensor(F_A[None], device=d0).to(BF)).float().norm()),
                   s_emb_shape=list(se.weight.shape), g0_shape=list(g0.weight.shape))
        print(f"TEST OVERLAY on {ck}: {json.dumps(OVL)}", flush=True)
        return m, ldev, cfgs, dict(binfo, s2p="cb0", s2p_zero_init=False, g0cond=True, g0cond_zero_init=False, test_overlay=True)

    W.FP.build_model = build_overlay
W.RUNS, W.RJOBS, W.ROUT, W.RJOBS_V, W.RJOBS_V50 = (f"{OUT}/runs", f"{OUT}/render/jobs", f"{OUT}/render/out", f"{OUT}/render/jobs_vansh",
                                                     f"{OUT}/render/jobs_vansh50")
_jw = W.jwrite


def jwrite_guard(path, obj):
    assert os.path.abspath(path).startswith(OUT + "/"), ("test would write outside tests/out/gpu", path)
    return _jw(path, obj)


W.jwrite = jwrite_guard
RES = {"ck": os.environ["GENWEB_CK"], "overlay": OVERLAY, "overlay_params": OVL, "worker_md5": WORKER_MD5, "runs": {}, "checks": {}}


def check(name, ok, detail=None):
    RES["checks"][name] = {"pass": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {json.dumps(detail, default=str)[:700]}" if detail is not None else ""), flush=True)


t0 = time.time()
wk = W.Worker()                                                         # build + check_mods + capture (no feed client: never streams)
RES["load_s"] = round(time.time() - t0, 1)
print(f"worker up in {RES['load_s']} s, step {wk.step}", flush=True)
check("W0 worker up: engine follows the model's s_emb (cb0) + inp.g0, both non-zero", wk.eng.s2p == "cb0" and wk.eng.g0 is not None
      and float(wk.eng.s_emb.weight.float().abs().sum()) > 0 and float(wk.eng.g0.weight.float().abs().sum()) > 0,
      {"s2p": wk.eng.s2p, "overlay": OVERLAY, "overlay_params": OVL})
PP = {p["id"]: p for p in wk.pp}
P01, P09, P06, ROS = "01_ab63d745", "09_d9ef3c94", "06_74807fe0", "rosie_xyvR1rjH"


def req(name, person, src="start", s=1.92, po=False, **kw):
    r = {"run_id": f"gpu-{name}", "seed": 4321, "clip_s": 4.0, "hidden": True, "label": "agent test pose6 A3",
         "prefill": {"person": person, "s": s, "src": src, "pose": True, "audio": True, "words": True},
         "prompt": {"voice": "same", "persona": ""}, "params": {"pose_only": True} if po else {}, "no_live": True, "stream": False}
    r.update(kw)
    return r


def go(name, rq):
    rd = f"{OUT}/runs/{rq['run_id']}"
    os.makedirs(rd, exist_ok=True)
    try:
        wk.run(rd, rq)
    except Exception:  # noqa: BLE001
        RES["runs"][name] = {"error": traceback.format_exc()[-1500:]}
        print(f"run {name} FAILED:\n{RES['runs'][name]['error']}", flush=True)
        return None
    sm = json.load(open(f"{rd}/summary.json"))
    st = json.load(open(f"{rd}/status.json"))
    tok = np.load(f"{rd}/tokens.npz")
    RES["runs"][name] = {"state": st.get("state"), "g0f": sm["g0f"], "g0_source": sm["g0_source"], "ms_per_frame": sm["ms_per_frame"],
                         "frames": sm["frames"], "words": sm["words"][:120]}
    return sm, st, {k: tok[k] for k in tok.files}


RUNS = {"R1": req("R1", P01), "R2": req("R2", P09, "clip"), "R3": req("R3", P06, po=True), "R4": req("R4", ROS)}
want_kind = {"R1": "openings", "R2": "clip", "R3": "openings-sam", "R4": "openings"}
out = {}
for k, rq in RUNS.items():
    out[k] = go(k, rq)
    if out[k] is None:
        check(f"{k} run", False)
        continue
    sm, st, _ = out[k]
    src = sm["g0_source"]
    f_ref = np.asarray(W.G0S.g0_from_opening(src["dir"], src["kind"]), np.float32)
    check(f"{k} done, kind {want_kind[k]}, summary g0f == g0src(dir, kind)", st["state"] == "done" and src["kind"] == want_kind[k]
          and np.array_equal(np.float32(sm["g0f"]), f_ref), RES["runs"][k])

if out.get("R1"):
    f1 = out["R1"][0]["g0f"]
    out["R5"] = go("R5", req("R5", P01, g0f=f1))
    out["R6"] = go("R6", req("R6", P01, g0f=[0.0] * 8))
    if out["R5"] and out["R6"]:
        same = all(np.array_equal(out["R1"][2][k], out["R5"][2][k]) for k in ("pose", "mimi", "lane"))
        npre = out["R1"][0]["pose_prefill_frames"]
        diff6 = int((out["R1"][2]["pose"][npre:] != out["R6"][2]["pose"][npre:]).any(1).sum())
        check("R5 override = R1's own f -> tokens bit-identical to R1", same and out["R5"][0]["g0_source"]["kind"] == "request override")
        check("R6 override f = 0 -> generated pose differs from R1 (g0f reaches the model)", diff6 > 0,
              {"generated_pose_frames_differing": diff6, "of": int(out["R1"][2]["pose"].shape[0] - npre)})

try:                                                                   # E1: the engine itself refuses a G0COND start without g0f
    import torch
    x = W.MIF.assemble(W.MIF.system_block(None, None), np.full(60, W.PAD, np.int64), np.zeros((W.DEPTH, 0), np.int64),
                       np.zeros((W.DEPTH, 0), np.int64), 50, 0)
    with torch.inference_mode():
        wk.eng.start([dict(x, a_mode="empty")], 1, dict(W.DEF), f0=0)
    check("E1 engine refuses start without g0f", False, "start() accepted an instance without g0f")
except ValueError as e:
    check("E1 engine refuses start without g0f", "g0f" in str(e), str(e)[:200])

RES["total_s"] = round(time.time() - t0, 1)
json.dump(RES, open(f"{OUT}/test_worker_gpu.json", "w"), indent=1, default=str)
n_fail = sum(not v["pass"] for v in RES["checks"].values())
print(f"SUMMARY {len(RES['checks']) - n_fail}/{len(RES['checks'])} PASS in {RES['total_s']} s", flush=True)
sys.exit(1 if n_fail else 0)
