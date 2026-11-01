#!/usr/bin/env python3
"""posefix PHYS-B 10-26 (user: "very unrealistic poses ... shouldnt be able to generate non physical motion"): SAMPLING ATTRIBUTION.
Replays the user's two live calls through the CURRENT live engine (aloe-posedup-live/live_ppm_pose.py + server.prefill_inputs), same
prompt (Person 01 voice-clip preset, P = 134), the same conversation-start full prefill (2 s -> N = 26), channel A = the call's own
processed mic samples (session.npz 'a' = exactly what server.py wrote with write_a), graphed steps, under a grid of pose-sampling settings
x seeds. Per generated pose token (code frame >= N) records the sampled codes, the cb0 log-prob under the COND distribution (log_softmax
of zp0 row 0, raw logits) and each cb1..31 code's log-prob under the cond-only 1-row pose decoder (dec_p.teacher_logits on the frame's cond
hv / zb, teacher-forced on the sampled codes). Plus a REAL reference: held-out conv c1's B track teacher-forced (lane + Mimi + pose),
log-probs of the REAL pose codes. Read-only instrumentation: no engine / maths change. -> posefix/phys/runs/<call>_<setting>_s<seed>.npz
usage: python replay_attr.py <ck> [settings comma list] [seeds per setting]"""
import json
import os
import sys
import time

LIVE = "/data/home/amartyavarma/ppm/aloe-posedup-live"
sys.path.insert(0, LIVE)
_argv = list(sys.argv)
sys.argv = ["server.py"]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import server as S  # noqa: E402
sys.argv = _argv
LP, PR = S.LP, S.PR
assert os.path.dirname(os.path.realpath(LP.__file__)) == LIVE, LP.__file__
from safetensors.numpy import load_file  # noqa: E402

OUT = f"{LIVE}/posefix/phys/runs"
SESS = f"{LIVE}/stage-lb/instS4/sessions"
CALLS = {"7b5a": ("20261026-205615-7b5a", 277), "b62a": ("20261026-205651-b62a", 400),
         "b62aL": ("20261026-205651-b62a", 998)}                       # L = the whole long call (drift from ~55 s), no teacher pass
SPF, CD, OFF, LOOK = LP.SPF, LP.CD, LP.OFF, LP.VS.LOOK24
GRID = {"U": {}, "C0": {"pose_cfg_on": 0}, "Cb": {"pose_cfg_light": 0}, "Cl": {"pose_cfg_cb0": 0},
        "Ltl": {"pose_light_temp": 0.8}, "Ltp": {"pose_temp": 0.8}, "K30": {"pose_light_topk": 30}}


def log(*a):
    print(f"[attr {time.strftime('%H:%M:%S')}]", *a, flush=True)


def prefix_for(person):
    codes = np.asarray(load_file(f"/data/home/amartyavarma/video/demo-data/prompts/{person}/gt/mimi_local.safetensors")["codes"], np.int64)[:, :125]
    clip = PR.upload_clip(codes)
    clip.update(src="video_person", src_key=f"VIDEO/{person}")
    return PR.build_prefix("", clip, prose=False)[0]


@torch.no_grad()
def light_logprobs(eng, m, hv0, hn0, codes):
    """cond-only 1-row pose decoder, teacher-forced on the sampled local codes [32] -> log-probs of cb1..31 [31]."""
    with torch.cuda.device(eng.d0):
        ct = torch.as_tensor(np.asarray(codes, np.int64)[None], device=eng.d0)
        zb = (m.decp_bb(hn0) * float(m.decp_bb_alpha)) if eng.bblink else None
        lg = torch.log_softmax(eng.dec_p.teacher_logits(hv0, ct, zb)[0], -1)        # [31, CBS]
        return lg.gather(1, ct[0, 1:, None])[:, 0].float().cpu().numpy()


TEACH = {"U", "C0", "Cb", "Cl"}        # light log-probs (teacher-forced, ~0.28 s / frame on the throttled pair) only where attribution needs them


@torch.no_grad()
def run_call(eng, m, tok, key, params, seed, setting):
    sid, nf = CALLS[key]
    sj = json.load(open(f"{SESS}/{sid}/session.json"))
    A = np.load(f"{SESS}/{sid}/session.npz")["a"].astype(np.float32)
    cfg = sj["cfg"]
    pp, pre_f, pinfo = S.prefill_inputs(sj["person"], float(cfg["prefill_s"]), int(cfg["prefill_src"]), bool(cfg["pose_prefill"]),
                                        bool(cfg["prefill_audio"]), bool(cfg["prefill_words"]), tok)
    N = int(pinfo["frames"])
    assert N == int(sj["pose_prefill_frames"]), (N, sj["pose_prefill_frames"])
    pre = prefix_for(sj["person"])
    P = int(pre["P"])
    assert P == int(sj["P"]), (P, sj["P"])
    eng.start(pre, seed, params, pose_prefill=pp, prefill=pre_f)
    eng.set_params(params)
    need_tot = (nf + 2) * SPF + LOOK
    if len(A) < need_tot:
        A = np.concatenate([A, np.zeros(need_tot - len(A), np.float32)])
    wrote = 0
    codes, lp0, keep, tok_l = [], [], [], []
    for f in range(nf):
        T = P + f
        need = (f + 1) * SPF + LOOK
        eng.write_a(wrote, A[wrote:need]); wrote = need
        dec = eng.step(T)
        tokB = eng.result(T, dec)[0]
        tok_l.append(int(tokB))
        if f >= CD:
            cf = f - CD
            c = list(eng.last_pose)
            codes.append(c)
            if cf >= N:
                with torch.cuda.device(eng.d0):
                    z = torch.log_softmax(eng.zp0[0].float(), -1)
                    lp0.append(float(z[c[0]]))
                    keep.append((eng.hv_p[0:1].clone(), eng.hn_p[0:1].clone(), c))
            else:
                lp0.append(np.nan)
    t0 = time.time()
    lpl = np.full((len(codes), 31), np.nan, np.float32)
    for i, (hv0, hn0, c) in enumerate(keep if (setting in TEACH and not key.endswith("L")) else ()):
        lpl[N + i] = light_logprobs(eng, m, hv0, hn0, c)
    ms = int(1e3 * (time.time() - t0))
    out = f"{OUT}/{key}_{setting}_s{seed}.npz"
    np.savez_compressed(out, codes=np.asarray(codes, np.int16), lp0=np.asarray(lp0, np.float32), lpl=lpl, N=N, P=P, seed=seed,
                        tok=np.asarray(tok_l), params=json.dumps(params), setting=setting, call=sid)
    g = int(eng.pcfg_guided.item()) if getattr(eng, "pcfg_guided", None) is not None else -1
    gen = np.asarray(lp0[N:])
    log(f"{key} {setting} seed {seed}: {len(codes)} code frames (gen {len(codes) - N}), cfg guided {g}, cb0 cond lp mean {np.nanmean(gen):.3f}, "
        f"light cond lp sum mean {np.nanmean(lpl[N:].sum(1)):.2f}, teacher {ms} ms -> {out}")


@torch.no_grad()
def run_gt(eng, m, nf=250):
    """REAL reference: conv c1 side 0 teacher-forced (lane + Mimi + pose, eager); log-probs of the real pose codes under the cond model."""
    z = np.load("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs/c1.npz")
    pre = {"P": int(z["pre_ids_0"].shape[1]), "ids": z["pre_ids_0"], "depth": z["pre_depth_0"], "present": z["pre_present_0"]}
    P = pre["P"]
    lane, dep, pdep, A = z["laneB_0"], z["depB_0"], z["pdepB_0"], z["stem_1"].astype(np.float32)
    eng.start(pre, 1234)
    eng.write_a(0, A[:(nf + 2) * SPF + LOOK])
    codes, lp0, lpl = [], [], []
    for f in range(nf):
        T = P + f
        frc = (int(lane[f]), dep[f] if dep[f, 0] >= 0 else None, pdep[f] if pdep[f, 0] >= 0 else None)
        eng.step(T, graphs=False, force=frc)
        eng.result(T, T >= P + CD)
        if T >= P + CD and pdep[f, 0] >= 0:
            c = (pdep[f] - OFF).astype(np.int64).tolist()
            with torch.cuda.device(eng.d0):
                lp0.append(float(torch.log_softmax(eng.zp0[0].float(), -1)[c[0]]))
            lpl.append(light_logprobs(eng, m, eng.hv_p[0:1].clone(), eng.hn_p[0:1].clone(), c))
            codes.append(c)
    out = f"{OUT}/gt_c1.npz"
    np.savez_compressed(out, codes=np.asarray(codes, np.int16), lp0=np.asarray(lp0, np.float32), lpl=np.asarray(lpl, np.float32), N=0)
    log(f"GT c1: {len(codes)} frames, cb0 cond lp mean {np.mean(lp0):.3f}, light cond lp sum mean {np.mean(np.sum(lpl, 1)):.2f} -> {out}")


def main():
    ck = sys.argv[1]
    settings = sys.argv[2].split(",") if len(sys.argv) > 2 else list(GRID)
    nseed = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    keys = sys.argv[4].split(",") if len(sys.argv) > 4 else ["7b5a", "b62a"]
    os.makedirs(OUT, exist_ok=True)
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    with torch.no_grad():
        m, ldev, cfgs, info = LP.build_model(ck, devs)
        eng = LP.LivePPM(m, ldev, cfgs, devs, max_s=90.0)                 # the long call is 79.8 s
        eng.capture()
        tok = LP.IP.tok()
        log(f"engine ready (pcfg {eng.pcfg}, bblink {eng.bblink}), {torch.cuda.memory_allocated(0) / 2**30:.1f} / {torch.cuda.memory_allocated(1) / 2**30:.1f} GiB")
        for st in settings:                                            # setting-major: the key comparisons (U vs C0) land first
            if st == "GT":
                if not os.path.exists(f"{OUT}/gt_c1.npz"):
                    run_gt(eng, m)
                continue
            for key in keys:
                sj = json.load(open(f"{SESS}/{CALLS[key][0]}/session.json"))
                base = dict(sj["params"])
                for k in range(nseed):
                    seed = int(sj["seed"]) + 1000 * k
                    if os.path.exists(f"{OUT}/{key}_{st}_s{seed}.npz"):
                        log(f"skip {key} {st} {seed}"); continue
                    run_call(eng, m, tok, key, dict(base, **GRID[st]), seed, st)
    log("DONE")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
