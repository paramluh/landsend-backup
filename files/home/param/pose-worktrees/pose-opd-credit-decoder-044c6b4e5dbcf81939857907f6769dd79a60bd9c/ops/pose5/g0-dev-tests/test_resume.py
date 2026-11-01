#!/usr/bin/env python3
"""pose5 G0COND test (c): RESUME from a checkpoint without inp.g0 (as dcp-17850), LR group, AdamW state, strictness.
Adapted from workstream A's code-811-s2p/tests/s2p/test_resume.py (same machinery; key / flag / group changed).

Each phase = one process (the trainer reads PPM_G0COND at import). Checkpoints are REAL trainer saves (train_ppm.save -> safety_ppm.save_local)
of the tiny FSDP2 model (CPU, world-1 gloo), loaded with the trainer's own train_ppm.s811_init + train_ppm.load (-> safety_ppm.load_local,
digest-checked), on node-local scratch SCR/ckpt/<phase>/ (never /data). Phases:
  A ORIG code, flag off: one 8,1,1 step (pose FULL + gemma), save dcp-17850 = stand-in for the real dcp-17850 (no inp.g0 key / AdamW
    state, cfg without g0cond; 8,1,1 + fresh-append blocks filled). The REAL dcp-17850 train_state.json (pose5/c/hf17850) is checked too:
    no g0cond in its cfg, LR groups = the 7 main groups.
  B COPY flag off: s811_init + load(A) strict; restored state digest == A's.
  C COPY PPM_G0COND=1: load(A) accepted with exactly inp.g0.weight missing (zero, no AdamW state); every other tensor + optimizer state == A's
    (digest without inp.g0); g0cond_from_step = 17850; inp.g0 in group 'audio' (with inp.pose / inp.aud). Then one 8,1,1 step (pose FULL,
    pose FROZEN, gemma): inp.g0 grad set by the FULL pose micro, unchanged by the FROZEN pose micro and by the text micro; AdamW: inp.g0 step
    1 vs the audio group's other params A + 1; update size ~ the group's LR (Adam step-1). Save dcp-17851 (carries inp.g0 + state + cfg).
  D COPY PPM_G0COND=1: load(C) STRICT: digest == C's save; g0cond_from_step carried (17850).
  E COPY flag off: load(C) must REFUSE (checkpoint has inp.g0).
  G COPY PPM_G0COND=1: A with inp.pose.weight removed must REFUSE (strict about everything else).
  H COPY PPM_G0COND=1: A whose train_state claims g0cond (but lacks inp.g0) must REFUSE.   I ORIG code: load(C) must REFUSE (unknown key)."""
import argparse
import json
import os
import pickle
import shutil
import sys

import torch

sys.path.insert(0, "/data/home/amartyavarma/ppm/pose5/code-811-g0/tests/g0")
import g0_cpu as H  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--phase", required=True)
ap.add_argument("--code", required=True)
ap.add_argument("--port", type=int, default=29851)
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
os.environ["PPM_TEST_LOAD_CPU"] = "1"
code = {"ORIG": H.ORIG, "COPY": H.COPY}[a.code]
sys.path.insert(0, code)
import train_ppm as T  # noqa: E402

M, SF = T.M, T.SF
assert os.path.dirname(os.path.abspath(M.__file__)) == code
H.cpu_patches()
SF.torch = H.TorchProxy(torch)
dist = H.init_dist(a.port)
dev = torch.device("cpu")
G0 = bool(getattr(T, "G0", False))
CK = f"{H.SCR}/ckpt"
GK = H.G0_KEY
OUTJ = f"{H.OUTD}/resume_{a.phase}.json"
R = {"phase": a.phase, "code": code, "G0": G0, "env_PPM_G0COND": os.environ.get("PPM_G0COND")}
REAL_TS = "/data/home/amartyavarma/ppm/pose5/c/hf17850/dl/dcp-17850/slinky-0/train_state.json"


def build():
    c32, c4, c06 = H.tiny_cfgs(M)
    model, _ = M.build(c32, c4, c06, dev, H.log, partial=True, proj_r=64, tower_vocab=M.TOWER_VOCAB, init=False, dec_ac=False,
                       dec_mode="replicate", pose_head=False, decp_bb=True, **({"g0cond": True} if G0 else {}))
    H.seeded_init(M, model, c4, c06)
    model.decp_bb_on = not T.BB_OFF
    model.decp_bb_alpha = 1.0
    model.train()
    opt, sched, _ = T.make_opt(model)
    dos = {"dec_s": T.DecOpt(model.dec_s, "dec_s"), "dec_p": T.DecOpt(model.dec_p, "dec_p")}
    return model, opt, sched, dos, M.Rot(c32, dev)


def state_digest(model, opt, dos, skip=()):
    return SF.digest_state(SF.local_state(model, T.opts_of(opt, dos), "all", skip=skip))


def micro(name):
    return T.prep(pickle.load(open(f"{H.MICD}/{name}.pkl", "rb")), dev)


def gparam(model):
    return dict(model.named_parameters()).get(GK)


def train_step(model, opt, sched, dos, rot, plan):
    """= the 8,1,1 loop body for the given micros (COEF index per micro), tower cadence 'step': one clip + AdamW + LambdaLR step."""
    info = []
    for name, ci in plan:
        gp = gparam(model)
        g_before = None if gp is None or gp.grad is None else H.loc(gp.grad).clone()
        v, loss = T.run_micro(model, micro(name), rot, dos, grad=True, cf=T.COEF[ci])
        g1 = None if gp is None or gp.grad is None else H.loc(gp.grad).clone()
        info.append({"micro": name, "coef": T.COEF[ci]["kind"] + "/" + T.COEF[ci]["mode"], "loss": loss,
                     "g0_grad_norm": None if g1 is None else float(g1.double().norm()),
                     "g0_grad_changed_by_micro": None if g1 is None else not (g_before is not None and torch.equal(g_before, g1))})
    lrs = {pg["name"]: pg["lr"] for pg in opt.param_groups}
    for pg in opt.param_groups:
        torch.nn.utils.clip_grad_norm_(pg["params"], T.CFG["clip"][pg["name"]])
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    return info, lrs


def save(model, opt, sched, dos, step, d):
    p = f"{CK}/{d}/dcp-{step}"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    assert T.save(model, opt, sched, dos, step, p, [T.HOST] * T.WORLD), "save skipped"
    return p


def expect_refusal(fn):
    try:
        fn()
    except AssertionError as e:
        return {"refused": True, "error": str(e)[:600]}
    return {"refused": False}


hosts = [T.HOST] * T.WORLD
ok = True
if a.phase == "A":
    assert not G0
    rts = json.load(open(REAL_TS))
    R["real_dcp17850"] = {"step": rts["step"], "cfg_has_g0cond": "g0cond" in rts["cfg"], "cfg_has_s2p": "s2p" in rts["cfg"],
                          "lr_groups": sorted(rts["cfg"]["lr"]), "world": rts["world"]}
    ok = rts["step"] == 17850 and "g0cond" not in rts["cfg"] and "audio" in rts["cfg"]["lr"]
    shutil.rmtree(f"{CK}/A", ignore_errors=True)
    model, opt, sched, dos, rot = build()
    T.CFG["s811"]["from_step"], T.CFG["s811"]["prev_ga_seq"] = 16861, ["pose", "speech", "speech", "text"]   # (= s811_init from a PPM ckpt)
    T.CFG["fresh"]["pose_append_at"] = [17352, 17641]                                                       # (= main() after the append)
    T.CFG["fresh"]["pose_digests"] = {T.HOST: ["plan1-digest", "npl2-digest", "npl3-digest"]}
    info, lrs = train_step(model, opt, sched, dos, rot, [("pose", 0), ("gemma", 9)])
    path = save(model, opt, sched, dos, 17850, "A")
    sd = torch.load(f"{path}/rank0.pt", map_location="cpu")
    R.update(path=path, step_info=info, lrs=lrs, digest=state_digest(model, opt, dos), ckpt_has_g0=GK in sd["model"],
             ckpt_opt_has_g0=GK in sd["opt"], ckpt_cfg_has_g0cond="g0cond" in json.load(open(f"{path}/train_state.json"))["cfg"])
    ok &= not R["ckpt_has_g0"] and not R["ckpt_opt_has_g0"] and not R["ckpt_cfg_has_g0cond"]
elif a.phase in ("B", "C", "D"):
    src = f"{CK}/A/dcp-17850" if a.phase in ("B", "C") else f"{CK}/C/dcp-17851"
    ref = json.load(open(f"{H.OUTD}/resume_{'A' if a.phase in ('B', 'C') else 'C'}.json"))
    model, opt, sched, dos, rot = build()
    if G0:
        assert float(H.loc(gparam(model)).abs().max()) == 0.0, "inp.g0 must be zero-initialised"
        if a.phase == "D":
            H.loc(gparam(model)).data.fill_(0.5)                 # poison: a strict load must overwrite it with the saved values
    T.s811_init(src)
    step = T.load(model, opt, sched, dos, src, hosts)
    R.update(src=src, step=step, cfg_g0cond=T.CFG.get("g0cond"), cfg_g0cond_from_step=T.CFG.get("g0cond_from_step"))
    skip = (GK,) if (G0 and a.phase == "C") else ()
    d_ = state_digest(model, opt, dos, skip=skip)
    R["digest_matches_ref"] = d_ == ref["digest"] if a.phase != "D" else d_ == ref["digest_after_step"]
    ok = R["digest_matches_ref"] and step == (17850 if a.phase in ("B", "C") else 17851)
    if G0:
        gp = gparam(model)
        grp = [pg["name"] for pg in opt.param_groups if any(p is gp for p in pg["params"])]
        pgrp = [pg["name"] for pg in opt.param_groups if any(p is dict(model.named_parameters())["inp.pose.weight"] for p in pg["params"])]
        R.update(g0_group=grp, pose_emb_group=pgrp, g0_abs_max=float(H.loc(gp).abs().max()), g0_has_adam_state=gp in opt.state,
                 groups=[pg["name"] for pg in opt.param_groups], sched_groups=len(sched.base_lrs))
        ok &= grp == ["audio"] == pgrp and R["sched_groups"] == len(opt.param_groups) and T.CFG["g0cond_from_step"] == 17850
        if a.phase == "C":
            ok &= R["g0_abs_max"] == 0.0 and not R["g0_has_adam_state"]
            tp = dict(model.named_parameters())["inp.pose.weight"]
            st0 = float(opt.state[tp]["step"])
            w0 = H.loc(gp).detach().clone()
            info, lrs = train_step(model, opt, sched, dos, rot, [("pose", 0), ("pose", 1), ("gemma", 9)])
            dw = (H.loc(gp).detach() - w0).double()
            touched = dw != 0
            lr_a = lrs["audio"]
            R.update(step_info=info, lrs=lrs, adam_step_g0=float(opt.state[gp]["step"]), adam_step_audio_other_before=st0,
                     adam_step_audio_other_after=float(opt.state[tp]["step"]), g0_elems_touched=int(touched.sum()), g0_elems=int(dw.numel()),
                     g0_update_absmax=float(dw.abs().max()), g0_update_median_touched=float(dw[touched].abs().median()) if touched.any() else 0.0,
                     audio_lr_used=lr_a,
                     note="Adam step 1 on a fresh state: update = lr * g / (|g| + eps) ~ lr per element with a gradient")
            pf = [x for x in info if x["coef"] == "pose/full"]
            pz = [x for x in info if x["coef"] == "pose/frozen"]
            tx = [x for x in info if x["micro"] == "gemma"]
            ok &= all(x["g0_grad_norm"] and x["g0_grad_changed_by_micro"] for x in pf)
            ok &= all(x["g0_grad_changed_by_micro"] is False for x in pz + tx)
            ok &= R["adam_step_g0"] == 1.0 and R["adam_step_audio_other_after"] == st0 + 1
            ok &= R["g0_elems_touched"] > 0 and R["g0_update_absmax"] <= lr_a * 1.001 + 1e-12 and R["g0_update_median_touched"] >= 0.5 * lr_a
            T.CFG["fresh"]["pose_digests"] = T.FRESH_SAVED_DIG[0]      # (= main(): the loaded plans' digests, re-recorded before any save; as A's test)
            shutil.rmtree(f"{CK}/C", ignore_errors=True)
            path = save(model, opt, sched, dos, 17851, "C")
            sd = torch.load(f"{path}/rank0.pt", map_location="cpu")
            tsc = json.load(open(f"{path}/train_state.json"))["cfg"]
            R.update(path=path, digest_after_step=state_digest(model, opt, dos), ckpt_has_g0=GK in sd["model"], ckpt_opt_has_g0=GK in sd["opt"],
                     ckpt_cfg_g0cond=tsc.get("g0cond"), ckpt_cfg_g0cond_from_step=tsc.get("g0cond_from_step"))
            ok &= R["ckpt_has_g0"] and R["ckpt_opt_has_g0"] and R["ckpt_cfg_g0cond"] == 1 and R["ckpt_cfg_g0cond_from_step"] == 17850
        if a.phase == "D":
            ok &= R["g0_has_adam_state"] and R["g0_abs_max"] > 0 and R["g0_abs_max"] != 0.5
    else:
        ok &= "g0cond" not in T.CFG
elif a.phase in ("E", "I"):
    src = f"{CK}/C/dcp-17851"
    model, opt, sched, dos, rot = build()
    T.s811_init(src)
    R.update(src=src, **expect_refusal(lambda: T.load(model, opt, sched, dos, src, hosts)))
    ok = R["refused"]
elif a.phase in ("G", "H"):
    src0 = f"{CK}/A/dcp-17850"
    src = f"{CK}/{a.phase}/dcp-17850"
    shutil.rmtree(f"{CK}/{a.phase}", ignore_errors=True)
    shutil.copytree(src0, src)
    if a.phase == "G":                                             # another key missing (weights + its AdamW state): must refuse
        sd = torch.load(f"{src}/rank0.pt", map_location="cpu")
        del sd["model"]["inp.pose.weight"]
        sd["opt"].pop("inp.pose.weight", None)
        torch.save(sd, f"{src}/rank0.pt")
    else:                                                          # the checkpoint claims g0cond but lacks the tensor: must refuse
        ts = json.load(open(f"{src}/train_state.json"))
        ts["cfg"].update(g0cond=1, g0cond_from_step=17850)
        json.dump(ts, open(f"{src}/train_state.json", "w"))
    model, opt, sched, dos, rot = build()
    T.s811_init(src)
    R.update(src=src, **expect_refusal(lambda: T.load(model, opt, sched, dos, src, hosts)))
    ok = R["refused"]
else:
    raise SystemExit(f"unknown phase {a.phase}")
R["PASS"] = bool(ok)
json.dump(R, open(OUTJ, "w"), indent=1, default=str)
H.log(f"phase {a.phase} ({a.code}, G0 {G0}): {'PASS' if ok else 'FAIL'} " + json.dumps({k: v for k, v in R.items() if k not in ("step_info",)}, default=str)[:1500])
dist.destroy_process_group()
