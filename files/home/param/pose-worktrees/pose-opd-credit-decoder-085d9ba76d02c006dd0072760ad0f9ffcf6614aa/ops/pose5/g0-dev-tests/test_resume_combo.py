#!/usr/bin/env python3
"""pose5 G0COND + POSE-S2P COMBINED test (c): RESUME from a checkpoint with NEITHER new key (as dcp-17850) under any flag combination
(PPM_G0COND 0/1 x PPM_S2P none/cb0/full), LR groups, AdamW state, strictness. Generalisation of test_resume.py (G0COND only) for the merged
copy (G0CODE=code-811-s2p); flags are read from the env at import exactly as the trainer does. Phase C / D / E checks apply to the set of
NEW keys of the run (inp.g0.weight -> group 'audio' and grads from backbone-FULL pose micros only; tower_p.s_emb.weight -> group 'tower_p'
and grads from every pose micro, never text). Extra phase X: the C checkpoint loaded with the OTHER combination must refuse.

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
S2P = getattr(T, "S2P", None)
NEWK = ([H.G0_KEY] if G0 else []) + ([H.S2P_KEY] if S2P else [])
GRP = {H.G0_KEY: "g0cond", H.S2P_KEY: "s2p"}                                   # OWN groups (main 10-27, audit S7)
JOINK = {H.G0_KEY: "audio", H.S2P_KEY: "tower_p"}                               # the group whose LR / clip / schedule each copies
SIB = {H.G0_KEY: "inp.pose.weight", H.S2P_KEY: "tower_p.code_emb.weight"}      # an existing parameter of the JOIN group
CK = f"{H.SCR}/ckpt"
GK = H.G0_KEY
TAG = f"g0{int(G0)}_s2p{S2P or 'none'}"
OUTJ = f"{H.OUTD}/resume_combo_{a.phase}_{TAG}.json" if a.phase != "X" else \
    f"{H.OUTD}/resume_combo_X_{TAG}_{a.code}_from_{os.environ['G0_X_SRC'].rstrip('/').split('/')[-3]}.json"
R = {"phase": a.phase, "code": code, "G0": G0, "S2P": S2P, "new_keys": NEWK, "env_PPM_G0COND": os.environ.get("PPM_G0COND"),
     "env_PPM_S2P": os.environ.get("PPM_S2P")}
REAL_TS = "/data/home/amartyavarma/ppm/pose5/c/hf17850/dl/dcp-17850/slinky-0/train_state.json"


def build():
    c32, c4, c06 = H.tiny_cfgs(M)
    model, _ = M.build(c32, c4, c06, dev, H.log, partial=True, proj_r=64, tower_vocab=M.TOWER_VOCAB, init=False, dec_ac=False,
                       dec_mode="replicate", pose_head=False, decp_bb=True, **({"g0cond": True} if G0 else {}), **({"s2p": S2P} if S2P else {}))
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


def gparam(model, k=None):
    return dict(model.named_parameters()).get(k or GK)


def train_step(model, opt, sched, dos, rot, plan):
    """= the 8,1,1 loop body for the given micros (COEF index per micro), tower cadence 'step': one clip + AdamW + LambdaLR step."""
    info = []
    for name, ci in plan:
        gb = {k: (None if gparam(model, k).grad is None else H.loc(gparam(model, k).grad).clone()) for k in NEWK}
        v, loss = T.run_micro(model, micro(name), rot, dos, grad=True, cf=T.COEF[ci])
        rec = {"micro": name, "coef": T.COEF[ci]["kind"] + "/" + T.COEF[ci]["mode"], "loss": loss}
        for k in NEWK:
            g1 = None if gparam(model, k).grad is None else H.loc(gparam(model, k).grad).clone()
            rec[f"{k}:grad_norm"] = None if g1 is None else float(g1.double().norm())
            rec[f"{k}:changed"] = None if g1 is None else not (gb[k] is not None and torch.equal(gb[k], g1))
        info.append(rec)
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
CKT = f"{CK}/combo-{TAG}"                                          # this combination's own C / X checkpoints (A: CK/A, shared)
if a.phase == "A":                                                 # = test_resume.py's A (ORIG code, no new key): the dcp-17850 stand-in
    assert not G0 and not S2P
    rts = json.load(open(REAL_TS))
    R["real_dcp17850"] = {"step": rts["step"], "cfg_has_g0cond": "g0cond" in rts["cfg"], "cfg_has_s2p": "s2p" in rts["cfg"],
                          "lr_groups": sorted(rts["cfg"]["lr"]), "world": rts["world"]}
    ok = rts["step"] == 17850 and "g0cond" not in rts["cfg"] and "s2p" not in rts["cfg"]
    shutil.rmtree(f"{CK}/A", ignore_errors=True)
    model, opt, sched, dos, rot = build()
    T.CFG["s811"]["from_step"], T.CFG["s811"]["prev_ga_seq"] = 16861, ["pose", "speech", "speech", "text"]
    T.CFG["fresh"]["pose_append_at"] = [17352, 17641]
    T.CFG["fresh"]["pose_digests"] = {T.HOST: ["plan1-digest", "npl2-digest", "npl3-digest"]}
    info, lrs = train_step(model, opt, sched, dos, rot, [("pose", 0), ("gemma", 9)])
    path = save(model, opt, sched, dos, 17850, "A")
    sd = torch.load(f"{path}/rank0.pt", map_location="cpu")
    tsa = json.load(open(f"{path}/train_state.json"))["cfg"]
    R.update(path=path, lrs=lrs, digest=state_digest(model, opt, dos), ckpt_new_keys=[k for k in (H.G0_KEY, H.S2P_KEY) if k in sd["model"] or k in sd["opt"]],
             ckpt_cfg_flags=[k for k in ("g0cond", "s2p") if k in tsa])
    ok &= not R["ckpt_new_keys"] and not R["ckpt_cfg_flags"]
elif a.phase in ("C", "D"):                                        # C: load A with this combination (+ one step, save); D: strict reload of C
    src = f"{CK}/A/dcp-17850" if a.phase == "C" else f"{CKT}/C/dcp-17851"
    ref = json.load(open(f"{H.OUTD}/resume_combo_A_g00_s2pnone.json" if a.phase == "C" else f"{H.OUTD}/resume_combo_C_{TAG}.json"))
    model, opt, sched, dos, rot = build()
    for k in NEWK:
        assert float(H.loc(gparam(model, k)).abs().max()) == 0.0, (k, "must be zero-initialised")
        if a.phase == "D":
            H.loc(gparam(model, k)).data.fill_(0.5)              # poison: a strict load must overwrite it with the saved values
    T.s811_init(src)
    step = T.load(model, opt, sched, dos, src, hosts)
    R.update(src=src, step=step, cfg={k: T.CFG.get(k) for k in ("g0cond", "g0cond_from_step", "s2p", "s2p_from_step")})
    d_ = state_digest(model, opt, dos, skip=tuple(NEWK) if a.phase == "C" else ())
    R["digest_matches_ref"] = d_ == (ref["digest"] if a.phase == "C" else ref["digest_after_step"])
    ok = R["digest_matches_ref"] and step == (17850 if a.phase == "C" else 17851)
    params = dict(model.named_parameters())
    R["keys"] = {}
    for k in NEWK:
        gp = params[k]
        grp = [pg["name"] for pg in opt.param_groups if any(p is gp for p in pg["params"])]
        sgrp = [pg["name"] for pg in opt.param_groups if any(p is params[SIB[k]] for p in pg["params"])]
        R["keys"][k] = {"group": grp, "sibling_group": sgrp, "abs_max": float(H.loc(gp).abs().max()), "has_adam_state": gp in opt.state}
        ok &= grp == [GRP[k]] and sgrp == [JOINK[k]]
        ig, ij = [pg["name"] for pg in opt.param_groups].index(GRP[k]), [pg["name"] for pg in opt.param_groups].index(JOINK[k])
        R["keys"][k].update(group_lr_now=opt.param_groups[ig]["lr"], join_lr_now=opt.param_groups[ij]["lr"],
                            group_initial_lr=opt.param_groups[ig]["initial_lr"], join_initial_lr=opt.param_groups[ij]["initial_lr"],
                            lam_equal_over_steps=all(sched.lr_lambdas[ig](s_) == sched.lr_lambdas[ij](s_) for s_ in (0, 1, 50, 99, 100, 17850, 17851, 30000)),
                            clip=[T.CFG["clip"][GRP[k]], T.CFG["clip"][JOINK[k]]], betas_wd=[[opt.param_groups[i_][x] for x in ("betas", "weight_decay")] for i_ in (ig, ij)])
        kk_ = R["keys"][k]
        ok &= kk_["group_lr_now"] == kk_["join_lr_now"] and kk_["group_initial_lr"] == kk_["join_initial_lr"] and kk_["lam_equal_over_steps"]
        ok &= kk_["clip"][0] == kk_["clip"][1] and kk_["betas_wd"][0] == kk_["betas_wd"][1]
    ok &= (T.CFG.get("g0cond_from_step") == 17850) == G0 and (T.CFG.get("s2p_from_step") == 17850) == bool(S2P)
    ok &= len(sched.base_lrs) == len(opt.param_groups)
    if a.phase == "C":
        ok &= all(v["abs_max"] == 0.0 and not v["has_adam_state"] for v in R["keys"].values())
        st0 = {k: float(opt.state[params[SIB[k]]]["step"]) for k in NEWK}
        R["groups_after_load"] = [pg["name"] for pg in opt.param_groups]
        w0 = {k: H.loc(params[k]).detach().clone() for k in NEWK}
        info, lrs = train_step(model, opt, sched, dos, rot, [("pose", 0), ("pose", 1), ("gemma", 9)])
        R.update(step_info=info, lrs=lrs)
        for k in NEWK:
            dw = (H.loc(params[k]).detach() - w0[k]).double()
            lr_ = lrs[GRP[k]]
            touched = dw != 0
            kk = R["keys"][k]
            kk["first_step_lr_equals_join"] = lrs[GRP[k]] == lrs[JOINK[k]]
            ok &= kk["first_step_lr_equals_join"]
            kk.update(adam_step=float(opt.state[params[k]]["step"]), sibling_step_before=st0[k], sibling_step_after=float(opt.state[params[SIB[k]]]["step"]),
                      touched=int(touched.sum()), update_absmax=float(dw.abs().max()),
                      update_median_touched=float(dw[touched].abs().median()) if touched.any() else 0.0, lr=lr_)
            pf = [x for x in info if x["coef"] == "pose/full"]
            pz = [x for x in info if x["coef"] == "pose/frozen"]
            tx = [x for x in info if x["micro"] == "gemma"]
            ok &= all(x[f"{k}:grad_norm"] and x[f"{k}:changed"] for x in pf)
            ok &= all(x[f"{k}:changed"] is False for x in tx)
            if k == H.G0_KEY:                                      # trunk input: the backbone-FROZEN pose micro gives no gradient
                ok &= all(x[f"{k}:changed"] is False for x in pz)
            else:                                                  # pose tower input: every pose micro (frozen trunk, live tower side)
                ok &= all(x[f"{k}:grad_norm"] and x[f"{k}:changed"] for x in pz)
            ok &= kk["adam_step"] == 1.0 and kk["sibling_step_after"] == st0[k] + 1
            ok &= kk["touched"] > 0 and kk["update_absmax"] <= lr_ * 1.001 + 1e-12 and kk["update_median_touched"] >= 0.5 * lr_
        T.CFG["fresh"]["pose_digests"] = T.FRESH_SAVED_DIG[0]      # (= main(): the loaded plans' digests, re-recorded before any save; as A's test)
        shutil.rmtree(f"{CKT}/C", ignore_errors=True)
        path = save(model, opt, sched, dos, 17851, f"combo-{TAG}/C")
        sd = torch.load(f"{path}/rank0.pt", map_location="cpu")
        tsc = json.load(open(f"{path}/train_state.json"))["cfg"]
        R.update(path=path, digest_after_step=state_digest(model, opt, dos), ckpt_keys={k: [k in sd["model"], k in sd["opt"]] for k in NEWK},
                 ckpt_cfg={k: tsc.get(k) for k in ("g0cond", "g0cond_from_step", "s2p", "s2p_from_step")})
        ok &= all(v == [True, True] for v in R["ckpt_keys"].values())
        ok &= (tsc.get("g0cond") == 1 and tsc.get("g0cond_from_step") == 17850) if G0 else "g0cond" not in tsc
        ok &= (tsc.get("s2p") == S2P and tsc.get("s2p_from_step") == 17850) if S2P else "s2p" not in tsc
    else:
        ok &= all(v["has_adam_state"] and v["abs_max"] > 0 and v["abs_max"] != 0.5 for v in R["keys"].values())
elif a.phase == "X":                                               # load another combination's C checkpoint (env PPM_* = the OTHER flags): refuse
    src = os.environ["G0_X_SRC"]
    model, opt, sched, dos, rot = build()
    T.s811_init(src)
    R.update(src=src, **expect_refusal(lambda: T.load(model, opt, sched, dos, src, hosts)))
    ok = R["refused"]
else:
    raise SystemExit(f"unknown phase {a.phase}")
R["PASS"] = bool(ok)
json.dump(R, open(OUTJ, "w"), indent=1, default=str)
H.log(f"phase {a.phase} ({a.code}, {TAG}): {'PASS' if ok else 'FAIL'} " + json.dumps({k: v for k, v in R.items() if k not in ("step_info",)}, default=str)[:1800])
dist.destroy_process_group()
