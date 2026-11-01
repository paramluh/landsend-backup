#!/usr/bin/env python3
"""pose5 s2p test (c): RESUME from a checkpoint without tower_p.s_emb (as dcp-17850), LR group, AdamW state, strictness.

Each phase = one process (the trainer's CFG / S2P are read at import). The checkpoints are REAL trainer saves (train_ppm.save ->
safety_ppm.save_local: per-rank sharded file + rep.pt + digests + train_state.json + COMPLETE) of the tiny FSDP2 model (CPU, world-1 gloo),
loaded with the trainer's own train_ppm.s811_init + train_ppm.load (-> safety_ppm.load_local, digest-checked), all on node-local scratch
SCR/ckpt/<phase>/ (never /data, never the run's checkpoint dir). Phases (driver: stage2.sbatch):
  A ORIG code, flag off: one 8,1,1 training step (pose FULL micro + gemma text micro, clip + AdamW + LambdaLR step), save dcp-17850 = a
    stand-in for the real dcp-17850 (no s_emb key, no s_emb AdamW state; cfg without s2p; 8,1,1 + fresh-append blocks filled).
  B COPY flag off: s811_init + load(A) strict; restored state digest == A's.
  C COPY PPM_S2P=1: load(A) -> accepted with exactly tower_p.s_emb.weight missing (zero, no AdamW state); every other tensor + optimizer
    state == A's (digest without s_emb); s2p_from_step = 17850; s_emb in group 'tower_p'. Then one training step (pose FULL, pose FROZEN,
    gemma): s_emb grads non-zero after the pose micros, unchanged by the text micro; AdamW: s_emb step 1 vs the group's other params A + 1;
    update size = the group's LR (Adam step-1 bias correction). Save dcp-17851 (carries s_emb + its state + cfg s2p).
  D COPY PPM_S2P=1: load(C) STRICT (nothing may be missing): digest == C's save, s2p_from_step carried (17850).
  E COPY flag off: load(C) must REFUSE (checkpoint has s_emb).          F COPY PPM_S2P=cb0: load(C) must REFUSE (mode differs).
  G COPY PPM_S2P=1: A with tower_p.code_emb.weight removed must REFUSE (strict about everything else).
  H COPY PPM_S2P=1: A whose train_state claims s2p (but lacks s_emb) must REFUSE.    I ORIG code: load(C) must REFUSE (unknown key).
  J/K/L = a LATER restart without s_emb (plan: leg 1 flags OFF 17850 -> 18022, then restart from the 18022 save with PPM_S2P=1):
  J COPY flag off: load(A) strict, one step, save dcp-18022 (the copy's flag-off save: no s_emb, no s2p cfg keys).
  K COPY PPM_S2P=1: load(J) -> only s_emb missing, s2p_from_step = 18022, s_emb fresh (step 1) while the rest continues; save dcp-18023.
  L COPY PPM_S2P=1: load(K) STRICT, s2p_from_step carried (18022)."""
import argparse
import json
import os
import pickle
import shutil
import sys

import torch

sys.path.insert(0, "/data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p")
import s2p_cpu as H  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--phase", required=True)
ap.add_argument("--code", required=True)
ap.add_argument("--port", type=int, default=29951)
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
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
S2P = getattr(T, "S2P", None)
CK = f"{H.SCR}/ckpt/run-{os.environ['S2P_RESUME_RUN']}"                 # one fresh dir per driver run (nothing is ever deleted)
SK = H.S2P_KEY
RT = os.environ.get("S2P_RESUME_TAG", "")                       # "" = the PPM_S2P=1 suite, "_cb0" = the PPM_S2P=cb0 suite
OUTJ = f"{H.OUTD}/resume{RT}_{a.phase}.json"
R = {"phase": a.phase, "code": code, "S2P": S2P, "env_PPM_S2P": os.environ.get("PPM_S2P")}


def build():
    c32, c4, c06 = H.tiny_cfgs(M, mini=True)
    model, _ = M.build(c32, c4, c06, dev, H.log, partial=True, proj_r=64, tower_vocab=M.TOWER_VOCAB, init=False, dec_ac=False,
                       dec_mode="replicate", pose_head=False, decp_bb=True, **({"s2p": S2P} if S2P else {}))
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


def sparam(model):
    return dict(model.named_parameters()).get(SK)


def train_step(model, opt, sched, dos, rot, plan):
    """= the 8,1,1 loop body for the given micros (COEF index per micro), tower cadence 'step': one clip + AdamW + LambdaLR step."""
    info = []
    for name, ci in plan:
        g0 = None if sparam(model) is None or sparam(model).grad is None else H.loc(sparam(model).grad).clone()
        v, loss = T.run_micro(model, micro(name), rot, dos, grad=True, cf=T.COEF[ci])
        sp = sparam(model)
        g1 = None if sp is None or sp.grad is None else H.loc(sp.grad).clone()
        info.append({"micro": name, "coef": T.COEF[ci]["kind"] + "/" + T.COEF[ci]["mode"], "loss": loss,
                     "s_emb_grad_norm": None if g1 is None else float(g1.double().norm()),
                     "s_emb_grad_changed_by_micro": None if g1 is None else not (g0 is not None and torch.equal(g0, g1))})
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
    assert S2P is None
    model, opt, sched, dos, rot = build()
    T.CFG["s811"]["from_step"], T.CFG["s811"]["prev_ga_seq"] = 16861, ["pose", "speech", "speech", "text"]   # (= s811_init from a PPM ckpt)
    T.CFG["fresh"]["pose_append_at"] = [17352, 17641]                                                       # (= main() after the append)
    T.CFG["fresh"]["pose_digests"] = {T.HOST: ["plan1-digest", "npl2-digest", "npl3-digest"]}
    info, lrs = train_step(model, opt, sched, dos, rot, [("pose", 0), ("gemma", 9)])
    path = save(model, opt, sched, dos, 17850, "A")
    sd = torch.load(f"{path}/rank0.pt", map_location="cpu")
    R.update(path=path, step_info=info, lrs=lrs, digest=state_digest(model, opt, dos), ckpt_has_s_emb=SK in sd["model"],
             ckpt_opt_has_s_emb=SK in sd["opt"], ckpt_cfg_has_s2p="s2p" in json.load(open(f"{path}/train_state.json"))["cfg"],
             tower_p_step=float(opt.state[dict(model.named_parameters())["tower_p.code_emb.weight"]]["step"]))
    ok = not R["ckpt_has_s_emb"] and not R["ckpt_opt_has_s_emb"] and not R["ckpt_cfg_has_s2p"]
elif a.phase in ("B", "C", "D", "J", "K", "L"):
    # (src phase, src step, ref phase, ref digest key, s_emb skipped in the digest, expected s2p_from_step, save step / None)
    P_ = {"B": ("A", 17850, "A", "digest", False, None, None), "C": ("A", 17850, "A", "digest", True, 17850, 17851),
          "D": ("C", 17851, "C", "digest_after_step", False, 17850, None), "J": ("A", 17850, "A", "digest", False, None, 18022),
          "K": ("J", 18022, "J", "digest_after_step", True, 18022, 18023), "L": ("K", 18023, "K", "digest_after_step", False, 18022, None)}
    sph, sst, rph, rkey, skip_s, from_exp, save_at = P_[a.phase]
    src = f"{CK}/{sph}/dcp-{sst}"
    ref = json.load(open(f"{H.OUTD}/resume{RT}_{rph}.json"))
    add = a.phase in ("C", "K")                                    # flag on, checkpoint WITHOUT s_emb (dcp-17850 / a later flag-off save)
    model, opt, sched, dos, rot = build()
    if S2P:
        assert float(H.loc(sparam(model)).abs().max()) == 0.0, "s_emb must be zero-initialised"
        if not add:
            H.loc(sparam(model)).data.fill_(0.5)                 # poison: a strict load must overwrite it with the saved values
    T.s811_init(src)
    step = T.load(model, opt, sched, dos, src, hosts)
    R.update(src=src, step=step, cfg_s2p=T.CFG.get("s2p"), cfg_s2p_from_step=T.CFG.get("s2p_from_step"))
    d_ = state_digest(model, opt, dos, skip=(SK,) if skip_s else ())
    R["digest_matches_ref"] = d_ == ref[rkey]
    ok = R["digest_matches_ref"] and step == sst
    if S2P:
        sp = sparam(model)
        grp = [pg["name"] for pg in opt.param_groups if any(p is sp for p in pg["params"])]
        R.update(s_emb_group=grp, s_emb_abs_max=float(H.loc(sp).abs().max()), s_emb_has_adam_state=sp in opt.state,
                 groups=[pg["name"] for pg in opt.param_groups], sched_groups=len(sched.base_lrs))
        ok &= grp == ["tower_p"] and R["sched_groups"] == len(opt.param_groups) and T.CFG["s2p_from_step"] == from_exp
        if add:
            ok &= R["s_emb_abs_max"] == 0.0 and not R["s_emb_has_adam_state"]
        else:
            ok &= R["s_emb_has_adam_state"] and R["s_emb_abs_max"] > 0 and R["s_emb_abs_max"] != 0.5
    else:
        ok &= "s2p" not in T.CFG
    if save_at is not None:                                        # one 8,1,1 training step, then the trainer's save
        tp = dict(model.named_parameters())["tower_p.code_emb.weight"]
        st0 = float(opt.state[tp]["step"])
        w0 = H.loc(sp).detach().clone() if S2P else None
        info, lrs = train_step(model, opt, sched, dos, rot, [("pose", 0), ("pose", 1), ("gemma", 9)])
        R.update(step_info=info, lrs=lrs, adam_step_tower_p_other_before=st0, adam_step_tower_p_other_after=float(opt.state[tp]["step"]))
        ok &= R["adam_step_tower_p_other_after"] == st0 + 1
        if S2P:
            dw = (H.loc(sp).detach() - w0).double()
            touched = dw.abs().sum(1) > 0
            lr_p = lrs["tower_p"]
            R.update(adam_step_s_emb=float(opt.state[sp]["step"]), s_emb_rows_touched=int(touched.sum()),
                     s_emb_update_absmax=float(dw.abs().max()), s_emb_update_median_touched=float(dw[touched].abs().median()) if touched.any() else 0.0,
                     tower_p_lr_used=lr_p,
                     note="Adam step 1 on a fresh state: update = lr * m_hat / (sqrt(v_hat) + eps) = lr * g / (|g| + eps) ~ lr per touched element")
            pose_g = [x for x in info if x["micro"] == "pose"]
            text_g = [x for x in info if x["micro"] == "gemma"]
            ok &= all(x["s_emb_grad_norm"] and x["s_emb_grad_changed_by_micro"] for x in pose_g)
            ok &= all(x["s_emb_grad_changed_by_micro"] is False for x in text_g)
            ok &= R["adam_step_s_emb"] == 1.0
            ok &= R["s_emb_rows_touched"] > 0 and R["s_emb_update_absmax"] <= lr_p * 1.001 + 1e-12 and R["s_emb_update_median_touched"] >= 0.5 * lr_p
        T.CFG["fresh"]["pose_digests"] = T.FRESH_SAVED_DIG[0]      # (= main(): the loaded plans' digests, re-recorded before any save)
        path = save(model, opt, sched, dos, save_at, a.phase)
        sd = torch.load(f"{path}/rank0.pt", map_location="cpu")
        tsc = json.load(open(f"{path}/train_state.json"))["cfg"]
        R.update(path=path, digest_after_step=state_digest(model, opt, dos), ckpt_has_s_emb=SK in sd["model"], ckpt_opt_has_s_emb=SK in sd["opt"],
                 ckpt_cfg_s2p=tsc.get("s2p"), ckpt_cfg_s2p_from_step=tsc.get("s2p_from_step"))
        if S2P:
            ok &= R["ckpt_has_s_emb"] and R["ckpt_opt_has_s_emb"] and R["ckpt_cfg_s2p"] == S2P and R["ckpt_cfg_s2p_from_step"] == from_exp
        else:
            ok &= not R["ckpt_has_s_emb"] and not R["ckpt_opt_has_s_emb"] and R["ckpt_cfg_s2p"] is None and R["ckpt_cfg_s2p_from_step"] is None
elif a.phase in ("E", "F", "I"):
    src = f"{CK}/C/dcp-17851"
    model, opt, sched, dos, rot = build()
    T.s811_init(src)
    R.update(src=src, **expect_refusal(lambda: T.load(model, opt, sched, dos, src, hosts)))
    ok = R["refused"]
elif a.phase in ("G", "H"):
    src0 = f"{CK}/A/dcp-17850"
    src = f"{CK}/{a.phase}/dcp-17850"
    shutil.copytree(src0, src)
    if a.phase == "G":                                             # another key missing (weights + its AdamW state): must refuse
        sd = torch.load(f"{src}/rank0.pt", map_location="cpu")
        del sd["model"]["tower_p.code_emb.weight"]
        sd["opt"].pop("tower_p.code_emb.weight", None)
        torch.save(sd, f"{src}/rank0.pt")
    else:                                                          # the checkpoint claims s2p but lacks the tensor: must refuse
        ts = json.load(open(f"{src}/train_state.json"))
        ts["cfg"].update(s2p=S2P, s2p_from_step=17850)
        json.dump(ts, open(f"{src}/train_state.json", "w"))
    model, opt, sched, dos, rot = build()
    T.s811_init(src)
    R.update(src=src, **expect_refusal(lambda: T.load(model, opt, sched, dos, src, hosts)))
    ok = R["refused"]
else:
    raise SystemExit(f"unknown phase {a.phase}")
R["PASS"] = bool(ok)
json.dump(R, open(OUTJ, "w"), indent=1, default=str)
H.log(f"phase {a.phase} ({a.code}, S2P {S2P}): {'PASS' if ok else 'FAIL'} " + json.dumps({k: v for k, v in R.items() if k not in ("step_info",)}, default=str)[:1500])
dist.destroy_process_group()
