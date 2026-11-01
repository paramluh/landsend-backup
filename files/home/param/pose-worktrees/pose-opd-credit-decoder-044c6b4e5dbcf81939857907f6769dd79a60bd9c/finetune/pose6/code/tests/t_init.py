#!/usr/bin/env python3
"""pose6 B1 test stage D (2): INIT from a base with the new keys loads them EXACTLY, FSDP2 sharding / resharding included (CPU gloo).

  --phase save  (torchrun nproc 3, the BASE's code pose5/code-811-s2p = job 31101's): model_ppm.build (FSDP2, world 3) with the 31101 config
                (decp_bb, no pose head, s2p cb0, g0cond), every tensor seeded N(0, 0.02) per NAME + global row chunk (world-invariant, all
                non-zero incl. tower_p.s_emb / inp.g0), then written in BOTH init formats the FT accepts:
                  base_dcp      = safety_ppm.export_weights' exact code path (get_model_state_dict -> bf16 -> dcp.save, train_cfg.json side
                                  {step, decp_bb, decp_bb_alpha, decp_bb_off, s2p, g0cond}, COMPLETE) = ckpt-weka/ppm-v1/w-<step>'s format;
                  base_local    = ppm-local-bf16-shards-v1 (world-3 dim-0 FSDP2 shards rank<r>.pt + rep.pt + meta.json + COMPLETE) =
                                  ckpt_local_export / export_local's format (mono8020's local-11488);
                  *_nokeys      = the same without tower_p.s_emb / inp.g0 (a pre-mods base such as w-15973 / local-11488).
  --phase load  (torchrun nproc 2, pose6 code, PPM_G0COND=1 PPM_S2P=cb0): train_aloe_pose.load_init into a world-2 FSDP2 model (resharding
                3 -> 2), every LOCAL shard compared BITWISE with the expected values (the seeded full tensor -> bf16 -> fp32, sliced to the rank's
                rows; decoders whole); then the D5 rules: flags on + keys missing -> FAIL; the TEST-ONLY override -> seeded random non-zero
                (world-invariant: = the full seeded tensor's slice); override with keys present -> FAIL; flags off + keys present -> FAIL; flags off
                + pre-mods base -> exact (mono8020's path)."""
import argparse
import json
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h6 as H  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--phase", required=True, choices=("save", "load"))
a = ap.parse_args()
torch.set_num_threads(1)
SB = 4242                                                          # the base's seed
D_ = f"{H.SCR}/initbase"
STD_NEW = {"tower_p.s_emb.weight": 0.02 / 32 ** 0.5, "inp.g0.weight": 0.02}
RANK, WORLD = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])

if a.phase == "save":
    H.use_code(H.C811)
    import model_ppm as MB  # noqa: E402
    assert os.path.dirname(os.path.abspath(MB.__file__)) == H.C811
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    dist.init_process_group("gloo")
    dev = torch.device("cpu")
    c32, c4, c06 = H.tiny_cfgs(MB)
    model, _ = MB.build(c32, c4, c06, dev, H.log, partial=True, proj_r=64, tower_vocab=MB.TOWER_VOCAB, init=False, dec_ac=False,
                        dec_mode="replicate", pose_head=False, decp_bb=True, s2p="cb0", g0cond=True)
    with torch.no_grad():
        for n, p in model.named_parameters():
            r0, _ = MB._rows(p)
            MB.H2.init_rows_(H.loc(p).data, r0, tuple(p.shape), "normal", 0.02, n.replace("_checkpoint_wrapped_module.", ""), SB)
    if RANK == 0:
        shutil.rmtree(D_, ignore_errors=True)
        os.makedirs(D_)
    dist.barrier()
    msd = get_model_state_dict(model)
    assert set(H.NEW) <= set(msd), sorted(msd)[:4]
    for nm, keep in (("base_dcp", None), ("base_dcp_nokeys", H.NEW)):
        path = f"{D_}/{nm}"
        dcp.save({k: v.to(torch.bfloat16) for k, v in msd.items() if not keep or k not in keep}, checkpoint_id=path)   # = export_weights
        dist.barrier()
        if RANK == 0:
            json.dump({"step": 1, "decp_bb": 1, "decp_bb_alpha": 0.0, "decp_bb_off": 1, "s2p": "cb0", "g0cond": 1}, open(f"{path}/train_cfg.json", "w"))
            open(f"{path}/COMPLETE", "w").write("1")
    for nm, keep in (("base_local", None), ("base_local_nokeys", H.NEW)):   # = ckpt_local_export / export_local
        path = f"{D_}/{nm}"
        os.makedirs(path, exist_ok=True)
        sh, rp = {}, {}
        for n, p in model.named_parameters():
            k = n.replace("_checkpoint_wrapped_module.", "")
            if keep and k in keep:
                continue
            if hasattr(p, "to_local"):
                sh[k] = p.to_local().detach().to(torch.bfloat16).contiguous().clone()
            else:
                rp[k] = p.detach().to(torch.bfloat16).contiguous().clone()
        torch.save(sh, f"{path}/rank{RANK}.pt")
        dist.barrier()
        if RANK == 0:
            torch.save(rp, f"{path}/rep.pt")
            json.dump({"format": "ppm-local-bf16-shards-v1", "src": "pose6 t_init", "step": 1, "world": WORLD, "run_tag": "t_init"}, open(f"{path}/meta.json", "w"))
            open(f"{path}/COMPLETE", "w").write("1")
    dist.barrier()
    H.log(f"rank {RANK}: base written (world {WORLD}): {sorted(os.listdir(D_))}")
    dist.destroy_process_group()
    raise SystemExit(0)

# ---------------------------------------------------------------------------------------------------- load (pose6 trainer)
H.use_code(H.P6)
import train_aloe_pose as T  # noqa: E402

M = T.M
dist = T.dist
assert T.G0 and T.S2P == "cb0" and not T.TEST_FILL, ("run with PPM_G0COND=1 PPM_S2P=cb0", T.G0, T.S2P)
dist.init_process_group("gloo")
dev = torch.device("cpu")
c32, c4, c06 = H.tiny_cfgs(M)
EXP = {}


def build(flags):
    kw = {"s2p": "cb0", "g0cond": True} if flags else {}
    m, _ = M.build(c32, c4, c06, dev, (lambda *x: None), partial=True, proj_r=64, tower_vocab=M.TOWER_VOCAB, init=False, dec_ac=False,
                   dec_mode="replicate", pose_head=False, decp_bb=True, **kw)
    return m


def check(model, fill=None):
    """every local shard == expected bitwise -> (n tensors, n exact, list of mismatches, new params checked)."""
    n_, ok_, bad, new = 0, 0, [], {}
    for n, p in model.named_parameters():
        k = n.replace("_checkpoint_wrapped_module.", "")
        r0, r = M._rows(p)
        if fill is not None and k in H.NEW:
            full = H.full_seeded(M, k, tuple(p.shape), "normal", STD_NEW[k], int(fill))
        else:
            if k not in EXP:
                EXP[k] = H.full_seeded(M, k, tuple(p.shape), "normal", 0.02, SB).to(torch.bfloat16).float()
            full = EXP[k]
        want = full[r0:r0 + r] if hasattr(p, "to_local") else full
        got = H.loc(p).detach()
        n_ += 1
        eq = bool(torch.equal(got, want))
        ok_ += int(eq)
        if not eq:
            bad.append(k)
        if k in H.NEW:
            new[k] = {"rows_local": [int(r0), int(r)], "exact": eq, "absmax_local": float(got.abs().max()) if got.numel() else 0.0}
    return n_, ok_, bad, new


def attempt(model, path):
    for p in model.parameters():
        H.loc(p).data.fill_(123.0)                                  # sentinel: anything not loaded stays visible
    T.CFG["init"] = path
    try:
        meta, gb = T.load_init(model, path, dev)
        return {"ok": True, "meta_step": meta.get("step"), "format": meta.get("format", "local"), "unloaded": meta.get("unloaded_keys"),
                "random_fill": meta.get("newparam_random_fill")}
    except AssertionError as e:
        return {"ok": False, "error": str(e)[:300]}


R = {}
mon = build(True)
for nm in ("base_dcp", "base_local"):
    r_ = attempt(mon, f"{D_}/{nm}")
    n_, ok_, bad, new = check(mon)
    r_.update(n_tensors=n_, n_exact=ok_, mismatches=bad[:6], new_params=new, PASS=r_["ok"] and ok_ == n_ and all(v["exact"] for v in new.values()) and len(new) == 2)
    R[f"flags_on <- {nm}"] = r_
for nm in ("base_dcp_nokeys", "base_local_nokeys"):
    r_ = attempt(mon, f"{D_}/{nm}")
    r_["PASS"] = (not r_["ok"]) and "D5" in r_["error"]
    R[f"flags_on <- {nm} (must FAIL)"] = r_
T.TEST_FILL = "7"
for nm in ("base_dcp_nokeys", "base_local_nokeys"):
    r_ = attempt(mon, f"{D_}/{nm}")
    n_, ok_, bad, new = check(mon, fill=7)
    r_.update(n_tensors=n_, n_exact=ok_, mismatches=bad[:6], new_params=new,
              PASS=r_["ok"] and ok_ == n_ and all(v["exact"] and v["absmax_local"] > 0 for v in new.values()) and len(new) == 2)
    R[f"flags_on + TEST override <- {nm} (random non-zero, world-invariant)"] = r_
r_ = attempt(mon, f"{D_}/base_dcp")
r_["PASS"] = (not r_["ok"]) and "unset it" in r_["error"]
R["flags_on + TEST override <- base_dcp (keys present: must FAIL)"] = r_
T.TEST_FILL = ""
T.NEW_KEYS = ()
moff = build(False)
for nm in ("base_dcp", "base_local"):
    r_ = attempt(moff, f"{D_}/{nm}")
    r_["PASS"] = (not r_["ok"]) and "carries trained new parameters" in r_["error"]
    R[f"flags_off <- {nm} (keys present: must FAIL)"] = r_
for nm in ("base_dcp_nokeys", "base_local_nokeys"):
    r_ = attempt(moff, f"{D_}/{nm}")
    n_, ok_, bad, new = check(moff)
    r_.update(n_tensors=n_, n_exact=ok_, mismatches=bad[:6], PASS=r_["ok"] and ok_ == n_ and not new)
    R[f"flags_off <- {nm} (mono8020 path)"] = r_
ok = torch.tensor([float(all(v["PASS"] for v in R.values()))])
dist.all_reduce(ok, op=dist.ReduceOp.MIN)
allr = [None] * WORLD
dist.all_gather_object(allr, R)
if RANK == 0:
    out = {"PASS": bool(float(ok) == 1.0), "world_save": 3, "world_load": WORLD, "per_rank": allr}
    json.dump(out, open(f"{H.OUTD}/stageD_init.json", "w"), indent=1, default=str)
    for k, v in R.items():
        H.log("INIT", k, {x: y for x, y in v.items() if x not in ("new_params",)}, "new:", v.get("new_params"))
    H.log("RESULT stage D", "PASS" if out["PASS"] else "FAIL")
dist.destroy_process_group()
