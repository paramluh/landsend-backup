#!/usr/bin/env python3
"""pose6 B1 test stage B: ONE trainer variant on CPU -> JSON of exact results (t_compare.py compares them).

The code dir's OWN trainer (train_aloe_pose: prep, train_step_multi = pass A forward / joint K-update nanosteps of BOTH decoders with their
own AdamW / pass B forward + one backward per round, make_opt's main AdamW groups + per-group clip + one optimizer step) and model (model_ppm.build:
FSDP2 fully_shard on a CPU world-1 gloo mesh, fp32 masters, bf16 compute, fp32 reduce = the production wrapping), tiny shapes (h6.tiny_cfgs),
every tensor seeded per NAME (h6.seeded_init: identical shared tensors in every variant). One step = 2 rounds: [packed micro of the 4
prompt-type items, packed micro of items 3-4] with tokens_step normalisers (step totals). Recorded: the stats vector + loss (float.hex),
the forward terms of a grad-free forward, sha256 of EVERY parameter's gradient, the per-group clip norms, the decoders after their
nanosteps, every parameter after the main AdamW step (at the groups' peak LRs), the resolved CFG. pose6 G0COND on: the gradient path of
inp.g0 (hooks: its output gradient vs the trunk-input gradient at the SLOT positions)."""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h6 as H  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True)
ap.add_argument("--variant", required=True)
ap.add_argument("--micros", required=True)                      # tag of SCR/micros_<tag>.pkl
ap.add_argument("--new", default="zero", choices=("zero", "nz"))
ap.add_argument("--port", type=int, default=29931)
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
H.use_code(a.code)
import train_aloe_pose as T  # noqa: E402

M = T.M
assert os.path.dirname(os.path.abspath(M.__file__)) == a.code and os.path.dirname(os.path.abspath(T.__file__)) == a.code
H.cpu_patches()
dist = H.init_dist(a.port)
dev = torch.device("cpu")
S2P, G0 = getattr(T, "S2P", None), bool(getattr(T, "G0", False))
H.log(f"variant {a.variant}: code {a.code} S2P {S2P} G0 {G0} new {a.new}; TRAIN_GROUPS {T.TRAIN_GROUPS}")
kw = ({"s2p": S2P} if S2P else {}) | ({"g0cond": True} if G0 else {})
c32, c4, c06 = H.tiny_cfgs(M)
model, _ = M.build(c32, c4, c06, dev, H.log, partial=True, proj_r=64, tower_vocab=M.TOWER_VOCAB, init=False, dec_ac=False,
                   dec_mode="replicate", pose_head=False, decp_bb=True, **kw)
H.seeded_init(M, model, c4, c06, new=a.new)
model.decp_bb_alpha = T.CFG["decp_bb_alpha"]
BB_OFF = bool(getattr(T, "BB_OFF", False))
if BB_OFF:                                                        # = train_aloe_pose.main's LINK OFF setup
    model.decp_bb_on = False
rot = M.Rot(c32, dev)
mic = pickle.load(open(f"{H.SCR}/micros_{a.micros}.pkl", "rb"))
bl = [mic["packed"], mic["packed2"]]
oc = np.sum([np.asarray(b_["own"]).sum(0) for b_ in bl], 0)
bts = [T.prep(b_, dev, oc[0], oc[1], oc[3]) for b_ in bl]     # tokens_step normalisers, WORLD 1 (= the trainer's tot / WORLD)
OUT = {"variant": a.variant, "code": a.code, "S2P": S2P, "G0": G0, "BB_OFF": BB_OFF, "new": a.new, "cfg": json.loads(json.dumps(T.CFG, default=str)),
       "train_groups": list(T.TRAIN_GROUPS),
       "before": {n.replace("_checkpoint_wrapped_module.", ""): H.sha(p) for n, p in model.named_parameters()}}

# ---- grad-free forward terms (pass-A-style) of round 0
with torch.no_grad():
    out = model(bts[0], rot, rot)
OUT["fwd"] = {k: H.sha(out[k]) for k in ("text", "cb0", "pcb0", "hv_s", "hv_p", "zb_p") if k in out}
OUT["zb_p_computed"] = "zb_p" in out
OUT["fwd_val"] = {k: float(out[k]) for k in ("text", "cb0", "pcb0")}
del out

# ---- G0COND gradient path hooks (pass B only: tensors that require grad)
cap = {"e": [], "y": []}
if G0:
    def h_inp(mod, inp, out):
        if out.requires_grad:
            out.register_hook(lambda g: cap["e"].append(g.detach().clone()))

    def h_g0(mod, inp, out):
        if out.requires_grad:
            cap.setdefault("f", []).append(inp[0].detach().clone())
            out.register_hook(lambda g: cap["y"].append(g.detach().clone()))
    model.inp.register_forward_hook(h_inp)
    model.inp.g0.register_forward_hook(h_g0)

dos = {"dec_s": T.DecOpt(model.dec_s, T.CFG["dec_lr"], T.CFG["dec_clip"]), "dec_p": T.DecOpt(model.dec_p, T.CFG["dec_p_lr"], T.CFG["dec_p_clip"])}
opt, sched, by, frozen, decn = T.make_opt(model)
v, loss, extra = T.train_step_multi(model, bts, rot, dos)
OUT["stats"] = [float(x).hex() for x in v]
OUT["loss"] = float(loss).hex()
OUT["passab"] = [extra["passab_loss_rel"], extra["passab_hv_abs"]]
names = {p: n.replace("_checkpoint_wrapped_module.", "") for n, p in model.named_parameters()}
OUT["grads"] = {names[p]: H.sha(p.grad) for p in names}
OUT["grad_none"] = sorted(names[p] for p in names if p.grad is None)
OUT["dec_after_nano"] = {names[p]: H.sha(p) for d in (model.dec_s, model.dec_p) for p in d.parameters()}
OUT["groups"] = [(pg["name"], len(pg["params"])) for pg in opt.param_groups]
for n_ in H.NEW:
    if n_ in names.values():
        p = next(q for q, m_ in names.items() if m_ == n_)
        g = H.loc(p.grad).float()
        OUT[f"grad_norm[{n_}]"] = float(g.norm())
        OUT[f"grad_rows_nonzero[{n_}]"] = int((g.abs().sum(1) > 0).sum())
        if n_ == "tower_p.s_emb.weight":                          # rows with a gradient = exactly the cb0 codes B's depth carries (all rounds)
            codes = sorted(set(np.concatenate([np.asarray(b_["depth"])[1][:, 0][np.asarray(b_["depth"])[1][:, 0] >= 0] for b_ in bl]).tolist()))
            rows = torch.nonzero(g.abs().sum(1) > 0)[:, 0].tolist()
            OUT["s_emb_rows_eq_codes_present"] = bool(set(rows) <= set(codes))
            OUT["s_emb_rows_nonzero_vs_codes"] = [len(rows), len(codes)]
if G0:                                                            # (4b) inp.g0's gradient comes ONLY through the SLOT positions
    r_ = []
    W = next(q for q, m_ in names.items() if m_ == "inp.g0.weight")
    exp = torch.zeros(W.shape, dtype=torch.float64)
    for r, b_ in enumerate(bts):
        gpos = b_["g0pos"]
        ge, gy = cap["e"][r], cap["y"][r]
        r_.append({"round": r, "g0pos": gpos.tolist(), "y_grad_eq_trunk_input_grad_at_SLOT": bool(torch.equal(gy, ge[gpos].to(gy.dtype))),
                   "trunk_input_grad_rows": int(ge.shape[0])})
        exp += gy.double().T @ cap["f"][r].double()                 # dL/dW = sum_k dL/dy_k (x) f_k (f as the layer saw it: bf16)
    g = H.loc(W.grad).double()
    OUT["g0_grad_path"] = {"per_round": r_, "W_grad_vs_sum_dy_f_max_abs": float((g - exp).abs().max()), "W_grad_max_abs": float(g.abs().max()),
                           "rel": float((g - exp).abs().max() / g.abs().max())}
    H.log("g0 grad path", OUT["g0_grad_path"])

# ---- main AdamW step: per-group clip (own groups), at the groups' peak LRs
gn = {}
for pg in opt.param_groups:
    g_ = torch.nn.utils.clip_grad_norm_(pg["params"], T.CFG["clip"][pg["name"]])
    gn[pg["name"]] = float(g_.full_tensor() if hasattr(g_, "full_tensor") else g_).hex()
    pg["lr"] = T.CFG["lr"][pg["name"]]
opt.step()
OUT["gn"] = gn
OUT["after_step"] = {names[p]: H.sha(p) for p in names}
for n_ in H.NEW:
    if n_ in names.values():
        p = next(q for q, m_ in names.items() if m_ == n_)
        OUT[f"after_step_absmax[{n_}]"] = float(H.loc(p).abs().max())
# ---- the trainer's two exports (node scratch only): DCP export_weights (train_cfg.json side) + node-local export_local (meta.json)
import shutil  # noqa: E402
ex = f"{H.SCR}/exp_{a.variant}"
shutil.rmtree(f"{ex}/dcp", ignore_errors=True); shutil.rmtree(f"{ex}/local", ignore_errors=True)
os.makedirs(ex, exist_ok=True)
T.CFG["export_wait_min"] = 0.0
assert T.export_weights(model, 1, f"{ex}/dcp", 0.0) == f"{ex}/dcp"
T.export_local(model, 1, f"{ex}/local")
tc = json.load(open(f"{ex}/dcp/train_cfg.json"))
mt = json.load(open(f"{ex}/local/meta.json"))
OUT["export"] = {"train_cfg": {k: v for k, v in tc.items() if k != "cfg"}, "train_cfg_cfg_eq_CFG": tc["cfg"] == OUT["cfg"],
                 "meta_wo_time": {k: v for k, v in mt.items() if k != "save_time"}, "meta_keys": sorted(mt), "side": mt.get("side")}
os.makedirs(H.OUTD, exist_ok=True)
json.dump(OUT, open(f"{H.OUTD}/stageB_{a.variant}.json", "w"), indent=1, default=str)
H.log(f"variant {a.variant} done: loss {float.fromhex(OUT['loss']):.6f} groups {OUT['groups']} gn {[(k, float.fromhex(x)) for k, x in gn.items()]}")
dist.destroy_process_group()
