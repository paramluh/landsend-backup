#!/usr/bin/env python3
"""pose5 G0COND tests (a) / (b): ONE variant of the 8,1,1 TRAINING MICRO on CPU -> JSON of exact results (compare.py compares them).
Adapted from workstream A's code-811-s2p/tests/s2p/run_variant.py (same harness: the trainer's own train_ppm.run_micro with the 8,1,1
coefficients of prod_811_fresh_append3.env -- pose micro 0 = backbone FULL, pose micro 1 = backbone FROZEN, speech micro 8 FULL, text
micro 9 FULL -- both decoders' nanosteps, the micro's one backward; sha256 of every gradient, decoders after their nanosteps).

--code ORIG (untouched ppm/code-811) | COPY (g0_cpu.COPY: code-811-g0, or the merged copy via G0CODE).
--g0 0 (flag off: no module) | 1 (inp.g0 ZERO = the init) | nz (inp.g0 non-zero: the pose-only check).
--s2p none | full | cb0 | full_nz (merged copy only: workstream A's tower_p.s_emb, zero / non-zero).
--model tiny: model_ppm.build (FSDP2 on a CPU world-1 mesh, fp32 masters, bf16 compute, fp32 reduce = the production wrapping), 811
  shapes reduced, every tensor seeded per NAME by the 811 init rules. --model real: REAL widths at reduced depth (--lb / --lt / --ld) with
  REAL 8,1,1 weights (step 17641, cache --wcache), plain module (bf16 params, fp32 decoder masters), activation checkpointing off.
Recorded for inp.g0: gradient norm, per-column norms, and the gradient recomputed from the trunk-input gradient at the SLOT positions
(hook on Inp's output: grad_W == sum_k dL/de[g0pos_k] (x) g0f_k, the exact chain rule of the index_add)."""
import argparse
import json
import os
import pickle
import sys
import time

import torch

sys.path.insert(0, "/data/home/amartyavarma/ppm/pose5/code-811-g0/tests/g0")
import g0_cpu as H  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True)
ap.add_argument("--g0", default="0", choices=("0", "1", "nz"))
ap.add_argument("--s2p", default="none", choices=("none", "full", "cb0", "full_nz", "cb0_nz"))
ap.add_argument("--model", required=True, choices=("tiny", "real"))
ap.add_argument("--micros", default="pose_full,pose_frozen,speech,gemma,chat")
ap.add_argument("--out", required=True)
ap.add_argument("--port", type=int, default=29811)
ap.add_argument("--lb", type=int, default=2)
ap.add_argument("--lt", type=int, default=2)
ap.add_argument("--ld", type=int, default=1)
ap.add_argument("--wcache", default="")
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
code = {"ORIG": H.ORIG, "COPY": H.COPY}.get(a.code, a.code)
g0_on, g0_nz = a.g0 != "0", a.g0 == "nz"
s2p_mode = None if a.s2p == "none" else a.s2p.split("_")[0]
s2p_nz = a.s2p.endswith("_nz")
assert code != H.ORIG or (not g0_on and not s2p_mode), "the original code has neither flag"
os.environ["PPM_G0COND"] = "1" if g0_on else "0"                  # (the trainer reads its flags at import)
if s2p_mode:
    os.environ["PPM_S2P"] = s2p_mode
else:
    os.environ.pop("PPM_S2P", None)
sys.path.insert(0, code)
import train_ppm as T  # noqa: E402

M = T.M
assert os.path.dirname(os.path.abspath(M.__file__)) == code and os.path.dirname(os.path.abspath(T.__file__)) == code, (M.__file__, T.__file__)
assert bool(getattr(T, "G0", False)) == g0_on and getattr(T, "S2P", None) == s2p_mode, (getattr(T, "G0", None), getattr(T, "S2P", None))
H.cpu_patches()
dist = H.init_dist(a.port)
dev = torch.device("cpu")
H.log(f"code {code} g0 {a.g0} s2p {a.s2p} model {a.model}; GA {T.SEQ}; COEF "
      f"{[(c['kind'], c['mode'], int(c['sp']), round(c['c_bb'], 6), round(c['c_s'], 6), round(c['c_p'], 6)) for c in T.COEF]}")
kw = ({"g0cond": True} if g0_on else {}) | ({"s2p": s2p_mode} if s2p_mode else {})

if a.model == "tiny":
    c32, c4, c06 = H.tiny_cfgs(M)
    model, _ = M.build(c32, c4, c06, dev, H.log, partial=True, proj_r=64, tower_vocab=M.TOWER_VOCAB, init=False, dec_ac=False,
                       dec_mode="replicate", pose_head=False, decp_bb=True, **kw)
    H.seeded_init(M, model, c4, c06, s_emb="nz" if s2p_nz else "zero", g0="nz" if g0_nz else "zero")
else:
    c32, c4, c06 = M.cfg32(a.lb), M.cfg4b(a.lt), M.cfg06(a.ld)
    with torch.device("meta"):
        model = M.PPM(c32, c4, c06, T.CFG["proj_r"], T.CFG["tower_vocab"], pose_head=False, decp_bb=True, **kw)
    for nm_, mod_ in model.named_children():
        if nm_ not in ("dec_s", "dec_p"):
            mod_.to(torch.bfloat16)                                # bf16 compute weights (FSDP's param_dtype); decoders: fp32 masters
    model.to_empty(device=dev)
    for d_ in (model.dec_s, model.dec_p):
        d_.model.rotary_emb = type(d_.model.rotary_emb)(config=c06, device=dev)
    W = torch.load(a.wcache, map_location="cpu", weights_only=True, mmap=True)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n == H.G0_KEY:
                if g0_nz:
                    p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(7)) * 0.02)
                else:
                    p.zero_()
                continue
            if n == H.S2P_KEY:
                if s2p_nz:
                    p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(5)) * 0.02 / 32 ** 0.5)
                else:
                    p.zero_()
                continue
            assert tuple(W[n].shape) == tuple(p.shape), (n, tuple(W[n].shape), tuple(p.shape))
            p.copy_(W[n])
    extra = sorted(set(W) - {n for n, _ in model.named_parameters()})
    assert not extra, ("weight cache has tensors the model lacks", extra[:4])
    for blk in list(model.layers) + list(model.tower_s.layers) + list(model.tower_p.layers):
        blk.ac = False
model.decp_bb_on = not T.BB_OFF                                    # = main(): LINK OFF (PPM_DECP_BB_OFF=1) -> never computed in training
model.decp_bb_alpha = 1.0
model.train()
names = [n for n, _ in model.named_parameters()]
assert (H.G0_KEY in names) == g0_on and hasattr(model.inp, "g0") == g0_on
if g0_on:                                                           # (code-811-g0: shared group 'audio'; merged code: own group 'g0cond')
    assert M.group_of(H.G0_KEY) in ("audio", "g0cond") and M.group_of("inp.pose.weight") == "audio"
GOF = {n: M.group_of(n) for n, _ in model.named_parameters()}
rot = M.Rot(c32, dev)
dos = {"dec_s": T.DecOpt(model.dec_s, "dec_s"), "dec_p": T.DecOpt(model.dec_p, "dec_p")}
dec_ids = {id(p) for d_ in (model.dec_s, model.dec_p) for p in d_.parameters()}
main_named = [(n, p) for n, p in model.named_parameters() if id(p) not in dec_ids]
dec_named = [(n, p) for n, p in model.named_parameters() if id(p) in dec_ids]
WHICH = {"pose_full": ("pose", 0), "pose_frozen": ("pose", 1), "speech": ("speech", 8), "gemma": ("text", 9), "chat": ("text", 9)}
res = {"code": code, "g0": a.g0, "s2p": a.s2p, "model": a.model, "n_params": len(names), "ce_chunk": M.H2.CHUNK,
       "param_shapes_sha": H.hashlib.sha256(json.dumps([(n, list(p.shape)) for n, p in model.named_parameters()]).encode()).hexdigest()[:32],
       "init_sha": {n: H.sha(p) for n, p in model.named_parameters()}, "micros": {}}
if a.model == "real":
    res["real"] = {"lb": a.lb, "lt": a.lt, "ld": a.ld, "wcache": a.wcache, "widths": [c32.hidden_size, c4.hidden_size, c06.hidden_size]}
CAP = {}


def _cap_hook(mod, inp, out):                                      # Inp's output (the trunk input e) -> keep it to read its gradient
    if out.requires_grad:
        out.retain_grad()
        CAP["e"] = out


hook = model.inp.register_forward_hook(_cap_hook) if (g0_on and a.model == "real") else None   # (tiny: FSDP2 wraps inp; skip)
for mk in a.micros.split(","):
    kind, ci = WHICH[mk]
    m = pickle.load(open(f"{H.MICD}/{mk.split('_')[0]}.pkl", "rb"))
    assert m["kind"] == kind and T.COEF[ci]["kind"] == kind, (mk, m["kind"], T.COEF[ci])
    assert (("g0pos" in m) == (kind == "pose")), "the test's pose micro must carry g0 (built with the G0COND pose cfg)"
    b = T.prep(m, dev)
    for _, p in main_named:
        p.grad = None
    CAP.clear()
    t0 = time.time()
    v, loss = T.run_micro(model, b, rot, dos, grad=True, cf=T.COEF[ci])
    dt = time.time() - t0
    r = {"coef": {k_: (v_ if not isinstance(v_, float) else float(v_)) for k_, v_ in T.COEF[ci].items()}, "seconds": round(dt, 1),
         "v": [float(x).hex() for x in v], "loss": float(loss).hex(), "v_float": [float(x) for x in v][:30], "loss_float": float(loss),
         "grad_sha": {n: H.sha(p.grad) for n, p in main_named}, "grad_vsha": {n: H.vsha(p.grad) for n, p in main_named},
         "grad_norm": {n: (None if p.grad is None else float(H.loc(p.grad).double().norm())) for n, p in main_named},
         "dec_after_sha": {n: H.sha(p) for n, p in dec_named}}
    # per main-AdamW group: the trainer's clip_grad_norm_ total norm and clip factor (max_norm = inf here: grads x 1.0, unchanged; after the shas)
    gn_, cf_ = {}, {}
    for gname in sorted({GOF[n] for n, _ in main_named}):
        ps_ = [p for n, p in main_named if GOF[n] == gname]
        if not any(p.grad is not None for p in ps_):
            gn_[gname] = None; continue
        g_ = torch.nn.utils.clip_grad_norm_(ps_, float("inf"))
        g_ = float(g_.full_tensor() if hasattr(g_, "full_tensor") else g_)
        gn_[gname] = g_.hex(); cf_[gname] = min(1.0, float(T.CFG["clip"].get(gname, 1.0)) / (g_ + 1e-6)).hex()
    r["group_norm"], r["group_clip_factor"] = gn_, cf_
    if g0_on:
        g = model.inp.g0.weight.grad
        if g is None:
            r["g0"] = {"grad": None}
        else:
            g = H.loc(g).float()
            r["g0"] = {"grad_norm": float(g.double().norm()), "col_norms": [float(x) for x in g.double().norm(dim=0)], "shape": list(g.shape),
                       "n_clips": int(len(b["g0pos"])), "n_valid": int((b["g0f"].abs().sum(1) > 0).sum()),
                       "f_col_nonzero": [bool(x) for x in (b["g0f"].abs().sum(0) > 0)]}
            if "e" in CAP and CAP["e"].grad is not None:            # chain rule of out.index_add(0, g0pos, g0(f)): dW = G[g0pos]^T f
                Ge = CAP["e"].grad.reshape(-1, CAP["e"].shape[-1])[b["g0pos"]].double()
                fb = b["g0f"].to(model.inp.g0.weight.dtype).double()
                ref = Ge.t() @ fb
                r["g0"]["recomputed_rel_err"] = float((ref - g.double()).norm() / ref.norm().clamp_min(1e-30))
    res["micros"][mk] = r
    H.log(f"{mk}: {dt:.1f}s loss {float(loss):.6f} v[:6] {[round(float(x), 5) for x in v[:6]]} "
          f"grads set {sum(x is not None for x in r['grad_sha'].values())}/{len(main_named)}" + (f" g0 {r['g0']}" if g0_on else ""))
if hook is not None:
    hook.remove()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=0)
H.log("->", a.out)
dist.destroy_process_group()
