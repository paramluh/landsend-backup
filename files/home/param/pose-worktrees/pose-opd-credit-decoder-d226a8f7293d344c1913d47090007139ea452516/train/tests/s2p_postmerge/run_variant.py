#!/usr/bin/env python3
"""pose5 s2p tests (a) / (b): ONE variant of the 8,1,1 TRAINING MICRO on CPU -> JSON of exact results (compare_variants.py compares them).

The micro runs through the trainer's own train_ppm.run_micro (the code dir's: --code ORIG = untouched ppm/code-811, --code COPY = the s2p
copy) with the 8,1,1 coefficients of prod_811_fresh_append3.env (COEF: pose micro 0 = backbone FULL, pose micro 1 = backbone FROZEN, speech
micro 8 FULL, text micro 9 FULL), i.e. the forward (model_ppm 8,1,1 bb / bridge / sp_parts path), BOTH decoders' nanosteps (K = 8 updates
with their own AdamW steps, all-reduce over a world-1 gloo group) and the micro's one backward with the roots + decoder input gradients.
Recorded per micro: the stats vector + loss (float.hex), a sha256 of EVERY parameter's gradient (bytes; plus a value-level digest), the
decoders' parameters after their nanosteps, and for tower_p.s_emb its gradient norm / non-zero rows vs the speech codes present.

--model tiny: model_ppm.build (FSDP2 fully_shard on a CPU world-1 mesh, fp32 masters, bf16 MixedPrecision compute + fp32 grad reduce =
  the production wrapping), 811 shapes reduced (backbone 2 x H128, towers 2 x H64, decoders 1 x H64), every tensor seeded per NAME by the
  811 init rules (s2p_cpu.seeded_init).
--model real: the REAL widths (Qwen3-32B backbone H5120 / vocab 151936, Qwen3-4B towers H2560 / tower vocab 151682, 0.6B decoders H1024)
  at reduced depth (--lb backbone, --lt tower, --ld decoder layers), REAL weights of 8,1,1 step 17641 (ckpt-infer/ppm-v1/local-17641,
  cached by real_weights.py on node scratch), plain module (no FSDP: bf16 parameters for everything but the decoders, which keep fp32
  masters as in production), activation checkpointing off (recompute = the same maths).
--variant off (no s2p) | full | cb0 (s_emb ZERO = the init) | full_nz | cb0_nz (s_emb non-zero: the optional pose-only check)."""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p_postmerge")
import s2p_cpu as H  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True)
ap.add_argument("--variant", required=True, choices=("off", "full", "cb0", "full_nz", "cb0_nz"))
ap.add_argument("--model", required=True, choices=("tiny", "real"))
ap.add_argument("--micros", default="pose_full,pose_frozen,speech,gemma,chat")
ap.add_argument("--out", required=True)
ap.add_argument("--port", type=int, default=29911)
ap.add_argument("--lb", type=int, default=2)
ap.add_argument("--lt", type=int, default=2)
ap.add_argument("--ld", type=int, default=1)
ap.add_argument("--wcache", default="")
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
code = {"ORIG": H.ORIG, "COPY": H.COPY}.get(a.code, a.code)
mode = None if a.variant == "off" else a.variant.split("_")[0]
nz = a.variant.endswith("_nz")
assert code == H.COPY or mode is None, "the original code has no s2p"
sys.path.insert(0, code)
import train_ppm as T  # noqa: E402

M = T.M
assert os.path.dirname(os.path.abspath(M.__file__)) == code and os.path.dirname(os.path.abspath(T.__file__)) == code, (M.__file__, T.__file__)
H.cpu_patches()
dist = H.init_dist(a.port)
dev = torch.device("cpu")
H.log(f"code {code} variant {a.variant} model {a.model}; trainer S2P {getattr(T, 'S2P', '<none: original code>')}; GA {T.SEQ}; "
      f"COEF {[(c['kind'], c['mode'], int(c['sp']), round(c['c_bb'], 6), round(c['c_s'], 6), round(c['c_p'], 6)) for c in T.COEF]}")
kw = {"s2p": mode} if mode else {}

if a.model == "tiny":
    c32, c4, c06 = H.tiny_cfgs(M)
    model, _ = M.build(c32, c4, c06, dev, H.log, partial=True, proj_r=64, tower_vocab=M.TOWER_VOCAB, init=False, dec_ac=False,
                       dec_mode="replicate", pose_head=False, decp_bb=True, **kw)
    H.seeded_init(M, model, c4, c06, s_emb="nz" if nz else "zero")
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
            if n == H.S2P_KEY:
                if nz:
                    p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(5)) * 0.02 / 32 ** 0.5)
                else:
                    p.zero_()
                continue
            assert tuple(W[n].shape) == tuple(p.shape), (n, tuple(W[n].shape), tuple(p.shape))
            p.copy_(W[n])
    extra = sorted(set(W) - {n for n, _ in model.named_parameters()})
    assert not extra, ("weight cache has tensors the model lacks", extra[:4])
    del W                                                          # (drop the mmap: its pages no longer count against the job's memory)
    for blk in list(model.layers) + list(model.tower_s.layers) + list(model.tower_p.layers):
        blk.ac = False
model.decp_bb_on = not T.BB_OFF                                    # = main(): LINK OFF (PPM_DECP_BB_OFF=1) -> never computed in training
model.decp_bb_alpha = 1.0
model.train()
assert hasattr(model.tower_p, "s_emb") == bool(mode)
names = [n for n, _ in model.named_parameters()]
assert (H.S2P_KEY in names) == bool(mode)
if mode:
    assert M.group_of(H.S2P_KEY) == "s2p"                         # (postmerge: own group, JOIN tower_p)
rot = M.Rot(c32, dev)
dos = {"dec_s": T.DecOpt(model.dec_s, "dec_s"), "dec_p": T.DecOpt(model.dec_p, "dec_p")}
dec_ids = {id(p) for d_ in (model.dec_s, model.dec_p) for p in d_.parameters()}
main_named = [(n, p) for n, p in model.named_parameters() if id(p) not in dec_ids]
dec_named = [(n, p) for n, p in model.named_parameters() if id(p) in dec_ids]
WHICH = {"pose_full": ("pose", 0), "pose_frozen": ("pose", 1), "speech": ("speech", 8), "gemma": ("text", 9), "chat": ("text", 9)}
res = {"code": code, "variant": a.variant, "model": a.model, "s2p": mode, "n_params": len(names), "ce_chunk": M.H2.CHUNK,
       "param_shapes_sha": H.hashlib.sha256(json.dumps([(n, list(p.shape)) for n, p in model.named_parameters()]).encode()).hexdigest()[:32],
       "init_sha": {n: H.sha(p) for n, p in model.named_parameters()}, "micros": {}}
if a.model == "real":
    res["real"] = {"lb": a.lb, "lt": a.lt, "ld": a.ld, "wcache": a.wcache, "widths": [c32.hidden_size, c4.hidden_size, c06.hidden_size]}
for mk in a.micros.split(","):
    kind, ci = WHICH[mk]
    m = pickle.load(open(f"{H.MICD}/{mk.split('_')[0]}.pkl", "rb"))
    assert m["kind"] == kind and T.COEF[ci]["kind"] == kind, (mk, m["kind"], T.COEF[ci])
    b = T.prep(m, dev)
    for _, p in main_named:
        p.grad = None
    t0 = time.time()
    v, loss = T.run_micro(model, b, rot, dos, grad=True, cf=T.COEF[ci])
    dt = time.time() - t0
    r = {"coef": {k_: (v_ if not isinstance(v_, float) else float(v_)) for k_, v_ in T.COEF[ci].items()}, "seconds": round(dt, 1),
         "v": [float(x).hex() for x in v], "loss": float(loss).hex(), "v_float": [float(x) for x in v][:24], "loss_float": float(loss),
         "grad_sha": {n: H.sha(p.grad) for n, p in main_named}, "grad_vsha": {n: H.vsha(p.grad) for n, p in main_named},
         "grad_norm": {n: (None if p.grad is None else float(H.loc(p.grad).double().norm())) for n, p in main_named},
         "dec_after_sha": {n: H.sha(p) for n, p in dec_named}}
    if mode:
        g = model.tower_p.s_emb.weight.grad
        if g is None:
            r["s_emb"] = {"grad": None}
        else:
            g = H.loc(g).float()
            nzr = torch.nonzero(g.abs().sum(1) > 0)[:, 0]
            d1 = b["depth"][1]
            hs = d1[:, 0] >= 0
            codes = d1[hs] if mode == "full" else d1[hs][:, :1]
            present = torch.unique(codes.reshape(-1))
            r["s_emb"] = {"grad_norm": float(g.double().norm()), "rows": int(g.shape[0]), "rows_nonzero": int(len(nzr)),
                          "distinct_codes_present": int(len(present)), "nonzero_rows_not_present": len(set(nzr.tolist()) - set(present.tolist())),
                          "present_rows_zero": len(set(present.tolist()) - set(nzr.tolist()))}
    res["micros"][mk] = r
    H.log(f"{mk}: {dt:.1f}s loss {float(loss):.6f} v[:6] {[round(float(x), 5) for x in v[:6]]} "
          f"grads set {sum(x is not None for x in r['grad_sha'].values())}/{len(main_named)}" + (f" s_emb {r['s_emb']}" if mode else ""))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=0)
H.log("->", a.out)
dist.destroy_process_group()
