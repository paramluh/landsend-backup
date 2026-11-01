#!/usr/bin/env python3
"""pose6 B1 test stage C (run with PPM_G0COND=1 PPM_S2P=cb0 + the mono8020 env):
  (3) LOSS PARITY with the base's own model (pose5/code-811-s2p/model_ppm = job 31101's code), plain modules (no FSDP), bf16 compute,
      identical weights with NON-ZERO tower_p.s_emb and inp.g0: every forward term (text, cb0, pcb0 + sums / counts / hits, hv_s, hv_p,
      zb_p) and every parameter gradient of the run_micro-shaped backward (lw_word x text + cb0 + pose cb0, plus fixed seeded upstream
      gradients into hv_s / hv_p / zb_p as the decoders' nanosteps would feed back) on: the Aloe packed micro (4 prompt types, G0 on), each
      Aloe single, and a REAL 8,1,1 TikTok pose micro (pose5/g0-test-micros/pose.pkl, built by the 811 code with g0pos / g0f).
  KEYS: the FT model's state-dict keys / shapes == the 811 model's (31101 config: decp_bb, no pose head, s2p cb0, g0cond) at REAL size
      (meta device), and vs the real export w-15973's DCP metadata (pre-mods: the FT model must need exactly the 2 new keys more).
  (4a) speech / text rows (811 micros speech / gemma / chat) through the pose6 model with the flags on and non-zero new params: the new
      parameters get NO gradient (None or exactly 0)."""
import importlib.util
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h6 as H  # noqa: E402

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
H.use_code(H.P6)
import train_aloe_pose as T  # noqa: E402

M = T.M
assert T.G0 and T.S2P == "cb0", ("run with PPM_G0COND=1 PPM_S2P=cb0", T.G0, T.S2P)
spec = importlib.util.spec_from_file_location("model_ppm_811", f"{H.C811}/model_ppm.py")
MO = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MO)
assert MO.FO is M.FO and MO.H2 is M.H2 and MO.SP is M.SP and MO.DA is M.DA, "the 811 model must share the (byte-identical) vendored modules"
H.cpu_patches()
dev = torch.device("cpu")
DT = torch.bfloat16
OUT = {}
ok_all = True

# ---- KEYS at real size (meta device)
with torch.device("meta"):
    kw = dict(pose_head=False, decp_bb=True, s2p="cb0", g0cond=True)
    ft = M.PPM(M.cfg32(64), M.cfg4b(36), M.cfg06(28), 2048, M.TOWER_VOCAB, **kw)
    bs = MO.PPM(MO.cfg32(64), MO.cfg4b(36), MO.cfg06(28), 2048, MO.TOWER_VOCAB, **kw)
    ft_off = M.PPM(M.cfg32(64), M.cfg4b(36), M.cfg06(28), 2048, M.TOWER_VOCAB, pose_head=False, decp_bb=True)
kf = {k: tuple(v.shape) for k, v in ft.state_dict().items()}
kb = {k: tuple(v.shape) for k, v in bs.state_dict().items()}
koff = {k: tuple(v.shape) for k, v in ft_off.state_dict().items()}
import torch.distributed.checkpoint as dcp  # noqa: E402
W15973 = "/data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-15973"
smd = dcp.FileSystemReader(W15973).read_metadata().state_dict_metadata
kw_ = {k: tuple(v.size) for k, v in smd.items()}
keys = {"ft_eq_811_keys_shapes": kf == kb, "n_keys": len(kf), "new_keys_shapes": {k: kf[k] for k in H.NEW},
        "ft_minus_ftoff": sorted(set(kf) - set(koff)),
        "w15973_minus_ftoff": sorted(set(kw_) - set(koff)), "ftoff_minus_w15973": sorted(set(koff) - set(kw_)),
        "ft_minus_w15973": sorted(set(kf) - set(kw_)),
        "shapes_equal_on_common": all(kw_[k] == kf[k] for k in set(kw_) & set(kf)),
        "w15973_dtypes": sorted({str(v.properties.dtype) for v in smd.values() if hasattr(v, "properties")})}
ok = keys["ft_eq_811_keys_shapes"] and keys["ft_minus_ftoff"] == sorted(H.NEW) and keys["ftoff_minus_w15973"] == [] and \
    keys["ft_minus_w15973"] == sorted(H.NEW) and keys["shapes_equal_on_common"]
keys["ok"] = ok
ok_all &= ok
OUT["keys"] = keys
H.log("KEYS", keys)
del ft, bs, ft_off, smd

# ---- tiny models with identical weights (non-zero new params)
c32, c4, c06 = H.tiny_cfgs(M)


def mk(mod):
    m = mod.PPM(c32, c4, c06, proj_r=64, tower_vocab=M.TOWER_VOCAB, pose_head=False, decp_bb=True, s2p="cb0", g0cond=True)
    m.decp_bb_alpha = 1.0
    return m


ref = mk(M)
H.seeded_init(M, ref, c4, c06, new="nz")
ref = ref.to(DT)
SD = {k: v.clone() for k, v in ref.state_dict().items()}
base = mk(MO).to(DT)
miss, unexp = base.load_state_dict(SD, strict=True), None
for m_ in (ref, base):
    m_.train()
    for blk in list(m_.layers) + list(m_.tower_s.layers) + list(m_.tower_p.layers):
        blk.ac = False                                             # same on both sides (recompute = the same maths)
assert all(torch.equal(SD[k], v) for k, v in base.state_dict().items())
for n_ in H.NEW:
    assert float(SD[n_].float().abs().max()) > 0, ("non-zero new params expected", n_)
rot = M.Rot(c32, dev)
KEYS = ("text", "cb0", "cb0_sum", "cb0_n", "cb0_hit", "hv_s", "pcb0", "pcb0_sum", "pcb0_n", "pcb0_hit", "hv_p", "zb_p")
LW = T.CFG["lw_word"]


def run(mdl, bt, seed):
    mdl.zero_grad(set_to_none=True)
    out = mdl(bt, rot, rot)
    g = torch.Generator().manual_seed(seed)
    up = [(torch.randn(out[k].shape, generator=g) * 1e-3).to(out[k].dtype) for k in ("hv_s", "hv_p", "zb_p")]
    loss = LW * out["text"] + T.CFG["w_cb0"] * out["cb0"] + T.CFG["w_pcb0"] * out["pcb0"]
    torch.autograd.backward([loss, out["hv_s"], out["hv_p"], out["zb_p"]], [None] + up)
    terms = {k: out[k].detach().clone() for k in KEYS}
    gr = {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in mdl.named_parameters()}
    return terms, gr


mic = pickle.load(open(f"{H.SCR}/micros_on.pkl", "rb"))
cases = [("aloe_packed", mic["packed"])] + [(f"aloe_single_{t}", s) for t, s in zip(("empty", "clip", "persona", "both"), mic["singles"])]
cases.append(("tiktok_811_pose", pickle.load(open("/data/home/amartyavarma/ppm/pose5/g0-test-micros/pose.pkl", "rb"))))
res = {}
for nm, b in cases:
    S = H.counts(b)
    bt = T.prep(b, dev, S[0], S[1], S[2])
    ta, ga = run(base, bt, 11)
    tb, gb = run(ref, bt, 11)
    td = {k: float((ta[k].float() - tb[k].float()).abs().max()) for k in KEYS}
    teq = all(torch.equal(ta[k], tb[k]) for k in KEYS)
    bad = [n for n in ga if (ga[n] is None) != (gb[n] is None) or (ga[n] is not None and not torch.equal(ga[n], gb[n]))]
    gmax = max((float((ga[n].float() - gb[n].float()).abs().max()) for n in ga if ga[n] is not None and gb[n] is not None), default=0.0)
    newg = {n: (None if gb[n] is None else float(gb[n].float().norm())) for n in H.NEW}
    ok = teq and not bad and all(v is not None and v > 0 for v in newg.values())
    ok_all &= ok
    res[nm] = {"ok": ok, "positions": int(len(b["pid"])), "K_slots": int(len(b["g0pos"])), "terms_bit_identical": teq, "terms_max_abs": max(td.values()),
               "pcb0_811": float(ta["pcb0"]), "pcb0_pose6": float(tb["pcb0"]), "text_811": float(ta["text"]), "text_pose6": float(tb["text"]),
               "cb0_811": float(ta["cb0"]), "cb0_pose6": float(tb["cb0"]), "n_grads": len(ga), "grads_differ": bad[:6], "grads_max_abs": gmax,
               "new_param_grad_norms": newg}
    H.log("PARITY", nm, res[nm])
OUT["parity_811"] = res

# ---- LINK OFF parity: the base's link-off path (811 model decp_bb_on False = 31101's PPM_DECP_BB_OFF) vs pose6 (QQA_DECP_BB_OFF's
#      model.decp_bb_on False): no zb_p, the pose decoder's input (hv_p, zb None) and its CE identical, decp_bb gets no gradient
rl = {}
for m_ in (ref, base):
    m_.decp_bb_on = False
for nm, b in (cases[0], cases[-1]):
    S = H.counts(b)
    bt = T.prep(b, dev, S[0], S[1], S[2])
    got = []
    for m_ in (base, ref):
        m_.zero_grad(set_to_none=True)
        out = m_(bt, rot, rot)
        fl = bt["plight"][:512]
        wz = torch.zeros(fl.numel())
        _, s_, n_, _ = m_.dec_p(out["hv_p"], bt["pdepth"], fl, wz)    # the pose decoder, link off: zb=None (both trainers' call)
        loss = LW * out["text"] + out["cb0"] + out["pcb0"] + 1e-3 * out["hv_p"].float().sum()
        loss.backward()
        got.append({"zb": "zb_p" in out, "hv_p": out["hv_p"].detach().clone(), "dec_ce": float(s_) / max(int(n_), 1),
                    "decp_bb_grad": [p.grad is None for p in m_.decp_bb.parameters()],
                    "g": {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in m_.named_parameters()}})
    a_, b_ = got
    gbad = [n for n in a_["g"] if (a_["g"][n] is None) != (b_["g"][n] is None) or (a_["g"][n] is not None and not torch.equal(a_["g"][n], b_["g"][n]))]
    ok = (not a_["zb"]) and (not b_["zb"]) and torch.equal(a_["hv_p"], b_["hv_p"]) and a_["dec_ce"] == b_["dec_ce"] and \
        all(a_["decp_bb_grad"]) and all(b_["decp_bb_grad"]) and not gbad
    ok_all &= ok
    rl[nm] = {"ok": ok, "zb_p_computed (811, pose6)": [a_["zb"], b_["zb"]], "hv_p_bit_identical": bool(torch.equal(a_["hv_p"], b_["hv_p"])),
              "pose_dec_ce (811, pose6)": [a_["dec_ce"], b_["dec_ce"]], "decp_bb_grad_is_None (811, pose6)": [a_["decp_bb_grad"], b_["decp_bb_grad"]],
              "grads_differ": gbad[:6]}
    H.log("LINK OFF parity", nm, rl[nm])
for m_ in (ref, base):
    m_.decp_bb_on = True
OUT["link_off_parity_811"] = rl

# ---- (4a) speech / text rows: no gradient into the new params
r4 = {}
for nm in ("speech", "gemma", "chat"):
    b = pickle.load(open(f"/data/home/amartyavarma/ppm/pose5/g0-test-micros/{nm}.pkl", "rb"))
    t = {"kind": b["kind"], "src": b.get("src", b["kind"]), "segs": [int(x) for x in b["segs"]]}
    for k in ("ids", "depth", "present", "pid", "sel", "y", "word", "pad", "eob", "nw", "cb0", "light"):
        if k in b:
            t[k] = torch.as_tensor(np.asarray(b[k]))
    t["pid"] = t["pid"][None]
    w = {k: t[k].float() for k in ("word", "pad", "eob", "nw")}
    S = float(len(b["sel"]))
    t["wt"] = ((w["word"] + w["pad"] + w["eob"] + w["nw"]) / S).float()
    t["wn"] = (5.0 * w["word"] / S).float()
    if "cb0" in t:
        t["cb0_w"] = (t["cb0"] >= 0).float() / max(float((t["cb0"] >= 0).sum()), 1.0)
    ref.zero_grad(set_to_none=True)
    out = ref(t, rot, rot)
    loss = LW * out["text"] + (out["cb0"] if "cb0" in out else 0.0)
    roots, grads = [loss], [None]
    if "hv_s" in out:
        roots.append(out["hv_s"]); grads.append((torch.randn(out["hv_s"].shape, generator=torch.Generator().manual_seed(3)) * 1e-3).to(out["hv_s"].dtype))
    torch.autograd.backward(roots, grads)
    g = {n: (None if p.grad is None else float(p.grad.float().abs().max())) for n, p in ref.named_parameters() if n in H.NEW}
    trunk_g = float(ref.inp.emb[1].weight.grad.float().abs().max())
    ok = all(v is None or v == 0.0 for v in g.values()) and trunk_g > 0 and "hv_p" not in out
    ok_all &= ok
    r4[nm] = {"ok": ok, "kind": b["kind"], "positions": int(len(b["pid"])), "new_param_grad_absmax (None = no grad)": g,
              "trunk_emb_grad_absmax": trunk_g, "pose_tower_ran": "hv_p" in out}
    H.log("(4a)", nm, r4[nm])
OUT["no_grad_on_speech_text"] = r4
OUT["PASS"] = bool(ok_all)
json.dump(OUT, open(f"{H.OUTD}/stageC_parity.json", "w"), indent=1, default=str)
H.log("RESULT stage C", "PASS" if ok_all else "FAIL")
