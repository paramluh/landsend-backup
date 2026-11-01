#!/usr/bin/env python3
"""pose-s2p test (c): TRAINING PATH on CPU with a tiny-config PPM (random init) and ONE real pose micro built by the branch's data code.

Checks (flag = PPM_S2P / PPM(s2p=...)):
  c0  flag OFF branch model_ppm == the ORIGINAL Rosie model_ppm (aloe-pose-rosie/code), same weights: every forward term bit-identical and
      every parameter grad bit-identical (flag off = today's training maths, proven not assumed).
  c1  flag ON ('full' and 'cb0') with s_emb = 0 vs flag OFF, same weights: every forward term bit-identical (torch.equal); after the
      backward of (a) the pose cb0 loss alone and (b) the run_micro total (lw x text + cb0 + pose cb0, plus the decoders' upstream
      gradient into hv_s / hv_p / zb_p, simulated by fixed seeded tensors exactly as run_micro's torch.autograd.backward(roots, grads)),
      every pre-existing parameter's grad bit-identical; tower_p.s_emb.weight grad non-zero, and non-zero ONLY on rows of speech codes
      that occur in B's depth (rows of absent codes exactly 0).
  c2  non-zero s_emb (seeded N(0, 0.02 / sqrt(32)), the code_emb rule): pose terms change, every speech / text term bit-identical to flag off
      (speech never depends on s_emb).
GPU kernels (vendor fused_triton rms / add_rms / qk_prep / silu_mul and FA3 varlen) are replaced IN THIS TEST ONLY by torch reference
implementations; both models of every comparison run the same replacement, so the comparison is unaffected."""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

C = "/data/home/amartyavarma/ppm/pose-s2p/code"
ORIG = "/data/home/amartyavarma/ppm/aloe-pose-rosie/code/model_ppm.py"
for p in (f"{C}/vendor", f"{C}/lib", C):
    sys.path.insert(0, p)
torch.manual_seed(0)
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
OUT = {}
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


import model_ppm as M  # noqa: E402
import fused_triton as FO  # noqa: E402
import fa3_op  # noqa: E402
import data_aloe_pose as A  # noqa: E402
import train_aloe_pose as T  # noqa: E402   (module level only: CFG, prep, ARR; no dist, no GPU)

spec = importlib.util.spec_from_file_location("model_ppm_orig", ORIG)
MO = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MO)
assert MO.FO is FO and MO.H2 is M.H2, "the original model_ppm must share the (identical) vendored modules"


# ------------------------------------------------------------------------------------------------ CPU reference kernels (test only)
def rms(x, w, eps):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w.to(x.dtype) * y.to(x.dtype)


def add_rms(h, a, w, eps):
    s = h + a
    return s, rms(s, w, eps)


def qk_prep(q, k, qw, kw, eps, cos, sin):
    hd = qw.shape[0]
    q = q.view(*q.shape[:-1], -1, hd)
    k = k.view(*k.shape[:-1], -1, hd)
    q, k = rms(q, qw, eps), rms(k, kw, eps)
    c, s = cos.unsqueeze(2).to(q.dtype), sin.unsqueeze(2).to(q.dtype)

    def rot(x):
        x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
        return torch.cat([-x2, x1], -1)
    return q * c + rot(q) * s, k * c + rot(k) * s


def silu_mul(g, u):
    return F.silu(g) * u


def fa3_varlen(q, k, v, cuq, cuk, mq, mk, scale, causal):
    out = []
    cu = cuq.tolist()
    rep = q.shape[1] // k.shape[1]
    for a, b in zip(cu[:-1], cu[1:]):
        qq = q[a:b].transpose(0, 1)[None]
        kk = k[a:b].repeat_interleave(rep, 1).transpose(0, 1)[None]
        vv = v[a:b].repeat_interleave(rep, 1).transpose(0, 1)[None]
        o = F.scaled_dot_product_attention(qq, kk, vv, is_causal=bool(causal), scale=scale)
        out.append(o[0].transpose(0, 1))
    return torch.cat(out, 0)


_addmm, _mm = torch.addmm, torch.mm


def addmm_cpu(inp, a, b_, *args, out_dtype=None, out=None, **kw):     # heads_s2q CE backward: bf16 GEMM, fp32 accumulate (CUDA-only op)
    if out_dtype is None:
        return _addmm(inp, a, b_, *args, out=out, **kw) if out is not None else _addmm(inp, a, b_, *args, **kw)
    r = _addmm(inp.to(out_dtype), a.to(out_dtype), b_.to(out_dtype), *args, **kw)
    return out.copy_(r) if out is not None else r


def mm_cpu(a, b_, *args, out_dtype=None, **kw):
    return _mm(a, b_, *args, **kw) if out_dtype is None else _mm(a.to(out_dtype), b_.to(out_dtype), *args, **kw)


torch.addmm, torch.mm = addmm_cpu, mm_cpu
FO.rms, FO.add_rms, FO.qk_prep, FO.silu_mul = rms, add_rms, qk_prep, silu_mul
fa3_op.fa3_varlen_func = fa3_varlen

# ------------------------------------------------------------------------------------------------ one real pose micro (branch data code)
cfg = T.CFG["data"]
sp = A.splits_v2()
ev = sorted(sp["eval"], key=lambda x: (x[3], x[4]))
st = A.Stores()
m = None
MAXN = int(os.environ.get("S2P_TEST_MAXN", "9600"))
for it in ev:
    if it[3] < 60 or it[3] > MAXN:
        continue
    mm = st.conv(it[:3], cfg, 0)
    P_ = int(mm["P"])
    clip = int((np.asarray(mm["depth"][1])[:P_, 0] >= 0).sum())
    if json.loads(mm["prov"]).get("clips") and len(mm["pid"]) <= MAXN:   # want B's voice clip in the system block (speech codes, no pose)
        m, item = mm, it
        break
assert m is not None, "no short eval item with a voice clip"
mic = A.pack_micro([m])
oc = np.asarray(mic["own"]).sum(0)
dev = torch.device("cpu")
b = T.prep(mic, dev, oc[0], oc[1], oc[3])
P = int(m["P"])
N = len(m["pid"])
d1 = b["depth"][1]
has_s = d1[:, 0] >= 0
has_p = b["pcb0"] >= 0
info_micro = dict(item=list(item[:3]) + [item[4]], prov=json.loads(m["prov"]), P=P, n_positions=N, n_frames=int(item[3]),
                  speech_code_positions=int(has_s.sum()), speech_codes_in_system_block=int(has_s[:P].sum()),
                  pose_label_positions=int(has_p.sum()), first_pose_label=int(torch.nonzero(has_p)[0]), first_conv_speech=int(P + 2),
                  pose_pos_with_speech_same_T=int((has_p & has_s).sum()))
assert info_micro["pose_label_positions"] == info_micro["pose_pos_with_speech_same_T"], "every pose-labelled T must carry speech codes at T"
OUT["micro"] = info_micro
log("micro", info_micro)

# ------------------------------------------------------------------------------------------------ tiny configs
c32 = M.cfg32(2)
c32.hidden_size, c32.intermediate_size, c32.num_attention_heads, c32.num_key_value_heads, c32.head_dim = 128, 256, 4, 2, 32
c4 = M.cfg4b(2)
c4.hidden_size, c4.intermediate_size, c4.num_attention_heads, c4.num_key_value_heads, c4.head_dim = 64, 128, 4, 2, 16
c06 = M.cfg06(1)
DT = torch.bfloat16                                                # = the FSDP param_dtype the trainer computes in


def build(mod, s2p):
    torch.manual_seed(1)
    kw = {} if mod is MO else {"s2p": s2p}
    mdl = mod.PPM(c32, c4, c06, proj_r=64, tower_vocab=M.TOWER_VOCAB, pose_head=False, decp_bb=True, **kw)
    mdl.decp_bb_alpha = 1.0
    return mdl


def init_(mdl):
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for n, p in mdl.named_parameters():
            if n == "tower_p.s_emb.weight":
                p.zero_()
            elif n.endswith("norm.weight") or n.endswith("layernorm.weight"):
                p.fill_(1.0)
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)


ref = build(MO, None)
init_(ref)
ref = ref.to(DT)
SD = {k: v.clone() for k, v in ref.state_dict().items()}


def clone_from(mod, s2p):
    mdl = build(mod, s2p).to(DT)
    miss, unexp = mdl.load_state_dict(SD, strict=False)
    assert not unexp and miss == (["tower_p.s_emb.weight"] if s2p else []), (miss, unexp)
    if s2p:
        with torch.no_grad():
            mdl.tower_p.s_emb.weight.zero_()
    mdl.train()
    for blk in list(mdl.layers) + list(mdl.tower_s.layers) + list(mdl.tower_p.layers):
        blk.ac = False                                             # (same on both sides; recompute = identical maths anyway)
    return mdl


rot = M.Rot(c32, dev)
rot_t = M.Rot(c4, dev)
KEYS = ("text", "cb0", "cb0_sum", "cb0_n", "cb0_hit", "hv_s", "pcb0", "pcb0_sum", "pcb0_n", "pcb0_hit", "hv_p", "zb_p")
SPEECH_KEYS = ("text", "cb0", "cb0_sum", "cb0_n", "cb0_hit", "hv_s")
CF = T.CFG
gen_up = torch.Generator().manual_seed(77)
UP = None


def run(mdl, mode):
    """forward + backward of mode 'pose' (pose cb0 only) or 'total' (run_micro's loss + decoder upstream grads) -> (terms, grads)."""
    global UP
    mdl.zero_grad(set_to_none=True)
    out = mdl(b, rot, rot_t)
    if mode == "pose":
        roots, grads = [out["pcb0"]], [None]
    else:
        loss = CF["lw_word"] * out["text"] + CF["lw_speech"] * CF["w_cb0"] * out["cb0"] + CF["lw_pose"] * CF["w_pcb0"] * out["pcb0"]
        if UP is None:
            UP = [torch.randn(out[k].shape, generator=gen_up).to(out[k].dtype) * 1e-3 for k in ("hv_s", "hv_p", "zb_p")]
        roots, grads = [loss, out["hv_s"], out["hv_p"], out["zb_p"]], [None] + UP
    torch.autograd.backward(roots, grads)
    terms = {k: out[k].detach().clone() for k in KEYS}
    gr = {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in mdl.named_parameters()}
    return terms, gr


def cmp_terms(a, c, keys=KEYS):
    res = {}
    for k in keys:
        x, y = a[k], c[k]
        res[k] = dict(equal=bool(torch.equal(x, y)), maxabs=float((x.float() - y.float()).abs().max()))
    return res


def cmp_grads(ga, gc, skip=("tower_p.s_emb.weight",)):
    bad, n = [], 0
    for k, x in ga.items():
        if k in skip:
            continue
        y = gc[k]
        n += 1
        if (x is None) != (y is None) or (x is not None and not torch.equal(x, y)):
            bad.append((k, None if x is None or y is None else float((x.float() - y.float()).abs().max())))
    return n, bad


ok_all = True
# ---- c0: flag off branch == original
res = {}
for mode in ("pose", "total"):
    ta, ga = run(clone_from(MO, None), mode)
    tb, gb = run(clone_from(M, None), mode)
    ct = cmp_terms(ta, tb)
    n, bad = cmp_grads(ga, gb, skip=())
    ok = all(v["equal"] for v in ct.values()) and not bad
    ok_all &= ok
    res[mode] = dict(ok=ok, terms_max_abs=max(v["maxabs"] for v in ct.values()), n_grads=n, grads_differ=bad[:5])
    log(f"c0 flag-off vs original [{mode}]: {'PASS' if ok else 'FAIL'}", res[mode])
    if mode == "total":
        BASE_T, BASE_G = ta, ga
OUT["c0_flagoff_vs_original"] = res

# ---- c1: flag on (zero s_emb) vs off;  c2: non-zero s_emb
for s2p in ("full", "cb0"):
    r = {}
    for mode in ("pose", "total"):
        toff, goff = (BASE_T, BASE_G) if mode == "total" else run(clone_from(M, None), mode)
        mon = clone_from(M, s2p)
        ton, gon = run(mon, mode)
        ct = cmp_terms(toff, ton)
        n, bad = cmp_grads(goff, gon)
        g = gon["tower_p.s_emb.weight"].float()
        rows_nz = torch.nonzero(g.abs().sum(1) > 0)[:, 0]
        codes = d1[has_s] if s2p == "full" else d1[has_s][:, :1]
        present = torch.unique(codes.reshape(-1))
        extra = sorted(set(rows_nz.tolist()) - set(present.tolist()))
        ok = all(v["equal"] for v in ct.values()) and not bad and float(g.norm()) > 0 and not extra
        ok_all &= ok
        r[mode] = dict(ok=ok, terms_all_equal=all(v["equal"] for v in ct.values()), terms_max_abs=max(v["maxabs"] for v in ct.values()),
                       n_other_grads=n, other_grads_differ=bad[:5], s_emb_grad_norm=float(g.norm()), s_emb_rows=int(g.shape[0]),
                       s_emb_rows_nonzero=int(len(rows_nz)), distinct_codes_present=int(len(present)), nonzero_rows_not_present=len(extra))
        log(f"c1 s2p={s2p} zero-init vs off [{mode}]: {'PASS' if ok else 'FAIL'}", r[mode])
    # c2: non-zero s_emb
    mon = clone_from(M, s2p)
    with torch.no_grad():
        mon.tower_p.s_emb.weight.copy_((torch.randn(mon.tower_p.s_emb.weight.shape, generator=torch.Generator().manual_seed(5)) * 0.02 / 32 ** 0.5).to(DT))
    tnz, _ = run(mon, "total")
    ct = cmp_terms(BASE_T, tnz)
    sp_eq = all(ct[k]["equal"] for k in SPEECH_KEYS)
    pose_changed = (not ct["pcb0"]["equal"]) and (not ct["hv_p"]["equal"])
    # same-position causality inside the pose tower: hv_p rows BEFORE the first position with speech codes must be unchanged
    first_s = int(torch.nonzero(has_s)[0])
    pre_eq = bool(torch.equal(BASE_T["hv_p"][:first_s], tnz["hv_p"][:first_s]))
    at_first_changed = not bool(torch.equal(BASE_T["hv_p"][first_s], tnz["hv_p"][first_s]))
    ok = sp_eq and pose_changed and pre_eq and at_first_changed
    ok_all &= ok
    r["nonzero"] = dict(ok=ok, speech_text_terms_bit_identical=sp_eq, pose_terms_changed=pose_changed,
                        pcb0_off=float(BASE_T["pcb0"]), pcb0_nonzero=float(tnz["pcb0"]), hv_p_max_abs_change=ct["hv_p"]["maxabs"],
                        first_speech_pos=first_s, hv_p_before_first_speech_unchanged=pre_eq, hv_p_at_first_speech_changed=at_first_changed)
    log(f"c2 s2p={s2p} non-zero s_emb: {'PASS' if ok else 'FAIL'}", r["nonzero"])
    OUT[f"c1c2_{s2p}"] = r

OUT["PASS"] = bool(ok_all)
OUT["tiny_cfg"] = dict(backbone=dict(layers=2, H=128), towers=dict(layers=2, H=64), decoders=dict(layers=1, H=c06.hidden_size), dtype=str(DT),
                       vocab=c32.vocab_size, tower_vocab=M.TOWER_VOCAB)
os.makedirs("/data/home/amartyavarma/ppm/pose-s2p/tests/out", exist_ok=True)
json.dump(OUT, open("/data/home/amartyavarma/ppm/pose-s2p/tests/out/test_train_path.json", "w"), indent=1, default=str)
log("RESULT", "PASS" if ok_all else "FAIL")
