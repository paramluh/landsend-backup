"""D33 (decp_bb backbone -> pose decoder link) GPU tests on ONE node (torchrun, reduced depth; env = prod_ppm.env + overrides, see
tests/test_d33.sh). Micros = the REAL micros test_d32.py builds, pre-dumped from slinky-7's stores (tests/dump_d33_micros.py) so the tests
run on a node without the stores.

usage: torchrun ... tests/test_d33.py <code dir> <mode> <out.json>
  grads  NSTEP (2) GA steps + optimizer steps, fresh init: per micro the loss + stats vector, per parameter a sha256 of its full gradient
         and of its weights after each step (decoders included via w_after), per group the clipped grad norm; with the link on: the gradient
         arriving at decp_bb's INPUT (hn, through decp_bb only: must be exactly 0 while W = 0), at its output, and its own grad norms.
         <code dir> = the pre-D33 backup or the new code (tests/cmp_d33.py compares runs bit for bit).
  align  hooks on real pose micros (link on, decp_bb test-randomised, alpha 0.7): decp_bb reads the very hn tensor the pose tower gets;
         zb[T] = alpha * decp_bb(hn)[T - 1] bitwise (segment starts 0); every pose-decoder call (training nanosteps and eval) gets zb /
         hv sharing storage with the model's zb_p / hv_p and ONE index vector for both; the scored frames = plight, each exactly once per
         micro, none at a segment start; slot 0 = proj(hv[flat]) + zb[flat] bitwise; the decoder's labels at flat T = the raw pose codes
         cb1..31 of row frame T - CD; the eval probes (tower-only / backbone-only) see the same frames.
  learn  NSTEP steps (micros cycled) on the PRODUCTION decoder path (compiled, attn32, real FA3): per step time, pose micro time, finite
         check, |decp_bb.weight|, gn_decp_bb; then an eval pass (grad=False, bbvar) over this rank's pose micros -> the summarised eval
         keys incl. decp_toweronly_* / decp_bbonly_*.
DETERMINISM (grads / align, test only): deterministic attention stand-in + eager HF decoder path, as tests/test_d32.py."""
import hashlib
import json
import os
import pickle
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
CODE, MODE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
for p in (os.path.join(CODE, "vendor"), os.path.join(CODE, "lib"), CODE):
    sys.path.insert(0, p)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

DET = os.environ.get("D33_DET", "1") == "1"
if DET:
    torch.use_deterministic_algorithms(True, warn_only=True)
import train_ppm as T  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402


def _att_group(qs, ks, vs, scale, causal):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(qs, ks, vs, is_causal=causal, scale=scale)


class _DetAttn(torch.autograd.Function):
    """= tests/test_d32.py's deterministic stand-in for the FA3 varlen op (per segment / KV head, SDPA MATH)."""

    @staticmethod
    def forward(ctx, q, k, v, cu, scale, causal):
        ctx.save_for_backward(q, k, v)
        ctx.cu, ctx.scale, ctx.causal = [int(x) for x in cu.tolist()], float(scale), bool(causal)
        rep = q.shape[1] // k.shape[1]
        out = torch.empty_like(q)
        for a, b in zip(ctx.cu[:-1], ctx.cu[1:]):
            for g in range(k.shape[1]):
                sl = slice(g * rep, (g + 1) * rep)
                o = _att_group(q[a:b, sl].transpose(0, 1)[None], k[a:b, g:g + 1].expand(-1, rep, -1).transpose(0, 1)[None],
                               v[a:b, g:g + 1].expand(-1, rep, -1).transpose(0, 1)[None], ctx.scale, ctx.causal)
                out[a:b, sl] = o[0].transpose(0, 1)
        return out

    @staticmethod
    def backward(ctx, go):
        q, k, v = ctx.saved_tensors
        rep = q.shape[1] // k.shape[1]
        dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        for a, b in zip(ctx.cu[:-1], ctx.cu[1:]):
            for g in range(k.shape[1]):
                sl = slice(g * rep, (g + 1) * rep)
                with torch.enable_grad():
                    qs = q[a:b, sl].detach().requires_grad_()
                    ks = k[a:b, g:g + 1].detach().requires_grad_()
                    vs = v[a:b, g:g + 1].detach().requires_grad_()
                    o = _att_group(qs.transpose(0, 1)[None], ks.expand(-1, rep, -1).transpose(0, 1)[None],
                                   vs.expand(-1, rep, -1).transpose(0, 1)[None], ctx.scale, ctx.causal)
                    gq, gk, gv = torch.autograd.grad(o, (qs, ks, vs), go[a:b, sl].transpose(0, 1)[None])
                dq[a:b, sl] = gq; dk[a:b, g:g + 1] = gk; dv[a:b, g:g + 1] = gv
        return dq, dk, dv, None, None, None


def _det_varlen(q, k, v, cu_q, cu_k, max_q, max_k, scale, causal):
    return _DetAttn.apply(q, k, v, cu_q, scale, causal)


if DET:
    T.M.fa3_op.fa3_varlen_func = _det_varlen

M, D, CFG = T.M, T.D, T.CFG
assert os.path.abspath(os.path.dirname(M.__file__)) == os.path.abspath(CODE), (M.__file__, CODE)
RANK, WORLD, LOCAL = T.RANK, T.WORLD, T.LOCAL
BB = bool(CFG.get("decp_bb", 0))
NSTEP = int(os.environ.get("D33_NSTEP", "2"))
MICROS = os.environ.get("D33_MICROS", "/data/home/amartyavarma/ppm/teststore/d33/micros")


def log(*a):
    if RANK == 0:
        print(*a, flush=True)


def gather_digest(t):
    loc = (t.to_local() if hasattr(t, "to_local") else t).detach().float().contiguous().cpu().numpy().tobytes()
    h = hashlib.sha256(loc).hexdigest()
    allh = [None] * WORLD
    dist.all_gather_object(allh, h)
    return hashlib.sha256("".join(allh).encode()).hexdigest()


def full_norm(t, dev):
    loc = (t.to_local() if hasattr(t, "to_local") else t).detach().double()
    s = torch.tensor([loc.pow(2).sum().item()], dtype=torch.float64, device=dev)
    if hasattr(t, "to_local"):
        dist.all_reduce(s)
    return float(s.sqrt())


def gmax(x, dev):
    t = torch.tensor([float(x)], dtype=torch.float64, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t)


REC = {}


def rec():
    if not REC:
        REC.update(pickle.load(open(f"{MICROS}/rank{RANK}.pkl", "rb")))
        assert REC["dcfg"] == CFG["data"], ("dumped micros made with another data cfg", REC["dcfg"], CFG["data"])
        assert tuple(REC["seq"]) == T.SEQ
    return REC


def micros(step):
    r = rec()
    n = len(r["pose"]) // T.SEQ.count("pose")
    step = step % n                                                    # learn mode cycles the dumped steps
    out, ks = [], {"pose": 0, "speech": 0, "text": 0}
    for kind in T.SEQ:
        i = step * T.SEQ.count(kind) + ks[kind]; ks[kind] += 1
        out.append((kind, r[kind][i]))
    return out


def setup(dev):
    c32, c4, c06 = M.cfg32(CFG["layers"]), M.cfg4b(CFG["tower_layers"]), M.cfg06(CFG["dec_layers"])
    kw = {"pose_head": bool(CFG["pose_head"])}
    if BB:
        kw["decp_bb"] = True
    model, _ = M.build(c32, c4, c06, dev, log, partial=True, proj_r=CFG["proj_r"], tower_vocab=CFG["tower_vocab"], dec_ac=CFG["dec_ac"],
                       dec_mode=CFG["dec_mode"], **kw)
    if CFG["dec_fast"]:
        for dec in (model.dec_s, model.dec_p):
            M.compile_decoder(dec, CFG["dec_compile_mode"], whole=CFG["dec_mode"] == "replicate", dynamic=CFG["dec_dyn"])
    rot = M.Rot(c32, dev)
    opt, sched, by = T.make_opt(model)
    dos = {"dec_s": T.DecOpt(model.dec_s, "dec_s"), "dec_p": T.DecOpt(model.dec_p, "dec_p")}
    if BB:
        model.decp_bb_alpha = T.decp_bb_alpha(0)
    return model, rot, opt, sched, dos


def opt_step(opt, sched):
    gn = {}
    for pg in opt.param_groups:
        g_ = torch.nn.utils.clip_grad_norm_(pg["params"], CFG["clip"][pg["name"]])
        gn[pg["name"]] = float(g_.full_tensor() if hasattr(g_, "full_tensor") else g_)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    return gn


# ------------------------------------------------------------------------------------------------ grads
def mode_grads(dev):
    model, rot, opt, sched, dos = setup(dev)
    out = {"code": CODE, "decp_bb": BB, "alpha": CFG.get("decp_bb_alpha"), "groups": [pg["name"] for pg in opt.param_groups], "steps": []}
    hk = {"in": [], "out": []}
    if BB:                                                       # the gradient reaching hn THROUGH decp_bb only (a view used by it alone)
        def _pre(mod, args):
            x = args[0]
            if x.requires_grad:
                xv = x.view_as(x)
                xv.register_hook(lambda g: hk["in"].append(g.detach().float().abs().max()))
                return (xv,)
            return None

        def _post(mod, args, o):
            if o.requires_grad:
                o.register_hook(lambda g: hk["out"].append(g.detach().float().norm()))
            return None
        model.decp_bb.register_forward_pre_hook(_pre)
        model.decp_bb.register_forward_hook(_post)
    for step in range(NSTEP):
        if BB:
            model.decp_bb_alpha = T.decp_bb_alpha(step)
        st = {"micros": [], "lr_used": [pg["lr"] for pg in opt.param_groups]}
        hk["in"].clear(); hk["out"].clear()
        for kind, mb in micros(step):
            v, loss = T.run_micro(model, T.prep(mb, dev), rot, dos, grad=True)
            st["micros"].append({"kind": kind, "v": v, "loss": loss})
        st["grad"] = {n: gather_digest(p.grad) for n, p in model.named_parameters() if p.grad is not None and not n.startswith("dec_")}
        st["grad_norm"] = {n: full_norm(p.grad, dev) for n, p in model.named_parameters() if p.grad is not None and not n.startswith("dec_")}
        if BB:
            st["bb_input_grad_absmax"] = gmax(max([float(x) for x in hk["in"]] or [-1.0]), dev)
            st["bb_output_grad_norm_max"] = gmax(max([float(x) for x in hk["out"]] or [-1.0]), dev)
            st["n_bb_hooks"] = len(hk["in"])
            st["bb_weight_norm_before"] = full_norm(model.decp_bb.weight, dev)
        st["gn"] = opt_step(opt, sched)
        st["w_after"] = {n: gather_digest(p) for n, p in model.named_parameters()}
        st["dec_micros"] = {k: d.micros for k, d in dos.items()}
        st["lr_next"] = sched.get_last_lr()
        if BB:
            st["bb_weight_norm_after"] = full_norm(model.decp_bb.weight, dev)
        out["steps"].append(st)
        log(f"[grads] step {step + 1}: losses {[round(m['loss'], 6) for m in st['micros']]} gn {st['gn']}"
            + (f" | bb input-grad |max| {st['bb_input_grad_absmax']:.3g}, output-grad norm {st['bb_output_grad_norm_max']:.4g}, "
               f"|W| {st['bb_weight_norm_before']:.3g} -> {st['bb_weight_norm_after']:.3g}, grad |W| {st['grad_norm'].get('decp_bb.weight')}" if BB else ""))
    if RANK == 0:
        json.dump(out, open(OUT, "w"))
    log(f"[grads] wrote {OUT}")


# ------------------------------------------------------------------------------------------------ align
def mode_align(dev):
    assert BB, "align needs PPM_DECP_BB=1"
    model, rot, opt, sched, dos = setup(dev)
    alpha = 0.7
    model.decp_bb_alpha = alpha
    with torch.no_grad():                                        # test-only: a non-zero link so values are distinguishable
        g_ = torch.Generator(device=dev).manual_seed(1234 + RANK)
        for p in (model.decp_bb.weight, model.decp_bb.bias):
            M._loc(p).normal_(0.0, 0.02, generator=g_)
    R = {"dec": []}
    model.tower_p.register_forward_pre_hook(lambda mod, args: R.__setitem__("tower_args", args))
    model.decp_bb.register_forward_pre_hook(lambda mod, args: R.__setitem__("bb_in", args[0]))
    model.decp_bb.register_forward_hook(lambda mod, args, o: R.__setitem__("bb_out", o))
    model.register_forward_hook(lambda mod, args, o: R.__setitem__("out", o))
    orig = M.Decoder.forward

    def dec_rec(self, hv, depth, flat, wf, n_valid=None, zb=None, tower=True):
        if self is model.dec_p:
            R["dec"].append({"hv": hv.data_ptr(), "zb": None if zb is None else zb.data_ptr(), "flat": flat.detach().clone(),
                             "n_valid": n_valid, "tower": tower, "depth": depth.data_ptr()})
        return orig(self, hv, depth, flat, wf, n_valid=n_valid, zb=zb, tower=tower)
    M.Decoder.forward = dec_rec
    r = rec()
    res = {"micros": 0, "scored": 0, "train_calls": 0, "eval_calls": 0, "label_checked": 0}
    ex = []
    for i in range(3):
        b = T.prep(r["pose"][i], dev)
        for grad in (True, False):
            R["dec"].clear()
            T.run_micro(model, b, rot, dos, grad=grad, **({} if grad else {"bbvar": True}))
            o = R["out"]
            hn, ctx = R["tower_args"][2], R["tower_args"][4]
            bi = R["bb_in"]                                        # (FSDP2's pre-forward hooks may re-wrap the args: compare storage + values)
            assert bi.dtype == hn.dtype and bi.shape == hn.shape and torch.equal(bi, hn), "decp_bb must read the hn the pose tower is conditioned on"
            res["bb_in_shares_storage"] = res.get("bb_in_shares_storage", 0) + int(bi.data_ptr() == hn.data_ptr())
            zb = o["zb_p"]
            N = hn.shape[0]
            starts = sorted(set(int(s) for s in ctx.starts.tolist()))
            notst = torch.tensor([t for t in range(N) if t not in set(starts)], device=dev)
            exp = torch.roll(R["bb_out"], 1, 0)
            msk = torch.zeros(N, dtype=torch.bool, device=dev); msk[ctx.starts] = True
            exp = exp.masked_fill(msk[:, None], 0.0) * float(alpha)
            assert torch.equal(zb, exp), "zb != alpha * shift1(decp_bb(hn))"
            assert torch.equal(zb[notst], R["bb_out"][notst - 1] * float(alpha)) and bool((zb[starts] == 0).all())
            calls = R["dec"]
            assert calls, "no pose-decoder call recorded"
            pl = b["plight"]
            if grad:
                seen = []
                for c in calls:
                    assert c["zb"] == zb.data_ptr() and c["hv"] == o["hv_p"].data_ptr() and c["depth"] == b["pdepth"].data_ptr() and c["tower"]
                    nv = c["n_valid"] if c["n_valid"] is not None else c["flat"].numel()
                    seen.append(c["flat"][:nv])
                seen = torch.cat(seen)
                assert seen.numel() == pl.numel() and torch.equal(torch.sort(seen).values, torch.sort(pl).values), "scored frames != plight (each once)"
                res["train_calls"] += len(calls)
            else:
                assert len(calls) == 3, len(calls)                 # both | tower-only | backbone-only, on the SAME frames
                assert calls[0]["zb"] == zb.data_ptr() and calls[0]["tower"] and calls[1]["zb"] is None and calls[2]["zb"] == zb.data_ptr() \
                    and not calls[2]["tower"]
                assert all(torch.equal(c["flat"], calls[0]["flat"]) and c["hv"] == o["hv_p"].data_ptr() for c in calls)
                res["eval_calls"] += len(calls)
                seen = calls[0]["flat"]
            assert not (set(seen.tolist()) & set(starts)), "a scored decoder frame at a segment start"
            # slot 0 = proj(hv[flat]) + zb[flat] (bitwise, = _dec_core) ; backbone-only: zb[flat] alone
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                fl = seen[:64]
                x1, lab1 = model.dec_p.inputs(o["hv_p"], b["pdepth"], fl, zb)
                x0, lab0 = model.dec_p.inputs(o["hv_p"], b["pdepth"], fl)
                x2, _ = model.dec_p.inputs(o["hv_p"], b["pdepth"], fl, zb, tower=False)
            assert torch.equal(x1[:, 0], x0[:, 0] + zb[fl].to(x0.dtype)) and torch.equal(x1[:, 1:], x0[:, 1:]) and torch.equal(lab1, lab0)
            assert torch.equal(x2[:, 0], zb[fl].to(x0.dtype)) and torch.equal(x2[:, 1:], x0[:, 1:])
            if grad:
                res["micros"] += 1; res["scored"] += int(seen.numel())
                # labels at flat T = raw pose codes cb1..31 of row frame T - CD (clip-local t >= CD)
                raw = r["raw_pose"][i]
                pose = raw["pose"]
                expf = np.full(N, -1, np.int64)
                pos = 0
                for (s0, L) in raw["segs"]:
                    for t in range(D.CD, L):
                        expf[pos + t] = s0 + t - D.CD
                    pos += L
                assert pos == N, (pos, N)
                labs = (b["pdepth"][seen, 1:] - torch.arange(1, M.DEPTH, device=dev) * M.CBS).cpu().numpy()
                fr = expf[seen.cpu().numpy()]
                assert (fr >= 0).all(), "a scored frame without a source frame"
                want = np.stack([pose[1:, f] for f in fr])
                assert (labs == want).all(), ("decoder labels != raw pose cb1..31 of frame T - CD", int((labs != want).sum()))
                res["label_checked"] += int(labs.size)
                if i == 0 and RANK == 0:
                    for T_ in seen[:3].tolist():
                        ex.append({"T": T_, "decp_bb_input_pos": T_ - 1, "tower_cond_pos": T_ - 1, "target_row_frame": int(expf[T_]),
                                   "target_cb1": int(pose[1, expf[T_]])})
    M.Decoder.forward = orig
    ks_ = ("micros", "scored", "train_calls", "eval_calls", "label_checked", "bb_in_shares_storage")
    t_ = torch.tensor([res.get(k, 0) for k in ks_], dtype=torch.float64, device=dev)
    dist.all_reduce(t_)
    res = dict(zip(("micros", "scored_frames", "train_decoder_calls", "eval_decoder_calls", "labels_checked", "forwards_bb_input_same_storage_as_tower_hn"),
                   [int(x) for x in t_.tolist()]))
    res["examples_rank0"] = ex
    res["checks"] = ["decp_bb input == the hn tensor passed to the pose tower (values; storage shared where reported) (its conditioning shift1(cond(hn)) reads hn[T-1], D32 align)",
                     "zb == alpha * shift1(decp_bb(hn)) bitwise; zb[T] == alpha * decp_bb(hn)[T-1]; segment starts 0",
                     "every training nanostep call: zb / hv / pdepth share storage with the model's zb_p / hv_p / the micro's pdepth, one flat index",
                     "training: scored frames == plight, each exactly once per micro; none at a segment start",
                     "slot 0 == proj(hv[flat]) + zb[flat] bitwise (other slots, labels unchanged); backbone-only slot 0 == zb[flat]",
                     "labels at flat T == raw pose cb1..31 of row frame T - CD",
                     "eval: both / tower-only / backbone-only = 3 calls on the same frames"]
    log(f"[align] PASS: {json.dumps(res)}")
    if RANK == 0:
        json.dump(res, open(OUT, "w"))


# ------------------------------------------------------------------------------------------------ learn (+ timing)
def mode_learn(dev):
    model, rot, opt, sched, dos = setup(dev)
    out = {"code": CODE, "decp_bb": BB, "dec_fast": CFG["dec_fast"], "steps": []}
    for step in range(NSTEP):
        if BB:
            model.decp_bb_alpha = T.decp_bb_alpha(step)
        torch.cuda.synchronize(); t0 = time.time()
        vs, tp = {}, 0.0
        for kind, mb in micros(step):
            b = T.prep(mb, dev)
            torch.cuda.synchronize(); t1 = time.time()
            v, loss = T.run_micro(model, b, rot, dos, grad=True)
            torch.cuda.synchronize()
            if kind == "pose":
                tp += time.time() - t1
                vs["pose"] = v
            assert np.isfinite(loss), (kind, loss)
        gn = opt_step(opt, sched)
        torch.cuda.synchronize(); dt = time.time() - t0
        tt = torch.tensor([dt, tp], device=dev); dist.all_reduce(tt, op=dist.ReduceOp.MAX)
        sm = T.summarize("pose", T.allsum(vs["pose"], dev), "train")
        st = {"step": step + 1, "step_s": float(tt[0]), "pose_micro_s": float(tt[1]), "gn": gn,
              "pose_light_ce": sm["train_pose/pose_light_ce"], "light_ce": sm["train_pose/light_ce"], "pose_cb0_ce": sm["train_pose/pose_cb0_ce"]}
        if BB:
            st["decp_bb_wnorm"] = full_norm(model.decp_bb.weight, dev); st["alpha"] = model.decp_bb_alpha
        assert all(np.isfinite(x) for x in [st["pose_light_ce"], st["light_ce"]] + list(gn.values())), st
        out["steps"].append(st)
        log(f"[learn] step {step + 1}: {dt:.2f}s (pose micro {float(tt[1]):.2f}s) pose_light {st['pose_light_ce']:.4f} light {st['light_ce']:.4f}"
            + (f" |W_bb| {st['decp_bb_wnorm']:.4g} gn_decp_bb {gn.get('decp_bb', 0):.4g}" if BB else ""))
    model.eval()
    acc = None
    for mb in rec()["pose"]:
        v, _ = T.run_micro(model, T.prep(mb, dev), rot, dos, grad=False, **({"bbvar": True} if BB else {}))
        acc = v if acc is None else [x + y for x, y in zip(acc, v)]
    out["eval"] = {k: x for k, x in T.summarize("pose", T.allsum(acc, dev), "eval").items() if "light" in k}
    log(f"[learn] eval (this node's {len(rec()['pose'])} pose micros/rank): " + json.dumps({k: round(x, 4) for k, x in out["eval"].items()}))
    sts = out["steps"][3:]
    out["mean_step_s_from4"] = float(np.mean([s["step_s"] for s in sts])) if sts else None
    out["mean_pose_micro_s_from4"] = float(np.mean([s["pose_micro_s"] for s in sts])) if sts else None
    log(f"[learn] mean step {out['mean_step_s_from4']}, pose micro {out['mean_pose_micro_s_from4']} (steps 4..{NSTEP})")
    if RANK == 0:
        json.dump(out, open(OUT, "w"))


def main():
    dev = torch.device("cuda", LOCAL)
    torch.cuda.set_device(dev)
    import datetime
    dist.init_process_group("nccl", device_id=dev, timeout=datetime.timedelta(seconds=1800))
    {"grads": mode_grads, "align": mode_align, "learn": mode_learn}[MODE](dev)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
