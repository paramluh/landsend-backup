"""D32 (pose head) GPU tests on ONE node (torchrun, reduced depth; env = prod_ppm.env + overrides, see tests/test_d32.sh).

usage: torchrun ... tests/test_d32.py <code dir> <mode> <out.json>
  mode grads  one GA step (pose, speech, speech, text) + optimizer step, twice (steps 1, 2), on REAL micros of this node's stores, fresh init:
              per micro the loss + stats vector, per parameter a sha256 of its full gradient (all ranks' shards) and of its weights after
              each step, per group the clipped grad norm. With the head on: the gradient arriving at the head's INPUT (must be exactly 0
              while W = 0) and the head's own grad norms. <code dir> = the code under test (the pre-D32 backup or the new code), so the
              outputs of two runs can be compared bit for bit (tests/cmp_d32.py).
  mode align  the alignment test: hooks on a real forward of a few pose rows record the pose tower's conditioning input / labels / weights and
              the head's input / labels / weights; asserts the head reads hn[T-1] at T exactly where the tower's cond shift reads hn[T-1],
              the labels / weights are the very same tensors, and every scored label = raw pose cb0 of row frame T - CD.
DETERMINISM (test only, nothing in the trainer changes): FA3 varlen backward with deterministic=True (production: False = atomics),
torch.use_deterministic_algorithms(warn_only), CUBLAS_WORKSPACE_CONFIG. Same code twice must then be bit-identical (checked by the caller)."""
import hashlib
import json
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
CODE, MODE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
for p in (os.path.join(CODE, "vendor"), os.path.join(CODE, "lib"), CODE):
    sys.path.insert(0, p)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

torch.use_deterministic_algorithms(True, warn_only=True)
import pinned_fa3  # noqa: E402

_fi = pinned_fa3._load_module().flash_attn_interface
_bwd0 = _fi._flash_attn_backward


def _bwd_det(*a, **k):
    a = list(a)
    if len(a) > 19:
        a[19] = True
    else:
        k["deterministic"] = True
    return _bwd0(*a, **k)


_fi._flash_attn_backward = _bwd_det
import train_ppm as T  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.utils.checkpoint  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402


def _att_group(qs, ks, vs, scale, causal):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(qs, ks, vs, is_causal=causal, scale=scale)


class _DetAttn(torch.autograd.Function):
    """test-only stand-in for the FA3 varlen op (its backward is run-to-run nondeterministic even with deterministic=True in this build:
    checked): per packed segment and per KV head (its `rep` query heads), SDPA MATH backend (deterministic); the backward recomputes one
    head group at a time (bounded memory). Same attention maths."""

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


if os.environ.get("D32_DET_ATTN", "1") == "1":
    T.M.fa3_op.fa3_varlen_func = _det_varlen

M, D, CFG = T.M, T.D, T.CFG
assert os.path.abspath(os.path.dirname(M.__file__)) == os.path.abspath(CODE), (M.__file__, CODE)
RANK, WORLD, LOCAL, LW = T.RANK, T.WORLD, T.LOCAL, T.LW
PH = bool(CFG.get("pose_head", 0))


def log(*a):
    if RANK == 0:
        print(*a, flush=True)


def gather_digest(t):
    """sha256 over every rank's local bytes (rank order)."""
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


def micros(step, dev):
    """the GA step's micros, from this node's stores (same rows for every code version)."""
    dcfg = CFG["data"]
    if not hasattr(micros, "c"):
        pnode = D.PS.load_node(CFG["pose_root"])
        pplan = D.PS.plan(pnode, CFG["pose_seed"], LW, LOCAL, 1, LW, LOCAL, 1)
        snode = D.R3.load_node(CFG["speech_root"])
        sidx = D.DS.load_index(CFG["speech_root"])
        splan = D.speech_plan(snode, sidx, CFG["speech_seed"], LW, LOCAL, CFG["speech_eval_frac"], CFG["speech_eval_salt"], 1)
        micros.c = (D.PoseDS(pnode, pplan["rows"][:16], dcfg, RANK), D.SpeechDS(CFG["speech_root"], splan["rows"][:32], dcfg, RANK),
                    D.TextDS(CFG["text_root"], "train", [j * WORLD + RANK for j in range(16)], dcfg), pnode, pplan)
    pds, sds, tds = micros.c[:3]
    out, ks = [], {"pose": 0, "speech": 0, "text": 0}
    for kind in T.SEQ:
        i = step * T.SEQ.count(kind) + ks[kind]; ks[kind] += 1
        out.append((kind, {"pose": pds, "speech": sds, "text": tds}[kind][i]))
    return out


def setup(dev):
    c32, c4, c06 = M.cfg32(CFG["layers"]), M.cfg4b(CFG["tower_layers"]), M.cfg06(CFG["dec_layers"])
    kw = {"pose_head": PH} if PH else {}
    model, _ = M.build(c32, c4, c06, dev, log, partial=True, proj_r=CFG["proj_r"], tower_vocab=CFG["tower_vocab"], dec_ac=CFG["dec_ac"],
                       dec_mode=CFG["dec_mode"], **kw)
    if CFG["dec_fast"]:
        for dec in (model.dec_s, model.dec_p):
            M.compile_decoder(dec, CFG["dec_compile_mode"], whole=CFG["dec_mode"] == "replicate", dynamic=CFG["dec_dyn"])
    rot = M.Rot(c32, dev)
    opt, sched, by = T.make_opt(model)
    dos = {"dec_s": T.DecOpt(model.dec_s, "dec_s"), "dec_p": T.DecOpt(model.dec_p, "dec_p")}
    return model, rot, opt, sched, dos


def mode_grads(dev):
    model, rot, opt, sched, dos = setup(dev)
    rec = {"code": CODE, "pose_head": PH, "groups": [pg["name"] for pg in opt.param_groups], "steps": []}
    hin = {"g": []}
    if PH:                                                       # the gradient arriving at the head's input x = shift1(hn)
        def _pre(mod, args):
            if args[0].requires_grad:
                args[0].register_hook(lambda g: hin["g"].append(g.detach().float().abs().max()))
            return None                                          # (a pre-hook's return value would replace the input)
        model.pose_head.register_forward_pre_hook(_pre)
    for step in range(2):
        st = {"micros": [], "lr_used": [pg["lr"] for pg in opt.param_groups]}
        hin["g"].clear()
        for kind, mb in micros(step, dev):
            b = T.prep(mb, dev)
            v, loss = T.run_micro(model, b, rot, dos, grad=True)
            st["micros"].append({"kind": kind, "v": v, "loss": loss})
        st["grad"] = {n: gather_digest(p.grad) for n, p in model.named_parameters() if p.grad is not None and not n.startswith("dec_")}
        st["grad_norm"] = {n: full_norm(p.grad, dev) for n, p in model.named_parameters() if p.grad is not None and not n.startswith("dec_")}
        if PH:
            mx = torch.stack(hin["g"]).max() if hin["g"] else torch.tensor(-1.0, device=dev)
            dist.all_reduce(mx, op=dist.ReduceOp.MAX)
            st["head_input_grad_absmax"] = float(mx)
            st["n_head_input_grads"] = len(hin["g"])
        gn = {}
        for pg in opt.param_groups:
            g_ = torch.nn.utils.clip_grad_norm_(pg["params"], CFG["clip"][pg["name"]])
            gn[pg["name"]] = float(g_.full_tensor() if hasattr(g_, "full_tensor") else g_)
        st["gn"] = gn
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        st["w_after"] = {n: gather_digest(p) for n, p in model.named_parameters()}
        st["lr_next"] = sched.get_last_lr()
        rec["steps"].append(st)
        log(f"[grads] step {step + 1}: losses {[round(m['loss'], 6) for m in st['micros']]} gn {gn}"
            + (f" | head-input grad |max| {st['head_input_grad_absmax']:.3g} over {st['n_head_input_grads']} micro(s)" if PH else ""))
    if RANK == 0:
        json.dump(rec, open(OUT, "w"))
    log(f"[grads] wrote {OUT}")


def mode_align(dev):
    model, rot, opt, sched, dos = setup(dev)
    assert PH, "align needs PPM_POSE_HEAD=1"
    R = {}
    orig_shift, orig_phce = M.shift1, M.pose_head_ce

    def shift_rec(x, ctx, fill):
        y = orig_shift(x, ctx, fill)
        R.setdefault("shifts", []).append((x, ctx, y))
        return y

    def phce_rec(head, hn, ctx, lab, wpos):
        R["head_args"] = (hn, ctx, lab, wpos)
        return orig_phce(head, hn, ctx, lab, wpos)
    M.shift1, M.pose_head_ce = shift_rec, phce_rec
    model.tower_p.register_forward_pre_hook(lambda mod, args: R.__setitem__("tower_args", args))
    model.tower_p.cond.register_forward_hook(lambda mod, args, out: R.__setitem__("cond_out", out))
    model.pose_head.register_forward_pre_hook(lambda mod, args: R.__setitem__("head_in", args[0]))
    pnode, pplan = None, None
    tot = {"rows": 0, "scored": 0, "clips": 0}
    ex = []
    for i in range(3):                                           # 3 real pose rows per rank
        if not hasattr(micros, "c"):
            micros(0, dev)
        pds, pnode, pplan = micros.c[0], micros.c[3], micros.c[4]
        mb = pds[i]
        b = T.prep(mb, dev)
        R.clear()
        with torch.no_grad():
            model(b, rot, rot)
        th, tctx, tlab, tw = R["tower_args"][2], R["tower_args"][4], R["tower_args"][6], R["tower_args"][7]
        hn, hctx, hlab, hw = R["head_args"]
        assert hn is th and hctx is tctx and hlab is tlab and hw is tw, "head and tower must get the same hn / ctx / labels / weights tensors"
        assert hlab is b["pcb0"] and hw is b["pcb0_w"]
        # the tower's conditioning shift = the shift1 call on cond(hn); the head's = the shift1 call on hn
        sh_t = [s for s in R["shifts"] if s[0] is R["cond_out"]]
        sh_h = [s for s in R["shifts"] if s[0] is hn]
        assert len(sh_t) == 1 and len(sh_h) == 1 and sh_t[0][1] is tctx and sh_h[0][1] is tctx
        x = R["head_in"]
        assert x is sh_h[0][2], "the head's input is shift1(hn)"
        N = hn.shape[0]
        starts = set(int(s) for s in tctx.starts.tolist())
        notst = torch.tensor([t for t in range(N) if t not in starts], device=dev)
        st_ = torch.tensor(sorted(starts), device=dev)
        for (xin, _, xout) in (sh_t[0], sh_h[0]):                # position T reads position T - 1 (segment starts: zeros) -- bitwise
            assert torch.equal(xout[notst], xin[notst - 1]) and bool((xout[st_] == 0).all())
        # targets: every scored label = raw pose cb0 of row frame (s0 + t - CD) for micro position T = clip start + t
        row = D.PS.PoseRows(pnode)[int(pplan["rows"][i])]
        lab = hlab.cpu().numpy()
        pos = 0
        exp = np.full(N, -100, np.int64)
        frame = np.full(N, -1, np.int64)
        pose = np.asarray(row["pose"])
        for (s0, L) in D.PS.row_segments(row):
            for t in range(D.CD, L):
                c = int(pose[0, s0 + t - D.CD])
                exp[pos + t] = c if c >= 0 else -100
                frame[pos + t] = s0 + t - D.CD
            pos += L
            tot["clips"] += 1
        assert pos == N, (pos, N)
        assert (exp == lab).all(), ("labels != raw pose cb0 of frame T - CD", int((exp != lab).sum()))
        sc = np.flatnonzero(lab >= 0)
        assert 0 not in set(sc.tolist()) or 0 not in starts, "a scored position at a segment start (no h[T-1])"
        pairs_t = [(int(T_) - 1, int(lab[T_])) for T_ in sc]      # tower: cond(hn[T-1]) at T, label lab[T]
        pairs_h = [(int(T_) - 1, int(lab[T_])) for T_ in sc]      # head: x[T] = hn[T-1] (asserted above), label lab[T] (same tensor)
        assert pairs_t == pairs_h
        w_ = hw.cpu().numpy()
        assert np.allclose(w_[sc], 1.0 / len(sc)) and (w_[lab < 0] == 0).all()
        tot["rows"] += 1; tot["scored"] += len(sc)
        if i == 0 and RANK == 0:
            for T_ in list(sc[:3]) + list(sc[-2:]):
                ex.append({"T": int(T_), "tower_cond_pos": int(T_) - 1, "head_input_pos": int(T_) - 1, "target_code": int(lab[T_]),
                           "target_row_frame": int(frame[T_]), "row_frame_of_T": int(frame[T_]) + D.CD})
    M.shift1, M.pose_head_ce = orig_shift, orig_phce
    t_ = torch.tensor([tot["rows"], tot["scored"], tot["clips"]], dtype=torch.float64, device=dev); dist.all_reduce(t_)
    res = {"rows": int(t_[0]), "scored_positions": int(t_[1]), "clips": int(t_[2]), "examples_rank0": ex,
           "checks": ["head and tower receive the same hn / ctx / pcb0 / pcb0_w tensor objects",
                      "tower cond: shift1(cond(hn)) position T <- T-1 (bitwise); head input: shift1(hn) position T <- T-1 (bitwise), same ctx",
                      "segment starts: zero input for both", "every scored label == raw pose[0, s0 + t - CD] (frame T - CD)",
                      "weights = 1/n on scored positions, 0 elsewhere (identical tensor)"]}
    log(f"[align] PASS: {res}")
    if RANK == 0:
        json.dump(res, open(OUT, "w"))


def main():
    dev = torch.device("cuda", LOCAL)
    torch.cuda.set_device(dev)
    import datetime
    dist.init_process_group("nccl", device_id=dev, timeout=datetime.timedelta(seconds=1800))
    {"grads": mode_grads, "align": mode_align}[MODE](dev)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
