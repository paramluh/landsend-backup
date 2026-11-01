"""qwen-1ch/stage2 COPY of mistral stage2_1ch/heads_s2.py. Unchanged except three ADDITIONS for the merged micro (N packed rows, each
row's own mean): shift_segs (the voice read shift restarts at every row), cb0_ce_w (cb0 CE with a per-position weight) and
LightTower.forward_w (light CE with a per-frame weight). With one row and weights 1/n they equal the original means (tests).
(original doc)
STAGE-2 heads and the two new towers' CPU-importable parts (no torch.distributed, no FA3; the GPU model is model_s2m.py).

ce_head        the trunk's SPEECH text-lane CE: full-softmax CE through the ONE head (lm_head with the spec columns -- T19's PAD row
               `spec_out` -- replaced by x . R^T, exactly e28head's column rule), per-position weight wt (speech_m.speech_targets:
               (w_text [word] + w_tim [PAD/NW/BND]) / n_keep). Chunked, gradient formed in the forward (e28head / klhead pattern): no
               [positions, V] fp32 tensor ever lives whole; logits = bf16 GEMM output, softmax math fp32, d logits rounded to the GEMM
               dtype once, dW / dR accumulated in fp32. Value: sum_i wt_i CE_i. Stats [3, N]: CE_i, top-1 hit, p(target).
code_feedback  Qwen s2model.code_feedback: AudioEmb = sum over the 32 codebooks of a fresh (32*2048, H) table; -1 slots = zero.
voice_read     Qwen A1 concat5: proj(concat_i RMSNorm_i(trunk residual after read layer i)) shifted by one position (position T reads T-1).
LightTower     Qwen's random-init Qwen3-0.6B-architecture depth decoder: per selected frame [proj(h8[T]), e_cb0(cb0), e(cb1..cb30)]
               -> cb1..cb31 of the same frame (32 positions, causal), heads (31*2048, Hl); Qwen3 HF layers (sdpa)."""
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F

DEPTH, CBS = 32, 2048
CHUNK = 1024

# Qwen3-0.6B architecture (config.json of Qwen/Qwen3-0.6B, the fields that shape the model); random init (Qwen light_init "random")
QWEN3_06B = dict(hidden_size=1024, intermediate_size=3072, num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, head_dim=128,
                 rms_norm_eps=1e-6, rope_theta=1000000.0, max_position_embeddings=40960, hidden_act="silu", attention_bias=False,
                 initializer_range=0.02, tie_word_embeddings=False, vocab_size=(DEPTH - 1) * CBS)


# ------------------------------------------------------------------------------------------------ the speech text CE (trunk head)
TE = 151643                                            # first special id of Qwen's vocab (word pieces are < TE)


@torch.no_grad()
def _ce_forward(x, W, R, spec, y, wt, chunk, grad, xg=None, xg_w=False, xr=None, xr_norm=False, wn=None):
    N, H = x.shape
    V = W.shape[0]
    dev = x.device
    spec_t = torch.tensor(list(spec), device=dev, dtype=torch.long)
    Wc, Rc = W.to(x.dtype), R.to(x.dtype)
    st = torch.zeros(4, N, dtype=torch.float32, device=dev)      # CE, top-1 hit, p(target), normalised CE (softmax over word pieces < TE)
    loss = torch.zeros((), dtype=torch.float64, device=dev)
    dx = dW = dR = None
    if grad:
        dx = torch.zeros(N, H, dtype=torch.float32, device=dev)
        dW = torch.zeros(V, H, dtype=torch.float32, device=dev)
        dR = torch.zeros(R.shape[0], H, dtype=torch.float32, device=dev)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        xc, yc, wc = x[s:e], y[s:e], wt[s:e]
        wnc = None if wn is None else wn[s:e]
        zs = xc @ Wc.t()
        zs[:, spec_t] = (xc @ Rc.t()).to(zs.dtype)
        z = zs.float(); del zs
        lse = torch.logsumexp(z, -1, keepdim=True)
        zy = z.gather(1, yc[:, None])
        ce = (lse - zy)[:, 0]
        st[0, s:e] = ce; st[1, s:e] = (z.argmax(-1) == yc).float(); st[2, s:e] = torch.exp(-ce)
        if V > TE:                                     # klpre's normalized_word_b: softmax restricted to ids < 151643 (drops PAD / NW / EOB / specials)
            st[3, s:e] = torch.where(yc < TE, (torch.logsumexp(z[:, :TE], -1, keepdim=True) - zy)[:, 0], torch.zeros_like(ce))
        if wnc is None:
            loss += (wc.double() * ce.double()).sum()
        else:                                          # user 10-02 FACTORISED CE: wt x CE over categories {PAD, NW, EOB, ..., kl_term = all word
            isw = yc < TE                              #   pieces < TE (p = sum of their softmax)} + wn x NORMALISED word CE (softmax over < TE)
            u = torch.logsumexp(z[:, :TE], -1, keepdim=True)
            zc = torch.where(isw[:, None], u, zy)
            loss += (wc.double() * (lse - zc)[:, 0].double()).sum() + (wnc.double() * torch.where(isw, (u - zy)[:, 0], torch.zeros_like(ce)).double()).sum()
        if grad:
            r_ = torch.arange(e - s, device=dev)
            g = (z - lse).exp_()
            if wnc is None:
                g[r_, yc] -= 1.0
                g.mul_(wc[:, None])
            else:                                      # d/dz: wt (p - T) + wn 1[W] (q - onehot), T = q on W for kl_term targets, onehot otherwise
                g.mul_(wc[:, None])
                q = (z[:, :TE] - u).exp_()
                cw = torch.where(isw, wnc - wc, torch.zeros_like(wc))
                g[:, :TE] += cw[:, None] * q; del q
                g[r_, yc] -= torch.where(isw, wnc, wc)
            gb = g.to(x.dtype); del g
            gs = gb[:, spec_t]
            gb[:, spec_t] = 0
            if xr is not None and xr_norm:                 # head-first v7: trunk-side (dx + lm_head) grad = xr * full-softmax + (1 - xr) * NORMALISED
                xc_ = xr[s:e, None].float()                #   (softmax over word pieces < TE only; zero on PAD/NW/EOB/specials)
                gn = torch.zeros_like(z)
                gn[:, :TE] = torch.softmax(z[:, :TE], -1)
                ok_ = yc < TE
                gn[torch.arange(e - s, device=dev)[ok_], yc[ok_]] -= 1.0
                gn.mul_(wc[:, None]); gn[~ok_] = 0
                gb = (xc_ * gb.float() + (1.0 - xc_) * gn).to(x.dtype); del gn
            gw = gb if (xg is None or not xg_w) else gb * xg[s:e, None].to(gb.dtype)    # head-first: lm_head rows' timing grad x alpha too
            if gb.dtype == torch.float32:
                dW.addmm_(gw.t(), xc); dR.addmm_(gs.t(), xc)
                gr = gs if xr is None else gs * xr[s:e, None].to(gs.dtype)      # head-first v6: own rows' (PAD/NW/EOB) part of dx x xr
                dx[s:e] = gb @ Wc + gr @ Rc
            else:                                          # bf16 GEMMs, fp32 accumulate (e28head / klhead)
                torch.addmm(dW, gw.t(), xc, out_dtype=torch.float32, out=dW)
                torch.addmm(dR, gs.t(), xc, out_dtype=torch.float32, out=dR)
                gr = gs if xr is None else gs * xr[s:e, None].to(gs.dtype)
                dx[s:e] = torch.mm(gb, Wc, out_dtype=torch.float32) + torch.mm(gr, Rc, out_dtype=torch.float32)
            if xg is not None:                             # user 10-02 head-first: scale ONLY the gradient into the trunk (x), per position
                dx[s:e].mul_(xg[s:e, None].float())
            del gb, gs, gw
        del z, lse
    return loss.float(), st, dx, dW, dR


class _CEHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, R, y, wt, spec, chunk, xg=None, xg_w=False, xr=None, xr_norm=False, wn=None):
        grad = any(ctx.needs_input_grad[:3])
        loss, st, dx, dW, dR = _ce_forward(x, W, R, spec, y, wt, int(chunk), grad, xg, bool(xg_w), xr, bool(xr_norm), wn)
        ctx.dt = (x.dtype, W.dtype, R.dtype)
        if grad:
            ctx.save_for_backward(dx, dW, dR)
        ctx.mark_non_differentiable(st)
        return loss, st

    @staticmethod
    def backward(ctx, gl, gst):
        dx, dW, dR = ctx.saved_tensors
        gl = gl.float()
        return (dx * gl).to(ctx.dt[0]), (dW * gl).to(ctx.dt[1]), (dR * gl).to(ctx.dt[2]), None, None, None, None, None, None, None, None, None


def ce_head(x, W, R, y, wt, spec, chunk=None, xg=None, xg_w=False, xr=None, xr_norm=False, wn=None):
    """x [N, H] final-normed trunk hidden at the scored positions; W [V, H] lm_head; R [k, H] the own-parameter rows of `spec` (their
    W rows get exactly 0 gradient); y [N] targets; wt [N] fp32 weights. -> (sum_i wt_i CE_i fp32 scalar, stats [3, N])."""
    assert R.shape[0] == len(spec) and y.shape == wt.shape == (x.shape[0],) and wt.dtype == torch.float32, (R.shape, spec, y.shape, wt.dtype)
    chunk = CHUNK if chunk is None else int(chunk)
    spec = tuple(int(s) for s in spec)
    if not torch.is_grad_enabled():
        return _ce_forward(x, W, R, spec, y, wt, chunk, False)[:2]
    assert xg is None or xg.shape == y.shape, (None if xg is None else xg.shape, y.shape)
    assert wn is None or (wn.shape == y.shape and wn.dtype == torch.float32), "wn: per-position normalised word CE weight, fp32"
    return _CEHead.apply(x, W, R, y, wt, spec, chunk, xg, bool(xg_w), xr, bool(xr_norm), wn)


def ce_factor_reference(x, W, R, y, wt, wn, spec):
    """autograd reference of the factorised CE: wt x CE over categories (kl_term = logsumexp of word pieces < TE) + wn x normalised word CE."""
    z = (x @ W.t()).clone()
    z[:, list(spec)] = x @ R.t()
    z = z.float(); lse = torch.logsumexp(z, -1); u = torch.logsumexp(z[:, :TE], -1); zy = z.gather(1, y[:, None])[:, 0]
    isw = y < TE
    return (wt * (lse - torch.where(isw, u, zy))).sum() + (wn * torch.where(isw, u - zy, torch.zeros_like(u))).sum()


def ce_reference(x, W, R, y, wt, spec, xg=None, xg_w=False, xr=None):
    """plain autograd reference (tests): the same logits rule, full fp32 softmax; xg = the per-position gradient scale into x (forward
    unchanged: x' = xg x + (1 - xg) x.detach())."""
    if xg is not None:
        x = xg[:, None] * x + (1 - xg[:, None]) * x.detach()
    if xg is not None and xg_w:                          # W's grad from position i x xg_i (forward unchanged)
        z = (xg[:, None] * (x @ W.t()) + (1 - xg[:, None]) * (x @ W.detach().t())).clone()
    else:
        z = (x @ W.t()).clone()
    xs_ = x if xr is None else xr[:, None] * x + (1 - xr[:, None]) * x.detach()     # own-row logits: x's grad x xr
    z[:, list(spec)] = xs_ @ R.t()
    return (wt * F.cross_entropy(z.float(), y, reduction="none")).sum()


# ------------------------------------------------------------------------------------------------ AudioEmb / voice read
def code_feedback(depth, weight):
    """depth [T, 32] (or [B, T, 32]) global code ids, -1 = no audio -> [.., H] sum of the 32 code embeddings (0 where no audio)."""
    d = depth if depth.dim() == 3 else depth[None]
    mask = d[:, :, 0] >= 0
    flat = d.clamp_min(0).reshape(-1, d.size(2))
    fb = F.embedding_bag(flat, weight, mode="sum").view(d.size(0), d.size(1), -1)
    fb = fb * mask.unsqueeze(-1).to(fb.dtype)
    return fb if depth.dim() == 3 else fb[0]


def shift_one(r):
    """[.., T, H] -> position T holds r[T - 1], position 0 holds zeros (Qwen voice_read_embeds' shift)."""
    return torch.cat([r.new_zeros(*r.shape[:-2], 1, r.shape[-1]), r[..., :-1, :]], dim=-2)


def shift_segs(r, segs):
    """[T_total, H] packed rows -> each row shifted by one position (row start = zeros): shift_one per segment."""
    out, a = [], 0
    for L in segs:
        out.append(shift_one(r[a:a + L])); a += L
    assert a == r.shape[0], (a, r.shape)
    return torch.cat(out, 0) if len(out) > 1 else out[0]


def default_read_layers(n_layers, ref=(8, 24, 40, 52, 64), ref_n=64):
    """Qwen's concat5 read points scaled to n_layers (88 -> 11, 33, 55, 72, 88); strictly increasing, 1-indexed, last = n_layers."""
    assert n_layers >= 5, ("concat5 needs >= 5 trunk layers", n_layers)
    out = []
    for k in ref:
        v = max(1, int(round(k * n_layers / ref_n)))
        out.append(max(v, out[-1] + 1) if out else v)
    out[-1] = n_layers
    assert out == sorted(set(out)) and out[-1] == n_layers and len(out) == 5, out
    return out


# ------------------------------------------------------------------------------------------------ light tower (Qwen3-0.6B arch)
def qwen3_light_config(over=None, attn="sdpa"):
    from transformers import Qwen3Config
    c = dict(QWEN3_06B); c.update(over or {})
    cfg = Qwen3Config(**c)
    cfg._attn_implementation = attn
    cfg.use_cache = False
    return cfg


class LightTower(nn.Module):
    """FQNs: proj (Hv -> Hl, bias: Qwen's nn.Linear default), cb0_emb (2048, Hl), model (Qwen3Model: embed_tokens = the (31*2048, Hl) code
    table for cb1..cb30, 28 layers, norm, rotary), heads ((31*2048), Hl)."""

    def __init__(self, Hv, cfg):
        super().__init__()
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
        Hl = cfg.hidden_size
        self.proj = nn.Linear(Hv, Hl)
        self.cb0_emb = nn.Embedding(CBS, Hl)
        self.model = Qwen3Model(cfg)
        assert self.model.embed_tokens.weight.shape == ((DEPTH - 1) * CBS, Hl)
        self.heads = nn.Linear(Hl, (DEPTH - 1) * CBS, bias=False)

    def inputs(self, hv, depth, flat):
        """hv [T, Hv], depth [T, 32] global ids, flat [N] selected positions -> x [N, 32, Hl], labels [N, 31] (local codes 0..2047)."""
        codes = depth.index_select(0, flat)
        p = self.proj(hv.index_select(0, flat))
        c0 = self.cb0_emb(codes[:, 0])
        rest = self.model.embed_tokens(codes[:, 1:DEPTH - 1] - CBS)          # cb1..cb30 -> (d-1)*2048 + code
        x = torch.cat([p[:, None].to(c0.dtype), c0[:, None], rest.to(c0.dtype)], dim=1)
        offs = torch.arange(1, DEPTH, device=codes.device) * CBS
        return x, codes[:, 1:] - offs

    def logits(self, hl):
        w = self.heads.weight.view(DEPTH - 1, CBS, -1)
        return torch.einsum("ndh,dvh->ndv", hl[:, 1:], w)                     # hidden after e(cb_{d-1}) predicts cb_d, d = 1..31

    def forward(self, hv, depth, flat, wf=None):
        """-> (sum CE over N x 31 fp32, count, per-codebook CE sums [31] (detached)). wf given: forward_w's 4-tuple (through __call__, so
        the FSDP2 unit's hooks fire)."""
        if wf is not None:
            return self.forward_w(hv, depth, flat, wf)
        x, labels = self.inputs(hv, depth, flat)
        assert bool((labels >= 0).all()) and bool((labels < CBS).all()), "light labels out of range (a selected frame without audio)"
        hl = self.model(inputs_embeds=x, use_cache=False).last_hidden_state
        lo = self.logits(hl).float()
        ce = F.cross_entropy(lo.reshape(-1, CBS), labels.reshape(-1), reduction="none").view(labels.shape)
        return ce.sum(), ce.numel(), ce.detach().sum(0)

    def forward_w(self, hv, depth, flat, wf):
        """weighted: wf [N] fp32 per selected frame -> (sum_n wf_n sum_d CE_nd, unweighted CE sum (detached), count, per-codebook sums)."""
        x, labels = self.inputs(hv, depth, flat)
        assert bool((labels >= 0).all()) and bool((labels < CBS).all()), "light labels out of range (a selected frame without audio)"
        hl = self.model(inputs_embeds=x, use_cache=False).last_hidden_state
        lo = self.logits(hl).float()
        ce = F.cross_entropy(lo.reshape(-1, CBS), labels.reshape(-1), reduction="none").view(labels.shape)
        return (ce * wf[:, None]).sum(), ce.detach().sum(), ce.numel(), ce.detach().sum(0)


def cb0_ce_w(hv, w, labels, wpos):
    """cb0 CE with weights: hv [T, Hv], w [2048, Hv], labels [T] (-100 = none), wpos [T] fp32 (0 where no label) -> (sum_p wpos_p CE_p,
    unweighted CE sum (detached), count, top-1 hits)."""
    lo = (hv @ w.to(hv.dtype).t()).float()
    m = labels >= 0
    ce = F.cross_entropy(lo, labels, ignore_index=-100, reduction="none")
    hit = ((lo.argmax(-1) == labels) & m).sum()
    return (ce * wpos).sum(), ce.detach().sum(), m.sum(), hit


def cb0_ce(hv, w, labels):
    """voice cb0 head: hv [T, Hv], w [2048, Hv], labels [T] (-100 = none) -> (sum CE fp32, count, top-1 hits)."""
    lo = (hv @ w.to(hv.dtype).t()).float()
    m = labels >= 0
    ce = F.cross_entropy(lo, labels, ignore_index=-100, reduction="sum")
    hit = ((lo.argmax(-1) == labels) & m).sum()
    return ce, m.sum(), hit


# ------------------------------------------------------------------------------------------------ deterministic init (world-invariant)
@torch.no_grad()
def init_rows_(loc, row0, full_shape, kind, std, name, seed, chunk_rows=1024):
    """fill the local rows [row0, row0 + loc.shape[0]) of a parameter of global shape full_shape: kind 'normal' (N(0, std)), 'ones',
    'zeros'. Normal values are generated per CHUNK of global rows from a generator seeded by (seed, crc32(name), chunk): the same tensor
    on any world size / sharding."""
    if loc.numel() == 0:
        return
    if kind in ("ones", "zeros"):
        loc.fill_(1.0 if kind == "ones" else 0.0); return
    assert kind == "normal", kind
    n = loc.shape[0]; rest = tuple(full_shape[1:])
    c0, c1 = row0 // chunk_rows, (row0 + n - 1) // chunk_rows
    for c in range(c0, c1 + 1):
        g = torch.Generator(device="cpu").manual_seed((int(seed) * 1000003 + zlib.crc32(name.encode()) * 7919 + c) % (2 ** 63))
        a = c * chunk_rows; b = min(a + chunk_rows, full_shape[0])
        blk = torch.randn((b - a, *rest), generator=g) * float(std)
        lo, hi = max(a, row0), min(b, row0 + n)
        loc[lo - row0:hi - row0].copy_(blk[lo - a:hi - a].to(loc.dtype))
