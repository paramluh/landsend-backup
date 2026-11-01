"""PPM MODEL (aloe-vox-ft copy: + VOX EDIT, optional vadapter on channel A; see lib/README_COPIES.txt) (pose + podcast + text-duplex), FSDP2, bf16 compute / fp32 masters + fp32 reduce. GPU only (FA3 varlen + fused Triton).

BACKBONE  base Qwen3-32B (models/Qwen3-32B), PLAIN RoPE on the frame index (pid restarts per packed segment), NO adapters / LoRA: every
          weight trains (full fine-tune). Layers = the lineage's fused layer (vendor/fused_triton rms / add_rms / silu_mul / qk_prep +
          pinned FA3 varlen), the packed-attention context passed EXPLICITLY to every layer (no globals; the activation-checkpoint
          recompute sees the same segments).
INPUT     per position p, both channels summed (stage 3):  e[p] = sum_c present_c[p] * ( emb_c(ids_c[p]) + aud_c(depth_c[p]) )
          + pose_emb(pdepth[p]) on pose rows (channel B's pose codes of frame p - 3). emb_A, emb_B = copies of Qwen3-32B's embed_tokens;
          aud_A, aud_B, pose_emb = 32 x 2048 code tables (sum over codebooks, -1 = absent -> 0), RANDOM N(0, tok_std / sqrt(32)).
          Text rows: no audio (depth None) -> e = sum_c emb_c(ids_c). A text-only row with channel A absent (present_A = 0) is EXACTLY
          base Qwen3-32B's input (step-0 check 1).
          New input rows (base Qwen never trained ids >= 151669): SOA/EOA/SOB <- data_1ch.INIT_FROM means; NW <- PAD (stage 2);
          <sosa>/<sosb>/<sop> <- SOB row, <eosa>/<eosb>/<eop> <- <|im_end|> row (stage 3); PAD = stock.
TEXT HEAD (stage 3's HeadS3, channel B only: user decision 10-11 'B is the only predicted channel'): final RMSNorm -> P_B (glu
          projector, EXACT identity at init: W = I + D, D = 0; U = 0) -> full-softmax CE through Qwen3-32B's lm_head with the speech
          control columns replaced by OWN rows: PAD <- spec_out (N(0, 0.02), stage 1), NEW_WORD <- nw_out (= spec_out at init, stage 2
          'pad' rule), EOB (<|im_end|>) <- eob_out (= lm_head[<|im_end|>] at init). heads_s2q.ce_head (chunked, factorised CE).
TOWERS    two Qwen3-4B-architecture towers (36 layers, H 2560, GQA 32/8, random init): speech (Mimi cb0 of channel B) and pose (pose cb0).
          Conditioning = ONLY the backbone's final-normed last hidden state hn = RMSNorm_final(h) at T - 1, projected 5120 -> 2560 (cond).
          tower input at T = text_emb(lane_B[T]) + code_emb(codes of frame T - 3 - 1 = depth[T - 1], 32 summed tables) + cond(hn[T - 1])
          (shifts restart per packed segment; row start gets no codes / no conditioning). Label at T = cb0 of frame T - 3 = depth[T, 0].
          text_emb rows = TOWER_VOCAB (151,682 = every input id a PPM row carries: Qwen ids + controls 151670-151679 + persona <sop>/<eop>
          151680/151681). cb0_head 2560 -> 2048.
POSE HEAD (D32, 10-12, PPM_POSE_HEAD=1; default OFF = the module does not exist): auxiliary direct pose cb0 head on the backbone (as SAM
          has): pose_head = Linear(5120 -> 2048, bias), weight AND bias ZERO at init. Position T reads hn[T - 1] (= the pose tower's
          conditioning at T: Tower.forward's shift1(cond(hn))) and predicts the pose tower's label at T (pcb0[T] = pose cb0 of frame T - CD) with
          the tower's weights pcb0_w: the SAME (conditioning position, target) pairs, the same mask. Pose rows only. Training only: inference
          uses the tower.
          D35 (10-13, user OK; default OFF = the head reads hn exactly as above): the head's gradient into the backbone is scaled by
          model.pose_head_bbgrad = s (None = off, the trainer sets it every step): the head reads posehead_in(hn, s) = hn.detach() +
          s * (hn - hn.detach()) (forward value = hn, so the head's loss is unchanged for every s; backbone grad from the head x s; the head's
          own grad unchanged); s <= 0 = hn.detach() (full detach: the head is a probe).
          D36 (10-14, user OK; default ON = as above): model.pose_head_on = False (the trainer sets it under PPM_POSEHEAD_OFF=1) skips the
          head in forward entirely (no phead* terms, no compute); the module and its weights stay (frozen, unused, saved) so it can be revived.
DECP_BB   (D33, 10-12, PPM_DECP_BB=1; default OFF = the module does not exist; TEMPORARY, user: "as a temporary measure to get the connection
          and then we detach from the backbone"): a direct link from the backbone into the POSE decoder: decp_bb = Linear(5120 -> 1024 = the
          0.6B decoder's hidden width, bias), weight AND bias ZERO at init. zb[T] = alpha * decp_bb(hn[T - 1]) (= shift1(decp_bb(hn)), segment
          starts <- 0: the very position the pose tower's conditioning reads), alpha = model.decp_bb_alpha (set by the trainer). The pose
          decoder's input slot 0 for frame T becomes proj(hv_p[T]) + zb[T] (additive, fp32 like the slot itself). zb is computed on the
          backbone side; the trainer's nanosteps hand the decoder a DETACHED copy and feed its accumulated gradient back through zb (so into
          decp_bb AND the backbone) in the micro's one backward, exactly like hv_p. Speech decoder unchanged.
DECODERS  two Qwen3-0.6B-architecture depth decoders (random init; heads_s2q.LightTower): per selected frame [proj(hv[T]), e(cb0), e(cb1..30)]
          -> cb1..cb31, one per tower (speech: Mimi; pose: pose codes). Trained by NANOSTEPS in the trainer (own AdamW, K chunked updates
          per micro on the detached tower output; the accumulated input gradient flows back through the tower once). T25 speed path (training
          nanosteps only): the compiled core runs _dec_trunk (= the HF Qwen3Model forward, op for op) with the fused T = 32 attention kernel
          vendor/dec_attn32 instead of SDPA's masked memory-efficient kernel; dynamic frame dimension. Eval keeps the eager HF path.
INIT      backbone tensors <- Qwen3-32B safetensors (row slices per FSDP shard, bf16 -> fp32 exact); every random tensor seeded per NAME and
          global row chunk (heads_s2q.init_rows_: world-size invariant)."""
import json
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "vendor"), os.path.join(HERE, "lib"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import fused_triton as FO  # noqa: E402
import fa3_op  # noqa: E402
import dec_attn32 as DA  # noqa: E402
import heads_s2q as H2  # noqa: E402
import speech_q as SP  # noqa: E402
import vadapter_s4q as VA  # noqa: E402   (aloe-vox-ft: frozen copy of stage4-asr2/vadapter_s4q.py = the ASR adapter run's class)
from transformers.models.qwen3 import modeling_qwen3 as Q3  # noqa: E402

QWEN32 = "/data/home/amartyavarma/models/Qwen3-32B"
QWEN4B_CFG = "/data/home/amartyavarma/ppm/configs/Qwen3-4B"
PAD, NW, EOB, SOB, SOA, EOA = SP.PAD_ID, SP.NW_ID, SP.EOB_ID, SP.SOB_ID, SP.SOA_ID, SP.EOA_ID
IM_START, IM_END, USER_ID, ASST_ID = 151644, 151645, 872, 77091
SOSA, EOSA, SOSB, EOSB, SOP, EOP = 151676, 151677, 151678, 151679, 151680, 151681
SPEC = (PAD, NW, EOB)                         # own output rows (spec_out, nw_out, eob_out), in this order
TOWER_VOCAB = 151682
INIT_SEED = 20261011
DEPTH, CBS = SP.DEPTH, SP.CBS


# ------------------------------------------------------------------------------------------------ configs
def _sdpa(c, layers=None):
    c._attn_implementation = "sdpa"           # construction only: every PPM attention runs the explicit FA3 varlen path below
    c.dtype = torch.bfloat16
    if layers is not None:
        c.num_hidden_layers = int(layers)
    return c


def cfg32(layers=None):
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(QWEN32)
    assert (c.hidden_size, c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.vocab_size) == (5120, 64, 64, 8, 128, 151936)
    assert not c.tie_word_embeddings
    return _sdpa(c, layers)


def cfg4b(layers=None):
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(QWEN4B_CFG)
    assert (c.hidden_size, c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.intermediate_size) == (2560, 36, 32, 8, 128, 9728)
    return _sdpa(c, layers)


def cfg06(layers=None):
    return H2.qwen3_light_config({"num_hidden_layers": int(layers)} if layers is not None else None)


# ------------------------------------------------------------------------------------------------ packed attention layer
class Ctx:
    """packed segments -> FA3 varlen arguments (cu int32 on device, max length as a python int: no host sync in the layers)."""

    def __init__(self, segs, dev):
        segs = [int(s) for s in segs]
        cu = [0]
        for s in segs:
            cu.append(cu[-1] + s)
        self.cu = torch.tensor(cu, dtype=torch.int32, device=dev)
        self.max, self.n, self.segs = max(segs), cu[-1], segs
        self.starts = torch.tensor(cu[:-1], dtype=torch.int64, device=dev)


def attn_fwd(at, x, pe, ctx):
    T = x.shape[1]
    assert x.shape[0] == 1 and T == ctx.n, (x.shape, ctx.n)
    cos, sin = pe
    eps = float(at.q_norm.variance_epsilon)
    q, k = FO.qk_prep(at.q_proj(x), at.k_proj(x), at.q_norm.weight, at.k_norm.weight, eps, cos, sin)
    v = at.v_proj(x).view(T, -1, at.head_dim)
    a = fa3_op.fa3_varlen_func(q[0], k[0], v, ctx.cu, ctx.cu, ctx.max, ctx.max, at.scaling, True)
    return at.o_proj(a.reshape(1, T, -1))


def layer_fwd(L, h, pe, ctx):
    """model_q.layer_forward (E28 fused layer) with the explicit packed context."""
    ln1, ln2, mlp = L.input_layernorm, L.post_attention_layernorm, L.mlp
    x = FO.rms(h, ln1.weight, float(ln1.variance_epsilon))
    a = attn_fwd(L.self_attn, x, pe, ctx)
    h, x = FO.add_rms(h, a, ln2.weight, float(ln2.variance_epsilon))
    return h + mlp.down_proj(FO.silu_mul(mlp.gate_proj(x), mlp.up_proj(x)))


class Block(nn.Module):
    """one Qwen3 decoder layer (FQN .layer.*) under optional non-reentrant activation checkpointing (ac)."""

    def __init__(self, cfg, i):
        super().__init__()
        self.layer, self.ac = Q3.Qwen3DecoderLayer(cfg, i), True

    def forward(self, h, pe, ctx):
        f = lambda hh: layer_fwd(self.layer, hh, pe, ctx)  # noqa: E731
        return checkpoint(f, h, use_reentrant=False) if (self.ac and torch.is_grad_enabled()) else f(h)


class Rot:
    def __init__(self, cfg, dev):
        r = Q3.Qwen3RotaryEmbedding(config=cfg)
        assert r.inv_freq.dtype == torch.float32 and r.attention_scaling == 1.0
        self.r = r.to(dev)

    def __call__(self, x, pid):
        return self.r(x, pid)


def shift1(x, ctx, fill):
    """position T <- x[T - 1] within each packed segment; segment starts <- fill."""
    y = torch.roll(x, 1, 0)
    m = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    m[ctx.starts] = True
    return y.masked_fill(m.view(-1, *([1] * (x.dim() - 1))), fill)


# ------------------------------------------------------------------------------------------------ modules
class CodeTable(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(DEPTH * CBS, H))

    def forward(self, depth):
        return H2.code_feedback(depth, self.weight)


class Inp(nn.Module):
    def __init__(self, V, H):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(V, H) for _ in range(2)])
        self.aud = nn.ModuleList([CodeTable(H) for _ in range(2)])
        self.pose = CodeTable(H)

    def forward(self, ids, depth=None, present=None, pdepth=None):
        out = None
        for c in range(2):
            x = self.emb[c](ids[c])
            if depth is not None:
                x = x + self.aud[c](depth[c]).to(x.dtype)
            if present is not None:
                x = x * present[c].to(x.dtype)[:, None]
            out = x if out is None else out + x
        if pdepth is not None:
            out = out + self.pose(pdepth).to(out.dtype)
        return out


class VGain(nn.Module):
    """aloe-vox-ft (user / coordinator 10-16): the channel-A scale s = exp(theta), one scalar for every vector, theta trained (own group
    'vgain'), kept in fp32 (own FSDP unit with param_dtype fp32: bf16 theta would quantise s in ~3 % steps)."""

    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(1))

    def forward(self, y):
        return y * torch.exp(self.theta.float()).to(y.dtype)


class GluProj(nn.Module):
    """stage 3's S3_PROJ=glu: P(x) = x + x D^T + (SiLU(x G^T) * (x V^T)) U^T; D, U zero -> exact identity at init."""

    def __init__(self, H, R):
        super().__init__()
        self.D = nn.Parameter(torch.empty(H, H))
        self.G = nn.Parameter(torch.empty(R, H))
        self.V = nn.Parameter(torch.empty(R, H))
        self.U = nn.Parameter(torch.empty(H, R))

    def forward(self, x):
        return x + F.linear(x, self.D) + F.linear(F.silu(F.linear(x, self.G)) * F.linear(x, self.V), self.U)


class Head(nn.Module):
    def __init__(self, cfg, proj_r):
        super().__init__()
        H = cfg.hidden_size
        self.norm = Q3.Qwen3RMSNorm(H, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(H, cfg.vocab_size, bias=False)
        self.spec_out = nn.Parameter(torch.empty(1, H))
        self.nw_out = nn.Parameter(torch.empty(1, H))
        self.eob_out = nn.Parameter(torch.empty(1, H))
        self.proj = GluProj(H, proj_r)

    def forward(self, h, sel, y, wt, wn, want_logits=False):
        """h [N, H] -> (weighted text CE, stats [4, S], hn [N, H] the final-normed states (tower conditioning)). want_logits: + raw
        lm_head logits [S, V] of P_B(hn[sel]) (no own-row replacement; step-0 HF check)."""
        hn = FO.rms(h, self.norm.weight, float(self.norm.variance_epsilon))
        x = self.proj(hn.index_select(0, sel))
        if want_logits:
            return None, None, hn, (x @ self.lm_head.weight.to(x.dtype).t()).float()
        R = torch.cat([self.spec_out, self.nw_out, self.eob_out], 0)
        loss, st = H2.ce_head(x, self.lm_head.weight, R, y, wt, SPEC, wn=wn)
        return loss, st, hn


TPROF = {"on": False, "ev": []}                 # PPM_PROFILE: CUDA events around each tower's forward and backward (name, kind, start, end)


def _ev():
    e = torch.cuda.Event(enable_timing=True); e.record(); return e


class Tower(nn.Module):
    def __init__(self, cfg, Ht, vocab):
        super().__init__()
        Hv = cfg.hidden_size
        self.text_emb = nn.Embedding(vocab, Hv)
        self.code_emb = CodeTable(Hv)
        self.cond = nn.Linear(Ht, Hv, bias=False)
        self.layers = nn.ModuleList([Block(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = Q3.Qwen3RMSNorm(Hv, eps=cfg.rms_norm_eps)
        self.cb0_head = nn.Linear(Hv, CBS, bias=False)

    def forward(self, ids, depth, hn, pid, ctx, rot, lab, wpos, info=None):
        """ids [M] lane, depth [M, 32] codes of frame T - 3 at T, hn [M, Ht] backbone final-normed states at the same positions, lab [M]
        cb0 labels (-100 none), wpos [M] fp32 -> (hv [M, Hv] final-normed, sum wpos CE, CE sum (detached), n labels, top-1 hits)."""
        prof = TPROF["on"] and torch.is_grad_enabled()
        if prof:
            ef0 = _ev()
        cnd = shift1(self.cond(hn), ctx, 0.0)
        e = self.text_emb(ids)
        ce_ = self.code_emb(shift1(depth, ctx, -1))
        if info is not None:
            info.update(tower_rms_text=float(e.float().pow(2).mean().sqrt()), tower_rms_codes=float(ce_.float().pow(2).mean().sqrt()),
                        tower_rms_cond=float(cnd.float().pow(2).mean().sqrt()), backbone_rms_hn=float(hn.float().pow(2).mean().sqrt()))
        e = (e + ce_.to(e.dtype) + cnd.to(e.dtype))[None]
        if prof:
            rec = {"name": getattr(self, "_pname", "tower")}
            e.register_hook(lambda g, rec=rec: rec.__setitem__("b1", _ev()))        # tower backward finished (grad of its input)
        pe = rot(e, pid)
        h = e
        for L in self.layers:
            h = L(h, pe, ctx)
        hv = FO.rms(h[0], self.norm.weight, float(self.norm.variance_epsilon))
        lo = (hv @ self.cb0_head.weight.to(hv.dtype).t()).float()
        m = lab >= 0
        ce = F.cross_entropy(lo, lab, ignore_index=-100, reduction="none")
        hit = ((lo.argmax(-1) == lab) & m).sum()
        if prof:
            rec.update(f0=ef0, f1=_ev())
            hv.register_hook(lambda g, rec=rec: rec.__setitem__("b0", _ev()))       # tower backward starts (grad of its output)
            TPROF["ev"].append(rec)
        return hv, (ce * wpos).sum(), ce.detach().sum(), m.sum(), hit


DEC_FAST = {"active": False, "head": None}      # T14: compiled decoder code runs only while active (training nanosteps)


def _dec_head_ce(hl, w, labels, wf):
    """31 x 2048-way heads + CE + weighted sum (one graph): hl [n, 32, Hl], w [31 x 2048, Hl], labels [n, 31], wf [n] -> (loss, CE [n, 31])."""
    lo = torch.einsum("ndh,dvh->ndv", hl[:, 1:], w.view(DEPTH - 1, CBS, -1)).float()
    ce = F.cross_entropy(lo.reshape(-1, CBS), labels.reshape(-1), reduction="none").view(labels.shape)
    return (ce * wf[:, None]).sum(), ce.detach()


DEC_ATTN = os.environ.get("PPM_DEC_ATTN", "attn32")   # speed knob (same maths): attn32 = _dec_trunk + fused T=32 kernel | hf = HF Qwen3Model forward


def _dec_trunk(m, x):
    """= HF Qwen3Model.forward(inputs_embeds=x, use_cache=False).last_hidden_state for the depth decoder (positions 0..31, causal, no
    mask tensor), op for op, with the attention call = vendor/dec_attn32 (fused causal GQA for T = 32; HF's SDPA call casts q, k, v to bf16
    under autocast, done explicitly here because a custom op has no autocast rule). Residual stream / norms / RoPE / MLP = the HF modules."""
    B, T, _ = x.shape
    cos, sin = m.rotary_emb(x, torch.arange(T, device=x.device)[None])
    h = x
    for L in m.layers[: m.config.num_hidden_layers]:
        a = L.self_attn
        y = L.input_layernorm(h)
        shp = (B, T, -1, a.head_dim)
        q = a.q_norm(a.q_proj(y).view(shp))
        k = a.k_norm(a.k_proj(y).view(shp))
        v = a.v_proj(y).view(shp)
        q, k = Q3.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
        bf = torch.bfloat16 if torch.is_autocast_enabled() else q.dtype
        o = DA.attn32(q.to(bf), k.to(bf), v.to(bf), a.scaling)
        h = h + a.o_proj(o.reshape(B, T, -1))
        h = h + L.mlp(L.post_attention_layernorm(h))
    return m.norm(h)


def _dec_core(dec, hs, codes, wf, zb=None):
    """decoder on PRE-SELECTED frames (static shapes for the compiled region): hs [n, Hv] tower states, codes [n, 32] global code ids, wf [n]
    -> (weighted CE sum, CE [n, 31] detached). = heads_s2q.LightTower.inputs + model + heads/CE. zb [n, Hl] (D33, pose decoder only): added
    to input slot 0 (proj(hs) + zb, in the slot's dtype); None = the pre-D33 graph exactly."""
    p = dec.proj(hs)
    c0 = dec.cb0_emb(codes[:, 0])
    rest = dec.model.embed_tokens(codes[:, 1:DEPTH - 1] - CBS)
    p0 = p[:, None].to(c0.dtype) if zb is None else p[:, None].to(c0.dtype) + zb[:, None].to(c0.dtype)
    x = torch.cat([p0, c0[:, None], rest.to(c0.dtype)], dim=1)
    labels = codes[:, 1:] - torch.arange(1, DEPTH, device=codes.device) * CBS
    hl = _dec_trunk(dec.model, x) if DEC_ATTN == "attn32" else dec.model(inputs_embeds=x, use_cache=False).last_hidden_state
    return _dec_head_ce(hl, dec.heads.weight, labels, wf)


def compile_decoder(dec, mode="default", whole=True, dynamic=False):
    """T14: torch.compile of the decoder (static shapes), ACTIVE-gated (training nanosteps only; eval stays eager). whole (replicated
    decoders): ONE compiled region = inputs + 28 layers + heads + CE; else (FSDP-sharded decoders) each layer's forward separately (the
    FSDP2 hooks stay on the module __call__) + the heads/CE. Same math, kernel-level rounding only."""
    import functools
    import types
    if whole:
        dec._core_c = torch.compile(functools.partial(_dec_core, dec), mode=mode, fullgraph=False, dynamic=None if dynamic else False)
        dec._core_dyn = bool(dynamic)
        return
    for L in dec.model.layers:
        inner = getattr(L, "_checkpoint_wrapped_module", L)
        eager = inner.forward
        comp = torch.compile(eager, mode=mode, fullgraph=True, dynamic=False)
        inner.forward = types.MethodType(lambda self, *a, _c=comp, _e=eager, **k: _c(*a, **k) if DEC_FAST["active"] else _e(*a, **k), inner)
    if DEC_FAST["head"] is None:
        DEC_FAST["head"] = torch.compile(_dec_head_ce, mode=mode, fullgraph=True, dynamic=False)


class Decoder(H2.LightTower):
    def inputs(self, hv, depth, flat, zb=None, tower=True):
        """= LightTower.inputs; zb [T, Hl] (D33): slot 0 of frame flat[i] = proj(hv[flat[i]]) + zb[flat[i]] (as _dec_core). tower=False
        (D33 eval probe 'backbone-only'): the tower term proj(hv) (incl. its bias) is dropped, slot 0 = zb[flat[i]] alone."""
        if zb is None:
            assert tower
            return super().inputs(hv, depth, flat)
        codes = depth.index_select(0, flat)
        c0 = self.cb0_emb(codes[:, 0])
        rest = self.model.embed_tokens(codes[:, 1:DEPTH - 1] - CBS)
        zs = zb.index_select(0, flat)[:, None].to(c0.dtype)
        p0 = self.proj(hv.index_select(0, flat))[:, None].to(c0.dtype) + zs if tower else zs
        x = torch.cat([p0, c0[:, None], rest.to(c0.dtype)], dim=1)
        return x, codes[:, 1:] - torch.arange(1, DEPTH, device=codes.device) * CBS

    def forward(self, hv, depth, flat, wf, n_valid=None, zb=None, tower=True):
        """weighted 0.6B CE: wf [n] per selected frame -> (sum_n wf_n sum_d CE_nd, CE sum over the first n_valid frames (detached; all if
        None), their count x 31, per-codebook sums [31]). Padded static chunks carry weight 0 after n_valid (zero loss, zero gradient).
        bf16 autocast (replicated decoders keep fp32 masters on the GPU; FSDP-sharded ones already run bf16 parameters).
        zb [T, Hl] (D33 backbone link, pose decoder only; None = unchanged): added to slot 0 of each selected frame (same `flat` as hv).
        tower=False (D33 eval probe only, eager path): slot 0 = zb alone (the tower term dropped)."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            core = getattr(self, "_core_c", None)
            if DEC_FAST["active"] and core is not None:              # index_select OUTSIDE the compiled region: static [CH, ..] shapes
                assert tower, "tower=False is an eval probe (eager path)"
                hs_, cd_ = hv.index_select(0, flat), depth.index_select(0, flat)
                zs_ = zb.index_select(0, flat) if zb is not None else None
                if getattr(self, "_core_dyn", False):                # one compiled graph for every micro-batch size (frames dim symbolic)
                    for t_ in (hs_, cd_, wf) + ((zs_,) if zs_ is not None else ()):
                        torch._dynamo.mark_dynamic(t_, 0)
                loss, ce = core(hs_, cd_, wf) if zs_ is None else core(hs_, cd_, wf, zs_)
            else:
                x, labels = self.inputs(hv, depth, flat) if zb is None else self.inputs(hv, depth, flat, zb, tower)
                hl = self.model(inputs_embeds=x, use_cache=False).last_hidden_state
                f = DEC_FAST["head"] if (DEC_FAST["active"] and DEC_FAST["head"] is not None) else _dec_head_ce
                loss, ce = f(hl, self.heads.weight, labels, wf)
        if n_valid is not None:
            ce = ce[:n_valid]
        return loss, ce.sum(), ce.numel(), ce.sum(0)


def pose_head_ce(head, hn, ctx, lab, wpos):
    """D32 auxiliary pose cb0 head: x[T] = hn[T - 1] (shift1 within each packed segment, segment starts <- 0: exactly the positions the pose
    tower's conditioning shift1(cond(hn)) reads), logits = head(x) [N, 2048], CE against the pose tower's labels lab [N] (-100 none) with its
    weights wpos [N] -> (sum wpos CE, CE sum (detached), n labels, top-1 hits) = Tower.forward's loss terms."""
    x = shift1(hn, ctx, 0.0)
    lo = head(x).float()
    m = lab >= 0
    ce = F.cross_entropy(lo, lab, ignore_index=-100, reduction="none")
    hit = ((lo.argmax(-1) == lab) & m).sum()
    return (ce * wpos).sum(), ce.detach().sum(), m.sum(), hit


def posehead_in(hn, s):
    """D35: the pose head's input. s None = hn itself (pre-D35 behaviour exactly); s <= 0 = hn.detach() (full detach); else the gradient-
    scaling straight-through hn.detach() + s * (hn - hn.detach()): forward value hn, gradient into hn = s x the head's input gradient."""
    if s is None:
        return hn
    d = hn.detach()
    if s <= 0.0:
        return d
    return d + float(s) * (hn - d)


class PPM(nn.Module):
    def __init__(self, c32, c4, c06, proj_r=2048, tower_vocab=TOWER_VOCAB, pose_head=False, decp_bb=False, vadapter=None):
        super().__init__()
        self.inp = Inp(c32.vocab_size, c32.hidden_size)
        self.layers = nn.ModuleList([Block(c32, i) for i in range(c32.num_hidden_layers)])
        self.head = Head(c32, proj_r)
        self.tower_s = Tower(c4, c32.hidden_size, tower_vocab)
        self.tower_p = Tower(c4, c32.hidden_size, tower_vocab)
        self.tower_s._pname, self.tower_p._pname = "tower_s", "tower_p"
        self.dec_s = Decoder(c4.hidden_size, c06)
        self.dec_p = Decoder(c4.hidden_size, c06)
        if pose_head:                                              # D32 (flag off: no such module, no key, no optimizer group)
            self.pose_head = nn.Linear(c32.hidden_size, CBS, bias=True)
            self.pose_head_bbgrad = None                           # D35: the head's backbone-gradient scale (None = off; trainer sets it)
            self.pose_head_on = True                               # D36: False = the head is skipped in forward (trainer: PPM_POSEHEAD_OFF=1)
        if decp_bb:                                                # D33 (flag off: no such module, no key, no optimizer group)
            self.decp_bb = nn.Linear(c32.hidden_size, c06.hidden_size, bias=True)
            self.decp_bb_alpha = 1.0                               # the link's scale (the trainer sets it every step: PPM_DECP_BB_ALPHA / fade)
        if vadapter:                                               # aloe-vox-ft VOX EDIT: channel A = vadapter(Voxtral latents) (flag off: no module)
            self.vadapter = VA.VAdapter(int(vadapter["enc_hidden"]), c32.hidden_size, int(vadapter["hidden"]), int(vadapter["stack"]),
                                        norm=vadapter["norm"])
            self.vgain = VGain()                                   # A latent = exp(theta) * adapter(v): ONE learnable scalar s = exp(theta)

    def forward(self, b, rot, rot_t, want_logits=False, info=None):
        """one micro (data_ppm common format, on the device, + trainer weights wt / wn / cb0_w / pcb0_w) -> dict of terms. The decoders
        are NOT called here (trainer nanosteps); speech / pose rows return the towers' outputs hv_s / hv_p."""
        kind = b["kind"]
        # 10-12 dependence probes (train_ppm codependence / xdep evals): optional TRUNK-input overrides ids_t / depth_t / present_t / pdepth_t;
        # the towers, decoders and every label keep the micro's own ids[1] / depth / pdepth / cb0 / pcb0. Absent keys = unchanged behaviour.
        e = self.inp(b["ids_t"] if "ids_t" in b else b["ids"], b["depth_t"] if "depth_t" in b else b.get("depth"),
                     b["present_t"] if "present_t" in b else b.get("present"), b["pdepth_t"] if "pdepth_t" in b else b.get("pdepth"))[None]
        if b.get("vlat") is not None:                              # VOX EDIT: + vadapter(latents [1, 4n, 1280]) at positions vpos (A's row frames)
            y = self.vgain(self.vadapter(b["vlat"])[0])
            if getattr(self, "vadapter_test_scale", None) is not None:   # TEST ONLY diagnostic (trainer: eval-only test mode); None in training
                y = y * float(self.vadapter_test_scale)
            assert y.shape[0] == b["vpos"].shape[0], (tuple(y.shape), tuple(b["vpos"].shape))
            e = e + torch.zeros_like(e[0]).index_copy(0, b["vpos"], y.to(e.dtype))[None]
        ctx = Ctx(b["segs"], e.device)
        pe = rot(e, b["pid"])
        h = e
        for L in self.layers:
            h = L(h, pe, ctx)
        out = {}
        if want_logits:
            _, _, hn, lg = self.head(h[0], b["sel"], None, None, None, want_logits=True)
            out["logits"] = lg
            return out
        lt, st, hn = self.head(h[0], b["sel"], b["y"], b["wt"], b.get("wn"))
        out.update(text=lt, text_stats=st)
        if kind in ("speech", "pose"):
            hv, l0, s0, n0, k0 = self.tower_s(b["ids"][1], b["depth"][1], hn, b["pid"], ctx, rot_t, b["cb0"], b["cb0_w"], info)
            out.update(hv_s=hv, cb0=l0, cb0_sum=s0, cb0_n=n0, cb0_hit=k0)
        if kind == "pose":
            hv, l0, s0, n0, k0 = self.tower_p(b["ids"][1], b["pdepth"], hn, b["pid"], ctx, rot_t, b["pcb0"], b["pcb0_w"])
            out.update(hv_p=hv, pcb0=l0, pcb0_sum=s0, pcb0_n=n0, pcb0_hit=k0)
            if hasattr(self, "pose_head") and self.pose_head_on:   # D32: same (hn[T - 1], pcb0[T]) pairs and weights as the pose tower (D36: off = skipped)
                l1, s1, n1, k1 = pose_head_ce(self.pose_head, posehead_in(hn, self.pose_head_bbgrad), ctx, b["pcb0"], b["pcb0_w"])
                out.update(phead=l1, phead_sum=s1, phead_n=n1, phead_hit=k1)
            if hasattr(self, "decp_bb"):                           # D33: zb[T] = alpha * decp_bb(hn[T - 1]) (segment starts 0) -> pose decoder
                out["zb_p"] = shift1(self.decp_bb(hn), ctx, 0.0) * float(self.decp_bb_alpha)
        return out


# ------------------------------------------------------------------------------------------------ groups
GROUPS = ("trunk", "spec", "proj", "audio", "tower_s", "tower_p", "pose_head", "decp_bb")   # main AdamW (pose_head only with D32 on, decp_bb
                                                                                  # only with D33 on: empty groups are skipped); dec_s / dec_p
                                                                                  # have their own nanostep AdamWs


def group_of(name):
    n = name.replace("_checkpoint_wrapped_module.", "")
    if n.startswith("inp.emb.") or n.startswith("layers.") or n in ("head.norm.weight", "head.lm_head.weight"):
        return "trunk"
    if n in ("head.spec_out", "head.nw_out", "head.eob_out"):
        return "spec"
    if n.startswith("head.proj."):
        return "proj"
    if n.startswith("inp.aud.") or n.startswith("inp.pose."):
        return "audio"
    if n.startswith("pose_head."):
        return "pose_head"
    if n.startswith("decp_bb."):
        return "decp_bb"
    if n.startswith("vadapter."):                                  # VOX EDIT
        return "vadapter"
    if n.startswith("vgain."):                                     # VOX EDIT: s = exp(theta), own group
        return "vgain"
    for g in ("tower_s", "tower_p", "dec_s", "dec_p"):
        if n.startswith(g + "."):
            return g
    raise KeyError(n)


# ------------------------------------------------------------------------------------------------ init
def _loc(p):
    return p.to_local() if hasattr(p, "to_local") else p


def _rows(p):
    if not hasattr(p, "to_local"):
        return 0, p.shape[0]
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
    shp, off = compute_local_shape_and_global_offset(p.shape, p.device_mesh, p.placements)
    return (int(off[0]) if len(off) else 0), (int(shp[0]) if len(shp) else 0)


def get_row(p, i, dev):
    a, r = _rows(p)
    row = torch.zeros(p.shape[1:], dtype=torch.float32, device=dev)
    if a <= i < a + r:
        row.copy_(_loc(p)[i - a].float())
    if dist.is_initialized():
        dist.all_reduce(row)
    return row


@torch.no_grad()
def set_row(p, i, row):
    a, r = _rows(p)
    if a <= i < a + r:
        _loc(p)[i - a].copy_(row.to(_loc(p).dtype))


def init_rule(n, shape, c4, c06, tok_std):
    """(kind, std) of every random / derived tensor (module doc)."""
    if n.startswith("inp.aud.") or n == "inp.pose.weight":
        return "normal", tok_std / DEPTH ** 0.5
    if n in ("head.spec_out", "head.nw_out", "head.eob_out"):
        return "normal", 0.02                 # nw_out / eob_out are overwritten right after (derived rows)
    if n in ("head.proj.D", "head.proj.U"):
        return "zeros", 0.0
    if n in ("pose_head.weight", "pose_head.bias"):          # D32: zero init (step 1: CE = ln 2048, exactly zero gradient into the backbone)
        return "zeros", 0.0
    if n in ("decp_bb.weight", "decp_bb.bias"):              # D33: zero init (zb = 0 exactly: the pose decoder's input unchanged at enable)
        return "zeros", 0.0
    if n in ("head.proj.G", "head.proj.V"):
        return "normal", 1.0 / shape[1] ** 0.5
    for t in ("tower_s.", "tower_p."):
        if n.startswith(t):
            k = n[len(t):]
            if k == "text_emb.weight":
                return "normal", 0.02
            if k == "code_emb.weight":
                return "normal", 0.02 / DEPTH ** 0.5
            if k == "cond.weight":
                return "normal", 0.02 / shape[1] ** 0.5
            if k == "cb0_head.weight":
                return "normal", 1.0 / (3 * c4.hidden_size) ** 0.5
            if k.endswith("norm.weight") or k.endswith("layernorm.weight"):
                return "ones", 0.0
            return "normal", 0.02                 # HF Qwen3 initializer_range for every layer matrix
    for t in ("dec_s.", "dec_p."):
        if n.startswith(t):
            return _light_rule("light." + n[len(t):], shape, c4, c06)
    raise KeyError(n)


def _light_rule(n, shape, c4, c06):
    import types
    vcfg = types.SimpleNamespace(hidden_size=c4.hidden_size)
    if n == "light.proj.weight":
        return "normal", 1.0 / (3 * vcfg.hidden_size) ** 0.5
    if n == "light.proj.bias":
        return "zeros", 0.0
    if n == "light.heads.weight":
        return "normal", 1.0 / (3 * c06.hidden_size) ** 0.5
    if n.endswith("norm.weight") or n.endswith("layernorm.weight"):
        return "ones", 0.0
    return "normal", 0.02


def hf_key(n):
    n = n.replace("_checkpoint_wrapped_module.", "")
    if n.startswith("inp.emb."):
        return "model.embed_tokens.weight"
    if n.startswith("layers."):
        return "model." + n.replace(".layer.", ".", 1)
    if n == "head.norm.weight":
        return "model.norm.weight"
    if n == "head.lm_head.weight":
        return "lm_head.weight"
    return None


class Shards:
    def __init__(self, root=QWEN32):
        from safetensors import safe_open
        self.root, self.ix, self._f, self._open = root, json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"], {}, safe_open

    def sl(self, key):
        fn = self.ix[key]
        if fn not in self._f:
            self._f[fn] = self._open(f"{self.root}/{fn}", "pt")
        return self._f[fn].get_slice(key)


@torch.no_grad()
def init_weights(model, c32, c4, c06, dev, log=print, partial=False):
    """backbone <- Qwen3-32B (row slices), every other tensor its seeded rule, then the derived rows. -> info dict."""
    sh = Shards()
    used, n_hf, n_rand = set(), 0, 0
    for name, p in model.named_parameters():
        k = hf_key(name)
        if k is None:
            continue
        a, r = _rows(p)
        s = sh.sl(k)
        assert tuple(s.get_shape()) == tuple(p.shape), (name, k, s.get_shape(), tuple(p.shape))
        if r:
            _loc(p).copy_(s[a:a + r].to(device=dev, dtype=torch.float32))
        used.add(k); n_hf += 1
    want = set(sh.ix) if not partial else {k for k in sh.ix if not k.startswith("model.layers.") or int(k.split(".")[2]) < c32.num_hidden_layers}
    assert used == want, ("Qwen3-32B tensors not loaded / unknown", sorted(want - used)[:4], sorted(used - want)[:4])
    # token-embedding std (global) for the code tables
    w = _loc(model.inp.emb[1].weight).float()
    t = torch.tensor([w.sum().item(), w.pow(2).sum().item(), float(w.numel())], dtype=torch.float64, device=dev)
    dist.all_reduce(t)
    tok_std = float(((t[1] / t[2]) - (t[0] / t[2]) ** 2).sqrt())
    for name, p in model.named_parameters():
        n = name.replace("_checkpoint_wrapped_module.", "")
        if hf_key(n) is not None:
            continue
        kind, std = init_rule(n, tuple(p.shape), c4, c06, tok_std)
        a, _ = _rows(p)
        H2.init_rows_(_loc(p), a, tuple(p.shape), kind, std, n, INIT_SEED)
        n_rand += 1
    # derived rows
    E = [model.inp.emb[0].weight, model.inp.emb[1].weight]
    W = model.head.lm_head.weight
    rows_before = {i: float(get_row(E[1], i, dev).norm()) for i in (PAD, NW, SOA, EOA, SOB, SOSA, SOP)}
    src = {SOA: (IM_START, USER_ID), EOA: (IM_END,), SOB: (IM_START, ASST_ID)}
    mean = {i: torch.stack([get_row(E[1], s_, dev) for s_ in ss]).mean(0) for i, ss in src.items()}
    pad_row = get_row(E[1], PAD, dev)
    for Wc in E:
        for i, v in mean.items():
            set_row(Wc, i, v)
        set_row(Wc, NW, pad_row)
        for i in (SOSA, SOSB, SOP):
            set_row(Wc, i, mean[SOB])
        for i in (EOSA, EOSB, EOP):
            set_row(Wc, i, get_row(Wc, IM_END, dev))
    set_row(model.head.nw_out, 0, get_row(model.head.spec_out, 0, dev))
    set_row(model.head.eob_out, 0, get_row(W, EOB, dev))
    eq = torch.tensor([float(torch.equal(_loc(E[0]), _loc(E[1])))], device=dev)
    dist.all_reduce(eq, op=dist.ReduceOp.MIN)
    assert float(eq) == 1.0, "emb_A != emb_B after init"
    hd = H2.CHUNK
    info = {"hf_tensors": n_hf, "random_tensors": n_rand, "tok_std": tok_std, "emb_row_norms_before": rows_before,
            "emb_typical_row_norm": float(get_row(E[1], 1000, dev).norm()), "ce_chunk": hd}
    log(f"[init] backbone <- {QWEN32}: {n_hf} tensors (emb copied to A and B, bit-equal); {n_rand} random tensors (seed {INIT_SEED}); "
        f"tok_std {tok_std:.5f}; control input rows (base norms {rows_before}, typical row 1000: {info['emb_typical_row_norm']:.3f}) <- SOA/EOA/SOB means, "
        f"NW <- PAD, sos*/sop <- SOB, eos*/eop <- <|im_end|>; nw_out <- spec_out, eob_out <- lm_head[<|im_end|>]")
    return info


# ------------------------------------------------------------------------------------------------ build (FSDP2)
def build(c32, c4, c06, dev, log=print, partial=False, proj_r=2048, tower_vocab=TOWER_VOCAB, init=True, dec_ac=False, dec_mode="replicate",
          pose_head=False, decp_bb=False, vadapter=None):
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    mpn = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
    with torch.device("meta"):
        model = PPM(c32, c4, c06, proj_r, tower_vocab, pose_head=pose_head, decp_bb=decp_bb, vadapter=vadapter)
    assert dec_mode in ("replicate", "fsdp"), dec_mode
    rep = dec_mode == "replicate"                                  # T14: decoders REPLICATED per GPU (default): no FSDP, one grad all-reduce per update
    for dec in (model.dec_s, model.dec_p) if (dec_ac and not rep) else ():   # decoder activation checkpointing off by default (chunks are small)
        lay = dec.model.layers
        for i in range(len(lay)):
            lay[i] = checkpoint_wrapper(lay[i], checkpoint_impl=CheckpointImpl.NO_REENTRANT, preserve_rng_state=False)
    fully_shard(model.inp, mp_policy=mp, reshard_after_forward=True)
    for L in model.layers:
        fully_shard(L, mp_policy=mp, reshard_after_forward=True)
    fully_shard(model.head, mp_policy=mpn, reshard_after_forward=True)
    for tw in (model.tower_s, model.tower_p):
        for L in tw.layers:
            fully_shard(L, mp_policy=mp, reshard_after_forward=True)
        fully_shard(tw, mp_policy=mpn, reshard_after_forward=True)
    if pose_head:                                                  # D32: its own FSDP unit (all-gathered when called on pose rows)
        fully_shard(model.pose_head, mp_policy=mpn, reshard_after_forward=True)
    if decp_bb:                                                    # D33: its own FSDP unit (backbone side; gradient via the main backward)
        fully_shard(model.decp_bb, mp_policy=mpn, reshard_after_forward=True)
    if vadapter:                                                   # VOX EDIT: own FSDP unit, bf16 compute / fp32 masters (= the ASR adapter run)
        fully_shard(model.vadapter, mp_policy=mp, reshard_after_forward=True)
        fully_shard(model.vgain, mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False),
                    reshard_after_forward=True)
    for dec in (model.dec_s, model.dec_p) if not rep else ():
        for L in dec.model.layers:
            fully_shard(L, mp_policy=mp, reshard_after_forward=True)
        fully_shard(dec, mp_policy=mpn, reshard_after_forward=True)
    ign = {p for d in (model.dec_s, model.dec_p) for p in d.parameters()} if rep else None
    fully_shard(model, mp_policy=mpn, reshard_after_forward=True, ignored_params=ign)
    model.to_empty(device=dev)
    for dec in (model.dec_s, model.dec_p):
        dec.model.rotary_emb = type(dec.model.rotary_emb)(config=c06, device=dev)
    for n_, p in model.named_parameters():
        assert p.dtype == torch.float32, ("fp32 masters", n_, p.dtype)
    info = init_weights(model, c32, c4, c06, dev, log, partial) if init else {}
    cnt = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    pc = {"inp": cnt(model.inp), "backbone_layers": cnt(model.layers), "head": cnt(model.head), "tower_s": cnt(model.tower_s),
          "tower_p": cnt(model.tower_p), "dec_s": cnt(model.dec_s), "dec_p": cnt(model.dec_p), **({"pose_head": cnt(model.pose_head)} if pose_head else {}),
          **({"decp_bb": cnt(model.decp_bb)} if decp_bb else {}), **({"vadapter": cnt(model.vadapter)} if vadapter else {}),
          "total": cnt(model)}
    info["param_counts"] = pc
    log("[build] PPM params " + ", ".join(f"{k} {v / 1e9:.3f}B" for k, v in pc.items()) +
        f" | backbone {c32.num_hidden_layers} layers, towers {c4.num_hidden_layers}, decoders {c06.num_hidden_layers}; tower vocab {tower_vocab}")
    return model, info
