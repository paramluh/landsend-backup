"""next-arch MODEL (10-17, user spec; PPM_CB0_HEAD=1 + PPM_DEC_LAYERS=14): the interleaved PPM below with the 4B speech tower REMOVED.
  cb0     a plain linear LM head on the backbone: cb0_head = Linear(5120 -> 2048, no bias, the tower cb0_head's rule N(0, 1/sqrt(3 x in))),
          input = the ASSISTANT-SPEECH slot's final-normed last hidden state hn_s = hn[3p + 2] (the very states the tower's cond read). Same
          (conditioning position, target) pairs, code delay, mask and weights as the tower: position p reads hn_s[p - 1] (shift1 per packed
          segment, segment starts <- 0) and predicts cb0[p] = cb0 of depth_B[p] (frame p - CD) with cb0_w (= Tower.forward's terms).
  decoder the same Qwen3-0.6B-design light decoder (hidden 1024, 16 / 8 heads, head_dim 128, MLP 3072, RMSNorm, RoPE 1e6, same heads / code
          tables / nanosteps) with PPM_DEC_LAYERS layers (14 = half depth); conditioning slot 0 = proj(hn_s[p - 1]) (proj = the decoder's own
          Linear(in -> 1024, bias) link, in = 5120 now, rule N(0, 1/sqrt(3 x in)), bias 0) + cb0_emb(cb0) + cb1..30 embeddings -> cb1..31 of
          depth_B[p] (= today's decoder interface with the tower state hv[p] replaced by the backbone speech-slot state the cb0 head reads).
          Random init (the 0.6B was never pretrained: every tensor seeded per NAME, so the 14 layers = the 28-layer decoder's layers 0..13).
  groups  cb0_head -> the main-AdamW group 'tower_s' (LR PPM_LR_TOWER 3e-4, clip 1.0, warmup 100 = the old tower's settings).
  Flag off (PPM_CB0_HEAD unset / 0) = the interleave-ppm model exactly (tower + 0.6B decoder on hv_s).
  READ POSITIONS (10-17 user change; read_pos 'v2' = the default with the cb0 head, PPM_READ_POS=v1 = the reads above): plain next-token
          prediction on the unchanged layout [user_text_p 3p, asst_text_p 3p+1, asst_speech_p 3p+2 (= depth_B[p], audio frame p - CD)]:
            text   target asst_text_{p+1} (y = ids_B[sel + 1]) read from user_text_{p+1} = backbone 3 (sel + 1) (was asst_text_p, 3 sel + 1);
                   same head / factorised loss / weights / target set; speech AND text rows; never across a packed segment (sel + 1 is in
                   sel's segment: asserted by the trainer on every micro)
            cb0    target cb0[p] (cb0 of depth_B[p]) and the decoder's cb1..31 of depth_B[p] read from asst_text_p = backbone 3p + 1 (was
                   asst_speech_{p-1}, 3p - 1): cb0 logits = cb0_head(hn[3p + 1]); decoder slot 0 = proj(hn[3p + 1]) + cb0_emb (same frame,
                   no shift, no cross-segment read; same labels / weights / nanosteps)
            the asst_speech slot's hidden has no loss and feeds nothing (context only); the user_text slot's hidden carries the text loss.
  backbone PPM_BACKBONE=qwen3-14b (default qwen3-32b): Qwen3-14B (40 layers, hidden 5120 = the 32B's: cb0 head / decoder link / text head /
          tables keep their shapes), step-0 init by PPM main's own builder model_ppm.init_weights (root = Qwen3-14B: emb copied to A and B,
          the same derived control rows, per-NAME seeded random tensors with the same rules and seed).
  RoPE    PPM_ROPE_POS_DIV=d (1 = off = interleave-ppm's original HF path exactly; next.env sets 3, user 10-17): the BACKBONE's RoPE with
          EVERY frequency divided uniformly by d (rope_theta untouched) = the slots of frame p at positions p, p + 1/3, p + 2/3 instead of
          3p, 3p + 1, 3p + 2 (linear position interpolation): frames N apart get PPM's angle at every frequency. IDENTICAL implementation to
          the live interleaved run's ILP_ROPE_POS_DIV: rope_frame.py copied VERBATIM from interleave-ppm/code (frame_rope: fp32 positions
          (3p + k) / d x inv_freq, fp32 cos / sin, cast to the activation dtype) and called exactly as there. The table is computed once per
          forward from the per-segment positions 0..3 T_s - 1 (il_positions) and passed explicitly to every layer (FA3 varlen + fused
          qk_prep; the activation-checkpoint recompute reuses it). Light decoder RoPE (its own HF rotary) and the old tower's RoPE (rot_t, row
          positions) are untouched. Independent of PPM_CB0_HEAD.

interleave-ppm MODEL (10-17, user spec): PPM main (model_ppm, NOPOSE: 32B backbone + Qwen3-4B speech tower + Qwen3-0.6B depth decoder, the
same parameter names and init) with the backbone input INTERLEAVED instead of summed. Mode 'sum' = PPM's forward exactly (exactness check).

LAYOUT (mode 'interleave', default)   every row position p (= PPM's frame position: system prefix, conversation, CD tail; text rows: PPM's
  two-track frame p) becomes THREE consecutive backbone positions, in this order:
      3p     USER TEXT        e = emb_A(ids_A[p])         PPM channel A's word lane (inp.emb.0). Conditioning only (no loss, as PPM).
                                                          Where A is absent in PPM (present_A 0: prefix positions owned by B, chat system
                                                          text) the lane holds PAD (asserted by the trainer) -> emb_A(PAD), never a zero
                                                          vector (interleave3's finding: an exactly-zero input at a segment's position 0
                                                          stays zero through every layer and RMSNorm's backward multiplies by 1/sqrt(eps)
                                                          per norm -> NaN gradients at >= 16 layers).
      3p + 1 ASSISTANT TEXT   e = emb_B(ids_B[p])         PPM channel B's word lane (inp.emb.1); PAD where B is absent.
      3p + 2 ASSISTANT SPEECH e = aud_B(depth_B[p])       the sum of B's 32 Mimi code embeddings at PPM's code delay (codes of frame p - CD at
                                                          p; inp.aud.1 = PPM's channel-B table). No codes (prefix outside B's voice clip,
                                                          the first CD frames, every TEXT row) -> exactly 0 = PPM's representation of an
                                                          absent audio channel (code_feedback of -1 codes; PPM text rows have no audio term).
                                                          Never at a segment's position 0 (slot 0 is always a token), so no zero-history.
  Channel A's audio (inp.aud.0) is NOT an input (user = word lane only): unused, out of the optimizer, asserted gradient-free + unchanged.
  Plain causal attention per packed segment (segments x 3, FA3 varlen); RoPE positions 0..3T_s - 1 per segment (plain RoPE on the backbone
  position, as PPM's plain RoPE on its position). Visibility: assistant text at p sees user text at p; assistant speech at p sees both
  texts at p; user text at p sees all three slots of p - 1.
HEADS
  text   PPM's head unchanged (final RMSNorm -> P_B glu -> lm_head + own PAD / NEW_WORD / EOB rows, factorised CE) on the ASSISTANT TEXT slots of
         PPM's scored positions: backbone 3 sel + 1 -> y = ids_B[sel + 1] (PPM's targets / weights / normalisers exactly; channel B only).
  speech ONLY the ASSISTANT SPEECH slots condition the speech tower: hn_s[p] = hn[3p + 2]; PPM's Tower unchanged on the T row positions:
         input at p = text_emb(ids_B[p]) + code_emb(depth_B[p - 1]) + cond(hn_s[p - 1]), label cb0 of depth_B[p] (frame p - CD) -> the speech
         slot of p - 1 predicts B's codes of the next frame (the codes input at slot 3p + 2). The 0.6B decoder reads the tower's output exactly
         as PPM (cb1..31, nanosteps in the trainer)."""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "vendor"), os.path.join(HERE, "lib"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import model_ppm as M  # noqa: E402
import rope_frame as RF  # noqa: E402   (VERBATIM copy of interleave-ppm/code/rope_frame.py: the live run's ILP_ROPE_POS_DIV implementation)

MODES = ("interleave", "sum")
SLOTS = ("user_text", "asst_text", "asst_speech")      # slot k of row position p sits at backbone position 3p + k
NS = len(SLOTS)
USER_SLOT, TEXT_SLOT, SPEECH_SLOT = 0, 1, 2
READ_POS = ("v1", "v2")                                 # v1: text <- 3 sel + 1, cb0 / decoder <- 3p - 1 | v2 (next-arch default): text <- 3 (sel + 1), cb0 / decoder <- 3p + 1
UNUSED_IN_INTERLEAVE = ("inp.aud.0.",)                 # channel-A (user) audio table: no user audio in the interleaved layout


class InpIL(M.Inp):
    """M.Inp (same tables and names, no pose table) + the interleaved layout. Called through the module (its FSDP2 unit hooks fire)."""

    def forward(self, ids, depth=None, present=None, pdepth=None, layout="sum", umask=None):
        if layout == "sum":
            return super().forward(ids, depth, present, pdepth)
        assert layout == "interleave" and pdepth is None
        u = self.emb[0](ids[0])                                         # user text (PAD where A is absent)
        if umask is not None:                                           # xdep eval 'absent' only: A contributes nothing (forward only)
            u = u * umask.to(u.dtype)[:, None]
        a = self.emb[1](ids[1])                                         # assistant text
        s = self.aud[1](depth[1]).to(a.dtype) if depth is not None else torch.zeros_like(a)   # assistant speech (0 = no codes / text row)
        return torch.stack([u, a, s], 1).reshape(-1, a.shape[-1])


def il_positions(segs, dev):
    """row segments [T_s] -> (backbone segments [3 T_s], RoPE positions [1, 3N] = 0..3T_s - 1 per segment)."""
    s3 = [NS * int(s) for s in segs]
    pid = torch.cat([torch.arange(s, dtype=torch.int64, device=dev) for s in s3])
    return s3, pid[None]


# ------------------------------------------------------------------------------------------------ next-arch: backbone RoPE frequency division
class RotDiv:
    """the backbone rotary with every frequency / div = interleave-ppm's ILP_ROPE_POS_DIV path verbatim: (cos, sin) = rope_frame.frame_rope(
    rot.r.inv_freq, pid, div, x.dtype, rot.r.attention_scaling) (the exact call in interleave-ppm/code/model_ilp.py). The source Rot is
    neither copied nor modified."""

    def __init__(self, rot, div):
        assert int(div) == div and int(div) >= 1, div
        self.div, self.src, self.r = int(div), rot, rot.r
        assert rot.r.attention_scaling == 1.0 and rot.r.rope_type == "default", (rot.r.attention_scaling, rot.r.rope_type)

    def __call__(self, x, pid):
        return RF.frame_rope(self.src.r.inv_freq, pid, self.div, x.dtype, self.src.r.attention_scaling)


# ------------------------------------------------------------------------------------------------ next-arch: cb0 head on the backbone
_INIT_RULE_PPM, _GROUP_OF_PPM = M.init_rule, M.group_of


def init_rule_next(n, shape, c4, c06, tok_std):
    """model_ppm.init_rule + the next-arch tensors (names that do not exist in the old model; the old model's rules are untouched):
    cb0_head.weight N(0, 1/sqrt(3 x in)) (= tower_s.cb0_head's rule at the backbone width); dec_s.proj.weight with a 5120-wide input
    N(0, 1/sqrt(3 x in)) (= the light proj rule at its input width; the old 2560-input proj keeps model_ppm's rule exactly)."""
    if n == "cb0_head.weight":
        return "normal", 1.0 / (3 * shape[1]) ** 0.5
    if n in ("dec_s.proj.weight", "dec_p.proj.weight") and shape[1] != c4.hidden_size:
        return "normal", 1.0 / (3 * shape[1]) ** 0.5
    return _INIT_RULE_PPM(n, shape, c4, c06, tok_std)


def group_of_next(name):
    """model_ppm.group_of + cb0_head -> 'tower_s' (the old tower's group: same LR / clip / warmup)."""
    n = name.replace("_checkpoint_wrapped_module.", "")
    if n.startswith("cb0_head."):
        return "tower_s"
    return _GROUP_OF_PPM(name)


M.init_rule, M.group_of = init_rule_next, group_of_next     # pure extensions: every old name maps exactly as before


def cb0_head_ce(head, hn_s, ctx, lab, wpos, info=None, shift=True):
    """x[p] = hn_s[p - 1] (shift1 within each packed segment, starts <- 0: the tower cond's positions), logits = head(x) [T, 2048] fp32,
    CE vs lab [T] (-100 none) x wpos -> (x = the decoder's conditioning [T, 5120], sum wpos CE, CE sum (detached), n labels, top-1 hits)."""
    prof = M.TPROF["on"] and torch.is_grad_enabled()
    x = M.shift1(hn_s, ctx, 0.0) if shift else hn_s.contiguous()        # v2 (shift False): position p reads its own row (asst_text_p)
    if info is not None:
        info.update(backbone_rms_hn_s=float(hn_s.float().pow(2).mean().sqrt()))
    if prof:
        ef0 = M._ev()
        rec = {"name": "cb0_head"}
    lo = head(x).float()
    m = lab >= 0
    ce = F.cross_entropy(lo, lab, ignore_index=-100, reduction="none")
    hit = ((lo.argmax(-1) == lab) & m).sum()
    if prof and lo.requires_grad:
        rec.update(f0=ef0, f1=M._ev())
        lo.register_hook(lambda g, rec=rec: rec.__setitem__("b0", M._ev()))         # head backward starts (grad of its logits)
        x.register_hook(lambda g, rec=rec: rec.__setitem__("b1", M._ev()))          # head backward done (grad of its input, + decoder's)
        M.TPROF["ev"].append(rec)
    return x, (ce * wpos).sum(), ce.detach().sum(), m.sum(), hit


class ILP(M.PPM):
    def __init__(self, c32, c4, c06, proj_r=2048, tower_vocab=M.TOWER_VOCAB, mode="interleave", cb0_head=False, rope_pos_div=1, read_pos="v1"):
        super().__init__(c32, c4, c06, proj_r, tower_vocab, pose_head=False, decp_bb=False, nopose=True)
        assert mode in MODES, mode
        self.inp = InpIL(c32.vocab_size, c32.hidden_size, pose=False)
        self.mode = mode
        self.cb0_mode = bool(cb0_head)
        assert int(rope_pos_div) == rope_pos_div and int(rope_pos_div) >= 1, rope_pos_div
        self.rope_pos_div = int(rope_pos_div)                           # next-arch RoPE (doc above): 1 = off (the original HF path)
        assert self.rope_pos_div == 1 or mode == "interleave", "PPM_ROPE_POS_DIV needs the interleaved layout"
        self._rotdiv = None
        assert read_pos in READ_POS and (read_pos == "v1" or (self.cb0_mode and mode == "interleave")), ("read_pos v2 needs the cb0 head", read_pos)
        self.read_pos = read_pos
        if self.cb0_mode:                                               # next-arch: no 4B tower; cb0 head + decoder on the backbone speech slot
            assert mode == "interleave", "PPM_CB0_HEAD needs the interleaved layout"
            del self.tower_s
            self.dec_s = M.Decoder(c32.hidden_size, c06)                # slot 0 = proj(hn_s[p - 1]): input width 5120 (was the tower's 2560)
            self.cb0_head = nn.Linear(c32.hidden_size, M.CBS, bias=False)

    def rot_bb(self, rot):
        """the backbone's rotary: rot itself (rope_pos_div 1 = interleave-ppm exactly) or a cached RotDiv(rot, d)."""
        if self.rope_pos_div == 1:
            return rot
        if self._rotdiv is None or self._rotdiv.src is not rot:
            self._rotdiv = RotDiv(rot, self.rope_pos_div)
        return self._rotdiv

    def forward(self, b, rot, rot_t, want_logits=False, info=None, keep_h=False):
        if self.mode == "sum":
            return M.PPM.forward(self, b, rot, rot_t, want_logits, info)
        kind = b["kind"]
        assert kind in ("speech", "text") and not want_logits, kind
        ids = b["ids_t"] if "ids_t" in b else b["ids"]                  # xdep trunk overrides (A's lane); depth_t (A's codes) has no slot
        um = b["present_t"][0] if "present_t" in b else None
        T = ids.shape[1]
        e = self.inp(ids, b.get("depth"), None, None, layout="interleave", umask=um)[None]
        s3, pid3 = il_positions(b["segs"], e.device)
        ctx3 = M.Ctx(s3, e.device)
        assert e.shape[1] == NS * T == ctx3.n, (e.shape, T, ctx3.n)
        pe = self.rot_bb(rot)(e, pid3)                                  # next-arch: backbone RoPE (frequency division when on)
        h = e
        for L in self.layers:
            h = L(h, pe, ctx3)
        rd = NS * (b["sel"] + 1) + USER_SLOT if self.read_pos == "v2" else NS * b["sel"] + TEXT_SLOT   # v2: user_text_{p+1} predicts asst_text_{p+1}
        lt, st, hn = self.head(h[0], rd, b["y"], b["wt"], b.get("wn"))
        out = {"text": lt, "text_stats": st}
        if keep_h:
            out.update(h=h[0], e=e[0], hn=hn)
        if kind == "speech":
            hn_s = hn.view(T, NS, -1)[:, SPEECH_SLOT]                    # ASSISTANT SPEECH slots only -> the speech tower's conditioning
            ctx = M.Ctx(b["segs"], e.device)
            if self.cb0_mode and self.read_pos == "v2":                 # v2: cb0 head + decoder read asst_text_p (3p + 1) for frame p's codes
                hv, l0, s0, n0, k0 = cb0_head_ce(self.cb0_head, hn.view(T, NS, -1)[:, TEXT_SLOT], ctx, b["cb0"], b["cb0_w"], info, shift=False)
            elif self.cb0_mode:                                         # next-arch v1: cb0 head + decoder read hn_s[p - 1] directly
                hv, l0, s0, n0, k0 = cb0_head_ce(self.cb0_head, hn_s, ctx, b["cb0"], b["cb0_w"], info)
            else:
                hv, l0, s0, n0, k0 = self.tower_s(b["ids"][1], b["depth"][1], hn_s, b["pid"], ctx, rot_t, b["cb0"], b["cb0_w"], info)
            out.update(hv_s=hv, cb0=l0, cb0_sum=s0, cb0_n=n0, cb0_hit=k0)
        return out


def build(c32, c4, c06, dev, log=print, mode="interleave", partial=False, proj_r=2048, tower_vocab=M.TOWER_VOCAB, init=True, dec_mode="replicate",
          cb0_head=False, rope_pos_div=1, backbone_root=M.QWEN32, read_pos="v1"):
    """= model_ppm.build(nopose=True) with the ILP class: identical FSDP2 units / mixed precision / replicated decoder / per-layer activation
    checkpointing (backbone + tower), then PPM's step-0 init (model_ppm.init_weights: Qwen3-32B + per-NAME seeded random tensors -> every
    tensor bit-identical to PPM main's step 0)."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    mpn = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
    with torch.device("meta"):
        model = ILP(c32, c4, c06, proj_r, tower_vocab, mode=mode, cb0_head=cb0_head, rope_pos_div=rope_pos_div, read_pos=read_pos)
    assert dec_mode == "replicate", dec_mode
    fully_shard(model.inp, mp_policy=mp, reshard_after_forward=True)
    for L in model.layers:
        fully_shard(L, mp_policy=mp, reshard_after_forward=True)
    fully_shard(model.head, mp_policy=mpn, reshard_after_forward=True)
    if cb0_head:                                                        # next-arch: the cb0 head is its own FSDP unit (as PPM's D32 pose head)
        fully_shard(model.cb0_head, mp_policy=mpn, reshard_after_forward=True)
    else:
        for L in model.tower_s.layers:
            fully_shard(L, mp_policy=mp, reshard_after_forward=True)
        fully_shard(model.tower_s, mp_policy=mpn, reshard_after_forward=True)
    fully_shard(model, mp_policy=mpn, reshard_after_forward=True, ignored_params={p for p in model.dec_s.parameters()})
    model.to_empty(device=dev)
    model.dec_s.model.rotary_emb = type(model.dec_s.model.rotary_emb)(config=c06, device=dev)
    for n_, p in model.named_parameters():
        assert p.dtype == torch.float32, ("fp32 masters", n_, p.dtype)
    info = M.init_weights(model, c32, c4, c06, dev, log, partial, root=backbone_root) if init else {}
    cnt = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    pc = {"inp": cnt(model.inp), "backbone_layers": cnt(model.layers), "head": cnt(model.head),
          **({"cb0_head": cnt(model.cb0_head)} if cb0_head else {"tower_s": cnt(model.tower_s)}),
          "dec_s": cnt(model.dec_s), "dec_s_layers": cnt(model.dec_s.model.layers), "total": cnt(model)}
    info["param_counts"] = pc
    log(f"[build] ILP mode {mode}{' + CB0 HEAD (no tower)' if cb0_head else ''}: params " + ", ".join(f"{k} {v / 1e9:.4f}B ({v:,})" for k, v in pc.items()) +
        f" | backbone {c32.num_hidden_layers} layers, " + ("cb0 head 5120 -> 2048" if cb0_head else f"tower {c4.num_hidden_layers}") +
        f", decoder {c06.num_hidden_layers} layers (hidden {c06.hidden_size}, heads {c06.num_attention_heads}/{c06.num_key_value_heads}, MLP {c06.intermediate_size}, "
        f"proj in {model.dec_s.proj.in_features}); tower vocab {tower_vocab}; backbone RoPE frequencies / {model.rope_pos_div:g}; read positions {read_pos} ("
        + ("text <- user_text_{p+1} 3(sel+1), cb0 / decoder <- asst_text_p 3p+1" if read_pos == "v2" else "text <- 3sel+1, cb0 / decoder <- 3p-1 / tower") + ")")
    return model, info
