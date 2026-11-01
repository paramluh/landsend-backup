"""aloe-nextvox-6242 MODEL (10-18, user: "finetune this [14B next-arch w-6242] on aloe using voxtral latents instead of the user text stream"):
the 14B next-arch model (model_ilp.ILP, the pretraining's own code, copied VERBATIM and NOT edited) + the PPM Voxtral-on-A edit of the
reference run (aloe-v2vox-raw j23081, its model_ppm VOX EDIT) placed the way the 14B ASR adapter (asr-adapter-14b, train_asr_14b.ASR14)
was trained to enter the 14B:
  vadapter  VAdapter(1280 x 4 -> SwiGLU 4096 -> 5120, norm 'none') = vadapter_s4q.VAdapter (the ASR runs' class); its own FSDP unit, bf16
            compute / fp32 masters (= PPM build's VOX EDIT and asr14b's build).
  vgain     s = exp(theta), ONE learnable scalar, fp32 FSDP unit (= PPM model_ppm.VGain, copied below verbatim).
  INPUT     interleaved frame p -> backbone positions 3p user_text, 3p+1 asst_text, 3p+2 asst_speech (model_ilp). On channel A's row frames
            (the conversation: vpos = P..P+n-1) the USER_TEXT slot = s x vadapter(Voxtral(A audio))_p and NOTHING else: emb_A(ids_A) is
            masked out there (InpIL umask mA = 0, = asr14b's `mA emb_A + vin` with mA = 0 on the clip frames, = PPM's channel A where
            present_A = 0 + vin). Everywhere else (system prefix, CD tail) the user_text slot = emb_A(PAD), exactly the base 14B's rule for
            an absent user (never a zero vector). asst_text / asst_speech = the base 14B's (B's words, B's Mimi codes at CD 2).
  READS     unchanged (model_ilp read_pos v2: text <- user_text_{p+1}, cb0 / decoder <- asst_text_p).
  path      'vox' (default) = the above; 'base' = model_ilp.ILP.forward ITSELF (unmodified function, the latents ignored) for the parity
            check: 'vox' with vscale 0 and mA = 1 (emb_A kept) must be bit-identical to 'base' on the same micro.
  groups    model_ppm.group_of as patched by model_ilp (cb0_head -> tower_s) + vadapter.* -> 'vadapter', vgain.* -> 'vgain' (= PPM VOX EDIT)."""
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "vendor"), os.path.join(HERE, "lib"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import model_ppm as M  # noqa: E402
import model_ilp as MI  # noqa: E402   (patches M.group_of / M.init_rule with the cb0-head extensions on import)
import vadapter_s4q as VA  # noqa: E402   (frozen copy of stage4-asr2/vadapter_s4q.py = the ASR adapter runs' class)

NS, USER = MI.NS, MI.USER_SLOT
assert NS == 3 and USER == 0


class VGain(nn.Module):
    """= PPM model_ppm.VGain (aloe-vox-ft, user / coordinator 10-16) verbatim: the channel-A scale s = exp(theta), one scalar for every vector,
    theta trained (own group 'vgain'), kept in fp32 (own FSDP unit with param_dtype fp32: bf16 theta would quantise s in ~3 % steps)."""

    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(1))

    def forward(self, y):
        return y * torch.exp(self.theta.float()).to(y.dtype)


_GROUP_OF_NEXT = M.group_of                                         # = model_ilp.group_of_next (already patched in by model_ilp)
assert _GROUP_OF_NEXT is MI.group_of_next


def group_of_vox(name):
    """model_ilp.group_of_next + the PPM VOX EDIT's two groups (every other name maps exactly as before)."""
    n = name.replace("_checkpoint_wrapped_module.", "")
    if n.startswith("vadapter."):
        return "vadapter"
    if n.startswith("vgain."):
        return "vgain"
    return _GROUP_OF_NEXT(name)


M.group_of = group_of_vox


class ILPVox(MI.ILP):
    def __init__(self, c32, c4, c06, vadapter, **kw):
        super().__init__(c32, c4, c06, **kw)
        self.vadapter = VA.VAdapter(int(vadapter["enc_hidden"]), c32.hidden_size, int(vadapter["hidden"]), int(vadapter["stack"]), norm=vadapter["norm"])
        self.vgain = VGain()

    def forward(self, b, rot, rot_t, want_logits=False, info=None, keep_h=False, path="vox"):
        if path == "base":                                              # the base 14B's own forward (unmodified function)
            return MI.ILP.forward(self, b, rot, rot_t, want_logits, info, keep_h)
        assert path == "vox" and self.mode == "interleave" and self.cb0_mode and self.read_pos == "v2", (path, self.mode, self.read_pos)
        kind = b["kind"]
        assert kind == "speech" and not want_logits and "ids_t" not in b and "present_t" not in b, kind
        ids = b["ids"]
        T = ids.shape[1]
        vpos = b["vpos"]
        mA = b.get("mA")
        if mA is None:                                                  # user_text = latent ONLY on A's row frames (emb_A masked there)
            mA = torch.ones(T, device=ids.device, dtype=torch.float32).index_fill_(0, vpos, 0.0)
        e = self.inp(ids, b.get("depth"), None, None, layout="interleave", umask=mA)      # [3T, H]: user_text * mA, asst_text, asst_speech
        y = self.vgain(self.vadapter(b["vlat"])[0])                     # A latent = exp(theta) * adapter(Voxtral latents) [n, H]
        if b.get("vscale") is not None:                                 # parity check only: adapter output x 0
            y = y * b["vscale"]
        assert y.shape[0] == vpos.shape[0], (tuple(y.shape), tuple(vpos.shape))
        e = e + torch.zeros_like(e).index_copy(0, NS * vpos + USER, y.to(e.dtype))
        if info is not None and info.get("check_user_slot"):            # data check: the user slots on vpos hold the latent and nothing else
            ue = e.view(T, NS, -1)[:, USER]
            chk = dict(user_slot_latent_max_abs_diff=float((ue.index_select(0, vpos).float() - y.float()).abs().max()),
                       user_slot_latent_rms=float(ue.index_select(0, vpos).float().pow(2).mean().sqrt()),
                       user_slot_mask_zero_frames=int((mA == 0).sum()), user_slot_frames=int(T))
            info.update(chk)
            self.user_slot_check = chk                                  # FSDP2's root forward passes a COPY of the info dict: read it here
        e = e[None]
        # ---- below = model_ilp.ILP.forward verbatim from the backbone on (same RoPE / 3, reads v2, heads)
        s3, pid3 = MI.il_positions(b["segs"], e.device)
        ctx3 = M.Ctx(s3, e.device)
        assert e.shape[1] == NS * T == ctx3.n, (e.shape, T, ctx3.n)
        pe = self.rot_bb(rot)(e, pid3)
        h = e
        for L in self.layers:
            h = L(h, pe, ctx3)
        rd = NS * (b["sel"] + 1) + USER                                 # v2: user_text_{p+1} predicts asst_text_{p+1}
        lt, st, hn = self.head(h[0], rd, b["y"], b["wt"], b.get("wn"))
        out = {"text": lt, "text_stats": st}
        if keep_h:
            out.update(h=h[0], e=e[0], hn=hn)
        ctx = M.Ctx(b["segs"], e.device)
        hv, l0, s0, n0, k0 = MI.cb0_head_ce(self.cb0_head, hn.view(T, NS, -1)[:, MI.TEXT_SLOT], ctx, b["cb0"], b["cb0_w"], info, shift=False)
        out.update(hv_s=hv, cb0=l0, cb0_sum=s0, cb0_n=n0, cb0_hit=k0)
        return out


def build(c32, c4, c06, dev, log=print, mode="interleave", partial=False, proj_r=2048, tower_vocab=M.TOWER_VOCAB, dec_mode="replicate",
          cb0_head=True, rope_pos_div=1, backbone_root=M.QWEN32, read_pos="v1", vadapter=None):
    """= model_ilp.build (init=False) with ILPVox + PPM build's VOX EDIT: vadapter its own FSDP unit (bf16 compute / fp32 masters), vgain its
    own fp32 unit. Same units / policies / replicated decoder otherwise."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    assert vadapter and cb0_head and mode == "interleave"
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    mpn = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
    with torch.device("meta"):
        model = ILPVox(c32, c4, c06, vadapter, proj_r=proj_r, tower_vocab=tower_vocab, mode=mode, cb0_head=cb0_head, rope_pos_div=rope_pos_div,
                       read_pos=read_pos)
    assert dec_mode == "replicate", dec_mode
    fully_shard(model.inp, mp_policy=mp, reshard_after_forward=True)
    for L in model.layers:
        fully_shard(L, mp_policy=mp, reshard_after_forward=True)
    fully_shard(model.head, mp_policy=mpn, reshard_after_forward=True)
    fully_shard(model.cb0_head, mp_policy=mpn, reshard_after_forward=True)
    fully_shard(model.vadapter, mp_policy=mp, reshard_after_forward=True)               # VOX EDIT (= PPM build / asr14b build)
    fully_shard(model.vgain, mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False),
                reshard_after_forward=True)
    fully_shard(model, mp_policy=mpn, reshard_after_forward=True, ignored_params={p for p in model.dec_s.parameters()})
    model.to_empty(device=dev)
    model.dec_s.model.rotary_emb = type(model.dec_s.model.rotary_emb)(config=c06, device=dev)
    for n_, p in model.named_parameters():
        assert p.dtype == torch.float32, ("fp32 masters", n_, p.dtype)
    cnt = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    pc = {"inp": cnt(model.inp), "backbone_layers": cnt(model.layers), "head": cnt(model.head), "cb0_head": cnt(model.cb0_head),
          "dec_s": cnt(model.dec_s), "dec_s_layers": cnt(model.dec_s.model.layers), "vadapter": cnt(model.vadapter), "vgain": cnt(model.vgain),
          "total": cnt(model)}
    info = {"param_counts": pc}
    log(f"[build] ILPVox mode {mode} + CB0 HEAD (no tower) + VOX (vadapter + vgain on the user_text slot): params "
        + ", ".join(f"{k} {v / 1e9:.4f}B ({v:,})" for k, v in pc.items())
        + f" | backbone {c32.num_hidden_layers} layers, cb0 head 5120 -> 2048, decoder {c06.num_hidden_layers} layers (proj in {model.dec_s.proj.in_features}); "
        f"tower vocab {tower_vocab}; backbone RoPE frequencies / {model.rope_pos_div:g}; read positions {read_pos}")
    return model, info
