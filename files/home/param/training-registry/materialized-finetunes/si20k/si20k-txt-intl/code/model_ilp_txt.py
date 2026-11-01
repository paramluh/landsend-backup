"""si20k-txt INTL MODEL (10-19, user: "do not sum the audio back into the backbone or predict audio at all, so it's streaming text in,
streaming text out basically"; confirmed: streaming SPEECH in (user = Voxtral latents), streaming text out) = model_ilp_vox.ILPVox (the
si20k-vox 14B next-arch model) with EVERY assistant-audio part removed:
  LAYOUT  the asst_speech slot is DROPPED from the interleaved frame: row position p -> TWO backbone positions [user_text_p, asst_text_p]
          (was [user_text_p, asst_text_p, asst_speech_p]). user_text_p = s x vadapter(Voxtral(A audio))_p on A's row frames (emb_A masked
          there), emb_A(PAD) elsewhere (= si20k-vox); asst_text_p = emb_B(ids_B[p]) (= si20k-vox). No audio code embedding anywhere
          (inp.aud.0 / inp.aud.1 not built; the micro's depth arrays are never read).
  RoPE    the two kept slots keep EXACTLY their si20k-vox / pretraining positions: frame p's user_text at 3p and asst_text at 3p + 1 of the
          3-slot layout, through the same frequency-divided backbone RoPE (rope_frame.frame_rope, div 3 = next.env) -> angles p and p + 1/3;
          only the asst_speech positions (3p + 2) are removed from the sequence. Causal attention per packed segment over the 2 T_s positions.
  OUTPUT  ONLY the text head with the next-arch v2 read: asst_text_{p+1} (y = ids_B[sel + 1]) predicted from user_text_{p+1} = backbone
          2 (sel + 1) (same head / factorised loss / weights / targets as si20k-vox). NO cb0 head, NO depth decoder (no light frames, no
          nanosteps, no decoder K): cb0_head / dec_s are not built.
  path    'vox' (training) = the above; 'base' = the same 2-slot forward with NO user input (user_text = emb_A(PAD) on every frame, the latents
          ignored) = the step-0 'without user input' eval and the parity reference ('vox' with the adapter output x 0 and emb_A kept must be
          bit-identical to 'base').
  Every remaining tensor keeps its si20k-vox name and init source (w-6242 + ASR adapter w-480); the init's audio tensors (inp.aud.*,
  cb0_head.*, dec_s.*) are skipped (asserted: only those)."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "vendor"), os.path.join(HERE, "lib"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import model_ppm as M  # noqa: E402
import model_ilp as MI  # noqa: E402
import model_ilp_vox as MV  # noqa: E402   (ILPVox + the vadapter / vgain groups; imports model_ilp's group / init extensions)

SLOTS2 = ("user_text", "asst_text")                              # slot k of row position p at backbone position 2p + k
NS2, USER2, TEXT2 = 2, 0, 1
AUDIO_PREFIX = ("inp.aud.", "cb0_head.", "dec_s.")               # si20k-vox tensors NOT built in the text-out model
assert MI.SLOTS[:2] == SLOTS2 and MI.USER_SLOT == USER2 and MI.TEXT_SLOT == TEXT2 and MI.NS == 3


class InpTxt(M.Inp):
    """M.Inp's embeddings (inp.emb.0 / inp.emb.1, same names) without the audio tables: [user_text, asst_text] per row position."""

    def forward(self, ids, umask=None):
        u = self.emb[0](ids[0])                                         # user text lane (PAD everywhere in si20k-txt: the user is Voxtral only)
        if umask is not None:
            u = u * umask.to(u.dtype)[:, None]
        a = self.emb[1](ids[1])                                         # assistant text lane (B's words, PAD / NEW_WORD timing)
        return torch.stack([u, a], 1).reshape(-1, a.shape[-1])


def il2_positions(segs, dev):
    """row segments [T_s] -> (backbone segments [2 T_s], integer RoPE positions [1, 2N] = the 3-slot layout's positions of the two kept
    slots, 3p and 3p + 1, per segment)."""
    s2 = [NS2 * int(s) for s in segs]
    pid = torch.cat([(MI.NS * torch.arange(int(s), dtype=torch.int64, device=dev)[:, None]
                      + torch.arange(NS2, dtype=torch.int64, device=dev)[None]).reshape(-1) for s in segs])
    return s2, pid[None]


class ILPTxt(MV.ILPVox):
    def __init__(self, c32, c4, c06, vadapter, **kw):
        super().__init__(c32, c4, c06, vadapter, **kw)
        assert self.mode == "interleave" and self.read_pos == "v2" and self.cb0_mode, (self.mode, self.read_pos, self.cb0_mode)
        del self.cb0_head, self.dec_s                                   # no cb0 head, no depth decoder
        self.inp = InpTxt(c32.vocab_size, c32.hidden_size, pose=False)
        del self.inp.aud                                                # no audio code tables

    def forward(self, b, rot, rot_t=None, want_logits=False, info=None, keep_h=False, path="vox"):
        assert path in ("vox", "base") and b["kind"] == "speech" and not want_logits, (path, b["kind"])
        assert b.get("depth") is None and "depth_t" not in b and "ids_t" not in b and "present_t" not in b, "si20k-txt: no audio codes / probes"
        ids = b["ids"]
        T = ids.shape[1]
        if path == "vox":
            vpos = b["vpos"]
            mA = b.get("mA")
            if mA is None:                                              # user_text = latent ONLY on A's row frames (emb_A masked there)
                mA = torch.ones(T, device=ids.device, dtype=torch.float32).index_fill_(0, vpos, 0.0)
            e = self.inp(ids, umask=mA)                                 # [2T, H]
            y = self.vgain(self.vadapter(b["vlat"])[0])
            if b.get("vscale") is not None:                             # parity check only: adapter output x 0
                y = y * b["vscale"]
            assert y.shape[0] == vpos.shape[0], (tuple(y.shape), tuple(vpos.shape))
            e = e + torch.zeros_like(e).index_copy(0, NS2 * vpos + USER2, y.to(e.dtype))
            if info is not None and info.get("check_user_slot"):        # data check: the user slots on vpos hold the latent and nothing else
                ue = e.view(T, NS2, -1)[:, USER2]
                chk = dict(user_slot_latent_max_abs_diff=float((ue.index_select(0, vpos).float() - y.float()).abs().max()),
                           user_slot_latent_rms=float(ue.index_select(0, vpos).float().pow(2).mean().sqrt()),
                           user_slot_mask_zero_frames=int((mA == 0).sum()), user_slot_frames=int(T))
                info.update(chk)
                self.user_slot_check = chk
        else:
            e = self.inp(ids)                                           # no user input: user_text = emb_A(PAD) on every frame
        e = e[None]
        s2, pid2 = il2_positions(b["segs"], e.device)
        ctx2 = M.Ctx(s2, e.device)
        assert e.shape[1] == NS2 * T == ctx2.n, (e.shape, T, ctx2.n)
        pe = self.rot_bb(rot)(e, pid2)                                  # frequency-divided backbone RoPE (div 3): angles p, p + 1/3
        h = e
        for L in self.layers:
            h = L(h, pe, ctx2)
        rd = NS2 * (b["sel"] + 1) + USER2                               # v2: user_text_{p+1} predicts asst_text_{p+1}
        lt, st, hn = self.head(h[0], rd, b["y"], b["wt"], b.get("wn"))
        out = {"text": lt, "text_stats": st}
        if keep_h:
            out.update(h=h[0], e=e[0], hn=hn)
        return out


def build(c32, c4, c06, dev, log=print, mode="interleave", proj_r=2048, tower_vocab=M.TOWER_VOCAB, rope_pos_div=3, read_pos="v2", vadapter=None):
    """= model_ilp_vox.build minus the cb0 head / decoder units: inp, every backbone layer, head (as si20k-vox), vadapter (bf16 / fp32
    masters), vgain (fp32)."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    assert vadapter and mode == "interleave" and read_pos == "v2"
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    mpn = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
    with torch.device("meta"):
        model = ILPTxt(c32, c4, c06, vadapter, proj_r=proj_r, tower_vocab=tower_vocab, mode=mode, cb0_head=True, rope_pos_div=rope_pos_div,
                       read_pos=read_pos)
    fully_shard(model.inp, mp_policy=mp, reshard_after_forward=True)
    for L in model.layers:
        fully_shard(L, mp_policy=mp, reshard_after_forward=True)
    fully_shard(model.head, mp_policy=mpn, reshard_after_forward=True)
    fully_shard(model.vadapter, mp_policy=mp, reshard_after_forward=True)
    fully_shard(model.vgain, mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False),
                reshard_after_forward=True)
    fully_shard(model, mp_policy=mpn, reshard_after_forward=True)
    model.to_empty(device=dev)
    names = [n.replace("_checkpoint_wrapped_module.", "") for n, _ in model.named_parameters()]
    assert not any(n.startswith(AUDIO_PREFIX + ("tower_s.", "tower_p.", "dec_p.", "pose_head.", "decp_bb.", "inp.pose.")) for n in names), "audio part built"
    for n_, p in model.named_parameters():
        assert p.dtype == torch.float32, ("fp32 masters", n_, p.dtype)
    cnt = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    pc = {"inp": cnt(model.inp), "backbone_layers": cnt(model.layers), "head": cnt(model.head), "vadapter": cnt(model.vadapter),
          "vgain": cnt(model.vgain), "total": cnt(model)}
    log(f"[build] ILP TEXT-OUT (frame [user_text, asst_text], no asst_speech slot, no audio tables / cb0 head / decoder) params "
        + ", ".join(f"{k} {v / 1e9:.4f}B ({v:,})" for k, v in pc.items())
        + f" | backbone {c32.num_hidden_layers} layers; backbone RoPE frequencies / {model.rope_pos_div:g} at the 3-slot positions 3p, 3p + 1; read v2")
    return model, {"param_counts": pc}
