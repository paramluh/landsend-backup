"""si20k-tdx INTL MODEL (10-19, user: 14B TEXT DUPLEX, "no audio anywhere"): the 14B next-arch ILP (model_ilp.ILP, the pretraining model;
no VOX, no adapter) with every audio part removed, = the si20k-txt text-out frame with the user as words:
  LAYOUT  row position p -> TWO backbone positions [user_text_p, asst_text_p] (the asst_speech slot dropped, as si20k-txt):
          user_text_p = emb_A(ids_A[p]) (the user's words at their TTS timing; emb_A(PAD) where silent / absent = the pretraining rule),
          asst_text_p = emb_B(ids_B[p]). No audio code embedding anywhere (inp.aud.* not built; depth arrays never read).
  RoPE    the two kept slots keep their 3-slot positions 3p and 3p + 1 through the frequency-divided backbone RoPE (div 3) = si20k-txt.
  OUTPUT  ONLY the text head, next-arch read v2: asst_text_{p+1} from user_text_{p+1} = backbone 2 (sel + 1) (= si20k-txt). No cb0 head,
          no depth decoder (not built).
  Tensor names = the 14B's (w-6242 loads directly; its audio tensors inp.aud.* / cb0_head.* / dec_s.* are skipped, asserted)."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "vendor"), os.path.join(HERE, "lib"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import model_ppm as M  # noqa: E402
import model_ilp as MI  # noqa: E402
import model_ilp_txt as MT  # noqa: E402   (InpTxt, il2_positions: the si20k-txt 2-slot layout)

NS2, USER2 = MT.NS2, MT.USER2
AUDIO_PREFIX = MT.AUDIO_PREFIX                                   # inp.aud. / cb0_head. / dec_s.: not built


class ILPTdx(MI.ILP):
    def __init__(self, c32, c4, c06, **kw):
        super().__init__(c32, c4, c06, **kw)
        assert self.mode == "interleave" and self.read_pos == "v2" and self.cb0_mode, (self.mode, self.read_pos, self.cb0_mode)
        del self.cb0_head, self.dec_s
        self.inp = MT.InpTxt(c32.vocab_size, c32.hidden_size, pose=False)
        del self.inp.aud

    def forward(self, b, rot, rot_t=None, want_logits=False, info=None, keep_h=False):
        assert b["kind"] == "speech" and not want_logits, b["kind"]
        assert b.get("depth") is None and "depth_t" not in b and "ids_t" not in b and "present_t" not in b, "si20k-tdx: no audio codes / probes"
        ids = b["ids"]
        T = ids.shape[1]
        e = self.inp(ids)[None]                                         # [user_text = emb_A(ids_A), asst_text = emb_B(ids_B)] per row position
        s2, pid2 = MT.il2_positions(b["segs"], e.device)
        ctx2 = M.Ctx(s2, e.device)
        assert e.shape[1] == NS2 * T == ctx2.n, (e.shape, T, ctx2.n)
        pe = self.rot_bb(rot)(e, pid2)
        h = e
        for L in self.layers:
            h = L(h, pe, ctx2)
        rd = NS2 * (b["sel"] + 1) + USER2                               # v2: user_text_{p+1} predicts asst_text_{p+1}
        lt, st, hn = self.head(h[0], rd, b["y"], b["wt"], b.get("wn"))
        out = {"text": lt, "text_stats": st}
        if keep_h:
            out.update(h=h[0], e=e[0], hn=hn)
        return out


def build(c32, c4, c06, dev, log=print, mode="interleave", proj_r=2048, tower_vocab=M.TOWER_VOCAB, rope_pos_div=3, read_pos="v2"):
    """= model_ilp_txt.build without the vadapter / vgain units: inp, every backbone layer, head."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    assert mode == "interleave" and read_pos == "v2"
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    mpn = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
    with torch.device("meta"):
        model = ILPTdx(c32, c4, c06, proj_r=proj_r, tower_vocab=tower_vocab, mode=mode, cb0_head=True, rope_pos_div=rope_pos_div, read_pos=read_pos)
    fully_shard(model.inp, mp_policy=mp, reshard_after_forward=True)
    for L in model.layers:
        fully_shard(L, mp_policy=mp, reshard_after_forward=True)
    fully_shard(model.head, mp_policy=mpn, reshard_after_forward=True)
    fully_shard(model, mp_policy=mpn, reshard_after_forward=True)
    model.to_empty(device=dev)
    names = [n.replace("_checkpoint_wrapped_module.", "") for n, _ in model.named_parameters()]
    assert not any(n.startswith(AUDIO_PREFIX + ("tower_s.", "vadapter.", "vgain.")) for n in names), "audio / adapter part built"
    for n_, p in model.named_parameters():
        assert p.dtype == torch.float32, ("fp32 masters", n_, p.dtype)
    cnt = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    pc = {"inp": cnt(model.inp), "backbone_layers": cnt(model.layers), "head": cnt(model.head), "total": cnt(model)}
    log("[build] ILP TEXT DUPLEX (frame [user_text = user words, asst_text], no audio tables / cb0 head / decoder / adapter) params "
        + ", ".join(f"{k} {v / 1e9:.4f}B ({v:,})" for k, v in pc.items()) + f" | RoPE / {model.rope_pos_div:g} at 3p, 3p + 1; read v2")
    return model, {"param_counts": pc}
