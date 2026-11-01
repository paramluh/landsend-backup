"""si20k-txt PPM MODEL (10-19, user: "do not sum the audio back into the backbone or predict audio at all, so it's streaming text in,
streaming text out basically"; confirmed: streaming SPEECH in (user = Voxtral latents), streaming text out) = model_ppm.PPM (nopose, VOX
edit: the si20k-vox model) with EVERY assistant-audio part removed:
  INPUT   e[p] = sum_c present_c[p] * emb_c(ids_c[p])  (+ s x vadapter(Voxtral(A audio)) at A's row frames vpos, unchanged): NO audio code
          sum (inp.aud.0 / inp.aud.1 not built; the micro's depth arrays are never read), i.e. model_ppm.Inp with depth None.
  OUTPUT  ONLY the text head (final RMSNorm -> P_B glu -> lm_head + own PAD / NEW_WORD / EOB rows, factorised CE, PPM's weights) on B's word
          stream: the same targets (words, PAD / NEW_WORD timing, EOB) and weights as si20k-vox. NO speech tower (no cb0 head), NO depth
          decoder (no light frames, no nanosteps, no decoder K): tower_s / dec_s are not built.
  Every remaining tensor keeps its si20k-vox name and init source (PPM local-12829 + ASR adapter w-1972), so the init loader reads them
  unchanged; the init's audio tensors (inp.aud.*, tower_s.*, dec_s.*) are skipped (asserted: only those + the pose parts)."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "vendor"), os.path.join(HERE, "lib"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import model_ppm as M  # noqa: E402

AUDIO_PREFIX = ("inp.aud.", "tower_s.", "dec_s.")               # si20k-vox tensors NOT built in the text-out model


class PPMTxt(M.PPM):
    def __init__(self, c32, c4, c06, proj_r=2048, tower_vocab=M.TOWER_VOCAB, vadapter=None):
        super().__init__(c32, c4, c06, proj_r, tower_vocab, pose_head=False, decp_bb=False, vadapter=vadapter, nopose=True)
        assert vadapter, "si20k-txt: the user enters as Voxtral latents (vadapter required)"
        del self.tower_s, self.dec_s                               # no speech tower / cb0 head, no depth decoder
        del self.inp.aud                                           # no audio code tables: nothing summed into the frame but text (+ A latents)

    def forward(self, b, rot, rot_t=None, want_logits=False, info=None):
        """one si20k / podcast micro -> {'text': weighted text CE, 'text_stats'}: text in (B's word lane, A = PAD + latents), text out."""
        assert b["kind"] == "speech" and not want_logits, b["kind"]
        assert b.get("depth") is None and "depth_t" not in b and "pdepth" not in b, "si20k-txt: no audio codes may enter the model"
        e = self.inp(b["ids"], None, b["present"], None)[None]     # = model_ppm.Inp with depth None: sum_c present_c emb_c(ids_c)
        y = self.vgain(self.vadapter(b["vlat"])[0])                # s x vadapter(Voxtral latents) at A's row frames (= si20k-vox)
        if getattr(self, "vadapter_test_scale", None) is not None:  # step-0 'no user input' eval only (x 0)
            y = y * float(self.vadapter_test_scale)
        assert y.shape[0] == b["vpos"].shape[0], (tuple(y.shape), tuple(b["vpos"].shape))
        e = e + torch.zeros_like(e[0]).index_copy(0, b["vpos"], y.to(e.dtype))[None]
        ctx = M.Ctx(b["segs"], e.device)
        pe = rot(e, b["pid"])
        h = e
        for L in self.layers:
            h = L(h, pe, ctx)
        lt, st, _ = self.head(h[0], b["sel"], b["y"], b["wt"], b.get("wn"))
        return {"text": lt, "text_stats": st}


def build(c32, c4, c06, dev, log=print, proj_r=2048, tower_vocab=M.TOWER_VOCAB, vadapter=None):
    """= model_ppm.build(nopose, vadapter, init False) minus the towers / decoders: the same FSDP2 units and mixed precision for every part
    that remains (inp, each backbone layer, head, vadapter bf16 / fp32 masters, vgain fp32)."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    mpn = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
    with torch.device("meta"):
        model = PPMTxt(c32, c4, c06, proj_r, tower_vocab, vadapter=vadapter)
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
    assert not any(n.startswith(AUDIO_PREFIX + ("tower_p.", "dec_p.", "pose_head.", "decp_bb.", "inp.pose.")) for n in names), "audio / pose part built"
    for n_, p in model.named_parameters():
        assert p.dtype == torch.float32, ("fp32 masters", n_, p.dtype)
    cnt = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    pc = {"inp": cnt(model.inp), "backbone_layers": cnt(model.layers), "head": cnt(model.head), "vadapter": cnt(model.vadapter),
          "vgain": cnt(model.vgain), "total": cnt(model)}
    log("[build] PPM TEXT-OUT (no audio in / out: no inp.aud, no speech tower, no depth decoder) params "
        + ", ".join(f"{k} {v / 1e9:.4f}B" for k, v in pc.items()) + f" | backbone {c32.num_hidden_layers} layers")
    return model, {"param_counts": pc}
