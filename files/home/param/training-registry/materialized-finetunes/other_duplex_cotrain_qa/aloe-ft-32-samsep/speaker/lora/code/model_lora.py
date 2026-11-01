"""aloe speaker LoRA FT (10-24, user via main: "train the tower and light head mainly, and just add a very low-rank LoRA to the backbone"):
LoRA on the PPM 32B trunk, FSDP2-clean. model_ppm.py is NOT edited (byte-identical copy); this module wraps it.
  build(..., lora=None | {"rank": r, "alpha": a, "targets": [...]})
            rank 0 / None -> model_ppm.build(...) EXACTLY (same call, same model: the identity path).
            rank > 0      -> model_ppm.build's code with ONE insertion before any fully_shard: every targeted nn.Linear of every trunk layer
                             (self_attn.q_proj / k_proj / v_proj / o_proj, mlp.gate_proj / up_proj / down_proj; HF Qwen3 FQNs) becomes a
                             LoRALinear IN PLACE: the base weight keeps its FQN (<...>.q_proj.weight: the init / export keys are unchanged)
                             and two NEW parameters are added to the same module, lora_A [r, in] and lora_B [out, r] (fp32 masters; the
                             layer's FSDP2 unit shards / all-gathers them with the layer, Shard(0)).
  forward   y = x W^T + (alpha / r) (x A^T) B^T  (the layer's bf16 compute dtype, as every other weight under the FSDP2 MixedPrecisionPolicy).
  init      init_lora(model): lora_B = 0 (step 0 == the base model exactly), lora_A = Kaiming-uniform(a = sqrt 5) bound 1 / sqrt(in) drawn
            per tensor from torch.Generator(seed = h64(FQN)) over the FULL [r, in] tensor, each rank copying its own rows (world-size invariant).
  merge     merged_weight(W, A, B, scale) = W + scale x B A (fp32 math, cast to the export dtype); merge_local(model) yields, per LoRA'd
            linear, this rank's merged weight rows (B and W share the Shard(0) row split; A is all-gathered: tiny)."""
import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import model_ppm as M

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
TARGETS = ATTN + MLP


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


class LoRALinear(nn.Linear):
    """nn.Linear + LoRA (module doc). Created by converting an existing nn.Linear in place (lora_inject); bias-free trunk projections."""

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        return y + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scale


def lora_inject(model, rank, alpha, targets=TARGETS):
    """(meta device, before fully_shard) convert the targeted trunk projections; -> list of converted module FQNs."""
    assert rank > 0 and set(targets) <= set(TARGETS), (rank, targets)
    out = []
    for i, L in enumerate(model.layers):
        lay = L.layer
        for nm in targets:
            par = lay.self_attn if nm in ATTN else lay.mlp
            lin = getattr(par, nm)
            assert type(lin) is nn.Linear and lin.bias is None, (i, nm, type(lin))
            dev, dt = lin.weight.device, lin.weight.dtype
            lin.__class__ = LoRALinear
            lin.lora_A = nn.Parameter(torch.empty(rank, lin.in_features, device=dev, dtype=dt))
            lin.lora_B = nn.Parameter(torch.empty(lin.out_features, rank, device=dev, dtype=dt))
            lin.lora_scale = float(alpha) / float(rank)
            lin.lora_rank = int(rank)
            out.append(f"layers.{i}.layer.{'self_attn' if nm in ATTN else 'mlp'}.{nm}")
    return out


def is_lora(name):
    return ".lora_A" in name or ".lora_B" in name


def build(c32, c4, c06, dev, log=print, lora=None, **kw):
    """= model_ppm.build (rank 0 / None: the SAME call); rank > 0: model_ppm.build's body with lora_inject before the first fully_shard."""
    if not lora or int(lora.get("rank", 0)) == 0:
        return M.build(c32, c4, c06, dev, log, **kw)
    partial, proj_r, tower_vocab = kw.get("partial", False), kw.get("proj_r", 2048), kw.get("tower_vocab", M.TOWER_VOCAB)
    init, dec_ac, dec_mode = kw.get("init", True), kw.get("dec_ac", False), kw.get("dec_mode", "replicate")
    pose_head, decp_bb, vadapter, nopose = kw.get("pose_head", False), kw.get("decp_bb", False), kw.get("vadapter"), kw.get("nopose", False)
    assert not init, "LoRA builds load an init (no random init path)"
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    mpn = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
    with torch.device("meta"):
        model = M.PPM(c32, c4, c06, proj_r, tower_vocab, pose_head=pose_head, decp_bb=decp_bb, vadapter=vadapter, nopose=nopose)
        conv = lora_inject(model, int(lora["rank"]), float(lora["alpha"]), tuple(lora.get("targets", TARGETS)))   # <- the ONE insertion
    decs = (model.dec_s,) if nopose else (model.dec_s, model.dec_p)
    towers = (model.tower_s,) if nopose else (model.tower_s, model.tower_p)
    assert dec_mode in ("replicate", "fsdp"), dec_mode
    rep = dec_mode == "replicate"
    for dec in decs if (dec_ac and not rep) else ():
        lay = dec.model.layers
        for i in range(len(lay)):
            lay[i] = checkpoint_wrapper(lay[i], checkpoint_impl=CheckpointImpl.NO_REENTRANT, preserve_rng_state=False)
    fully_shard(model.inp, mp_policy=mp, reshard_after_forward=True)
    for L in model.layers:
        fully_shard(L, mp_policy=mp, reshard_after_forward=True)
    fully_shard(model.head, mp_policy=mpn, reshard_after_forward=True)
    for tw in towers:
        for L in tw.layers:
            fully_shard(L, mp_policy=mp, reshard_after_forward=True)
        fully_shard(tw, mp_policy=mpn, reshard_after_forward=True)
    if pose_head:
        fully_shard(model.pose_head, mp_policy=mpn, reshard_after_forward=True)
    if decp_bb:
        fully_shard(model.decp_bb, mp_policy=mpn, reshard_after_forward=True)
    if vadapter:
        fully_shard(model.vadapter, mp_policy=mp, reshard_after_forward=True)
        fully_shard(model.vgain, mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False),
                    reshard_after_forward=True)
    for dec in decs if not rep else ():
        for L in dec.model.layers:
            fully_shard(L, mp_policy=mp, reshard_after_forward=True)
        fully_shard(dec, mp_policy=mpn, reshard_after_forward=True)
    ign = {p for d in decs for p in d.parameters()} if rep else None
    fully_shard(model, mp_policy=mpn, reshard_after_forward=True, ignored_params=ign)
    model.to_empty(device=dev)
    for dec in decs:
        dec.model.rotary_emb = type(dec.model.rotary_emb)(config=c06, device=dev)
    for n_, p in model.named_parameters():
        assert p.dtype == torch.float32, ("fp32 masters", n_, p.dtype)
    cnt = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    nl = sum(p.numel() for n_, p in model.named_parameters() if is_lora(n_))
    pc = {"inp": cnt(model.inp), "backbone_layers": cnt(model.layers) - nl, "lora": nl, "head": cnt(model.head), "tower_s": cnt(model.tower_s),
          "dec_s": cnt(model.dec_s), **({"vadapter": cnt(model.vadapter)} if vadapter else {}), "total": cnt(model)}
    log("[build] PPM + LoRA params " + ", ".join(f"{k} {v / 1e9:.4f}B" for k, v in pc.items()) +
        f" | LoRA rank {lora['rank']} alpha {lora['alpha']} on {len(conv)} trunk projections ({', '.join(lora.get('targets', TARGETS))})")
    return model, {"param_counts": pc, "lora_modules": conv}


@torch.no_grad()
def init_lora(model):
    """lora_B = 0, lora_A = Kaiming-uniform rows of a per-FQN seeded full tensor (module doc) -> number of LoRA tensors initialised."""
    n = 0
    for name, p in model.named_parameters():
        k = name.replace("_checkpoint_wrapped_module.", "")
        if not is_lora(k):
            continue
        loc = M._loc(p)
        if k.endswith(".lora_B"):
            loc.zero_()
        else:
            r, d_in = p.shape
            g = torch.Generator().manual_seed(h64("aloe-spk-lora-A|" + k) & 0x7FFFFFFFFFFFFFFF)
            bound = 1.0 / math.sqrt(d_in)                           # = nn.init.kaiming_uniform_(a=sqrt(5)) on [r, in]
            full = (torch.rand(r, d_in, generator=g, dtype=torch.float64) * 2.0 - 1.0).mul_(bound).float()
            a, rr = M._rows(p)
            if rr:
                loc.copy_(full[a:a + rr].to(loc.device))
        n += 1
    return n


def merged_weight(W, A, B, scale, dtype=None):
    """W [out, in] (rows of), A [r, in] (full), B [out, r] (the same rows as W) -> W + scale B A in fp32, cast to dtype (default W's)."""
    out = W.float() + float(scale) * (B.float() @ A.float())
    return out.to(dtype or W.dtype)


def lora_modules(model):
    """-> [(FQN of the projection, module)] for every LoRALinear, in named_modules order (identical on every rank)."""
    return [(n.replace("_checkpoint_wrapped_module.", ""), m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]
