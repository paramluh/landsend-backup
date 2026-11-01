"""speaker LoRA FT: DELTA export + its BASE export -> one MERGED full export (CPU, any node with room; 10-24).
  base   ppm-local-bf16-shards-v1 dir (world W0: rank<s>.pt = dim-0 chunk s of every FSDP2 tensor, rep.pt = the replicated decoder, meta.json)
         = the run's init (meta 'base' of the delta), e.g. 29345's export-249.
  delta  ppm-local-bf16-delta-v1 dir (world W: rank<r>.pt = chunk r of every TRAINED tensor + the LoRA lora_A / lora_B, rep.pt = the trained
         replicated decoder, meta.json with lora {rank, alpha, targets}, frozen_groups, vgain_theta_fp32).
  out    ppm-local-bf16-shards-v1 in the BASE layout (world W0, same keys as the base: NO LoRA keys): every trained tensor = the delta's (full
         tensor re-chunked to W0), every LoRA'd <proj>.weight = W_base + (alpha / rank) * lora_B @ lora_A (fp32 math, bf16 out, = model_lora.
         merged_weight = the trainer's 'merged' export rule), every other tensor = the base's; rep.pt = base rep.pt updated with the delta's;
         meta.json = base meta + merge provenance + the delta's fp32 vgain theta. Written to <out>.part, COMPLETE last, renamed.
usage: python -I merge_lora_export.py <base dir> <delta dir> <out dir>"""
import json
import os
import sys
import time

import torch

base, delta, out = sys.argv[1:4]
t0 = time.time()
mb, md = json.load(open(f"{base}/meta.json")), json.load(open(f"{delta}/meta.json"))
assert mb["format"] == "ppm-local-bf16-shards-v1" and md["format"] == "ppm-local-bf16-delta-v1", (mb["format"], md["format"])
assert os.path.isfile(f"{base}/COMPLETE") and os.path.isfile(f"{delta}/COMPLETE")
W0, W = int(mb["world"]), int(md["world"])
lora = md.get("lora") or {}
scale = float(lora["alpha"]) / float(lora["rank"]) if lora else 0.0


def load(d, n):
    return torch.load(f"{d}/rank{n}.pt", map_location="cpu", mmap=True, weights_only=True)


dsh = [load(delta, r) for r in range(W)]
keys = sorted(dsh[0])
assert all(set(d) == set(keys) for d in dsh), "delta ranks hold different keys"
full = {k: torch.cat([dsh[r][k] for r in range(W)], 0) for k in keys}     # delta shards are dim-0 chunks in rank order
lw = {k[:-len(".lora_A")] for k in keys if k.endswith(".lora_A")}
assert lw == {k[:-len(".lora_B")] for k in keys if k.endswith(".lora_B")}
trained = {k for k in keys if not (k.endswith(".lora_A") or k.endswith(".lora_B"))}
part = out + ".part"
os.makedirs(part, exist_ok=True)
seen = set()
for s in range(W0):
    sd = dict(load(base, s))
    for k in list(sd):
        p = k[:-len(".weight")] if k.endswith(".weight") else None
        if k in trained:
            R = int(full[k].shape[0])
        elif p in lw:
            R = int(full[p + ".lora_B"].shape[0])
        else:
            continue
        c0 = -(-R // W0)
        a, b = min(s * c0, R), min((s + 1) * c0, R)
        assert sd[k].shape[0] == b - a, ("base shard rows != chunk(W0) rows", k, s, tuple(sd[k].shape), a, b)
        if k in trained:
            assert tuple(full[k].shape[1:]) == tuple(sd[k].shape[1:]), (k, tuple(full[k].shape), tuple(sd[k].shape))
            sd[k] = full[k][a:b].clone()
        else:
            A, B = full[p + ".lora_A"], full[p + ".lora_B"]
            assert A.shape[1] == sd[k].shape[1] and B.shape[1] == A.shape[0], (k, tuple(A.shape), tuple(B.shape), tuple(sd[k].shape))
            sd[k] = (sd[k].float() + scale * (B[a:b].float() @ A.float())).to(torch.bfloat16)
        seen.add(k)
    torch.save(sd, f"{part}/rank{s}.pt")
    del sd
assert seen == trained | {p + ".weight" for p in lw}, ("delta keys missing from the base", sorted((trained | {p + ".weight" for p in lw}) - seen)[:4])
rb = dict(torch.load(f"{base}/rep.pt", map_location="cpu", weights_only=True))
rd = torch.load(f"{delta}/rep.pt", map_location="cpu", weights_only=True)
assert set(rd) <= set(rb), sorted(set(rd) - set(rb))[:4]
rb.update(rd)
torch.save(rb, f"{part}/rep.pt")
mo = dict(mb, merged_from={"base": base, "delta": delta, "delta_step": md.get("step"), "delta_run": md.get("run_tag"), "lora": lora,
                           "rule": "trained tensors = delta; <proj>.weight = W_base + (alpha / rank) lora_B @ lora_A (fp32 -> bf16); rest = base",
                           "trained_tensors": len(trained), "merged_weights": len(lw), "time": time.strftime("%Y-%m-%dT%H:%M:%S")},
          step=md.get("step"), vgain_theta_fp32=md.get("vgain_theta_fp32", mb.get("vgain_theta_fp32")),
          vgain_s_fp32=md.get("vgain_s_fp32", mb.get("vgain_s_fp32")), vgain_theta_dtype="float32")
json.dump(mo, open(f"{part}/meta.json", "w"), indent=1, default=str)
open(f"{part}/COMPLETE", "w").write(str(md.get("step")))
os.rename(part, out)
print(f"merged {delta} (step {md.get('step')}, world {W}) onto {base} (world {W0}) -> {out}: {len(trained)} trained tensors replaced, {len(lw)} LoRA'd "
      f"weights merged (scale {scale:g}), {time.time() - t0:.0f}s")
