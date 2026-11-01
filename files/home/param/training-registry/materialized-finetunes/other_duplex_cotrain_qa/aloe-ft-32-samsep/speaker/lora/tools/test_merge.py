"""CPU test (10-24) of merge_lora_export.py on a synthetic tiny base (world 4) + delta (world 2) built with the trainer's chunk rule: trained
tensor replaced, LoRA'd weight == W + (alpha / r) B A (bf16 of the fp32 result), untouched frozen tensor == base, rep.pt updated, meta merged.
usage: python -I test_merge.py <tools dir> <work dir>"""
import json
import os
import subprocess
import sys

import torch

TOOLS, WD = sys.argv[1], sys.argv[2]
os.makedirs(WD, exist_ok=True)
g = torch.Generator().manual_seed(0)


def chunks(t, w):
    c = -(-t.shape[0] // w)
    return [t[min(i * c, t.shape[0]):min((i + 1) * c, t.shape[0])].clone() for i in range(w)]


def write(d, w, tensors, rep, meta):
    os.makedirs(d, exist_ok=True)
    for i in range(w):
        torch.save({k: chunks(v, w)[i] for k, v in tensors.items()}, f"{d}/rank{i}.pt")
    torch.save(rep, f"{d}/rep.pt")
    json.dump(meta, open(f"{d}/meta.json", "w"))
    open(f"{d}/COMPLETE", "w").write("1")


q = "layers.0.layer.self_attn.q_proj"
Wq = torch.randn(10, 6, generator=g).to(torch.bfloat16)
base_t = {q + ".weight": Wq, "tower_s.x": torch.randn(7, 3, generator=g).to(torch.bfloat16), "inp.emb.0.weight": torch.randn(5, 6, generator=g).to(torch.bfloat16)}
base_rep = {"dec_s.w": torch.zeros(4, dtype=torch.bfloat16)}
write(f"{WD}/base", 4, base_t, base_rep, {"format": "ppm-local-bf16-shards-v1", "world": 4, "step": 249, "vgain_theta_fp32": -5.0})
A = torch.randn(2, 6, generator=g).to(torch.bfloat16)
B = torch.randn(10, 2, generator=g).to(torch.bfloat16)
tx = torch.randn(7, 3, generator=g).to(torch.bfloat16)
delta_t = {q + ".lora_A": A, q + ".lora_B": B, "tower_s.x": tx}
write(f"{WD}/delta", 2, delta_t, {"dec_s.w": torch.ones(4, dtype=torch.bfloat16)},
      {"format": "ppm-local-bf16-delta-v1", "world": 2, "step": 7, "lora": {"rank": 2, "alpha": 4.0}, "vgain_theta_fp32": -5.6})
out = f"{WD}/merged"
r = subprocess.run([sys.executable, "-I", f"{TOOLS}/merge_lora_export.py", f"{WD}/base", f"{WD}/delta", out], capture_output=True, text=True)
print(r.stdout.strip(), r.stderr.strip()[-500:])
m = {}
for i in range(4):
    for k, v in torch.load(f"{out}/rank{i}.pt", weights_only=True).items():
        m.setdefault(k, []).append(v)
m = {k: torch.cat(v, 0) for k, v in m.items()}
exp_q = (Wq.float() + 2.0 * (B.float() @ A.float())).to(torch.bfloat16)
ok = {"keys_same_as_base": sorted(m) == sorted(base_t), "lora_weight_merged": torch.equal(m[q + ".weight"], exp_q),
      "trained_replaced": torch.equal(m["tower_s.x"], tx), "frozen_untouched": torch.equal(m["inp.emb.0.weight"], base_t["inp.emb.0.weight"]),
      "rep_updated": torch.equal(torch.load(f"{out}/rep.pt", weights_only=True)["dec_s.w"], torch.ones(4, dtype=torch.bfloat16)),
      "meta_theta_from_delta": json.load(open(f"{out}/meta.json"))["vgain_theta_fp32"] == -5.6, "complete": os.path.isfile(f"{out}/COMPLETE")}
print(ok)
print("TEST", "PASS" if all(ok.values()) else "FAIL")
