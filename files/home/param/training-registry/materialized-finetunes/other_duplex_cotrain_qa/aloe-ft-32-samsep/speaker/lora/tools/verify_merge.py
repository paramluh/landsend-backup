"""verify a merged LoRA export (merge_lora_export.py output) vs its base + delta, on full tensors of sampled keys:
untouched -> byte-equal to the base; trained (delta) -> byte-equal to the delta; LoRA'd <proj>.weight -> == bf16(fp32(W_base) + (alpha/rank) B @ A)
(model_lora.merged_weight); rep.pt keys: delta's where trained else base's; meta carries the fp32 vgain theta. usage: python -I verify_merge.py <base> <delta> <merged>"""
import sys, json, random, torch
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/code")
import model_lora as ML
base, delta, out = sys.argv[1:4]
mb, md, mo = (json.load(open(f"{p}/meta.json")) for p in (base, delta, out))
W0, W = int(mb["world"]), int(md["world"])
ld = lambda p, r: torch.load(f"{p}/rank{r}.pt", map_location="cpu", mmap=True, weights_only=True)
B0, D0, O0 = ld(base, 0), ld(delta, 0), ld(out, 0)
dkeys = set(D0); lora = {k[:-len(".lora_B")] for k in dkeys if k.endswith(".lora_B")}; trained = {k for k in dkeys if ".lora_" not in k}
assert set(O0) == set(B0), "merged keys != base keys"
assert not any(".lora_" in k for k in O0), "LoRA keys left in the merged export"
untouched = sorted(k for k in B0 if k not in trained and k.rsplit(".", 1)[0] not in lora)
rnd = random.Random(0)
pick = rnd.sample(untouched, 6) + rnd.sample(sorted(trained), min(4, len(trained))) + [p + ".weight" for p in rnd.sample(sorted(lora), 4)]
full = lambda p, n, k: torch.cat([ld(p, r)[k] for r in range(n)], 0)
alpha, rank = float(md["lora"]["alpha"]), int(md["lora"]["rank"])
res = []
for k in pick:
    o = full(out, W0, k); b = full(base, W0, k)
    if k in trained:
        ok = torch.equal(o, full(delta, W, k)); kind = "trained == delta"
    elif k.rsplit(".", 1)[0] in lora:
        p = k.rsplit(".", 1)[0]; A = full(delta, W, p + ".lora_A"); Bm = full(delta, W, p + ".lora_B")
        ref = (b.float() + (alpha / rank) * (Bm.float() @ A.float())).to(torch.bfloat16)
        ok = torch.equal(o, ref); kind = f"lora == base + {alpha / rank:g} BA (|BA|max {float((Bm.float() @ A.float()).abs().max()):.2e})"
    else:
        ok = torch.equal(o, b); kind = "untouched == base"
    res.append((k, kind, ok))
rb, rd, ro = (torch.load(f"{p}/rep.pt", map_location="cpu", mmap=True, weights_only=True) for p in (base, delta, out))
rep_ok = all(torch.equal(ro[k], rd[k] if k in rd else rb[k]) for k in ro) and set(ro) == set(rb)
th_src = md["vgain_theta_fp32"] if "vgain_theta_fp32" in md else mb.get("vgain_theta_fp32")   # lazy default (aloedemo 10-24: base meta may lack it)
th_ok = "vgain_theta_fp32" in mo and th_src is not None and float(mo["vgain_theta_fp32"]) == float(th_src)
for k, kind, ok in res: print(("PASS " if ok else "FAIL ") + kind + "  " + k)
print(f"rep.pt {'PASS' if rep_ok else 'FAIL'} ({len(ro)} keys); meta vgain_theta_fp32 {'PASS' if th_ok else 'FAIL'} ({mo.get('vgain_theta_fp32')})")
print("VERIFY", "PASS" if all(r[2] for r in res) and rep_ok and th_ok else "FAIL")
