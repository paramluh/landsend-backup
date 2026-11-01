"""w64 relaunch: which ppm-local-bf16-shards-v1 source rank files (world W0=128) each target host needs when the trainer runs at world 64
(8 hosts x 8 ranks, rank = 8*h + local, FSDP2 Shard(0) = chunk rows ceil(R/W)), exactly as train_samsep.load_init reads them (+ rank0.pt, which
load_init always opens for the key set, + rep.pt / meta.json / COMPLETE). usage: python -I init_subset_hosts.py <init dir> <out json> <ranks per host, e.g. 8,6>"""
import sys, json, torch
P, OUT = sys.argv[1], sys.argv[2]
meta = json.load(open(f"{P}/meta.json")); W0 = int(meta["world"]); NR = [int(x) for x in sys.argv[3].split(",")]; W, NH = sum(NR), len(NR); OFF = [sum(NR[:i]) for i in range(NH)]; host_of = [h for h in range(NH) for _ in range(NR[h])]
rows = {}
for s in range(W0):
    sd = torch.load(f"{P}/rank{s}.pt", map_location="cpu", mmap=True, weights_only=True)
    for k, t in sd.items():
        rows.setdefault(k, [0] * W0)[s] = int(t.shape[0]) if t.dim() else 1
need = {h: {0} for h in range(NH)}
for k, rs in rows.items():
    R = sum(rs); c0 = -(-R // W0)
    assert all(rs[s] == max(0, min((s + 1) * c0, R) - s * c0) for s in range(W0)), ("source shards not chunk(W0)", k, rs[:4])
    c = -(-R // W)
    for rank in range(W):
        a, b = rank * c, min((rank + 1) * c, R)
        if b <= a: continue
        for s in range(a // c0, min(W0, -(-b // c0))): need[host_of[rank]].add(s)
res = {"init": P, "W0": W0, "W": W, "n_tensors": len(rows), "hosts": {h: sorted(v) for h, v in need.items()}}
json.dump(res, open(OUT, "w"))
print({h: (len(v), min(v), max(v)) for h, v in need.items()})
