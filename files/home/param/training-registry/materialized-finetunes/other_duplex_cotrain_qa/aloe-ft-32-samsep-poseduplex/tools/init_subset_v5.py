"""poseduplex init staging: which source rank files (world W0 = 128, ppm-local-bf16-shards-v1) each target host needs when the trainer runs at
world 64 (8 hosts x 8 ranks, rank = 8 h + local, FSDP2 Shard(0) = chunk rows ceil(R / W)), exactly as train_samsep_posedup.load_init reads
them (+ rank0.pt, which load_init always opens for the key set, + rep.pt / meta.json / COMPLETE).
v5's final export is not written yet (job 29863 runs), so the per-tensor row counts R come from a /data export of the SAME architecture with
the same pose parts: the 8,1,1 PPM export local-17641 (world 128: backbone, towers, inp tables, decp_bb; decoders in rep.pt) + the Voxtral adapter
/ gain tensors v5 adds (vadapter_s4q.VAdapter(1280, 5120, 4096, 4, norm none) on the meta device; vgain.theta R = 1). The staging script adds a
+-1 source margin on top (stage_export.py) and re-checks against v5's real file list on HF.
usage (compute node): python -I init_subset_v5.py <code dir> <out json>"""
import json
import sys

import torch

CODE, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, f"{CODE}/lib")
SRC = "/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17641"
meta = json.load(open(f"{SRC}/meta.json"))
W0, W, NH, L = int(meta["world"]), 64, 8, 8
assert W0 == 128 and meta["format"] == "ppm-local-bf16-shards-v1"
rows = {}
for s in range(W0):
    sd = torch.load(f"{SRC}/rank{s}.pt", map_location="cpu", mmap=True, weights_only=True)
    for k, t in sd.items():
        rows.setdefault(k, [0] * W0)[s] = int(t.shape[0]) if t.dim() else 1
R_of = {}
for k, rs in rows.items():
    R = sum(rs); c0 = -(-R // W0)
    assert all(rs[s] == max(0, min((s + 1) * c0, R) - s * c0) for s in range(W0)), ("source shards not chunk(W0)", k, rs[:4])
    R_of[k] = R
import vadapter_s4q as VA  # noqa: E402
with torch.device("meta"):
    va = VA.VAdapter(1280, 5120, 4096, 4, norm="none")
added = {f"vadapter.{n}": int(p.shape[0]) if p.dim() else 1 for n, p in va.named_parameters()}
added["vgain.theta"] = 1
for k, R in added.items():
    assert k not in R_of, k
    R_of[k] = R
need = {h: {0} for h in range(NH)}
for k, R in R_of.items():
    c0 = -(-R // W0)
    c = -(-R // W)
    for rank in range(W):
        a, b = rank * c, min((rank + 1) * c, R)
        if b <= a:
            continue
        for s in range(a // c0, min(W0, -(-b // c0))):
            need[rank // L].add(s)
res = {"basis": SRC, "added": added, "W0": W0, "W": W, "n_tensors": len(R_of), "pose_tensors": sum(k.startswith(("tower_p.", "inp.pose", "decp_bb.")) for k in R_of),
       "hosts": {h: sorted(v) for h, v in need.items()}}
json.dump(res, open(OUT, "w"), indent=1)
print({h: (len(v), min(v), max(v)) for h, v in need.items()}, "tensors", len(R_of), "added", added)
