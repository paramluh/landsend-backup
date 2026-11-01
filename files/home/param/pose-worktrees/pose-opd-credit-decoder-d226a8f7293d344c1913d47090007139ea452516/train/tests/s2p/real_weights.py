#!/usr/bin/env python3
"""pose5 s2p tests: REAL 8,1,1 weights for a reduced-depth model -> one node-local cache file (bf16; decoders as saved).

Source: ckpt-infer/ppm-v1/local-17641 = the 8,1,1 main run's step 17641 (job 27204), ckpt_local_export format (128 per-rank bf16 dim-0
FSDP2 shards + rep.pt for the replicated decoders), read-only with mmap (only the needed tensors' pages are read). The reduced model keeps
the first --lb backbone layers, the first --lt layers of each tower, the first --ld layers of each decoder, and every non-layer tensor
(embeddings, code tables, head, towers' text_emb / code_emb / cond / norm / cb0_head, decp_bb, decoders' proj / cb0_emb / embed / norm /
heads). usage: real_weights.py <out.pt> [lb lt ld]"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p")
import s2p_cpu as H  # noqa: E402

sys.path.insert(0, H.ORIG)
import model_ppm as M  # noqa: E402   (the ORIGINAL model: the cache holds exactly its tensors)

SRC = "/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17641"
out = sys.argv[1]
lb, lt, ld = (int(x) for x in (sys.argv[2:5] if len(sys.argv) > 4 else (2, 2, 1)))
meta = json.load(open(f"{SRC}/meta.json"))
assert meta["format"] == "ppm-local-bf16-shards-v1" and meta["step"] == 17641, (meta["format"], meta["step"])
with torch.device("meta"):
    m = M.PPM(M.cfg32(lb), M.cfg4b(lt), M.cfg06(ld), meta["cfg"]["proj_r"], meta["cfg"]["tower_vocab"], pose_head=False, decp_bb=True)
want = {n: tuple(p.shape) for n, p in m.named_parameters()}
pieces = {n: [] for n in want}
t0 = time.time()
for r in range(int(meta["world"])):
    sd = torch.load(f"{SRC}/rank{r}.pt", map_location="cpu", weights_only=True, mmap=True)
    for n in want:
        if n in sd and sd[n].dim() and sd[n].shape[0]:
            pieces[n].append(sd[n])                                 # (mmap-backed; read by the cat below)
    del sd
    if r % 16 == 0:
        H.log(f"rank file {r}/{meta['world']} ({time.time() - t0:.0f}s)")
rep = torch.load(f"{SRC}/rep.pt", map_location="cpu", weights_only=True, mmap=True)
W = {}
for n, shp in want.items():
    if n in rep:
        assert not pieces[n], n
        W[n] = rep[n].clone()
    else:
        W[n] = torch.cat(pieces[n], 0)
    assert tuple(W[n].shape) == shp, (n, tuple(W[n].shape), shp)
    assert torch.isfinite(W[n].float()).all(), n
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save(W, out + ".tmp")
os.replace(out + ".tmp", out)                                      # atomic: readers never see a partial cache
gb = sum(t.numel() * t.element_size() for t in W.values()) / 1e9
H.log(f"saved {len(W)} tensors ({gb:.1f} GB, dtypes {sorted({str(t.dtype) for t in W.values()})}) of step {meta['step']} (lb {lb} lt {lt} ld {ld}) -> {out}")
json.dump({"src": SRC, "step": meta["step"], "run_tag": meta.get("run_tag"), "lb": lb, "lt": lt, "ld": ld, "tensors": len(W), "gb": gb},
          open(out + ".json", "w"), indent=1)
