"""meta-device parameter census of the PPM model (full width), full and reduced depth; dim-0 shard sizes at world 64 / 96."""
import os, sys
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/code")
import torch
import model_ppm as M
for L, L4, L06 in ((64, 36, 28), (2, 2, 2), (1, 1, 1)):
    with torch.device("meta"):
        m = M.PPM(M.cfg32(L), M.cfg4b(L4), M.cfg06(L06), 2048, M.TOWER_VOCAB, pose_head=False, decp_bb=True)
    top = {}
    for n, p in m.named_parameters():
        k = n.split(".")[0] + ("." + n.split(".")[1] if n.split(".")[0] in ("inp", "head") else "")
        top[k] = top.get(k, 0) + p.numel()
    print(L, L4, L06, {k: round(v / 1e9, 4) for k, v in top.items()}, "total", round(sum(top.values()) / 1e9, 3))
    if L == 64:
        dims = {}
        for n, p in m.named_parameters():
            if n.startswith(("dec_s.", "dec_p.")):
                continue
            dims.setdefault(p.shape[0], []).append(n)
        for R in sorted(dims):
            def info(W):
                c = -(-R // W); nch = -(-R // c); empty = W - nch; last = R - c * (nch - 1)
                return f"chunk {c} nchunks {nch} empty_ranks {empty} last {last}"
            print(f"dim0 {R}: {len(dims[R])} tensors e.g. {dims[R][0]} | W64 {info(64)} | W96 {info(96)} | W128 {info(128)}")
