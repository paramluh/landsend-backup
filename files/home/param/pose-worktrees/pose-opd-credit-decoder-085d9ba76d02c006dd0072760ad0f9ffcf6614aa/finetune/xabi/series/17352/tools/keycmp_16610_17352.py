import torch, sys
A="/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-16610"; B="/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17352"
for f in ["rank0.pt", "rank127.pt", "rep.pt"]:
    a = torch.load(f"{A}/{f}", map_location="cpu", mmap=True, weights_only=True)
    b = torch.load(f"{B}/{f}", map_location="cpu", mmap=True, weights_only=True)
    ka, kb = set(a), set(b)
    print(f, len(ka), len(kb), "only16610:", sorted(ka - kb)[:10], "only17352:", sorted(kb - ka)[:10])
    bad = [k for k in ka & kb if a[k].shape != b[k].shape or a[k].dtype != b[k].dtype]
    print("  shape/dtype mismatches:", bad[:10])
    dk = [k for k in sorted(ka & kb) if "decp_bb" in k]
    for k in dk:
        print("  ", k, tuple(b[k].shape), "equal to 16610:", bool(torch.equal(a[k], b[k])), "absmax", float(b[k].float().abs().max()) if b[k].numel() else None)
