"""v2init checks (CPU, run as a low-priority step on a training node):
 (1) --ck: the trainer's init_checksum.json (exact int64 sums of the loaded weights' bf16 bits, per tensor) == the same sums computed here
     straight from the consolidated DCP init (e.g. w-10351) -> proves the loaded weights are exactly the checkpoint's.
 (2) --ri: the run's rank_items.json (per-rank step-0 / step-1 train items + eval items) == 22383's, re-derived with aloe-v2-ft's ORIGINAL
     code (data_aloe_v2.make_plan(64) / rank_items) + plan digest == 22383's logged ba2d70729ea437ab."""
import argparse, hashlib, json, sys
import torch
ap = argparse.ArgumentParser(); ap.add_argument("--ck", default=""); ap.add_argument("--dcp", default=""); ap.add_argument("--ri", default="")
a = ap.parse_args()
if a.ri:
    sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-v2-ft/code")
    import data_aloe_v2 as A
    PL = A.make_plan(64, roots=A.SRC_STORES)
    dg = hashlib.sha256(json.dumps(PL["items"]).encode()).hexdigest()[:16]
    R = json.load(open(a.ri))
    bad = 0
    for x in R["ranks"]:
        tr, ev = A.rank_items(PL, x["rank"], 64)
        ref = {"step0": tr[0], "step1": tr[1], "eval": ev}
        got = {"step0": x["step0"], "step1": x["step1"], "eval": x["eval"]}
        ref = json.loads(json.dumps(ref))
        if ref != got:
            bad += 1; print("MISMATCH rank", x["rank"], ref, got)
    print(f"RANKITEMS {a.ri}: plan digest {dg} (22383 logged ba2d70729ea437ab: {dg == 'ba2d70729ea437ab'}); {len(R['ranks'])} ranks, "
          f"{bad} mismatching vs 22383's step-0 / step-1 / eval items -> {'MATCH' if bad == 0 and dg == 'ba2d70729ea437ab' else 'FAIL'}")
if a.ck:
    import pickle
    import torch.distributed.checkpoint as dcp
    C = json.load(open(a.ck)); mod = int(C["mod"]); T = C["tensors"]
    md = pickle.load(open(f"{a.dcp}/.metadata", "rb")).state_dict_metadata
    names = list(T)
    def stats(t):
        b = t.contiguous().view(torch.int16).reshape(t.shape[0] if t.dim() else 1, -1)
        s = ws = 0
        for i in range(0, b.shape[0], 4096):
            rs = b[i:i + 4096].to(torch.int64).sum(1)
            w = (torch.arange(i, i + rs.shape[0], dtype=torch.int64) + 1) % mod
            s += int(rs.sum()); ws += int((rs * w).sum())
        return [s, ws, t.numel()]
    bad, nb, batch, bb = [], 0, [], 0
    def flush(batch):
        sd = {n: torch.empty(tuple(md[n].size), dtype=torch.bfloat16) for n in batch}
        dcp.load(sd, checkpoint_id=a.dcp, no_dist=True)
        for n in batch:
            assert md[n].properties.dtype == torch.bfloat16, (n, md[n].properties.dtype)
            if stats(sd[n]) != T[n]:
                bad.append(n); print("MISMATCH", n, stats(sd[n]), T[n], flush=True)
    for n in names:
        assert n in md, ("tensor not in DCP", n)
        batch.append(n); bb += T[n][2] * 2
        if bb > 2e9:
            flush(batch); nb += bb; batch, bb = [], 0
    if batch:
        flush(batch); nb += bb
    print(f"CHECKSUM {a.ck} vs {a.dcp}: {len(names)} tensors, {nb / 1e9:.1f} GB, {len(bad)} mismatching -> {'EXACT MATCH' if not bad else 'FAIL'}")
