"""= evalvox/init_subset_hosts.py's rule VERBATIM (FSDP2 Shard(0) = chunk rows ceil(R/W); + rank0.pt always) for the world-64 export-249 whose
rank files are spread over 8 hosts: the per-tensor row counts come from rows_<host>.json (tools/rows_dump.py) instead of one dir.
usage: python -I subset_w16.py <rows dir> <out json> <ranks per host, e.g. 8,8>"""
import glob, json, sys
D, OUT = sys.argv[1], sys.argv[2]
W0 = 64; NR = [int(x) for x in sys.argv[3].split(",")]; W, NH = sum(NR), len(NR); host_of = [h for h in range(NH) for _ in range(NR[h])]
per = {}
for f in glob.glob(f"{D}/rows_*.json"):
    for s, d in json.load(open(f))["ranks"].items():
        per[int(s)] = d
assert sorted(per) == list(range(W0)), sorted(per)
rows = {}
for s in range(W0):
    for k, n in per[s].items():
        rows.setdefault(k, [0] * W0)[s] = n
need = {h: {0} for h in range(NH)}
for k, rs in rows.items():
    R = sum(rs); c0 = -(-R // W0)
    assert all(rs[s] == max(0, min((s + 1) * c0, R) - s * c0) for s in range(W0)), ("source shards not chunk(W0)", k, rs[:4])
    c = -(-R // W)
    for rank in range(W):
        a, b = rank * c, min((rank + 1) * c, R)
        if b <= a: continue
        for s in range(a // c0, min(W0, -(-b // c0))): need[host_of[rank]].add(s)
res = {"init": "/scratch/amartyavarma/aloe-ft-32-samsep/exports-vox17127/export-249 (8 hosts)", "W0": W0, "W": W, "NR": NR, "n_tensors": len(rows),
       "hosts": {h: sorted(v) for h, v in need.items()}}
json.dump(res, open(OUT, "w"))
print({h: (len(v), sorted(v)[:3], sorted(v)[-3:]) for h, v in need.items()})
