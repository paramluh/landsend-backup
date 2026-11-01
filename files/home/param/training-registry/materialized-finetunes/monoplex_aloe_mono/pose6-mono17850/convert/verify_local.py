"""pose6 C1 verify (2) on a converted ppm-local-bf16-shards-v1 dir vs a reference local dir (local-17641): per file (rank0..W-1, rep.pt) the
same keys in the same order, same shapes, every tensor bf16, every tensor finite (read back from the written files); meta.json same keys /
order, same world / hosts / format; COMPLETE == step. Read-only. -> json report; exit 1 on any failure."""
import argparse
import json
import os
import sys
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--new", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--json", required=True)
ap.add_argument("--mmap", type=int, default=1, help="0 = plain sequential read of the new files (much faster on Weka than mmap page faults)")
a = ap.parse_args()
t0 = time.time()
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
mn, mr = json.load(open(f"{a.new}/meta.json")), json.load(open(f"{a.ref}/meta.json"))
res = {"new": a.new, "ref": a.ref, "fail": [], "files": 0, "tensors": 0, "elements": 0}
if list(mn) != list(mr):
    res["fail"].append(f"meta keys/order differ: {list(mn)} vs {list(mr)}")
for k in ("format", "world", "hosts", "run_base"):
    if mn.get(k) != mr.get(k):
        res["fail"].append(f"meta {k} differs: {str(mn.get(k))[:80]} vs {str(mr.get(k))[:80]}")
res["meta_differ_keys"] = [k for k in mr if mn.get(k) != mr.get(k)]
res["meta_new"] = {k: mn[k] for k in ("format", "src", "step", "world", "run_tag", "run_base", "save_time", "made")}
c = open(f"{a.new}/COMPLETE").read() if os.path.isfile(f"{a.new}/COMPLETE") else None
res["complete"] = c
if c != str(mn["step"]):
    res["fail"].append(f"COMPLETE {c!r} != step {mn['step']}")
ln, lr = sorted(os.listdir(a.new)), sorted(os.listdir(a.ref))
res["listing_equal"] = ln == lr
if ln != lr:
    res["fail"].append(f"file listing differs: only new {sorted(set(ln) - set(lr))[:6]}, only ref {sorted(set(lr) - set(ln))[:6]}")
W = int(mn["world"])
for fn in [f"rank{r}.pt" for r in range(W)] + ["rep.pt"]:
    n = torch.load(f"{a.new}/{fn}", map_location="cpu", mmap=bool(a.mmap), weights_only=True)
    r = torch.load(f"{a.ref}/{fn}", map_location="cpu", mmap=True, weights_only=True)
    if list(n) != list(r):
        res["fail"].append(f"{fn}: keys/order differ")
    for k, t in n.items():
        u = r.get(k)
        if u is None or tuple(u.shape) != tuple(t.shape) or u.dtype != t.dtype:
            res["fail"].append(f"{fn}:{k} shape/dtype {tuple(t.shape)} {t.dtype} vs ref {None if u is None else (tuple(u.shape), u.dtype)}")
        if t.dtype != torch.bfloat16:
            res["fail"].append(f"{fn}:{k} not bf16 ({t.dtype})")
        if t.numel() and not bool(torch.isfinite(t).all()):
            res["fail"].append(f"{fn}:{k} NON-FINITE")
        res["tensors"] += 1
        res["elements"] += t.numel()
    res["files"] += 1
    del n, r
    if len(res["fail"]) > 50:
        break
res["ok"] = not res["fail"]
res["secs"] = round(time.time() - t0, 1)
json.dump(res, open(a.json, "w"), indent=1)
print(f"[verify_local] {'PASS' if res['ok'] else 'FAIL'}: {res['files']} files, {res['tensors']} tensors, {res['elements']:,} elements bf16+finite;"
      f" listing equal {res['listing_equal']}; meta differs only in {res['meta_differ_keys']}; {res['secs']}s" + ("" if res["ok"] else f"; FAIL {res['fail'][:5]}"), flush=True)
sys.exit(0 if res["ok"] else 1)
