"""MONOFT (10-22): PPM node-local training save dcp-N -> this host's ranks as bf16 shards in NODE-LOCAL /scratch (no /data copy: /data is at its
floor). = aloe-v2pure-17641/tools/ckpt_local_export_slot.py (the converter of every scaling-FT init) with ONLY these changes:
  * --out is a node-local stage dir; the /data floor gate is replaced by a /scratch gate (this host's own bytes + --min-free-gb left);
  * a per-host manifest (--manifest, small JSON on /data): for every rank file written here its size, sha256 and per-tensor dim-0 rows (the
    fetchers use it to compute which source ranks each FT host needs and to verify every copied byte); rank 0's host adds rep.pt / meta.json.
The conversion maths (mmap 'model' section, .to(bfloat16).contiguous().clone(), torch.save, the meta.json fields) is the original's verbatim.
  CKX_HOST_AS=<logical host> python ckpt_node_export.py --src <dcp-N> --out <stage> --manifest <ctl>/manifest-<physical host>.json"""
import argparse
import hashlib
import json
import os
import shutil
import socket
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--min-free-gb", type=float, default=30.0)
a = ap.parse_args()
phys = socket.gethostname().split(".")[0]
# the 8,1,1 run records slinky-7 under slinky-10's LOGICAL slot (nodes16_s7.txt swap) in train_state.json hosts: CKX_HOST_AS=<logical host>
host = os.environ.get("CKX_HOST_AS") or phys


def log(*x):
    print(f"[{time.strftime('%H:%M:%S')} {phys}]", *x, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                return h.hexdigest()
            h.update(b)


assert os.path.isfile(f"{a.src}/COMPLETE"), ("source save not COMPLETE on this node", a.src)
ts = json.load(open(f"{a.src}/train_state.json"))
hosts, step, world = ts["hosts"], int(ts["step"]), int(ts["world"])
mine = [r for r, h in enumerate(hosts) if h == host]
assert mine, (host, "holds no rank of this save")
assert a.out.startswith("/scratch/"), ("node-local stage expected", a.out)
os.makedirs(a.out, exist_ok=True)
free = shutil.disk_usage(a.out).free / 1e9
need = 5.6 + (2.3 if 0 in mine else 0.0)                       # this host's 8 bf16 rank files (5.38 GB) + rep.pt (2.2 GB) on rank 0's host
assert free - need >= a.min_free_gb, (f"/scratch free {free:.0f} GB: writing ~{need:.1f} GB would go below {a.min_free_gb} GB")
t0 = time.time()
nb = 0
man = {"phys": phys, "host_as": host, "step": step, "world": world, "src": a.src, "stage": a.out, "files": {}}
for r in mine:
    sd = torch.load(f"{a.src}/rank{r}.pt", map_location="cpu", mmap=True, weights_only=True)
    out = {n: t.to(torch.bfloat16).contiguous().clone() for n, t in sd["model"].items()}
    nb += sum(t.numel() * 2 for t in out.values())
    rows = {n: int(t.shape[0]) for n, t in out.items()}
    torch.save(out, f"{a.out}/rank{r}.pt.tmp")
    os.replace(f"{a.out}/rank{r}.pt.tmp", f"{a.out}/rank{r}.pt")
    del sd, out
    man["files"][f"rank{r}.pt"] = {"rank": r, "size": os.path.getsize(f"{a.out}/rank{r}.pt"), "sha256": sha256(f"{a.out}/rank{r}.pt"), "rows": rows}
if 0 in mine:
    sr = torch.load(f"{a.src}/rep.pt", map_location="cpu", mmap=True, weights_only=True)
    out = {n: t.to(torch.bfloat16).contiguous().clone() for n, t in sr["model"].items()}
    torch.save(out, f"{a.out}/rep.pt")
    nb += sum(t.numel() * 2 for t in out.values())
    json.dump({"format": "ppm-local-bf16-shards-v1", "src": a.src, "step": step, "world": world, "hosts": hosts, "run_tag": ts["cfg"].get("run_tag"),
               "run_base": ts["cfg"].get("run_base"), "cfg": ts["cfg"], "save_time": ts.get("time"), "made": time.strftime("%Y-%m-%d %H:%M:%S %Z")},
              open(f"{a.out}/meta.json", "w"), indent=1)
    for f in ("rep.pt", "meta.json"):
        man["files"][f] = {"size": os.path.getsize(f"{a.out}/{f}"), "sha256": sha256(f"{a.out}/{f}")}
    man["rep_keys"] = sorted(out)
    del sr, out
man["seconds"] = round(time.time() - t0, 1)
os.makedirs(os.path.dirname(a.manifest), exist_ok=True)
json.dump(man, open(a.manifest + ".tmp", "w"))
os.replace(a.manifest + ".tmp", a.manifest)
log(f"ranks {mine} -> {a.out} ({nb / 1e9:.2f} GB bf16, {time.time() - t0:.0f}s incl. sha256); manifest {a.manifest}")
