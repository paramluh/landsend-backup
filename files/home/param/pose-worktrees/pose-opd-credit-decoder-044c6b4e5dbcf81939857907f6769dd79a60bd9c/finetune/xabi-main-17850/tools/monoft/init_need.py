"""MONOFT (10-22): the FT init as a PER-HOST partial copy of the bf16 shard set (ppm-local-bf16-shards-v1), node-local on each FT host.

The aloe trainer's load_init (code/train_aloe_pose.py) has each FT rank k (world 64; rank = 8 x (host index in the sort -V host list) + local
id, task_ppm.sh) read ONLY the source rank files (world 128) holding its FSDP2 dim-0 rows, plus rank0.pt (key set), rep.pt and meta.json /
COMPLETE. So an FT host needs ~16-24 of the 128 rank files (~10-16 GB + rep 2.2 GB) instead of the full 83 GB copy.

  need   --ctl <dir of manifest-<host>.json x 16> --ft-hosts h1,..,h8 --host <FT host> -> JSON list of {file, owner, size, sha256} this host needs
         (exact: torch's Shard(0) chunking for the FT shards x load_init's own source-range loop; + rank0.pt, rep.pt, meta.json)
  check  --init <dir> --ctl <dir> --ft-hosts .. --host .. --step S [--ref <a full local-N set that a real FT already loaded>] [--complete]
         = runs the trainer's REAL load_init source (extracted from train_aloe_pose.py with ast, exec'd unchanged) on CPU for each of this host's
           8 FT ranks, with shims only for the FSDP2 row map (torch's own Shard(0) size / offset for world 64), the local buffers and the
           all-reduce: every row is copied from the files present, coverage / shape / non-finite asserts are the trainer's. With --ref: the key
           set and every global shape == the reference set's (local-17641 was loaded by the 17641 FT, job 27622). --complete writes COMPLETE = S."""
import argparse
import ast
import glob
import json
import os
import re
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["need", "check"])
ap.add_argument("--ctl", required=True)
ap.add_argument("--ft-hosts", required=True)
ap.add_argument("--host", required=True)
ap.add_argument("--world-ft", type=int, default=64)
ap.add_argument("--init", default="")
ap.add_argument("--step", type=int, default=-1)
ap.add_argument("--ref", default="")
ap.add_argument("--trainer", default="/data/home/amartyavarma/ppm/aloe-v2pure-17641/code/train_aloe_pose.py")
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--complete", action="store_true")
a = ap.parse_args()


def vkey(h):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", h)]


def log(*x):
    print(f"[{time.strftime('%H:%M:%S')} {a.host}]", *x, flush=True)


def shard(R, W, k):
    """(offset, size) of FT rank k's dim-0 shard: torch's Shard(0) (DTensor = FSDP2 fully_shard), cross-checked with torch.chunk's rule."""
    c = -(-R // W)
    off, size = min(k * c, R), max(0, min(R, (k + 1) * c) - min(k * c, R))
    from torch.distributed.tensor.placement_types import Shard
    s2, o2 = Shard._local_shard_size_and_offset(R, W, k)
    assert s2 == size and (size == 0 or o2 == off), ("Shard(0) rule mismatch", R, W, k, (s2, o2), (size, off))
    return off, size


# ---- manifests: rank -> (owner host, entry); per-tensor global rows R
mans = [json.load(open(p)) for p in sorted(glob.glob(f"{a.ctl}/manifest-*.json"))]
own = {}
for m in mans:
    for f, e in m["files"].items():
        assert f not in own, ("file in two manifests", f)
        own[f] = (m["phys"], e)
W0 = int(mans[0]["world"]); S0 = int(mans[0]["step"])
assert all(int(m["world"]) == W0 and int(m["step"]) == S0 for m in mans), "manifests disagree on world / step"
miss = [r for r in range(W0) if f"rank{r}.pt" not in own] + [f for f in ("rep.pt", "meta.json") if f not in own]
assert not miss, ("manifests incomplete (a source host not done?)", miss[:8])
rows = {}
for r in range(W0):
    for n, x in own[f"rank{r}.pt"][1]["rows"].items():
        rows.setdefault(n, [0] * W0)[r] = x
R = {n: sum(v) for n, v in rows.items()}
for n, v in rows.items():                                         # the source save is torch.chunk-sharded over W0 (sanity)
    c0 = -(-R[n] // W0)
    assert all(v[s] == max(0, min(R[n], (s + 1) * c0) - min(s * c0, R[n])) for s in range(W0)), ("source rows not chunk-sharded", n)
hosts = sorted(a.ft_hosts.split(","), key=vkey)
assert a.host in hosts and len(hosts) * 8 == a.world_ft, (a.host, hosts, a.world_ft)
I = hosts.index(a.host)
ranks = list(range(8 * I, 8 * I + 8))

need = {0}
for k in ranks:
    for n in R:
        off, r = shard(R[n], a.world_ft, k)
        if r:
            c0 = -(-R[n] // W0)
            need.update(range(off // c0, min(W0, -(-(off + r) // c0))))   # = load_init's source loop
files = [f"rank{s}.pt" for s in sorted(need)] + ["rep.pt", "meta.json"]

if a.mode == "need":
    print(json.dumps([{"file": f, "owner": own[f][0], "size": own[f][1]["size"], "sha256": own[f][1]["sha256"]} for f in files]))
    raise SystemExit(0)

# ---- check: the trainer's load_init, unchanged, on this host's files
assert a.init and os.path.isdir(a.init), a.init
meta = json.load(open(f"{a.init}/meta.json"))
assert meta["format"] == "ppm-local-bf16-shards-v1" and int(meta["world"]) == W0, (meta["format"], meta["world"])
assert a.step < 0 or int(meta["step"]) == a.step == S0, ("init step", meta["step"], a.step, S0)
have = sorted(os.listdir(a.init))
log(f"FT host {I} of {hosts} (ranks {ranks[0]}-{ranks[-1]}): {len(files)} files needed, {len(have)} present; meta step {meta['step']} world {W0}")
rep = torch.load(f"{a.init}/rep.pt", map_location="cpu", mmap=True, weights_only=True)
r0 = torch.load(f"{a.init}/rank0.pt", map_location="cpu", mmap=True, weights_only=True)
assert set(r0) == set(R) and not (set(rep) & set(R)), "rank0 key set != manifests' / overlaps rep"
gshape = {n: (R[n],) + tuple(r0[n].shape[1:]) for n in R}
assert all(t.dtype == torch.bfloat16 for t in list(r0.values()) + list(rep.values())), "bf16 only"
if a.ref:
    rr = {}
    for f in glob.glob(f"{a.ref}/rank*.pt"):
        for n, t in torch.load(f, map_location="cpu", mmap=True, weights_only=True).items():
            rr.setdefault(n, [0, tuple(t.shape[1:])])[0] += int(t.shape[0])
    ref_g = {n: (v[0],) + v[1] for n, v in rr.items()}
    ref_rep = {n: tuple(t.shape) for n, t in torch.load(f"{a.ref}/rep.pt", map_location="cpu", mmap=True, weights_only=True).items()}
    d1 = sorted(set(ref_g) ^ set(gshape)) + [n for n in ref_g if n in gshape and ref_g[n] != gshape[n]]
    d2 = sorted(set(ref_rep) ^ set(rep)) + [n for n in ref_rep if n in rep and ref_rep[n] != tuple(rep[n].shape)]
    assert not d1 and not d2, ("key set / global shapes differ from the reference set", d1[:6], d2[:6])
    log(f"== reference {a.ref}: {len(gshape)} sharded + {len(rep)} replicated tensors, key set and every global shape identical")
del r0

src = open(a.trainer).read()
fn = [x for x in ast.parse(src).body if isinstance(x, ast.FunctionDef) and x.name == "load_init"]
assert len(fn) == 1, "load_init not found in the trainer"
code = ast.get_source_segment(src, fn[0])
dt = getattr(torch, a.dtype)


class P:                                                          # an FSDP2 DTensor param as load_init sees it: global .shape, local rows
    def __init__(self, n, k):
        self.shape = torch.Size(gshape[n])
        self.off, r = shard(gshape[n][0], a.world_ft, k)
        self.loc = torch.empty((r,) + tuple(gshape[n][1:]), dtype=dt)


class Mshim:
    @staticmethod
    def _rows(p):
        return p.off, p.loc.shape[0]

    @staticmethod
    def _loc(p):
        return p.loc if isinstance(p, P) else p


class Dshim:
    @staticmethod
    def all_reduce(t, *x, **y):
        return None


reps = {n: torch.empty(tuple(t.shape), dtype=dt) for n, t in rep.items()}
del rep
names = [n for n in list(R) + list(reps) if not n.startswith("pose_head.")]   # the FT model builds no pose head (QQA_POSE_HEAD_BUILD=0)
tot = 0.0
# load_init asserts COMPLETE exists: write it (= S) for the check; removed again if any rank fails (or if --complete is not given)
open(f"{a.init}/COMPLETE", "w").write(str(meta["step"]))
try:
    for k in ranks:
        t0 = time.time()
        ns = {"json": json, "os": os, "torch": torch, "M": Mshim, "dist": Dshim, "CFG": {"pose_head_build": 0}}
        exec(code, ns)

        class Model:
            def named_parameters(self, k=k):
                for n in names:
                    yield (n, reps[n]) if n in reps else (n, P(n, k))
        with torch.no_grad():
            m_, gb = ns["load_init"](Model(), a.init, "cpu")
        tot += gb
        log(f"FT rank {k}: load_init OK ({gb:.2f} GB of shard rows copied, unloaded keys {m_['unloaded_keys']}, all finite) {time.time() - t0:.0f}s")
        import gc
        gc.collect()
except BaseException:
    os.remove(f"{a.init}/COMPLETE")
    raise
log(f"== CHECK PASS: the trainer's load_init accepts {a.init} for FT ranks {ranks[0]}-{ranks[-1]} ({tot:.1f} GB shard rows)")
if a.complete:
    log(f"COMPLETE {a.init} = {meta['step']}")
else:
    os.remove(f"{a.init}/COMPLETE")
