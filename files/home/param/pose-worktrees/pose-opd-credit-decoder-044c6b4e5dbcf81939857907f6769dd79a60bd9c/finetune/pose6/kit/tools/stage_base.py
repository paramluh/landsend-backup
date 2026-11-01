#!/usr/bin/env python3
"""pose6 kit (B3, 10-27): stage the FT BASE (a consolidated bf16 DCP export of the 8,1,1 pose run 31101, ckpt-weka/ppm-v1/w-<step>) as a
NODE-LOCAL ppm-local-bf16-shards-v1 copy, the ONLY init format the mono8020 trainer's load_init reads (train_aloe_pose.load_init: meta.json
format + COMPLETE, rank<r>.pt = dim-0 FSDP2 chunks of world W0 concatenated in rank order, rep.pt = the replicated decoders dec_s / dec_p).
mono8020 itself read local-11488 (that format, W0 128) in place from /data; a w-N export is a different format, so it is converted here.
Same precedent as aloe-v2pure-main-17850 (dcp-17850 staged node-local by monoft/tools/init_dist.sh because /data was at its floor).

MATHS: none. bf16 values are copied, never cast (the DCP must be bf16 everywhere, else FAIL), split along dim 0 exactly as torch.chunk /
FSDP2 shards them (c0 = ceil(R / W0) rows per rank, trailing ranks may be empty), and every written file is re-read and compared
bit-exactly (torch.equal) with the source rows before COMPLETE is written; a per-key sha256 of the source is kept in manifest.json.
W0 = this job's world, so each FT rank reads exactly its own rank file (load_init's resharding loop then copies rows 1:1).

One task per node (run script: srun -N<n> --ntasks-per-node=1). Node i (index in PPM_HOSTS, the trainer's own rank rule = task_ppm.sh:
rank = 8 x host index + local id) writes rank<8i..8i+7>.pt + rank0.pt (load_init lists the checkpoint keys from rank0.pt on every rank) +
rep.pt + meta.json, then COMPLETE (= the step). An existing COMPLETE copy of the same source + world is verified (sizes) and reused.
A base that is ALREADY ppm-local-bf16-shards-v1 is not staged (the run script reads it in place, as mono8020 read local-11488).

usage: stage_base.py --src <w-N dir> --out <node-local dir> --world W --hosts h1,h2,.. [--batch-gb 6] [--keys-prefix P,..(TEST ONLY)]"""
import argparse
import hashlib
import json
import math
import os
import socket
import sys
import time

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

REP_PREFIX = ("dec_s.", "dec_p.")          # model_ppm.build(dec_mode="replicate"): the decoders are the ONLY non-FSDP (ignored) params


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} {socket.gethostname().split('.')[0]}] [stage]", *a, flush=True)


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes() if t.numel() else b"").hexdigest()


def read_md(src):
    md = dcp.FileSystemReader(src).read_metadata().state_dict_metadata
    bad = [k for k, v in md.items() if not isinstance(v, TensorStorageMetadata)]
    assert not bad, ("non-tensor entries in the base DCP", bad[:6])
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--hosts", required=True, help="PPM_HOSTS (sorted -V, comma list) = the trainer's rank -> host order")
    ap.add_argument("--host", default=socket.gethostname().split(".")[0])
    ap.add_argument("--batch-gb", type=float, default=6.0)
    ap.add_argument("--keys-prefix", default="", help="TEST ONLY: stage only keys with these prefixes (meta marks it partial)")
    a = ap.parse_args()
    t0 = time.time()
    src, out, W0 = a.src.rstrip("/"), a.out.rstrip("/"), a.world
    hosts = a.hosts.split(",")
    assert len(hosts) * 8 == W0, ("world must be 8 x hosts (one 8-GPU task set per host)", W0, hosts)
    assert a.host in hosts, (a.host, "not in", hosts)
    idx = hosts.index(a.host)
    mine = sorted(set(range(8 * idx, 8 * idx + 8)) | {0})
    assert os.path.isfile(f"{src}/COMPLETE") and os.path.isfile(f"{src}/.metadata"), ("base is not a COMPLETE consolidated DCP", src)
    step = open(f"{src}/COMPLETE").read().strip()
    side = {}
    for f in ("train_cfg.json", "side.json"):
        if os.path.isfile(f"{src}/{f}"):
            side = json.load(open(f"{src}/{f}")); break
    md = read_md(src)
    keys = sorted(md)
    if a.keys_prefix:
        keys = [k for k in keys if k.startswith(tuple(a.keys_prefix.split(",")))]
        assert keys, "no key matches --keys-prefix"
    nondt = [k for k in keys if md[k].properties.dtype != torch.bfloat16]
    assert not nondt, ("base DCP has non-bf16 tensors (a cast would be needed: refused, bf16 only)", [(k, str(md[k].properties.dtype)) for k in nondt[:6]])
    zero_d = [k for k in keys if len(md[k].size) == 0 and not k.startswith(REP_PREFIX)]
    assert not zero_d, ("0-dim FSDP tensors cannot be dim-0 sharded", zero_d[:6])
    ident = {"src": os.path.realpath(src), "step": step, "world": W0, "hosts": hosts, "keys_prefix": a.keys_prefix, "n_keys": len(keys)}
    if os.path.isfile(f"{out}/COMPLETE"):
        old = json.load(open(f"{out}/meta.json"))
        same = all(old.get("stage", {}).get(k) == v for k, v in ident.items())
        assert same, ("node-local stage dir holds a DIFFERENT base / world (never overwritten; pick another --out)", out, old.get("stage"))
        miss = [r for r in mine if not os.path.isfile(f"{out}/rank{r}.pt")]
        assert not miss and os.path.isfile(f"{out}/rep.pt"), ("COMPLETE stage dir lacks files", miss)
        log(f"reuse COMPLETE {out} (step {step}, world {W0}, ranks {mine})")
        return
    os.makedirs(out, exist_ok=True)
    shards = {r: {} for r in mine}
    rep, man = {}, {}
    batch, bbytes, nb = [], 0, 0

    def flush():
        nonlocal batch, bbytes, nb
        if not batch:
            return
        sd = {k: torch.empty(tuple(md[k].size), dtype=torch.bfloat16) for k in batch}
        dcp.load(sd, checkpoint_id=src, no_dist=True)
        for k in batch:
            t = sd[k]
            man[k] = {"shape": list(t.shape), "sha256": sha(t)}
            if k.startswith(REP_PREFIX):
                rep[k] = t
                continue
            R = t.shape[0]
            c0 = -(-R // W0)                                       # torch.chunk / FSDP2 dim-0 rows per rank
            for r in mine:
                lo, hi = min(r * c0, R), min((r + 1) * c0, R)
                shards[r][k] = t[lo:hi].clone()
            nb += t.numel() * 2
        batch, bbytes = [], 0

    for k in keys:
        batch.append(k)
        bbytes += math.prod(md[k].size) * 2
        if bbytes >= a.batch_gb * 2 ** 30:
            flush()
    flush()
    log(f"read {len(keys)} tensors ({nb / 1e9:.1f} GB FSDP + {sum(t.numel() * 2 for t in rep.values()) / 1e9:.2f} GB replicated) from {src} in {time.time() - t0:.0f}s")
    for r in mine:
        torch.save(shards[r], f"{out}/rank{r}.pt.tmp")
        os.replace(f"{out}/rank{r}.pt.tmp", f"{out}/rank{r}.pt")
    torch.save(rep, f"{out}/rep.pt.tmp")
    os.replace(f"{out}/rep.pt.tmp", f"{out}/rep.pt")
    # ---- verify: every written file re-read (mmap) and compared bit-exactly with the source rows
    nbad = 0
    for r in mine:
        back = torch.load(f"{out}/rank{r}.pt", map_location="cpu", mmap=True, weights_only=True)
        assert set(back) == set(shards[r]), ("rank file key set", r)
        nbad += sum(int(not (back[k].dtype == torch.bfloat16 and back[k].shape == shards[r][k].shape and torch.equal(back[k], shards[r][k]))) for k in back)
    back = torch.load(f"{out}/rep.pt", map_location="cpu", mmap=True, weights_only=True)
    assert set(back) == set(rep), "rep.pt key set"
    nbad += sum(int(not torch.equal(back[k], rep[k])) for k in back)
    assert nbad == 0, (nbad, "tensors differ after write: FAIL")
    if len(mine) == W0 or set(mine) >= set(range(W0)):            # this node holds every rank: full reconstruction == source sha
        for k in keys:
            if k.startswith(REP_PREFIX):
                continue
            full = torch.cat([shards[r][k] for r in range(W0)], 0)
            assert sha(full) == man[k]["sha256"], ("reconstruction sha", k)
    json.dump(man, open(f"{out}/manifest.json", "w"))
    json.dump({"format": "ppm-local-bf16-shards-v1", "src": os.path.realpath(src), "step": int(step) if step.isdigit() else step, "world": W0,
               "hosts": [h for h in hosts for _ in range(8)], "run_tag": side.get("run", side.get("run_tag", "ppm-v1 (8,1,1 pose run 31101)")),
               "save_time": time.strftime("%m-%d %H:%M:%S", time.localtime(os.path.getmtime(f"{src}/COMPLETE"))) + " (src COMPLETE mtime, node TZ)",
               "made_by": "pose6/kit/tools/stage_base.py", "stage": ident, "ranks_on_this_node": mine, "partial_TEST": bool(a.keys_prefix)},
              open(f"{out}/meta.json", "w"), indent=1)
    open(f"{out}/COMPLETE", "w").write(step)
    log(f"COMPLETE {out}: step {step}, world {W0}, ranks {mine} + rep.pt, all {sum(len(v) for v in shards.values()) + len(rep)} written tensors "
        f"bit-identical to the source on re-read; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
