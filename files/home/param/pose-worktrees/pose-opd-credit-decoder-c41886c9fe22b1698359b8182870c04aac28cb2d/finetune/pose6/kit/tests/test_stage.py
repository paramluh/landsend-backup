#!/usr/bin/env python3
"""pose6 kit tests (CPU, node /scratch only; nothing on /data except the JSON result):
S1  stage_base on a REAL consolidated DCP (w-15973, a key SUBSET: --keys-prefix, test only) at world 8 -> node-local; then load_init's own
    row loop (copied verbatim from pose6/code/train_aloe_pose.load_init) run with fake FSDP2 params for EVERY target rank of world 8 and of
    world 16 (resharding) -> every tensor bit-identical to an independent dcp.load of the source.
S2  synthetic DCP with the two new keys' shapes (tower_p.s_emb.weight [2048, 2560], inp.g0.weight [5120, 8]), a 13-row bias (R % W0 != 0,
    trailing empty ranks), a 0.6B-decoder key (rep.pt): stage at world 64 as host index 3 of 8 (ranks 24..31 + 0) -> the loop for ranks
    24..31 bit-identical; and at world 8 (all ranks).
S3  preflight comparison logic with patched tables: ref + 2 new keys -> PASS; s_emb 'full' (65536 rows) -> FAIL; extra pose_head -> FAIL;
    missing g0 -> FAIL; non-bf16 -> FAIL.
usage: test_stage.py <scratch test dir> <out.json>"""
import json
import os
import socket
import subprocess
import sys
import time

import torch
import torch.distributed.checkpoint as dcp

K = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
HOST = socket.gethostname().split(".")[0]
T, OUT = sys.argv[1], sys.argv[2]
W15973 = "/data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-15973"
res = {"host": HOST, "time": time.strftime("%F %T")}


class FakeP:
    """an FSDP2 dim-0 shard of a parameter (torch.chunk rows for rank `rank` of `world`), like M._rows / M._loc see it."""
    def __init__(self, shape, world, rank, rep=False):
        self.shape = torch.Size(shape)
        R = shape[0]
        c = -(-R // world)
        self.a = min(rank * c, R) if not rep else 0
        self.r = (min((rank + 1) * c, R) - self.a) if not rep else R
        self.loc = torch.full((self.r, *shape[1:]), float("nan"), dtype=torch.float32)


def load_loop(path, params, rep_keys):
    """= pose6/code/train_aloe_pose.load_init's row loop (resharding part), verbatim apart from M._rows / M._loc -> FakeP fields."""
    meta = json.load(open(f"{path}/meta.json"))
    assert meta["format"] == "ppm-local-bf16-shards-v1" and os.path.isfile(f"{path}/COMPLETE"), path
    W0 = int(meta["world"])
    srcs = {}

    def src(r):
        if r not in srcs:
            srcs[r] = torch.load(f"{path}/rank{r}.pt", map_location="cpu", mmap=True, weights_only=True)
        return srcs[r]
    rep = torch.load(f"{path}/rep.pt", map_location="cpu", mmap=True, weights_only=True)
    for n, p in params.items():
        if n in rep:
            p.loc.copy_(rep[n].to(dtype=p.loc.dtype)); continue
        R = p.shape[0]
        c0 = -(-R // W0)
        a, r = p.a, p.r
        loc = p.loc
        got = 0
        if r:
            for s in range(a // c0, min(W0, -(-(a + r) // c0))):
                lo, hi = s * c0, min((s + 1) * c0, R)
                t = src(s)[n]
                assert t.shape[0] == hi - lo and tuple(t.shape[1:]) == tuple(p.shape[1:]), (n, s, tuple(t.shape), lo, hi)
                x0, x1 = max(lo, a), min(hi, a + r)
                if x1 > x0:
                    loc[x0 - a:x1 - a].copy_(t[x0 - lo:x1 - lo].to(dtype=loc.dtype)); got += x1 - x0
        assert got == r, ("rows not covered", n, got, r)
    return set(rep) | set(src(0))


def check(path, srcd, world, ranks):
    """run the loop for each target rank and compare with the source rows (bf16 -> fp32 copy must be exact)."""
    nbad, nt = 0, 0
    for rk in ranks:
        params = {k: FakeP(tuple(v.shape), world, rk, rep=k.startswith(("dec_s.", "dec_p."))) for k, v in srcd.items()}
        ck = load_loop(path, params, None)
        assert ck == set(srcd), ("checkpoint key set != source", sorted(ck ^ set(srcd))[:6])
        for k, p in params.items():
            want = srcd[k][p.a:p.a + p.r].float() if not k.startswith(("dec_s.", "dec_p.")) else srcd[k].float()
            nt += 1
            nbad += int(not torch.equal(p.loc, want))
    return nbad, nt


def stage(src, out, world, hosts, host, prefix=""):
    cmd = [PY, "-u", f"{K}/tools/stage_base.py", "--src", src, "--out", out, "--world", str(world), "--hosts", hosts, "--host", host]
    if prefix:
        cmd += ["--keys-prefix", prefix]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    assert r.returncode == 0, ("stage_base failed", r.returncode)
    return r.stdout.strip().splitlines()[-1]


# ---- S1: real DCP subset
pre = "layers.0.,tower_p.layers.0.,dec_s.model.layers.0.,tower_p.norm.,norm."
md = dcp.FileSystemReader(W15973).read_metadata().state_dict_metadata
keys = sorted(k for k in md if k.startswith(tuple(pre.split(","))))
srcd = {k: torch.empty(tuple(md[k].size), dtype=md[k].properties.dtype) for k in keys}
dcp.load(srcd, checkpoint_id=W15973, no_dist=True)
o1 = f"{T}/s1-w15973sub-W8"
t0 = time.time()
line = stage(W15973, o1, 8, HOST, HOST, pre)
b8, n8 = check(o1, srcd, 8, range(8))
b16, n16 = check(o1, srcd, 16, range(16))
res["S1"] = {"src": W15973, "keys": len(keys), "GB": round(sum(v.numel() * 2 for v in srcd.values()) / 1e9, 2), "stage_log": line,
             "world8_tensors_checked": n8, "world8_differ": b8, "world16_tensors_checked": n16, "world16_differ": b16,
             "pass": b8 == 0 and b16 == 0, "secs": round(time.time() - t0, 1)}
print("S1", res["S1"], flush=True)
del srcd

# ---- S2: synthetic DCP with the new keys' shapes (node /scratch only)
g = torch.Generator().manual_seed(1027)
syn = {"tower_p.s_emb.weight": torch.randn(2048, 2560, generator=g).to(torch.bfloat16),
       "inp.g0.weight": torch.randn(5120, 8, generator=g).to(torch.bfloat16),
       "layers.0.mlp.fake_bias": torch.randn(13, generator=g).to(torch.bfloat16),
       "dec_s.model.fake.weight": torch.randn(7, 5, generator=g).to(torch.bfloat16)}
sp = f"{T}/s2-synth-dcp"
if not os.path.isfile(f"{sp}/COMPLETE"):
    dcp.save(syn, checkpoint_id=sp, no_dist=True)
    open(f"{sp}/COMPLETE", "w").write("123")
    json.dump({"step": 123, "run": "synthetic-test"}, open(f"{sp}/train_cfg.json", "w"))
fake8 = ",".join(["slinky-x0", "slinky-x1", "slinky-x2", HOST, "slinky-x4", "slinky-x5", "slinky-x6", "slinky-x7"])
o2 = f"{T}/s2-synth-W64-idx3"
stage(sp, o2, 64, fake8, HOST)
b64, n64 = check(o2, syn, 64, range(24, 32))
o3 = f"{T}/s2-synth-W8"
stage(sp, o3, 8, HOST, HOST)
b8s, n8s = check(o3, syn, 8, range(8))
files2 = sorted(os.listdir(o2))
res["S2"] = {"world64_idx3_files": files2, "world64_ranks24_31_checked": n64, "world64_differ": b64, "world8_checked": n8s, "world8_differ": b8s,
             "pass": b64 == 0 and b8s == 0 and "rank0.pt" in files2 and "rank24.pt" in files2 and "rank31.pt" in files2 and "rank8.pt" not in files2}
print("S2", res["S2"], flush=True)

# ---- S3: preflight logic (patched tables)
sys.path.insert(0, f"{K}/tools")
import preflight as PF  # noqa: E402
ref = json.load(open(PF.REF))
base_tab = dict(ref["shapes"])
cases = {}


def run_case(name, tab, dt=None, expect_ok=True):
    PF.table = lambda path: (tab, dt or {k: "torch.bfloat16" for k in tab}, [], {"format": "dcp", "step": "TEST"})
    PF.new_tensors = lambda path, fmt, keys: {k: {"finite": True, "norm": 1.0, "absmax": 1.0, "nonzero_frac": 1.0} for k in keys}
    sys.argv = ["preflight.py", "--base", "/nonexistent-test", "--skip-sidecar"]
    rc = PF.main()
    cases[name] = {"rc": rc, "want_ok": expect_ok, "pass": (rc == 0) == expect_ok}


good = dict(base_tab, **{"tower_p.s_emb.weight": [2048, 2560], "inp.g0.weight": [5120, 8]})
run_case("ref+2new", good, expect_ok=True)
run_case("s_emb full", dict(good, **{"tower_p.s_emb.weight": [65536, 2560]}), expect_ok=False)
run_case("extra pose_head", dict(good, **{"pose_head.weight": [2048, 5120], "pose_head.bias": [2048]}), expect_ok=False)
run_case("missing g0", {k: v for k, v in good.items() if k != "inp.g0.weight"}, expect_ok=False)
run_case("non-bf16", good, dt=dict({k: "torch.bfloat16" for k in good}, **{"inp.g0.weight": "torch.float32"}), expect_ok=False)
res["S3"] = {"cases": cases, "pass": all(c["pass"] for c in cases.values())}
print("S3", res["S3"], flush=True)
res["ALL_PASS"] = all(res[k]["pass"] for k in ("S1", "S2", "S3"))
json.dump(res, open(OUT, "w"), indent=1, default=str)
print("ALL_PASS", res["ALL_PASS"], "->", OUT, flush=True)
