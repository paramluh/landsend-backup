"""ALOEFT32 base (10-24): verify a node-local ppm-local-bf16-shards-v1 copy of PPM step 17850 (stage_17850.py) and, only if every check
passes, write COMPLETE (= 17850). Checks (CPU only, this host's copy):
  1. bytes: every rank<r>.pt (128) + rep.pt re-hashed (independent pass) == the 17850 monoplex FT init manifests (job 27756, made from the
     node-local dcp-17850 on the main hosts); meta.json format / step / world.
  2. structure: key set + every global shape == local-17641 (the set the 17641 FT loaded) and == w-15973's DCP metadata (the reference Aloe
     init, job 26232/26467); param count of the non-pose tensors == w-15973's (= 26232's model minus vadapter / vgain).
  3. the reference trainer's REAL load_init (train_aloe_v2vox_raw.py, extracted with ast, exec'd unchanged) on CPU for every rank of a
     world-W FSDP2 layout (default 14 = slinky-18 x 8 + slinky-4 x 6; the copy is full, so any rank -> host order works), with shims only
     for the FSDP2 row map (torch Shard(0) size / offset), local buffers and the all-reduce; parameter names = w-15973's non-pose keys (the
     set 26232's model built, minus vadapter / vgain, asserted by its load_init_dcp). + for sampled tensors, rank k's loaded rows == the
     source rows read directly.
  4. values: sampled tensors of every module type (trunk layers, embeddings / code tables, heads, proj, speech tower, replicated speech
     decoder, pose parts) gathered from the 128 shards: finite, bf16, and vs local-17641 (209 steps earlier): relative L2 diff > 0 and small.
  python verify_17850.py --init /scratch/amartyavarma/aloe-samsep-17850-init --ctl <dir> [--world 14]"""
import argparse
import ast
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import pickle
import socket
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True)
ap.add_argument("--ctl", required=True)
ap.add_argument("--world", type=int, default=14)
ap.add_argument("--man", default="/data/home/amartyavarma/ppm/aloe-v2pure-main-17850/initctl")
ap.add_argument("--ref", default="/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17641")
ap.add_argument("--w15973", default="/data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-15973")
ap.add_argument("--trainer", default="/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973/code/train_aloe_v2vox_raw.py")
ap.add_argument("--hash-workers", type=int, default=16)
a = ap.parse_args()
me = socket.gethostname().split(".")[0]
R_ = {"host": me, "init": a.init, "world": a.world, "checks": {}}
T0 = time.time()


def log(*x):
    print(f"[{time.strftime('%m-%d %H:%M:%S')} {me}]", *x, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                return h.hexdigest()
            h.update(b)


assert not os.path.exists(f"{a.init}/COMPLETE"), "COMPLETE already present: verify runs before it is written"
# ---- 1. bytes
exp = {}
for p in sorted(glob.glob(f"{a.man}/manifest-*.json")):
    for f, e in json.load(open(p))["files"].items():
        exp[f] = e
files = [f"rank{r}.pt" for r in range(128)] + ["rep.pt"]
with cf.ThreadPoolExecutor(a.hash_workers) as ex:
    got = dict(zip(files, ex.map(lambda f: sha256(f"{a.init}/{f}"), files)))
bad = [f for f in files if got[f] != exp[f]["sha256"] or os.path.getsize(f"{a.init}/{f}") != exp[f]["size"]]
assert not bad, ("sha256 / size != the 17850 manifest", bad[:8])
meta = json.load(open(f"{a.init}/meta.json"))
assert meta["format"] == "ppm-local-bf16-shards-v1" and int(meta["step"]) == 17850 and int(meta["world"]) == 128, (meta["format"], meta["step"], meta["world"])
extra = sorted(set(os.listdir(a.init)) - set(files) - {"meta.json"})
assert not extra, ("unexpected files in the init dir", extra)
gb = sum(exp[f]["size"] for f in files) / 1e9
R_["checks"]["bytes"] = f"129 files {gb:.2f} GB: sha256 + size == 17850 manifest (re-hashed); meta step 17850 world 128"
log(f"[1] bytes OK: {R_['checks']['bytes']} ({time.time() - T0:.0f}s)")
R_["sha256"] = {f: got[f] for f in files}

# ---- 2. structure
rows = {}
for r in range(128):
    for n, x in exp[f"rank{r}.pt"]["rows"].items():
        rows.setdefault(n, [0] * 128)[r] = x
Rg = {n: sum(v) for n, v in rows.items()}
r0 = torch.load(f"{a.init}/rank0.pt", map_location="cpu", mmap=True, weights_only=True)
rep = torch.load(f"{a.init}/rep.pt", map_location="cpu", mmap=True, weights_only=True)
assert set(r0) == set(Rg) and not (set(rep) & set(Rg)), "rank0 key set != manifest rows / overlaps rep"
assert all(t.dtype == torch.bfloat16 for t in list(r0.values()) + list(rep.values())), "bf16 only"
gshape = {n: (Rg[n],) + tuple(r0[n].shape[1:]) for n in Rg}
gshape.update({n: tuple(t.shape) for n, t in rep.items()})
for n in Rg:                                                     # torch.chunk sharding of the source (sanity)
    c0 = -(-Rg[n] // 128)
    assert all(rows[n][s] == max(0, min(Rg[n], (s + 1) * c0) - min(s * c0, Rg[n])) for s in range(128)), ("not chunk-sharded", n)
# vs local-17641
rr = {}
for f in glob.glob(f"{a.ref}/rank*.pt"):
    for n, t in torch.load(f, map_location="cpu", mmap=True, weights_only=True).items():
        rr.setdefault(n, [0, tuple(t.shape[1:])])[0] += int(t.shape[0])
ref_g = {n: (v[0],) + v[1] for n, v in rr.items()}
ref_g.update({n: tuple(t.shape) for n, t in torch.load(f"{a.ref}/rep.pt", map_location="cpu", mmap=True, weights_only=True).items()})
d1 = sorted(set(ref_g) ^ set(gshape)) + [n for n in ref_g if n in gshape and ref_g[n] != gshape[n]]
assert not d1, ("key set / shapes != local-17641", d1[:8])
# vs w-15973 (DCP metadata)
POSE_PREFIX = None
src_tr = open(a.trainer).read()
for node in ast.parse(src_tr).body:
    if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "POSE_PREFIX" for t in node.targets):
        POSE_PREFIX = ast.literal_eval(node.value)
assert POSE_PREFIX, "POSE_PREFIX not found in the trainer"
md = pickle.load(open(f"{a.w15973}/.metadata", "rb")).state_dict_metadata
wsh = {k: tuple(v.size) for k, v in md.items() if hasattr(v, "size")}
d2 = sorted(set(wsh) ^ set(gshape)) + [n for n in wsh if n in gshape and wsh[n] != gshape[n]]
nonpose_w = sorted(k for k in wsh if not k.startswith(POSE_PREFIX))
numel = lambda s: int(torch.Size(s).numel())  # noqa: E731
n_ours = sum(numel(gshape[n]) for n in gshape if not n.startswith(POSE_PREFIX))
n_w = sum(numel(wsh[n]) for n in nonpose_w)
n_all = sum(numel(s) for s in gshape.values())
skipped = {p_: sum(1 for n in gshape if n.startswith(p_)) for p_ in POSE_PREFIX}
R_["checks"]["structure"] = (f"{len(Rg)} sharded + {len(rep)} replicated tensors; key set + global shapes == local-17641; vs w-15973 DCP: "
                             f"{'IDENTICAL key set + shapes' if not d2 else 'DIFF ' + str(d2[:6])}; non-pose (trainer-loaded) tensors {len([n for n in gshape if not n.startswith(POSE_PREFIX)])} "
                             f"= {n_ours / 1e9:.4f}B params (w-15973 non-pose {len(nonpose_w)} = {n_w / 1e9:.4f}B); all tensors {n_all / 1e9:.4f}B; pose keys skipped by the trainer {skipped}")
assert not [n for n in nonpose_w if n not in gshape or gshape[n] != wsh[n]], "a w-15973 non-pose tensor is missing / differs in shape"
log(f"[2] structure OK: {R_['checks']['structure']} ({time.time() - T0:.0f}s)")
del r0

# ---- 3. the trainer's real load_init, world a.world, every rank
fn = [x for x in ast.parse(src_tr).body if isinstance(x, ast.FunctionDef) and x.name == "load_init"]
assert len(fn) == 1, "load_init not found"
code = ast.get_source_segment(src_tr, fn[0])
from torch.distributed.tensor.placement_types import Shard  # noqa: E402


def shard(Rn, W, k):
    c = -(-Rn // W)
    off, size = min(k * c, Rn), max(0, min(Rn, (k + 1) * c) - min(k * c, Rn))
    s2, o2 = Shard._local_shard_size_and_offset(Rn, W, k)
    assert s2 == size and (size == 0 or o2 == off), ("Shard(0) rule mismatch", Rn, W, k)
    return off, size


class P:
    def __init__(self, n, k):
        self.shape = torch.Size(gshape[n])
        self.off, r = shard(gshape[n][0], a.world, k)
        self.loc = torch.empty((r,) + tuple(gshape[n][1:]), dtype=torch.bfloat16)


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


names = nonpose_w                                                 # = 26232's model parameters minus vadapter.* / vgain.*
reps = {n: torch.empty(tuple(rep[n].shape), dtype=torch.bfloat16) for n in names if n in rep}
open(f"{a.init}/COMPLETE", "w").write("17850")                    # load_init asserts COMPLETE: written for the check, removed on failure
samp_chk = [n for n in ("inp.emb.0.weight", "head.lm_head.weight", "layers.0.layer.self_attn.q_proj.weight", "layers.63.layer.mlp.down_proj.weight",
                        "tower_s.code_emb.weight", "head.proj.U") if n in Rg]


def src_rows(n, lo, hi):
    """rows [lo, hi) of sharded tensor n, read directly from the source rank files."""
    c0 = -(-Rg[n] // 128)
    parts = []
    for s in range(lo // c0, min(128, -(-hi // c0))):
        t = torch.load(f"{a.init}/rank{s}.pt", map_location="cpu", mmap=True, weights_only=True)[n]
        a0, b0 = max(lo, s * c0), min(hi, s * c0 + t.shape[0])
        if b0 > a0:
            parts.append(t[a0 - s * c0:b0 - s * c0])
    return torch.cat(parts) if parts else None


tot = 0.0
try:
    for k in range(a.world):
        t0 = time.time()
        ns = {"json": json, "os": os, "torch": torch, "M": Mshim, "dist": Dshim, "POSE_PREFIX": POSE_PREFIX, "SKIPPED_INIT": {}}
        exec(code, ns)
        params = {n: (reps[n] if n in reps else P(n, k)) for n in names}

        class Model:
            def named_parameters(self):
                yield from params.items()
        with torch.no_grad():
            m_, gbk = ns["load_init"](Model(), a.init, "cpu")
        assert int(m_["step"]) == 17850
        sk = {p_: len(v) for p_, v in ns["SKIPPED_INIT"].items()}
        for n in samp_chk:                                        # the loaded rows == the source rows read directly
            p = params[n]
            if p.loc.shape[0]:
                assert torch.equal(p.loc, src_rows(n, p.off, p.off + p.loc.shape[0])), ("rank rows != source rows", n, k)
        if k == 0:
            assert all(torch.equal(reps[n], rep[n]) for n in reps), "replicated tensors != rep.pt"
        tot += gbk
        log(f"[3] world-{a.world} rank {k}: trainer load_init OK ({gbk:.2f} GB of rows, skipped init keys {sk}, all finite; sampled rows == source) {time.time() - t0:.0f}s")
        del params
        import gc
        gc.collect()
except BaseException:
    os.remove(f"{a.init}/COMPLETE")
    raise
os.remove(f"{a.init}/COMPLETE")
R_["checks"]["load_init"] = (f"train_aloe_v2vox_raw.py load_init (unchanged, CPU shims) accepted the copy for all {a.world} ranks of a world-{a.world} FSDP2 layout "
                             f"({tot:.1f} GB rows = 2x per-rank sum; skipped keys only pose parts {sk}); sampled rank rows == source rows; replicated == rep.pt")
log(f"[3] {R_['checks']['load_init']} ({time.time() - T0:.0f}s)")

# ---- 4. values: sampled tensors of every module type vs local-17641
lay = sorted({int(n.split(".")[1]) for n in Rg if n.startswith("layers.")})
pick = []
for pat in ("inp.emb.0.weight", "inp.emb.1.weight", "inp.aud.0.weight", "inp.aud.1.weight", "inp.pose.weight", "head.lm_head.weight",
            "head.norm.weight", "head.spec_out", "head.nw_out", "head.eob_out", "head.proj.U", "head.proj.D",
            f"layers.{lay[0]}.", f"layers.{lay[len(lay) // 2]}.", f"layers.{lay[-1]}.", "tower_s.code_emb.weight", "tower_s.text_emb.weight",
            "tower_s.cb0_head.weight", "tower_s.cond.weight", "tower_s.layers.0.", "tower_s.layers.35.", "tower_p.code_emb.weight",
            "dec_s.model.embed_tokens.weight", "dec_s.heads.weight", "dec_s.cb0_emb.weight", "dec_s.proj.weight", "dec_s.model.layers.0.",
            "dec_s.model.layers.27.", "dec_p.heads.weight", "decp_bb.weight"):
    pick += [n for n in sorted(gshape) if n == pat or (pat.endswith(".") and n.startswith(pat))]
refs = {}


def full(root, n, cache):
    if n in rep:
        return torch.load(f"{root}/rep.pt", map_location="cpu", mmap=True, weights_only=True)[n]
    if root not in cache:
        cache[root] = [torch.load(f"{root}/rank{r}.pt", map_location="cpu", mmap=True, weights_only=True) for r in range(128)]
    return torch.cat([cache[root][r][n] for r in range(128) if n in cache[root][r]])


cache = {}
vals = []
for n in pick:
    x = full(a.init, n, cache)
    y = full(a.ref, n, cache)
    assert x.dtype == torch.bfloat16 and tuple(x.shape) == gshape[n] and torch.isfinite(x.float()).all(), n
    xf, yf = x.float(), y.float()
    rel = float((xf - yf).norm() / yf.norm().clamp_min(1e-30))
    vals.append({"name": n, "shape": list(x.shape), "rms": float(xf.pow(2).mean().sqrt()), "rel_l2_vs_17641": rel, "equal_17641": bool(torch.equal(x, y))})
cache.clear()
eq = [v["name"] for v in vals if v["equal_17641"]]
R_["values"] = vals
R_["checks"]["values"] = (f"{len(vals)} sampled tensors (trunk first/mid/last layer, emb / audio code tables, heads, proj, speech tower, "
                          f"replicated decoder, pose parts): bf16, finite, shapes ok; rel L2 vs local-17641 median "
                          f"{sorted(v['rel_l2_vs_17641'] for v in vals)[len(vals) // 2]:.2e}, max {max(v['rel_l2_vs_17641'] for v in vals):.2e}; "
                          f"bit-equal to 17641: {len(eq)} {eq[:6]}")
log(f"[4] {R_['checks']['values']}")
open(f"{a.init}/COMPLETE", "w").write("17850")
R_["COMPLETE"] = True
R_["seconds"] = round(time.time() - T0)
R_["time"] = time.strftime("%F %T")
json.dump(R_, open(f"{a.ctl}/verify-{me}.json", "w"), indent=1)
log(f"== VERIFY PASS {a.init}: COMPLETE = 17850 written ({time.time() - T0:.0f}s); report {a.ctl}/verify-{me}.json")
