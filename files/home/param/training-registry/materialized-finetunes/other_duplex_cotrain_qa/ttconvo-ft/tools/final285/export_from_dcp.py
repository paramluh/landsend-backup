"""ttconvo-ft final state (10-21): node-local resume save dcp-285 -> node-local weights export export-285 in the trainer's own export
format (train_ppm.export_local, ppm-local-bf16-shards-v1, world 128), on CPU, read-only on the source. One task per host
(srun --overlap -c4 --cpu-bind=none nice ... inside whatever job holds the 16 hosts; no GPU).

Why this is exact: export_local writes, per rank R, {fqn: p.to_local().to(bf16)} for every FSDP2-sharded parameter and, on rank 0,
rep.pt = {fqn: p.to(bf16)} for every replicated parameter. save_local's dcp rank<R>.pt 'model' section is {fqn: p.to_local()} for the
same sharded parameters in the same (named_parameters) order, and its rep.pt 'model' section the same replicated parameters. So the export
is the dcp's 'model' sections cast to bf16; fp32 -> bf16 is round-to-nearest-even on CPU and CUDA alike (identical for every finite value;
non-finite values are asserted absent).

Per rank of this host (train_state.json hosts map):
  1. torch.load(mmap) the full rank file; safety_ppm.digest_state(it) == digest-<R>.txt (the trainer's own resume / FT-init check)
  2. every 'model' tensor finite; out = {fqn: t.to(bf16).contiguous().clone()} in the file's order -> rank<R>.pt.tmp -> rank<R>.pt
  3. re-load the written file: same keys / order, every tensor torch.equal to the cast of the source
rank 0's host also: rep.pt (same 3 checks against digest-rep.txt) + meta.json (export_local's fields; src / made note this conversion).
--finalize (rank 0's host, after every host's state/<host>.ok): writes COMPLETE (= str(step)), as export_local does after its barrier."""
import argparse
import json
import os
import socket
import sys
import time

import torch

sys.path.insert(0, "/data/home/amartyavarma/ppm/ttconvo-ft/code")
import safety_ppm as SF  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/scratch/amartyavarma/ttconvo-ft/ckpt/ttconvo-ft/dcp-285")
ap.add_argument("--out", default="/scratch/amartyavarma/ttconvo-ft/exports/export-285")
ap.add_argument("--state", default="/data/home/amartyavarma/ppm/ttconvo-ft/tools/final285/state")
ap.add_argument("--finalize", action="store_true")
ap.add_argument("--min-free-gb", type=float, default=30.0)
a = ap.parse_args()
H = socket.gethostname().split(".")[0]
torch.set_num_threads(1)                      # 1 thread: no OpenMP spin on the cores shared with training rank 0 (4 threads slowed the 14B ~8%)
os.makedirs(a.state, exist_ok=True)


def log(*m):
    print(f"[{time.strftime('%H:%M:%S')} {H}]", *m, flush=True)


def drop_cache(p):
    try:
        fd = os.open(p, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except Exception:
        pass


ts = json.load(open(f"{a.src}/train_state.json"))
step, world, hosts = int(ts["step"]), int(ts["world"]), ts["hosts"]

if a.finalize:
    hs = sorted(set(hosts), key=lambda h: [int(x) if x.isdigit() else x for x in __import__("re").split(r"(\d+)", h)])
    oks = {h: json.load(open(f"{a.state}/{h}.ok")) for h in hs if os.path.isfile(f"{a.state}/{h}.ok")}
    assert len(oks) == len(hs), ("hosts not done", sorted(set(hs) - set(oks)))
    assert all(o["step"] == step and o["ok"] for o in oks.values()), "a host reported a failed check"
    ranks = sorted(r for o in oks.values() for r in o["ranks"])
    assert ranks == list(range(world)), ("rank files", len(ranks), world)
    assert hosts[0] == H and os.path.isfile(f"{a.out}/rep.pt") and os.path.isfile(f"{a.out}/meta.json"), "finalize runs on rank 0's host"
    open(f"{a.out}/COMPLETE", "w").write(str(step))
    log(f"COMPLETE {a.out}: {world} rank files on {len(hs)} hosts + rep.pt + meta.json, step {step}")
    raise SystemExit(0)

assert os.path.isfile(f"{a.src}/COMPLETE") and open(f"{a.src}/COMPLETE").read().strip() == str(step), ("source not COMPLETE", a.src)
mine = [r for r, h in enumerate(hosts) if h == H]
assert len(mine) == 8, (H, mine)
assert not os.path.exists(f"{a.out}/COMPLETE"), ("export already COMPLETE here", a.out)
os.makedirs(a.out, exist_ok=True)
t0 = time.time()
rec = {"host": H, "step": step, "ranks": mine, "files": {}, "ok": False}


def convert(srcf, digf, dstf):
    sd = torch.load(srcf, map_location="cpu", mmap=True, weights_only=True)
    dg = SF.digest_state(sd)
    want = open(digf).read().strip()
    assert dg == want, ("digest mismatch", srcf, dg, want)
    m = sd["model"]
    dts = {}
    for n, t in m.items():
        dts[str(t.dtype)] = dts.get(str(t.dtype), 0) + t.numel()
        assert bool(torch.isfinite(t).all()), ("non-finite weight", srcf, n)
    out = {n: t.to(torch.bfloat16).contiguous().clone() for n, t in m.items()}
    # the archive name inside a torch.save zip = the written file's name: the trainer saves rank<R>.pt via rank<R>.pt.tmp (archive
    # 'rank<R>.pt') but rep.pt directly (archive 'rep'); write each the same way (rep via a temp dir) so the files match its exports byte-size
    tmpf = dstf + ".tmp" if not dstf.endswith("/rep.pt") else os.path.join(os.path.dirname(dstf), ".repfix", "rep.pt")
    os.makedirs(os.path.dirname(tmpf), exist_ok=True)
    torch.save(out, tmpf)
    os.replace(tmpf, dstf)
    if tmpf.endswith("/.repfix/rep.pt"):
        os.rmdir(os.path.dirname(tmpf))
    back = torch.load(dstf, map_location="cpu", mmap=True, weights_only=True)
    assert list(back) == list(m), ("key order", dstf)
    for n, t in m.items():
        assert back[n].dtype == torch.bfloat16 and torch.equal(back[n], t.to(torch.bfloat16)), ("re-read mismatch", dstf, n)
    nb = sum(t.numel() * 2 for t in out.values())
    nt = len(m)
    del sd, m, out, back
    drop_cache(srcf); drop_cache(dstf)
    return {"tensors": nt, "src_dtypes": dts, "bf16_bytes": nb, "size": os.path.getsize(dstf), "digest_ok": True}


free0 = __import__("shutil").disk_usage(a.out).free / 1e9
need = 8 * 0.68 + (2.4 if 0 in mine else 0)
assert free0 - need >= a.min_free_gb, (f"/scratch free {free0:.1f} GB - {need:.1f} GB export < {a.min_free_gb} GB floor")
for r in mine:
    rec["files"][f"rank{r}.pt"] = convert(f"{a.src}/rank{r}.pt", f"{a.src}/digest-{r}.txt", f"{a.out}/rank{r}.pt")
    log(f"rank{r}: digest == saved, finite, bf16 cast written + re-read equal ({rec['files'][f'rank{r}.pt']['size']} B)")
if 0 in mine:
    rec["files"]["rep.pt"] = convert(f"{a.src}/rep.pt", f"{a.src}/digest-rep.txt", f"{a.out}/rep.pt")
    log(f"rep.pt: digest == saved, finite, bf16 cast written + re-read equal ({rec['files']['rep.pt']['size']} B)")
    json.dump({"format": "ppm-local-bf16-shards-v1",
               "src": f"ttconvo-ft {a.src} (STOP_AFTER_SAVE step {step}) -> CPU bf16 cast of the 'model' sections "
                      "(tools/final285/export_from_dcp.py; == train_ppm export_local)",
               "step": step, "world": world, "hosts": hosts, "run_tag": ts["cfg"]["run_tag"], "init": ts["cfg"]["ft"]["init"],
               "cfg": ts["cfg"], "save_time": ts.get("time"), "decp_bb": 1, "decp_bb_alpha": 0.0, "decp_bb_off": 1,
               "made": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, open(f"{a.out}/meta.json", "w"), indent=1, default=str)
rec["ok"] = True
rec["time_s"] = round(time.time() - t0, 1)
rec["scratch_free_gb_after"] = round(__import__("shutil").disk_usage(a.out).free / 1e9, 1)
json.dump(rec, open(f"{a.state}/{H}.ok.tmp", "w"), indent=1)
os.replace(f"{a.state}/{H}.ok.tmp", f"{a.state}/{H}.ok")
log(f"done: ranks {mine} -> {a.out} in {rec['time_s']} s; /scratch free {rec['scratch_free_gb_after']} GB")
