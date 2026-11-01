"""ALOEFT32 base (10-24): PPM 8,1,1 training save dcp-17850 (private HF backup InternalCan/ppm-v1-c, verified 10-23) -> a NODE-LOCAL full
ppm-local-bf16-shards-v1 copy (world 128: rank<r>.pt x 128 + rep.pt + meta.json + COMPLETE) on THIS host, the format the reference Aloe
trainer (aloe-v2vox-asr-w15973/code/train_aloe_v2vox_raw.py load_init) reads as QQA_INIT.

Per source file: hf_hub_download (xet) into a node-local download dir -> sha256 == HF lfs sha256 (end-to-end) -> conversion = the MONOFT
converter's maths verbatim (aloe-v2pure-main-17850/tools/monoft/ckpt_node_export.py: mmap 'model' section, .to(bfloat16).contiguous().clone(),
torch.save) -> sha256 of the converted file == the 17850 monoplex FT's init manifest (initctl/manifest-<host>.json, written 10-22 by job
27756 from the node-local dcp-17850 on the main hosts) -> the raw downloaded copy (our own transient duplicate; HF + the main hosts keep
theirs) is removed. Resumable: converted files already present with the manifest sha256 are kept. COMPLETE is NOT written here
(verify_17850.py writes it after the trainer's load_init check). Never touches anything outside --init / --dl.
  python stage_17850.py --init /scratch/amartyavarma/aloe-samsep-17850-init --dl /scratch/amartyavarma/aloe-samsep-17850-dl --ctl <dir>"""
import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import shutil
import socket
import sys
import threading
import time

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import torch  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

sys.path.insert(0, "/data/home/amartyavarma/ppm")
from scratch_guard import assert_deletable  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True)
ap.add_argument("--dl", required=True)
ap.add_argument("--ctl", required=True)
ap.add_argument("--repo", default="InternalCan/ppm-v1-c")
ap.add_argument("--step", type=int, default=17850)
ap.add_argument("--tag", default="")
ap.add_argument("--man", default="/data/home/amartyavarma/ppm/aloe-v2pure-main-17850/initctl")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--min-free-gb", type=float, default=30.0)
a = ap.parse_args()
a.tag = a.tag or f"dcp-{a.step}"
me = socket.gethostname().split(".")[0]
for p in (a.init, a.dl):
    assert p.startswith("/scratch/amartyavarma/") and ".." not in p and p.rstrip("/") != "/scratch/amartyavarma", p
os.makedirs(a.init, exist_ok=True); os.makedirs(a.dl, exist_ok=True); os.makedirs(a.ctl, exist_ok=True)
LK = threading.Lock()


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


def free_gb(p):
    return shutil.disk_usage(p).free / 1e9


# ---- expected converted files: a monoft init manifest of THIS step if one exists (17850: job 27756); else none (10-24: dcp-16861 has
#      none -> integrity = raw sha256 == HF lfs sha256; converted sha256 recorded in a ledger in --ctl, used for resume)
exp = {}
for p in sorted(glob.glob(f"{a.man}/manifest-*.json")) if a.man else []:
    m = json.load(open(p))
    if int(m["step"]) != a.step:
        continue
    assert int(m["world"]) == 128, (p, m["world"])
    for f, e in m["files"].items():
        assert f not in exp, f
        exp[f] = e
assert not exp or (all(f"rank{r}.pt" in exp for r in range(128)) and "rep.pt" in exp), "manifests incomplete"
LEDGER = f"{a.ctl}/ledger-{a.step}-{me}.json"
ledger = json.load(open(LEDGER)) if os.path.isfile(LEDGER) else {}
# ---- HF listing: rank<r>.pt -> its host folder (+ size, lfs sha256)
api = HfApi()
src = {}
reps_ = {}
for f in api.list_repo_tree(a.repo, path_in_repo=a.tag, recursive=True):
    if not hasattr(f, "size") or f.lfs is None and not f.path.endswith(".json"):
        continue
    host, name = f.path.split("/")[-2], f.path.split("/")[-1]
    if name.startswith("rank") and name.endswith(".pt"):
        assert name not in src, ("rank file in two host folders", name)
        src[name] = {"path": f.path, "size": f.size, "sha": f.lfs.sha256}
    elif name == "rep.pt":
        reps_[host] = {"path": f.path, "size": f.size, "sha": f.lfs.sha256}
H0 = src["rank0.pt"]["path"].split("/")[-2]                     # rep.pt of rank 0's host (= the converter's choice)
src["rep.pt"] = reps_[H0]
assert sorted(src) == sorted([f"rank{r}.pt" for r in range(128)] + ["rep.pt"]), ("HF listing incomplete", len(src))
ts = json.load(open(hf_hub_download(a.repo, f"{a.tag}/{H0}/train_state.json", local_dir=a.dl)))
assert int(ts["step"]) == a.step and int(ts["world"]) == 128 and len(ts["hosts"]) == 128, (ts["step"], ts["world"])
log(f"HF {a.repo}/{a.tag}: 128 rank files + {H0} rep.pt, {sum(e['size'] for e in src.values()) / 1e9:.1f} GB raw; train_state step {ts['step']} "
    f"world {ts['world']}; bf16 ~{sum(e['size'] for e in src.values()) / 6e9:.1f} GB; manifest of this step: {'yes' if exp else 'none (ledger ' + LEDGER + ')'}")

todo = []
for f in sorted(src, key=lambda x: (x != "rep.pt", int(x[4:-3]) if x.startswith("rank") else -1)):
    d = f"{a.init}/{f}"
    if os.path.isfile(d) and os.path.getsize(d) == (exp[f]["size"] if exp else ledger.get(f, {}).get("size", -1)):
        todo.append((f, True))
    else:
        todo.append((f, False))
need_gb = sum(src[f]["size"] / 6 for f, ok in todo if not ok) / 1e9
fr = free_gb(a.init)
log(f"/scratch free {fr:.0f} GB; still to write {need_gb:.1f} GB + transient raw <= {a.workers} x 4.1 GB; floor {a.min_free_gb:.0f} GB")
# peak = (almost) all converted bytes + W raw files in flight; rep.pt (13.8 GB raw) is converted FIRST, while the init is still ~empty
assert fr - need_gb - a.workers * 4.1 >= a.min_free_gb, "not enough /scratch"
stats = {"done": 0, "raw_gb": 0.0, "dl_s": 0.0}
res = {}


def one(item):
    f, maybe = item
    d = f"{a.init}/{f}"
    if maybe and sha256(d) == (exp[f]["sha256"] if exp else ledger.get(f, {}).get("sha256")):
        return f, "present", 0.0, 0.0
    e = src[f]
    for t in range(4):
        t0 = time.time()
        p = hf_hub_download(a.repo, e["path"], local_dir=a.dl)
        dl = time.time() - t0
        ok = os.path.getsize(p) == e["size"] and sha256(p) == e["sha"]
        if ok:
            break
        log(f"{f}: raw sha256 / size != HF lfs (attempt {t + 1}); re-download")
        assert_deletable(p); os.remove(p)
    else:
        raise RuntimeError(f"{f}: raw download never matched HF lfs sha256")
    t1 = time.time()
    sd = torch.load(p, map_location="cpu", mmap=True, weights_only=True)
    out = {n: t.to(torch.bfloat16).contiguous().clone() for n, t in sd["model"].items()}   # = ckpt_node_export.py
    nsrc = len(sd["model"])
    # torch.save names the zip archive after the target's basename: ckpt_node_export.py wrote rank files via rank<r>.pt.tmp but rep.pt
    # directly (archive 'rep'), so rep.pt goes through a tmp DIR with the same basename to reproduce its bytes
    if f == "rep.pt":
        os.makedirs(f"{a.init}/.tmp-rep", exist_ok=True)
        torch.save(out, f"{a.init}/.tmp-rep/rep.pt")
        os.replace(f"{a.init}/.tmp-rep/rep.pt", d)
        os.rmdir(f"{a.init}/.tmp-rep")
    else:
        torch.save(out, d + ".tmp")
        os.replace(d + ".tmp", d)
    del sd, out
    got = sha256(d)
    if not exp:
        with LK:
            ledger[f] = {"size": os.path.getsize(d), "sha256": got, "raw_sha256_hf": e["sha"]}
            json.dump(ledger, open(LEDGER + ".tmp", "w")); os.replace(LEDGER + ".tmp", LEDGER)
    elif got != exp[f]["sha256"]:
        raise RuntimeError(f"{f}: converted sha256 {got[:16]} != 17850 manifest {exp[f]['sha256'][:16]} (size {os.path.getsize(d)} vs {exp[f]['size']})")
    assert_deletable(p); os.remove(p)                             # our transient raw duplicate (HF + main hosts keep theirs)
    with LK:
        stats["done"] += 1; stats["raw_gb"] += e["size"] / 1e9; stats["dl_s"] += dl
        n = stats["done"]
    if n % 8 == 0 or f == "rep.pt":
        log(f"{n} converted ({f}: raw {e['size'] / 1e9:.2f} GB in {dl:.0f}s, {nsrc} tensors, convert+sha {time.time() - t1:.0f}s); /scratch free {free_gb(a.init):.0f} GB")
        json.dump({"host": me, "converted": n, "of": sum(1 for _, ok in todo if True), "time": time.strftime("%F %T")}, open(f"{a.ctl}/progress-{me}.json", "w"))
    return f, "converted", e["size"] / 1e9, dl


t0 = time.time()
ex = cf.ThreadPoolExecutor(a.workers)
futs = [ex.submit(one, it) for it in todo]
try:
    for fu in cf.as_completed(futs):
        f, how, gb, dl = fu.result()                              # first failure: cancel everything not started, re-raise
        res[f] = how
except BaseException:
    for fu in futs:
        fu.cancel()
    ex.shutdown(wait=True)
    raise
ex.shutdown(wait=True)
for sub in (f"{a.dl}/{a.tag}",):                                    # empty dirs left by the downloads
    for dp, dn, fn in sorted(os.walk(sub, topdown=False), reverse=False):
        if not fn and not dn and os.path.isdir(dp):
            os.rmdir(dp)
# ---- meta.json (= ckpt_node_export.py's fields; src names the HF copy of the same save)
meta = {"format": "ppm-local-bf16-shards-v1", "src": f"hf://{a.repo}/{a.tag}",
        "step": int(ts["step"]), "world": int(ts["world"]), "hosts": ts["hosts"], "run_tag": ts["cfg"].get("run_tag"), "run_base": ts["cfg"].get("run_base"),
        "cfg": ts["cfg"], "save_time": ts.get("time"), "made": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "made_by": "ALOEFT32 base stage_17850.py (ckpt_node_export maths; raw sha256 == HF lfs" + (", converted == monoft manifest" if exp else "") + ")"}
json.dump(meta, open(f"{a.init}/meta.json.tmp", "w"), indent=1)
os.replace(f"{a.init}/meta.json.tmp", f"{a.init}/meta.json")
tot = sum(os.path.getsize(f"{a.init}/{f}") for f in src) / 1e9
log(f"== STAGE DONE {a.init}: step {a.step}, 129 bf16 files {tot:.1f} GB, raw sha256 == HF lfs" + (", converted == manifest" if exp else "") + " ({sum(1 for v in res.values() if v == 'converted')} converted, "
    f"{sum(1 for v in res.values() if v == 'present')} already present); raw {stats['raw_gb']:.0f} GB from HF in {time.time() - t0:.0f}s wall; "
    f"/scratch free {free_gb(a.init):.0f} GB")
json.dump({"host": me, "step": a.step, "stage": "done", "gb": round(tot, 2), "files": len(res), "free_gb": round(free_gb(a.init), 1), "time": time.strftime("%F %T")},
          open(f"{a.ctl}/stage-{a.step}-{me}.json", "w"))
