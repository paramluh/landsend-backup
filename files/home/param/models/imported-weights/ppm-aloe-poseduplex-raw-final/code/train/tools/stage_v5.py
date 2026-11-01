"""poseduplex: stage v5's final export (ppm-local-bf16-shards-v1, world 128) from OUR private HF repo InternalCan/ppmvox-v5-17850 into ONE
flat dir on THIS host = evalvox/stage_export.py (aloe-ft-32-samsep) with: the repo / thread count from env, TAG 'auto' = the newest w-<N> tag on
HF whose 128 rank files + rep.pt / meta.json / COMPLETE are all present, and the subset json of tools/init_subset_v5.py (+-1 source margin).
Every file's sha256 == the HF LFS sha256 (small files: size == HF). Writes <dest>/STAGE.json; the dir appears only when complete (.part rename).
NEVER downloads anything but our own private repo.
usage (compute node): python -I stage_v5.py <tag | auto> <subset json> <host index 0..7> <dest root>   (dest = <dest root>/<tag>)"""
import hashlib
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, hf_hub_download

REPO = os.environ.get("POSEDUP_V5_REPO", "InternalCan/ppmvox-v5-17850")
assert REPO.startswith("InternalCan/"), REPO
THREADS = int(os.environ.get("STAGE_THREADS", "16"))
RESOLVE = sys.argv[1] == "resolve"                                 # 'stage_v5.py resolve': print the newest complete w-N tag and exit
tag, sj, hi, root = ("auto", None, None, None) if RESOLVE else (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
W0 = 128
api = HfApi()
files_all = api.list_repo_files(REPO)


def rank_map(t):
    where = {}
    for f in files_all:
        if not f.startswith(f"{t}/"):
            continue
        h, n = f[len(t) + 1:].split("/", 1)
        if n.startswith("rank") and n.endswith(".pt"):
            where[int(n[4:-3])] = f
    return where


if tag == "auto":
    cands = sorted({int(m.group(1)) for f in files_all for m in [re.match(r"^w-(\d+)/", f)] if m}, reverse=True)
    tag = None
    for n in cands:
        w = rank_map(f"w-{n}")
        if sorted(w) == list(range(W0)) and all(any(f == f"w-{n}/{w[0].split('/')[1]}/{x}" for f in files_all) for x in ("rep.pt", "meta.json", "COMPLETE")):
            tag = f"w-{n}"
            break
    assert tag, ("no complete w-N export on HF", REPO, cands[:5])
if RESOLVE:
    print(tag); sys.exit(0)
dest = f"{root}/{tag}"
if os.path.isfile(f"{dest}/COMPLETE"):
    print(f"{dest} already COMPLETE"); sys.exit(0)
need = set(json.load(open(sj))["hosts"][hi])
need |= {s + d for s in need for d in (-1, 1) if 0 <= s + d < W0}
where = rank_map(tag)
assert sorted(where) == list(range(W0)), ("export incomplete on HF", tag, len(where))
r0h = where[0].split("/")[1]
want = [where[s] for s in sorted(need)] + [f"{tag}/{r0h}/{n}" for n in ("rep.pt", "meta.json", "COMPLETE")]
info = {p.path: p for p in api.get_paths_info(REPO, want)}
part, dl = dest + ".part", dest + ".dl"
os.makedirs(part, exist_ok=True); os.makedirs(dl, exist_ok=True)
t0 = time.time()


def one(f):
    p = hf_hub_download(REPO, f, local_dir=dl)
    n = f.rsplit("/", 1)[1]
    lfs = getattr(info[f], "lfs", None)
    if lfs is not None:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""):
                h.update(b)
        assert h.hexdigest() == lfs.sha256, ("sha256 != HF LFS", f)
    else:
        assert os.path.getsize(p) == info[f].size, ("size != HF", f)
    shutil.move(p, f"{part}/{n}")
    return n


with ThreadPoolExecutor(THREADS) as ex:
    got = list(ex.map(one, want))
m = json.load(open(f"{part}/meta.json"))
assert m["format"] == "ppm-local-bf16-shards-v1" and int(m["world"]) == W0 and "vgain_theta_fp32" in m, ("not a PPMVOX export", m.get("format"), m.get("world"))
json.dump({"repo": REPO, "tag": tag, "host_index": hi, "files": len(got), "sources": sorted(need), "time_s": round(time.time() - t0),
           "step": m.get("step"), "vox_step": m.get("vox_step"), "run_tag": m.get("run_tag")}, open(f"{part}/STAGE.json", "w"))
shutil.rmtree(dl, ignore_errors=True)
os.rename(part, dest)
print(f"staged {REPO}:{tag} -> {dest}: {len(got)} files ({sum(os.path.getsize(f'{dest}/{n}') for n in got) / 1e9:.1f} GB), sha256 == HF LFS, "
      f"{time.time() - t0:.0f}s; step {m['step']} vox_step {m.get('vox_step')} run {m.get('run_tag')}")
