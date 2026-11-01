"""evalvox: stage a PPMVOX node-local export (ppm-local-bf16-shards-v1, world 128) from OUR private HF backup InternalCan/ppmvox-16861/<tag>/<host>/
into one flat dir on THIS host, only the rank files this host's eval ranks read (subset json from init_subset_hosts.py, +-1 margin) + rep.pt /
meta.json / COMPLETE (from rank 0's host dir). Every file's sha256 == the HF LFS sha256 (or the git blob for small files is size-checked).
usage (compute node): python -I stage_export.py <tag e.g. w-16911> <subset json> <host index> <dest dir>"""
import os, sys, json, hashlib, shutil, time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download
REPO = os.environ.get("EVALVOX_HF_REPO", "InternalCan/ppmvox-16861")   # v2: InternalCan/ppmvox-v2-16861
tag, sj, hi, dest = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
if os.path.isfile(f"{dest}/COMPLETE"):
    print(f"{dest} already COMPLETE"); sys.exit(0)
need = set(json.load(open(sj))["hosts"][hi])
W0 = 128
need |= {s + d for s in need for d in (-1, 1) if 0 <= s + d < W0}
api = HfApi()
files = [f for f in api.list_repo_files(REPO) if f.startswith(f"{tag}/")]
where = {}
for f in files:
    h, n = f[len(tag) + 1:].split("/", 1)          # tag may carry an HF path prefix (ppmvox-v2/w-N): <tag>/<host>/<file>
    if n.startswith("rank") and n.endswith(".pt"):
        where[int(n[4:-3])] = f
assert sorted(where) == list(range(W0)), ("export incomplete on HF", len(where))
r0h = where[0].split("/")[1]
want = [where[s] for s in sorted(need)] + [f"{tag}/{r0h}/{n}" for n in ("rep.pt", "meta.json", "COMPLETE")]
info = {p.path: p for p in api.get_paths_info(REPO, want)}
part = dest + ".part"; dl = dest + ".dl"
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
with ThreadPoolExecutor(8) as ex:
    got = list(ex.map(one, want))
m = json.load(open(f"{part}/meta.json")); assert m["format"] == "ppm-local-bf16-shards-v1" and int(m["world"]) == W0
json.dump({"repo": REPO, "tag": tag, "host_index": hi, "files": len(got), "sources": sorted(need), "time_s": round(time.time() - t0)}, open(f"{part}/EVALVOX_STAGE.json", "w"))
shutil.rmtree(dl, ignore_errors=True)
os.rename(part, dest)
print(f"staged {tag} -> {dest}: {len(got)} files ({sum(os.path.getsize(f'{dest}/{n}') for n in got) / 1e9:.1f} GB), sha256 == HF LFS, {time.time() - t0:.0f}s; step {m['step']} vox_step {m.get('vox_step')}")
