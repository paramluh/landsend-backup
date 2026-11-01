"""pdsamples: stage a poseduplex node-local export (ppm-local-bf16-shards-v1, world 64: rank*.pt + rep.pt + meta.json + COMPLETE) from our
private HF backup into ONE node-local dir for the fast engine (fast_ppm.load_weights reads every rank). sha256 == HF LFS checked per file.
usage (compute node): python -I stage_pd.py <repo> <tag> <dest>"""
import os, sys, json, hashlib, shutil, time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download
repo, tag, dest = sys.argv[1], sys.argv[2], sys.argv[3]
if os.path.isfile(f"{dest}/COMPLETE"):
    print(f"{dest} already COMPLETE"); sys.exit(0)
api = HfApi()
files = [f for f in api.list_repo_files(repo) if f.startswith(f"{tag}/")]
by = {f.rsplit("/", 1)[1]: f for f in files}
ranks = sorted(int(n[4:-3]) for n in by if n.startswith("rank") and n.endswith(".pt"))
assert ranks and ranks == list(range(len(ranks))), ("rank files not contiguous on HF", ranks[:5], len(ranks))
for n in ("rep.pt", "meta.json", "COMPLETE"):
    assert n in by, ("missing on HF", n)
want = [by[f"rank{r}.pt"] for r in ranks] + [by[n] for n in ("rep.pt", "meta.json", "COMPLETE")]
info = {p.path: p for p in api.get_paths_info(repo, want)}
part, dl = dest + ".part", dest + ".dl"
os.makedirs(part, exist_ok=True); os.makedirs(dl, exist_ok=True)
t0 = time.time()
def one(f):
    n = f.rsplit("/", 1)[1]
    if os.path.isfile(f"{part}/{n}") and os.path.getsize(f"{part}/{n}") == info[f].size:
        return n
    p = hf_hub_download(repo, f, local_dir=dl)
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
with ThreadPoolExecutor(24) as ex:
    got = list(ex.map(one, want))
m = json.load(open(f"{part}/meta.json")); assert m["format"] == "ppm-local-bf16-shards-v1" and int(m["world"]) == len(ranks), (m["format"], m["world"], len(ranks))
json.dump({"repo": repo, "tag": tag, "files": len(got), "time_s": round(time.time() - t0)}, open(f"{part}/PDSAMPLES_STAGE.json", "w"))
shutil.rmtree(dl, ignore_errors=True)
os.rename(part, dest)
print(f"staged {repo}/{tag} -> {dest}: {len(got)} files ({sum(os.path.getsize(f'{dest}/{n}') for n in got) / 1e9:.1f} GB), {time.time() - t0:.0f}s; step {m['step']}", flush=True)
