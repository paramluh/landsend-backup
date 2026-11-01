# download one export dir from our own private HF repo into a node-local dir, then check every file's size + LFS sha256 against the repo tree
import sys, os, hashlib, json, time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, snapshot_download
repo, sub, dst = sys.argv[1], sys.argv[2], sys.argv[3]
t0 = time.time()
api = HfApi()
tree = {f.path: f for f in api.list_repo_tree(repo, path_in_repo=sub, recursive=True) if getattr(f, "size", None) is not None}
print(f"{len(tree)} files {sum(f.size for f in tree.values())/1e9:.1f} GB", flush=True)
snapshot_download(repo, allow_patterns=[f"{sub}/*"], local_dir=dst, max_workers=int(os.environ.get("HFW", 32)))
print(f"downloaded in {time.time()-t0:.0f}s", flush=True)
def chk(p):
    f = tree[p]; lp = os.path.join(dst, p)
    if not os.path.isfile(lp) or os.path.getsize(lp) != f.size:
        return p, "SIZE"
    if f.lfs is None:
        return p, "ok-nolfs"
    h = hashlib.sha256()
    with open(lp, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return p, "ok" if h.hexdigest() == f.lfs.sha256 else "SHA"
with ThreadPoolExecutor(int(os.environ.get("HASHW", 8))) as ex:
    res = list(ex.map(chk, sorted(tree)))
bad = [r for r in res if not r[1].startswith("ok")]
json.dump({"repo": repo, "sub": sub, "dst": dst, "n": len(res), "bad": bad, "secs": round(time.time()-t0)}, open(os.path.join(dst, sub, "DL_VERIFY.json"), "w"), indent=1)
print("VERIFY", "PASS" if not bad else f"FAIL {bad[:5]}", f"{time.time()-t0:.0f}s", flush=True)
sys.exit(1 if bad else 0)
