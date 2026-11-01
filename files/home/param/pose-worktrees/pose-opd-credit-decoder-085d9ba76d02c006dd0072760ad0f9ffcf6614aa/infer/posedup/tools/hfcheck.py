# every file of a local dir vs a private-HF path: size + sha256 (HF LFS sha256 for LFS files, git blob size for small files) -> json report
import hashlib, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi
local, repo, sub, out = sys.argv[1:5]
tree = {f.path: f for f in HfApi().list_repo_tree(repo, path_in_repo=sub, recursive=True) if getattr(f, "size", None) is not None}
files = sorted(os.path.relpath(os.path.join(dp, f), local) for dp, _, fs in os.walk(local) for f in fs)
def chk(rel):
    hp = f"{sub}/{rel}"; f = tree.get(hp); lp = os.path.join(local, rel); sz = os.path.getsize(lp)
    if f is None:
        return rel, "MISSING_ON_HF", sz
    if f.size != sz:
        return rel, f"SIZE {sz} != {f.size}", sz
    if f.lfs is None:
        h = hashlib.sha1(b"blob %d\0" % sz + open(lp, "rb").read()).hexdigest()
        return rel, "ok-blob" if h == getattr(f, "blob_id", h) else "BLOB_SHA", sz
    h = hashlib.sha256()
    with open(lp, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return rel, "ok" if h.hexdigest() == f.lfs.sha256 else "SHA", sz
with ThreadPoolExecutor(int(os.environ.get("HASHW", 4))) as ex:
    res = list(ex.map(chk, files))
bad = [r for r in res if not r[1].startswith("ok")]
rep = {"local": local, "repo": repo, "sub": sub, "files": len(res), "bytes": sum(r[2] for r in res), "bad": bad, "PASS": bool(res) and not bad}
json.dump(rep, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != "bad"}), "bad:", bad[:5])
