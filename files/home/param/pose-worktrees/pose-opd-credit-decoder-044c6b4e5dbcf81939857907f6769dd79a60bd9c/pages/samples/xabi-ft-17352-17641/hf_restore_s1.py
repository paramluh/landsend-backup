# One-off restore (coordinator-approved 10-22 18:20 PT) of FT-(a) aloe-v2pure-17352 export-200 slinky-1 shard set (rank 0-7 + extras) from our own
# private HF backup -> spare-node /scratch; verify size + sha256 vs HF LFS sha (= upload-time local sha, per the verify json) and the manifest.
import hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
try:
    import hf_transfer  # noqa
except Exception:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import HfApi, hf_hub_download
R = "InternalCan/ppm-aloe-v2pure-17352-final"; DST = "/scratch/amartyavarma/xabi-samples/w17352"
VJ = "/data/home/amartyavarma/ppm/aloe-v2pure-17352/logs/hf_verify/InternalCan__ppm-aloe-v2pure-17352-final__export-200__slinky-1.json"
api = HfApi()
try:
    man = hf_hub_download(R, "_backup/MANIFEST.export-200.slinky-1.tsv", local_dir=DST + "/_man")
except Exception:
    man = hf_hub_download(R, "export-200/_backup/MANIFEST.export-200.slinky-1.tsv", local_dir=DST + "/_man")
rows = [l.rstrip("\n").split("\t") for l in open(man) if l.strip()]
files = {r[0]: int(r[1]) for r in rows if len(r) > 1 and r[1].isdigit() and "_backup" not in r[0]}
print("manifest", files, flush=True)
info = {}
for e in api.list_repo_tree(R, path_in_repo="export-200", expand=True):
    if hasattr(e, "size"):
        info[e.path] = (e.size, e.lfs.sha256 if getattr(e, "lfs", None) else None)
def sha(p):
    m = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            m.update(b)
    return m.hexdigest()
def one(rel):
    rp = rel if rel.startswith("export-200/") else "export-200/" + rel
    t = time.time(); p = hf_hub_download(R, rp, local_dir=DST)
    sz = os.path.getsize(p); hs = info.get(rp, (None, None))
    ok_sz = sz == files[rel] == hs[0]
    loc = sha(p) if hs[1] else None
    ok_sha = (hs[1] is None) or (loc == hs[1])
    return rp, sz, ok_sz, ok_sha, loc, round(time.time() - t)
with ThreadPoolExecutor(11) as ex:
    res = list(ex.map(one, sorted(files)))
vj = json.load(open(VJ)); samp = {s["path"]: s.get("sha256_local") or s.get("sha256_hf") for s in vj.get("sample_download_sha256", [])}
bad = 0
for rp, sz, ok_sz, ok_sha, loc, dt in res:
    vs = samp.get(rp) or samp.get(rp.split("/", 1)[1])
    vjok = (vs is None) or (vs == loc)
    bad += not (ok_sz and ok_sha and vjok)
    print(rp, sz, "size_ok" if ok_sz else "SIZE_BAD", "sha_ok" if ok_sha else "SHA_BAD", "vj_sample_ok" if vs and vjok else ("vj_n/a" if vs is None else "VJ_BAD"), f"{dt}s", flush=True)
print("RESTORE", "PASS" if bad == 0 and len(res) == len(files) else f"FAIL bad={bad}", len(res), "files", flush=True)
