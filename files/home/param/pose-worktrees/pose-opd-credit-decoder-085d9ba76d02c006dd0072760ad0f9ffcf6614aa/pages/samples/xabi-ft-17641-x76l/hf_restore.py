# X76L (10-23): one-off restore of FT-(b) aloe-v2pure-17641 export-200 slinky-0 + slinky-2 shard sets from our own private HF backup ->
# slinky-15 /scratch; verify size (manifest + HF tree) and sha256 (HF LFS sha = upload-time local sha) + the hf_verify json samples.
import hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download
R = "InternalCan/ppm-aloe-v2pure-17641-final"; DST = "/scratch/amartyavarma/xabi-samples/w17641"
VJD = "/data/home/amartyavarma/ppm/aloe-v2pure-17641/logs/hf_verify"
HOSTS = sys.argv[1:] or ["slinky-0", "slinky-2"]
api = HfApi()
files = {}
for h in HOSTS:
    man = None
    for p in (f"_backup/MANIFEST.export-200.{h}.tsv", f"export-200/_backup/MANIFEST.export-200.{h}.tsv"):
        try:
            man = hf_hub_download(R, p, local_dir="/scratch/amartyavarma/xabi-samples/_man17641"); break
        except Exception as e:
            print("manifest miss", p, type(e).__name__, flush=True)
    rows = [l.rstrip("\n").split("\t") for l in open(man) if l.strip()]
    for r in rows:
        if len(r) > 1 and r[1].isdigit() and "_backup" not in r[0]:
            files[r[0] if r[0].startswith("export-200/") else "export-200/" + r[0]] = int(r[1])
print("manifest", len(files), "files", sum(files.values()), "B", flush=True)
info = {}
for e in api.list_repo_tree(R, path_in_repo="export-200", expand=True):
    if hasattr(e, "size"):
        info[e.path] = (e.size, e.lfs.sha256 if getattr(e, "lfs", None) else None)
samp = {}
for h in HOSTS:
    vj = json.load(open(f"{VJD}/InternalCan__ppm-aloe-v2pure-17641-final__export-200__{h}.json"))
    assert vj["PASS"], h
    for s in vj.get("sample_download_sha256", []):
        samp[s["path"]] = s.get("sha256_local") or s.get("sha256_hf")
def sha(p):
    m = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            m.update(b)
    return m.hexdigest()
def one(rp):
    t = time.time(); p = hf_hub_download(R, rp, local_dir=DST)
    sz = os.path.getsize(p); hs = info.get(rp, (None, None))
    loc = sha(p)
    ok_sz = sz == files[rp] == hs[0]
    ok_sha = (hs[1] is None and rp.endswith(("COMPLETE", "meta.json"))) or loc == hs[1]
    vs = samp.get(rp)
    vjok = vs is None or vs == loc
    return rp, sz, ok_sz, ok_sha, vs is not None, vjok, round(time.time() - t)
with ThreadPoolExecutor(len(files)) as ex:
    res = list(ex.map(one, sorted(files)))
bad = 0
for rp, sz, ok_sz, ok_sha, hv, vjok, dt in res:
    bad += not (ok_sz and ok_sha and vjok)
    print(rp, sz, "size_ok" if ok_sz else "SIZE_BAD", "sha_ok" if ok_sha else "SHA_BAD", ("vj_sample_ok" if vjok else "VJ_BAD") if hv else "vj_n/a", f"{dt}s", flush=True)
print("RESTORE", "PASS" if bad == 0 and len(res) == len(files) else f"FAIL bad={bad}", len(res), "files", time.strftime("%H:%M:%S"), flush=True)
