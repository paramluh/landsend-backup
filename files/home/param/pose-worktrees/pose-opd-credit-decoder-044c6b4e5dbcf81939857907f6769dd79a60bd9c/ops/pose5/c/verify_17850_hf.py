"""pose5 workstream C: verify the HF copy of the 8,1,1 pause save dcp-17850 (READ ONLY; nothing is written to HF or to any host).
Checks, against the watcher's own records (setup-logs/hfb/lists/shards-dcp-17850.tsv = the file list + bytes it uploaded from the
node-local saves, and verify/InternalCan__ppm-v1-c__dcp-17850__<host>.json = its per-host VERIFY PASS with sha vs upload cache):
  1. HF InternalCan/ppm-v1-c lists exactly 16 host dirs x 20 files under dcp-17850/ (+ 16 _backup/MANIFEST.dcp-17850.<host>.tsv);
  2. every file's HF size == the recorded size;
  3. every LFS file's HF sha256 == the sha256 in the per-host MANIFEST tsv (if the manifest carries one);
  4. small files (train_state.json, digest-*.txt, COMPLETE) streamed back: train_state step == 17850 and the host list / rank layout;
  5. the bundle in InternalCan/ppm-v1-d (RESTORE.md, SHARDS.json) is present.
Writes /data/home/amartyavarma/ppm/pose5/c/hf17850/VERIFY_17850.json (+ per-file table hf_files.tsv). HF token: the default HF path
(never printed). Run on a spare node (tiny srun), never on the login node."""
import hashlib
import json
import os
import sys
import time

OUT = "/data/home/amartyavarma/ppm/pose5/c/hf17850"
REC = "/data/home/amartyavarma/setup-logs/hfb/lists/shards-dcp-17850.tsv"
VREC = "/data/home/amartyavarma/setup-logs/hfb/verify/InternalCan__ppm-v1-c__dcp-17850__{h}.json"
REPO, BREPO, TYPE, TAG = "InternalCan/ppm-v1-c", "InternalCan/ppm-v1-d", "model", "dcp-17850"
os.makedirs(OUT, exist_ok=True)
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

api = HfApi()
rec = {}
for ln in open(REC).read().splitlines()[1:]:
    p, s, src = ln.split("\t")
    rec[p] = int(s)
hosts = sorted({p.split("/")[1] for p in rec if p.startswith(TAG + "/")}, key=lambda h: int(h.split("-")[1]))
res = {"repo": REPO, "bundle_repo": BREPO, "tag": TAG, "time_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
       "recorded_files": len(rec), "hosts": hosts, "problems": []}
info = api.repo_info(REPO, repo_type=TYPE)
res["private"] = bool(info.private)
# 1+2: HF listing of each host dir (expand -> sizes + LFS sha256)
hf = {}
for h in hosts:
    for e in api.list_repo_tree(REPO, path_in_repo=f"{TAG}/{h}", repo_type=TYPE, expand=True):
        if getattr(e, "size", None) is None:
            continue
        hf[e.path] = (int(e.size), e.lfs.sha256 if getattr(e, "lfs", None) else None)
man_paths = [p for p in rec if p.startswith("_backup/")]
for e in api.get_paths_info(REPO, man_paths, repo_type=TYPE, expand=True):
    hf[e.path] = (int(e.size), e.lfs.sha256 if getattr(e, "lfs", None) else None)
per_host = {}
for h in hosts:
    want = sorted(p for p in rec if p.startswith(f"{TAG}/{h}/"))
    got = sorted(p for p in hf if p.startswith(f"{TAG}/{h}/"))
    miss = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    bad = [p for p in want if p in hf and hf[p][0] != rec[p]]
    per_host[h] = {"files_recorded": len(want), "files_hf": len(got), "missing": miss, "extra": extra, "size_mismatch": bad,
                   "bytes_hf": sum(hf[p][0] for p in got)}
    if miss or extra or bad or len(want) != 20:
        res["problems"].append(f"{h}: missing {len(miss)} extra {len(extra)} size_mismatch {len(bad)} recorded {len(want)}")
# 3: manifest sha256 vs HF LFS sha256
man_cmp = {}
for h in hosts:
    mp = f"_backup/MANIFEST.{TAG}.{h}.tsv"
    if mp not in hf:
        res["problems"].append(f"{h}: manifest {mp} not on HF")
        continue
    f = hf_hub_download(REPO, mp, repo_type=TYPE, local_dir=f"{OUT}/dl")
    lines = open(f).read().splitlines()
    hdr = lines[0].split("\t")
    rows = [dict(zip(hdr, ln.split("\t"))) for ln in lines[1:] if ln.strip()]
    shacol = next((c for c in hdr if "sha" in c.lower()), None)
    n_cmp = n_bad = 0
    for r in rows:
        p = r.get("path") or r.get(hdr[0])
        rp = p if p.startswith(TAG) else f"{TAG}/{h}/{os.path.basename(p)}"
        if shacol and rp in hf and hf[rp][1]:
            n_cmp += 1
            if r[shacol] != hf[rp][1]:
                n_bad += 1
                res["problems"].append(f"{h}: sha256 mismatch {rp}")
    man_cmp[h] = {"manifest_cols": hdr, "rows": len(rows), "sha_col": shacol, "lfs_sha_compared": n_cmp, "lfs_sha_mismatch": n_bad}
# 4: small files streamed back
small = {}
for h in hosts:
    d = f"{OUT}/dl"
    ts = json.load(open(hf_hub_download(REPO, f"{TAG}/{h}/train_state.json", repo_type=TYPE, local_dir=d)))
    comp = open(hf_hub_download(REPO, f"{TAG}/{h}/COMPLETE", repo_type=TYPE, local_dir=d)).read().strip()
    ranks = sorted(int(os.path.basename(p)[4:-3]) for p in hf if p.startswith(f"{TAG}/{h}/rank"))
    small[h] = {"step": ts.get("step"), "COMPLETE": comp, "ranks": [ranks[0], ranks[-1]] if ranks else None,
                "host_of_first_rank_in_state": (ts.get("hosts") or [None] * 128)[ranks[0]] if ranks and ts.get("hosts") else None,
                "keys": sorted(ts.keys())[:40]}
    if ts.get("step") != 17850:
        res["problems"].append(f"{h}: train_state step {ts.get('step')}")
# 5: bundle
try:
    b = [e.path for e in api.list_repo_tree(BREPO, path_in_repo=TAG, repo_type=TYPE)]
    res["bundle_entries"] = b
except Exception as ex:  # noqa: BLE001
    res["problems"].append(f"bundle listing failed: {type(ex).__name__}")
# the watcher's recorded per-host VERIFY PASS
vp = {}
for h in hosts:
    try:
        v = json.load(open(VREC.format(h=h)))
        vp[h] = {"PASS": v.get("PASS"), "time": v.get("time"), "local_files": v.get("local_files"), "local_bytes": v.get("local_bytes"),
                 "sha_checked_vs_upload_cache": v.get("sha_checked_vs_upload_cache"), "sha_mismatch": len(v.get("sha_mismatch") or [])}
    except Exception as ex:  # noqa: BLE001
        vp[h] = {"error": type(ex).__name__}
        res["problems"].append(f"{h}: no watcher verify record")
res.update(per_host=per_host, manifest_vs_lfs=man_cmp, small_files=small, watcher_verify=vp,
           total_files_hf=sum(v["files_hf"] for v in per_host.values()), total_bytes_hf=sum(v["bytes_hf"] for v in per_host.values()))
res["PASS"] = not res["problems"]
with open(f"{OUT}/hf_files.tsv", "w") as f:
    f.write("path\tsize\tlfs_sha256\n")
    for p in sorted(hf):
        f.write(f"{p}\t{hf[p][0]}\t{hf[p][1] or ''}\n")
json.dump(res, open(f"{OUT}/VERIFY_17850.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("PASS", "problems", "total_files_hf", "total_bytes_hf", "private")}), flush=True)
print(json.dumps(man_cmp.get(hosts[0])), json.dumps(small.get(hosts[0]))[:600], flush=True)
sys.exit(0 if res["PASS"] else 3)
