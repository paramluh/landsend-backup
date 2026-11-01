"""ttconvo-ft (10-21): combined HF check of export-N/ in the PRIVATE repo InternalCan/ppm-ttconvo-ft, recomputed from the per-host verify
jsons (hfb_up.py) + the HF listing (metadata only; nothing downloaded). Replaces the watcher's combined check, whose expected bytes came from
the .done markers' "bytes" = mawk printf %d clamped at 2147483647 per host (-> 16 x 2^31-1 = 34,359,738,352 B, a false MISMATCH).
Expected per host = verify local_files / local_bytes minus that host's _backup manifest (its HF size, which the verify matched).
PASS = repo private, every host's verify PASS (sizes, sha256 of every LFS file vs upload-time local sha256, 3 files streamed back),
rank0..rank{world-1}.pt + rep.pt + meta.json + COMPLETE all on HF, file count and bytes == the hosts' totals, every *.pt has an LFS sha256.
--ref M: also compare every rank<R>.pt / rep.pt size with export-M's (same keys / shapes / dtypes -> identical torch.save sizes).
--write: corrected HF_BACKUP-export-N.json (old check kept under "superseded_check"), fixed .done bytes, one BACKUP_LOG line per export."""
import argparse
import datetime
import json
import os
import zoneinfo

from huggingface_hub import HfApi

B = "/data/home/amartyavarma/ppm/ttconvo-ft"
HB = f"{B}/hf_backup"
REPO = "InternalCan/ppm-ttconvo-ft"
HOSTS = ("slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-8,slinky-9,slinky-10,slinky-11,slinky-12,slinky-13,slinky-14,"
         "slinky-16,slinky-17,slinky-19").split(",")
ap = argparse.ArgumentParser()
ap.add_argument("exports", nargs="+", type=int)
ap.add_argument("--ref", type=int, default=0)
ap.add_argument("--write", action="store_true")
a = ap.parse_args()
api = HfApi()
info = api.repo_info(REPO, repo_type="model")
now = datetime.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles")).strftime("%m-%d %H:%M:%S PT")


def listing(path):
    out = {}
    for e in api.list_repo_tree(REPO, path_in_repo=path, recursive=True, expand=True, repo_type="model"):
        if hasattr(e, "blob_id") and getattr(e, "size", None) is not None:
            out[e.path] = (e.size, e.lfs.sha256 if getattr(e, "lfs", None) else None)
    return out


man = listing("_backup")
ref = listing(f"export-{a.ref}") if a.ref else {}
res = {}
for n in a.exports:
    T = f"export-{n}"
    L = listing(T)
    hosts, ef, eb, bad = {}, 0, 0, []
    for h in HOSTS:
        v = json.load(open(f"{HB}/verify/{REPO.replace('/', '__')}__{T}__{h}.json"))
        mp = f"_backup/MANIFEST.{T}.{h}.tsv"
        ms = man.get(mp, (None,))[0]
        if ms is None:
            bad.append(f"{h}: manifest {mp} not on HF")
            ms = 0
        f_, b_ = v["local_files"] - 1, v["local_bytes"] - ms
        ok_h = bool(v["PASS"] and v["private"] and v["n_missing"] == 0 and not v["size_mismatch"] and not v["sha_mismatch"]
                    and all(s.get("match") is True for s in v["sample_download_sha256"]))
        if not ok_h:
            bad.append(f"{h}: verify json not clean")
        hosts[h] = {"files": f_, "bytes": b_, "verify_PASS": ok_h, "sha_checked": v["sha_checked_vs_upload_cache"], "verify_time": v["time"]}
        ef += f_; eb += b_
    world = 8 * len(HOSTS)
    want = {f"{T}/rank{r}.pt" for r in range(world)} | {f"{T}/rep.pt", f"{T}/meta.json", f"{T}/COMPLETE"}
    miss = sorted(want - set(L))
    extra = sorted(set(L) - want)
    nolfs = sorted(p for p in L if p.endswith(".pt") and not L[p][1])
    hb = sum(s for s, _ in L.values())
    if miss: bad.append(f"missing on HF: {miss[:5]} ({len(miss)})")
    if extra: bad.append(f"unexpected on HF: {extra[:5]} ({len(extra)})")
    if nolfs: bad.append(f"*.pt without LFS sha256: {nolfs[:5]}")
    if len(L) != ef or hb != eb: bad.append(f"listing {len(L)} files {hb} B != hosts {ef} files {eb} B")
    if not info.private: bad.append("repo NOT private")
    refcmp = None
    if ref:
        diff = [p for p in L if (p.endswith(".pt")) and ref.get(p.replace(T, f"export-{a.ref}"), (None,))[0] != L[p][0]]
        refcmp = {"ref": f"export-{a.ref}", "pt_files_compared": sum(p.endswith(".pt") for p in L), "size_differs": diff[:10],
                  "n_size_differs": len(diff)}
        if diff: bad.append(f"{len(diff)} *.pt sizes differ from export-{a.ref}")
    ok = not bad
    r = {"export": T, "PASS": ok, "problems": bad, "hf_files": len(L), "hf_bytes": hb, "expected_files": ef, "expected_bytes": eb,
         "private": bool(info.private), "hosts_verify_pass": sum(x["verify_PASS"] for x in hosts.values()), "ref_size_check": refcmp,
         "time": now}
    res[n] = r
    print(json.dumps(r))
    if a.write:
        sf = f"{HB}/HF_BACKUP-{T}.json"
        if os.path.isfile(sf):
            old = json.load(open(sf))
        else:
            old = {"repo": REPO, "repo_type": "model", "path": f"{T}/", "url": f"https://huggingface.co/{REPO}/tree/main/{T}", "step": n,
                   "format": "ppm-local-bf16-shards-v1", "hosts": {}}
        sup = {k: old.get(k) for k in ("PASS", "files", "bytes", "gb", "hf_listing", "time") if k in old}
        for h in HOSTS:
            dn = f"{HB}/state/{T}/{h}.done"
            if os.path.isfile(dn):
                d = json.load(open(dn))
                if d.get("bytes") != hosts[h]["bytes"]:
                    d["bytes_recorded_bad"] = d.get("bytes"); d["bytes"] = hosts[h]["bytes"]
                    json.dump(d, open(dn + ".tmp", "w"), indent=1); os.replace(dn + ".tmp", dn)
                old.setdefault("hosts", {})[h] = d
            try:
                old.setdefault("node_local", {})[h] = open(f"{HB}/state/{T}/{h}.local").read().strip()
            except Exception:
                pass
        old.update({"private": bool(info.private), "PASS": ok, "files": ef, "bytes": eb, "gb": round(eb / 1e9, 2),
                    "hosts_verified": r["hosts_verify_pass"], "hosts_total": len(HOSTS),
                    "hf_listing": {"files": len(L), "bytes": hb, "private": bool(info.private), "expected_files": ef, "expected_bytes": eb,
                                   "match": len(L) == ef and hb == eb},
                    "recheck": r, "time": now})
        if sup and "superseded_check" not in old:
            old["superseded_check"] = {**sup, "why": "expected bytes from .done 'bytes' = mawk %d clamp 2147483647 per host (bug, fixed in "
                                                      "hf_local_exports.sh 10-21); recomputed by tools/final285/hf_recheck.py"}
        json.dump(old, open(sf + ".tmp", "w"), indent=1); os.replace(sf + ".tmp", sf)
        with open(f"{B}/logs/BACKUP_LOG.txt", "a") as f:
            f.write(f"[{now}] {T} HF backup RECHECK {'PASS' if ok else 'FAIL'}: private {REPO}/{T}/ {len(L)} files {hb / 1e9:.2f} GB == 16 hosts' "
                    f"{ef} files {eb} B; {r['hosts_verify_pass']}/16 host verifies clean (size + LFS sha256 of every file + 3 streamed back)"
                    + (f"; *.pt sizes == export-{a.ref}" if refcmp and not refcmp["n_size_differs"] else "")
                    + (f"; problems {bad}" if bad else "") + f"; summary {sf}\n")
