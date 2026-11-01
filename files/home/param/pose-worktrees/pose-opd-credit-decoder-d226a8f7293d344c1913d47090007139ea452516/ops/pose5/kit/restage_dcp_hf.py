#!/usr/bin/env python3
"""pose5 watchdog v2 (main 10-27): RESTAGE an HF-verified save of THIS run onto THIS host from HF, sha256-checked; idempotent; never deletes,
moves or overwrites anything that exists (an existing destination that does not verify = ABORT, ask main).
Runs ON a training host (one task per host of a CPU-only srun, nice). Source = setup-logs/hfb records.jsonl {"kind": "dcp"|"dcp-pinned",
"step": N, "PASS": true, "shard_repo": R, "path": "dcp-N[-suf]/", "hosts": [...]} -> HF R/<path><this host>/ (the physical host that wrote it).
Destination (node-local preferred): /scratch/amartyavarma/pose5-restage/dcp-N when /scratch free >= save + 53 GB (next save's write room) +
the host floor (gate_rules: s13 20, others 30); else Weka /data/home/amartyavarma/ppm/pose5/restage/dcp-N/<host> when /data free >= save + 350.
Every LFS file: size + sha256 == HF; small files: size == HF and byte-equal to a second HF read; train_state.json step == N; COMPLETE last;
then the per-host link /scratch/amartyavarma/pose5-resume/dcp-N -> destination (ln -sfn on a LINK only; a real dir there = ABORT).
Prints one line "RESTAGE <host> <step> OK|STAGED|ABORT|DRY ..." and writes kit/state/restage/dcp-N-<host>.json.
usage: python -I restage_dcp_hf.py <step> [--dry-run] [--workers 16] [--test-root DIR --test-small]   (tests: spare only, small files only)"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import socket
import sys
import time

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
ap = argparse.ArgumentParser()
ap.add_argument("step", type=int)
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--host", default="", help="tests only: act as this host's HF dir")
ap.add_argument("--test-root", default="", help="tests only (spare): stage under <dir>/dcp-N, no link")
ap.add_argument("--test-small", action="store_true", help="tests only: small files only")
a = ap.parse_args()
assert not (a.test_small or a.host) or a.test_root, "--test-small / --host need --test-root"
N = a.step
H = socket.gethostname().split(".")[0]
assert H != "slinky-9", "never on slinky-9"
HD = a.host or H
K = "/data/home/amartyavarma/ppm/pose5/kit"
HFB = "/data/home/amartyavarma/setup-logs/hfb"
sys.path.insert(0, K)
import gate_rules as GR  # noqa: E402
REC = {"host": H, "hf_dir_host": HD, "step": N, "time": time.strftime("%Y-%m-%d %H:%M:%S %Z")}


def out(status, **kw):
    REC.update(status=status, **kw)
    os.makedirs(f"{K}/state/restage", exist_ok=True)
    if not a.dry_run:
        json.dump(REC, open(f"{K}/state/restage/dcp-{N}-{H}{'-TEST' if a.test_root else ''}.json", "w"), indent=1)
    print(f"RESTAGE {H} {N} {status} " + json.dumps(kw)[:400], flush=True)
    sys.exit(0 if status in ("OK", "STAGED", "DRY") else 2)


recs = [json.loads(l) for l in open(f"{HFB}/records.jsonl") if l.strip()]
rs = [r for r in recs if r.get("kind") in ("dcp", "dcp-pinned") and r.get("step") == N and r.get("PASS") is True]
if not rs:
    out("ABORT", why=f"no PASS hfb dcp record for step {N}")
r = rs[-1]
REPO, PATH = r.get("shard_repo") or r["repo"], r["path"].rstrip("/") + "/"
hosts = r["hosts"] if isinstance(r["hosts"], list) else json.loads(r["hosts"].replace("'", '"'))
if HD not in hosts:
    out("ABORT", why=f"{HD} has no folder in dcp-{N} ({REPO} {PATH}; hosts {hosts})")
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
want = {}
for e in HfApi().list_repo_tree(REPO, path_in_repo=f"{PATH}{HD}", recursive=True, expand=True, repo_type="model"):
    if getattr(e, "size", None) is not None:
        want[e.path.rsplit("/", 1)[1]] = (int(e.size), e.lfs.sha256 if getattr(e, "lfs", None) else None)
if "COMPLETE" not in want or "train_state.json" not in want:
    out("ABORT", why=f"HF {REPO}/{PATH}{HD} lacks COMPLETE / train_state.json ({len(want)} files)")
if a.test_small:
    want = {f: v for f, v in want.items() if not v[1]}
total = sum(s for s, _ in want.values()) / 1e9
if a.test_root:
    DEST = f"{a.test_root.rstrip('/')}/dcp-{N}"
else:
    sfree = shutil.disk_usage("/scratch").free / 1e9
    dfree = shutil.disk_usage("/data").free / 1e9
    if sfree >= total + 53.0 + GR.floor(H):
        DEST = f"/scratch/amartyavarma/pose5-restage/dcp-{N}"
    elif dfree >= total + 350:
        DEST = f"/data/home/amartyavarma/ppm/pose5/restage/dcp-{N}/{H}"
    else:
        out("ABORT", why=f"no room: /scratch {sfree:.1f} GB < {total:.1f} + 53 + floor, /data {dfree:.0f} GB < {total:.1f} + 350")
REC.update(repo=REPO, hf_path=f"{PATH}{HD}", dest=DEST, files=len(want), GB=round(total, 2))
TMP = DEST + ".restagetmp"
LINK = f"/scratch/amartyavarma/pose5-resume/dcp-{N}"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def check_file(d, f):
    p = f"{d}/{f}"
    if not os.path.isfile(p):
        return f, "missing"
    size, lfs = want[f]
    if os.path.getsize(p) != size:
        return f, f"size {os.path.getsize(p)} != {size}"
    if lfs and sha(p) != lfs:
        return f, "sha256 != HF LFS"
    return f, None


def verify(d):
    with cf.ThreadPoolExecutor(a.workers) as ex:
        bad = [(f, e) for f, e in ex.map(lambda f: check_file(d, f), sorted(want)) if e]
    if not a.test_small or "train_state.json" in want:
        try:
            if json.load(open(f"{d}/train_state.json")).get("step") != N:
                bad.append(("train_state.json", "step != N"))
        except Exception as e:
            bad.append(("train_state.json", str(e)[:80]))
    return bad


def link():
    if a.test_root:
        return "test: no link"
    if os.path.exists(LINK) and not os.path.islink(LINK):
        out("ABORT", why=f"{LINK} exists and is not a symlink (never replaced)")
    os.makedirs(os.path.dirname(LINK), exist_ok=True)
    tmpl = LINK + ".new"
    if os.path.islink(tmpl):
        os.unlink(tmpl)
    os.symlink(DEST, tmpl); os.replace(tmpl, LINK)
    return f"{LINK} -> {DEST}"


t0 = time.time()
if os.path.isdir(DEST):
    if a.dry_run:
        out("DRY", plan=f"verify existing {DEST}")
    bad = verify(DEST)
    if bad:
        out("ABORT", why=f"existing {DEST} does not verify (never replaced: ask main)", bad=bad[:6])
    out("OK", how="existing restage verified (size + sha256 == HF)", link=link(), secs=round(time.time() - t0))
if a.dry_run:
    out("DRY", plan=f"download {REPO}/{PATH}{HD} ({len(want)} files, {total:.1f} GB) -> {TMP} -> sha256 -> {DEST} -> link {LINK}")
if os.path.exists(TMP) and not os.path.isdir(TMP):
    out("ABORT", why=f"{TMP} exists and is not a dir")
os.makedirs(TMP, exist_ok=True)


def dl(f):
    if check_file(TMP, f)[1] is None:
        return f, "reused"
    p = hf_hub_download(REPO, f"{PATH}{HD}/{f}", repo_type="model", local_dir=f"{TMP}/.dl")
    os.replace(p, f"{TMP}/{f}" if f != "COMPLETE" else f"{TMP}/COMPLETE.part")
    return f, "ok"


order = sorted([f for f in want if f != "COMPLETE"], key=lambda f: -want[f][0])
with cf.ThreadPoolExecutor(a.workers) as ex:
    list(ex.map(dl, order))
if "COMPLETE" in want and not os.path.exists(f"{TMP}/COMPLETE"):
    dl("COMPLETE")
bad = [b for b in verify(TMP) if b[0] != "COMPLETE"]
if bad:
    out("ABORT", why=f"downloaded files do not verify in {TMP} (kept for inspection)", bad=bad[:6])
if os.path.exists(f"{TMP}/COMPLETE.part"):
    os.replace(f"{TMP}/COMPLETE.part", f"{TMP}/COMPLETE")
bad = verify(TMP)
if bad:
    out("ABORT", why="final verify failed", bad=bad[:6])
if os.path.isdir(f"{TMP}/.dl"):
    shutil.rmtree(f"{TMP}/.dl")                                  # this script's own download cache only
if os.path.exists(DEST):
    out("ABORT", why=f"{DEST} appeared while staging (another stager?)")
os.makedirs(os.path.dirname(DEST), exist_ok=True)
os.rename(TMP, DEST)
json.dump({"step": N, "repo": REPO, "hf_path": f"{PATH}{HD}", "files": len(want), "verified": "size + sha256 == HF LFS", "time": REC["time"]},
          open(f"{DEST}.RESTAGE_VERIFIED.json", "w"))          # NEXT TO the save dir (the save dir holds exactly the HF files)
out("STAGED", how=f"HF {REPO} download + verify", link=link(), secs=round(time.time() - t0))
