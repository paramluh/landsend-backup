"""pose5 launch kit: put the 8,1,1 pause save dcp-17850 of ONE logical slot on THIS host at the trainer's resume path, verified; idempotent.
Runs ON the training host (inside the CPU-only pose5-stage job, nice). Never deletes, moves or overwrites anything that exists: an existing
dcp-17850 that does not verify is an ABORT (ask main), never replaced.

  DEST = /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17850   (20 files: rank<8i..8i+7>.pt, rep.pt, digest-*.txt, train_state.json, COMPLETE)
  expected = the HF listing recorded by pose5 C (c/hf17850/hf_files.tsv: size + LFS sha256 of every file of InternalCan/ppm-v1-c dcp-17850/<dir>/,
             checked 10-27 against the watcher's upload records: 320 files, 737.404 GB, all sizes ==) + the small files fetched from HF
             (c/hf17850/dl/dcp-17850/<dir>/*: byte-compared).
  <dir> on HF = the PHYSICAL host that wrote the save: slot slinky-10 -> slinky-7 (27653 placement), every other slot -> its own name (slot
             slinky-9 -> slinky-9, Faisal's node: its files come from HF only, never from slinky-9).
Order of sources: 1) DEST already there -> verify (size + sha256 of the 9 LFS files, bytes of the 11 small files) -> OK / ABORT;
  2) slot slinky-9 only: /scratch/amartyavarma/ppmvox/init-17850-slots/dcp-17850 (PPMVOX v5's 10-25 staging, 19 files sha256 == HF LFS) ->
     verify again -> hard-link (0 bytes) into DEST.pose5tmp -> rename to DEST;
  3) download from HF InternalCan/ppm-v1-c into DEST.pose5tmp/.dl (16 threads) -> sha256 == HF LFS -> COMPLETE last -> rename to DEST.
Staged files get the ORIGINAL save time (10-22 22:29:33 PT = the node-local save of job 27653) as mtime, so the HF watcher's lineage check
sees the recorded save (no needless 737 GB re-upload as dcp-17850-j<job>); nothing else depends on mtimes.
usage: python -I stage_dcp17850.py <logical slot> [--dry-run] [--workers 16]   -> prints one line "DCP17850 <host> <slot> OK|STAGED|ABORT ..."
and writes /data/home/amartyavarma/ppm/pose5/kit/state/dcp17850-<host>.json"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import socket
import sys
import time

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
ap = argparse.ArgumentParser()
ap.add_argument("slot")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--test-root", default="", help="SELF-TEST only (spare node): stage under <dir>/dcp-17850 instead of the trainer path")
ap.add_argument("--test-small", action="store_true", help="SELF-TEST only: the 11 small files (no rank / rep shards)")
a = ap.parse_args()
assert not a.test_small or a.test_root, "--test-small needs --test-root"
STEP, REPO, TYPE = 17850, "InternalCan/ppm-v1-c", "model"
ORIG_MTIME = 1792733373                         # 2026-10-23 05:29:33 UTC = 10-22 22:29:33 PT ("[save] .../dcp-17850 ... (41.1s)" in ppm-27653.log)
C = "/data/home/amartyavarma/ppm/pose5/c/hf17850"
DEST = f"{a.test_root.rstrip('/')}/dcp-{STEP}" if a.test_root else f"/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-{STEP}"
TMP = DEST + ".pose5tmp"
V5 = "/scratch/amartyavarma/ppmvox/init-17850-slots/dcp-17850"
SLOTS = ["slinky-0", "slinky-1", "slinky-2", "slinky-3", "slinky-5", "slinky-6", "slinky-8", "slinky-9", "slinky-10", "slinky-11", "slinky-12",
         "slinky-13", "slinky-14", "slinky-16", "slinky-17", "slinky-19"]
H = socket.gethostname().split(".")[0]
assert H != "slinky-9", "never on slinky-9"
assert a.slot in SLOTS, a.slot
HFDIR = {"slinky-10": "slinky-7"}.get(a.slot, a.slot)
i0 = 8 * SLOTS.index(a.slot)
want = {}
for ln in open(f"{C}/hf_files.tsv").read().splitlines()[1:]:
    p, s, sha = ln.split("\t")
    if p.startswith(f"dcp-{STEP}/{HFDIR}/"):
        want[p.rsplit("/", 1)[1]] = (int(s), sha or None)
exp_names = {f"rank{r}.pt" for r in range(i0, i0 + 8)} | {f"digest-{r}.txt" for r in range(i0, i0 + 8)} | \
    {"rep.pt", "digest-rep.txt", "train_state.json", "COMPLETE"}
assert set(want) == exp_names and len(want) == 20, ("recorded HF listing != the slot's 20 files", sorted(set(want) ^ exp_names))
if a.test_small:
    want = {f: v for f, v in want.items() if not v[1]}
SMALL = f"{C}/dl/dcp-{STEP}/{HFDIR}"
REC = {"host": H, "slot": a.slot, "hf_dir": HFDIR, "ranks": [i0, i0 + 7], "dest": DEST, "time": time.strftime("%Y-%m-%d %H:%M:%S %Z")}


def out(status, **kw):
    REC.update(status=status, **kw)
    os.makedirs("/data/home/amartyavarma/ppm/pose5/kit/state", exist_ok=True)
    if not a.dry_run:
        json.dump(REC, open(f"/data/home/amartyavarma/ppm/pose5/kit/state/dcp17850-{H}{'-TEST' if a.test_root else ''}.json", "w"), indent=1)
    print(f"DCP17850 {H} {a.slot} {status} " + json.dumps(kw)[:400], flush=True)
    sys.exit(0 if status in ("OK", "STAGED", "DRY") else 2)


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
    if lfs:
        return (f, None) if sha(p) == lfs else (f, "sha256 != HF LFS")
    ref = f"{SMALL}/{f}"
    if not os.path.isfile(ref):
        return f, "no HF reference copy of the small file"
    return (f, None) if open(p, "rb").read() == open(ref, "rb").read() else (f, "bytes != HF")


def verify(d):
    with cf.ThreadPoolExecutor(a.workers) as ex:
        bad = [(f, e) for f, e in ex.map(lambda f: check_file(d, f), sorted(want)) if e]
    extra = sorted(set(os.listdir(d)) - set(want) - {".dl"}) if os.path.isdir(d) else []
    ts = json.load(open(f"{d}/train_state.json")) if os.path.isfile(f"{d}/train_state.json") else {}
    if ts.get("step") != STEP or ts.get("hosts", [None] * 128)[i0:i0 + 8] != [a.slot] * 8:
        bad.append(("train_state.json", f"step {ts.get('step')} hosts {ts.get('hosts', [])[i0:i0 + 8]}"))
    return bad, extra


t0 = time.time()
if os.path.isdir(DEST):                                          # 1. already there (the 4 hosts D1 never reached: slinky-1/2/7/14 per records)
    if a.dry_run:
        out("DRY", plan=f"verify existing {DEST} (20 files, sha256 of 9 LFS files ~20-30 s)")
    bad, extra = verify(DEST)
    if bad or extra:
        out("ABORT", why=f"existing {DEST} does not verify (never replaced by this script: ask main)", bad=bad[:6], extra=extra[:6])
    out("OK", how="existing node-local copy verified (size + sha256 == HF LFS, small files == HF)", secs=round(time.time() - t0))
if os.path.exists(TMP) and not os.path.isdir(TMP):
    out("ABORT", why=f"{TMP} exists and is not a dir")
src = None
if a.slot == "slinky-9" and os.path.isfile(f"{V5}/COMPLETE"):  # 2. PPMVOX v5's verified slot-9 staging (slinky-15)
    if a.dry_run:
        out("DRY", plan=f"verify {V5} then hard-link its 20 files into {DEST} (0 bytes); else HF download 46.1 GB")
    bad, _ = verify(V5)
    if not bad:
        src = V5
if a.dry_run:
    out("DRY", plan=f"download {REPO}/dcp-{STEP}/{HFDIR}/ (20 files, 46.1 GB) -> {TMP} -> sha256 -> {DEST}")
os.makedirs(TMP, exist_ok=True)
how = ""
if src:
    for f in want:
        if not os.path.exists(f"{TMP}/{f}"):
            os.link(f"{src}/{f}", f"{TMP}/{f}")
    how = f"hard links of {src} (verified == HF LFS)"
else:                                                            # 3. HF download
    from huggingface_hub import hf_hub_download

    def dl(f):
        if check_file(TMP, f)[1] is None:
            return f, "reused"
        t = time.time()
        p = hf_hub_download(REPO, f"dcp-{STEP}/{HFDIR}/{f}", repo_type=TYPE, local_dir=f"{TMP}/.dl")
        os.replace(p, f"{TMP}/{f}")
        return f, f"{want[f][0] / 1e9:.2f} GB {time.time() - t:.0f}s"
    order = sorted([f for f in want if f != "COMPLETE"], key=lambda f: -want[f][0])
    with cf.ThreadPoolExecutor(a.workers) as ex:
        log = list(ex.map(dl, order))
    if not os.path.exists(f"{TMP}/COMPLETE"):
        os.replace(hf_hub_download(REPO, f"dcp-{STEP}/{HFDIR}/COMPLETE", repo_type=TYPE, local_dir=f"{TMP}/.dl"), f"{TMP}/COMPLETE.part")
    how = f"HF {REPO} download ({round(time.time() - t0)} s)"
bad, extra = verify(TMP) if src is None else ([], [])
bad = [b for b in bad if b[0] != "COMPLETE"] if src is None else bad
if bad:
    out("ABORT", why=f"staged files do not verify in {TMP} (kept for inspection)", bad=bad[:6])
if src is None:
    os.replace(f"{TMP}/COMPLETE.part", f"{TMP}/COMPLETE") if os.path.exists(f"{TMP}/COMPLETE.part") else None
    bad, extra = verify(TMP)
    if bad:
        out("ABORT", why="COMPLETE / final verify failed", bad=bad[:6])
if os.path.isdir(f"{TMP}/.dl"):
    import shutil
    shutil.rmtree(f"{TMP}/.dl")                                  # this script's own download cache only
if src is None:                                                  # original save time (hard links share the source's inode: leave them)
    for f in want:
        os.utime(f"{TMP}/{f}", (ORIG_MTIME, ORIG_MTIME))
    os.utime(f"{TMP}/COMPLETE", (ORIG_MTIME + 5, ORIG_MTIME + 5))
if os.path.exists(DEST):
    out("ABORT", why=f"{DEST} appeared while staging (another stager?)")
os.rename(TMP, DEST)
out("STAGED", how=how, secs=round(time.time() - t0))
