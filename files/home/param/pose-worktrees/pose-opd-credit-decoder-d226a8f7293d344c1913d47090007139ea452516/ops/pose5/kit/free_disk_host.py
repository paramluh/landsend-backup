"""pose5 launch kit, disk step, ONE host (run by launch_pose5.sh in a tiny CPU step on each of the 16 hosts AFTER MoT has stopped; never on
slinky-9). Main's rule (10-27, launch day only, after the user's go): free /scratch only by deleting HF-VERIFIED node-local items, never anything
without a PASS HF verify record, never any dcp-17850 copy, never a PROTECTED path (PROTECTED_SCRATCH.txt via scratch_guard; those need the
user's path-specific OK and are only PRINTED here).

  need  = npl5 share still to copy (B's PLACEMENT_PHYS.json GB minus what is already in /scratch/amartyavarma/npl-append/npl5)
        + 46.1 GB if /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17850 is not COMPLETE here (slinky-15: 0 if ppmvox/init-17850-slots is there)
        + 9.2 GB slot-9 pose store (slinky-15 only, if /scratch/amartyavarma/pose-s3-samdata/rows is absent)
        + GATE (122 GB = 30 floor + 46 save + 46 HF hold; coordinator rule)
  gap   = need - live free (df). Candidates = c/DISK_NEED.json hosts[<host>].candidates, tier "1_hf_verified" only, in the listed order.
--dry-run (default): prints free / need / gap and which candidates WOULD go (live du of each). --delete: deletes in order until the gap is
closed, each only after re-checking: path exists, no 'dcp-17850' / 'init-17850' / excluded store in it, every verify file present with
PASS true, scratch_guard guard_check OK; logs one line per deletion to pose5/kit/logs/DELETED-<date>.log; re-measures at the end.
Exit 0 = no gap left (or dry run); 3 = gap remains (printed)."""
import argparse, datetime, json, os, shutil, socket, subprocess, sys, zoneinfo

ap = argparse.ArgumentParser()
ap.add_argument("--delete", action="store_true")
ap.add_argument("--gate-gb", type=float, default=122.0)
ap.add_argument("--need-json", default="/data/home/amartyavarma/ppm/pose5/c/DISK_NEED.json")
ap.add_argument("--log", default="", help="deletion log (default pose5/kit/logs/DELETED-<date>.log); self-tests pass their own")
a = ap.parse_args()
H = socket.gethostname().split(".")[0]
assert H != "slinky-9", "never on slinky-9"
PT = zoneinfo.ZoneInfo("America/Los_Angeles")
now = lambda: datetime.datetime.now(PT).strftime("%m-%d %H:%M:%S PT")  # noqa: E731
LOG = a.log or f"/data/home/amartyavarma/ppm/pose5/kit/logs/DELETED-{datetime.datetime.now(PT):%Y%m%d}.log"
GB = 1e9
EXCL = ("dcp-17850", "init-17850", "/scratch/amartyavarma/s3-rows", "/scratch/amartyavarma/ppm/txtdup-full", "/scratch/amartyavarma/pose-s3-samdata",
        "/scratch/amartyavarma/ppm-fresh-pose", "/scratch/amartyavarma/new-pose-llm-data", "/scratch/amartyavarma/npl-append",
        "/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-16861", "/scratch/amartyavarma/ppm-mot", "/scratch/amartyavarma/inductor-cache-ppm/")


def free_b():
    return shutil.disk_usage("/scratch").free


def du_b(p):
    try:
        return int(subprocess.run(["du", "-sb", "--one-file-system", p], capture_output=True, text=True, timeout=600).stdout.split()[0])
    except Exception:
        return 0


D = json.load(open(a.need_json))
hd = D["hosts"][H]
share = float(hd["need"]["npl5_GB"]) * GB
have5 = du_b("/scratch/amartyavarma/npl-append/npl5") if os.path.isdir("/scratch/amartyavarma/npl-append/npl5") else 0
n_npl5 = max(0.0, share - have5)
dcp_ok = os.path.isfile("/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17850/COMPLETE") or \
    (H == "slinky-15" and os.path.isfile("/scratch/amartyavarma/ppmvox/init-17850-slots/dcp-17850/COMPLETE"))
n_dcp = 0.0 if dcp_ok else 46.09 * GB
n_pose = 9.2 * GB if (H == "slinky-15" and not os.path.isdir("/scratch/amartyavarma/pose-s3-samdata/rows")) else 0.0
need = n_npl5 + n_dcp + n_pose + a.gate_gb * GB
f0 = free_b()
gap = need - f0
print(f"DISK {H} free {f0 / GB:.1f} GB; need npl5 {n_npl5 / GB:.1f} + dcp-17850 {n_dcp / GB:.1f} + slot9-pose {n_pose / GB:.1f} + gate "
      f"{a.gate_gb:.0f} = {need / GB:.1f} GB -> gap {max(0, gap) / GB:.1f} GB", flush=True)
for c in hd.get("candidates", []):
    if c.get("tier") != "1_hf_verified":
        print(f"  (not deletable here: {c.get('tier')} {c['path']} {c.get('bytes', 0) / GB:.1f} GB{' ' + c['protected_by'] if c.get('protected_by') else ''})")
freed, out = 0, []
for c in [c for c in hd.get("candidates", []) if c.get("tier") == "1_hf_verified"]:
    p = c["path"]
    why = None
    if gap - freed <= 0:
        why = "gap closed"
    elif not os.path.exists(p):
        why = "absent"
    elif any(x in p for x in EXCL):
        why = "excluded path"
    else:
        for vf in c.get("verify_files", []):
            try:
                v = json.load(open(vf))
                if v.get("PASS") is not True:
                    why = f"verify not PASS: {vf}"
            except Exception as e:
                why = f"verify record unreadable: {vf} ({type(e).__name__})"
        if not c.get("verify_files"):
            why = "no verify file"
        g = subprocess.run(["bash", "-c", 'source /data/home/amartyavarma/ppm/scratch_guard.sh; guard_check "$1"', "_", p], capture_output=True, text=True)
        if g.returncode != 0:
            why = "PROTECTED: " + g.stderr.strip()[:200]
    if why:
        print(f"  skip {p}: {why}")
        continue
    sz = du_b(p)
    if not a.delete:
        print(f"  WOULD delete {p} ({sz / GB:.1f} GB; HF {c.get('hf_repo')}/{c.get('hf_path', '')}; verify {c['verify_files'][0]})")
        freed += sz
        continue
    fb = free_b()
    r = subprocess.run(["nice", "-n", "10", "ionice", "-c3", "bash", "-c", 'source /data/home/amartyavarma/ppm/scratch_guard.sh; guard_rm "${1:?}"', "_", p],
                       capture_output=True, text=True)
    fa = free_b()
    line = (f"[{now()}] pose5 launch --free-disk (main's rule 10-27, user go): {H} {'deleted' if r.returncode == 0 and not os.path.exists(p) else 'FAILED to delete'} "
            f"{p} ({sz} B, rc {r.returncode}{' ' + r.stderr.strip()[:150] if r.returncode else ''}) free {fb / GB:.1f} -> "
            f"{fa / GB:.1f} GB :: HF {c.get('hf_repo')}/{c.get('hf_path', '')} VERIFY PASS {';'.join(c['verify_files'])}")
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print("  " + line, flush=True)
    if r.returncode == 0:
        freed += fa - fb
f1 = free_b() if a.delete else f0 + freed
left = need - f1
print(f"DISK {H} {'after' if a.delete else 'would be after'}: free {f1 / GB:.1f} GB, need {need / GB:.1f} GB -> "
      f"{'OK' if left <= 0 else f'STILL SHORT {left / GB:.1f} GB'}", flush=True)
sys.exit(0 if left <= 0 else 3)                          # dry run: 3 = would still be short even after the tier-1 deletions
