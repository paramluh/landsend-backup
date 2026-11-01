"""hfb_up_stub.py: DRY-TEST stand-in for setup-logs/hfb/hfb_up.py (same CLI, NO network, never imports huggingface_hub) =
aloe-v2vox-raw-w14503/tests/hfb_up_stub.py + the throttled upload loop (--control) used beside training. Refuses any repo not
InternalCan/DRYRUN-*. Used only by tests/hf_dry.sh (HFLX_DRY=1 HFLX_UP=<this>).
  --verify-only --no-verify : build the symlink farm + _backup/<manifest> like hfb_up.py, exit 0
  --child                   : "upload" nothing; exit 1 once per farm whose name contains $HFLX_STUB_CHILD_FAIL_ONCE (tests the retry loop)
  --control F --no-verify   : (no --verify-only) farm, then wait while F holds 0 (hfb_up.py's upload loop: paused), "upload" for
                              $HFLX_STUB_UPLOAD_S s with an UPLOAD-START / -END event line in $HFLX_STUB_EVENTS; fails once like --child;
                              rc 4 if --require vanished
  --verify-only             : write the verify json (same name + keys as hfb_up.py); PASS unless --tag is listed in $HFLX_STUB_VFAIL; rc 0 / 3"""
import argparse
import json
import os
import socket
import sys
import time

ap = argparse.ArgumentParser()
for k in ("--repo", "--type", "--farm", "--spec", "--manifest-name", "--tag", "--require", "--verify-out", "--control", "--owner-job",
          "--readme", "--manifest-extra"):
    ap.add_argument(k, default="")
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--max-fails", type=int, default=0)
ap.add_argument("--file-limit", type=int, default=0)
for k in ("--verify-only", "--no-verify", "--child", "--defer-sample"):
    ap.add_argument(k, action="store_true")
a = ap.parse_args()
assert a.repo.startswith("InternalCan/DRYRUN-"), f"stub refuses real repo {a.repo}"
H = socket.gethostname().split(".")[0]
if not a.tag and a.spec:                                # --child calls carry no --tag (as hfb_up.py): the spec's dest prefix
    a.tag = json.load(open(a.spec))["dirs"][0][1]


def event(m):
    p = os.environ.get("HFLX_STUB_EVENTS")
    if p:
        with open(p, "a") as f:
            f.write(f"{time.time():.2f} {H} {a.tag} {m}\n")


def fail_once():
    tok = os.environ.get("HFLX_STUB_CHILD_FAIL_ONCE", "")
    mark = os.path.join(a.farm, ".stub_child_failed_once")
    if tok and tok in os.path.basename(a.farm) and not os.path.exists(mark):
        open(mark, "w").write("1")
        print(f"[stub] {H} upload: simulated upload failure (once) for {a.farm}", flush=True)
        return True
    return False


if a.child:
    if fail_once():
        sys.exit(1)
    event(f"CHILD-UPLOAD workers {a.workers}")
    print(f"[stub] {H} child: no upload (dry) for {a.farm} workers {a.workers}", flush=True)
    sys.exit(0)
spec = json.load(open(a.spec))
files = {}
for src, dest in spec["dirs"]:
    for root, _, fs in os.walk(src):
        for f in fs:
            rel = os.path.relpath(os.path.join(root, f), src)
            if f in spec.get("exclude", []) or f.endswith(".tmp"):
                continue
            files[os.path.join(dest, rel)] = os.path.join(root, f)
for rp, s in files.items():
    t = os.path.join(a.farm, rp)
    os.makedirs(os.path.dirname(t), exist_ok=True)
    if not os.path.lexists(t):
        os.symlink(os.path.realpath(s), t)
os.makedirs(os.path.join(a.farm, "_backup"), exist_ok=True)
with open(os.path.join(a.farm, "_backup", a.manifest_name), "w") as f:
    f.write("path\tsize\tsource\n" + "".join(f"{rp}\t{os.path.getsize(s)}\t{H}:{s}\n" for rp, s in sorted(files.items())))
print(f"[stub] {H} farm {a.farm}: {len(files) + 1} files (dry, no network)", flush=True)
if a.verify_only and a.no_verify:                      # repo create + farm only
    sys.exit(0)
if not a.verify_only:                                 # throttled upload loop (hfb_up.py upload() with --control)
    w, said = 0, False
    while True:
        if a.require and not os.path.exists(a.require):
            print(f"[stub] {H} ABORT: {a.require} disappeared", flush=True); sys.exit(4)
        try:
            w = int(open(a.control).read().strip()) if a.control else a.workers
        except Exception:
            w = a.workers
        if w > 0:
            break
        if not said:
            print(f"[stub] {H} control {a.control}: workers 0 -> paused", flush=True); event("PAUSED"); said = True
        time.sleep(0.5)
    if fail_once():
        sys.exit(1)
    event(f"UPLOAD-START workers {w}")
    print(f"[stub] {H} upload (dry, nothing sent) workers {w}", flush=True)
    time.sleep(float(os.environ.get("HFLX_STUB_UPLOAD_S", "1")))
    event("UPLOAD-END")
    if a.no_verify:
        sys.exit(0)
fail = a.tag in os.environ.get("HFLX_STUB_VFAIL", "").split(",")
big = sum(1 for rp in files if rp.endswith((".pt", ".distcp")))
res = {"repo": a.repo, "tag": a.tag, "type": a.type, "private": True, "host": H, "time": time.strftime("%F %T %Z"),
       "local_files": len(files) + 1, "n_missing": 1 if fail else 0, "missing_on_hf": [], "size_mismatch": [], "sha_mismatch": [],
       "sha_checked_vs_upload_cache": big, "PASS": not fail, "dry_stub": True,
       "sample_download_sha256": [{"path": "dry", "match": None, "deferred": True}]}
os.makedirs(a.verify_out, exist_ok=True)
json.dump(res, open(os.path.join(a.verify_out, a.repo.replace("/", "__") + f"__{a.tag}__{H}.json"), "w"), indent=1)
print(f"[stub] {H} VERIFY {'FAIL' if fail else 'PASS'} {a.tag} (dry)", flush=True)
sys.exit(3 if fail else 0)
