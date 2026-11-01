"""npl5 ROLLING FETCHER, v2 (10-27 08:00 PT; audit round 1 fixes B2 B4 S10 S11 S12 N6). The hosts have no room for full npl5 shares and old
pose data may not be deleted (main GO), so one daemon per host keeps only a WINDOW of npl5 parquet node-local; the trainer reads the FULL npl5
plan (digests unchanged). One launch from dcp-17850 with npl5 appended at 17850: every rank first reads its npl3 leftover (~172-184 steps).
Runs as ONE light `srun --overlap` step inside the trainer job (bin/npl5_roll_super.sbatch), or once per host in 'window' mode (npl5_prep.sh).
  keep  = every npl5 file read by this host's 8 ranks at a step in [floor, target)
    live   = the trainer's position, from ITS job log (ppm-$SLURM_JOB_ID.log, else --log, else control/CURRENT_JOB_811): the newest of
             'resumed <src> at step R' (-> R) and '] step N/M ' (-> N); none yet -> the persisted value (else the oldest resume candidate)
    floor  = MONOTONIC, persisted node-local (<dest>/.roll-state.json): raised to the oldest resume candidate > START among COMPLETE saves in
             CKPT, the hold dir (ckpt-keep-pose5/hold) and the pin dir (ckpt-keep-pose5), never above the trainer; lowered ONLY by a 'resumed
             ... at step R' log line with R < floor (then [R, R + ahead) is refetched). A host that loses its CKPT saves keeps its floor (no
             re-download of consumed data). BACKMODE 'newest' was dropped (with keep=1 + the hold it saved nothing and was unsafe).
    target = min(live + AHEAD, END); AHEAD = kit/state/AHEAD-<host> > env NPL5_AHEAD > --ahead, re-read every pass
  fetch = keep files not present, earliest first, from the HF bucket (resolve -> CDN, as src_fetch.py), xet hash checked BEFORE the rename,
          unique temp names (<file>.roll.<pid>.<thread>); a file is fetched only if /scratch free after it stays >= MIN_FREE_GB (default
          66 = one 46 GB save + 20; kit/state/MIN_FREE_GB-<host> > env NPL5_MIN_FREE_GB), else NOSPACE
  drop  = present npl5 files of this host's plan whose LAST reading step < floor, path-checked to lie under <dest>/jobs and in the plan
  guards: one daemon per host (flock <dest>/.roll.lock); startup removes stale *.roll.* partials and xet-hashes every present plan file
          (mismatch -> deleted, refetched); memory: the trainer JOB cgroup's headroom = memory.max - (memory.current - file cache from
          memory.stat) < 6 GB -> LOWMEM: no fetch that pass (the step's own --mem=8G caps the daemon itself)
  status: W/roll/<host>.json rewritten every 30 s by a heartbeat thread (also inside long fetch passes / 429 pauses) with ahead_steps, live,
          floor, covered_until, GB_present, last_fetch_time, alerts[]; markers W/roll/<host>.{BEHIND <30, CRIT <20, STARVED, LOWMEM, NOSPACE}
          are set / cleared every pass. C's hold_chain alerts on markers and on a JSON older than 900 s.
Row -> step: step t, micro m of local rank l reads stream row (t - 16861) x 8 + m of fresh_pose.append_stream over [plan 1, npl2, npl3, npl5]
(tail 1); rows past END are never read. TRAINER-SIDE FAILURE MODE if a file is missing anyway: FileNotFoundError in that rank's DataLoader
worker -> the job dies (--kill-on-bad-exit=1); no row is skipped or substituted (no maths change); the watchdog relaunches from a save.
usage (qwen-s2 venv, python -I):  npl5_roll.py daemon|window|plan --end <PPM_MAX_STEPS> [..]   (see --help)"""
import argparse
import fcntl
import glob
import json
import os
import random
import re
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import numpy as np

CODE = os.environ.get("NPL5_CODE", "/data/home/amartyavarma/ppm/pose5/code-811-s2p")       # (the launch trainer's tree; fresh_pose identical)
W = os.environ.get("NPL5_W", "/data/home/amartyavarma/ppm/pose5/npl5")
KIT = os.environ.get("NPL5_KIT_STATE", "/data/home/amartyavarma/ppm/pose5/kit/state")
SF, NPM, LW = 16861, 8, 8
DIRS = ("ppm-fresh-pose", "ppm-fresh-pose-npl2", "ppm-fresh-pose-npl3", "ppm-fresh-pose-npl5")
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["daemon", "window", "plan"])
ap.add_argument("--end", type=int, required=True)
ap.add_argument("--upto", type=int, default=0)
ap.add_argument("--upto-rel", type=int, default=0, help="window mode: fetch the files read before (this host's first npl5 step + N)")
ap.add_argument("--ahead", type=int, default=100)
ap.add_argument("--low-water", type=int, default=30)
ap.add_argument("--crit", type=int, default=20)
ap.add_argument("--aheads", default="100,80,60,40", help="plan mode: WINDOW-<key>-a<ahead>-b<back>.json for each")
ap.add_argument("--alt-dir", default="")
ap.add_argument("--back", type=int, default=110, help="plan mode: steps kept behind the trainer (keep=1 save + a chained HF hold)")
ap.add_argument("--poll", type=float, default=120)
ap.add_argument("--hf-rate", type=float, default=20)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--start", type=int, default=17850)
ap.add_argument("--phys", default=socket.gethostname().split(".")[0])
ap.add_argument("--key", default="")
ap.add_argument("--root", default="/scratch/amartyavarma")
ap.add_argument("--ckpt-dir", default="")
ap.add_argument("--hold-dir", default="")
ap.add_argument("--pin-dir", default="")
ap.add_argument("--log", default="")
ap.add_argument("--audit13", action="store_true", help="rehearsal/plan only: plans 1-3 row counts from the Weka audit copies")
ap.add_argument("--no-startup-hash", action="store_true")
a = ap.parse_args()
P = a.phys
K = a.key or ("slinky-9" if P == "slinky-15" else P)
assert P != "slinky-9" and socket.gethostname().split(".")[0] != "slinky-9", "never on slinky-9"
CK = a.ckpt_dir or f"{a.root}/ppm/ckpt/ppm-v1"
HOLD = a.hold_dir or f"{a.root}/ppm/ckpt-keep-pose5/hold"
PIN = a.pin_dir or f"{a.root}/ppm/ckpt-keep-pose5"
R = os.environ.get("NPL5_ROLL_DIR", f"{W}/roll")
os.makedirs(R, exist_ok=True)
sys.path.insert(0, f"{CODE}/fresh")
import fresh_pose as FP  # noqa: E402

assert os.path.abspath(FP.__file__).startswith(CODE), FP.__file__


def pt():
    return time.strftime("%m-%d %H:%M:%S %Z", time.localtime())


def log(msg):
    line = f"[{pt()}] [roll {P}={K}] {msg}"
    print(line, flush=True)
    with open(f"{R}/{P}.log", "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- the host's stream -> per npl5 file: first / last step that reads it
if a.audit13:
    _ad = ["/data/home/amartyavarma/ppm/code-811/fresh/scan/plans"] + [f"/data/home/amartyavarma/ppm/code-811/fresh/append/{t}/scan/plans" for t in ("npl2", "npl3")]
    pl = [json.load(open(f"{d}/PLAN-{K}.json")) for d in _ad]
else:                                                              # plans 1-3: only host / n_rows / digest are needed (the trainer checks digests)
    pl = []
    for d in DIRS[:3]:
        _j = json.load(open(f"{a.root}/{d}/PLAN.json"))
        pl.append({k_: _j[k_] for k_ in ("host", "n_rows", "digest", "tag") if k_ in _j}); del _j
pl.append(FP.load_plan(f"{a.root}/{DIRS[3]}"))                     # npl5: full plan, digest-asserted
assert all(p["host"] == K for p in pl), ([p["host"] for p in pl], K)
p5 = pl[3]
DEST = os.path.dirname(p5["src_root"])
JOBS_LINK = f"{DEST}/jobs"                                         # the plan's canonical path (src_root)
JOBS = os.path.realpath(JOBS_LINK)                                 # where the files really are (s13: a symlink into npl5-weka/<host>/jobs)
assert p5.get("tag") == "npl5" and p5["src_root"].endswith("/npl-append/npl5/jobs"), (p5.get("tag"), p5["src_root"])
z = p5["z"]
nf = len(p5["files"])
first = np.full(nf, np.iinfo(np.int64).max, np.int64); last = np.full(nf, -1, np.int64)
npos = (a.end - SF) * NPM
for l in range(LW):
    s = FP.append_stream([int(p["n_rows"]) for p in pl], l, LW, NPM, SF, 1, ())[:npos]
    for q in np.flatnonzero((s >> FP.PLAN_SHIFT) == 3):
        r = int(s[q]) & ((1 << FP.PLAN_SHIFT) - 1)
        st = SF + int(q) // NPM
        fi = np.unique(z["seg_file"][z["row_seg"][z["row_off"][r]:z["row_off"][r + 1]]])
        first[fi] = np.minimum(first[fi], st); last[fi] = np.maximum(last[fi], st)
used = last >= 0
p5.pop("segment_ids", None)
lst = {}
for ln in open(f"{W}/launch/npl5/placement/{P if P != 'slinky-9' else 'slinky-15'}.tsv"):
    rel, sz, xh = ln.rstrip("\n").split("\t"); lst[rel] = (int(sz), xh)
F = [f"jobs/{f}" for f in p5["files"]]
FSET = set(F)
assert all(f in lst for f in F), "npl5 plan files missing from this host's placement"
SZ = np.array([lst[f][0] for f in F], np.int64)
FIRST5 = int(first[used].min()) if used.any() else a.end
log(f"v2: plans {[p['digest'] for p in pl]}; npl5 {nf} files / {SZ.sum() / 1e9:.1f} GB, {int(used.sum())} read before END {a.end} "
    f"(first npl5 step {FIRST5}); code {CODE}")
if a.mode == "plan":
    rk = sorted({int(SF + int(np.flatnonzero((FP.append_stream([int(p['n_rows']) for p in pl], l, LW, NPM, SF, 1, ()) >> FP.PLAN_SHIFT) == 3)[0]) // NPM) for l in range(LW)})
    print(f"first npl5 step on {K}: {FIRST5} ({FIRST5 - a.start} steps after {a.start}); per-rank first npl5 steps {rk}", flush=True)
    WD = a.alt_dir or R
    os.makedirs(WD, exist_ok=True)
    for ah in [int(x) for x in a.aheads.split(",") if x]:
        mx, mt = 0.0, a.start
        for t in range(a.start, a.end + 1, 2):
            k = used & (first < t + ah) & (last >= max(a.start, t - a.back))
            g = float(SZ[k].sum()) / 1e9
            if g > mx:
                mx, mt = g, t
        print(f"PEAK window ahead {ah} + back {a.back}: {mx:.2f} GB at trainer step {mt} on {K}; full share {SZ[used].sum() / 1e9:.1f} GB", flush=True)
        json.dump({"key": K, "ahead": ah, "back": a.back, "peak_window_GB": round(mx, 2), "peak_at_step": mt, "full_share_GB": round(float(SZ[used].sum()) / 1e9, 2),
                   "first_npl5_step": FIRST5, "end": a.end, "GB_per_step": round(float(SZ[used].sum()) / 1e9 / max(1, a.end - FIRST5), 4)},
                  open(f"{WD}/WINDOW-{K}-a{ah}-b{a.back}.json", "w"), indent=1)
    sys.exit(0)

# ---------------------------------------------------------------- dest checks, single daemon per host, startup hygiene
WEKA_DIR = os.environ.get("NPL5_WEKA_DIR", f"/data/home/amartyavarma/ppm/pose5/npl5-weka/{P}")
if os.path.islink(JOBS_LINK):                                      # (main 10-27: s13's window served from Weka through a symlink)
    want = os.path.realpath(f"{WEKA_DIR}/jobs")
    if JOBS != want or not want.startswith("/data/home/amartyavarma/ppm/pose5/npl5-weka/"):
        log(f"REFUSED: {JOBS_LINK} -> {JOBS}, allowed only {want} (under npl5-weka/)"); sys.exit(4)
elif os.path.exists(JOBS_LINK) and not JOBS.startswith("/scratch/"):
    log(f"REFUSED: {JOBS_LINK} resolves to {JOBS}, not under /scratch"); sys.exit(4)
ON_DATA = JOBS.startswith("/data/")
os.makedirs(JOBS, exist_ok=True)
LOCKF = open(f"{DEST}/.roll.lock", "w")
try:
    fcntl.flock(LOCKF, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    log("another npl5_roll daemon holds the host lock: exiting"); sys.exit(3)
import hf_xet  # noqa: E402
import requests  # noqa: E402


def path_ok(rel):
    """a delete / overwrite target: one of this host's npl5 plan files, whose REAL path lies inside the resolved dest root (s13: the Weka dir)."""
    p = os.path.realpath(f"{DEST}/{rel}")
    return rel in FSET and p.startswith(JOBS + "/")


def safe_remove(p):
    rp = os.path.realpath(p)
    assert rp.startswith(JOBS + "/"), ("refusing a delete outside the dest root", p, rp)
    os.remove(rp)


stale = [os.path.join(dp, f) for dp, _, fs in os.walk(JOBS) for f in fs if ".roll." in f or f.endswith(".roll")]
for f in stale:
    safe_remove(f)
if stale:
    log(f"removed {len(stale)} stale partial files")


def stat_ok(i):
    p = f"{DEST}/{F[i]}"
    return os.path.isfile(p) and os.path.getsize(p) == int(SZ[i])


have = np.array([stat_ok(i) for i in range(nf)], bool)            # in-memory presence; re-stat only the window each pass (Weka metadata)
if not a.no_startup_hash:                                          # size AND xet hash for every plan file already present
    pr0 = np.flatnonzero(have)
    bad = 0
    for c in range(0, len(pr0), 256):
        ch = [int(i) for i in pr0[c:c + 256]]
        for i, h in zip(ch, hf_xet.hash_files([f"{DEST}/{F[i]}" for i in ch])):
            if h.hash != lst[F[i]][1] and path_ok(F[i]):
                safe_remove(f"{DEST}/{F[i]}"); have[i] = False; bad += 1
    log(f"startup: dest {JOBS_LINK} -> {JOBS} ({'Weka /data' if ON_DATA else 'node-local'}); {len(pr0)} present files xet-checked, {bad} mismatching removed")

L = json.load(open(f"{W}/LISTING.json"))["src"]
TOKEN = os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
BASE = f"https://huggingface.co/buckets/{L['bucket']}/resolve/"
PFX = (L["prefix"] + "/") if L["prefix"] else ""
lock, tl = threading.Lock(), threading.local()
bucket, pause = [time.time()], [0.0]
STATE_F = f"{DEST}/.roll-state.json"
state = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {"floor": a.start, "live": None, "resume_seen": None}
ST = {"phys": P, "key": K, "pid": os.getpid(), "started": pt(), "phase": "init", "alerts": [], "fetched_in_pass": 0, "todo_in_pass": 0,
      "r429": 0, "dest": JOBS_LINK, "dest_real": JOBS, "dest_on_data": ON_DATA}
FAKE429 = float(os.environ.get("NPL5_FAKE_429", "0"))              # test only: probability that a resolve answers 429


def save_state():
    tmp = f"{STATE_F}.{os.getpid()}"
    json.dump(state, open(tmp, "w")); os.replace(tmp, STATE_F)


def knob(name, default, cast):
    try:
        return cast(open(f"{KIT}/{name}-{P}").read().split()[0])
    except Exception:  # noqa: BLE001
        v = os.environ.get(f"NPL5_{name}")
        return cast(v) if v else default


def free_gb():
    s_ = os.statvfs(JOBS)                                          # the filesystem the files really land on (s13: /data)
    return s_.f_bavail * s_.f_frsize / 1e9


def scratch_free_gb():
    s_ = os.statvfs(a.root)                                        # the node-local /scratch (always reported, for C's /scratch trend alert)
    return s_.f_bavail * s_.f_frsize / 1e9


def min_free_gb():
    return knob("MIN_FREE_GB", 350.0 if ON_DATA else 66.0, float)   # /data: keep >= 350 GB (main); node-local: one save + 20


def job_headroom_gb(cg_dir=None):
    """the trainer JOB cgroup's memory.max - (memory.current - reclaimable page cache), GB; page cache = memory.stat file - shmem
    (shmem / tmpfs, e.g. the DataLoader's /dev/shm, is not reclaimable); None outside a job / unlimited."""
    try:
        d = cg_dir or os.environ.get("NPL5_FAKE_CGROUP")
        if not d:
            c = open("/proc/self/cgroup").read().strip().split(":", 2)[2]
            if "/job_" not in c:
                return None
            i = c.index("/job_"); j = c.find("/", i + 1)
            d = "/sys/fs/cgroup" + (c[:j] if j > 0 else c)
        mx = open(f"{d}/memory.max").read().strip()
        if mx == "max":
            return None
        cur = int(open(f"{d}/memory.current").read())
        ms = dict((ln.split()[0], int(ln.split()[1])) for ln in open(f"{d}/memory.stat") if len(ln.split()) == 2)
        return round((int(mx) - (cur - (ms.get("file", 0) - ms.get("shmem", 0)))) / 1e9, 1)
    except Exception:  # noqa: BLE001
        return None


def sess():
    if not hasattr(tl, "s"):
        tl.s = requests.Session(); tl.s.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=4, max_retries=0))
    return tl.s


def slot():
    while True:
        with lock:
            now = time.time(); w = max(pause[0] - now, bucket[0] - now)
            if w <= 0:
                bucket[0] = max(bucket[0], now) + 1.0 / a.hf_rate; return
        time.sleep(min(w, 5) + random.random() * 0.05)


NOSPACE = [False]


def fetch(i):
    rel = F[i]; sz, xh = lst[rel]; dst = f"{DEST}/{rel}"
    if not path_ok(rel):
        return False
    if free_gb() - sz / 1e9 < min_free_gb():
        NOSPACE[0] = True
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for att in range(12):
        tmp = f"{dst}.roll.{os.getpid()}.{threading.get_ident()}"
        try:
            slot()
            if FAKE429 and random.random() < FAKE429:
                code = 429
            else:
                r = sess().get(BASE + quote(PFX + rel), headers={"Authorization": f"Bearer {TOKEN}"}, allow_redirects=False, timeout=60)
                code = r.status_code
            if code == 429:
                with lock:
                    pause[0] = max(pause[0], time.time() + (5 if FAKE429 else 60)); ST["r429"] += 1
                continue
            if code not in (301, 302, 303, 307, 308):
                raise IOError(f"resolve {code}")
            with sess().get(r.headers["location"], stream=True, timeout=120) as g:
                if g.status_code != 200:
                    raise IOError(f"cdn {g.status_code}")
                with open(tmp, "wb") as o:
                    for c in g.iter_content(4 << 20):
                        o.write(c)
            if os.path.getsize(tmp) != sz:
                raise IOError("size")
            if hf_xet.hash_files([tmp])[0].hash != xh:
                raise IOError("xet hash")
            os.replace(tmp, dst)
            with lock:
                have[i] = True; ST["fetched_in_pass"] += 1; ST["last_fetch_time"] = pt()
            return True
        except Exception as e:  # noqa: BLE001
            if os.path.exists(tmp):
                safe_remove(tmp)
            if att % 4 == 3:
                log(f"retry {att} {rel}: {type(e).__name__} {str(e)[:120]}")
            time.sleep(min(30, 2 + 2 * att) + random.random())
    return False


def candidates():
    """COMPLETE saves > START a relaunch could resume from: CKPT, the hold dir, the pin dir (dcp-N/COMPLETE)."""
    out = set()
    for root in (CK, HOLD, PIN):
        for d in glob.glob(f"{root}/dcp-*"):
            m = re.fullmatch(r"dcp-(\d+)", os.path.basename(d))
            if m and os.path.isfile(f"{d}/COMPLETE") and int(m.group(1)) > a.start:
                out.add(int(m.group(1)))
    return sorted(out)


def trainer_log():
    if a.log:
        return a.log
    j = os.environ.get("SLURM_JOB_ID")
    if not j:
        try:
            j = open("/data/home/amartyavarma/ppm/control/ppm-v1/CURRENT_JOB_811").read().split()[0]
        except Exception:  # noqa: BLE001
            return ""
    return f"/data/home/amartyavarma/ppm/logs/ppm-{j}.log"


def from_log():
    """-> (live step or None, resume step of the newest 'resumed ... at step R' line or None, its key) from the trainer's own log."""
    lg = trainer_log()
    if not lg or not os.path.isfile(lg):
        return None, None, None
    with open(lg, "rb") as f:
        f.seek(max(0, os.path.getsize(lg) - 2000000)); tail = f.read().decode("utf-8", "replace")
    st_ = [(m.start(), int(m.group(1))) for m in re.finditer(r"\] step (\d+)/(\d+) ", tail)]
    rs = [(m.start(), int(m.group(1))) for m in re.finditer(r"resumed \S+ at step (\d+)", tail)]
    res = rs[-1][1] if rs else None
    rkey = f"{lg}:{len(rs)}:{res}" if rs else None                  # a NEW resume event (new job log or a new line), even at the same step
    if st_ and (not rs or st_[-1][0] > rs[-1][0]):
        return st_[-1][1], res, rkey
    return res, res, rkey


def coverage(live):
    with lock:
        hv = have.copy()
    miss = used & ~hv & (last >= live) & (first < a.end)
    return int(np.maximum(first[miss], live).min()) if miss.any() else a.end


def alerts_for(live, covered, extra=()):
    al = list(extra)
    if covered < a.end and covered <= live + 1:
        al.append("STARVED")
    elif covered < a.end and covered - live < a.crit:
        al += ["CRIT", "BEHIND"]
    elif covered < a.end and covered - live < a.low_water:
        al.append("BEHIND")
    if ST.get("phase") == "fetching" and time.time() - ST.get("pass_started_epoch", time.time()) > 900:
        al.append("SLOWPASS")
    return al


def markers(al):
    for m_ in ("BEHIND", "CRIT", "STARVED", "LOWMEM", "NOSPACE", "SLOWPASS"):
        f_ = f"{R}/{P}.{m_}"
        if m_ in al:
            open(f_, "w").write(json.dumps({k_: ST.get(k_) for k_ in ("time", "live_step", "covered_until", "ahead_steps", "job_mem_headroom_GB",
                                                                     "scratch_free_GB", "dest_free_GB", "r429", "alerts")}) + "\n")
        elif os.path.exists(f_):
            os.remove(f_)


def write_status():
    with lock:
        ST["heartbeat"] = pt(); ST["heartbeat_epoch"] = round(time.time())
        tmp = f"{R}/{P}.json.{os.getpid()}.{threading.get_ident()}"
        json.dump(ST, open(tmp, "w"), indent=1, default=str); os.replace(tmp, f"{R}/{P}.json")


def heartbeat():
    """R2-9: every 30 s recompute the CURRENT numbers (trainer position from its log, coverage from the in-memory presence the fetch threads
    update, alerts + markers), also in the middle of a long / 429-slowed pass."""
    while True:
        time.sleep(30)
        try:
            live_l, _, _ = from_log()
            live = live_l if live_l is not None else (state["live"] if state["live"] is not None else a.start)
            cov = coverage(live)
            keep_extra = [x for x in ST.get("alerts", []) if x in ("LOWMEM", "NOSPACE")]
            with lock:
                ST.update(live_step=live, covered_until=cov, ahead_steps=cov - live, files_present=int(have.sum()),
                          GB_present=round(float(SZ[have].sum()) / 1e9, 2), scratch_free_GB=round(scratch_free_gb(), 1),
                          dest_free_GB=round(free_gb(), 1))
            al = alerts_for(live, cov, keep_extra)
            ST["alerts"] = al
            write_status(); markers(al)
        except Exception as e:  # noqa: BLE001
            print(f"heartbeat error {e}", flush=True)


def one_pass(upto=None):
    ahead = knob("AHEAD", a.ahead, int)
    live_l, res, rkey = from_log()
    if rkey is not None and rkey != state.get("resume_seen"):
        state["resume_seen"] = rkey
        if res < state["floor"]:
            log(f"trainer resumed at step {res} < floor {state['floor']}: floor lowered, window refetched from {res}")
            state["floor"] = res
    cand = candidates()
    live = live_l if live_l is not None else (state["live"] if state["live"] is not None else (min(cand) if cand else a.start))
    state["live"] = live
    if cand:
        state["floor"] = max(state["floor"], min(min(cand), live))   # monotonic up; never above a resume candidate or the trainer
    floor = state["floor"]
    save_state()
    target = min(upto if upto else live + ahead, a.end)
    hr = job_headroom_gb()
    lowmem = hr is not None and hr < float(os.environ.get("NPL5_MIN_HEADROOM_GB", "6"))
    keep = used & (first < target) & (last >= floor)
    for i in np.flatnonzero(keep):                                 # re-stat only the window (catches outside deletions; cheap on Weka)
        have[i] = stat_ok(int(i))
    todo = sorted(np.flatnonzero(keep & ~have), key=lambda i: first[i])
    with lock:
        ST.update(phase="fetching", pass_started=pt(), pass_started_epoch=time.time(), todo_in_pass=len(todo), fetched_in_pass=0,
                  live_step=live, floor=floor, target=target, ahead_cfg=ahead)
    t0 = time.time(); ok = 0; NOSPACE[0] = False
    if todo and not lowmem:
        with ThreadPoolExecutor(a.threads) as ex:
            ok = sum(ex.map(fetch, todo))
    drop = [int(i) for i in np.flatnonzero(have & used & (last < floor)) if path_ok(F[int(i)])]
    for i in drop:
        safe_remove(f"{DEST}/{F[i]}")
        have[i] = False
    live_e, _, rkey_e = from_log()                                 # audit R3: the trainer moved during the pass; never report an older step
    if live_e is not None and (rkey_e == rkey or rkey_e is None):
        live = max(live, live_e)
    elif live_e is not None:                                       # a new resume event during the pass: its own position, as read now
        live = live_e
    state["live"] = live; save_state()
    covered = coverage(live)
    extra = (["LOWMEM"] if lowmem else []) + (["NOSPACE"] if NOSPACE[0] else [])
    with lock:
        ST.update(phase="idle", time=pt(), live_step=live, floor=floor, target=target, covered_until=covered, ahead_steps=covered - live,
                  files_present=int(have.sum()), GB_present=round(float(SZ[have].sum()) / 1e9, 2), fetched=int(ok),
                  fetch_failed=int(len(todo) - ok) if not lowmem else 0, fetch_s=round(time.time() - t0, 1), dropped=len(drop),
                  scratch_free_GB=round(scratch_free_gb(), 1), dest_free_GB=round(free_gb(), 1), end=a.end, ahead_cfg=ahead, low_water=a.low_water, crit=a.crit, job_mem_headroom_GB=hr,
                  min_free_GB=min_free_gb(), resume_candidates=cand, resume_seen=state.get("resume_seen"), first_npl5_step=FIRST5)
    alerts = alerts_for(live, covered, extra)
    ST["alerts"] = alerts
    write_status(); markers(alerts)
    msg = (f"live {live}, floor {floor}, covered to {covered} ({covered - live} ahead, cfg {ahead}), {ST['files_present']} files "
           f"{ST['GB_present']} GB, fetched {ok}/{len(todo)} in {ST['fetch_s']} s, dropped {len(drop)}, dest free {ST['dest_free_GB']} GB "
           f"({'data' if ON_DATA else 'scratch'}), scratch free {ST['scratch_free_GB']} GB, headroom {hr}, 429s {ST['r429']}")
    log(("ROLL " + "+".join(alerts) + ": " if alerts else "ok: ") + msg)
    return ST


threading.Thread(target=heartbeat, daemon=True).start()
if a.mode == "window":
    if a.upto_rel:
        a.upto = FIRST5 + a.upto_rel
    st = one_pass(upto=a.upto or a.start + a.ahead)
    sys.exit(0 if st["covered_until"] >= min(a.upto or a.start + a.ahead, a.end) and st["fetch_failed"] == 0 and
             not (set(st["alerts"]) & {"LOWMEM", "NOSPACE"}) else 1)
while True:
    st = one_pass()
    if st["live_step"] >= a.end:
        log("trainer reached END: daemon exits"); break
    time.sleep(knob("POLL", a.poll, float))
