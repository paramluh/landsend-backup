"""hf_backup_watch.py : (sbatch job ttconvo-hfb-watch on slinky-15, 1 CPU, no GPU; hf_backup_watch.sbatch) HF-ONLY backup (NEVER GCS) of
the ttconvo-ft node-local weight exports DURING training, into the PRIVATE repo InternalCan/ppm-ttconvo-ft, path export-<N>/.
Every TTC_POLL_S (60 s):
 1. reads plan/EXPORT_STEPS.txt (comma-separated export steps; the last one = the final export) and RUNNING_JOB.txt (training job id,
    overwritten by relaunches). The job is used ONLY if squeue shows it RUNNING, named ttconvo-ft, on exactly the 16 hosts - never any
    other job (ppm / ppm811 / demos ...).
 2. COMPLETE probe: a 1-CPU, CPU-only overlap step of that job on slinky-0 (nice 19 / ionice idle) lists exports/export-*/COMPLETE; every
    60 s once the job's log shows it at / past a pending export step (or an [export-local] line for it), else every 10 min.
 3. the oldest export of the plan that is COMPLETE (or already verified on slinky-0) and not verified on all 16 hosts -> hf_local_exports.sh
    as ONE overlap step:  srun --jobid=<job> --overlap -N16 --ntasks-per-node=1 -c 4 --mem=40G --cpu-bind=none --gres=none
    nice -n 19 ionice -c3 (per host: hfb_up.py farm -> upload_large_folder -> verify; guarded deletion of an intermediate export).
    One upload at a time; hosts already verified are skipped by the script, so a relaunch only redoes what is missing.
 4. step-time throttle while it runs (= setup-logs/hfb/hfb_throttle.py's rules, every TTC_CTL_S = 15 s) through control/<host>.workers
    (hfb_up.py --control: 0 = paused): uploads paused until BASE = median of this job's logged steps 4..15; then N hosts with pending work
    get 2 workers, the rest 0:  newest step > BASE + 5 s -> N = 0 + 10 min cool-off;  median of last 3 > BASE + 1 s -> N //= 2;
    median of last 5 < BASE + 0.4 s and >= 3 min since the last change -> N += 1 (N < 2) or += 2, max 16. N is kept per job and only ramps
    while an upload is active. No BASE 15 min after an upload started (log unreadable / too few steps) -> fixed N = 2 (logged).
 5. every host .done -> combined HF listing check of export-N/ (count + bytes vs the 16 hosts' records, repo private) ->
    hf_backup/HF_BACKUP-export-N.json + one BACKUP_LOG line.
 6. training job gone (not running, no ttconvo-ft job pending) for 10 min and an export still not on HF that was seen COMPLETE (probe) or
    has its [export-local] log line -> fallback: hf_local_exports.sh as a normal CPU-only job (sbatch -p all --gres=none -N16 -w <hosts>
    --ntasks-per-node=1 -c 4 --mem=40G), only once the 16 nodes are free (no GPU job, 4 CPUs + 40G free each, no pending, non-held job
    of mine waiting for them); a fallback job still pending after 30 min or when a ttconvo-ft job starts pending is cancelled (scancel).
 Exit: final export verified on all 16 hosts and every export handled (rc 0; rc 1 if an export gave up after TTC_MAX_LAUNCH = 6
 launches); or the job gone for TTC_GONE_EXIT_S (2 h) with no COMPLETE export pending (rc 2 if the final export never appeared).
Clean stop: touch hf_backup/state/STOP_WATCH (or scancel the watcher job).
Log: logs/hfb_watch.log; results: logs/BACKUP_LOG.txt; state: hf_backup/state/ (watch.json, base-<job>, export-N/<host>.{done,end,local}).
Dry test (tests/hf_dry.sh): HFLX_DRY=1 HFLX_SHARED=<ttconvo-ft/tests/dry-*> HFLX_ROOT=/scratch/amartyavarma/hflx-dry-ttconvo* HFLX_HOSTS=a,b
HFLX_JOBMAP=a:<job>,b:<job> HFLX_JOBNAME=<name of a's job> HFLX_TRAIN_LOG=<fake log> HFLX_UP=<stub>: same loop, uploads nothing."""
import datetime
import getpass
import json
import os
import re
import subprocess
import sys
import time
import zoneinfo

B = "/data/home/amartyavarma/ppm/ttconvo-ft"
PT = zoneinfo.ZoneInfo("America/Los_Angeles")
E = os.environ.get
DRY = E("HFLX_DRY") == "1"
HOSTS16 = ("slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-8,slinky-9,slinky-10,slinky-11,slinky-12,slinky-13,slinky-14,"
           "slinky-16,slinky-17,slinky-19").split(",")
if DRY:
    SHD, ROOT = E("HFLX_SHARED", ""), E("HFLX_ROOT", "")
    assert SHD.startswith(f"{B}/tests/dry-") and ROOT.startswith("/scratch/amartyavarma/hflx-dry-ttconvo"), ("bad dry dirs", SHD, ROOT)
    LD, HB, PLAN, RJ = f"{SHD}/logs", f"{SHD}/hf_backup", f"{SHD}/plan/EXPORT_STEPS.txt", f"{SHD}/RUNNING_JOB.txt"
    EXR, HOSTS, REPO = f"{ROOT}/exports", E("HFLX_HOSTS").split(","), "InternalCan/DRYRUN-no-upload"
    JOBNAME, TRAIN_LOG = E("HFLX_JOBNAME"), E("HFLX_TRAIN_LOG")
    JOBMAP = dict(x.split(":") for x in E("HFLX_JOBMAP").split(","))
else:
    LD, HB, PLAN, RJ = f"{B}/logs", f"{B}/hf_backup", f"{B}/plan/EXPORT_STEPS.txt", f"{B}/RUNNING_JOB.txt"
    EXR, HOSTS, REPO = "/scratch/amartyavarma/ttconvo-ft/exports", HOSTS16, "InternalCan/ppm-ttconvo-ft"
    JOBNAME, TRAIN_LOG, JOBMAP = "ttconvo-ft", None, None
SCRIPT = f"{B}/hf_local_exports.sh"
POLL_S = float(E("TTC_POLL_S", "60"))
CTL_S = float(E("TTC_CTL_S", "15"))
PROBE_IDLE_S = float(E("TTC_PROBE_IDLE_S", "600"))
W = int(E("TTC_W", "2"))                     # upload workers per active host (PPM HFB_W)
NMAX = len(HOSTS)
SLOW, HALVE, OKM = 5.0, 1.0, 0.4             # PPM: zero above base + 5 s, halve above base + 1 s, ramp below base + 0.4 s
RAMP_S = float(E("TTC_RAMP_S", "180"))
COOL_S = float(E("TTC_COOL_S", "600"))
NOBASE_S, NOBASE_N = float(E("TTC_NOBASE_S", "900")), 2
MAX_LAUNCH = int(E("TTC_MAX_LAUNCH", "6"))
FALLBACK_WAIT_S = float(E("TTC_FALLBACK_WAIT_S", "600"))
FB_PEND_MAX_S = float(E("TTC_FB_PEND_MAX_S", "1800"))
GONE_EXIT_S = float(E("TTC_GONE_EXIT_S", "7200"))
STEP_RE = re.compile(r"^\[[^]]+\] step (\d+)/\d+ ([0-9.]+)s ")           # = hfb_throttle.py's step-time regex
EXP_RE = re.compile(r"\[export-local\] \S*/export-(\d+) on every node")
USER = getpass.getuser()
os.makedirs(f"{HB}/state", exist_ok=True); os.makedirs(f"{HB}/control", exist_ok=True); os.makedirs(LD, exist_ok=True)
SUMF = lambda n: f"{HB}/HF_BACKUP-export-{n}.json"                       # noqa: E731


def now_pt():
    return datetime.datetime.now(PT).strftime("%m-%d %H:%M:%S PT")


def log(m):
    with open(f"{LD}/hfb_watch.log", "a") as f:
        f.write(f"[{now_pt()}] {m}\n")


def blog(m):
    with open(f"{LD}/BACKUP_LOG.txt", "a") as f:
        f.write(f"[{now_pt()}] {m}\n")


def clean_env():
    return {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_", "SRUN_", "SALLOC_"))}


def sh(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=clean_env())
        return r.returncode, r.stdout + r.stderr
    except Exception as e:
        return 99, repr(e)


def wr(path, v):
    try:
        if open(path).read().strip() == str(v):
            return
    except Exception:
        pass
    with open(path + ".tmp", "w") as f:
        f.write(f"{v}\n")
    os.replace(path + ".tmp", path)


def natkey(h):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", h)]


def expand(nodes):
    rc, out = sh(["scontrol", "show", "hostnames", nodes])
    return sorted(out.split(), key=natkey) if rc == 0 else []


# ------------------------------------------------------------------------------------------------ state
ST = {"launches": {}, "gaveup": [], "complete": {}, "logs": {}, "last_job": None, "listing_fail": {}}


def load():
    try:
        ST.update(json.load(open(f"{HB}/state/watch.json")))
    except Exception:
        pass


def save():
    json.dump(ST, open(f"{HB}/state/watch.json.tmp", "w"), indent=1)
    os.replace(f"{HB}/state/watch.json.tmp", f"{HB}/state/watch.json")


def plan():
    try:
        s = [int(x) for x in re.split(r"[,\s]+", open(PLAN).read().strip()) if x]
    except Exception:
        return None, None
    return (sorted(set(s)), s[-1]) if s else (None, None)


def done_host(n, h):
    return os.path.exists(f"{HB}/state/export-{n}/{h}.done")


def done(n):
    return all(done_host(n, h) for h in HOSTS)


def n_done(n):
    return sum(done_host(n, h) for h in HOSTS)


# ------------------------------------------------------------------------------------------------ the training job
def job_state(j):
    rc, out = sh(["squeue", "-h", "-j", str(j), "-o", "%j|%T|%N"])
    ln = [x for x in out.splitlines() if x.count("|") == 2]
    return ln[0].split("|") if rc == 0 and ln else None


def live_job():
    """-> (job id or None, reason). Only a RUNNING job named JOBNAME on exactly HOSTS (dry: on its own host)."""
    try:
        j = open(RJ).read().split()[0].strip()
    except Exception:
        return None, f"no job id in {RJ} yet"
    if not j.isdigit():
        return None, f"{RJ} holds {j!r}, not a job id"
    s = job_state(j)
    if not s:
        return None, f"job {j} not in squeue (gone)"
    name, state, nodes = s
    if name != JOBNAME:
        return None, f"job {j} is named {name!r}, not {JOBNAME!r}: REFUSED (never used)"
    if state != "RUNNING":
        return None, f"job {j} is {state}"
    if not DRY:
        hs = expand(nodes)
        if hs != HOSTS:
            return None, f"job {j} runs on {','.join(hs)}, not the 16 export hosts: REFUSED"
    return j, f"job {j} {name} RUNNING"


def stdout_of(j):
    rc, out = sh(["scontrol", "show", "job", str(j), "-o"])
    m = re.search(r"(?:^|\s)StdOut=(\S+)", out)
    return m.group(1) if m else None


def my_pending(names=None):
    """my pending jobs that are not held: [(id, name, reason, n nodes, requested nodes)]"""
    rc, out = sh(["squeue", "-h", "-u", USER, "-t", "PD", "-o", "%i|%j|%r|%D|%n"])
    res = []
    for ln in out.splitlines():
        p = (ln.split("|") + [""] * 5)[:5]
        if not p[0].isdigit() or p[2].startswith("JobHeld"):
            continue
        if names is None or p[1] in names:
            res.append(p)
    return res


# ------------------------------------------------------------------------------------------------ training log (step times, export lines)
class LogReader:
    def __init__(self):
        self.path, self.off, self.part, self.T, self.steps, self.exports = None, 0, "", [], [], set()

    def attach(self, path):
        if path and path != self.path:
            self.__init__(); self.path = path
            log(f"training log: {path}")

    def read(self):
        if not self.path:
            return
        try:
            with open(self.path, errors="replace") as f:
                f.seek(self.off)
                txt = f.read()
                self.off = f.tell()
        except Exception:
            return
        lines = (self.part + txt).split("\n")
        self.part = lines.pop()
        for ln in lines:
            m = STEP_RE.match(ln)
            if m:
                self.steps.append(int(m.group(1))); self.T.append(float(m.group(2)))
                continue
            m = EXP_RE.search(ln)
            if m and int(m.group(1)) not in self.exports:
                self.exports.add(int(m.group(1)))
                log(f"training log: export-{m.group(1)} written on every node ([export-local] line)")


# ------------------------------------------------------------------------------------------------ step-time throttle (hfb_throttle.py)
class Throttle:
    def __init__(self):
        self.j, self.N, self.last, self.cool, self.nl, self.fixed = None, 0, 0.0, 0.0, 0, False

    def job(self, j, lr):
        if j != self.j:
            self.j, self.N, self.last, self.cool, self.nl, self.fixed = j, 0, time.time(), 0.0, len(lr.T), False
            log(f"throttle: job {j}: uploads paused until its step-time baseline (steps 4..15)")

    def tick(self, lr, pending, t_start):
        lr.read()
        T, now, bf = lr.T, time.time(), f"{HB}/state/base-{self.j}"
        if not os.path.exists(bf):
            if len(T) >= 15:
                b = sorted(T[3:15]); wr(bf, (b[5] + b[6]) / 2)
                self.N, self.last, self.nl = 1, now, len(T)
                log(f"throttle: job {self.j} BASE {open(bf).read().strip()} s (median of steps 4..15) -> N 0 -> 1")
            elif not self.fixed and now - t_start >= NOBASE_S:
                self.fixed, self.N = True, NOBASE_N
                log(f"throttle: job {self.j}: no step-time baseline {NOBASE_S:.0f} s after the upload started ({len(T)} step lines in "
                    f"{lr.path}) -> fixed N {NOBASE_N}")
        else:
            base = float(open(bf).read())
            if len(T) > self.nl:
                self.nl = len(T)
                med3 = sorted(T[-3:])[len(T[-3:]) // 2]
                med5 = sorted(T[-5:])[len(T[-5:]) // 2]
                s = lr.steps[-1]
                if T[-1] > base + SLOW:
                    if self.N or now > self.cool:
                        log(f"throttle: job {self.j} step {s} {T[-1]}s > base {base}+{SLOW:g} -> N {self.N} -> 0, cool-off {COOL_S:.0f} s")
                    self.N, self.cool, self.last = 0, now + COOL_S, now
                elif med3 > base + HALVE and self.N > 0:
                    log(f"throttle: job {self.j} step {s} med3 {med3}s > base {base}+{HALVE:g} -> N {self.N} -> {self.N // 2}")
                    self.N, self.last = self.N // 2, now
                elif med5 < base + OKM and now - self.last >= RAMP_S and now > self.cool and self.N < NMAX:
                    n2 = min(NMAX, self.N + (1 if self.N < 2 else 2))
                    log(f"throttle: job {self.j} step {s} med5 {med5}s ok (base {base}) -> N {self.N} -> {n2}")
                    self.N, self.last = n2, now
        order = [h for h in HOSTS if h in pending]
        act = set(order[:self.N])
        for h in HOSTS:
            wr(f"{HB}/control/{h}.workers", W if h in act else 0)
        wr(f"{HB}/state/throttle.now", f"{now_pt()} job {self.j} step {lr.steps[-1] if lr.steps else '-'} last {T[-1] if T else '-'}s base "
           f"{open(bf).read().strip() if os.path.exists(bf) else '-'} N {self.N}{' (fixed)' if self.fixed else ''} active {sorted(act, key=natkey)} "
           f"pending {len(order)}")


def ctl_all(v):
    for h in HOSTS:
        wr(f"{HB}/control/{h}.workers", v)


# ------------------------------------------------------------------------------------------------ probes / launches
def probe(j):
    """{export step: COMPLETE mtime} on rank 0's node (HOSTS[0]) via a 1-CPU, CPU-only overlap step; None if the step failed."""
    h0 = HOSTS[0]
    jid = JOBMAP[h0] if DRY else j
    cmd = ["srun", f"--jobid={jid}", "--overlap", "-w", h0, "-N1", "-n1", "-c1", "--mem=0" if DRY else "--mem=1G", "--gres=none",
           "--cpu-bind=none", "-J", "ttconvo-hfb-probe", "nice", "-n", "19", "ionice", "-c3", "bash", "-c",
           f'for d in {EXR}/export-*; do [ -f "$d/COMPLETE" ] && echo "COMPLETE ${{d##*/export-}} $(stat -c %Y "$d/COMPLETE")"; done; echo PROBE_OK'
           f' $(df -BG --output=avail /scratch | tail -1)']
    rc, out = sh(cmd, timeout=180)
    if "PROBE_OK" not in out:
        log(f"probe of {h0} (job {jid}) failed rc {rc}: {out.strip()[-300:]}")
        return None
    res = {}
    for ln in out.splitlines():
        p = ln.split()
        if len(p) == 3 and p[0] == "COMPLETE" and p[1].isdigit():
            res[int(p[1])] = int(p[2])
    return res


def ensure_repo():
    if DRY:
        return True
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
        info = api.repo_info(REPO, repo_type="model")
        if not info.private:
            log(f"ALERT: {REPO} exists but is NOT private; uploads would abort (hfb_up.py asserts private)")
            return False
        log(f"repo {REPO}: exists, private")
        return True
    except Exception as e:
        log(f"repo {REPO}: create / check failed ({e!r}); hfb_up.py creates it at the first upload")
        return False


def launch(ns, j, mode):
    """mode overlap / dry: Popen of the script (srun overlap step); alloc: sbatch of the script (fallback); dry-alloc: Popen."""
    k = max(ST["launches"].get(str(n), 0) for n in ns) + 1
    lid = f"export-{'+'.join(map(str, ns))}-l{k}-{int(time.time())}"
    ctl_all(0)
    env = clean_env()
    env.update(TTC_LAUNCH=mode, TTC_EXPORTS=",".join(map(str, ns)), TTC_LAUNCH_ID=lid)
    if j:
        env["TTC_JOB"] = str(j)
    cur = {"ns": ns, "lid": lid, "mode": mode, "t": time.time(), "j": j}
    if mode == "alloc":
        cmd = ["sbatch", "--parsable", "-p", "all", "--gres=none", f"-N{len(HOSTS)}", "-w", ",".join(HOSTS), "--ntasks-per-node=1", "-c", "4",
               "--mem=40G", "-t", "0-06:00:00", "-J", "ttconvo-hfb-final", "-o", f"{LD}/hf_final-%j.log",
               f"--export=ALL,TTC_LAUNCH=alloc,TTC_EXPORTS={env['TTC_EXPORTS']},TTC_LAUNCH_ID={lid}", SCRIPT]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        sj = r.stdout.strip().split(";")[0]
        if r.returncode != 0 or not sj.isdigit():
            log(f"fallback sbatch FAILED rc {r.returncode}: {(r.stdout + r.stderr).strip()[-300:]}")
            return None
        cur["sj"] = sj
        log(f"launch {lid}: fallback CPU-only job {sj} (sbatch -p all --gres=none -N{len(HOSTS)} -c 4 --mem=40G) for export(s) {ns}")
    else:
        out = open(f"{LD}/hf_launch-export-{'+'.join(map(str, ns))}.log", "a")
        cur["p"] = subprocess.Popen(["bash", SCRIPT], env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        log(f"launch {lid}: {mode} upload of export(s) {ns}" + (f" as an overlap step of job {j} (-N{len(HOSTS)} -c 4 --mem=40G, "
                                                                   f"throttled)" if mode in ("overlap", "dry") else " (fixed workers)"))
    for n in ns:                                  # counted only once the launch really started
        ST["launches"][str(n)] = ST["launches"].get(str(n), 0) + 1
    save()
    return cur


def launch_finished(cur):
    """-> (finished, rc). A fallback job still pending after FB_PEND_MAX_S or while a ttconvo-ft job pends is cancelled (own job)."""
    if "p" in cur:
        rc = cur["p"].poll()
        return rc is not None, rc
    s = job_state(cur["sj"])
    if not s:
        return True, None
    if s[1] == "PENDING":
        pend = my_pending({JOBNAME})
        if (pend or time.time() - cur["t"] > FB_PEND_MAX_S) and not cur.get("cancelled"):
            sh(["scancel", cur["sj"]]); cur["cancelled"] = True
            log(f"fallback job {cur['sj']} cancelled while pending ({'ttconvo-ft job pending: ' + pend[0][0] if pend else 'pending > 30 min'})")
    return False, None


def pending_hosts(cur):
    n = cur["ns"][0]
    res = set()
    for h in HOSTS:
        try:
            ok = open(f"{HB}/state/export-{n}/{h}.end").read().split()[0] == cur["lid"]
        except Exception:
            ok = False
        if not ok and not done_host(n, h):
            res.add(h)
    return res


def nodes_free():
    bad = []
    for h in HOSTS:
        rc, out = sh(["scontrol", "show", "node", h, "-o"])
        kv = dict(x.split("=", 1) for x in out.split() if "=" in x)
        st, tres = kv.get("State", "?"), kv.get("AllocTRES", "")
        try:
            fc, fm = int(kv["CPUTot"]) - int(kv["CPUAlloc"]), int(kv["RealMemory"]) - int(kv["AllocMem"])
        except Exception:
            fc, fm = 0, 0
        if rc != 0 or any(x in st for x in ("DRAIN", "DOWN", "FAIL", "MAINT", "NOT_RESPONDING", "RESERVED", "POWER")) or fc < 4 or fm < 40960 \
                or "gres/gpu" in tres:
            bad.append(f"{h}:{st} free cpu {fc} mem {fm // 1024}G{' gpu-job' if 'gres/gpu' in tres else ''}")
    want = []
    for p in my_pending():
        if p[1].startswith("ttconvo-hfb"):
            continue
        req = set(expand(p[4])) if p[4] and p[4] != "(null)" else set()
        if (req & set(HOSTS)) or (not req and p[3].isdigit() and int(p[3]) >= 8):
            want.append(f"{p[0]}:{p[1]}")
    return not bad and not want, bad, want


# ------------------------------------------------------------------------------------------------ results
def hf_listing(n):
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.repo_info(REPO, repo_type="model")
    nf = nb = 0
    for e in api.list_repo_tree(REPO, path_in_repo=f"export-{n}", recursive=True, repo_type="model"):
        if hasattr(e, "blob_id") and getattr(e, "size", None) is not None:
            nf += 1; nb += e.size
    return {"files": nf, "bytes": nb, "private": bool(info.private)}


def summarize(n, final):
    """all hosts verified -> combined listing check -> HF_BACKUP-export-N.json + BACKUP_LOG line. False = retry later (listing error)."""
    hosts = {}
    for h in HOSTS:
        d = json.load(open(f"{HB}/state/export-{n}/{h}.done"))
        try:
            d["node_local"] = open(f"{HB}/state/export-{n}/{h}.local").read().strip()
        except Exception:
            d["node_local"] = "?"
        hosts[h] = d
    files, nbytes = sum(d["files"] for d in hosts.values()), sum(d["bytes"] for d in hosts.values())
    if DRY:
        listing = {"skipped": "dry run (no network)"}
        ok = True
    else:
        try:
            listing = hf_listing(n)
        except Exception as e:
            k = str(n); ST["listing_fail"][k] = ST["listing_fail"].get(k, 0) + 1; save()
            log(f"export-{n}: HF listing failed ({e!r}) [{ST['listing_fail'][k]}]")
            if ST["listing_fail"][k] < 5:
                return False
            listing = {"error": repr(e)[:300]}
        listing.update(expected_files=files, expected_bytes=nbytes)
        listing["match"] = listing.get("files") == files and listing.get("bytes") == nbytes and listing.get("private") is True
        ok = listing["match"]
    ok = ok and all(d.get("private") is True for d in hosts.values())
    nl = sorted({d["node_local"].split(":")[0].split()[0] for d in hosts.values()})
    out = {"repo": REPO, "repo_type": "model", "private": True, "path": f"export-{n}/", "url": f"https://huggingface.co/{REPO}/tree/main/export-{n}",
           "step": n, "final": n == final, "PASS": ok, "format": "ppm-local-bf16-shards-v1", "hosts_verified": len(hosts), "hosts_total": len(HOSTS),
           "files": files, "bytes": nbytes, "gb": round(nbytes / 1e9, 2), "hf_listing": listing, "complete_mtime_slinky0": ST["complete"].get(str(n)),
           "node_local": {h: d["node_local"] for h, d in hosts.items()},
           "verify": "per host (hfb_up.py): repo private, every file on HF with the same size, LFS sha256 == upload-time local sha256 of every "
                     "LFS file, 3 files streamed back + sha256'd vs the local file; plus this combined listing check",
           "restore": f"hf download {REPO} --include 'export-{n}/*' --local-dir <dir>: rank<R>.pt (FSDP2 dim-0 shards, world {8 * len(HOSTS)}, "
                      f"R = 8 x host index (sort -V) + local rank) + rep.pt + meta.json + COMPLETE; put each host's 8 rank files in its "
                      f"/scratch/amartyavarma/ttconvo-ft/exports/export-{n}/ (rep.pt, meta.json, COMPLETE on {HOSTS[0]})",
           "time": now_pt(), "hosts": hosts}
    json.dump(out, open(SUMF(n) + ".tmp", "w"), indent=1)
    os.replace(SUMF(n) + ".tmp", SUMF(n))
    blog(f"export-{n} {'(FINAL) ' if n == final else ''}HF backup {'VERIFIED' if ok else 'CHECK FAILED'} on {len(hosts)}/{len(HOSTS)} hosts: "
         f"private {REPO}/export-{n}/ {files} files {nbytes / 1e9:.1f} GB; HF listing {listing.get('files', '-')} files "
         f"{'match' if listing.get('match') else listing.get('skipped') or 'MISMATCH'}; node-local {'/'.join(nl)}; summary {SUMF(n)}")
    log(f"export-{n}: all {len(hosts)} hosts verified; summary written (PASS {ok})")
    return True


def reexport_check(n, mtime):
    """a COMPLETE newer than the one backed up (a relaunch re-exported step n) -> its records are superseded and it is uploaded again."""
    if not os.path.exists(SUMF(n)):
        return
    try:
        old = json.load(open(SUMF(n))).get("complete_mtime_slinky0")
    except Exception:
        return
    if old and mtime > int(old) + 1:
        t = int(time.time())
        os.replace(SUMF(n), f"{SUMF(n)}.superseded-{t}")
        for h in HOSTS:
            for suf in ("done", "end", "local"):
                p = f"{HB}/state/export-{n}/{h}.{suf}"
                if os.path.exists(p):
                    os.replace(p, f"{p}.superseded-{t}")
        ST["launches"][str(n)] = 0; ST["complete"][str(n)] = mtime; save()
        log(f"export-{n}: COMPLETE on {HOSTS[0]} is newer ({mtime}) than the backed-up one ({old}): re-exported, uploading again")
        blog(f"export-{n} RE-EXPORTED (COMPLETE newer than the HF backup): previous records superseded, uploading again")


# ------------------------------------------------------------------------------------------------ main loop
def main():
    if not DRY:
        me = E("SLURM_JOB_ID", "")
        rc, out = sh(["squeue", "-h", "-u", USER, "-n", "ttconvo-hfb-watch", "-t", "R", "-o", "%i"])
        others = [x for x in out.split() if x.isdigit() and x != me]
        if others:
            log(f"another ttconvo-hfb-watch job runs ({others}): exiting"); return 0
    load()
    log(f"watcher start pid {os.getpid()} job {E('SLURM_JOB_ID', '-')} dry {DRY}: repo {REPO}, hosts {','.join(HOSTS)}, exports {EXR}, "
        f"poll {POLL_S:g} s, throttle W {W} ramp {RAMP_S:g} s cool {COOL_S:g} s")
    repo_ok, repo_t = ensure_repo(), time.time()
    lr, thr = LogReader(), Throttle()
    cur, gone_since, last_probe, why_prev, plan_prev, fb_note = None, None, 0.0, None, None, 0.0
    while True:
        try:
            if os.path.exists(f"{HB}/state/STOP_WATCH"):          # clean stop without scancel (touch it; removed here)
                os.remove(f"{HB}/state/STOP_WATCH")
                log(f"STOP_WATCH found: exit (running launch: {cur['lid'] if cur else '-'})"); return 3
            if not repo_ok and time.time() - repo_t > 3600:
                repo_ok, repo_t = ensure_repo(), time.time()
            steps, final = plan()
            if (steps, final) != plan_prev:
                log(f"plan: export steps {steps}, final {final}" if steps else f"plan: {PLAN} missing / empty: waiting")
                plan_prev = (steps, final)
            j, why = live_job()
            if why != why_prev:
                log(f"training job: {why}"); why_prev = why
            if j:
                if ST["last_job"] != j:
                    ST["last_job"] = j; save()
                p = TRAIN_LOG if DRY else (ST["logs"].get(j) or stdout_of(j))
                if p and ST["logs"].get(j) != p:
                    ST["logs"][j] = p; save()
                lr.attach(p)
                thr.job(j, lr)
            elif ST["last_job"]:
                lr.attach(TRAIN_LOG if DRY else ST["logs"].get(ST["last_job"]))   # the ended job's log (its last [export-local] line)
            lr.read()
            now = time.time()
            # 1. the running launch: throttle while it runs; when it ends, count it
            if cur:
                fin, rc = launch_finished(cur)
                if not fin:
                    if cur["mode"] in ("overlap", "dry"):
                        thr.tick(lr, pending_hosts(cur), cur["t"])
                    time.sleep(CTL_S)
                    continue
                for n in cur["ns"]:
                    if cur.get("cancelled") and not done(n):   # a fallback cancelled while pending never ran: not counted
                        ST["launches"][str(n)] = max(0, ST["launches"].get(str(n), 1) - 1); save()
                    log(f"launch {cur['lid']} ({cur['mode']}) ended rc {rc}: export-{n} verified on {n_done(n)}/{len(HOSTS)} hosts")
                    if not done(n) and ST["launches"].get(str(n), 0) >= MAX_LAUNCH and str(n) not in ST["gaveup"]:
                        ST["gaveup"].append(str(n)); save()
                        log(f"export-{n}: GAVE UP after {MAX_LAUNCH} launches")
                        blog(f"export-{n} HF backup GAVE UP after {MAX_LAUNCH} launches: {n_done(n)}/{len(HOSTS)} hosts verified; "
                             f"the other hosts' node-local copies are kept (see logs/hf_up-export-{n}-<host>.log)")
                ctl_all(0)
                cur = None
            if not steps:
                time.sleep(POLL_S); continue
            # 2. results
            for n in steps:
                if done(n) and not os.path.exists(SUMF(n)):
                    summarize(n, final)
            todo = [n for n in steps if not done(n) and str(n) not in ST["gaveup"]]
            if not todo and not [n for n in steps if done(n) and not os.path.exists(SUMF(n))]:
                fin_ok = done(final) and os.path.exists(SUMF(final))
                log(f"every export handled (final export-{final} {'verified on all hosts' if fin_ok else 'NOT verified'}; gave up: "
                    f"{ST['gaveup'] or 'none'}): exit")
                return 0 if fin_ok and not ST["gaveup"] else 1
            # 3. job running: probe for COMPLETE, launch the oldest pending one
            if j and todo:
                gone_since = None
                near = (not lr.steps) or lr.steps[-1] >= min(todo) - 1 or any(n in lr.exports for n in todo)
                if now - last_probe >= (POLL_S if near else PROBE_IDLE_S):
                    last_probe = now
                    C = probe(j)
                    if C is not None:
                        for n, m in sorted(C.items()):
                            if n not in steps:
                                continue
                            if ST["complete"].get(str(n)) != m:
                                if str(n) in ST["complete"]:
                                    reexport_check(n, m)
                                ST["complete"][str(n)] = m; save()
                                log(f"export-{n}: COMPLETE on {HOSTS[0]} (mtime {m})")
                        cands = [n for n in todo if n in C or done_host(n, HOSTS[0])]
                        if cands:
                            cur = launch([cands[0]], j, "dry" if DRY else "overlap")
                            time.sleep(CTL_S)
                            continue
            # 4. job gone (only after one was seen running: before the run starts this just waits): fallback CPU-only job for a
            #    COMPLETE export still not on HF, or exit
            elif ST["last_job"] and todo:
                gone_since = gone_since or now
                pend = my_pending({JOBNAME})
                if pend:
                    if now - fb_note > 1800:
                        log(f"ttconvo-ft job {pend[0][0]} pending: waiting for it (uploads overlap into it once it runs)"); fb_note = now
                    gone_since = now
                else:
                    ev = [n for n in todo if str(n) in ST["complete"] or n in lr.exports or done_host(n, HOSTS[0])]
                    if ev and now - gone_since >= FALLBACK_WAIT_S:
                        ok, bad, want = nodes_free()
                        if ok or DRY:
                            if DRY and not ok:
                                log(f"(dry) real-mode verdict: nodes not free ({bad[:3]} {want[:3]}); dry fallback runs anyway in the klweb jobs")
                            cur = launch(ev, None, "dry-alloc" if DRY else "alloc")
                            if cur:
                                time.sleep(CTL_S); continue
                        elif now - fb_note > 1800:
                            log(f"job gone; export(s) {ev} COMPLETE but not on HF: fallback waits for the 16 nodes to be free "
                                f"(busy: {bad[:4]}{' ...' if len(bad) > 4 else ''}; pending jobs wanting them: {want[:3]})"); fb_note = now
                    elif not ev and now - gone_since >= GONE_EXIT_S:
                        log(f"training job gone for {GONE_EXIT_S / 3600:.1f} h, no COMPLETE export pending (not on HF: {todo}): exit")
                        return 2 if not done(final) else 0
            wr(f"{HB}/state/watch.now", f"{now_pt()} {why}; pending {todo}; done {[n for n in steps if done(n)]}; gave up {ST['gaveup']}")
        except Exception as e:
            log(f"ERROR {e!r}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
