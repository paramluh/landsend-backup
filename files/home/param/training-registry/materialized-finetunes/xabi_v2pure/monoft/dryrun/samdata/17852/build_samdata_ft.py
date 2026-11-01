#!/usr/bin/env python3
"""PPM 32B fine-tune on SAM 8B's data (ppm/aloe-v2pure-16610 = job 23051 aloe-v2pure-ft verbatim, init PPM step 16610, world 64) vs the 8B SAM pose
Aloe FT "v2pure" (HF InternalCan/sam-pose-aloe-v2pure-step636) live curve page, in the standard live PPM page format (samdata_ft.tpl.html =
a copy of ppm.tpl.html with the same tools; its data model swapped for these runs). Stdlib only. Rebuilt every 60 s by samdata_ft_refresh.sbatch;
served by the klweb tunnel (klweb-ppm, ppm_serve.sbatch: http.server on this directory).

x = epochs of SAM 8B's training data (Aloe pose release v2 "v2pure" train set): SAM step / 636 (its total_steps = exactly 1 epoch), ours step / 200
(PAGE_CFG.json total_steps; 200 steps x 64 rows = 1 epoch of the same data); the page also has a step mode.

Reads  ppm/aloe-v2pure-16610/RUNNING_JOB.txt ("job J nodes .. started .." / ".. ended .. rc N"; the newest 'job J' line = the current job),
       RUN_TAG.txt ('ppm-aloe-v2pure-16610-j<first job>', kept across resumes) -> runs/<RUN_TAG>/metrics.jsonl (else the newest runs/ppm-aloe-v2pure-16610-j*;
       train lines train_speech/*, train/*; eval lines eval_speech/* = the combined held-out eval (3,481 clips of SAM's 113 held-out recordings);
       last record per step wins); if no records yet, the trainer log logs/*-<JOB>.log; PAGE_CFG.json (total_steps, rows_per_step, eval_every).
       SAM: sam-aloe-ft-v2pure/logs/log.jsonl + run.json (per-step train lines; {"step","heldout"} = its 113 held-out recordings, 3,482 items, the
       SAME held-out clips our eval uses; {"step","eval"} = 1 packed row per local rank (8 rows) cut from the head of ITS TRAINING permutation and
       excluded from training (train_pose.py eval_ds / packloader.plan eval_rows): an in-distribution split of the training recordings, shown
       separately and off by default). Same parsing as build_ppm_aloe_v2pure.py load_sam(); SAM's word CE is not comparable (text upfront) and absent.
       Grey reference (off by default): the earlier PPM v2pure FT 23051 (ppm/aloe-v2pure-ft/runs/ppm-aloe-v2pure-ft-j23051/metrics.jsonl).
       10-21 scaling: two more PPM FT runs on the same recipe (= 26089 verbatim except the init), parsed exactly as the 16,610 run (FT_RUNS):
       ppm/aloe-v2pure-13533 (init PPM step 13,533) and ppm/aloe-v2pure-7932 (init PPM step 7,932: the nearest HF-verified save to 8,057, which was
       GCS-only; never label it 8,057) -> s['fts'][id] (same fields as the top-level 16,610 doc; pending / empty until the dir has a job / records).
       10-22: + ppm/aloe-v2pure-17352 and ppm/aloe-v2pure-17641 (PPM 8,1,1 inits: x = the run's own eval_pose, i.e. decp_bb link OFF; the FT trains the
       link on, as 26089). Scaling (s['scaling'], one point per PPM FT run 7,932 / 11,488 (= 23051) / 13,533 / 16,610 / 17,352 / 17,641): x = the init's PPM pretraining held-out eval
       (eval_pose/pose_cb0_ce, eval_pose/pose_light_ce) from ppm/logs/ppm-<job>.log 'EVAL step N: k v | ...' lines at the init step, else the nearest
       eval step (a step in several logs: the newest log's line, spread across logs kept); y = the FT's eval_speech/pose_cb0_ce / pose_light_ce at its
       last eval step, only once that is >= its total steps (200); 'latest' = the newest eval (> step 0) of a run not there yet (hollow marker).
Writes <OUT>/samdata_ft_live.json (polled by the page), <OUT>/samdata_ft.html, <OUT>/ver_samdata_ft.txt (= template hash), atomically.
Usage: python3 build_samdata_ft.py [out_dir (default = this dir)]   Test env: SDFT_BASE (16,610 run dir root), SDFT_NOQ=1 (ignore squeue)."""
import datetime
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

W = os.path.dirname(os.path.abspath(__file__))
O = sys.argv[1] if len(sys.argv) > 1 else W
H = "/data/home/amartyavarma"
BASE = os.environ.get("SDFT_BASE", f"{H}/ppm/aloe-v2pure-16610")
RUNS, LOGS, CFG, RJ = f"{BASE}/runs", f"{BASE}/logs", f"{BASE}/PAGE_CFG.json", f"{BASE}/RUNNING_JOB.txt"
RUNPFX = "ppm-aloe-v2pure-16610-j"
JOBRE = re.compile(r"aloev2p16610|v2pure.?16610", re.I)   # squeue name of the training job (before RUNNING_JOB.txt exists)
JOBNOT = re.compile(r"export|gcs|prune|hfb|clean|refresh|klweb|check|test|smoke", re.I)
REF = f"{H}/ppm/aloe-v2pure-ft/runs/ppm-aloe-v2pure-ft-j23051/metrics.jsonl"   # earlier PPM v2pure FT (grey reference)
REF_CFG = f"{H}/ppm/aloe-v2pure-ft/PAGE_CFG.json"
SAMD = f"{H}/sam-aloe-ft-v2pure-cmp"
SAM_RUNS = [("sam", SAMD), ("sam3", f"{H}/sam-aloe-ft-v2pure-cmp-lr3")]   # SAM-kind runs on the page (id, dir); template RUNC/SAMS give label + colour   # SAM 8B FT on the SAME plan as 26089 / 23051 (200 steps, same clips per step, same held-out eval)
DEF = {"total_steps": 200, "rows_per_step": 64, "eval_every": 25}
PT = ZoneInfo("America/Los_Angeles")
TCOLS = ["step", "rows", "nword", "cb0", "light", "pcb0", "plight", "step_s"]
TKEYS = [None, None, "train_speech/word_ce_norm", "train_speech/cb0_ce", "train_speech/light_ce",
         "train_speech/pose_cb0_ce", "train_speech/pose_light_ce", "train/step_time"]
ECOLS = ["step", "rows", "nword", "cb0", "light", "pcb0", "plight"]
EKEYS = [None, None, "eval_speech/word_ce_norm", "eval_speech/cb0_ce", "eval_speech/light_ce",
         "eval_speech/pose_cb0_ce", "eval_speech/pose_light_ce"]
SCOLS = ["step", "cb0", "light", "pcb0", "plight", "step_s"]  # SAM 8B cmp: our column <- its metrics.jsonl key (token-weighted, PPM-style keys)
STKEYS = [None, "train_speech/cb0_ce", "train_speech/light_ce", "train_speech/pose_cb0_ce", "train_speech/pose_light_ce", "train/step_time"]
SEKEYS = [None, "eval_speech/cb0_ce", "eval_speech/light_ce", "eval_speech/pose_cb0_ce", "eval_speech/pose_light_ce", None]
LIVE_ST = ("RUNNING", "PENDING", "COMPLETING", "CONFIGURING")
TRK = ("word_ce_norm", "pad_ce", "cb0_ce", "light_ce", "pose_cb0_ce", "pose_light_ce")
PRE_LOGS = f"{H}/ppm/logs/ppm-*.log"   # PPM 32B pretraining trainer logs (EVAL lines -> the scaling chart's x)
PRE_KEYS = (("pcb0", "eval_pose/pose_cb0_ce"), ("plight", "eval_pose/pose_light_ce"))
FT_KEYS = (("pcb0", "eval_speech/pose_cb0_ce"), ("plight", "eval_speech/pose_light_ce"))


class Run:
    """One PPM FT run dir (aloe-v2pure-* layout: RUN_TAG.txt, RUNNING_JOB.txt, runs/<RUN_TAG>/metrics.jsonl, logs/<name>-<job>.log)."""

    def __init__(self, rid, init, base, pfx, jobre):
        self.id, self.init, self.base, self.pfx = rid, init, base, pfx
        self.runs, self.logs, self.cfg, self.rj = f"{base}/runs", f"{base}/logs", f"{base}/PAGE_CFG.json", f"{base}/RUNNING_JOB.txt"
        self.jobre = re.compile(jobre, re.I)


# PPM FT runs with a live doc, all = 26089's recipe (200 steps, eval every 25) except the init: 'ft' = the 16,610 run (top level of the live JSON, as
# before), the others under s['fts'][id]; the template's PPMS gives label + colour. 7,932 = the nearest HF-verified save to 8,057 (GCS-only).
FT_RUNS = [Run("ft", 16610, BASE, RUNPFX, JOBRE.pattern),
           Run("p13533", 13533, f"{H}/ppm/aloe-v2pure-13533", "ppm-aloe-v2pure-13533-j", r"aloev2p13533|v2pure.?13533"),
           Run("p7932", 7932, f"{H}/ppm/aloe-v2pure-7932", "ppm-aloe-v2pure-7932-j", r"aloev2p7932|v2pure.?7932"),
           # 10-22 scaling points 5 + 6 (user via coordinator): the PPM 8,1,1 inits = 26089 verbatim except the init; both world 64 on 8 PPM hosts each,
           # launched together after the delivery-2 run (17,641 = its end); the killed first 17,352 run (job 27211) is archived in runs-killed-27211
           # (job-name regex anchored: their backup / cleanup jobs are named aloev2p<step>-hffinal / -rmsaves)
           Run("p17352", 17352, f"{H}/ppm/aloe-v2pure-17352", "ppm-aloe-v2pure-17352-j", r"^aloev2p17352$"),
           Run("p17641", 17641, f"{H}/ppm/aloe-v2pure-17641", "ppm-aloe-v2pure-17641-j", r"^aloev2p17641$"),
           # 10-23 scaling point 7: init = PPM 8,1,1 delivery-3 append run (job 27653) step 17852; = aloe-v2pure-17641 verbatim except the init
           Run("p17852", 17852, f"{H}/ppm/aloe-v2pure-main-17852", "ppm-aloe-v2pure-main-17852-j", r"^aloev2pm17852$")]
REF_INIT = 11488   # the 23051 reference's init PPM step (scaling point)
SC_ORDER = ("p7932", "ref", "p13533", "ft", "p17352", "p17641", "p17852")   # scaling points, by init step
EXACT_PRE = {"p17352", "p17641", "p17852"}   # 10-22: x must be the pretraining eval at exactly the init step


def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def parse_pt(s):
    try:
        d = datetime.datetime.strptime(f"{datetime.datetime.now(PT).year}-" + s.replace(" PT", ""), "%Y-%m-%d %H:%M:%S")
        return d.replace(tzinfo=PT).timestamp()
    except Exception:
        return None


def cfg(path=CFG):
    c = dict(DEF)
    try:
        j = json.load(open(path))
        for k in DEF:
            if isinstance(j, dict) and num(j.get(k)):
                c[k] = int(j[k])
        c["_cfg"] = True
    except Exception:
        c["_cfg"] = False
    c["eval_every"] = max(1, c["eval_every"])
    return c


def _jid(p, pfx):
    t = os.path.basename(p)[len(pfx):]
    return int(t) if t.isdigit() else None


def live_run(R=FT_RUNS[0]):
    try:
        t = open(f"{R.base}/RUN_TAG.txt").read().strip()
        if t and os.path.isdir(f"{R.runs}/{t}"):
            return f"{R.runs}/{t}"
    except Exception:
        pass
    rs = [p for p in glob.glob(f"{R.runs}/{R.pfx}*") if _jid(p, R.pfx) is not None]
    return max(rs, key=lambda p: _jid(p, R.pfx)) if rs else None


def log_of(job, R=FT_RUNS[0]):
    ls = [p for p in glob.glob(f"{R.logs}/*-{job}.log") if not JOBNOT.search(os.path.basename(p))]
    return sorted(ls)[0] if ls else None


def running_job_file(R=FT_RUNS[0]):
    """RUNNING_JOB.txt: the newest 'job J' (5-7 digits) named on any line = the current job (a resume appends a new line)."""
    try:
        t = open(R.rj).read()
    except Exception:
        return None
    js = [int(x) for x in re.findall(r"\bjob\D{0,6}(\d{5,7})\b", t, re.I)] or [int(x) for x in re.findall(r"\b(\d{5,7})\b", t)]
    return max(js) if js else None


def ended_line(job, R=FT_RUNS[0]):
    try:
        for line in open(R.rj, errors="replace"):
            m = re.search(r"\bjob (\d{5,7})\b.*\bended (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d PT)(?: rc (-?\d+))?", line)
            if m and int(m.group(1)) == int(job):
                return m.group(2), (int(m.group(3)) if m.group(3) is not None else None)
    except Exception:
        pass
    return None


def load_log(path):
    """metrics.jsonl fallback: the trainer log's step / EVAL lines (train_aloe_pose.py format, as build_ppm_aloe_v2pure.py load_log)."""
    rs = re.compile(r"^\[(\d\d-\d\d \d\d:\d\d:\d\d PT)\] step (\d+)/\d+ ([\d.]+)s .*?peak ([\d.]+) GiB \| (.*?)(?: \| ETA.*)?$")
    re_ = re.compile(r"^\[(\d\d-\d\d \d\d:\d\d:\d\d PT)\] EVAL step (\d+): (.*)$")
    tr, ev = {}, {}
    try:
        for line in open(path, errors="replace"):
            line = line.rstrip("\n")
            m = re_.match(line)
            if m:
                d = {"step": int(m.group(2)), "time_pt": m.group(1)}
                for kv in m.group(3).split(" | "):
                    k, _, v = kv.strip().rpartition(" ")
                    try:
                        d[k] = float(v)
                    except ValueError:
                        pass
                if "eval_speech/word_ce_norm" in d:
                    ev[d["step"]] = d
                continue
            m = rs.match(line)
            if m:
                d = {"step": int(m.group(2)), "time_pt": m.group(1), "train/step_time": float(m.group(3)), "train/peak_gib": float(m.group(4))}
                for kv in m.group(5).split():
                    k, _, v = kv.partition("=")
                    try:
                        d[("train_speech/" if k in TRK else "train/") + k] = float(v)
                    except ValueError:
                        pass
                tr[d["step"]] = d
    except (FileNotFoundError, TypeError):
        pass
    return [tr[x] for x in sorted(tr)], [ev[x] for x in sorted(ev)]


def load(path):
    """metrics.jsonl -> (train records, eval records), last record per step wins; one-off identity records (eval_ref*) are neither."""
    tr, ev = {}, {}
    try:
        for line in open(path, errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if not isinstance(d, dict) or not num(d.get("step")):
                continue
            if "eval_speech/word_ce_norm" in d or "eval_speech/pose_cb0_ce" in d:
                ev[int(d["step"])] = d
            elif "train_speech/word_ce_norm" in d or "train/loss" in d:
                tr[int(d["step"])] = d
    except (FileNotFoundError, TypeError):
        pass
    return [tr[s] for s in sorted(tr)], [ev[s] for s in sorted(ev)]


def load_sam(SAMD=SAMD):
    """SAM 8B comparable FT (sam-aloe-ft-v2pure-cmp, train_pose_cmp.py): logs/metrics.jsonl (rank 0, token-weighted PPM-style keys train_speech/*,
    eval_speech/* = the same 3,481 held-out clips), job from its RUNNING_JOB.txt ('job J nodes .. started ..' / 'job J ended .. rc N').
    The old SAM 8B v2pure 636 x 24 curve is NOT shown (different schedule)."""
    tr, ev = load(f"{SAMD}/logs/metrics.jsonl")
    total = 200
    try:
        rj = json.load(open(f"{SAMD}/logs/run.json"))
        total = int(rj.get("steps") or rj.get("total_steps") or 200)
    except Exception:
        pass
    row = lambda d, keys: [int(d["step"])] + [(round(d[k], 6) if k and num(d.get(k)) else None) for k in keys[1:]]   # noqa: E731
    out = {"total": total, "train": [row(d, STKEYS) for d in tr], "heldout": [row(d, SEKEYS) for d in ev], "phase": "starting",
           "live": False, "step": int(tr[-1]["step"]) if tr else None}
    job, end = None, None
    try:
        t = open(f"{SAMD}/RUNNING_JOB.txt").read()
        js = [int(x) for x in re.findall(r"\bjob (\d{5,7})\b", t)]
        job = max(js) if js else None
        if job:
            m = re.search(rf"\bjob {job}\b.*\bended (\d{{4}}-\d\d-\d\d \d\d:\d\d:\d\d PT)(?: rc (-?\d+))?", t)
            end = m.group(1) if m else None
    except Exception:
        pass
    if job:
        state, _ = job_state(job)
        out.update(jobid=str(job), state=state)
        live = state in LIVE_ST and not end
        out["live"] = live
        if state == "PENDING":
            out["phase"] = "pending"
        elif live:
            out["phase"] = "training" if tr else "starting"
        else:
            out["phase"] = "done" if (out["step"] or 0) >= total else "ended"
    st = [d["train/step_time"] for d in tr[-20:] if num(d.get("train/step_time"))]
    if st:
        out["mean_s"] = round(sum(st) / len(st), 2)
    out["done"] = out["phase"] in ("done", "ended")
    return out


# LR chart: one line per param group, generic over runs. LR_GROUPS[kind] = [(chart group, label, metrics key)]; chart groups:
# trunk (trunk + embeddings / heads trained with it), towers (the 4B towers), decoders (the 0.6B decoders / light towers + their link).
# Decoder LRs = the LR of the step's LAST nanostep (the K = 8 nanosteps decay within the step on the same shape).
LR_GROUPS = {
    "ppm": [("trunk", "trunk", "train/lr_trunk"), ("trunk", "spec", "train/lr_spec"), ("trunk", "proj", "train/lr_proj"),
            ("trunk", "audio+pose emb", "train/lr_audio"),
            ("towers", "speech tower", "train/lr_tower_s"), ("towers", "pose tower", "train/lr_tower_p"),
            ("decoders", "speech dec (last nanostep)", "train/dec_lr_last"), ("decoders", "pose dec (last nanostep)", "train/dec_p_lr_last"),
            ("decoders", "decp_bb link", "train/lr_decp_bb")],
    "sam": [("trunk", "trunk", "train/lr_trunk"), ("trunk", "new pose params", "train/lr_new"),
            ("decoders", "audio light", "train/lr_audio_light"), ("decoders", "pose light (last nanostep)", "train/pose_light_lr_last")],
}


def lr_runs(paths):
    """Registry of the page's runs for the LR chart: (id, metrics.jsonl, kind). Add a run here (id + label/colour in the template's RUNC).
    paths = {PPM FT run id: its live metrics.jsonl or None}."""
    return ([("ft", paths.get("ft"), "ppm"), ("p13533", paths.get("p13533"), "ppm"), ("ref", REF, "ppm"), ("p7932", paths.get("p7932"), "ppm"),
             ("p17352", paths.get("p17352"), "ppm"), ("p17641", paths.get("p17641"), "ppm"), ("p17852", paths.get("p17852"), "ppm")]
            + [(k, f"{d}/logs/metrics.jsonl", "sam") for k, d in SAM_RUNS])


def lr_doc(paths):
    out = []
    for rid, path, kind in lr_runs(paths if isinstance(paths, dict) else {"ft": paths}):
        tr, _ = load(path) if path else ([], [])
        ser = []
        for grp, lab, key in LR_GROUPS[kind]:
            pts = [[int(d["step"]), d[key]] for d in tr if num(d.get(key))]
            if pts:
                ser.append([grp, lab, key, pts])
        out.append({"id": rid, "kind": kind, "series": ser})
    return out


def job_state(job):
    try:
        r = subprocess.run(["squeue", "-h", "-j", str(job), "-o", "%T|%N"], capture_output=True, text=True, timeout=20).stdout.strip()
        if r:   # a het job prints one line per component: first state, nodes joined
            ls = [x.split("|") for x in r.splitlines() if "|" in x]
            return [ls[0][0], ",".join(x[1] for x in ls)]
        r = subprocess.run(["sacct", "-n", "-X", "-j", str(job), "-o", "State%20"], capture_output=True, text=True, timeout=20).stdout.split()
        return [r[0] if r else "ENDED", ""]
    except Exception:
        return ["?", ""]


_SQ = []


def queued_job(R=FT_RUNS[0]):
    if os.environ.get("SDFT_NOQ"):
        return None
    try:
        if not _SQ:   # one squeue call per build, shared by every run
            _SQ.append(subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", "amartyavarma"), "-o", "%i|%T|%N|%j"],
                                      capture_output=True, text=True, timeout=20).stdout.split())
        r = _SQ[0]
        r = [x.split("|")[0].split("+")[0] + "|" + x.split("|", 1)[1] for x in r if "|" in x]   # het job components "<id>+<k>" -> <id>
        js = [x.split("|") for x in r if x.count("|") >= 3 and x.split("|")[0].isdigit()
              and R.jobre.search(x.split("|")[3]) and not JOBNOT.search(x.split("|")[3])]
        return max(js, key=lambda x: int(x[0]))[:3] if js else None
    except Exception:
        return None


def rowify(recs, keys, rows):
    return [[int(d["step"]), int(d["step"]) * rows] + [(round(d[k], 9) if num(d.get(k)) else None) for k in keys[2:]] for d in recs]


def ft_doc(R):
    """One PPM FT run's live doc (the fields the 16,610 run has always had at the top level) + its eval records + its metrics.jsonl path.
    No job / no records yet (e.g. 7,932 before launch) -> phase 'pending', empty curves, never an exception."""
    c = cfg(R.cfg)
    TOTAL, ROWS = c["total_steps"], c["rows_per_step"]
    s = {"max_steps": TOTAL, "epoch_steps": TOTAL, "rows": ROWS, "eval_every": c["eval_every"], "cfg": c["_cfg"], "init": R.init,
         "train": [], "eval": [], "phase": "pending", "live": False, "done": False}
    run = live_run(R)
    q = queued_job(R)
    rj = running_job_file(R)
    rjob = _jid(run, R.pfx) if run else None
    jobs = [x for x in ((int(q[0]) if q else None), rj, rjob) if x]
    mpath = f"{run}/metrics.jsonl" if run else None
    if not jobs:
        s["nojob"] = True   # not launched yet
        return s, [], mpath
    job = max(jobs)
    if q and int(q[0]) == job:
        state, nodes = q[1], (q[2] if len(q) > 2 else "")
    else:
        state, nodes = job_state(job)
    tr, ev = load(mpath) if run else ([], [])
    if not tr and not ev:
        tr, ev = load_log(log_of(job, R))
        if tr or ev:
            s["src"] = "log"
    end = ended_line(job, R)
    live = state in LIVE_ST and not end
    step = int(tr[-1]["step"]) if tr else None
    s.update(jobid=str(job), state=state, nodes=nodes, live=live, train=rowify(tr, TKEYS, ROWS), eval=rowify(ev, EKEYS, ROWS), step=step)
    if run:
        s["run"] = os.path.basename(run)
    if end:
        s["ended_pt"] = end[0][5:]
    if state == "PENDING":
        s["phase"] = "pending"
    elif live:
        s["phase"] = "training" if tr else "starting"
    else:
        s["phase"] = "done" if (state == "COMPLETED" or (step or 0) >= TOTAL) else "ended"
    s["done"] = s["phase"] in ("done", "ended")
    if tr:
        last = tr[-1]
        s["last_pt"] = last.get("time_pt")
        s["t_last"] = parse_pt(last.get("time_pt", ""))
        st = [r["train/step_time"] for r in tr[-20:] if num(r.get("train/step_time"))]
        if st:
            s["mean_s"] = round(sum(st) / len(st), 2)
    if ev:
        s["ev_last_step"] = int(ev[-1]["step"])
    return s, ev, mpath


def pre_evals():
    """PPM pretraining held-out evals from ppm/logs/ppm-<job>.log: {step: [(job, {pcb0, plight})]} for EVAL lines carrying both pose keys
    ('[ts] EVAL step N: eval_pose/word_ce_norm 1.95 | ... | eval_pose/pose_cb0_ce 1.7792 | ... | eval_pose/pose_light_ce 3.9931 | ...')."""
    out = {}
    rx = re.compile(r"EVAL step (\d+): (.*)$")
    for p in glob.glob(PRE_LOGS):
        m = re.search(r"/ppm-(\d+)\.log$", p)
        if not m:
            continue
        try:
            fh = open(p, errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                if "EVAL step" not in line:
                    continue
                mm = rx.search(line.rstrip("\n"))
                if not mm:
                    continue
                d = {}
                for kv in mm.group(2).split(" | "):
                    k, _, v = kv.strip().rpartition(" ")
                    try:
                        d[k] = float(v)
                    except ValueError:
                        pass
                vals = {c: d.get(k) for c, k in PRE_KEYS}
                if all(num(v) for v in vals.values()):
                    out.setdefault(int(mm.group(1)), []).append((int(m.group(1)), vals))
    return out


def pre_at(pe, init):
    """The pretraining eval at the init step, else the nearest eval step (ties -> the earlier step); a step in several logs -> the newest log's
    line (largest job id), with the spread across logs kept (they should agree)."""
    if not pe:
        return None
    st = init if init in pe else min(pe, key=lambda x: (abs(x - init), x))
    rows = sorted(pe[st], key=lambda r: r[0])
    job, vals = rows[-1]
    spread = max((abs(r[1][c] - vals[c]) for r in rows for c, _ in PRE_KEYS), default=0.0)
    return dict(step=st, exact=st == init, job=job, n_logs=len(rows), spread=round(spread, 6), **{c: round(vals[c], 6) for c, _ in PRE_KEYS})


def ft_pt(e):
    return {"step": int(e["step"]), **{c: (round(e[k], 6) if num(e.get(k)) else None) for c, k in FT_KEYS}}


def scaling(evs, totals, docs):
    """One entry per PPM FT run (SC_ORDER): pre = x (pretraining eval at the init), final = y (the FT's eval at its last eval step, only once that
    step >= its total steps), latest = its newest eval (> step 0) while not final. Pending runs keep an entry (no final / latest)."""
    pe = pre_evals()
    inits = {R.id: R.init for R in FT_RUNS}
    inits["ref"] = REF_INIT
    out = []
    for rid in SC_ORDER:
        ev = [e for e in (evs.get(rid) or []) if all(num(e.get(k)) for _, k in FT_KEYS)]
        tot = totals.get(rid) or DEF["total_steps"]
        last = ev[-1] if ev else None
        fin = last is not None and int(last["step"]) >= tot
        d = docs.get(rid) or {}
        pre = pre_at(pe, inits[rid])
        if rid in EXACT_PRE and pre and not pre["exact"]:
            pre = None   # 8,1,1 inits: x only from the eval AT the init step (the run evals at its final step; never a neighbour)
        out.append({"id": rid, "init": inits[rid], "total": tot, "pre": pre,
                    "final": ft_pt(last) if fin else None,
                    "latest": ft_pt(last) if (last is not None and not fin and int(last["step"]) > 0) else None,
                    "phase": d.get("phase"), "jobid": d.get("jobid"), "nojob": bool(d.get("nojob"))})
    return out


def doc():
    now = time.time()
    rc = cfg(REF_CFG)
    rtr, rev = load(REF)
    s = {"tcols": TCOLS, "ecols": ECOLS, "scols": SCOLS, "sam": load_sam(), "sams": {k: load_sam(d) for k, d in SAM_RUNS},
         "ref": {"run": "ppm-aloe-v2pure-ft-j23051", "total": rc["total_steps"], "rows": rc["rows_per_step"],
                 "train": rowify(rtr, TKEYS, rc["rows_per_step"]), "eval": rowify(rev, EKEYS, rc["rows_per_step"])},
         "updated": datetime.datetime.fromtimestamp(now, PT).strftime("%a %m-%d %H:%M PT")}
    docs, evs, paths, totals = {}, {"ref": rev}, {}, {"ref": rc["total_steps"]}
    for R in FT_RUNS:
        try:
            d, ev, mp = ft_doc(R)
        except Exception as ex:   # one broken run dir must not take the page down
            d, ev, mp = {"phase": f"builder error: {ex!r}", "train": [], "eval": [], "live": False, "done": False, "init": R.init}, [], None
        docs[R.id], evs[R.id], paths[R.id], totals[R.id] = d, ev, mp, d.get("max_steps")
    s.update(docs["ft"])   # the 16,610 run stays at the top level (as before)
    s["fts"] = {k: v for k, v in docs.items() if k != "ft"}
    s["lr"] = lr_doc(paths)
    try:
        s["scaling"] = scaling(evs, totals, docs)
    except Exception as ex:
        s["scaling"], s["scaling_err"] = [], repr(ex)
    return s


def wr(path, text):
    open(path + ".tmp", "w").write(text)
    os.replace(path + ".tmp", path)


def main():
    try:
        s = doc()
    except Exception as ex:
        import traceback
        traceback.print_exc()
        s = {"phase": f"builder error: {ex!r}", "max_steps": DEF["total_steps"], "epoch_steps": DEF["total_steps"], "train": [], "eval": [],
             "tcols": TCOLS, "ecols": ECOLS, "scols": SCOLS, "sam": load_sam(), "sams": {k: load_sam(d) for k, d in SAM_RUNS}, "ref": None,
             "fts": {}, "scaling": []}
    wr(f"{O}/samdata_ft_live.json", json.dumps(s, separators=(",", ":")))
    tpl = open(f"{W}/samdata_ft.tpl.html").read()
    ver = hashlib.sha1(tpl.encode()).hexdigest()[:12]
    built = datetime.datetime.now(PT).strftime("%H:%M PT")
    head = ("<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content=\"width=device-width,initial-scale=1\">"
            "<title>SAM-data FT curves</title></head><body>")
    body = tpl.replace("__DATA__", json.dumps({"ver": ver, "built": built, "s3": s}, separators=(",", ":")).replace("</", "<\\/"))
    wr(f"{O}/samdata_ft.html", head + body + "</body></html>")
    wr(f"{O}/ver_samdata_ft.txt", ver)
    sm = s.get("sam") or {}
    print("samdata_ft.html", "★" if s.get("live") else "-", "job", s.get("jobid"), "step", s.get("step"), "/", s.get("max_steps"),
          "phase", s.get("phase"), "| SAM cmp job", sm.get("jobid"), sm.get("phase"), "train", len(sm.get("train") or []), "heldout", len(sm.get("heldout") or []),
          "| 23051", len(((s.get("ref") or {}).get("train")) or []),
          "|", " ".join(f"{k[1:]} job {v.get('jobid')} {v.get('phase')} step {v.get('step')}" for k, v in (s.get("fts") or {}).items()),
          "| scaling", " ".join(f"{q['init']}:{'final' if q.get('final') else ('s' + str(q['latest']['step']) if q.get('latest') else '-')}"
                                for q in (s.get("scaling") or [])) + (" ERR " + s["scaling_err"] if s.get("scaling_err") else ""), "ver", ver)


if __name__ == "__main__":
    main()
