#!/usr/bin/env python3
"""PPM Aloe v2pure fine-tune (aloe-v2pure-ft: PPM 32B, monoplex Aloe pose on the release-v2 data: channel A empty, B generates speech + pose;
recipe + metrics = job 22317 aloe-uncond-ft) live curve page.
= build_ppm_aloe_uncond.py (10-17 copy, standard live-curve format, stdlib only) with only: run dir aloe-v2pure-ft, outputs ppm_aloe_v2pure*,
template ppm_aloe_v2pure.tpl.html (= ppm_aloe_uncond.tpl.html + the 8B refs); job id also read from RUNNING_JOB.txt; the muted grey reference =
the finished uncond run 22317 (same recipe, Model-1 handover data); PLUS the 8B SAM pose fine-tunes as reference series on the four metrics
that mean the same on both models (pose cb0 CE, pose light cb1-31 CE, audio cb0 CE, audio light cb1-31 CE; train + the 8B's held-out eval).
Rebuilt every 60 s by ppm_aloe_v2pure_refresh.sbatch; served by the klweb tunnel (klweb-ppm, ppm_serve.sbatch).

Reads  /data/home/amartyavarma/ppm/aloe-v2pure-ft/RUNNING_JOB.txt (job id + nodes, written at launch; the job id = the first 5-7 digit number,
       preferring one after "job"), .../runs/$(cat RUN_TAG.txt)/metrics.jsonl (RUN_TAG.txt = 'ppm-aloe-v2pure-ft-j<FIRST job id>', kept across
       crash-resumes; until it exists: the newest runs/ppm-aloe-v2pure-ft-j*) (train lines: train_speech/*, train/*; eval lines: eval_speech/* =
       combined; per-side eval_userB_ / eval_asstB_ keys and the extra train/rounds, train/passab_* keys are NOT read; a step logged twice after a
       resume -> the LAST record wins). If metrics.jsonl has no records yet, falls back to the current job's trainer log logs/*-<JOB>.log
       (22317's step / EVAL line format). Current job = the newest of: RUNNING_JOB.txt, queued squeue job named aloe*v2pure*, log *-<JOB>.log,
       the run dir's job id (after a resume the current job is newer than the run dir's).
       .../aloe-v2pure-ft/PAGE_CFG.json on every rebuild (total_steps, rows_per_step, eval_every, exports, train_items, eval_items);
       22317 (ppm/aloe-uncond-ft/runs/ppm-aloe-uncond-ft-j22317/metrics.jsonl + its PAGE_CFG.json) for the grey reference;
       the 8B SAM pose FTs /data/home/amartyavarma/sam-aloe-ft-{v2pure,v2,posev3,posev2}/logs/log.jsonl + run.json (whichever exist):
       train lines (audio_cb0, audio_light, pose_cb0, pose_light per step) and "heldout" lines (HO_EVAL, 3,482 items / 113 held-out recordings).
Writes <OUT>/ppm_aloe_v2pure_live.json (polled by the page), <OUT>/ppm_aloe_v2pure.html (template with the data embedded),
       <OUT>/ver_ppm_aloe_v2pure.txt (= template hash: open pages reload only when the page itself changes), all atomically.
10-18 + clip80: the run aloe-v2pure-clip80 (= 23051 with ONE change, the system-prompt rule: no persona text, voice clip on a seeded 80 % of
       items; 23051: 1/3) as s.c80, NORMALISED WORD CE ONLY (train + combined eval), from /data/home/amartyavarma/ppm/aloe-v2pure-clip80 (same
       layout: RUN_TAG.txt 'ppm-aloe-v2pure-clip80-j<first job>', RUNNING_JOB.txt, runs/<RUN_TAG>/metrics.jsonl, PAGE_CFG.json, trainer log
       logs/aloeclip80-<job>.log, squeue name aloeclip80); last record per step wins; its one-off step-0 record eval_ref23051rule_speech/* (the
       init under 23051's prompt rule) is NOT its step-0 eval: kept apart as s.c80.idchk (one labelled marker). live (★) = job state live in
       squeue AND no 'job <J> ... ended' line in RUNNING_JOB.txt (same rule now for 23051). Until the job exists: phase 'waiting' (page: starting).
10-18 + posemono8020: the run aloe-pose-mono-8020 ("PPM · w-11488 · monoplex pose · Aloe rows · 80% clip / 20% persona" = the 23051 recipe
       (v2pure monoplex) on the Aloe handover pose rows: 22091's qualifying tracks, 1 item per track; prompt rule voice clip p 0.8, persona p 0.2,
       independent draws) as s.pm, built exactly like s.c80 (same function, side_doc): NORMALISED WORD CE ONLY (train + combined eval), from
       /data/home/amartyavarma/ppm/aloe-pose-mono-8020 (RUN_TAG.txt 'ppm-aloe-pose-mono-8020-j<first job>', RUNNING_JOB.txt,
       runs/<RUN_TAG>/metrics.jsonl, PAGE_CFG.json = 357 steps / 64 rows / eval 25 / 27,517 train / 531 eval items, trainer log
       logs/posemono8020-<job>.log, squeue name posemono8020, never posemono-release). Its own epoch = 357 steps (23051: 200): the page places it
       like 22317, at the fraction of its own epoch. Its one-off step-0 record eval_ref23051rule_speech/* = s.pm.idchk (one labelled marker, not
       its step-0 eval); the identity reference = 22317's step-0 eval_speech/word_ce_norm (1.9574: same Aloe eval items, same monoplex format).
Usage: python3 build_ppm_aloe_v2pure.py [out_dir (default = this dir; use a scratch dir to test)]
Test env overrides: PPMV2P_BASE (run dir root, e.g. to replay 22317 = /data/home/amartyavarma/ppm/aloe-uncond-ft), PPMV2P_PREFIX (run dir prefix),
       PPMV2P_NOQ=1 (ignore squeue), PPMV2P_C80_BASE (clip80 run dir root, e.g. a fake dir to test), PPMV2P_PM_BASE (posemono8020 run dir root)."""
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
BASE = os.environ.get("PPMV2P_BASE", f"{H}/ppm/aloe-v2pure-ft")   # env overrides for testing only
RUNS, LOGS, CFG, RJ = f"{BASE}/runs", f"{BASE}/logs", f"{BASE}/PAGE_CFG.json", f"{BASE}/RUNNING_JOB.txt"
RUNPFX = os.environ.get("PPMV2P_PREFIX", "ppm-aloe-v2pure-ft-j")
JOBRE = re.compile(r"^aloe(?!.*clip80).*v2pure", re.I)              # squeue job name of the training job (pending state before RUNNING_JOB.txt); never clip80's
# 10-18 clip80 run (= 23051 + one prompt-rule change): normalised word CE only
C80 = os.environ.get("PPMV2P_C80_BASE", f"{H}/ppm/aloe-v2pure-clip80")
C80_PFX = "ppm-aloe-v2pure-clip80-j"
C80_JOBRE = re.compile(r"^aloe.?clip80", re.I)                      # sbatch -J aloeclip80
C80_LOG = "aloeclip80-"                                             # trainer log logs/aloeclip80-<job>.log
C80_IDK = "eval_ref23051rule_speech/word_ce_norm"                   # step-0 identity check under 23051's prompt rule (not the run's own eval)
# 10-18 posemono8020 run (23051 recipe on the Aloe handover pose rows, 80% clip / 20% persona): normalised word CE only, built like clip80
PM = os.environ.get("PPMV2P_PM_BASE", f"{H}/ppm/aloe-pose-mono-8020")
PM_PFX = "ppm-aloe-pose-mono-8020-j"
PM_JOBRE = re.compile(r"^posemono.?8020", re.I)                     # sbatch -J posemono8020 (never posemono-release)
PM_LOG = "posemono8020-"                                            # trainer log logs/posemono8020-<job>.log
PM_DEF = {"total_steps": 357, "rows_per_step": 64, "eval_every": 25, "exports": [60, 119, 178, 238, 298, 357]}   # until its PAGE_CFG.json exists
REF = f"{H}/ppm/aloe-uncond-ft/runs/ppm-aloe-uncond-ft-j22317/metrics.jsonl"   # 22317 = same recipe on the Model-1 handover rows (reference)
REF_CFG = f"{H}/ppm/aloe-uncond-ft/PAGE_CFG.json"
DEF = {"total_steps": 195, "rows_per_step": 64, "eval_every": 25, "exports": [], "train_items": None, "eval_items": None}
PT = ZoneInfo("America/Los_Angeles")
TCOLS = ["step", "rows", "nword", "loss", "cb0", "light", "pcb0", "plight",
         "lr_trunk", "lr_tower_s", "lr_tower_p", "lr_audio", "lr_decp_bb", "lr_dec", "lr_dec_p", "step_s", "loss_u"]
TKEYS = [None, None, "train_speech/word_ce_norm", "train/loss", "train_speech/cb0_ce", "train_speech/light_ce",
         "train_speech/pose_cb0_ce", "train_speech/pose_light_ce",
         "train/lr_trunk", "train/lr_tower_s", "train/lr_tower_p", "train/lr_audio", "train/lr_decp_bb",
         "train/dec_lr_last", "train/dec_p_lr_last", "train/step_time",
         "train/loss_unweighted"]   # train/loss = weighted (B word lane x10, optimised); loss_unweighted = the loss without the x10
ECOLS = ["step", "rows", "nword", "cb0", "light", "pcb0", "ptop1", "plight"]
EKEYS = [None, None, "eval_speech/word_ce_norm", "eval_speech/cb0_ce", "eval_speech/light_ce",
         "eval_speech/pose_cb0_ce", "eval_speech/pose_cb0_top1", "eval_speech/pose_light_ce"]
# 8B SAM pose fine-tunes (reference series): our column <- the 8B log key. Same Mimi + same pose tokenizer; word CE NOT comparable (text upfront).
SCOLS = ["step", "cb0", "light", "pcb0", "plight"]
SKEYS = [None, "audio_cb0", "audio_light", "pose_cb0", "pose_light"]
SAM = [  # (key, run dir, label, colour var, main, note: measured 10-17 from the eval caches / logs)
    ("v2pure", f"{H}/sam-aloe-ft-v2pure", "8B v2pure · Xabi’s data (ref)", "--rt1", True,
     "trained on Xabi’s Aloe pose release v2 (= this run's data); held-out = 3,482 items from the same 113 recordings as our eval, same Mimi + pose targets"),
    ("v2", f"{H}/sam-aloe-ft-v2", "8B v2 (ref)", "--rt2", False,
     "held-out items, Mimi codes and pose codes byte-identical to v2pure's"),
    ("posev3", f"{H}/sam-aloe-ft-posev3", "8B posev3 (ref)", "--rt3", False,
     "same held-out items + pose codes; its Mimi codes are the pre-v2 audio (step-0 held-out audio cb0 1.82 vs 3.08 on v2): audio CE on other targets"),
    ("posev2", f"{H}/sam-aloe-ft-posev2", "8B posev2 (ref)", "--rt4", False,
     "older pose release (held-out cache gone; step-0 held-out pose cb0 1.92 vs 1.71): pose CE on other targets; audio = posev3's"),
]
LIVE_ST = ("RUNNING", "PENDING", "COMPLETING", "CONFIGURING")


def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def parse_pt(s):
    try:
        d = datetime.datetime.strptime("2026-" + s.replace(" PT", ""), "%Y-%m-%d %H:%M:%S")
        return d.replace(tzinfo=PT).timestamp()
    except Exception:
        return None


def cfg():
    c = dict(DEF)
    j = None
    try:
        j = json.load(open(CFG))
        if isinstance(j, dict):
            for k in DEF:
                if k in j and j[k] is not None:
                    c[k] = j[k]
            c["_cfg"] = True
    except Exception:
        c["_cfg"] = False
    try:   # PAGE_CFG.json still the verbatim copy of 22317's (not yet rewritten for this run)
        c["_stale"] = bool(c["_cfg"]) and j == json.load(open(REF_CFG)) and not BASE.endswith("aloe-uncond-ft")
    except Exception:
        c["_stale"] = False
    c["total_steps"] = int(c["total_steps"])
    c["rows_per_step"] = int(c["rows_per_step"])
    c["eval_every"] = max(1, int(c["eval_every"]))
    c["exports"] = sorted(int(x) for x in (c["exports"] or []) if num(x))
    return c


def _jid(p, pfx, sfx=""):
    t = os.path.basename(p)[len(pfx):]
    t = t[:-len(sfx)] if sfx and t.endswith(sfx) else t
    return int(t) if t.isdigit() else None


def live_run(base=BASE, pfx=RUNPFX):
    """RUN_TAG.txt (written at launch, kept across resumes) names the run dir; else the newest run dir."""
    runs = f"{base}/runs"
    try:
        t = open(f"{base}/RUN_TAG.txt").read().strip()
        if t and os.path.isdir(f"{runs}/{t}"):
            return f"{runs}/{t}"
    except Exception:
        pass
    rs = [p for p in glob.glob(f"{runs}/{pfx}*") if _jid(p, pfx) is not None]
    return max(rs, key=lambda p: _jid(p, pfx)) if rs else None


def log_of(job, logs=LOGS, name="*"):
    """The trainer log of a job: logs/<name>-<JOB>.log (any name except the export / GCS helper logs)."""
    ls = [p for p in glob.glob(f"{logs}/{name}-{job}.log") if not os.path.basename(p).startswith(("gcs", "export", "prune"))]
    return sorted(ls)[0] if ls else None


def newest_log_job(logs=LOGS, name=""):
    js = []
    for p in glob.glob(f"{logs}/{name}*-*.log"):
        b = os.path.basename(p)
        m = re.match(r"^(.+)-(\d{5,7})\.log$", b)
        if m and not b.startswith(("gcs", "export", "prune")):
            js.append(int(m.group(2)))
    return max(js) if js else None


def running_job_file(rj=RJ):
    """RUNNING_JOB.txt (written at launch): job id = a 5-7 digit number after 'job' / 'id', else the first 5-7 digit number."""
    try:
        t = open(rj).read()
    except Exception:
        return None
    m = re.search(r"(?:job|id)\D{0,6}(\d{5,7})\b", t, re.I) or re.search(r"\b(\d{5,7})\b", t)
    return int(m.group(1)) if m else None


def ended_line(rj, job):
    """RUNNING_JOB.txt 'job <J> nodes .. ended YYYY-MM-DD HH:MM:SS PT rc N' (appended by the run script on exit) -> (ended PT string, rc) or None."""
    try:
        for line in open(rj, errors="replace"):
            m = re.search(r"\bjob (\d{5,7})\b.*\bended (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d PT)(?: rc (-?\d+))?", line)
            if m and int(m.group(1)) == int(job):
                return m.group(2), (int(m.group(3)) if m.group(3) is not None else None)
    except Exception:
        pass
    return None


TRK = ("word_ce_norm", "pad_ce", "cb0_ce", "light_ce", "pose_cb0_ce", "pose_light_ce")   # speech keys on the trainer's step lines


def load_log(path):
    """Fallback when metrics.jsonl has no records yet: rebuild the same records from the trainer log (22317 format).
    step line: "[MM-DD HH:MM:SS PT] step N/T 11.6s (micro .. wait ..) peak 57.3 GiB | word_ce_norm=.. cb0_ce=.. lr_trunk=.. wloss_word=.. | ETA x min"
    eval line: "[MM-DD HH:MM:SS PT] EVAL step N: eval_speech/word_ce_norm 1.8576 | eval_speech/... | eval/time_s 25.59". Unknown keys are kept, never plotted."""
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
    tr, ev = {}, {}
    try:
        for line in open(path, errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if not isinstance(d, dict) or not num(d.get("step")):
                continue
            # clip80's / posemono8020's one-off step-0 identity record (eval_ref23051rule_speech/* only, no eval_speech/* key) is never an eval / train record
            if "eval_speech/word_ce_norm" in d:
                ev[int(d["step"])] = d
            elif "train_speech/word_ce_norm" in d or "train/loss" in d:
                tr[int(d["step"])] = d
    except FileNotFoundError:
        pass
    return [tr[s] for s in sorted(tr)], [ev[s] for s in sorted(ev)]


def load_sam():
    """8B SAM pose FT logs -> [{key, label, col, main, note, total, warmup, rows, train: [[step, cb0, light, pcb0, plight]], eval: [...]}].
    train = the per-step lines (have 'loss'); eval = the 'heldout' lines (HO_EVAL); the 8B 'eval' lines (1 in-distribution row / rank) are skipped."""
    out = []
    for key, d, label, col, main, note in SAM:
        lj = f"{d}/logs/log.jsonl"
        if not os.path.exists(lj):
            continue
        try:
            rj = json.load(open(f"{d}/logs/run.json"))
        except Exception:
            rj = {}
        tr, ho = {}, {}
        for line in open(lj, errors="replace"):
            try:
                x = json.loads(line)
            except Exception:
                continue
            if not isinstance(x, dict) or not num(x.get("step")):
                continue
            if isinstance(x.get("heldout"), dict):
                ho[int(x["step"])] = x["heldout"]
            elif "eval" in x:
                continue
            elif "loss" in x:
                tr[int(x["step"])] = x
        if not tr and not ho:
            continue
        total = int(rj.get("total_steps") or (max(tr) if tr else 0) or 1)
        row = lambda s, r: [s] + [(round(r[k], 6) if num(r.get(k)) else None) for k in SKEYS[1:]]   # noqa: E731
        out.append({"key": key, "label": label, "col": col, "main": main, "note": note, "total": total,
                    "warmup": int(rj.get("warmup") or 0), "rows": int(rj.get("world") or 0),
                    "train": [row(s, tr[s]) for s in sorted(tr)], "eval": [row(s, ho[s]) for s in sorted(ho)],
                    "done": bool(tr) and max(tr) >= total})
    return out


def job_state(job):
    try:
        r = subprocess.run(["squeue", "-h", "-j", str(job), "-o", "%T|%N"], capture_output=True, text=True, timeout=20).stdout.strip()
        if r:
            return r.split("|")
        r = subprocess.run(["sacct", "-n", "-X", "-j", str(job), "-o", "State%20"], capture_output=True, text=True, timeout=20).stdout.split()
        return [r[0] if r else "ENDED", ""]
    except Exception:
        return ["?", ""]


def queued_job(jobre=JOBRE):
    """Newest queued/running job whose name matches aloe*v2pure* (clip80: aloeclip80) (for the state before its run dir exists)."""
    if os.environ.get("PPMV2P_NOQ"):
        return None
    try:
        r = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", "amartyavarma"), "-o", "%i|%T|%N|%j"],
                           capture_output=True, text=True, timeout=20).stdout.split()
        js = [x.split("|") for x in r if x.count("|") >= 3 and x.split("|")[0].isdigit() and jobre.match(x.split("|")[3])]
        return max(js, key=lambda x: int(x[0]))[:3] if js else None
    except Exception:
        return None


def rowify(recs, keys, rows):
    out = []
    for d in recs:
        s = int(d["step"])
        out.append([s, s * rows] + [(round(d[k], 9) if num(d.get(k)) else None) for k in keys[2:]])
    return out


def doc():
    c = cfg()
    TOTAL, ROWS = c["total_steps"], c["rows_per_step"]
    run = live_run()
    now = time.time()
    rtr, rev = load(REF)
    try:
        rc = json.load(open(REF_CFG))
    except Exception:
        rc = {}
    s = {"max_steps": TOTAL, "rows": ROWS, "eval_every": c["eval_every"], "exports": c["exports"], "cfg": c["_cfg"], "cfg_stale": c["_stale"],
         "train_items": c["train_items"], "eval_items": c["eval_items"],
         "tcols": TCOLS, "ecols": ECOLS, "train": [], "eval": [],
         "ref_total": int(rc.get("total_steps") or 430), "ref_rows": int(rc.get("rows_per_step") or 64),
         "ref_train_items": rc.get("train_items"), "ref_eval_items": rc.get("eval_items"),
         "ref_train": rowify(rtr, TKEYS, int(rc.get("rows_per_step") or 64)), "ref_eval": rowify(rev, EKEYS, int(rc.get("rows_per_step") or 64)),
         "scols": SCOLS, "sam": load_sam(),
         "updated": datetime.datetime.fromtimestamp(now, PT).strftime("%a %m-%d %H:%M:%S PT"), "phase": "waiting", "live": False}
    if (c["train_items"] or c["eval_items"]) and not c["_stale"]:   # a stale 22317 copy would mislabel 22317's counts as release-v2
        s["data_note"] = "current: Xabi’s Aloe pose release v2" + (f" · {c['train_items']:,} train clips" if c["train_items"] else "") + \
            (f" · {c['eval_items']:,} eval clips (from the 8B's 113 held-out recordings)" if c["eval_items"] else "")
    if rev:
        s["ref_ev_last"] = rev[-1]["eval_speech/word_ce_norm"]
        s["ref_ev_last_step"] = int(rev[-1]["step"])
    # run dir = RUN_TAG.txt ('ppm-aloe-v2pure-ft-j<FIRST job id>', kept across crash-resumes), else the newest run dir; the job shown = the
    # NEWEST of RUNNING_JOB.txt / queued squeue job / trainer log / run-dir job (after a resume it is newer than the run dir's job id)
    q = queued_job()
    rj = running_job_file()
    lj = newest_log_job()
    rjob = _jid(run, RUNPFX) if run else None
    jobs = [x for x in ((int(q[0]) if q else None), rj, lj, rjob) if x]
    if not jobs:
        return s
    job = max(jobs)
    if q and int(q[0]) == job:
        state, nodes = q[1], (q[2] if len(q) > 2 else "")
    else:
        state, nodes = job_state(job)
    tr, ev = load(f"{run}/metrics.jsonl") if run else ([], [])   # last record per step (a crash-resume can repeat steps)
    if not tr and not ev:   # metrics.jsonl missing / empty: same records from the trainer log
        tr, ev = load_log(log_of(job))
        if tr or ev:
            s["src"] = "log"
    end = ended_line(RJ, job)   # 10-18: RUNNING_JOB.txt 'ended' line for this job = finished (no ★), whatever squeue still says
    live = state in LIVE_ST and not end
    step = int(tr[-1]["step"]) if tr else None
    s.update(jobid=str(job), state=state, nodes=nodes, live=live, train=rowify(tr, TKEYS, ROWS), eval=rowify(ev, EKEYS, ROWS), step=step)
    if end:
        s["ended_pt"] = end[0][5:]
    if run:
        s["run"] = os.path.basename(run)
        if rjob and rjob != job:
            s["resumed_from"] = str(rjob)
    if state == "PENDING":
        s["phase"] = "pending"
    elif live:
        s["phase"] = "training" if tr else "starting"
    else:
        s["phase"] = "done" if (state == "COMPLETED" or (step or 0) >= TOTAL) else "ended"
    if tr:
        last = tr[-1]
        s["last_pt"] = last.get("time_pt")
        s["t_last"] = parse_pt(last.get("time_pt", ""))
        st = [r["train/step_time"] for r in tr[-20:] if num(r.get("train/step_time"))]
        if st:
            s["mean_s"] = round(sum(st) / len(st), 2)
        if num(last.get("train/peak_gib")):
            s["peak_gib"] = round(last["train/peak_gib"], 1)
        fut = {x for x in range(c["eval_every"], TOTAL + 1, c["eval_every"])} | set(c["exports"])
        s["evals_left"] = len([x for x in fut if x > step])
    if ev:
        e = ev[-1]
        s["ev_last_step"] = int(e["step"])
        s["ev_last_pt"] = e.get("time_pt")
        for k, key in (("ev_last", "eval_speech/word_ce_norm"), ("ev_pcb0", "eval_speech/pose_cb0_ce"), ("ev_ptop1", "eval_speech/pose_cb0_top1"),
                       ("ev_plight", "eval_speech/pose_light_ce"), ("ev_cb0", "eval_speech/cb0_ce"), ("ev_light", "eval_speech/light_ce")):
            if num(e.get(key)):
                s[k] = e[key]
        es = [x["eval/time_s"] for x in ev if num(x.get("eval/time_s"))]
        if es:
            s["eval_s"] = round(sum(es) / len(es), 1)
    return s


C80_TKEYS = [None, None, "train_speech/word_ce_norm"]   # clip80 / posemono8020: normalised word CE only (cols step, rows, nword)
C80_EKEYS = [None, None, "eval_speech/word_ce_norm"]    # combined eval only (per-side eval_asstB_ / eval_userB_ not shown, as for 23051)


def side_doc(base, pfx, jobre, logn, fb, ref0):
    """A side run (clip80 / posemono8020): normalised word CE only, same job / phase rules as 23051's. fb = run parameters until its
    PAGE_CFG.json exists; ref0 = the reference value of its one-off step-0 identity check (eval_ref23051rule_speech/word_ce_norm)."""
    c = dict(DEF)
    c.update(fb)
    try:
        j = json.load(open(f"{base}/PAGE_CFG.json"))
        for k in DEF:
            if isinstance(j, dict) and j.get(k) is not None:
                c[k] = j[k]
        c["_cfg"] = True
    except Exception:
        c["_cfg"] = False
    TOTAL, ROWS, EV = int(c["total_steps"]), int(c["rows_per_step"]), max(1, int(c["eval_every"]))
    exports = sorted(int(x) for x in (c["exports"] or []) if num(x))
    r = {"max_steps": TOTAL, "rows": ROWS, "eval_every": EV, "exports": exports, "cfg": c["_cfg"],
         "tcols": ["step", "rows", "nword"], "ecols": ["step", "rows", "nword"], "train": [], "eval": [], "phase": "waiting", "live": False}
    for k in ("train_items", "eval_items"):
        if c.get(k) is not None:
            r[k] = c[k]
    run = live_run(base, pfx)
    rjf = f"{base}/RUNNING_JOB.txt"
    q = queued_job(jobre)
    rj = running_job_file(rjf)
    lre = re.compile(r"^" + re.escape(logn) + r"(\d{5,7})\.log$")   # trainer log <logn><job>.log only (never a helper / release log)
    ljs = [int(m.group(1)) for m in (lre.match(os.path.basename(p)) for p in glob.glob(f"{base}/logs/{logn}*.log")) if m]
    lj = max(ljs) if ljs else None
    rjob = _jid(run, pfx) if run else None
    jobs = [x for x in ((int(q[0]) if q else None), rj, lj, rjob) if x]
    if not jobs:
        return r
    job = max(jobs)
    if q and int(q[0]) == job:
        state, nodes = q[1], (q[2] if len(q) > 2 else "")
    else:
        state, nodes = job_state(job)
    end = ended_line(rjf, job)
    met = f"{run}/metrics.jsonl" if run else None
    tr, ev = load(met) if met else ([], [])   # last record per step (a crash-resume can repeat steps)
    if not tr and not ev:
        tr, ev = load_log(log_of(job, f"{base}/logs", logn.rstrip("-")))
        if tr or ev:
            r["src"] = "log"
    idc = None   # the one-off step-0 identity record (last one wins, like every step)
    try:
        for line in open(met, errors="replace"):
            if C80_IDK in line:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if isinstance(d, dict) and num(d.get(C80_IDK)):
                    idc = d
    except (FileNotFoundError, TypeError):
        pass
    live = state in LIVE_ST and not end
    step = int(tr[-1]["step"]) if tr else None
    r.update(jobid=str(job), state=state, nodes=nodes, live=live, step=step,
             train=rowify(tr, C80_TKEYS, ROWS), eval=rowify(ev, C80_EKEYS, ROWS))
    if run:
        r["run"] = os.path.basename(run)
        if rjob and rjob != job:
            r["resumed_from"] = str(rjob)
    if end:
        r["ended_pt"] = end[0][5:]
        r["rc"] = end[1]
    if state == "PENDING" and not end:
        r["phase"] = "pending"
    elif live:
        r["phase"] = "training" if tr else "starting"
    else:
        r["phase"] = "done" if (step or 0) >= TOTAL else "ended"
    if tr:
        last = tr[-1]
        r["last_pt"] = last.get("time_pt")
        r["t_last"] = parse_pt(last.get("time_pt", ""))
        st = [x["train/step_time"] for x in tr[-20:] if num(x.get("train/step_time"))]
        if st:
            r["mean_s"] = round(sum(st) / len(st), 2)
        if num(last.get("train/peak_gib")):
            r["peak_gib"] = round(last["train/peak_gib"], 1)
        fut = set(range(EV, TOTAL + 1, EV)) | set(exports)
        r["evals_left"] = len([x for x in fut if x > step])
    if ev:
        e = ev[-1]
        r["ev_last_step"] = int(e["step"])
        r["ev_last_pt"] = e.get("time_pt")
        r["ev_last"] = e["eval_speech/word_ce_norm"]
        es = [x["eval/time_s"] for x in ev if num(x.get("eval/time_s"))]
        if es:
            r["eval_s"] = round(sum(es) / len(es), 1)
    if idc:
        r["idchk"] = {"v": idc[C80_IDK], "step": int(idc["step"]), "pt": idc.get("time_pt"), "ref": ref0}
    return r


def c80_doc(s):
    """clip80 run state; s = 23051's doc: its parameters = the fallback until clip80's PAGE_CFG.json exists (same recipe), its step-0 eval =
    the identity reference."""
    fb = {k_: s[v_] for k_, v_ in (("total_steps", "max_steps"), ("rows_per_step", "rows"), ("eval_every", "eval_every"), ("exports", "exports"))
          if s.get(v_) is not None}
    e23 = [x for x in (s.get("eval") or []) if x and x[0] == 0]   # 23051's own step-0 eval (row: step, rows, nword, ...)
    ref0 = e23[0][2] if e23 and num(e23[0][2]) else None
    return side_doc(C80, C80_PFX, C80_JOBRE, C80_LOG, fb, ref0)


def pm_doc(s):
    """posemono8020 run state; identity reference = 22317's step-0 eval_speech/word_ce_norm (the grey reference: same Aloe eval items, same
    monoplex format; 1.9574)."""
    rows = s.get("ref_eval")
    if rows is None:   # 23051's doc failed: read 22317 directly
        rows = rowify(load(REF)[1], EKEYS, 64)
    e22 = [x for x in rows if x and x[0] == 0]
    ref0 = e22[0][2] if e22 and num(e22[0][2]) else None
    return side_doc(PM, PM_PFX, PM_JOBRE, PM_LOG, PM_DEF, ref0)


def wr(path, text):
    open(path + ".tmp", "w").write(text)
    os.replace(path + ".tmp", path)


def main():
    try:
        s = doc()
    except Exception as ex:
        import traceback
        traceback.print_exc()
        s = {"phase": f"builder error: {ex!r}", "max_steps": DEF["total_steps"], "rows": DEF["rows_per_step"]}
    try:   # clip80 apart: an error there never takes down 23051's page
        s["c80"] = c80_doc(s)
    except Exception as ex:
        import traceback
        traceback.print_exc()
        s["c80"] = {"phase": f"builder error: {ex!r}", "live": False, "train": [], "eval": []}
    try:   # posemono8020 apart too: an error there never takes down the rest of the page
        s["pm"] = pm_doc(s)
    except Exception as ex:
        import traceback
        traceback.print_exc()
        s["pm"] = {"phase": f"builder error: {ex!r}", "live": False, "train": [], "eval": []}
    wr(f"{O}/ppm_aloe_v2pure_live.json", json.dumps(s, separators=(",", ":")))
    tpl = open(f"{W}/ppm_aloe_v2pure.tpl.html").read()
    ver = hashlib.sha1(tpl.encode()).hexdigest()[:12]
    head = "<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content=\"width=device-width,initial-scale=1\"></head><body>"
    wr(f"{O}/ppm_aloe_v2pure.html", head + tpl.replace("__DATA__", json.dumps({"ver": ver, "s": s}, separators=(",", ":")).replace("</", "<\\/")) + "</body></html>")
    wr(f"{O}/ver_ppm_aloe_v2pure.txt", ver)
    c8, pm = s.get("c80") or {}, s.get("pm") or {}
    print("ppm_aloe_v2pure.html", "★" if s.get("live") else "-", "job", s.get("jobid"), "step", s.get("step"), "/", s.get("max_steps"),
          "phase", s.get("phase"), "8B refs", len(s.get("sam") or []), "| clip80", "★" if c8.get("live") else "-", "job", c8.get("jobid"),
          "step", c8.get("step"), "/", c8.get("max_steps"), "phase", c8.get("phase"), "| posemono8020", "★" if pm.get("live") else "-",
          "job", pm.get("jobid"), "step", pm.get("step"), "/", pm.get("max_steps"), "phase", pm.get("phase"), "ver", ver)


if __name__ == "__main__":
    main()
