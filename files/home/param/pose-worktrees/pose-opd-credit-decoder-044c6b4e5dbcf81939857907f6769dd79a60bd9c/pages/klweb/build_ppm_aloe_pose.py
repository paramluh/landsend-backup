#!/usr/bin/env python3
"""PPM Aloe POSE fine-tune (aloe-pose-ft: PPM fine-tune on the Aloe handover rows with POSE on channel B) live curve page.
Same standard live-curve format as build_ppm_aloe.py / ppm_aloe.tpl.html (stdlib only). Rebuilt every 60 s by ppm_aloe_pose_refresh.sbatch;
served by the klweb tunnel.

Reads  the newest /data/home/amartyavarma/ppm/aloe-pose-ft/runs/ppm-aloe-pose-ft-j<JOB>/metrics.jsonl (train lines: train_speech/*, train/*;
       eval lines: eval_speech/* = combined; per-side eval_userB_ / eval_asstB_ keys are deliberately NOT read) + squeue/sacct state
       (job name aloeposeft, used for the pending state before the run dir exists);
       /data/home/amartyavarma/ppm/aloe-pose-ft/PAGE_CFG.json on every rebuild (total_steps, rows_per_step, eval_every, exports, train_items, eval_items);
       the finished no-pose reference run 21980 (ppm/aloe-ft/runs/ppm-aloe-ft-j21980) for the muted overlay on the main word-CE chart.
Writes <OUT>/ppm_aloe_pose_live.json (polled by the page), <OUT>/ppm_aloe_pose.html (template ppm_aloe_pose.tpl.html with the data embedded),
       <OUT>/ver_ppm_aloe_pose.txt (= template hash: open pages reload only when the page itself changes), all atomically.
Usage: python3 build_ppm_aloe_pose.py [out_dir (default = this dir; use a scratch dir to test)]"""
import datetime
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

W = os.path.dirname(os.path.abspath(__file__))
O = sys.argv[1] if len(sys.argv) > 1 else W
BASE = "/data/home/amartyavarma/ppm/aloe-pose-ft"
RUNS = os.environ.get("PPMAP_RUNS", f"{BASE}/runs")  # env overrides for testing only
CFG = os.environ.get("PPMAP_CFG", f"{BASE}/PAGE_CFG.json")
JOBNAME = "aloeposeft"
REF = "/data/home/amartyavarma/ppm/aloe-ft/runs/ppm-aloe-ft-j21980/metrics.jsonl"
REF_TOTAL = 366
DEF = {"total_steps": 430, "rows_per_step": 64, "eval_every": 25, "exports": [], "train_items": None, "eval_items": None}
PT = ZoneInfo("America/Los_Angeles")
TCOLS = ["step", "rows", "nword", "loss", "cb0", "light", "pcb0", "plight",
         "lr_trunk", "lr_tower_s", "lr_tower_p", "lr_audio", "lr_decp_bb", "lr_dec", "lr_dec_p", "step_s"]
TKEYS = [None, None, "train_speech/word_ce_norm", "train/loss", "train_speech/cb0_ce", "train_speech/light_ce",
         "train_speech/pose_cb0_ce", "train_speech/pose_light_ce",
         "train/lr_trunk", "train/lr_tower_s", "train/lr_tower_p", "train/lr_audio", "train/lr_decp_bb",
         "train/dec_lr_last", "train/dec_p_lr_last", "train/step_time"]
ECOLS = ["step", "rows", "nword", "cb0", "light", "pcb0", "ptop1", "plight"]
EKEYS = [None, None, "eval_speech/word_ce_norm", "eval_speech/cb0_ce", "eval_speech/light_ce",
         "eval_speech/pose_cb0_ce", "eval_speech/pose_cb0_top1", "eval_speech/pose_light_ce"]
RCOLS = ["step", "rows", "nword"]
NSTORE = 8
LOGS = f"{BASE}/logs"
STORES = f"{BASE}/stores"
DATA_NOTE = ("18,093 qualifying B tracks · 28,517 rows counted (469 under 125 frames dropped) · 28,048 used: 27,517 train / 531 eval · ~3,092 h")
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
    try:
        j = json.load(open(CFG))
        if isinstance(j, dict):
            for k in DEF:
                if k in j and j[k] is not None:
                    c[k] = j[k]
            c["_cfg"] = True
    except Exception:
        c["_cfg"] = False
    c["total_steps"] = int(c["total_steps"])
    c["rows_per_step"] = int(c["rows_per_step"])
    c["eval_every"] = max(1, int(c["eval_every"]))
    c["exports"] = sorted(int(x) for x in (c["exports"] or []) if num(x))
    return c


def live_run():
    rs = [p for p in glob.glob(f"{RUNS}/ppm-aloe-pose-ft-j*") if p.rsplit("-j", 1)[1].isdigit()]
    return max(rs, key=lambda p: int(p.rsplit("-j", 1)[1])) if rs else None


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
            if "eval_speech/word_ce_norm" in d:
                ev[int(d["step"])] = d
            elif "train_speech/word_ce_norm" in d or "train/loss" in d:
                tr[int(d["step"])] = d
    except FileNotFoundError:
        pass
    return [tr[s] for s in sorted(tr)], [ev[s] for s in sorted(ev)]


def job_state(job):
    try:
        r = subprocess.run(["squeue", "-h", "-j", str(job), "-o", "%T|%N"], capture_output=True, text=True, timeout=20).stdout.strip()
        if r:
            return r.split("|")
        r = subprocess.run(["sacct", "-n", "-X", "-j", str(job), "-o", "State%20"], capture_output=True, text=True, timeout=20).stdout.split()
        return [r[0] if r else "ENDED", ""]
    except Exception:
        return ["?", ""]


def queued_job():
    """Newest queued/running job named aloeposeft (for the state before its run dir exists)."""
    try:
        r = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", "amartyavarma"), "-n", JOBNAME, "-o", "%i|%T|%N"],
                           capture_output=True, text=True, timeout=20).stdout.split()
        js = [x.split("|") for x in r if x.split("|")[0].isdigit()]
        return max(js, key=lambda x: int(x[0])) if js else None
    except Exception:
        return None


def sacct_name(name):
    """[(jobid, state)] for jobs of this name in the last 3 days, newest first."""
    try:
        r = subprocess.run(["sacct", "-n", "-X", "-P", "-u", os.environ.get("USER", "amartyavarma"), "--name", name, "-S", "now-3days",
                            "-o", "JobID,State"], capture_output=True, text=True, timeout=20).stdout.split()
        js = [x.split("|") for x in r if x.split("|")[0].isdigit()]
        return sorted(((j, st.split()[0] if st else "?") for j, st in js), key=lambda x: -int(x[0]))
    except Exception:
        return []


def prep_state():
    """Pre-training pipeline: 8 row-build jobs aloepose-build-p0..p7 (store done = stores/p<K>/BUILD_DONE), then aloepose-stage."""
    try:
        sq = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", "amartyavarma"), "-o", "%i|%j|%T|%N"],
                            capture_output=True, text=True, timeout=20).stdout.split()
    except Exception:
        sq = []
    q = {}
    for x in sq:
        p = x.split("|")
        if len(p) >= 3 and p[0].isdigit():
            q.setdefault(p[1], []).append((p[0], p[2], p[3] if len(p) > 3 else ""))
    stores = []
    for k in range(NSTORE):
        nm = f"aloepose-build-p{k}"
        done = os.path.exists(f"{STORES}/p{k}/BUILD_DONE")
        jobs = [int(f.rsplit("-", 1)[1][:-4]) for f in glob.glob(f"{LOGS}/build-{nm}-*.log") if f.rsplit("-", 1)[1][:-4].isdigit()]
        job = str(max(jobs)) if jobs else None
        st = "done" if done else None
        if not done:
            inq = sorted(q.get(nm, []), key=lambda x: -int(x[0]))
            if inq:
                job, st = inq[0][0], inq[0][1].lower()
            elif job:
                st = job_state(job)[0].lower()
            else:
                sa = sacct_name(nm)
                job, st = (sa[0][0], sa[0][1].lower()) if sa else (None, "not launched")
        stores.append({"k": k, "job": job, "st": st})
    out = {"stores": stores, "done": sum(x["st"] == "done" for x in stores), "n": NSTORE,
           "running": sum(x["st"] == "running" for x in stores)}
    sg = sorted(q.get("aloepose-stage", []), key=lambda x: -int(x[0]))
    if sg:
        out["stage"] = {"job": sg[0][0], "st": sg[0][1].lower(), "nodes": sg[0][2]}
    else:
        sa = sacct_name("aloepose-stage")
        if sa:
            out["stage"] = {"job": sa[0][0], "st": sa[0][1].lower(), "nodes": ""}
    return out


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
    s = {"max_steps": TOTAL, "rows": ROWS, "eval_every": c["eval_every"], "exports": c["exports"], "cfg": c["_cfg"],
         "train_items": c["train_items"], "eval_items": c["eval_items"],
         "tcols": TCOLS, "ecols": ECOLS, "train": [], "eval": [],
         "rcols": RCOLS, "ref_total": REF_TOTAL, "ref_train": rowify(rtr, [None, None, "train_speech/word_ce_norm"], 64),
         "ref_eval": rowify(rev, [None, None, "eval_speech/word_ce_norm"], 64),
         "data_note": DATA_NOTE,
         "updated": datetime.datetime.fromtimestamp(now, PT).strftime("%a %m-%d %H:%M:%S PT"), "phase": "waiting", "live": False}
    if rev:
        s["ref_ev_last"] = rev[-1]["eval_speech/word_ce_norm"]
        s["ref_ev_last_step"] = int(rev[-1]["step"])
    q = queued_job()
    if not run:
        if q:
            s.update(jobid=q[0], state=q[1], nodes=q[2] if len(q) > 2 else "", live=True, phase="pending" if q[1] == "PENDING" else "starting")
            return s
        p = prep_state()
        s["prep"] = p
        sg = p.get("stage")
        if sg and sg["st"] in [x.lower() for x in LIVE_ST]:
            s["phase"] = "staging"
        elif sg and sg["st"] == "completed":
            s["phase"] = "staged"
        elif sg and p["done"] == p["n"]:
            s["phase"] = "stage_failed"
        elif p["done"] == p["n"]:
            s["phase"] = "built"
        elif any(x["st"] != "not launched" for x in p["stores"]):
            s["phase"] = "building"
        return s
    job = run.rsplit("-j", 1)[1]
    if q and int(q[0]) > int(job):
        s["next_job"] = q[0]
        s["next_state"] = q[1]
    state, nodes = job_state(job)
    tr, ev = load(f"{run}/metrics.jsonl")
    live = state in ("RUNNING", "PENDING", "COMPLETING", "CONFIGURING")
    step = int(tr[-1]["step"]) if tr else None
    s.update(run=os.path.basename(run), jobid=job, state=state, nodes=nodes, live=live,
             train=rowify(tr, TKEYS, ROWS), eval=rowify(ev, EKEYS, ROWS), step=step)
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
    wr(f"{O}/ppm_aloe_pose_live.json", json.dumps(s, separators=(",", ":")))
    tpl = open(f"{W}/ppm_aloe_pose.tpl.html").read()
    ver = hashlib.sha1(tpl.encode()).hexdigest()[:12]
    head = "<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content=\"width=device-width,initial-scale=1\"></head><body>"
    wr(f"{O}/ppm_aloe_pose.html", head + tpl.replace("__DATA__", json.dumps({"ver": ver, "s": s}, separators=(",", ":")).replace("</", "<\\/")) + "</body></html>")
    wr(f"{O}/ver_ppm_aloe_pose.txt", ver)
    print("ppm_aloe_pose.html", "★" if s.get("live") else "-", "step", s.get("step"), "/", s.get("max_steps"), "phase", s.get("phase"), "ver", ver)


if __name__ == "__main__":
    main()
