#!/usr/bin/env python3
"""PPM Aloe v2 fine-tune (aloe-v2-ft, job 22383+: first chunks only, no-text system prompt + B voice clip on 30 %, decoder K 1 on 10 % of frames)
= build_ppm_aloe.py (21980's page) with only: run dir aloe-v2-ft, 253 steps, saves 42/84/126/169/211/253, outputs ppm_aloe_v2*, template ppm_aloe_v2.tpl.html,
+ the finished 21980 run as a muted reference overlay (ref_train / ref_eval, same columns). Refreshed by ppm_aloe_v2_refresh.sbatch (slinky-15).
Original doc: PPM Aloe fine-tune (aloe-ft) live curve page, standard live-curve format (= s4qqa.tpl.html / stage3.html engine; stdlib only).
Rebuilt every 60 s by ppm_aloe_refresh.sbatch; served by the klweb tunnel.

Reads  the newest /data/home/amartyavarma/ppm/aloe-ft/runs/ppm-aloe-ft-j<JOB>/metrics.jsonl (train lines: train_speech/*, train/*;
       eval lines: eval_speech/* = combined; the per-side eval_userB_ / eval_asstB_ keys are deliberately NOT read) + squeue/sacct state.
Writes <OUT>/ppm_aloe_v2_live.json (polled by the page), <OUT>/ppm_aloe.html (template ppm_aloe_v2.tpl.html with the data embedded),
       <OUT>/ver_ppm_aloe_v2.txt (= template hash: open pages reload only when the page itself changes), all atomically.
Usage: python3 build_ppm_aloe.py [out_dir (default = this dir; use a scratch dir to test)]
Previous custom SVG builder: build_ppm_aloe.py.bak-pre-stdformat."""
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
RUNS = "/data/home/amartyavarma/ppm/aloe-v2-ft/runs"
REF = "/data/home/amartyavarma/ppm/aloe-ft/runs/ppm-aloe-ft-j21980/metrics.jsonl"   # 21980 = the base speech+text Aloe FT (muted overlay)
REF_TOTAL = 366
TOTAL = 253
ROWS = 64
EVAL_EVERY = 25
EXPORTS = (42, 84, 126, 169, 211, 253)
PT = ZoneInfo("America/Los_Angeles")
TCOLS = ["step", "rows", "nword", "loss", "cb0", "light", "lr_trunk", "lr_tower", "lr_audio", "step_s"]
TKEYS = [None, None, "train_speech/word_ce_norm", "train/loss", "train_speech/cb0_ce", "train_speech/light_ce",
         "train/lr_trunk", "train/lr_tower_s", "train/lr_audio", "train/step_time"]
ECOLS = ["step", "rows", "nword", "cb0", "light"]
EKEYS = [None, None, "eval_speech/word_ce_norm", "eval_speech/cb0_ce", "eval_speech/light_ce"]


def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def parse_pt(s):
    try:
        d = datetime.datetime.strptime("2026-" + s.replace(" PT", ""), "%Y-%m-%d %H:%M:%S")
        return d.replace(tzinfo=PT).timestamp()
    except Exception:
        return None


def live_run():
    rs = [p for p in glob.glob(f"{RUNS}/ppm-aloe-v2-ft-j*") if p.rsplit("-j", 1)[1].isdigit()]
    return max(rs, key=lambda p: int(p.rsplit("-j", 1)[1])) if rs else None


def load(run):
    tr, ev = {}, {}
    try:
        for line in open(f"{run}/metrics.jsonl", errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if not num(d.get("step")):
                continue
            if "eval_speech/word_ce_norm" in d:
                ev[int(d["step"])] = d
            elif "train_speech/word_ce_norm" in d:
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


def rowify(recs, keys):
    out = []
    for d in recs:
        s = int(d["step"])
        out.append([s, s * ROWS] + [(round(d[k], 9) if num(d.get(k)) else None) for k in keys[2:]])
    return out


def doc():
    run = live_run()
    now = time.time()
    s = {"max_steps": TOTAL, "exports": list(EXPORTS), "tcols": TCOLS, "ecols": ECOLS, "train": [], "eval": [],
         "updated": datetime.datetime.fromtimestamp(now, PT).strftime("%a %m-%d %H:%M:%S PT"), "phase": "waiting", "live": False}
    rtr, rev = load(REF.rsplit("/", 1)[0])
    s.update(ref_total=REF_TOTAL, ref_train=rowify(rtr, TKEYS), ref_eval=rowify(rev, EKEYS))
    if rev:
        s["ref_ev0"] = rev[0]["eval_speech/word_ce_norm"]; s["ref_ev_last"] = rev[-1]["eval_speech/word_ce_norm"]; s["ref_ev_last_step"] = int(rev[-1]["step"])
    if not run:
        return s
    job = run.rsplit("-j", 1)[1]
    state, nodes = job_state(job)
    tr, ev = load(run)
    live = state in ("RUNNING", "PENDING", "COMPLETING", "CONFIGURING")
    step = int(tr[-1]["step"]) if tr else None
    s.update(run=os.path.basename(run), jobid=job, state=state, nodes=nodes, live=live, train=rowify(tr, TKEYS), eval=rowify(ev, EKEYS), step=step)
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
        fut = {x for x in range(EVAL_EVERY, TOTAL + 1, EVAL_EVERY)} | set(EXPORTS)
        s["evals_left"] = len([x for x in fut if x > step])
    if ev:
        s["ev_last"] = ev[-1]["eval_speech/word_ce_norm"]
        s["ev_last_step"] = int(ev[-1]["step"])
        es = [e["eval/time_s"] for e in ev if num(e.get("eval/time_s"))]
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
        s = {"phase": f"builder error: {ex!r}", "max_steps": TOTAL}
    wr(f"{O}/ppm_aloe_v2_live.json", json.dumps(s, separators=(",", ":")))
    tpl = open(f"{W}/ppm_aloe_v2.tpl.html").read()
    ver = hashlib.sha1(tpl.encode()).hexdigest()[:12]
    head = "<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content=\"width=device-width,initial-scale=1\"></head><body>"
    wr(f"{O}/ppm_aloe_v2.html", head + tpl.replace("__DATA__", json.dumps({"ver": ver, "s": s}, separators=(",", ":")).replace("</", "<\\/")) + "</body></html>")
    wr(f"{O}/ver_ppm_aloe_v2.txt", ver)
    print("ppm_aloe_v2.html", "★" if s.get("live") else "-", "step", s.get("step"), "phase", s.get("phase"), "ver", ver)


if __name__ == "__main__":
    main()
