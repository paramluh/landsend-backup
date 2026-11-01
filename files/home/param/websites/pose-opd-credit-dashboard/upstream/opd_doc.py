#!/usr/bin/env python3
"""opd_doc.py <out.json>: one OPD (on-policy distillation) run -> the live-page doc that build_stage3.py / ppm.tpl.html read (same keys as aloe_doc.py;
S3_PAGE=opd: s3branches_opd.json "doc" for the main run and every branch). 10-29 (user: "concurrently set up my own style curves page" for the OPD
reproduction of Param's run; "i dont want to plot on wandb").
Sources (OPD_SRC, '|' = several candidates, the newest existing file wins):
  * a live metrics.jsonl (our run, ppm/opd/run24/metrics.jsonl): one JSON object per update, Param's metric names (train/lexical_kl, timing/round_seconds, ...);
  * a W&B history export {config, summary, history: [rows]} (Param's reference runs, ppm/opd/param-curves/*.json; local copies, never W&B).
x = policy/version (else _step, else step): a later row for an update already seen (a restart) replaces it and every later one; a row without a version
(an eval logged between updates) belongs to the update before the next versioned row. Train columns = OPD_TRAIN_KEYS (+ every key under OPD_MORE_PREFIXES,
the MORE tab); timing/round_seconds is the column "train/step_time" (the template's step-time key: header "last update" + s/update). Eval column =
genghis/successes (points at saves). Header: OPD_MAX_STEPS updates; checkpoint every OPD_SAVE_EVERY_MIN min of wall clock (first update that ends past
the last save, as Param's trainer: interval_seconds 5400 checked after each update) + the final update; GenghisBench right after each save."""
import glob, json, os, re, subprocess, sys, datetime, zoneinfo
OUT = sys.argv[1]
RUN = os.environ.get("S3_RUN", "opd")
SRC = os.environ.get("OPD_SRC", "")
MAXS = int(os.environ.get("OPD_MAX_STEPS", "110"))
SAVE_MIN = float(os.environ.get("OPD_SAVE_EVERY_MIN", "90"))
RPS = int(os.environ.get("OPD_ROWS_PER_STEP", "384"))          # answers per update
EPR = int(os.environ.get("OPD_EPOCH_ROWS", "84319"))           # admitted questions (< 20 s) = 1 epoch; 110 updates = half
PT = zoneinfo.ZoneInfo("America/Los_Angeles")
TRAIN = ["train/lexical_kl", "train/loss", "train/category_kl", "train/aloe_category_ce", "train/aloe_cb0_ce", "train/aloe_depth_ce",
         "train/opd", "train/aloe_ar", "train/weighted_lexical_kl", "train/weighted_category_kl", "train/weighted_aloe_ar", "train/weighted_opd",
         "timing/round_seconds", "timing/rollout_phase_seconds", "timing/teacher_phase_seconds", "timing/critic_phase_seconds",
         "timing/learner_phase_seconds", "timing/publication_seconds", "train/learning_rate", "grad_norm/pre_clip", "grad_norm/post_clip"]
TRAIN = [k for k in os.environ.get("OPD_TRAIN_KEYS", ",".join(TRAIN)).split(",") if k]
MORE = tuple(p for p in os.environ.get("OPD_MORE_PREFIXES", "kl/,signal/,timing_rollout/,timing_credit/").split(",") if p)   # 10-29 nextrun: + rollout timing stats (TIMING GATE tab)
EVAL = [k for k in os.environ.get("OPD_EVAL_KEYS", "genghis/successes").split(",") if k]
ALIAS = {"timing/round_seconds": "train/step_time"}


def pick_src():
    c = [p for g in SRC.split("|") if g for p in glob.glob(g) if os.path.isfile(p)]
    return max(c, key=os.path.getmtime) if c else None


def tpt(s):
    """'10-29 08:11:30 PT' -> unix seconds."""
    try:
        m = re.match(r"(\d+)-(\d+) (\d+):(\d+):(\d+)", s or ""); now = datetime.datetime.now(PT)
        d = datetime.datetime(now.year, int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), tzinfo=PT)
        return (d.replace(year=now.year - 1) if d > now + datetime.timedelta(days=2) else d).timestamp()
    except Exception:
        return None


def ts(r):
    for k in ("_timestamp", "timestamp", "time", "t"):
        v = r.get(k)
        if isinstance(v, (int, float)) and v > 1e9: return float(v)
    return tpt(r.get("time_pt"))


def load(p):
    if p is None: return [], {}
    if p.endswith(".jsonl"):
        rows = []
        try:
            for line in open(p, errors="replace"):
                try: r = json.loads(line)
                except Exception: continue          # partial last line while the trainer writes
                if isinstance(r, dict): rows.append(r)
        except Exception: pass
        return rows, {}
    try: d = json.load(open(p))
    except Exception: return [], {}
    if isinstance(d, dict) and isinstance(d.get("history"), list): return [r for r in d["history"] if isinstance(r, dict)], d
    if isinstance(d, list): return [r for r in d if isinstance(r, dict)], {}
    return [], {}


def num(v): return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


def ver(r):
    for k in ("policy/version", "_step", "step", "update"):
        if num(r.get(k)): return int(r[k])
    return None


def timing_gate(tr):
    """TIMING GATE block for the page (s3 "timing"): static config = OPD_TIMING_CFG's "timing" key (charts, default thresholds, base export-253 and
    Param u40/u80/u110 values on the 96 paramsamples eval rollouts); live = the run's out/TIMING.jsonl (OPD_TIMING; per-update stats, breached names,
    streak, thresholds) and out/TIMING-ALERT.json (OPD_TIMING_ALERT; written once OPD_TIMING_GATE_UPDATES consecutive updates breach). The latest
    update's breach comes from TIMING.jsonl, else from the metrics rows (timing_rollout/*) against the thresholds."""
    try: cfg = dict(json.load(open(os.environ["OPD_TIMING_CFG"])).get("timing") or {})
    except Exception: cfg = {}
    thr = {k: list(v) for k, v in (cfg.get("thresholds") or {}).items()}
    tj = [p for g in os.environ.get("OPD_TIMING", "").split("|") if g for p in glob.glob(g) if os.path.isfile(p)]
    tj = max(tj, key=os.path.getmtime) if tj else None
    last = None
    if tj:
        for line in open(tj, errors="replace"):
            try: r = json.loads(line)
            except Exception: continue
            if isinstance(r, dict) and num(r.get("update")): last = r
    if last and isinstance(last.get("thresholds"), dict): thr = {k: list(v) for k, v in last["thresholds"].items()}
    mk = lambda n: "timing_rollout/" + (n[:-4] if n.endswith("_low") else n)      # latency_p50_s_low = latency_p50_s (lower bound)
    def br(stats, names=None):
        out = []
        for n, (op, t) in thr.items():
            v = stats.get(n, stats.get(n[:-4]) if n.endswith("_low") else None)
            hit = (n in names) if names is not None else (num(v) and ((op == ">" and v > t) or (op == "<" and v < t)))
            if hit: out.append([n, R5(v) if num(v) else None, op, t])
        return out
    cand = []
    if last: cand.append({"update": int(last["update"]), "breached": br(last.get("stats") or {}, list(last.get("breached") or [])),
                          "streak": last.get("streak"), "t": last.get("unix_seconds"), "src": "TIMING.jsonl"})
    mr = [(u, r) for u, r in tr if any(k.startswith("timing_rollout/") and num(r.get(k)) for k in r)]
    if mr:
        u, r = mr[-1]; st = {k[len("timing_rollout/"):]: v for k, v in r.items() if k.startswith("timing_rollout/") and num(v)}
        if not cand or u > cand[0]["update"]: cand = [{"update": u, "breached": br(st), "streak": None, "t": ts(r), "src": "metrics.jsonl"}]
    al = None; ap = os.environ.get("OPD_TIMING_ALERT", "")
    if ap and os.path.isfile(ap):
        try: a = json.load(open(ap))
        except Exception: a = {}
        al = {"update": a.get("update"), "breached": list(a.get("breached") or []), "streak": a.get("streak"), "t": os.path.getmtime(ap), "file": os.path.basename(ap)}
    tl = {}
    for n, (op, t) in thr.items(): tl.setdefault(mk(n), []).append([op, t, n])
    return {"charts": cfg.get("charts") or [], "thr": tl, "base": cfg.get("base") or {}, "base_label": cfg.get("base_label"), "param": cfg.get("param") or {},
            "param_label": cfg.get("param_label"), "latest": cand[0] if cand else None, "alert": al, "src": tj, "gate_updates": cfg.get("gate_updates", 2)}


src = pick_src(); raw, meta = load(src)
has_pv = any(num(r.get("policy/version")) for r in raw)
order = sorted(range(len(raw)), key=lambda i: (raw[i].get("_step") if num(raw[i].get("_step")) else i, i))
rows = []                                    # [(update, row)] in log order, restarts stitched
for n, i in enumerate(order):
    r = raw[i]; v = int(r["policy/version"]) if has_pv and num(r.get("policy/version")) else (None if has_pv else ver(r))
    if v is None:                            # unversioned row (eval between updates): the update before the next versioned row
        nx = next((int(raw[j]["policy/version"]) for j in order[n + 1:] if num(raw[j].get("policy/version"))), None)
        v = nx - 1 if nx is not None else (rows[-1][0] if rows else 0)
        rows.append((v, r)); continue
    if rows and v == rows[-1][0]:            # another row for the same update (e.g. its eval / checkpoint logged separately): merged, newer keys win
        rows[-1] = (v, {**rows[-1][1], **r}); continue
    rows = [(u, q) for u, q in rows if u < v]   # a restart (the version goes back): this update and every later one are redone (the newer rows win)
    rows.append((v, r))
R5 = lambda v: None if not num(v) else (round(v, 9) if abs(v) < 1e-3 else round(v, 6))
tr = [(u, r) for u, r in rows if any(num(r.get(k)) for k in TRAIN)]
ev = list({u: (u, r) for u, r in rows if any(num(r.get(k)) for k in EVAL)}.values())   # one eval per update: the newest (a re-run eval after a restart wins)
more = sorted({k for _, r in tr for k in r if k.startswith(MORE) and num(r.get(k))})
tk = [k for k in TRAIN if any(num(r.get(k)) for _, r in tr)] + more
ek = [k for k in EVAL if any(num(r.get(k)) for _, r in ev)]
jid = os.environ.get("OPD_JOB") or ""
if not jid and os.environ.get("OPD_JOB_FILE"):
    try: jid = (re.search(r"\d+", open(os.environ["OPD_JOB_FILE"]).read()) or [""])[0]
    except Exception: jid = ""
if not jid:
    for _, r in reversed(rows):
        for k in ("slurm/job_id", "slurm_job_id", "job_id", "job"):
            if r.get(k) not in (None, ""): jid = str(r[k]); break
        if jid: break
JOB = jid or os.environ.get("OPD_JOB_NAME") or ""      # slurm job id (OPD_JOB / OPD_JOB_FILE / the rows), else the W&B run id (references); none = no jobs yet
d = {"run": RUN, "jobs": [JOB] if JOB else [], "job_runs": {JOB: RUN} if JOB else {}, "started": bool(tr), "src": src,
     "train_cols": ["step", "rows"] + [ALIAS.get(k, k) for k in tk], "train": [[u, u * RPS] + [R5(r.get(k)) for k in tk] for u, r in tr],
     "cols": ["step", "rows"] + ek, "pts": [[u, u * RPS] + [R5(r.get(k)) for k in ek] for u, r in ev],
     "lr": {"cols": ["step", "lr"], "rows": [[u, R5(r.get("train/learning_rate"))] for u, r in tr if num(r.get("train/learning_rate"))]} if any(num(r.get("train/learning_rate")) for _, r in tr) else None,
     "rows_per_step": RPS, "epoch_rows": EPR, "epoch_steps": EPR / RPS, "ppm": {"speech_rows_per_step": RPS, "speech_epoch_rows": EPR},
     "label": os.environ.get("OPD_LABEL"), "short": os.environ.get("OPD_SHORT"), "max_steps": MAXS, "export_steps": [MAXS],
     "updated": datetime.datetime.now(PT).strftime("%H:%M PT")}
if os.environ.get("OPD_DIFF"):      # Plot difference metrics for the ★ doc (ppm.tpl.html PA + S().diff, page-local replace in s3branches_opd.json)
    d["diff"] = [[ALIAS.get(k, k) + ":t", k] for k in TRAIN] + [[k + ":e", k + " (at saves)"] for k in EVAL]   # static list (the page reads it once at load, also before the ★ logs)
st = None
if jid.isdigit():
    try:
        st = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%T"], capture_output=True, text=True, timeout=15).stdout.strip() or None
        if not st:
            o = subprocess.run(["sacct", "-X", "-n", "-j", jid, "-o", "State%20"], capture_output=True, text=True, timeout=15).stdout.split()
            st = o[0] if o else None
    except Exception: st = None
d["state"] = st
if tr:
    u1, r1 = tr[0]; uL, rL = tr[-1]
    rs = [r.get("timing/round_seconds") for _, r in tr if num(r.get("timing/round_seconds"))]
    tl = [ts(r) for _, r in tr]
    if not rs:                               # no round timer logged: wall-clock gaps between updates
        rs = [b - a for a, b in zip(tl, tl[1:]) if a and b and b > a]
    d["step"] = uL; d["t_last"] = tl[-1]; d["t_step1"] = tl[0]
    if d["t_last"] is None and src and src.endswith(".jsonl"): d["t_last"] = os.path.getmtime(src)
    _rw = [r.get("timing/round_seconds") for u, r in tr if u > 2 and num(r.get("timing/round_seconds"))]      # 10-29 opdpage-fix (main): updates 1-2 = compile warm-up
    if _rw: rs = _rw      # s/update = median of the last <= 10 non-warm-up updates (all updates until update 3 exists); same number as opd_prog.py's run ETA
    _m = sorted(rs[-10:]); _md = ((_m[len(_m) // 2] + _m[(len(_m) - 1) // 2]) / 2) if _m else None
    d["mean50_s"] = round(_md, 2) if _md else None; d["mean_s"] = d["mean50_s"]
    d["done"] = uL >= MAXS or st == "COMPLETED"
    if not d["done"] and st and st not in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING"): d["stopped"] = uL
    t0 = (tl[0] - r1["timing/round_seconds"]) if tl[0] and num(r1.get("timing/round_seconds")) else tl[0]   # the update loop's start
    sv = [(u, r) for u, r in rows if num(r.get("checkpoint/complete")) and r["checkpoint/complete"] >= 1]
    if sv:
        u, r = sv[-1]; t = ts(r)
        if t and num(r.get("genghis/phase_wall_seconds")): t -= r["genghis/phase_wall_seconds"]   # the row is logged after the post-save GenghisBench
        d["last_save"] = [t, u]
    else:
        d["last_save"] = [t0, None]
    d["save_every_min"] = SAVE_MIN; d["save_wall"] = True
    if MAXS > uL: d["next_export_step"] = MAXS
    d["last_eval"] = list(d["last_save"]); d["eval_every_min"] = SAVE_MIN       # GenghisBench runs right after each save
    if ev: d["last_genghis"] = [ts(ev[-1][1]), ev[-1][0]]
else:
    d["step"] = None
if os.environ.get("OPD_TIMING_CFG"):      # 10-29 nextrun (main -07: gate the next OPD run on rollout timing, not lexical KL): the ★ doc only (main_env)
    d["timing"] = timing_gate(tr)
json.dump(d, open(OUT, "w"), separators=(",", ":"))
