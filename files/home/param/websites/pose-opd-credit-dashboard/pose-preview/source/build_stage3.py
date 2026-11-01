#!/usr/bin/env python3
"""Live training-curves page (page-templates/curves; README.md). Originally the Qwen stage-3 live page. Standalone:  python3 build_stage3.py [klweb_dir] [--no-doc]
  1. runs ../s3_doc.py -> <W>/s3live.json (unless --no-doc; S3_JOBS=<jobs|chain:job> pins the jobs; S3_RUN defaults to s3branches.json main_run)
     plus one s3_doc.py run per branch in s3branches.json -> s3live.json "branches" (polled by the tunnel page with the main data)
  2. reads <W>/s3ref_hestia.json (static Hestia reference, from s3tools/hestia_fetch.py) and the stage-2 reference from <W>/s2live.json
  3. writes <W>/stage3.html (tunnel page; polls s3live.json every 60 s) and <W>/stage3_artifact.html (snapshot body for a claude.ai artifact)
ver_s3.txt = hash of template + static data, so open tunnel pages reload only when the page itself changes, not on every data refresh."""
import json, sys, os, subprocess, hashlib, datetime, zoneinfo
CANOPY_ROOT = os.environ.get("CANOPY_ROOT") or os.path.expanduser("~")   # cluster home holding ppm/, venvs/, models/ (see README: paths)
HOME_DIR = os.path.expanduser("~")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
W = args[0] if args else os.path.dirname(os.path.abspath(__file__))
DOC = os.environ.get("CURVES_DOC") or os.path.join(W, "opd_doc.py")      # default doc script when s3branches has no "doc"
# S3_PAGE=<name> builds a second instance of this same page from s3branches_<name>.json -> stage3_<name>.html / s3live_<name>.json / ver_s3_<name>.txt
# (e.g. S3_PAGE=pose: the pose runs, same template; their pose rows + codependence series show in the pose sections, hidden when a run has none)
PG = os.environ.get("S3_PAGE", ""); SFX = f"_{PG}" if PG else ""
# s3branches.json: {"main_run": <run name of the ★ chain>, "branches": [{run, label, from_step, color, color_dark[, job]}]}. The main chain follows main_run
# (s3_doc.py S3_RUN), so a branch resuming from the same checkpoint is never chained into it; each branch is its own series (steps >= from_step).
try: BRC = json.load(open(f"{W}/s3branches{SFX}.json"))
except Exception: BRC = {}
XROOT = CANOPY_ROOT + "/qwen-1ch/xbench/results"
def xdep_jobs(jobs, rps):
    """cross-channel dependence per export of the given jobs: xbench/results/s3probe-j<job>-w<N>/summary.json (pooled normalised word CE).
    pts [step, rows, gap_oth-silent, gap_oth-absent, real, gap_oth-wrong]: SILENT = the OTHER channel's text all PAD + its audio = the codec's
    silence codes (s3probe sK); ABSENT = the other channel contributes nothing to the trunk input (s3probe xK); WRONG = the other channel's
    whole track (text lane + audio) taken from a different held-out conversation (s3probe wK); each minus real."""
    import glob, re
    out = []
    for j in jobs:
        for f in glob.glob(f"{XROOT}/s3probe-j{j}-w*/summary.json"):
            try:
                w = json.load(open(f))["pooled"]["word_norm"]; n = int(re.search(r"-w(\d+)/", f)[1])
                g = lambda k: round(w[k], 5) if w.get(k) is not None else None
                if g("gap_oth-silent") is None and g("gap_oth-absent") is None and g("gap_oth-wrong") is None: continue
                out.append([n, n * rps, g("gap_oth-silent"), g("gap_oth-absent"), round(w["real"], 5), g("gap_oth-wrong")])
            except Exception: pass
    return sorted({r[0]: r for r in out}.values())
def ckpt_points(c):
    """10-24 aloe32 page (main: "add PPMVOX checkpoint evals as POINTS"): s3branches "ckpt_points" {results: RESULTS.jsonl (one line per checkpoint
    {"tag": "w-N", "eval_job", "word_ce_norm", "run" (absent = ppmvox-v1)}), runs: metrics.jsonl glob with {tag} / {rtag} (= tag for v1, else
    "<run>-<tag>") / {job}, from_step, label, note, controls: CONTROLS.jsonl} -> {cols: [vox_step, step, eval keys...], pts, tags, runs, controls}: per
    (run, checkpoint) (newest line wins) the step-0 eval of its eval-only run, else RESULTS' word CE as eval_speech/word_ce_norm; controls = the last
    line per control that has a word CE (flat references)."""
    import glob, re
    last = {}
    for line in open(c["results"]) if os.path.exists(c["results"]) else []:
        try: r = json.loads(line)
        except Exception: continue
        if re.match(r"w-\d+$", str(r.get("tag", ""))): last[(r.get("run") or "ppmvox-v1", r["tag"])] = r
    rows = []
    for (run, t), r in last.items():
        ev = {}; rt = t if run == "ppmvox-v1" else f"{run}-{t}"
        for p in sorted(glob.glob(c["runs"].format(tag=t, rtag=rt, job=r.get("eval_job") or "*"))):
            for l in open(p):
                try: d = json.loads(l)
                except Exception: continue
                if d.get("step") == 0 and any(k.startswith("eval_") for k in d): ev = {k: v for k, v in d.items() if k.startswith("eval_") and isinstance(v, (int, float))}
        if not ev:
            try: ev = {"eval_speech/word_ce_norm": float(r["word_ce_norm"])}
            except Exception: continue
        rows.append(((c.get("merge") or {}).get(run, run), int(t[2:]), t, ev))      # 10-24 "merge": {run: series} (ppmvox-v3x continues the ppmvox-v3 series)
    rows.sort(); keys = sorted({k for _, _, _, e in rows for k in e}); s0 = int(c.get("from_step", 0)); ctl = {}
    for line in open(c["controls"]) if c.get("controls") and os.path.exists(c["controls"]) else []:
        try: r = json.loads(line); ctl[r["control"]] = dict(r, word_ce_norm=float(r["word_ce_norm"]))
        except Exception: continue
    return {"label": c.get("label"), "note": c.get("note"), "from_step": s0, "tags": [t for _, _, t, _ in rows], "runs": [u for u, _, _, _ in rows], "cols": ["vox_step", "step"] + keys,
            "pts": [[st - s0, st] + [round(e[k], 5) if k in e else None for k in keys] for _, st, _, e in rows],
            "controls": [{"control": k, "word_ce_norm": v["word_ce_norm"], "note": v.get("note")} for k, v in sorted(ctl.items())]}
def branch_doc(b):
    """run s3_doc.py for one branch (its own chain, which includes the shared history), keep steps >= from_step; dependence probes of the branch's own jobs."""
    tmp = f"{W}/s3branch{SFX}.json.tmp"; env = dict(os.environ, S3_RUN=b["run"], **{k: os.path.expandvars(str(v)) for k, v in (b.get("env") or {}).items()})   # 10-17 branch "env": extra env for its doc script (ppm page: the interleaved run's log root / prefix)
    cmd = [sys.executable, os.path.expandvars(b.get("doc") or BRC.get("doc") or DOC), tmp] + ([f"chain:{b['job']}"] if b.get("job") else [])   # "doc": another doc script (S3_PAGE=ppm: ppm_doc.py)
    if subprocess.run(cmd, stdout=subprocess.DEVNULL, env=env).returncode != 0: return None
    d = json.load(open(tmp)); os.remove(tmp)
    own = [j for j in d.get("jobs", []) if d.get("job_runs", {}).get(j) == b["run"]]
    if not own: return None
    s0 = b.get("from_step", 0); rps = d.get("rows_per_step") or 304
    return {k: b.get(k) for k in ("run", "label", "color", "color_dark", "from_step", "archived", "note", "fit", "rel")} | {
        "jobs": own, "step": d.get("step"), "stopped": d.get("stopped"), "done": d.get("done"), "max_steps": d.get("max_steps"), "updated": d.get("updated"),
        "cols": d["cols"], "pts": [r for r in d["pts"] if r[0] >= s0], "train_cols": d["train_cols"], "train": [r for r in d["train"] if r[0] >= s0],
        "xdep": xdep_jobs(own, rps), "mean50_s": d.get("mean50_s"), "t_last": d.get("t_last"), "rows_per_step": d.get("rows_per_step"), "epoch_rows": d.get("epoch_rows")} | \
        ({"segments": b["segments"]} if b.get("segments") else {}) | \
        {k: b[k] for k in ("base_label", "desc", "tag") if b.get(k)} | \
        {k: d.get(k) for k in (b.get("pass") or [])}   # 10-17: header step time / ETA for a branch; 10-23 branch "pass": extra doc keys passed through (ppm811 LR-decay branch: lr, max_steps_set); no "pass" = as before
# ---- GenghisBench section (s3live.json "gb", so open tunnel pages pick up new results on their 60 s poll) ----
GBDIR = CANOPY_ROOT + "/qwen-1ch/infer/gb_s3"
GB_BASE = {"n60_file": CANOPY_ROOT + "/qwen-1ch/infer/gb_s2final/gb_v6fix_n60_j7862_w4289.json", "kl": 0.0965,
           "label": "stage-3 init = stage-2 final w-4289 (job 7862)"}
def gb_doc(jobs):
    """GenghisBench + game KL per export of the ★ chain's jobs. Each job's exports count only up to the step the next job resumed from
    (same cut as s3_doc.py), so a re-exported step is taken from the job that is actually in the chain.
    pts rows = GB_COLS: step, job, EOT wins / games (end of turn = the model's own EOT; gb_j<job>-w<N>.json), qmark wins / games (turn forced
    to end at a '?' piece; ...-qmark.json), 60-game wins / games (EOT; ...-n60.json), game KL (xbench/results/s3gbkl-j<job>-w<N>/summary.json ALL;
    teacher agreement on 40 real GenghisBench dialogs, lower = better). None where that result has not landed yet."""
    import glob, re
    LD = CANOPY_ROOT + "/qwen-1ch/stage3/logs"
    cut = {}
    for a, b in zip(jobs, jobs[1:]):
        try:
            for line in open(f"{LD}/q1ch-s3-{b}.log", errors="replace"):
                m = re.search(r"full resume from \S*?dcp-(\d+)", line)
                if m: cut[a] = int(m[1]); break
        except Exception: pass
    CACHE = {}
    def ld(p):
        if p not in CACHE:
            try: CACHE[p] = json.load(open(p))
            except Exception: CACHE[p] = None
        return CACHE[p]
    def sn(p):
        d = ld(p)
        try: return (int(d["successes"]), int(d["n"])) if d and d.get("n") else (None, None)
        except Exception: return (None, None)
    def games(p, seeds):
        """per game [won 0/1, questions used (-1 = aborted)] in seed order, or None unless the file holds exactly these seeds.
        Also exports the transcripts to <W>/gbtx/<file name> ({seed: conv}) for the page to fetch on click (rewritten only when the source is newer)."""
        d = ld(p)
        try: m = {int(a["seed"]): a for a in d["attempts"]}
        except Exception: return None
        if sorted(m) != seeds: return None
        try:
            o = f"{W}/gbtx/{os.path.basename(p)}"
            if not os.path.exists(o) or os.path.getmtime(o) < os.path.getmtime(p):
                os.makedirs(f"{W}/gbtx", exist_ok=True)
                json.dump({str(k): a.get("conv") or [] for k, a in m.items()}, open(o + ".tmp", "w"), separators=(",", ":")); os.replace(o + ".tmp", o)
        except Exception: pass
        return [[int(bool(m[k].get("success"))), int(m[k].get("questions", -1))] for k in seeds]
    S20, S60 = list(range(100, 120)), list(range(200, 260))
    rows, per = {}, {}
    for j in jobs:
        steps = set()
        for p in glob.glob(f"{GBDIR}/gb_j{j}-w*.json") + glob.glob(f"{XROOT}/s3gbkl-j{j}-w*/summary.json"):
            m = re.search(rf"j{j}-w(\d+)(?:[-./]|$)", p)
            if m: steps.add(int(m[1]))
        for n in sorted(steps):
            if n > cut.get(j, 10 ** 9): continue
            t = f"j{j}-w{n}"; e, q, s = sn(f"{GBDIR}/gb_{t}.json"), sn(f"{GBDIR}/gb_{t}-qmark.json"), sn(f"{GBDIR}/gb_{t}-n60.json")
            try: kl = round(float(json.load(open(f"{XROOT}/s3gbkl-{t}/summary.json"))["ALL"]), 5)
            except Exception: kl = None
            if e[0] is None and q[0] is None and s[0] is None and kl is None: continue
            rows[n] = [n, j, *e, *q, *s, kl]
            g = {k: games(f"{GBDIR}/gb_{t}{suf}.json", sd) for k, suf, sd in (("eot", "", S20), ("qm", "-qmark", S20), ("n60", "-n60", S60))}
            per[n] = {"tag": t} | {k: v for k, v in g.items() if v}
    b60 = sn(GB_BASE["n60_file"])
    # per-game: games[step] = {tag, eot, qm, n60}; each a list aligned to seeds20 / seeds60 of [won, questions]; transcripts at gbtx/gb_<tag><suffix>.json
    return {"games": {str(k): per[k] for k in sorted(per) if len(per[k]) > 1}, "seeds20": S20, "seeds60": S60,
            "base_games": {"n60": games(GB_BASE["n60_file"], S60), "tx": "gbtx/" + os.path.basename(GB_BASE["n60_file"])},
            "cols": ["step", "job", "eot", "eot_n", "qm", "qm_n", "n60", "n60_n", "kl"], "pts": [rows[k] for k in sorted(rows)],
            "base": {"label": GB_BASE["label"], "n60": b60[0], "n60_n": b60[1], "kl": GB_BASE["kl"]}}
# ---- InteractionBench section (s3live.json "ib"; results infer/ibench/runs/<TAG>/ibench.json from infer/ibench/ibench.py) ----
IBDIR = CANOPY_ROOT + "/qwen-1ch/infer/ibench/runs"
def ib_doc(jobs):
    """InteractionBench per export of the ★ chain's jobs (runs/j<job>-w<N>/ibench.json; the 24-row continue set). pts rows = cols:
    step, job, total (0-100), 90% CI lo / hi, part scores td / rs / dg (0-1), median switches / min, median response latency (s), judged
    'responds' rate, dead-air events / min (mean), faff score (0-100, closeness to real), mean judged waffle 0-3, new content
    words / min, fillers / min, clips-only prompt (export trained before persona text), cross-channel gap_oth-wrong.
    ref = the calibration conditions (real continuations = ceiling; same-episode time-shifted and other-episode mismatched pairs)."""
    import glob, re
    def one(d):
        g = d["groups"]["continue"]; m = g.get("metrics", {}); r = g.get("responsiveness") or {}; ci = g.get("total_ci90") or [None, None]
        f = lambda k, s="median": (m.get(k) or {}).get(s)
        return [g.get("total"), ci[0], ci[1], g["parts"].get("td"), g["parts"].get("rs"), g["parts"].get("dg"), f("switch_per_min"),
                f("latency_p50"), r.get("resp"), f("dead_air_per_min", "mean"), (g.get("faff") or {}).get("score"), f("waffle_mean", "mean"),
                f("info_per_min"), f("filler_per_min")]
    rows = {}
    for j in jobs:
        for p in glob.glob(f"{IBDIR}/j{j}-w*/ibench.json"):
            mm = re.search(rf"/j{j}-w(\d+)/ibench\.json$", p)
            if not mm: continue
            try:
                d = json.load(open(p)); x = d.get("xdep") or {}
                rows[int(mm[1])] = [int(mm[1]), j, *one(d), int(bool(d.get("clips_only"))), x.get("gap_oth-wrong")]
            except Exception: pass
    ref = {}
    for c in ("real", "shift", "mismatch"):
        try:
            d = json.load(open(f"{IBDIR}/calib-{c}/ibench.json")); v = one(d)
            ref[c] = dict(zip(["total", "lo", "hi", "td", "rs", "dg", "sw", "lat", "resp", "dead", "faff", "waf", "info", "fill"], v))
        except Exception: pass
    var = []
    for p in sorted(glob.glob(f"{IBDIR}/j*-w*-*/ibench.json")):
        mm = re.search(r"/(j\d+)-w(\d+)-([a-z]+)/ibench\.json$", p)
        try:
            if mm: var.append({"step": int(mm[2]), "job": mm[1], "variant": mm[3], "total": json.load(open(p))["groups"]["continue"]["total"]})
        except Exception: pass
    return {"cols": ["step", "job", "total", "lo", "hi", "td", "rs", "dg", "sw", "lat", "resp", "dead", "faff", "waf", "info", "fill", "clips", "xw"],
            "pts": [rows[k] for k in sorted(rows)], "ref": ref, "variants": var}
# ---- MathBench (text) section (s3live.json "mb"; results infer/mathbench/out/<TAG>/summary.json from infer/mathbench/mathbench.py) ----
MBDIR = CANOPY_ROOT + "/qwen-1ch/infer/mathbench/out"
MB_COLS = ["step", "job"] + [f"{d}_{t}{s}" for d in ("g", "s") for t in ("gsm", "math") for s in ("", "_lo", "_hi")]
def mb_doc(jobs):
    """MathBench (text) accuracy % per export of the ★ chain's jobs (out/j<job>-w<N>/summary.json): rows = MB_COLS (g = greedy, s = T1.0 /
    top-k 60; gsm = GSM8K 200, math = MATH-500 100; _lo / _hi = Wilson 95% CI); ref = base Qwen3-32B (out/base*) and the stage-3 init (out/init-*)."""
    import glob, re
    def one(p):
        R = json.load(open(p))["results"]; o = []
        for dec in ("greedy", "sample"):
            for t in ("gsm8k", "math500"):
                r = R.get(dec) or {}; x = r.get(t) or {}
                o += [round(100 * x["acc"], 1), round(100 * x["ci95"][0], 1), round(100 * x["ci95"][1], 1)] if x.get("n") and r.get("complete") else [None] * 3
        return o
    rows = {}
    for j in jobs:
        for p in glob.glob(f"{MBDIR}/j{j}-w*/summary.json"):
            m = re.search(rf"/j{j}-w(\d+)/summary\.json$", p)
            try:
                if m: rows[int(m[1])] = [int(m[1]), j, *one(p)]
            except Exception: pass
    ref = {}
    for k, pat in (("base", "base*"), ("init", "init-*")):
        for p in sorted(glob.glob(f"{MBDIR}/{pat}/summary.json"))[:1]:
            try: ref[k] = {"tag": p.split("/")[-2], "v": one(p)}
            except Exception: pass
    return {"cols": MB_COLS, "pts": [rows[k] for k in sorted(rows)], "ref": ref}
def ptabs_doc(dd):
    """pose-page tabs data (pose_tabs.py): the ★ chain + every non-archived branch of s3branches_<page>.json; only when it sets "posetabs"."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import pose_tabs
    # main = s3branches_<page>.json main_run (the ★ chain s3_doc.py followed); branches = every non-archived branch, with optional
    # posetabs_label / posetabs_tabs (which tabs it appears on; default all) / posetabs_counts (its pose store's counts.json)
    cfg = {b.get("run"): b for b in BRC.get("branches", [])}
    ch = [("main", "★ " + BRC.get("main_run", ""), BRC.get("main_run", ""), dd.get("jobs", []), None, BRC.get("posetabs_counts"))]
    for i, b in enumerate(dd.get("branches", [])):
        if b.get("archived"): continue
        c = cfg.get(b.get("run"), {})
        if c.get("posetabs") is False: continue      # 10-17: a branch without pose (ppm page: the interleaved run) stays off the pose tabs
        ch.append(("b%d" % i, c.get("posetabs_label") or b.get("label"), b.get("run"), b.get("jobs", []), c.get("posetabs_tabs"), c.get("posetabs_counts")))
    if BRC.get("posetabs_refs"):      # 10-12 ppm page: reference runs from the doc's "refs" (kind "pose") also go on the tabs
        for r in dd.get("refs", []):
            if r.get("kind") == "pose": ch.append(("ref_" + str(r.get("id")), r.get("label"), r.get("run"), r.get("jobs", []), None, None))
    return pose_tabs.doc(ch)
if "--no-doc" not in sys.argv:
    env = dict(os.environ); env.setdefault("S3_RUN", BRC.get("main_run", ""))
    env.update({k: os.path.expandvars(str(v)) for k, v in (BRC.get("main_env") or {}).items()})      # 10-21 s3branches_<page>.json "main_env": extra env for the ★ doc script only (ppm14b page: its own PPM_ROOT / PPM_LOG_PREFIX / PPM_HEAD); pages without it unchanged
    cmd = [sys.executable, os.path.expandvars(BRC.get("doc") or DOC), f"{W}/s3live{SFX}.json.tmp"] + ([os.environ["S3_JOBS"]] if os.environ.get("S3_JOBS") else [])   # s3branches_<page>.json "doc" (10-12, ppm page: ppm_doc.py); default s3_doc.py
    if subprocess.run(cmd, stdout=subprocess.DEVNULL, env=env).returncode == 0:
        try:
            dd = json.load(open(f"{W}/s3live{SFX}.json.tmp")); brs = []
            for b in BRC.get("branches", []):
                try:
                    x = branch_doc(b)
                    if x: brs.append(x)
                except Exception as e: print("branch", b.get("run"), "failed:", e)
            dd["branches"] = brs
            if BRC.get("base_line"):      # 10-23 aloe32 page: the ★ FT's base checkpoint (step-0 eval values), polled with the rest
                bl = dict(BRC["base_line"]); r0 = next((r for r in dd.get("pts") or [] if r[0] == 0), None)
                if bl.get("from_main_step0") and r0:      # 10-24: follow the ★ run's own step-0 eval once it has one (else the config's values)
                    bl["vals"] = {c: v for c, v in zip(dd.get("cols") or [], r0) if c.startswith("eval_") and v is not None}; bl["src"] = f"the ★ run {dd.get('run')} step-0 eval (metrics.jsonl)"
                dd["base_line"] = bl
            if BRC.get("xwin"): dd["xwin"] = BRC["xwin"]
            if BRC.get("lead"): dd["lead"] = BRC["lead"]      # 10-25 aloe32 page: the ★ FT's lead charts first on CRITICAL [[key, eval?, title], ...]      # 10-25 aloe32 page: default x window [from, to] (steps) unless the header "Full history"
            if BRC.get("ckpt_points"):      # 10-24 aloe32 page: PPMVOX checkpoints on the Aloe eval (points), polled with the rest
                try: dd["ckpt_pts"] = ckpt_points(BRC["ckpt_points"])
                except Exception as e: print("ckpt_points failed:", e)
            if BRC.get("posetabs"):
                try: dd["ptabs"] = ptabs_doc(dd)      # pose page POSE | SPEECH | KL tabs (polled with the rest of s3live_<page>.json)
                except Exception as e: print("ptabs failed:", e)
            try: dd["gb"] = gb_doc(dd.get("jobs", []))
            except Exception as e: print("gb failed:", e)
            try: dd["ib"] = ib_doc(dd.get("jobs", []))      # InteractionBench section
            except Exception as e: print("ib failed:", e)
            try: dd["mb"] = mb_doc(dd.get("jobs", []))      # MathBench (text) section
            except Exception as e: print("mb failed:", e)
            for _k in BRC.get("strip", []): dd.pop(_k, None)      # 10-29 s3branches_<page>.json "strip": s3live keys this page drops (OPD page: stage-3 gb / ib / mb); no key = unchanged
            json.dump(dd, open(f"{W}/s3live{SFX}.json.tmp", "w"), separators=(",", ":"))
        except Exception as e: print("branches failed:", e)
        os.replace(f"{W}/s3live{SFX}.json.tmp", f"{W}/s3live{SFX}.json")
TUN = open(f"{W}/TUNNEL_URL").read().strip() if os.path.exists(f"{W}/TUNNEL_URL") else ""
try: s3 = json.load(open(f"{W}/s3live{SFX}.json"))
except Exception: s3 = {"started": False}
if "gb" not in s3 and s3.get("jobs"):
    try: s3["gb"] = gb_doc(s3["jobs"])       # --no-doc snapshot builds still carry the GenghisBench section
    except Exception: pass
if "ib" not in s3 and s3.get("jobs"):
    try: s3["ib"] = ib_doc(s3["jobs"])       # InteractionBench section (--no-doc snapshot builds)
    except Exception: pass
if "mb" not in s3 and s3.get("jobs"):
    try: s3["mb"] = mb_doc(s3["jobs"])       # MathBench (text) section (--no-doc snapshot builds)
    except Exception: pass
if BRC.get("posetabs") and "ptabs" not in s3 and s3.get("jobs"):
    try: s3["ptabs"] = ptabs_doc(s3)          # --no-doc snapshot builds
    except Exception as e: print("ptabs failed:", e)
for _k in BRC.get("strip", []): s3.pop(_k, None)      # 10-29 "strip" (see above): also out of the --no-doc snapshot fill-ins
try: hestia = json.load(open(f"{W}/s3ref_hestia.json"))      # optional static reference runs
except Exception: hestia = {"runs": []}
try: dmour = json.load(open(f"{W}/s3ref_demeter_ourrows.json"))      # Demeter checkpoints on OUR eval rows (s3hestia/ourrows/make_ref.py)
except Exception: dmour = None
try: dmtxd = json.load(open(f"{W}/s3ref_demeter_txd_ourrows.json"))   # Demeter checkpoints on OUR held-out text-duplex packs (s3hestia/dmrepro/make_txd_ref.py)
except Exception: dmtxd = None

def s2ref():
    """stage-2 eval at the step stage 3 was initialised from (its init_step), else stage 2's latest eval. Assistant / single channel."""
    if os.environ.get("S2_FINAL_KL"): return {"text_kl": float(os.environ["S2_FINAL_KL"]), "src": "S2_FINAL_KL override"}
    try: d = json.load(open(f"{W}/s2live.json"))
    except Exception: return {}
    c, pts = d.get("cols", []), d.get("pts", [])
    if not pts: return {}
    want = s3.get("init_step")
    row = min(pts, key=lambda r: abs(r[0] - want)) if want else pts[-1]
    o = {k: row[c.index(k)] for k in ("text_kl", "nword", "word_ce", "cb0", "light", "nw_ce", "pad_ce") if k in c and c.index(k) < len(row)}
    o["src"] = f"stage-2 job {','.join(d.get('jobs', [])[-1:])} eval at step {row[0]}" + (" = stage-3 init" if want else " (latest stage-2 eval)")
    o["step"] = row[0]
    return o

def xdep():
    """dependence probes of the ★ chain's jobs (see xdep_jobs)."""
    return xdep_jobs(s3.get("jobs", []), s3.get("rows_per_step") or 304)
XDEP = xdep()
try: DMX = json.load(open(f"{W}/s3ref_demeter_xdep.json"))         # Demeter checkpoints, same probe (s3hestia/ourrows/dm_xdep.py + make_xdep.py)
except Exception: DMX = None
DMXB = None      # 10-12 s3branches_<page>.json "dmxdep_b": Demeter's CHANNEL-B dependence on A (s3hestia/ourrows/make_xdep_B.py), PPM page only
if BRC.get("dmxdep_b"):
    try: DMXB = json.load(open(f"{W}/s3ref_demeter_xdep_B.json"))
    except Exception: DMXB = None

tpl = open(f"{W}/{BRC.get('page_tpl') or 'ppm.tpl.html'}").read()   # 10-12 "page_tpl": a page's own template (ppm: ppm.tpl.html; no injections)
tpl = tpl.replace("'s3live.json?t='", f"'s3live{SFX}.json?t='").replace("'ver_s3.txt?t='", f"'ver_s3{SFX}.txt?t='")
if PG:     # second instance: its own browser-saved view state, and its title
    import re as _re
    tpl = _re.sub(r"'s3(arch1|gbsel1|gshift1|hidden1|modes1|ranges1|shift1)'", rf"'s3{PG}_\1'", tpl)
    if BRC.get("title"): tpl = tpl.replace("<title>Qwen Stage 3 Curves</title>", f"<title>{BRC['title']}</title>").replace("<h1>Qwen stage 3 curves", f"<h1>{BRC['title']}")
# SAM 8B speech+pose reference (10-10): s3branches_<page>.json "samref" = path to a static W&B snapshot (qwen-1ch/infer/sam_pose/wandb_history.json,
# from qwen-1ch/infer/sam_pose/wbpull/snap.py). Only such pages get it: stage3_samref.inc.js is injected into THIS page's template copy and the
# data goes to D.samref (static, never polled), so stage3.html (no "samref" in s3branches.json) is byte-identical to before.
SAMREF = None; INJECT_ERR = None
if BRC.get("samref") or BRC.get("posetabs"):
    try:
        _h = json.load(open(BRC["samref"])); _p = _h["page"]; _r = lambda rows: [[x[0]] + [None if v is None else round(v, 5) for v in x[1:]] for x in rows]
        SAMREF = {"source": _h.get("source"), "fetched": _h.get("fetched_utc"), "planned": _h.get("planned_steps"), "last_step": _h.get("last_step"),
                  "export_step": _h.get("export_step"), "state": (_h.get("run") or {}).get("state"),
                  "eval_cols": _p["eval_cols"], "eval": _r(_p["eval"]), "train_cols": _p["train_cols"], "train": _r(_p["train_mean50"])}
        if _h.get("page_cb"): SAMREF.update({"rows_per_step": _h.get("rows_per_step", 40), "cb_eval_cols": _h["page_cb"]["eval_cols"], "cb_eval": _r(_h["page_cb"]["eval"]),
                                             "cb_train_cols": _h["page_cb"]["train_cols"], "cb_train": _r(_h["page_cb"]["train_mean50"])})
        if BRC.get("page_tpl"): pass      # own template: data only (D.samref / s3live ptabs), no fragment injection
        elif BRC.get("posetabs"):
            # POSE | SPEECH | KL tabs (hardened 10-10): ONE injection point, just before the template's LAST </script>, so edits inside the
            # template's functions can never break it; the fragment hooks build / fit / fitY / slopeTxt at runtime and shows a red banner itself
            # if a template name it needs is gone. Only a missing </script> (or a missing fragment file) can fail here -> banner + loud log below.
            _frag = open(f"{W}/{BRC.get('posetabs_js') or 'stage3_posetabs.inc.js'}").read(); _k = tpl.rfind("</script>")   # 10-12 "posetabs_js": a page's own copy (ppm)
            if "</script" in _frag.lower(): raise ValueError("stage3_posetabs.inc.js contains a literal closing script tag (it would end the script early)")
            if _k < 0: raise ValueError("template anchor missing: the final </script> of stage3.tpl.html")
            # its OWN <script> after the template's: top-level const/let/functions of a classic script are shared, and a syntax error in the
            # fragment can then only kill the tabs, never the classic page; the guard script turns "fragment never ran" into the red banner too
            _guard = ("<script>setTimeout(()=>{if(!window.PT_OK&&!document.getElementById('pt_fail')){const b=document.createElement('div');b.id='pt_fail';"
                      "b.style.cssText='background:#c0392b;color:#fff;font:600 14px/1.4 system-ui,sans-serif;padding:10px 14px;border-radius:8px;margin:8px 0';"
                      "b.textContent='pose tabs failed to load: the tabs script did not run (syntax error? see the browser console)';"
                      "(document.querySelector('.wrap')||document.body).prepend(b);console.error(b.textContent)}},1500)</script>")
            _e = _k + len("</script>")
            tpl = tpl[:_e] + "\n<script>\n" + _frag + "\n</script>" + _guard + tpl[_e:]
        else:
            # older faded-grey SAM lines on the classic charts (string anchors; any miss -> banner + loud log below)
            _frag = open(f"{W}/stage3_samref.inc.js").read()
            _R = [("for(const k of Object.keys(sets))rngUI(k);", _frag + "\nfor(const k of Object.keys(sets))rngUI(k);"), ("const defHidden=l=>{", "const defHidden=l=>{if(l.startsWith('SAM 8B'))return false;")]
            for _a, _b in _R:
                if tpl.count(_a) != 1: raise ValueError("template anchor missing: " + _a[:60])
            for _a, _b in _R: tpl = tpl.replace(_a, _b, 1)
    except Exception as e:
        SAMREF = None; INJECT_ERR = f"{type(e).__name__}: {e}"
        msg = f"!!! stage3{SFX}.html: pose tabs / SAM reference NOT injected: {INJECT_ERR} (page built with a red banner)"
        print(msg); print(msg, file=sys.stderr)
# 10-12 s3branches_<page>.json "inject_js" = a fragment file (e.g. stage3_ppm.inc.js) added as its OWN <script> after the template's last </script>
# (same single injection point as the pose tabs); pages without the key are unchanged. A missing file / anchor -> red banner + loud log.
if BRC.get("inject_js") and not BRC.get("page_tpl"):
    try:
        _frag = open(f"{W}/{BRC['inject_js']}").read(); _k = tpl.rfind("</script>")
        if "</script" in _frag.lower(): raise ValueError(f"{BRC['inject_js']} contains a literal closing script tag")
        if _k < 0: raise ValueError("template anchor missing: the final </script> of stage3.tpl.html")
        _e = _k + len("</script>"); tpl = tpl[:_e] + "\n<script>\n" + _frag + "\n</script>" + tpl[_e:]
    except Exception as e:
        INJECT_ERR = f"{type(e).__name__}: {e}"; print(f"!!! stage3{SFX}.html: {BRC.get('inject_js')} NOT injected: {INJECT_ERR}", file=sys.stderr)
# per-page text / visibility for runs the duplex wording does not fit (10-10, tt runs): s3branches.json "replace" = [[old, new], ...] literal swaps in THIS
# page's template copy (header paragraph, chart titles, tile notes); "hide" = element ids whose stat tile (.si) or chart section is hidden (e.g. sm_pt = the
# persona-text tile, c_kl = the text-KL chart when KL is off). Branch keys "fit": false (not counted in Fit to ★ run) and "rel" (how it relates to ★).
# No such keys (the duplex config) = template unchanged.
for _a, _b in BRC.get("replace", []):
    if _a not in tpl: print("replace anchor missing:", _a[:80])
    tpl = tpl.replace(_a, _b)
if BRC.get("hide"):
    tpl = "<style>" + ",".join(f".si:has(#{i}),section:has(#{i})" for i in BRC["hide"]) + "{display:none!important}</style>" + tpl
ver = hashlib.sha1((tpl + json.dumps(hestia) + json.dumps(dmour) + json.dumps(dmtxd) + json.dumps(XDEP) + json.dumps(DMX) + (json.dumps(SAMREF) if SAMREF else "") + (json.dumps(DMXB) if DMXB else "")).encode()).hexdigest()[:12]
built = datetime.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles")).strftime("%H:%M PT")
try: FC = json.load(open(f"{W}/s3forecast.json"))      # frozen forecast lines for the normalised word chart
except Exception: FC = None
if BRC.get("no_forecast"): FC = None
if BRC.get("page_tpl"): hestia = dict(hestia, runs=[dict(r, train=[]) for r in hestia.get("runs", [])])      # own-template pages use Hestia eval only (keeps the page small)      # 10-12 ppm page: the frozen stage-3 forecast lines do not apply
if BRC.get("refs") is False:      # s3branches.json "refs": false = the ★ run is not on the duplex Demeter eval (e.g. tt runs): drop Hestia / Demeter / frozen-forecast lines
    FC = None; hestia = dict(hestia, runs=[]); dmour = dmtxd = DMX = None
d = {"default_xm": BRC.get("default_xm", "ep"), "forecast": FC, "ver": ver, "built": built, "live": False, "s3": s3, "hestia": hestia, "s2ref": s2ref(), "dmour": dmour, "dmtxd": dmtxd, "xdep": XDEP, "dmxdep": DMX}
if BRC.get("refs") is False: d["norefs"] = True      # page drops the (empty) Demeter series
for _k in BRC.get("strip", []):      # 10-29 "strip": page-level reference blocks this page drops (OPD page: s2ref stage-2 eval, hestia / Demeter)
    if _k == "s2ref": d["s2ref"] = {}
    elif _k in ("hestia", "dmour", "dmtxd"): d[_k] = None
if BRC.get("fit_y_from0"): d["fit_y_from0"] = True      # y-range fit includes the early steps (default skips the first 20%)
if BRC.get("empty_notes"): d["empty_notes"] = BRC["empty_notes"]      # {chart id: note shown while that chart has no data}
if SAMREF: d["samref"] = SAMREF      # pose page only (see SAMREF above)
if DMXB: d["dmxdepB"] = DMXB      # PPM page only (see DMXB above)
t = tpl.replace("__TUN__", TUN or ".").replace("__DATA__", json.dumps(d, separators=(",", ":")))
head = "<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content=\"width=device-width,initial-scale=1\"></head><body>"
import html as _html
BANNER = (f'<div id="pt_fail" style="background:#c0392b;color:#fff;font:600 14px/1.4 system-ui,sans-serif;padding:10px 14px;margin:8px">pose tabs failed to load: '
          f'{_html.escape(INJECT_ERR)} (build_stage3.py; the rest of the page is unaffected)</div>') if INJECT_ERR else ""
try:      # <W>/stage3_<page>.inject_error.txt exists exactly while the injection is failing (for watchers); removed on the next good build
    _ef = f"{W}/stage3{SFX}.inject_error.txt"
    if INJECT_ERR: open(_ef, "w").write(INJECT_ERR + "\n")
    elif os.path.exists(_ef): os.remove(_ef)
except Exception: pass
open(f"{W}/stage3{SFX}.html.tmp", "w").write(head + BANNER + t + "</body></html>"); os.replace(f"{W}/stage3{SFX}.html.tmp", f"{W}/stage3{SFX}.html")
open(f"{W}/stage3{SFX}_artifact.html.tmp", "w").write(tpl.replace("__TUN__", TUN or "https://trycloudflare.com").replace("__DATA__", json.dumps(d, separators=(",", ":")))); os.replace(f"{W}/stage3{SFX}_artifact.html.tmp", f"{W}/stage3{SFX}_artifact.html")
open(f"{W}/ver_s3{SFX}.txt.tmp", "w").write(ver); os.replace(f"{W}/ver_s3{SFX}.txt.tmp", f"{W}/ver_s3{SFX}.txt")
print(f"stage3{SFX}.html", "live" if s3.get("started") else "not started", "step", s3.get("step"), "ver", ver, "s2ref", d["s2ref"].get("src"))
