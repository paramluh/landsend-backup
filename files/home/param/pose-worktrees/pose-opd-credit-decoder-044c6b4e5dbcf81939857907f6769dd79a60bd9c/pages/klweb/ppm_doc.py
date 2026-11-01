#!/usr/bin/env python3
"""PPM live doc for klweb/stage3_ppm.html (S3_PAGE=ppm; s3branches_ppm.json "doc" points build_stage3.py here instead of s3_doc.py).
Usage: [S3_RUN=<run base, default ppm-v1>] ppm_doc.py <out.json> [chain:<job>]
Writes the s3_doc.py schema (jobs, step, max_steps, rows, epoch, mean50_s, t_step1, t_last, cols/pts, train_cols/train ...) so the stage-3 template's
header tiles / ETA / step-time / grad-norm charts work, with PPM's own column names (stage3_ppm.inc.js draws the PPM charts), plus "ppm" (x constants
and eval-row descriptions) and "refs" (reference series from s3_doc.py runs; s3branches_ppm.json "ppm_refs").
Chain: newest ppm/logs/ppm-<job>.log whose run tag (first line "run <tag>") minus -j<job> == S3_RUN; it extends back while a log says
"[resume] .../dcp-N: step N" (the previous leg = the newest earlier log of the same run that logged step >= N; its rows count up to N).
Source per job: ppm/runs/<tag>/metrics.jsonl (one JSON per line; train keys train_<task>/*, train/*; eval keys eval_<task>/*, eval/*).
x: speech epoch = world x FULL speech train rows/rank (log "[data]" line); rows / step = world x speech micros per step (GA list);
   pose epoch = world x pose train rows/rank; pose rows / step = world x pose micros per step.
10-21 (8,1,1 schedule): GA is read PER LEG; when it changes between legs (PPM 1,2,1 = GA pose,speech,speech,text -> 8,1,1 = 8 pose, 1 speech, 1 text
   micros per step, resumed from a PPM checkpoint under the same run base), step -> rows is piecewise (ppm.segs; the rows column, rows / epoch,
   epoch_steps follow it) so the PPM legs keep their x and the new legs continue from them; ppm.marks = one {step, label, ...} per GA change (the page
   draws them as dashed vertical lines). With one schedule in the chain, every output is exactly as before (no segs / marks / epoch_bounds keys).
10-21 later (user): 8,1,1 has its OWN page (S3_PAGE=ppm811, s3branches_ppm811.json). Leg kind: '811' = env file .../code-811/... or a GA with
   >= 2 pose micros, else 'ppm'. PPM page (PPM_SCHED unset / ppm): 8,1,1 legs never enter the chain (PPM main ends where it stopped, 16,861).
   PPM_SCHED=811 (default when S3_PAGE=ppm811): chain = newest 8,1,1 leg back through its resumes (8,1,1 legs first, then the PPM legs it continued
   from); rows / segs / marks / hp use the whole chain, the output rows (train, evals, jobs, step times, GenghisBench) are the 8,1,1 legs' only.
   Page-scoped config / caches: s3branches_<S3_PAGE>.json ppm_refs, s3ref_<S3_PAGE>_<id>.json, <S3_PAGE>_gb_cache.json (S3_PAGE ppm = old names).
10-21 ppm14b page (S3_PAGE=ppm14b, s3branches_ppm14b.json "main_env": PPM_ROOT=ppm/ppm14b, PPM_LOG_PREFIX=ppm14b-, PPM_HEAD=400): run base ppm14b, logs
   "== ppm14b job ... env <file> run ppm14b-j<job> mode sum" (the job-line regex also takes "ppm<N>b") and "ILP <tag>: mode ..." (tag_of); with PPM_HEAD set,
   hp() looks for the "cfg {" line within the first PPM_HEAD lines (its ILP line is ~line 18; default 6 lines otherwise, so the PPM / 8,1,1 docs are unchanged)."""
import re, sys, json, glob, os, datetime, zoneinfo, subprocess, time
OUT = sys.argv[1]; ARG = sys.argv[2] if len(sys.argv) > 2 else ""
# 10-17 env overrides (defaults = PPM main, unchanged): the interleaved re-run (s3branches_ppm.json branch "ppm-ilp") runs this same parser with
# PPM_ROOT=ppm/interleave-ppm (or ppm/next-arch, prefix ppm-next-), PPM_LOG_PREFIX=ppm-ilp-, PPM_HEAD=400 (its log header is longer), PPM_DOC_LITE=1 (no refs / GenghisBench / hub: main doc owns those caches)
P = os.environ.get("PPM_ROOT") or "/data/home/amartyavarma/ppm"; LD = os.environ.get("PPM_LD") or f"{P}/logs"; RD = os.environ.get("PPM_RD") or f"{P}/runs"; W = os.path.dirname(os.path.abspath(__file__))   # 10-17 PPM_LD / PPM_RD: a stopped run kept under logs/aborted, runs/aborted
PFX = os.environ.get("PPM_LOG_PREFIX") or "ppm-"; HEADN = int(os.environ.get("PPM_HEAD") or 60); LITE = os.environ.get("PPM_DOC_LITE") == "1"
PT = zoneinfo.ZoneInfo("America/Los_Angeles"); RUN = os.environ.get("S3_RUN") or "ppm-v1"
PG = os.environ.get("S3_PAGE") or "ppm"; SCHED = os.environ.get("PPM_SCHED") or ("811" if PG == "ppm811" else "ppm"); S811 = SCHED == "811"      # 10-21 (see docstring)
LOGRE = re.compile(re.escape(PFX) + r"(\d+)\.log$")
f = lambda x: None if x is None else round(float(x), 5)
# 10-20 glitchfix: from job 25627 each of the 128 ranks prints a torch C++ warning ("[W1019 18:07:10.239430409 Utils.hpp:137] Warning: ...
# TORCH_NCCL_TRACE_BUFFER_SIZE is deprecated ...") into the log header, some of them spliced into the middle of rank 0's long "cfg {...}" line.
# The "[resume] .../dcp-N" line moved to line 138, past the 60-line head -> resume None -> the chain kept only 25627 + 26271 and the page lost
# steps 0..13035. lines(L) = the log with those warnings removed (a line they were spliced into is re-joined), used for every log read here.
TORCHW = re.compile(r"\[[WEI]\d{4} \d\d:\d\d:\d\d\.\d+ [^\]]*\] .*$")
def lines(L):
    carry = ""
    with open(L, errors="replace") as fh:
        for line in fh:
            m = TORCHW.search(line) if "[" in line else None
            if m:
                carry += line[:m.start()]; continue
            if carry: line, carry = carry + line, ""
            yield line
    if carry: yield carry
def head(L, n=40):
    out = []
    try:
        for line in lines(L):
            if len(out) >= n: break
            out.append(line)
    except Exception: pass
    return out
def tag_of(L):
    for line in head(L, 3):
        m = re.search(r" run (\S+)\s*$", line) or re.search(r"PPM (\S+): world", line)
        if m: return m[1]
    for line in head(L, HEADN):      # 10-17 interleaved run: "== ppm-ilp job ... run <tag> mode interleave" / "ILP <tag>: mode ..." (patterns absent from PPM main logs)
        m = re.search(r"^== ppm-[a-z]+ job .* run (\S+) mode \S+\s*$", line) or re.search(r"\] ILP (\S+): mode ", line)
        if m: return m[1]
    return None
def run_of(L):
    t = tag_of(L); return re.sub(r"-j\d+$", "", t) if t else None
def max_step(L):
    m = None
    for line in lines(L):
        x = re.search(r"\] step (\d+)/\d+ ", line)
        if x: m = int(x[1])
    return m
def resume_of(L):
    for line in head(L, HEADN):
        m = re.search(r"\[resume\] \S*/dcp-(\d+): step (\d+)", line)
        if m: return int(m[2])
    return None
_KIND = {}
def kind(L):
    """10-21: '811' = an 8,1,1 leg (first line "... PT env .../code-811/... run ..." or a GA with >= 2 pose micros), else 'ppm'.
    10-23: + code-811-<x>/ copies (the LR-decay branch: env .../code-811-decay/prod_811_decay.env, log ppmdecay-<job>.log; no ppm- log uses one)."""
    if L not in _KIND:
        k = "ppm"
        for line in head(L, HEADN):
            if re.search(r" PT env \S*/code-811(?:-[\w.]+)?/\S* run ", line): k = "811"; break
            m = re.search(r"world \d+ \(lw \d+\); GA \(([^)]*)\)", line)
            if m:
                if re.findall(r"'(\w+)'", m[1]).count("pose") >= 2: k = "811"
                break
        _KIND[L] = k
    return _KIND[L]
alllogs = sorted([p for p in glob.glob(f"{LD}/{PFX}*.log") if LOGRE.search(p)], key=lambda p: int(LOGRE.search(p)[1]))
mine = [p for p in alllogs if run_of(p) == RUN and (S811 or kind(p) == "ppm")]      # the PPM page never sees 8,1,1 legs
def earlier(L, ok):
    """chain predecessors of leg L: earlier legs of its own kind passing ok(); an 8,1,1 leg with none falls back to the PPM legs (its switch checkpoint)"""
    j0 = int(LOGRE.search(L)[1]); c = [q for q in mine if int(LOGRE.search(q)[1]) < j0 and ok(q)]
    same = [q for q in c if kind(q) == kind(L)]
    return same if same or kind(L) == "ppm" else [q for q in c if kind(q) == "ppm"]
if ARG.startswith("chain:"): chain = [f"{LD}/{PFX}{ARG[6:]}.log"]
else: chain = [p for p in mine if kind(p) == SCHED][-1:]
while chain:
    r = resume_of(chain[0])
    if r is None:
        # 10-12: a relaunch whose resume decision is not logged yet (no "[resume] .../dcp-N" and no "fresh start" line): keep the previous leg in the
        # chain meanwhile (whole, uncut) so the page never drops to "not started" during the minutes a new job spends loading
        if len(chain) == 1 and not any("fresh start" in x for x in head(chain[0], HEADN)):
            prev = earlier(chain[0], lambda q: (max_step(q) or -1) >= 0)
            if prev: chain.insert(0, prev[-1]); continue
        break
    # 10-21 23:30 (fresh 8,1,1 restarted from dcp-16861 by job 27100): the predecessor must have STARTED before the resume step (it wrote dcp-r);
    # legs of an abandoned branch that started at / after r (26391 from 16861, 27030 from 17012) never enter the chain
    prev = earlier(chain[0], lambda p: (max_step(p) or -1) >= r and (resume_of(p) or 0) < r)
    if not prev or prev[-1] in chain: break
    chain.insert(0, prev[-1])
chain = [L for L in chain if os.path.exists(L)]
job = lambda L: LOGRE.search(L)[1]
CUT = {}
for a, b in zip(chain, chain[1:]):
    r = resume_of(b)
    if r is not None: CUT[a] = r
# ---- column maps (PPM metrics.jsonl key -> page column)
TASKS = {"sp": "speech", "po": "pose", "gm": "gemma", "ch": "chat"}
MET = {"nword": "word_ce_norm", "top1": "word_top1", "pad": "pad_ce", "nw": "nw_ce", "text": "text_ce", "cb0": "cb0_ce", "cb0_top1": "cb0_top1", "light": "light_ce"}
def keys(pre):
    o = []
    for t, tn in TASKS.items():
        for m, mk in MET.items():
            if t in ("gm", "ch") and m in ("cb0", "cb0_top1", "light"): continue
            o.append((f"{t}_{m}", f"{pre}_{tn}/{mk}"))
    o += [("pose_cb0", f"{pre}_pose/pose_cb0_ce"), ("pose_cb0_top1", f"{pre}_pose/pose_cb0_top1"), ("pose_light", f"{pre}_pose/pose_light_ce")]
    o += [("pose_head_cb0", f"{pre}_pose/pose_head_cb0_ce"), ("pose_head_cb0_top1", f"{pre}_pose/pose_head_cb0_top1")]    # D32 aux pose head (10-12)
    return o
# eval-only tasks (from the D2 resume, 10-12): the stage-3 eval rows (Demeter 'ourrows' 103, lanes A / B + pooled) and the stage-3 Gemma
# text-duplex eval packs (gemma-txtdup-v1 0..151, lanes A / B + pooled); no train counterpart
EVAL_ONLY = {"s3": "speech_s3", "s3a": "speech_s3_A", "s3b": "speech_s3_B", "g3": "gemma_s3", "g3a": "gemma_s3_A", "g3b": "gemma_s3_B"}
EK = keys("eval") + [(f"{t}_{m}", f"eval_{tn}/{mk}") for t, tn in EVAL_ONLY.items() for m, mk in MET.items()
                     if not (t.startswith("g3") and m in ("cb0", "cb0_top1", "light"))]
TK = keys("train")
# per-codebook light CE (logging only, from the D2 resume): speech decoder cb 8/16/31 on speech and pose rows, pose decoder cb 1/8/16/31;
# per-channel speech eval: A = the eval rows with sides flipped (PPM scores only channel B), B = eval_speech
def cbkeys(pre):
    return [(f"sp_light_cb{c}", f"{pre}_speech/light_ce_cb{c}") for c in (8, 16, 31)] + [(f"po_light_cb{c}", f"{pre}_pose/light_ce_cb{c}") for c in (8, 16, 31)] + \
           [(f"pose_light_cb{c}", f"{pre}_pose/pose_light_ce_cb{c}") for c in (1, 8, 16, 31)]
EK += cbkeys("eval") + [("spa_nword", "eval_speech_A/word_ce_norm"), ("spb_nword", "eval_speech_B/word_ce_norm")]
# 10-12 B-given-A dependence probe (train_ppm.py xdep: speech_s3 rows 0..31 x lanes A, B; channel A silent / absent on the trunk input; ablated - intact)
EK += [(f"xd_{m}_{c}", f"xdep/b_given_a/d_{mk}_{cn}") for m, mk in (("w", "word_ce_norm"), ("c", "cb0_ce"), ("l", "light_ce")) for c, cn in (("sil", "silent"), ("abs", "absent"))] + \
      [("xd_real", "xdep/intact/word_ce_norm"), ("xd_real_cb0", "xdep/intact/cb0_ce"), ("xd_s", "xdep/time_s"), ("cd_s", "codependence/time_s")]
TK += cbkeys("train")
GN = ["trunk", "spec", "proj", "audio", "tower_s", "tower_p"]
TRAIN_COLS = ["step", "rows"] + [c for c, _ in TK] + ["step_s", "micro_pose", "micro_speech", "micro_text", "wait_s", "gn", "gn_towers"] + \
             [f"gn_{g}" for g in GN] + ["lr_trunk", "peak_gib", "gemma_frac", "A_nword", "B_nword", "gn_pose_head", "lr_pose_head"]   # D32: head's own group (not in gn)
# 10-21 8,1,1 page only (appended, so the PPM page's columns are unchanged): per-step timing of the 8,1,1 micros, decoder updates / frame coverage; the
# link-on pose probe (eval main pose light = link off in 8,1,1; decp_linkon_* = the frozen D33 link re-added)
XT811 = [("micro_pose_full", "train/micro_s_pose_full"), ("micro_pose_frozen", "train/micro_s_pose_frozen"), ("micro_speech_full", "train/micro_s_speech_full"),
         ("micro_text_full", "train/micro_s_text_full"), ("opt_s", "train/opt_s"), ("dec_p_updates", "train/dec_p_updates"), ("dec_s_updates", "train/dec_s_updates"),
         ("dec_p_frame_frac", "train/dec_p_frame_frac"), ("dec_s_frame_frac", "train/dec_s_frame_frac")] if S811 else []
XE811 = [("pose_linkon_light", "eval_pose/decp_linkon_light_ce")] + [(f"pose_linkon_cb{c}", f"eval_pose/decp_linkon_light_ce_cb{c}") for c in (1, 8, 16, 31)] if S811 else []
TRAIN_COLS += [c for c, _ in XT811]
EVAL_COLS = ["step", "rows"] + [c for c, _ in EK] + ["eval_s", "A_nword", "B_nword"] + [c for c, _ in XE811]   # A_ / B_nword = the pooled speech nword in both (template c_nword chart = their mean)
def ts_of(s, year):
    try: return datetime.datetime.strptime(f"{year}-{s[:-3]}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=PT).timestamp()
    except Exception: return None
train, ev = {}, {}; tsrc, esrc = {}, {}; tags = []; world = None; sp_rpr = po_rpr = None; ga = None; max_steps = None; t_job0 = t_step1 = t_last = None
stopped = None; done = False; tstamp = {}; d2 = None
last_save = None; save_min = None; last_eval = None; eval_min = None      # 10-22 (user): "time to next checkpoint" on the live page: newest own [save] line (t, step) + cfg save_every_min
legs = []      # 10-21: per chain leg {job, start (its "[resume] .../dcp-N: step N", 0 for the first leg), world, ga, sched ("[811] SCHEDULE ..." line)}
env_file = cfg_max = sp_min = None      # 10-19 maxsteps: the newest leg's env file / resolved cfg max_steps / min FULL speech rows per rank (see max_steps_set below)
for L in chain:
    cut = CUT.get(L, 10 ** 9); j = job(L); tag = tag_of(L) or f"{RUN}-j{j}"; year = None; stopped = None; done = False; sp_min_leg = None
    leg_w = leg_ga = leg_sched = None; own = not S811 or kind(L) == "811"      # 10-21: on the 8,1,1 page only its own legs give rows / jobs / start times
    leg_fpose = None                            # 10-22: the leg's "[fresh] POSE from step index S_f" (fresh pose rows, never repeated)
    leg_data = leg_sphi = leg_store = None      # 10-21 8,1,1 data-plan marks: the leg's "[811] data from step R: ..." line, D2 max speech rows / rank, pose store
    if own: tags.append(tag)
    for line in lines(L):
        m = re.search(r"^== ppm(?:-[a-z]+|\d+b)? job \S+ .*?(\d{4})-(\d\d-\d\d) (\d\d:\d\d:\d\d) PT", line)      # 10-21: + "== ppm14b job"
        if m:
            year = m[1]
            if t_job0 is None and own: t_job0 = datetime.datetime.fromisoformat(f"{m[1]}-{m[2]} {m[3]}").replace(tzinfo=PT).timestamp()
            m = re.search(r" PT env (\S+) run ", line)
            if m: env_file = m[1]
        m = re.search(r"world (\d+) \(lw \d+\); GA \(([^)]*)\)", line)
        if m: world = int(m[1]); ga = re.findall(r"'(\w+)'", m[2]); leg_w, leg_ga = world, ga
        m = re.search(r"\[811\] (SCHEDULE \S+ from step \d+.*)$", line)
        if m and leg_sched is None: leg_sched = m[1].strip()
        m = re.search(r"\[811\] data from step (\d+): (.*)$", line)
        if m and leg_data is None: leg_data = (int(m[1]), m[2])
        m = re.search(r"\[fresh\] POSE from step index (\d+)", line)
        if m and leg_fpose is None: leg_fpose = int(m[1])
        if '"pose_root"' in line:
            m = re.search(r'"pose_root": "/scratch/amartyavarma/([^/"]+)/rows', line)
            if m: leg_store = m[1]
        m = re.search(r"^\[(\d\d-\d\d \d\d:\d\d:\d\d PT)\] \[save\] \S*/dcp-(\d+) node-local", line)
        if m and own and year:
            t_ = ts_of(m[1], year)
            if t_: last_save = [t_, int(m[2])]
        m = re.search(r"^\[(\d\d-\d\d \d\d:\d\d:\d\d PT)\] EVAL step (\d+):", line)      # 10-22 (user): + time to next eval
        if m and own and year:
            t_ = ts_of(m[1], year)
            if t_: last_eval = [t_, int(m[2])]
        if " cfg {" in line and own:
            m = re.search(r'"save_every_min": ([\d.]+)', line)
            if m: save_min = float(m[1])
            m = re.search(r'"eval_every_min": ([\d.]+)', line)
            if m: eval_min = float(m[1])
        if " cfg {" in line:
            m = re.search(r'"max_steps": (\d+)', line)
            if m: cfg_max = int(m[1])
        m = re.search(r"\[data\] D2 extra slices .*?; speech rows/rank ([\d,]+)\.\.([\d,]+)", line)
        if m: sp_min_leg = int(m[1].replace(",", "")); leg_sphi = int(m[2].replace(",", ""))
        m = re.search(r"pose node \S+: train rows/rank ([\d,]+); speech node \S+: FULL train rows/rank ([\d,]+).*max_steps (\d+)", line)
        if m: po_rpr, sp_rpr, max_steps = int(m[1].replace(",", "")), int(m[2].replace(",", "")), int(m[3]); sp_min = sp_min_leg
        m = re.search(r"\[data\] pose EXCLUDED; speech node \S+: FULL train rows/rank ([\d,]+).*max_steps (\d+)", line)     # 10-17 interleaved run (no pose)
        if m: sp_rpr, max_steps = int(m[1].replace(",", "")), int(m[2]); sp_min = sp_min_leg
        m = re.search(r"STOP\w*\b.*?\bstep (\d+)", line)
        if m and "[safety]" not in line: stopped = int(m[1])
        if re.search(r"\] done \d+ steps", line): done = True
        m = re.search(r"\[data\] D2 extra slices .*? from step (\d+):.*?ALREADY SEEN before step \d+: ([\d,]+) rows", line)
        if m: d2 = {"from": int(m[1]), "seen_holdout_rows": int(m[2].replace(",", ""))}
    year = year or str(datetime.date.today().year)
    legs.append({"job": j, "start": 0 if not legs else resume_of(L), "world": leg_w, "ga": leg_ga, "sched": leg_sched, "kind": kind(L),
                 "data": leg_data, "sp_lo": sp_min_leg, "sp_hi": leg_sphi, "store": leg_store, "fpose": leg_fpose})
    mf = f"{RD}/{tag}/metrics.jsonl"
    if not os.path.exists(mf):
        c = glob.glob(f"{RD}/*-j{j}/metrics.jsonl"); mf = c[0] if c else None
    if not mf: continue
    for line in open(mf, errors="replace"):
        try: d = json.loads(line)
        except Exception: continue
        s = d.get("step")
        if s is None or int(s) > cut: continue
        s = int(s); ts = ts_of(d.get("time_pt", ""), year)
        if "train/step_time" in d:
            g = lambda k: f(d.get(k))
            gns = [d.get(f"train/gn_{x}") for x in GN]
            tot = (sum(float(v) ** 2 for v in gns if v is not None)) ** 0.5 if any(v is not None for v in gns) else None
            tw = [d.get("train/gn_tower_s"), d.get("train/gn_tower_p")]
            tws = (sum(float(v) ** 2 for v in tw if v is not None)) ** 0.5 if any(v is not None for v in tw) else None
            train[s] = [s, None] + [g(k) for _, k in TK] + [g("train/step_time"), g("train/micro_s_pose"), g("train/micro_s_speech"), g("train/micro_s_text"),
                        g("train/data_wait"), f(tot), f(tws)] + [g(f"train/gn_{x}") for x in GN] + [d.get("train/lr_trunk"), g("train/peak_gib"), g("train/text_gemma_frac"), g("train_speech/word_ce_norm"), g("train_speech/word_ce_norm"),
                        g("train/gn_pose_head"), d.get("train/lr_pose_head")] + [g(k) for _, k in XT811]
            tsrc[s] = own
            if ts:
                if t_step1 is None and s >= 1 and own: t_step1 = ts
                t_last = ts
        if any(k.startswith("eval_") for k in d):
            ev[s] = [s, None] + [f(d.get(k)) for _, k in EK] + [f(d.get("eval/time_s")), f(d.get("eval_speech/word_ce_norm")), f(d.get("eval_speech/word_ce_norm"))] + [f(d.get(k)) for _, k in XE811]
            esrc[s] = own
nsp = (ga or []).count("speech") or 2; npo = (ga or []).count("pose") or 1
rps = (world or 128) * nsp; prps = (world or 128) * npo
epoch_rows = (world or 128) * sp_rpr if sp_rpr else rps * (max_steps or 13766)
pose_epoch_rows = (world or 128) * po_rpr if po_rpr else prps * (max_steps or 13766)
# 10-21 piecewise step -> rows. eff = the schedule in force per step range: legs in chain order, a leg from its resume step on replaces what came before
# (a leg without a GA line yet inherits the previous one; one without a resume line yet has no steps). segs = maximal ranges of constant rates:
# {s: first boundary step, sp / po: speech / pose rows per step for steps > s, sp0 / po0: rows at step s, ns: speech micros per rank}.
def _rates(w, g): return (w or 128) * ((g or []).count("speech") or 2), (w or 128) * ((g or []).count("pose") or 1), (g or []).count("speech") or 2
def _sname(g): return ",".join(str((g or []).count(t)) for t in ("pose", "speech", "text"))
eff = []
for lg in legs:
    if lg["start"] is None: continue
    w_, g_ = lg["world"], lg["ga"]
    if g_ is None and eff: w_, g_ = eff[-1]["world"], eff[-1]["ga"]
    eff = [e for e in eff if e["s"] < lg["start"]] + [{"s": lg["start"], "world": w_, "ga": g_, "job": lg["job"], "sched": lg["sched"]}]
segs = []
for e in eff:
    sp_, po_, ns_ = _rates(e["world"], e["ga"])
    if segs and (segs[-1]["sp"], segs[-1]["po"], segs[-1]["ns"]) == (sp_, po_, ns_): continue
    q = segs[-1] if segs else None
    segs.append({"s": e["s"], "sp": sp_, "po": po_, "sp0": q["sp0"] + (e["s"] - q["s"]) * q["sp"] if q else 0, "po0": q["po0"] + (e["s"] - q["s"]) * q["po"] if q else 0,
                 "ns": ns_, "r0": q["r0"] + (e["s"] - q["s"]) * q["ns"] if q else 0, "ga": _sname(e["ga"]), "job": e["job"]})
marks = []
for a, b in zip(eff, eff[1:]):
    if a["ga"] is not None and b["ga"] is not None and a["ga"] != b["ga"]:
        (spa, poa, _), (spb, pob, _) = _rates(a["world"], a["ga"]), _rates(b["world"], b["ga"])
        marks.append({"step": b["s"], "kind": "ga", "name": _sname(b["ga"]), "was": _sname(a["ga"]), "label": f"{_sname(b['ga'])} from step {b['s']}", "job": b["job"],
                      "detail": f"{_sname(b['ga'])} from step {b['s']:,} (was {_sname(a['ga'])}; pose, speech, text micros per step) · job {b['job']}\n"
                                f"speech rows / step {spa:,} → {spb:,} · pose rows / step {poa:,} → {pob:,}", "sched": (b["sched"] or "")[:300]})      # detail = the 2-line hover
PIECE = len(segs) > 1
def rows_at(s):
    g = segs[0]
    for q in segs:
        if s >= q["s"]: g = q
    return g["sp0"] + (s - g["s"]) * g["sp"]
def step_at_rank_rows(n):      # last step whose cumulative speech rows per rank <= n (one schedule: n // speech micros per step, as before)
    g = segs[0]
    for q in segs:
        if q["r0"] <= n: g = q
    return g["s"] + (n - g["r0"]) // g["ns"]
for s, r in train.items(): r[1] = rows_at(s) if PIECE else s * rps
for s, r in ev.items(): r[1] = rows_at(s) if PIECE else s * rps
if S811:      # 10-21: the 8,1,1 page keeps its own legs' rows only (the PPM legs before the switch are its reference branch)
    train = {s: r for s, r in train.items() if tsrc.get(s)}; ev = {s: r for s, r in ev.items() if esrc.get(s)}
S0_811 = next((lg["start"] for lg in legs if lg["start"] is not None and lg["kind"] == "811"), None)      # first 8,1,1 leg's resume step (the switch)
def _rows_at(R, key):      # world-level rows of stream key ('sp' speech / 'po' pose) used by step R, through the segs
    g = segs[0]
    for q in segs:
        if R >= q["s"]: g = q
    return g[key + "0"] + (R - g["s"]) * g[key]
def _pose_rem(store, used, wr):
    """per-rank pose rows left of the epoch-1 order after `used` rows per rank: the store's per-node train rows (counts.json), each node's rows striped
    over its ranks (lw = world / nodes); None if unreadable"""
    try:
        nodes = json.load(open(f"/data/home/amartyavarma/qwen-1ch-data/{store}/counts.json"))["nodes"]; lw = wr // len(nodes); o = []
        for v in nodes.values():
            t = int(v["train"]["rows"]); o += [t // lw + (1 if r < t % lw else 0) - used for r in range(lw)]
        return o
    except Exception: return None
def data_marks():
    """10-21 (user): where each 8,1,1 data stream starts repeating rows (epoch e >= 2), from the leg's "[811] data from step R: pose = ... (N rows over all ranks,
    rank 0 a) ... rank-0 stream epochs at stream row [(1, 0), (2, a)]; speech = ... (M rows, rank 0 b) ... rank-0 epochs at [(1, 0), (2, b)]" line: the mark is
    rank 0's first step with an epoch-e row (R + row // micros per step + 1); band = the same step over all ranks, for the epoch-2 crossing only and only when
    it reproduces the logged total (pose: per-node store rows; speech: the D2 per-rank range)"""
    out = {}
    for lg in legs:
        if lg["kind"] != "811" or not lg.get("data") or lg["start"] is None: continue
        R, txt = lg["data"]; wr = lg["world"] or world or 128
        npo = (lg["ga"] or []).count("pose") or 1; nsp_ = (lg["ga"] or []).count("speech") or 1
        for stream, rx, nper in (("pose", r"pose = .*?\(([\d,]+) rows over all ranks, rank 0 ([\d,]+)\).*?rank-0 stream epochs at stream row \[([^\]]*)\]", npo),
                                 ("speech", r"speech = .*?\(([\d,]+) rows, rank 0 ([\d,]+)\).*?rank-0 epochs at \[([^\]]*)\]", nsp_)):
            if stream == "pose" and lg.get("fpose") is not None: continue   # 10-22 (user): fresh-pose legs never repeat pose rows (the [811] line's pose epochs are the unused old store plan)
            m = re.search(rx, txt)
            if not m: continue
            tot, r0 = int(m[1].replace(",", "")), int(m[2].replace(",", ""))
            for e, row in ((int(a), int(b)) for a, b in re.findall(r"\((\d+), (\d+)\)", m[3])):
                if e < 2 or (stream, e) in out: continue
                step = R + row // nper + 1; band = None; how = ""
                if e == 2 and row == r0 and PIECE:
                    used = _rows_at(R, "po" if stream == "pose" else "sp") // wr
                    if stream == "pose":
                        rem = _pose_rem(lg.get("store"), used, wr)
                        if rem and len(rem) == wr and sum(rem) == tot and min(rem) > 0:
                            band = [R + min(rem) // nper + 1, R + max(rem) // nper + 1]; how = f"PPM epoch-1 pose rows left per rank {min(rem):,}..{max(rem):,} ({tot:,} in all)"
                    elif lg.get("sp_lo") and lg.get("sp_hi"):
                        lo, hi = lg["sp_lo"] - used, lg["sp_hi"] - used
                        if 0 < lo <= r0 <= hi and wr * lo <= tot <= wr * hi:
                            band = [R + lo // nper + 1, R + hi // nper + 1]; how = f"PPM D2 speech rows left per rank {lo:,}..{hi:,} ({tot:,} in all)"
                lab = f"{stream} epoch {e} begins (repeated rows)"
                det = f"{lab}: rank 0 at step {step:,}" + (f"\nall {wr} ranks between steps {band[0]:,} and {band[1]:,} (shaded)" if band else "") + (f"\n{how}" if how else "")
                out[(stream, e)] = {"step": step, "kind": "data", "name": f"{stream} epoch {e}", "label": lab, "band": band, "job": lg["job"], "detail": det,
                                    "hide": stream == "speech" and e == 2}      # 10-27 user: remove the speech epoch 2 vertical line (kept in data; not drawn)
    return sorted(out.values(), key=lambda x: x["step"])
if S811: marks = marks + data_marks()
tr = [train[s] for s in sorted(train)]; pts = [ev[s] for s in sorted(ev)]
st = max(train) if train else None
si = TRAIN_COLS.index("step_s")
t5 = [r[si] for r in tr if r[0] >= 5 and r[si] is not None]; t50 = [r[si] for r in tr[-50:] if r[si] is not None]
if st is not None and max_steps and st >= max_steps: done = True
# 10-19 (user): the page's forecast / fit lines run to the USER-SET max steps, not to the trainer's max_steps (train_ppm.py clamps it to the data:
# min(PPM_MAX_STEPS, steps available)). max_steps_set = PPM_MAX_STEPS in the env file the newest leg was launched with (its "== ppm job ... env <file>"
# line), re-read on every build so an edit shows on the next refresh; fallback the newest leg's resolved cfg max_steps, then max_steps.
# epoch_steps = one full epoch of the speech data plan = min over ranks of FULL speech rows/rank (D2 "[data]" line) // speech micros per step
# (= the trainer's "steps available" = where the data ends); without a D2 line, this host's FULL rows/rank // speech micros per step.
def env_max(p):
    v = None
    try:
        for line in open(p, errors="replace"):
            m = re.match(r"\s*(?:export\s+)?PPM_MAX_STEPS=[\"']?(\d+)", line)
            if m: v = int(m[1])      # last assignment wins, as when the file is sourced
    except Exception: pass
    return v
_em = env_max(env_file) if env_file else None
max_steps_set = _em or cfg_max or max_steps
max_steps_src = f"PPM_MAX_STEPS in {env_file}" if _em else "resolved cfg max_steps (newest leg)" if cfg_max else "trainer max_steps"
_spr = sp_min or sp_rpr
epoch_steps = (step_at_rank_rows(_spr) if PIECE else _spr // nsp) if _spr else None      # 10-21: piecewise when the schedule changed (the speech data runs out later at 1 speech micro / step)
# ---- reference series (stage-3 runs through s3_doc.py; cached, refreshed every 20 min since they are stopped runs)
def refs():
    try: cfg = json.load(open(f"{W}/s3branches_{PG}.json")).get("ppm_refs", [])
    except Exception: cfg = []
    out = []
    for rc in cfg:
        cache = f"{W}/s3ref_{PG}_{rc['id']}.json"
        try:
            if time.time() - os.path.getmtime(cache) < 1200: out.append(json.load(open(cache))); continue
        except Exception: pass
        tmp = cache + ".tmp"; env = dict(os.environ, S3_RUN=rc["run"])
        cmd = [sys.executable, "/data/home/amartyavarma/setup-logs/s3_doc.py", tmp] + ([f"chain:{rc['job']}"] if rc.get("job") else [])
        try:
            if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, timeout=240).returncode != 0: raise RuntimeError("s3_doc failed")
            d = json.load(open(tmp)); os.remove(tmp)
            o = {k: rc.get(k) for k in ("id", "label", "kind", "color", "color_dark", "eval_rows", "note")} | {
                "run": rc["run"], "jobs": d.get("jobs"), "step": d.get("step"), "epoch_rows": d.get("epoch_rows"), "rows_per_step": d.get("rows_per_step"),
                "cols": d["cols"], "pts": d["pts"], "train_cols": d["train_cols"], "train": d["train"][::max(1, len(d["train"]) // 1500)]}
            if rc.get("kind") == "pose":
                sys.path.insert(0, W); import pose_tabs
                x = next((x for x, _ in (pose_tabs.pose_epoch(jj) for jj in d.get("jobs", [])) if x), None)
                o["pose_x"] = x
            json.dump(o, open(cache + ".w", "w"), separators=(",", ":")); os.replace(cache + ".w", cache); out.append(o)
        except Exception as e:
            print("ref", rc.get("id"), "failed:", e, file=sys.stderr)
            try: out.append(json.load(open(cache)))
            except Exception: pass
    return out
def s3xdep():
    """the stage-3 ★ run's B-given-A probe (xbench s3probe on its exports, results s3probe-j<job>-w<N>/summary.json by_channel.B; 32 rows x
    swaps 0, 1 = the rows / probe of PPM's xdep): pts [step, rows, w_sil, w_abs, c_sil, c_abs, real, real_cb0] (gap = oth-silent / oth-absent
    minus real, normalised word CE and cb0). Jobs / rows per step from the cached s3duplex reference (s3ref_ppm_<id>.json); later jobs win a step."""
    try:
        rc = next(r for r in json.load(open(f"{W}/s3branches_{PG}.json")).get("ppm_refs", []) if r.get("kind") == "speech")
        c = json.load(open(f"{W}/s3ref_{PG}_{rc['id']}.json")); rps = c.get("rows_per_step") or 304
    except Exception: return None
    out = {}
    for j in c.get("jobs") or []:
        for fn in glob.glob(f"/data/home/amartyavarma/qwen-1ch/xbench/results/s3probe-j{j}-w*/summary.json"):
            try:
                b = json.load(open(fn))["by_channel"]["B"]; n = int(re.search(r"-w(\d+)/", fn)[1]); w, cb = b["word_norm"], b["cb0"]
                g = lambda t, k: round(t[k], 5) if t.get(k) is not None else None
                if g(w, "gap_oth-silent") is None and g(w, "gap_oth-absent") is None: continue
                out[n] = [n, n * rps, g(w, "gap_oth-silent"), g(w, "gap_oth-absent"), g(cb, "gap_oth-silent"), g(cb, "gap_oth-absent"), g(w, "real"), g(cb, "real")]
            except Exception: pass
    return {"run": rc["run"], "cols": ["step", "rows", "w_sil", "w_abs", "c_sil", "c_abs", "real", "real_cb0"], "pts": [out[k] for k in sorted(out)]}
EVROWS = {"sp": "PPM's own speech holdout: s3-rows episodes hash-selected with salt 'ppm-v1' (frac 0.0005), FULL 7,500-frame rows only, 2 rows per rank (256 rows); NOT the stage-3 eval rows (salt 's3eval-v1')",
          "po": "PPM's own pose holdout: pose-s3-samdata rows, pose_seed 20261011, 2 rows per rank (256 rows)",
          "gm": "PPM's own Gemma text-duplex eval: txtdup-full 'eval' split packs rank*2..rank*2+1 (256 packs); NOT the stage-3 gemma-txtdup-v1 eval packs 0..151",
          "ch": "PPM's chat eval: chat split 2 holdout packs (pack plan, unshuffled), 2 per rank",
          "s3": "the stage-3 eval rows Demeter 'ourrows' was scored on: slinky-5 s3-rows, salt 's3eval-v1' (frac 0.0005), 103 FULL rows, PPM row format (system prompt, CD 2); lane B = stage-3 channel B (dataset-A rows unswapped, dataset-B rows swapped), lane A = the same rows flipped (PPM predicts only its channel B); every 's3eval-v1' episode is excluded from PPM training from the D2 resume on",
          "g3": "the stage-3 Gemma text-duplex eval: gemma-txtdup-v1 eval packs 0..151 (192 conversations, cap 7,000), swap = parity of the eval-list position (lane B), lane A flipped; these conversations are in txtdup-full's eval split, so PPM never trained on them"}
# ---- 10-12 LR tab: logged per-group learning rates (train/lr_<group>, train/dec_<s|p>_lr), raw floats (no rounding), chain cuts as above; a row is kept
# when any LR changed vs the previous kept row (plus every 25th step), so constant stretches stay small and warmup ramps stay exact
def lrs():
    rowsd, cols = {}, []
    for L in chain:
        cut = CUT.get(L, 10 ** 9); j = job(L); tag = tag_of(L) or f"{RUN}-j{j}"
        mf = f"{RD}/{tag}/metrics.jsonl"
        if not os.path.exists(mf):
            c = glob.glob(f"{RD}/*-j{j}/metrics.jsonl"); mf = c[0] if c else None
        if not mf: continue
        for line in open(mf, errors="replace"):
            if '_lr' not in line: continue
            try: d = json.loads(line)
            except Exception: continue
            st_ = d.get("step")
            if st_ is None or int(st_) > cut: continue
            v = {}
            for k, x in d.items():
                if x is None: continue
                if k.startswith("train/lr_"): v[k[9:]] = float(x)
                elif k.startswith("train/dec_") and k.endswith("_lr"): v["dec_" + k[10:-3]] = float(x)
            if v:
                for k in v:
                    if k not in cols: cols.append(k)
                rowsd[int(st_)] = v
    rows = sorted(rowsd.items()); out, prev = [], None
    for i, (st_, v) in enumerate(rows):
        if prev is None or v != prev or st_ % 25 == 0 or i == len(rows) - 1: out.append([st_] + [v.get(c) for c in cols]); prev = v
    return {"cols": ["step"] + cols, "rows": out}
# ---- 10-12 run events for the charts' vertical markers: "[resume] D<n> <what> ADDED at step N" lines in the chain logs (e.g. D32 pose head at 1083)
def events():
    o = []
    for L in chain:
        for line in head(L, 60):
            m = re.search(r"\[resume\] (D\d+) (.+?) ADDED at step (\d+)", line)
            if m: o.append({"tag": m[1], "what": m[2].strip(), "step": int(m[3]), "job": job(L)})
            m = re.search(r"\[train set\] resume adds trainable backbone layers \[([0-9, ]+)\]", line)      # 10-22 m128r: io switch marker
            if m:
                st = next((int(x[1]) for x in (re.search(r"\[resume\] /\S+/dcp-(\d+): step", l2) for l2 in head(L, 400)) if x), None)
                if st is not None:
                    ls = m[1].replace(" ", "")
                    o.append({"tag": "SWITCH", "what": f"backbone layers {ls} trainable (io: all frozen before)", "step": st, "job": job(L)})
        for line in head(L, 400):      # 10-27 pose5 (user: dotted line where the run continued with the new changes): s2p + G0COND resume lines -> one P5 marker
            m = re.search(r"\[resume\] (POSE-S2P|G0COND) ADDED at step (\d+)", line)
            if m: o.append({"tag": "P5", "what": "speech conditions pose (s2p) + frame-0 root/width conditioning (G0COND) + pose delivery 5", "step": int(m[2]), "job": job(L)})
    return sorted({(e["tag"], e["step"]): e for e in o}.values(), key=lambda e: e["step"])
# ---- 10-12 HYPERPARAMETERS tab: the resolved trainer cfg (the "cfg {...}" JSON on each chain log's header) of every leg; current = the newest leg,
# a key whose value differs from the previous leg is reported as changed at that leg's resume step (its "[resume] .../dcp-N: step N" line)
def hp():
    def flat(d, pre=""):
        o = {}
        for k, v in d.items():
            if isinstance(v, dict): o.update(flat(v, f"{pre}{k}."))
            else: o[f"{pre}{k}"] = v
        return o
    legs = []
    for L in chain:
        for line in head(L, HEADN if os.environ.get("PPM_HEAD") else 6):      # 10-21 ppm14b: its cfg line is past line 6 (PPM_HEAD runs only)
            i = line.find(" cfg {")
            if i >= 0:
                try: legs.append((job(L), resume_of(L) or 0, flat(json.loads(line[i + 5:].strip()))))
                except Exception: pass
                break
    if not legs: return None
    cur = legs[-1][2]; chg = {}
    for (j0, s0, c0), (j1, s1, c1) in zip(legs, legs[1:]):
        for k in set(c0) | set(c1):
            if c0.get(k) != c1.get(k): chg.setdefault(k, []).append({"step": s1, "job": j1, "was": c0.get(k)})
    return {"job": legs[-1][0], "from_step": legs[-1][1], "cfg": cur, "changes": chg, "legs": [[j, s] for j, s, _ in legs]}
# ---- 10-12 GenghisBench (ppm/code/gb_ppm.sh: v5 oracle, T 0.7, run_chat protocol): per checkpoint ppm/gb/<RUN>-{local,w}-N/games/<seed>.json; a set is
# plotted once all its games are in (seeds 100-119 = 20 games, 200-259 = 60 games). Per-game results cached in ppm_gb_cache.json by (path, mtime).
GB_REFS = [["base Qwen3-32B", 10, "the PPM init, its own chat template (no think), vLLM, seeds 100-119"],
           ["Qwen stage-1 w-4000", 12, "qwen-1ch stage 1 end, seeds 100-119"], ["Qwen stage-2 w-2500", 19, "qwen-1ch stage 2 j7671 w-2500, seeds 100-119"]]
def gb(smin=None, smax=None):      # (not lo / hi: the seed-set loop below uses lo)
    cf = f"{W}/{PG}_gb_cache.json"
    try: cache = json.load(open(cf))
    except Exception: cache = {}
    sets, new = {}, {}
    for d in glob.glob(f"{P}/gb/{RUN}-*/games"):
        m = re.search(r"-(?:local|w)-(\d+)/games$", d)
        if not m: continue
        stp = int(m[1])
        for fn in glob.glob(f"{d}/*.json"):
            try:
                mt = os.path.getmtime(fn); c = cache.get(fn)
                if not c or c[0] != mt:
                    r = json.load(open(fn)); c = [mt, int(r["seed"]), bool(r["success"]), int(r["questions"])]
                new[fn] = c
            except Exception: continue
            for lo, n in ((100, 20), (200, 60)):
                if lo <= c[1] < lo + n: sets.setdefault((stp, n), []).append(c)
    try: json.dump(new, open(cf + ".w", "w")); os.replace(cf + ".w", cf)
    except Exception: pass
    def med(v):
        v = sorted(v); k = len(v)
        return None if not k else (v[k // 2] if k % 2 else (v[k // 2 - 1] + v[k // 2]) / 2)
    out = {"cols": ["step", "wins", "n", "median_q"], "g20": [], "g60": [], "pending": [], "refs": GB_REFS}
    for (stp, n), cs in sorted(sets.items()):
        if (smin is not None and stp < smin) or (smax is not None and stp > smax): continue      # 10-21: shared gb/ppm-v1-* dirs (PPM page: <= its last step; 8,1,1: >= the switch)
        w = sum(c[2] for c in cs); row = [stp, w, len(cs), med([c[3] for c in cs if c[2]])]
        (out["g20" if n == 20 else "g60"] if len(cs) >= n else out["pending"]).append(row if len(cs) >= n else row + [n])
    return out
XPROBE = "/data/home/amartyavarma/ppm/next-arch/probe_xdep/results/history.jsonl"
def xprobe():
    """10-17 offline partner-channel probe (next-arch/probe_xdep, results/history.jsonl, still being extended): condition "swap" (the partner's text
    replaced by another conversation's), split "all"; per run [step, dCE, err (1 s.e.)], later lines win a (run, step). Two line formats: nested
    {step, [run], d: {cond: {split: v}}, se: {cond: {split: v}}} and flat {run, step, cond, split, dCE, err}; no "run" = next14b. A missing file or
    partial / bad line is skipped (what parses is kept), so each refresh picks up new lines."""
    out = {}
    try: fh = open(XPROBE, errors="replace")
    except Exception: return {"cond": "swap", "split": "all", "runs": {}}
    with fh:
        for line in fh:
            try:
                r = json.loads(line)
                if not isinstance(r, dict) or r.get("step") is None: continue
                if isinstance(r.get("d"), dict):
                    v = (r["d"].get("swap") or {}).get("all"); e = ((r.get("se") or {}).get("swap") or {}).get("all")
                elif r.get("cond") == "swap" and r.get("split", "all") == "all":
                    v = r.get("dCE", r.get("dce")); e = r.get("err", r.get("se"))
                else: continue
                if v is None: continue
                s = int(r["step"]); out.setdefault(str(r.get("run") or "next14b"), {})[s] = [s, round(float(v), 5), None if e is None else round(float(e), 5)]
            except Exception: continue
    return {"cond": "swap", "split": "all", "runs": {k: [v[s] for s in sorted(v)] for k, v in out.items()}}
def hub():
    """10-14: HF backups tab = setup-logs/hfb/hub_status.json (written by hfb_watch.py every cycle; rows checked against the HF listing)."""
    try: h = json.load(open("/data/home/amartyavarma/setup-logs/hfb/hub_status.json"))
    except Exception: return None
    h["age_s"] = round(time.time() - h.get("updated", 0)); return h
S0_PPM_END = st if not S811 else None      # the PPM page's last step (GenghisBench sets after it belong to the 8,1,1 run)
JOBS = [job(L) for L in chain if not S811 or kind(L) == "811"]
doc = {"jobs": JOBS, "run_tags": tags, "job_runs": {j_: RUN for j_ in JOBS}, "run": RUN, "started": bool(tr or pts), "step": st,
       "max_steps": max_steps, "max_steps_set": max_steps_set, "max_steps_src": max_steps_src, "epoch_steps": epoch_steps, "stopped": stopped, "done": done, "rows": tr[-1][1] if tr else 0, "rows_per_step": rps, "world": world, "s3_rows": nsp,
       "epoch_rows": epoch_rows, "epoch": (tr[-1][1] / epoch_rows) if tr else 0.0, "init_from": f"PPM main dcp-{S0_811} (8,1,1 switch)" if S811 else "Qwen3-32B (fresh towers / decoders)", "init_step": None,
       "mean_s": round(sum(t5) / len(t5), 2) if t5 else None, "mean50_s": round(sum(t50) / len(t50), 2) if t50 else None,
       "t_job0": t_job0, "t_step1": t_step1, "t_last": t_last, "last_save": last_save, "save_every_min": save_min, "last_eval": last_eval, "eval_every_min": eval_min, "train_cols": TRAIN_COLS, "train": tr[::max(1, len(tr) // 4000)] if len(tr) > 4000 else tr,
       "cols": EVAL_COLS, "pts": pts, "updated": datetime.datetime.now(PT).strftime("%H:%M PT"),
       "ppm": {"ga": ga, "speech_rows_per_step": rps, "speech_epoch_rows": epoch_rows, "pose_rows_per_step": prps, "pose_epoch_rows": pose_epoch_rows, "eval_rows": EVROWS, "d2": d2},
       "refs": [] if LITE else refs(), "s3xdep": None if LITE else s3xdep(), "hp": hp(), "events": events(), "lr": lrs(), "gbp": None if LITE else gb(smin=S0_811) if S811 else gb(smax=S0_PPM_END), "hub": None if LITE else hub(), "xprobe": None if LITE or S811 else xprobe()}   # "gbp": build_stage3.py owns s3["gb"] (stage-3 GenghisBench)
if PIECE:
    doc["ppm"]["segs"] = [{k: q[k] for k in ("s", "sp", "po", "sp0", "po0", "ga", "job")} for q in segs]
    if _spr: doc["ppm"]["epoch_bounds"] = [step_at_rank_rows(i * _spr) for i in range(1, 51)]      # 1..50 epochs of the speech plan (page: epoch lines)
if marks: doc["ppm"]["marks"] = marks
if S811 and not LITE: doc["gbp_ref"] = gb(smax=S0_811)      # 10-21: PPM main's GenghisBench sets up to the switch (the 8,1,1 page's dimmed reference)
if S811 and not LITE:                                       # 10-27 pose5: npl5 rolling fetchers' lead per host (pose5/kit/hold_chain.sh writes it; fresh <= 15 min)
    try:
        _ra = "/data/home/amartyavarma/ppm/pose5/kit/state/hold_chain/roll_ahead.json"
        if time.time() - os.path.getmtime(_ra) < 900:
            _rj = json.load(open(_ra))
            doc["roll_ahead"] = {h_: r_.get("ahead") for h_, r_ in _rj["hosts"].items()}
            doc["roll_disk"] = _rj.get("disk_warn") or {}      # hosts whose /scratch is near / trending to their floor (hold_chain.sh)
    except Exception:
        pass
json.dump(doc, open(OUT, "w"), separators=(",", ":"))
print(len(pts), "evals", len(tr), "steps", doc["jobs"], "epoch %.4f" % doc["epoch"])
