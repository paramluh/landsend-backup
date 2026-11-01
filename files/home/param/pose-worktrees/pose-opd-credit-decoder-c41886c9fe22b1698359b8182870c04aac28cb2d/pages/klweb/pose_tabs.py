"""Pose-page tabs data (stage3_pose.html POSE | SPEECH | KL tabs; build_stage3.py, s3branches_pose.json "posetabs").
doc(chains) -> {"tcols", "ecols", "runs": [{id, label, jobs, train, eval}], "x": {...pose-epoch constants...}}
chains = [(id, label, run_name, [jobs])]. Per job: qwen-1ch/stage3/runs/<run_name>-j<job>/metrics.jsonl; a job's rows count only up to the step
the next job of its chain resumed from ("full resume from ...dcp-N" in logs/q1ch-s3-<next>.log), later jobs override.
Pose epoch (verified 10-10): S3_MIX_CYCLE kl,conv,pose,pose on 16 whole nodes (node k at step t runs cycle[(t - k) % 4]) -> 8 pose nodes x 8 GPUs
x POSE_ROWS 1 = 64 pose rows / step; one pose epoch = 128 GPUs x (min node rows 43,743 // 8 = 5,467) = 699,776 rows = 10,934 steps
(POSE_STEPS 10,932 = 2 x 43,743 // 8 - 2, PREFLIGHT_<job>.log)."""
import json, os, re, glob
RD = "/data/home/amartyavarma/qwen-1ch/stage3/runs"; LD = "/data/home/amartyavarma/qwen-1ch/stage3/logs"
PF = "/data/home/amartyavarma/qwen-1ch-data/pose-s3/PREFLIGHT_{}.log"
CB = (8, 16, 31)
T = {"pose_cb0": "train_pose/pose_cb0_ce", **{f"pose_cb{c}": f"train_pose/pose_light_ce_cb{c}" for c in CB}, "sp_cb0": "train_pose/cb0_ce",
     "conv_nword": "train_conv/word_ce_norm", "conv_cb0": "train_conv/cb0_ce",
     "kl": "train/kl_word_invariant", "kl_noeos": "train/kl_word_invariant_noeos", "top1": "train/top_1_word_agree_invariant", "eos_kl": "train/eos_kl"}
E = {"pose_cb0": "eval_pose/pose_cb0_ce", **{f"pose_cb{c}": f"eval_pose/pose_light_ce_cb{c}" for c in CB}, "sp_cb0": "eval_pose/cb0_ce",
     **{f"sp_cb{c}": f"eval_pose/light_ce_cb{c}" for c in CB}, "conv_nword": "eval_conv/word_ce_norm", "conv_cb0": "eval_conv/cb0_ce",
     **{f"conv_cb{c}": f"eval_conv/light_ce_cb{c}" for c in CB},
     "kl": "eval/kl_word_invariant", "kl_noeos": "eval/kl_word_invariant_noeos", "top1": "eval/top_1_word_agree_invariant", "eos_kl": "eval/eos_kl"}
TC, EC = ["step"] + list(T), ["step"] + list(E)
r5 = lambda v: None if v is None else round(float(v), 5)
def resume_step(job):
    try:
        for line in open(f"{LD}/q1ch-s3-{job}.log", errors="replace"):
            m = re.search(r"full resume from \S*?dcp-(\d+)", line)
            if m: return int(m[1])
    except Exception: pass
    return None
def chain(run, jobs):
    tr, ev = {}, {}
    for i, j in enumerate(jobs):
        cut = resume_step(jobs[i + 1]) if i + 1 < len(jobs) else None
        cut = 10 ** 9 if cut is None else cut
        for mf in glob.glob(f"{RD}/{run}-j{j}/metrics.jsonl")[:1]:
            for line in open(mf, errors="replace"):
                try: d = json.loads(line)
                except Exception: continue
                s = d.get("step")
                if s is None or int(s) > cut: continue
                s = int(s)
                if any(d.get(k) is not None for k in T.values()): tr[s] = [s] + [r5(d.get(k)) for k in T.values()]
                if any(d.get(k) is not None for k in E.values()): ev[s] = [s] + [r5(d.get(k)) for k in E.values()]
    return [tr[s] for s in sorted(tr)], [ev[s] for s in sorted(ev)]
def _cfg_get(cf, *names):
    for n in names:
        for k in (n, n.lower(), n.upper(), "s3_" + n.lower()):
            if cf.get(k) not in (None, ""): return cf[k]
    return None
def _as_list(v): return [str(x).strip().lower() for x in (v.split(",") if isinstance(v, str) else v) if str(x).strip()]
def pose_epoch(job):
    """({pose_rows_per_step, pose_epoch_rows, mix}, None) from cfg-<job>.json + a PREFLIGHT_<job>.log, or (None, reason).
    pose rows / step by the run's MIX MODE:
      - mix_mode absent / "nodes" (rotating whole nodes, S3_MIX_CYCLE e.g. kl,conv,pose,pose): world x pose share of the cycle x pose_rows
        (node k at step t runs cycle[(t - k) % P]: 8 of 16 nodes on pose -> 64 rows);
      - mix_mode "ga" (each optimiser step = one micro per cycle entry, ALL GPUs on the same task per micro, e.g. pose,pose,conv,kl):
        world x (pose micros per step) x pose_rows (-> 256 rows at 128 GPUs); the micro list must be in the cfg (ga_cycle / mix_cycle / ga_micros);
      - any other mix_mode, or ga without a readable micro list -> None (the page shows the red fallback line).
    pose epoch rows = world x (smallest node's train rows // 8): every GPU reads its own node's store (pre-flight "min pose train rows per node")."""
    try: cf = json.load(open(f"{RD}/cfg-{job}.json"))
    except Exception: return None, f"no cfg-{job}.json"
    try:
        world, pr = int(cf["world"]), int(cf.get("pose_rows", 1) or 1)
        mode = str(_cfg_get(cf, "mix_mode") or "nodes").strip().lower()
        # 10-10: MIX_MODE=ga runs (train_pose_s3q.ga*.py) are not recorded in cfg-<job>.json -> read the trainer's own "[ga] MIX_MODE=ga: N micros
        # per optimizer step ('pose', ..., 'conv', 'kl')" log line (5:1:1 -> 640 rows / step, 3:1:1 -> 384 at 128 GPUs)
        try:
            for line in open(f"/data/home/amartyavarma/qwen-1ch/stage3/logs/q1ch-s3-{job}.log", errors="replace"):
                m_ = re.search(r"\[ga\] MIX_MODE=ga: \d+ micros per optimizer step \(([^)]*)\)", line)
                if m_:
                    mic_ = re.findall(r"'(\w+)'", m_[1]); mode = "ga"; cf = dict(cf, ga_cycle=",".join(mic_)); break
        except Exception:
            pass
        if mode in ("nodes", "node", "rotate", "rotating"):
            cyc = _as_list(_cfg_get(cf, "mix_cycle") or "kl,conv,pose,pose")
            if not cyc or "pose" not in cyc: return None, f"mix cycle {cyc} has no pose"
            rps = world * cyc.count("pose") / len(cyc) * pr; mix = "rotating nodes " + ",".join(cyc)
        elif mode == "ga":
            mic = _cfg_get(cf, "ga_cycle", "ga_micros", "mix_cycle", "micro_cycle")
            mic = _as_list(mic) if mic is not None else []
            if not mic or "pose" not in mic: return None, f"mix_mode ga but no readable micro list with pose in cfg-{job}.json"
            rps = world * mic.count("pose") * pr; mix = "ga micros " + ",".join(mic)
        else: return None, f"unknown mix_mode {mode!r} in cfg-{job}.json"
        pf = ([PF.format(job)] if os.path.exists(PF.format(job)) else []) + sorted(glob.glob(f"/data/home/amartyavarma/qwen-1ch-data/*/PREFLIGHT_{job}.log"))
        if not pf: return None, f"no PREFLIGHT_{job}.log"
        minrows = None
        for line in open(pf[0], errors="replace"):        # any pose store's pre-flight log for this job (pose-s3, pose-s3-samdata, ...)
            m = re.search(r"min pose train rows per node \(READY files\) (\d+)", line)
            if m: minrows = int(m[1])
        if not minrows: return None, f"no 'min pose train rows per node' line in {os.path.basename(pf[0])}"
        return {"pose_rows_per_step": int(round(rps)), "pose_epoch_rows": world * (minrows // 8), "mix": mix}, None
    except Exception as e: return None, f"cfg-{job}.json: {type(e).__name__}: {e}"
COUNTS = "/data/home/amartyavarma/qwen-1ch-data/pose-s3/counts.json"
def store_counts(path):
    """(clip-hours per train row, train rows, train clip-hours, dataset name) from a pose store's counts.json; None if unreadable."""
    try:
        d = json.load(open(path)); c = d["train"]; return c["clip_hours"] / c["rows"], c["rows"], c["clip_hours"], d.get("dataset")
    except Exception: return None
def counts_for(jobs):
    """the run's own store: cfg-<job>.json pose_data_root /scratch/amartyavarma/<store>/rows -> qwen-1ch-data/<store>/counts.json (None if unknown)."""
    for j in jobs:
        try:
            m = re.match(r"/scratch/amartyavarma/([^/]+)/rows/?$", json.load(open(f"{RD}/cfg-{j}.json")).get("pose_data_root", ""))
            if m: return f"/data/home/amartyavarma/qwen-1ch-data/{m[1]}/counts.json"
        except Exception: pass
    return None
# ---- 10-12 PPM runs (run name "ppm-*", the PPM page S3_PAGE=ppm only): metrics in ppm/runs/<run>-j<job>/metrics.jsonl with PPM key names,
# mapped onto the columns above; PPM-only extra columns (XT / XE) are appended ONLY when a doc() call contains a PPM run, so the pose page is unchanged.
PPM_RD, PPM_LD = "/data/home/amartyavarma/ppm/runs", "/data/home/amartyavarma/ppm/logs"
PPM_T = {"pose_cb0": "train_pose/pose_cb0_ce", "sp_cb0": "train_pose/cb0_ce", "conv_nword": "train_speech/word_ce_norm", "conv_cb0": "train_speech/cb0_ce"}
PPM_T.update({f"pose_cb{c}": f"train_pose/pose_light_ce_cb{c}" for c in CB}); PPM_T.update({f"sp_cb{c}": f"train_pose/light_ce_cb{c}" for c in CB})   # 10-12 per-codebook keys (from job 18906)
PPM_E = {"pose_cb0": "eval_pose/pose_cb0_ce", "sp_cb0": "eval_pose/cb0_ce", "conv_nword": "eval_speech/word_ce_norm", "conv_cb0": "eval_speech/cb0_ce"}
PPM_E.update({f"pose_cb{c}": f"eval_pose/pose_light_ce_cb{c}" for c in CB}); PPM_E.update({f"sp_cb{c}": f"eval_pose/light_ce_cb{c}" for c in CB})
XT = {"pose_cb1": "train_pose/pose_light_ce_cb1", "pose_light": "train_pose/pose_light_ce", "sp_light": "train_pose/light_ce", "pose_nword": "train_pose/word_ce_norm", "conv_light": "train_speech/light_ce",
      "gm_nword": "train_gemma/word_ce_norm", "ch_nword": "train_chat/word_ce_norm"}
XE = {"pose_cb1": "eval_pose/pose_light_ce_cb1", "pose_light": "eval_pose/pose_light_ce", "sp_light": "eval_pose/light_ce", "pose_nword": "eval_pose/word_ce_norm", "conv_light": "eval_speech/light_ce",
      "gm_nword": "eval_gemma/word_ce_norm", "ch_nword": "eval_chat/word_ce_norm"}
# from the D2 resume (10-12) PPM also logs per-codebook light CE (stage-3 key names) -> the cb columns
PPM_T.update({f"pose_cb{c}": f"train_pose/pose_light_ce_cb{c}" for c in (1,) + CB})
PPM_E.update({**{f"pose_cb{c}": f"eval_pose/pose_light_ce_cb{c}" for c in (1,) + CB}, **{f"sp_cb{c}": f"eval_pose/light_ce_cb{c}" for c in CB},
              **{f"conv_cb{c}": f"eval_speech/light_ce_cb{c}" for c in CB}})
XT["pose_cb1"] = "train_pose/pose_light_ce_cb1"; XE["pose_cb1"] = "eval_pose/pose_light_ce_cb1"
# D32 (10-12): PPM's auxiliary direct backbone pose cb0 head (same positions / targets as the pose tower's cb0); PPM docs only
XT.update({"pose_head_cb0": "train_pose/pose_head_cb0_ce", "pose_head_cb0_top1": "train_pose/pose_head_cb0_top1"})
XE.update({"pose_head_cb0": "eval_pose/pose_head_cb0_ce", "pose_head_cb0_top1": "eval_pose/pose_head_cb0_top1"})
# D33 (10-12): the pose decoder's two input paths, eval-only probes on the same held-out pose rows / frames as eval_pose/pose_light_*:
# tower-only (backbone link term zeroed) and backbone-only (tower term zeroed); cb1-31 mean + cb 1/8/16/31; PPM docs only
for _v, _c in (("toweronly", "decp_tow"), ("bbonly", "decp_bbo")):
    XE[f"{_c}_light"] = f"eval_pose/decp_{_v}_light_ce"
    XE.update({f"{_c}_cb{c}": f"eval_pose/decp_{_v}_light_ce_cb{c}" for c in (1, 8, 16, 31)})
XK_T = {"pose_cb1": "train_pose/pose_light_ce_cb1", "pose_light": "train_pose/pose_light_ce", "sp_light": "train_pose/light_ce", "pose_nword": "train_pose_B/word_ce_norm", "conv_light": "train_conv/light_ce"}     # the same extras for stage-3 pose runs
XK_E = {"pose_cb1": "eval_pose/pose_light_ce_cb1", "pose_light": "eval_pose/pose_light_ce", "sp_light": "eval_pose/light_ce", "pose_nword": "eval_pose_B/word_ce_norm", "conv_light": "eval_conv/light_ce"}
# 10-12 codependence probe (pose rows): PPM (train_ppm.py, from job 18910's successor) logs the pose trainer's codependence/* keys verbatim,
# so the ★ PPM and the pose run share the cd_<key> columns (s3_doc.py CODEP list); eval-only, both runs read the same key names
CODEP = ["pose_to_speech/d_word_ce_norm_pose_shuf", "pose_to_speech/d_word_ce_norm_pose_zero", "pose_to_speech/d_cb0_ce_pose_shuf", "pose_to_speech/d_cb0_ce_pose_zero",
         "pose_to_speech/d_light_ce_pose_shuf", "pose_to_speech/d_light_ce_pose_zero", "speech_to_pose/d_pose_cb0_ce_speech_shuf", "speech_to_pose/d_pose_cb0_ce_speech_sil",
         "speech_to_pose/d_pose_light_ce_speech_shuf", "speech_to_pose/d_pose_light_ce_speech_sil", "timing/r_pnw_real_motion_real_input", "timing/r_pnw_real_motion_pose_shuf_input"]
XE.update({f"cd_{k}": f"codependence/{k}" for k in CODEP}); XK_E.update({f"cd_{k}": f"codependence/{k}" for k in CODEP})
def ppm_resume(job):
    try:
        i = 0
        for line in open(f"{PPM_LD}/ppm-{job}.log", errors="replace"):
            if re.match(r"^\[[WEI]\d{4} \d\d:\d\d:\d\d", line): continue      # 10-20 glitchfix: 128 per-rank torch warnings precede "[resume]" from job 25627 (line 138)
            m = re.search(r"\[resume\] \S*/dcp-(\d+): step (\d+)", line)
            if m: return int(m[2])
            i += 1
            if i > 80: break
    except Exception: pass
    return None
def ppm_chain(run, jobs, tc, ec):
    """PPM chain rows in the columns tc / ec (names -> PPM keys via PPM_T + XT / PPM_E + XE; columns PPM does not log stay None)."""
    tk = {**PPM_T, **XT}; ek = {**PPM_E, **XE}; tr, ev = {}, {}
    for i, j in enumerate(jobs):
        cut = ppm_resume(jobs[i + 1]) if i + 1 < len(jobs) else None
        cut = 10 ** 9 if cut is None else cut
        for mf in glob.glob(f"{PPM_RD}/{run}-j{j}/metrics.jsonl")[:1]:
            for line in open(mf, errors="replace"):
                try: d = json.loads(line)
                except Exception: continue
                s = d.get("step")
                if s is None or int(s) > cut: continue
                s = int(s)
                if "train/step_time" in d: tr[s] = [s] + [r5(d.get(tk[c])) if c in tk else None for c in tc[1:]]
                if any(k.startswith("eval_") for k in d): ev[s] = [s] + [r5(d.get(ek[c])) if c in ek else None for c in ec[1:]]
    return [tr[s] for s in sorted(tr)], [ev[s] for s in sorted(ev)]
def ppm_x(jobs):
    """PPM pose x: GA list + world from the log's 'world N (lw 8); GA (...)' line; pose epoch = world x pose train rows/rank ('[data]' line);
    clip-hours per row from the pose store's counts.json (cfg pose_root /scratch/amartyavarma/<store>/rows)."""
    for j in jobs:
        try:
            world = ga = prr = store = None
            i = -1
            for line in open(f"{PPM_LD}/ppm-{j}.log", errors="replace"):
                if re.match(r"^\[[WEI]\d{4} \d\d:\d\d:\d\d", line): continue      # 10-20 glitchfix: per-rank torch warnings (128 lines from job 25627) do not count
                i += 1
                m = re.search(r"world (\d+) \(lw \d+\); GA \(([^)]*)\)", line)
                if m: world, ga = int(m[1]), re.findall(r"'(\w+)'", m[2])
                m = re.search(r'"pose_root": "/scratch/amartyavarma/([^/"]+)/rows', line)
                if m: store = m[1]
                m = re.search(r"pose node \S+: train rows/rank ([\d,]+)", line)
                if m: prr = int(m[1].replace(",", "")); break
                if i > 80: break
            if world and ga and prr:
                return {"pose_rows_per_step": world * ga.count("pose"), "pose_epoch_rows": world * prr, "mix": "PPM GA " + ",".join(ga) + f" x {world} ranks x 1 pose row"}, \
                       (f"/data/home/amartyavarma/qwen-1ch-data/{store}/counts.json" if store else None)
        except Exception: pass
    return None, None
def doc(chains):
    """chains = [(id, label, run_name, jobs, tabs or None, counts.json path or None)]. Each run gets its OWN x constants (pose rows / step and
    rows / pose epoch from its first job with a cfg + PREFLIGHT log; clip-hours / row from its counts.json), so a restart on another data store
    (new run name in s3branches_<page>.json main_run) is placed on its own data. x.derived False = fallback constants (the page says so)."""
    runs = []
    isp = lambda run: str(run or "").startswith("ppm-")
    has_ppm = any(isp(c[2]) for c in chains)
    tc, ec = (TC + list(XT), EC + list(XE)) if has_ppm else (TC, EC)
    for cid, label, run, jobs, tabs, counts in chains:
        jobs = [str(j) for j in jobs if j]
        if not jobs: continue
        if isp(run):
            tr, ev = ppm_chain(run, jobs, tc, ec); xe, cpp = ppm_x(jobs); why = None if xe else "PPM log has no world / GA / pose rows line"
            cp = counts or cpp or COUNTS; sc = store_counts(cp)
        else:
            tr, ev = chain(run, jobs)
            if has_ppm:      # PPM page: the stage-3 pose run gets the PPM extra columns too (light means, pose-row word CE)
                tr0, ev0 = tr, ev; tr, ev = [], []
                xt, xe_ = {}, {}
                for j in jobs:
                    for mf in glob.glob(f"{RD}/{run}-j{j}/metrics.jsonl")[:1]:
                        for line in open(mf, errors="replace"):
                            try: d = json.loads(line)
                            except Exception: continue
                            if d.get("step") is None: continue
                            s_ = int(d["step"])
                            if any(d.get(k) is not None for k in XK_T.values()): xt[s_] = [r5(d.get(XK_T[c])) if c in XK_T else None for c in XT]
                            if any(d.get(k) is not None for k in XK_E.values()): xe_[s_] = [r5(d.get(XK_E[c])) if c in XK_E else None for c in XE]
                tr = [r + xt.get(r[0], [None] * len(XT)) for r in tr0]; ev = [r + xe_.get(r[0], [None] * len(XE)) for r in ev0]
            pe = [pose_epoch(j) for j in jobs]; xe = next((x for x, _ in pe if x), None); why = None if xe else "; ".join(r for _, r in pe if r)
            cp = counts or counts_for(jobs) or COUNTS; sc = store_counts(cp)        # explicit posetabs_counts > the run's cfg data root > the v3 store
        rps, er = (xe["pose_rows_per_step"], xe["pose_epoch_rows"]) if xe else (64, 699776)
        runs.append({"id": cid, "label": label, "run": run, "jobs": jobs, "tabs": tabs, "train": tr, "eval": ev,
                     "x": {"pose_rows_per_step": rps, "pose_epoch_rows": er, "clip_h_per_row": sc[0] if sc else 114043.3 / 703935,
                           "derived": bool(xe), "why": why, "mix": xe["mix"] if xe else None, "clip_h_derived": sc is not None, "counts": cp,
                           "store_rows": sc[1] if sc else None, "store_clip_h": sc[2] if sc else None, "dataset": sc[3] if sc else None}})
    main = next((r for r in runs if r["id"] == "main"), runs[0] if runs else None)
    return {"tcols": tc, "ecols": ec, "runs": runs, "x": dict(main["x"]) if main else {}}
