"""npl5 OFF-HOST DRY RUN (pose5 workstream B, 10-27): what code-811/fresh/append/append_dryrun.py host/reduce/guard check, done on a spare
for the resume from dcp-17850 with plans [plan 1, npl2, npl3, npl5] (append_dryrun.py itself assumes resume == the end of the earlier
plans and keys plans by the local hostname, so it cannot run here unchanged; this file reuses the trainer's own code the same way):
  ends    <OUT>        row counts of all 4 plans per plan host (audit copies; npl5 = W/scan/plans) -> fresh_pose.append_ends (= the trainer's
                       all_reduce MIN boundaries), tail 1: ends, max_steps, rows never repeated per host over the WHOLE fresh run, steps
                       < 18022 unchanged vs the 3-plan run (27653's stream), where every rank is at 17850 and where it enters npl5, flags vs
                       STREAM_LIMITS.json; npl5 plan files == their audit copies (digest, n_rows) -> <OUT>/RESULT.json
  need    <OUT> <host> parquet files needed for `fresh_pose.py check` (rows 0 / mid / last of every slot) and the micros below -> a src_fetch
                       work dir <OUT>/need (placement/<host>.tsv; fetched to the canonical /scratch/amartyavarma/npl-append/npl5 on the spare)
  micros  <OUT>        every slot's 8 ranks: the trainer's micro (FreshPoseMultiDS -> data_ppm.pose_micro_ppm -> packed) for its first npl5
                       row and its last step -> <OUT>/<slot>.json   (the resume-step micro reads npl3 parquet: on-host on launch day)
  guard   <OUT>        (ENV sourced by the caller, PPM_MAX_STEPS = RESULT max_steps) import the trainer, s811_init on dcp-17850's real
                       train_state.json: npl5 added at 17850 <= 18022, saved digests == the loaded prefix, max_steps -> <OUT>/GUARD.json
usage: qwen-s2 venv, python -I (guard: plain python, the trainer's imports)."""
import glob
import json
import os
import sys
import time

import numpy as np

W = os.environ.get("NPL5_W", "/data/home/amartyavarma/ppm/pose5/npl5")   # (NPL5_W: a test work dir)
C = os.environ.get("CODE811", "/data/home/amartyavarma/ppm/code-811")
A = "/data/home/amartyavarma/ppm/code-811/fresh/append"
SF, NPM, LW, RESUME = 16861, 8, 8, 17850
ENDS3 = [17352, 17641, 18022]                                     # job 27114 / 27204 / 27653 plan ends (27653 paused at 17850)
TAGS = ("npl", "npl2", "npl3", "npl5")
LOGICAL = {"slinky-7": "slinky-10"}                              # plan key -> 811 logical slot (plan key slinky-9 = slot slinky-9, on slinky-15)
TS17850 = os.environ.get("TS17850", "/data/home/amartyavarma/ppm/pose5/npl5/dcp17850_ts")   # (copies of the HF dcp-17850 train_state.json, 16 hosts)
NPL5_DIR = "/scratch/amartyavarma/ppm-fresh-pose-npl5"


def fp():
    sys.path.insert(0, C)
    import data_ppm  # noqa: F401,E402  (the trainer's speech_q first, as train_ppm does)
    sp_ = list(sys.path); sys.path.insert(0, f"{C}/fresh")
    try:
        import fresh_pose as FP_  # noqa: E402
    finally:
        sys.path[:] = sp_
    return FP_


def audits(tag):
    d = {"npl": f"{A}/../scan/plans", "npl5": f"{W}/scan/plans"}.get(tag, f"{A}/{tag}/scan/plans")
    out = {json.load(open(f))["host"]: json.load(open(f)) for f in glob.glob(f"{d}/PLAN-*.json")}
    assert len(out) == 16, ("audit copies", tag, d, sorted(out))
    return out


def npl5_plan(FP, h):
    return FP.load_plan(f"{W}/plans/{h}/ppm-fresh-pose-npl5")


def cmd_ends(out):
    FP = fp()
    au = [audits(t) for t in TAGS]
    hosts = sorted(au[0], key=lambda h: int(h.split("-")[1]))
    assert hosts == sorted(FP.HOSTS, key=lambda h: int(h.split("-")[1])) and all(set(a) == set(hosts) for a in au)
    P = json.load(open(f"{W}/plans/PLANS.json"))["slots"]
    for h in hosts:                                                # the built plan files == their audit copies
        assert P[h]["digest"] == au[3][h]["digest"] and P[h]["n_rows"] == au[3][h]["n_rows"], h
    ts = json.load(open(f"{TS17850}/slinky-0/train_state.json"))
    dg = ts["cfg"]["fresh"]["pose_digests"]
    assert set(dg) == set(hosts) and all(dg[h] == [au[k][h]["digest"] for k in range(3)] for h in hosts), "dcp-17850 digests != plan 1-3 audits"
    nrows = {h: [a[h]["n_rows"] for a in au] for h in hosts}
    ends, st = FP.append_ends(nrows, LW, NPM, SF, 1)
    ends3, st3 = FP.append_ends({h: v[:3] for h, v in nrows.items()}, LW, NPM, SF, 1)
    assert ends3 == ENDS3 and ends[:3] == ENDS3, (ends3, ends)
    end = ends[-1]
    E3 = (ENDS3[2] - SF) * NPM
    pos = []
    for (h, l), s in st.items():
        assert np.array_equal(s[:E3], st3[(h, l)][:E3]), (h, l, "steps < 18022 differ from the 3-plan stream")
        p0 = (RESUME - SF) * NPM
        k0, r0 = FP.split_row(int(s[p0]))
        i5 = int(np.flatnonzero((s >> FP.PLAN_SHIFT) == 3)[0])
        kl, rl = FP.split_row(int(s[(end - SF) * NPM - 1]))
        pos.append({"host": h, "local": l, "resume_plan": k0 + 1, "resume_row": r0, "npl5_first_pos": i5, "npl5_first_step": SF + i5 // NPM,
                    "last_plan": kl + 1, "last_row": rl, "npl3_rows_left_at_17850": int(((s[p0:] >> FP.PLAN_SHIFT) == 2).sum())})
    uniq, unused = {}, {}
    for h in hosts:
        u = np.concatenate([st[(h, l)][:(end - SF) * NPM] for l in range(LW)])
        uniq[h] = bool(len(np.unique(u)) == len(u)); unused[h] = int(sum(nrows[h])) - int(len(u))
        assert uniq[h], (h, "a fresh pose row repeats")
        n3u = int(((u >> FP.PLAN_SHIFT) == 2).sum())
        assert n3u == nrows[h][2] or os.environ.get("STRICT_NPL3") == "0", (h, "tail 1: every npl3 row used once", n3u, nrows[h][2])
    lim = json.load(open(f"{A}/STREAM_LIMITS.json"))
    flags = []
    if end > lim["gemma"]["run_end_step_index_min_over_ranks"]:
        flags.append(f"GEMMA RUNS OUT at {lim['gemma']['run_end_step_index_min_over_ranks']} < pose end {end}: the trainer would stop there")
    if end > lim["chat"]["second_pass_from_step_index_min_over_ranks"]:
        flags.append(f"chat re-reads (2nd pass) from {lim['chat']['second_pass_from_step_index_min_over_ranks']}")
    if end > lim["speech"]["epoch2_from_step_index_min_over_ranks"]:
        flags.append(f"speech enters epoch 2 (flipped repeats) from {lim['speech']['epoch2_from_step_index_min_over_ranks']}")
    hours = [round(sum(a[h]["kept_frames"] for h in hosts) / 12.5 / 3600) for a in au]
    res = {"max_steps": end, "ends": ends, "resume_step": RESUME, "steps_after_resume": end - RESUME, "tail": 1,
           "rows_per_plan": [sum(nrows[h][k] for h in hosts) for k in range(4)], "hours_per_plan": hours, "rows_per_host": nrows,
           "rows_unused_all_hosts": sum(unused.values()), "rows_unused_per_host": unused, "rows_unique_per_host": uniq,
           "resume_in_plan": sorted({p["resume_plan"] for p in pos}), "npl3_rows_left_at_17850_min_max": [min(p["npl3_rows_left_at_17850"] for p in pos), max(p["npl3_rows_left_at_17850"] for p in pos)],
           "npl5_first_step_min_max": [min(p["npl5_first_step"] for p in pos), max(p["npl5_first_step"] for p in pos)],
           "last_step_plan": sorted({p["last_plan"] for p in pos}), "history_lt_18022_equal_3plan_stream": True,
           "digests": {h: [a[h]["digest"] for a in au] for h in hosts}, "positions": pos, "flags": flags,
           "global_dedup_npl5": au[3][hosts[0]]["global"], "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(res, open(f"{out}/RESULT.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("positions", "digests", "rows_per_host", "global_dedup_npl5")}), flush=True)


def need_rows(FP, h, res):
    p = npl5_plan(FP, h)
    n = int(p["n_rows"])
    rows = {"check": sorted({0, n // 2, n - 1})}
    rows["micro"] = sorted(set(range(LW)) | {q["last_row"] for q in res["positions"] if q["host"] == h and q["last_plan"] == 4})
    return p, rows


def cmd_need(out, host):
    FP = fp()
    res = json.load(open(f"{out}/RESULT.json"))
    lst = {}
    for ln in open(f"{W}/listing.tsv"):
        r = ln.rstrip("\n").split("\t"); lst[r[0]] = (r[1], r[2])
    files, per = set(), {}
    for h in sorted(FP.HOSTS):
        p, rows = need_rows(FP, h, res)
        z = p["z"]
        fs = set()
        for r in sorted(set(rows["check"]) | set(rows["micro"])):
            for j in z["row_seg"][z["row_off"][r]:z["row_off"][r + 1]]:
                fs.add(p["files"][int(z["seg_file"][j])])
        per[h] = {"rows": rows, "files": len(fs)}; files |= fs
    nd = f"{out}/need"
    os.makedirs(f"{nd}/placement", exist_ok=True)
    json.dump(json.load(open(f"{W}/LISTING.json")), open(f"{nd}/LISTING.json", "w"))
    with open(f"{nd}/placement/{host}.tsv", "w") as f:
        for x in sorted(files):
            s, xh = lst[f"jobs/{x}"]
            f.write(f"jobs/{x}\t{s}\t{xh}\n")
    json.dump(per, open(f"{nd}/NEED.json", "w"), indent=1)
    print(f"need: {len(files)} parquet files for {len(per)} slots", flush=True)


def cmd_micros(out):
    FP = fp()
    import data_ppm as D
    res = json.load(open(f"{out}/RESULT.json"))
    au = [audits(t) for t in TAGS[:3]]
    cfg = D.default_cfg(pose_delay=2)
    order = sorted((LOGICAL.get(h, h) for h in FP.HOSTS), key=lambda h: int(h.split("-")[1]))
    for h in sorted(FP.HOSTS):
        t0 = time.time()
        p5 = npl5_plan(FP, h)
        plans = [au[0][h], au[1][h], au[2][h], p5]                  # plans 1-3: audit copies (only their tags are read: no npl3 row here)
        r = {"host": h, "logical": LOGICAL.get(h, h), "micros": []}
        for q in [q for q in res["positions"] if q["host"] == h]:
            l = q["local"]; rank = 8 * order.index(LOGICAL.get(h, h)) + l
            ids = [(3 << FP.PLAN_SHIFT) | l]                        # the rank's first npl5 row (rank_rows starts at row l)
            nm = ["npl5_first"]
            if q["last_plan"] == 4:
                ids.append((3 << FP.PLAN_SHIFT) | q["last_row"]); nm.append("last")
            ds = FP.FreshPoseMultiDS(plans, ids, cfg, rank)
            for j, w in enumerate(nm):
                t1 = time.time(); m = ds[j]
                N = len(m["pid"])
                assert m["ids"].shape == (2, N) and m["pdepth"].shape == (N, 32) and (m["present"][1] == 1).all()
                r["micros"].append({"local": l, "rank": rank, "what": w, "row": FP.split_row(ids[j])[1], "N": int(N), "n_text": m["n_text"],
                                    "pcb0": int((m["pcb0"] >= 0).sum()), "s": round(time.time() - t1, 3)})
        r["seconds"] = round(time.time() - t0, 1)
        json.dump(r, open(f"{out}/{h}.json", "w"), indent=1)
        print(json.dumps({k: v for k, v in r.items() if k != "micros"}), len(r["micros"]), "micros", flush=True)


def cmd_guard(out):
    import importlib.util
    import shutil
    res = json.load(open(f"{out}/RESULT.json"))
    tp = os.environ.get("TRAIN_PY", f"{C}/train_ppm.py")
    spec = importlib.util.spec_from_file_location("train_ppm", tp)
    T = importlib.util.module_from_spec(spec); sys.modules["train_ppm"] = T; spec.loader.exec_module(T)
    assert T.CFG["max_steps"] == res["max_steps"], (T.CFG["max_steps"], res["max_steps"])
    want2 = ["/scratch/amartyavarma/ppm-fresh-pose-npl2", "/scratch/amartyavarma/ppm-fresh-pose-npl3", NPL5_DIR]
    assert list(T.FRESH_POSE2) == want2, T.FRESH_POSE2
    ts = sorted(glob.glob(f"{TS17850}/slinky-*/train_state.json"))
    st = {json.load(open(f))["step"] for f in ts}
    same = len({json.dumps(json.load(open(f))["cfg"]["fresh"], sort_keys=True) for f in ts}) == 1
    assert len(ts) == 16 and st == {RESUME} and same, ("dcp-17850 train_state copies", len(ts), st, same)
    d = f"{out}/guard-ckpt/dcp-{RESUME}"
    os.makedirs(d, exist_ok=True); shutil.copyfile(ts[0], f"{d}/train_state.json")
    T.s811_init(d)
    ap = T.CFG["fresh"]["pose_append_at"]
    assert ap == [17352, 17641, RESUME], ap
    assert all(ap[k] <= res["ends"][k] for k in range(3)), (ap, res["ends"])
    sv = T.FRESH_SAVED_DIG[0]
    H = res["digests"]
    assert sv is not None and set(sv) == set(H) and all(H[h][:len(sv[h])] == list(sv[h]) for h in H), ("saved digests != loaded prefix", sv)
    assert T.CFG["s811"]["from_step"] == SF, T.CFG["s811"]
    fo, fn = json.load(open(ts[0]))["cfg"]["fresh"], {k: v for k, v in T.CFG["fresh"].items() if k not in T.FRESH_APPEND_LATE}
    neg = {}
    for nm, f2 in (("npl3 dropped", dict(fn, pose_root2=[want2[0], want2[2]])), ("reordered", dict(fn, pose_root2=list(reversed(want2)))),
                   ("tail 0", dict(fn, pose_tail=0)), ("other key", dict(fn, speech_flip_from=SF + 1))):
        try:
            T.fresh_append_guard(fo, f2, RESUME); neg[nm] = "NOT refused"
        except AssertionError:
            neg[nm] = "refused"
    assert all(v == "refused" for v in neg.values()), neg
    g = {"guard": "ok", "refusals": neg, "train_py": tp, "resume_ckpt_step": RESUME, "pose_append_at": ap, "fresh_block": T.CFG["fresh"],
         "saved_digests_prefix_ok": True, "s811_from_step": T.CFG["s811"]["from_step"], "max_steps": T.CFG["max_steps"],
         "host_alias": getattr(T, "HOST_ALIAS", None), "s2p": T.CFG.get("s2p", T.CFG.get("tower_p_s2p", "n/a"))}
    json.dump(g, open(f"{out}/GUARD{os.environ.get('GUARD_SUFFIX', '')}.json", "w"), indent=1, default=str)
    print(json.dumps(g, default=str)[:3000], flush=True)


PLAN_DIRS = ("ppm-fresh-pose", "ppm-fresh-pose-npl2", "ppm-fresh-pose-npl3", "ppm-fresh-pose-npl5")
LISTINGS = {1: f"{A}/npl2/listing.tsv", 2: f"{A}/npl3/listing.tsv"}     # (plan-1 files: data/new-pose-llm-data-dl; never needed after 17850)


def cmd_need3(out, root, key, phys):
    """LAUNCH DAY (or off-host on a copy): the parquet files of plans 1-3 the slot's 8 ranks still read from the resume step 17850 on
    (stream positions >= (17850 - 16861) x 8 up to the end of the rank's plan 1-3 rows), from the 3 plan dirs under <root>; missing / wrong-size
    files at their src_root -> <out>/need3-<key>/placement/<phys>.tsv (+ LISTING.json of that bucket) for src_fetch. -> <out>/need3-<key>.json"""
    FP = fp()
    pl = [FP.load_plan(f"{root}/{d}") for d in PLAN_DIRS[:3]]
    assert all(p["host"] == key for p in pl), ([p["host"] for p in pl], key)
    p0 = (RESUME - SF) * NPM
    need = [set() for _ in pl]
    for l in range(LW):
        s = FP.append_stream([int(p["n_rows"]) for p in pl], l, LW, NPM, SF, 1, ())
        for x in s[p0:]:
            k, r = FP.split_row(int(x)); z = pl[k]["z"]
            for j in z["row_seg"][z["row_off"][r]:z["row_off"][r + 1]]:
                need[k].add(pl[k]["files"][int(z["seg_file"][j])])
    res = {"key": key, "phys": phys, "root": root, "files_per_plan": [len(x) for x in need], "missing": {}, "bytes_needed": 0}
    for k in (1, 2):
        if not need[k]:
            continue
        lst = {}
        for ln in open(LISTINGS[k]):
            r = ln.rstrip("\n").split("\t"); lst[r[0]] = (int(r[1]), r[2])
        miss = []
        for f in sorted(need[k]):
            sz, xh = lst[f"jobs/{f}"]
            res["bytes_needed"] += sz
            q = f"{pl[k]['src_root']}/{f}"
            if not (os.path.isfile(q) and os.path.getsize(q) == sz):
                miss.append((f"jobs/{f}", sz, xh))
        res["missing"][PLAN_DIRS[k]] = {"n": len(miss), "bytes": sum(m[1] for m in miss), "src_root": pl[k]["src_root"]}
        if miss:
            d = f"{out}/need3-{key}-{PLAN_DIRS[k]}"
            os.makedirs(f"{d}/placement", exist_ok=True)
            json.dump(json.load(open(f"{os.path.dirname(LISTINGS[k])}/LISTING.json")), open(f"{d}/LISTING.json", "w"))
            with open(f"{d}/placement/{phys}.tsv", "w") as f_:
                for m in miss:
                    f_.write(f"{m[0]}\t{m[1]}\t{m[2]}\n")
    assert not need[0], ("plan-1 rows still read after 17850: not expected", len(need[0]))
    json.dump(res, open(f"{out}/need3-{key}.json", "w"), indent=1)
    print(json.dumps(res), flush=True)


def cmd_host(out, key):
    """LAUNCH DAY on each host (after the copy): its 4 node-local plans (canonical dirs) load + digest == RESULT's for its plan key; sha256 of
    the npl5 plan files == PLANS.json; micros of its 8 ranks at the resume step (npl3 row), the first npl5 row and the last step."""
    import hashlib
    FP = fp()
    import data_ppm as D
    res = json.load(open(f"{W}/dryrun/RESULT.json")); P = json.load(open(f"{W}/plans/PLANS.json"))["slots"][key]
    pl = [FP.load_plan(f"/scratch/amartyavarma/{d}") for d in PLAN_DIRS]
    assert [p["digest"] for p in pl] == res["digests"][key] and all(p["host"] == key for p in pl), ([p["digest"] for p in pl], res["digests"][key])
    sha = {n: hashlib.sha256(open(f"/scratch/amartyavarma/{PLAN_DIRS[3]}/{n}", "rb").read()).hexdigest() for n in ("PLAN.json", "PLAN.npz")}
    assert sha == P["sha256"], (sha, P["sha256"])
    cfg = D.default_cfg(pose_delay=2)
    order = sorted((LOGICAL.get(h, h) for h in FP.HOSTS), key=lambda h: int(h.split("-")[1]))
    r = {"key": key, "phys": os.uname().nodename.split(".")[0], "digests_ok": True, "npl5_sha_ok": True, "micros": []}
    end = res["max_steps"]
    for l in range(LW):
        s = FP.append_stream([int(p["n_rows"]) for p in pl], l, LW, NPM, SF, 1, ())[:(end - SF) * NPM]
        picks = {} if os.environ.get("SKIP_RESUME_MICRO") == "1" else {"resume_first": (RESUME - SF) * NPM}
        w5 = np.flatnonzero((s >> FP.PLAN_SHIFT) == 3)
        picks.update({"npl5_first": int(w5[0])} if len(w5) else {})
        if os.environ.get("SKIP_LAST_MICRO") != "1":            # (rolling fetch: the last step's files arrive later; built off-host instead)
            picks["last"] = len(s) - 1
        if os.environ.get("SKIP_RESUME_MICRO") == "1":          # (spare rehearsal: no npl3 parquet here -> npl5 rows only)
            picks = {k_: i_ for k_, i_ in picks.items() if int(s[i_]) >> FP.PLAN_SHIFT == 3}
        ds = FP.FreshPoseMultiDS(pl, [int(s[i]) for i in picks.values()], cfg, 8 * order.index(LOGICAL.get(key, key)) + l)
        for j, (nm, i) in enumerate(picks.items()):
            t1 = time.time(); m = ds[j]; N = len(m["pid"])
            assert m["ids"].shape == (2, N) and m["pdepth"].shape == (N, 32) and (m["present"][1] == 1).all()
            k_, r_ = FP.split_row(int(s[i]))
            r["micros"].append({"local": l, "what": nm, "step_index": SF + i // NPM, "plan": k_ + 1, "row": r_, "N": int(N), "s": round(time.time() - t1, 3)})
    st = os.statvfs("/scratch"); r["scratch_free_GB"] = round(st.f_bavail * st.f_frsize / 1e9, 1)
    json.dump(r, open(f"{out}/host-{key}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k != "micros"}), len(r["micros"]), "micros", flush=True)


if __name__ == "__main__":
    cmd, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    {"ends": lambda: cmd_ends(out), "need": lambda: cmd_need(out, sys.argv[3]), "micros": lambda: cmd_micros(out),
     "guard": lambda: cmd_guard(out), "need3": lambda: cmd_need3(out, sys.argv[3], sys.argv[4], sys.argv[5]),
     "host": lambda: cmd_host(out, sys.argv[3])}[cmd]()
