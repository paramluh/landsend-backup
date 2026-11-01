"""APPEND ingest step 7: check_resume DRY RUN of the combined fresh pose plan, on CPU, no GPUs, no process group (the trainer's own code:
fresh_pose.load_plan / append_stream / FreshPoseMultiDS, train_ppm.fresh_append_guard / s811_init).
  host   <ENV> <OUT>            (every host, qwen-s2 venv, python -I) load plan 1 + the appended plans named by ENV (digest-asserted), compare
                                their digests with the Weka audit copies, sha256 the plan files, compute EVERY rank's combined stream from the
                                audit row counts (append_ends: = the trainer's all_reduce MIN boundaries), build this host's 8 ranks' micros at
                                the resume step, the first appended-plan row and the last step; copy the resume checkpoint's train_state.json
                                -> <OUT>/<host>.json, <OUT>/ts-<host>.json
  reduce <ENV> <OUT> [--fill-env] (one host) every host's result -> ends, max_steps, row-uniqueness over the whole fresh run per host, the other
                                streams' limits (STREAM_LIMITS.json) -> <OUT>/RESULT.json; --fill-env writes PPM_MAX_STEPS into ENV (backup first)
  guard  <ENV> <OUT>            (one host, ENV sourced by the caller incl. PPM_MAX_STEPS) import the trainer under ENV, run s811_init on the
                                resume checkpoint's train_state.json (the fresh-block guard) and check append step / max_steps / plan dirs."""
import glob
import hashlib
import json
import os
import re
import shutil
import socket
import sys
import time

import numpy as np

C = "/data/home/amartyavarma/ppm/code-811"
A = f"{C}/fresh/append"
CKPT = "/scratch/amartyavarma/ppm/ckpt/ppm-v1"
END1_27114 = 17352
# 10-23 (delivery 3): RESUME_STEP = the resume checkpoint = the final save of the run over the plans BEFORE the newest one (17352: job 27114's,
# plans [1] -> + npl2; 17641: job 27204's, plans [1, npl2] -> + npl3). It must equal the end over those plans (ends[-2]).
RESUME = int(os.environ.get("RESUME_STEP", str(END1_27114)))
HOST = socket.gethostname().split(".")[0]
LOGICAL = {"slinky-7": "slinky-10"}                                    # prod env PPM_HOST_ALIAS=slinky-7:slinky-10
LOGICAL_ORDER = sorted((LOGICAL.get(h, h) for h in open(f"{C}/nodes16_s7.txt").read().strip().split(",")), key=lambda h: int(h.split("-")[1]))


def env_file(path):
    e = {}
    for ln in open(path):
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1); e[k] = v
    return e


def fp():
    sys.path.insert(0, C)
    import data_ppm  # noqa: F401,E402  (the trainer's speech_q first, as train_ppm does)
    sp_ = list(sys.path); sys.path.insert(0, f"{C}/fresh")
    try:
        import fresh_pose as FP_  # noqa: E402
    finally:
        sys.path[:] = sp_
    return FP_


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def audits(tag):
    """Weka audit copies (written by fresh_pose.py plan) of the plan with this tag: {host: PLAN json minus files / segment ids}."""
    d = f"{C}/fresh/scan/plans" if tag == "npl" else f"{A}/{tag}/scan/plans"
    out = {json.load(open(f))["host"]: json.load(open(f)) for f in glob.glob(f"{d}/PLAN-*.json")}
    assert len(out) == 16, ("audit copies of plan", tag, d, sorted(out))
    return out


def plan_dirs(E):
    return [E["PPM_FRESH_POSE_ROOT"]] + [x for x in E["PPM_FRESH_POSE_ROOT2"].split(",") if x]


def cmd_host(envp, out):
    E = env_file(envp)
    FP = fp()
    sf, tail = int(E["PPM_FRESH_POSE_FROM"]), int(E["PPM_FRESH_POSE_TAIL"])
    t0 = time.time()
    plans = [FP.load_plan(d) for d in plan_dirs(E)]                    # digest-asserted
    tags = [p.get("tag", "npl") for p in plans]
    assert all(p["host"] == HOST for p in plans) and len(set(tags)) == len(tags), ([p["host"] for p in plans], tags)
    au = [audits(t) for t in tags]
    for p, a in zip(plans, au):
        assert a[HOST]["digest"] == p["digest"] and a[HOST]["n_rows"] == p["n_rows"], ("plan != its audit copy", p.get("tag"), a[HOST]["digest"], p["digest"])
    nrows = {h: [a[h]["n_rows"] for a in au] for h in au[0]}
    ends, streams = FP.append_ends(nrows, 8, 8, sf, tail)
    assert ends[0] == END1_27114, ("plan-1 end from the audit copies != 27114's", ends[0])
    assert len(ends) >= 2 and ends[-2] == RESUME, ("RESUME_STEP != the end over the plans before the newest one", RESUME, ends)
    end = ends[-1]
    res = {"host": HOST, "tags": tags, "digests": [p["digest"] for p in plans], "n_rows": [p["n_rows"] for p in plans],
           "kept_hours": [round(p["kept_frames"] / 12.5 / 3600, 1) for p in plans], "ends": ends,
           "plan_sha256": {f"{t}/{n}": sha(f"{d}/{n}") for t, d in zip(tags, plan_dirs(E)) for n in ("PLAN.json", "PLAN.npz")},
           "global_dedup": [p["global"] for p in plans[1:]], "micros": []}
    cfg = __import__("data_ppm").default_cfg(pose_delay=2)
    # the micros the trainer will build: the first step after the resume, the first row of each appended plan, the run's last step
    for l in range(8):
        s = streams[(HOST, l)][:(end - sf) * 8]
        rank = 8 * LOGICAL_ORDER.index(LOGICAL.get(HOST, HOST)) + l      # the trainer's global rank (PPM_HOST_ALIAS slot order)
        picks = {"resume_first": (RESUME - sf) * 8, "last": len(s) - 1}
        for k in range(1, len(plans)):
            w = np.flatnonzero((s >> FP.PLAN_SHIFT) == k)
            if len(w):
                picks[f"plan{k + 1}_first"] = int(w[0])
        ds = FP.FreshPoseMultiDS(plans, [int(s[i]) for i in picks.values()], cfg, rank)
        for j, (nm, i) in enumerate(picks.items()):
            t1 = time.time(); m = ds[j]
            N = len(m["pid"])
            assert m["ids"].shape == (2, N) and m["pdepth"].shape == (N, 32) and (m["present"][1] == 1).all()
            k_, r_ = FP.split_row(int(s[i]))
            res["micros"].append({"local": l, "what": nm, "stream_index": int(i), "step_index": sf + int(i) // 8, "plan": k_ + 1, "row": r_,
                                  "N": int(N), "n_text": m["n_text"], "pcb0": int((m["pcb0"] >= 0).sum()), "s": round(time.time() - t1, 3)})
    # the run's rows per host: never repeated over the whole fresh run (27114's steps + the append)
    u = np.concatenate([streams[(HOST, l)][:(end - sf) * 8] for l in range(8)])
    res["rows_used_this_host"], res["rows_unique"] = int(len(u)), bool(len(np.unique(u)) == len(u))
    res["rows_unused_this_host"] = [int(sum(nrows[HOST])) - int(len(u))]
    # the resume checkpoint on this host
    cs = sorted(int(d.rsplit("-", 1)[1]) for d in glob.glob(f"{CKPT}/dcp-*") if re.fullmatch(r"dcp-\d+", os.path.basename(d))
                and os.path.isfile(f"{d}/COMPLETE"))
    res["ckpts"] = cs
    rs = RESUME if RESUME in cs else max(cs)
    res["resume_ckpt"], res["resume_is_final_save"] = rs, rs == RESUME
    shutil.copyfile(f"{CKPT}/dcp-{rs}/train_state.json", f"{out}/ts-{HOST}.json")
    st = os.statvfs("/scratch")
    res["scratch_free_GB"] = round(st.f_bavail * st.f_frsize / 1e9, 1)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(f"{out}/{HOST}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "micros"}), flush=True)


def cmd_reduce(envp, out, fill):
    E = env_file(envp)
    FP = fp()
    sf, tail = int(E["PPM_FRESH_POSE_FROM"]), int(E["PPM_FRESH_POSE_TAIL"])
    R = {json.load(open(f))["host"]: json.load(open(f)) for f in glob.glob(f"{out}/slinky-*.json")}
    assert len(R) == 16, ("host results", sorted(R))
    ends, streams = FP.append_ends({h: r["n_rows"] for h, r in R.items()}, 8, 8, sf, tail)
    assert all(r["ends"] == ends for r in R.values()), ("hosts disagree on the boundaries", {h: r["ends"] for h, r in R.items()})
    assert all(r["rows_unique"] for r in R.values()), "a fresh pose row repeats"
    lim = json.load(open(f"{A}/STREAM_LIMITS.json"))
    end = ends[-1]
    flags = []
    if end > lim["gemma"]["run_end_step_index_min_over_ranks"]:
        flags.append(f"GEMMA RUNS OUT at {lim['gemma']['run_end_step_index_min_over_ranks']} < pose end {end}: the trainer would stop there")
    if end > lim["chat"]["second_pass_from_step_index_min_over_ranks"]:
        flags.append(f"chat re-reads (2nd pass) from {lim['chat']['second_pass_from_step_index_min_over_ranks']}")
    if end > lim["speech"]["epoch2_from_step_index_min_over_ranks"]:
        flags.append(f"speech enters epoch 2 (flipped repeats) from {lim['speech']['epoch2_from_step_index_min_over_ranks']}")
    sum_rows = [sum(r["n_rows"][k] for r in R.values()) for k in range(len(ends))]
    hours = [round(sum(r["kept_hours"][k] for r in R.values())) for k in range(len(ends))]
    res = {"max_steps": end, "ends": ends, "steps_after_17352": end - END1_27114, "resume_step": RESUME, "steps_after_resume": end - RESUME, "tail": tail, "rows_per_plan": sum_rows, "hours_per_plan": hours,
           "rows_unused_all_hosts": sum(r["rows_unused_this_host"][0] for r in R.values()),
           "resume_ckpt": sorted({r["resume_ckpt"] for r in R.values()}), "resume_is_final_save": all(r["resume_is_final_save"] for r in R.values()),
           "scratch_free_GB": {h: r["scratch_free_GB"] for h, r in sorted(R.items())}, "micros_built": sum(len(r["micros"]) for r in R.values()),
           "global_dedup_plan2": R[sorted(R)[0]]["global_dedup"], "flags": flags, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    assert len(res["resume_ckpt"]) == 1, ("hosts hold different newest checkpoints", res["resume_ckpt"])
    json.dump(res, open(f"{out}/RESULT.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)
    if fill:
        txt = open(envp).read()
        i = txt.rindex(os.environ.get("FILL_MARKER", "10-22 FRESH POSE APPEND"))     # (10-23: the newest append block)
        head, blk = txt[:i], txt[i:]
        new = re.sub(r"(?m)^PPM_MAX_STEPS=.*$", f"PPM_MAX_STEPS={end}", blk)
        assert new.count("PPM_MAX_STEPS=") == 1, "the append block must hold exactly one PPM_MAX_STEPS line"
        if not os.path.exists(envp + ".bak-pre-fill"):
            shutil.copyfile(envp, envp + ".bak-pre-fill")
        open(envp + ".tmp", "w").write(head + new); os.replace(envp + ".tmp", envp)
        print(f"filled PPM_MAX_STEPS={end} into {envp}", flush=True)


def cmd_guard(envp, out):
    """ENV already in os.environ (sourced by ingest_more.sh with PPM_MAX_STEPS filled); rank-0 identity, CPU only."""
    import importlib.util
    res = json.load(open(f"{out}/RESULT.json"))
    spec = importlib.util.spec_from_file_location("train_ppm", f"{C}/train_ppm.py")
    T = importlib.util.module_from_spec(spec); sys.modules["train_ppm"] = T; spec.loader.exec_module(T)
    assert T.CFG["max_steps"] == res["max_steps"], (T.CFG["max_steps"], res["max_steps"])
    assert T.FRESH_POSE2 == plan_dirs(env_file(envp))[1:], (T.FRESH_POSE2, envp)
    ts = sorted(glob.glob(f"{out}/ts-*.json"))
    steps = {json.load(open(f))["step"] for f in ts}
    assert len(ts) == 16 and len(steps) == 1, ("resume train_state copies", len(ts), steps)
    d = f"{out}/guard-ckpt/dcp-{steps.pop()}"
    os.makedirs(d, exist_ok=True); shutil.copyfile(ts[0], f"{d}/train_state.json")
    T.s811_init(d)
    ap = T.CFG["fresh"]["pose_append_at"]
    assert len(ap) == len(res["ends"]) - 1 and all(ap[k] <= res["ends"][k] for k in range(len(ap))), ("appended plan added past the end of the plans before it", ap, res["ends"])
    assert ap[-1] == RESUME, ("the newest plan is not added at the resume step", ap, RESUME)
    sv = T.FRESH_SAVED_DIG[0]
    if sv is not None:                                   # the checkpoint's plan digests == the prefix of this run's (main() asserts it again)
        H = {json.load(open(f))["host"]: json.load(open(f))["digests"] for f in glob.glob(f"{out}/slinky-*.json")}
        assert set(sv) == set(H) and all(H[h][:len(sv[h])] == list(sv[h]) for h in H), ("checkpoint plan digests != the loaded plans' prefix", sv, H)
    assert T.CFG["s811"]["from_step"] == 16861, T.CFG["s811"]
    g = {"guard": "ok", "resume_ckpt_step": int(json.load(open(ts[0]))["step"]), "pose_append_at": ap, "fresh_block": T.CFG["fresh"],
         "saved_digests_prefix_ok": T.FRESH_SAVED_DIG[0] is not None,
         "s811_from_step": T.CFG["s811"]["from_step"], "max_steps": T.CFG["max_steps"]}
    json.dump(g, open(f"{out}/GUARD.json", "w"), indent=1)
    print(json.dumps(g), flush=True)


if __name__ == "__main__":
    cmd, envp, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    {"host": lambda: cmd_host(envp, out), "reduce": lambda: cmd_reduce(envp, out, "--fill-env" in sys.argv),
     "guard": lambda: cmd_guard(envp, out)}[cmd]()
