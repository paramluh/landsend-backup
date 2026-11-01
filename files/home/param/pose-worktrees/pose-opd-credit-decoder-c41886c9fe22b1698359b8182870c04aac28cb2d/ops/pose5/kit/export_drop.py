"""pose5 EXPORT /data DROP (main 10-27: the standing MoT policy of 10-24, applied to this run). Every COMPLETE consolidated export
/data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-N WRITTEN BY THIS RUN is deleted from /data once ALL of these hold (fail closed: anything
missing / mismatched / ambiguous -> KEEP + ALERT):
  1 N > 17850, N != 15973 (shared export, hard refusal), the dir is exactly ckpt-weka/ppm-v1/w-N (nothing MoT, nothing outside), COMPLETE
  2 this run wrote it: one of this run's jobs (kit/state/LAUNCHED_RUN J + the pose5 watchdog's relaunches + CURRENT_JOB_811) logged
    "[export] <root>/ppm-v1/w-N: consolidated" in ppm/logs/ppm-<job>.log, and the hfb record's job is that same job
  3 the HF watcher's upload record: setup-logs/hfb/records.jsonl {"kind": "w", "step": N, "job": J, "dest": "hf", "PASS": true, "repo": R}
    (exactly one such record) + done/w-N with PASS true. The upload is a BUNDLE (export + run_state/ + hfb's gen/w-N/ top-level files such as
    RESTORE.md; audit 2 R2-5): its local_files / local_bytes count all of it, so the on-disk export must be a strict subset of them
  4 a LIVE re-check right before deleting: HF <R> w-N/ minus run_state/ and the gen top-level names == the local export (same names, count,
    every size), every LFS file's sha256 == the local file's, and the WHOLE HF w-N/ listing == the verified totals (files + bytes)
  (--explain N: steps 3-4 for one step even if refused, e.g. w-15973 -> "would match but REFUSED: 15973"; never deletes)
Then rm -rf ONLY that w-N dir; one line each to pose5/STATUS.md, kit/logs/DELETED-<date>.log and setup-logs/hfb/DELETED-1021-hfb-hold.log.
usage: python3 -I export_drop.py [--dry-run]       (hold_chain.sh runs it in the background under a lock every poll)
tests: --selftest (synthetic records incl. a w-15973 decoy, another run's export, a MoT dir, a bytes mismatch, a missing record)"""
import argparse, glob, hashlib, json, os, re, shutil, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--root", default="/data/home/amartyavarma/ppm/ckpt-weka")
ap.add_argument("--hfb", default="/data/home/amartyavarma/setup-logs/hfb")
ap.add_argument("--logs", default="/data/home/amartyavarma/ppm/logs")
ap.add_argument("--jobs", default="", help="override this run's job ids (tests)")
ap.add_argument("--fake-hf", default="", help="tests: JSON {repo: {relpath: [size, sha256|null]}} instead of the live HF listing")
ap.add_argument("--status", default="/data/home/amartyavarma/ppm/pose5/STATUS.md")
ap.add_argument("--klog", default="")
ap.add_argument("--selftest", action="store_true")
ap.add_argument("--no-live", action="store_true", help="list candidates without the live HF re-check / sha256 (login-node dry runs); never deletes")
ap.add_argument("--explain", type=int, default=0, help="tests: record + bundle match for ONE step even if refused; never deletes")
ap.add_argument("--sha", action="store_true", help="with --explain: also sha256 every LFS file (slow: ~89 GB per export)")
a = ap.parse_args()
os.environ["TZ"] = "America/Los_Angeles"; time.tzset()
pt = lambda: time.strftime("%m-%d %H:%M:%S PT")  # noqa: E731
K = "/data/home/amartyavarma/ppm/pose5/kit"
PIN_STEP, SHARED = 17850, {15973}


def run_jobs():
    if a.jobs:
        return set(a.jobs.split(","))
    j = set()
    try:
        j |= set(re.findall(r"^J=(\d+)", open(f"{K}/state/LAUNCHED_RUN").read(), re.M))
    except Exception:
        pass
    try:
        j |= {l.split()[1] for l in open("/data/home/amartyavarma/ppm/control/ppm-v1/watchdog_pose5_state/relaunches") if len(l.split()) > 1}
    except Exception:
        pass
    try:
        c = open("/data/home/amartyavarma/ppm/control/ppm-v1/CURRENT_JOB_811").read().strip()
        if j and c.isdigit():
            j.add(c)                                           # (only alongside a launch record: never adopt a foreign job alone)
    except Exception:
        pass
    return j


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(16 << 20), b""):
            h.update(c)
    return h.hexdigest()


def hf_listing(repo, n):
    if a.fake_hf:
        return {k: tuple(v) for k, v in json.load(open(a.fake_hf)).get(repo, {}).get(f"w-{n}", {}).items()}
    from huggingface_hub import HfApi
    out = {}
    for e in HfApi().list_repo_tree(repo, path_in_repo=f"w-{n}", recursive=True, expand=True, repo_type="model"):
        if getattr(e, "size", None) is not None:
            out[e.path[len(f"w-{n}/"):]] = (int(e.size), e.lfs.sha256 if getattr(e, "lfs", None) else None)
    return out


def hf_manifest(repo, n):
    """the bundle's own per-file listing, written by hfb_up.py and uploaded with it: _backup/MANIFEST.w-N.tsv (path \\t size \\t host:source)
    -> (rows {repo_path: (size, source)}, manifest size in bytes)"""
    name = f"_backup/MANIFEST.w-{n}.tsv"
    if a.fake_hf:
        txt = json.load(open(a.fake_hf)).get(repo, {}).get(name)
        if txt is None:
            raise FileNotFoundError(name)
    else:
        import tempfile
        from huggingface_hub import hf_hub_download
        td = tempfile.mkdtemp(prefix="exdrop-man-", dir=f"{K}/state")
        try:
            txt = open(hf_hub_download(repo, name, repo_type="model", cache_dir=td), errors="replace").read()
        finally:
            shutil.rmtree(td, ignore_errors=True)               # this call's own temp download dir only
    rows = {}
    for ln in txt.splitlines()[1:]:
        p = ln.split("\t")
        if len(p) >= 3 and p[1].isdigit():
            rows[p[0]] = (int(p[1]), p[2])
    return rows, len(txt.encode())


def check(w, jobs, recs):
    """-> (ok, why, info)"""
    n_s = os.path.basename(w)[2:]
    if not n_s.isdigit():
        return False, "not a w-<step> dir", {}
    n = int(n_s)
    if n in SHARED:
        return False, "w-15973 is shared (hard refusal)", {}
    if n <= PIN_STEP:
        return False, f"step {n} <= {PIN_STEP}: not this run's", {}
    if os.path.realpath(w) != os.path.join(os.path.realpath(a.root), "ppm-v1", f"w-{n}") or "mot" in w.lower():
        return False, "outside ckpt-weka/ppm-v1 (or MoT)", {}
    if not os.path.isfile(f"{w}/COMPLETE"):
        return False, "not COMPLETE", {}
    writers = [j for j in jobs if re.search(rf"\[export\] \S*/ppm-v1/w-{n}: consolidated", open(f"{a.logs}/ppm-{j}.log", errors="replace").read()
                                          if os.path.isfile(f"{a.logs}/ppm-{j}.log") else "")]
    if len(writers) != 1:
        return False, f"written by {len(writers)} of this run's jobs {sorted(jobs)} (need exactly 1)", {}
    rs = [r for r in recs if r.get("kind") == "w" and int(r.get("step", -1)) == n]
    if len(rs) != 1:
        return False, f"{len(rs)} hfb 'w' records for step {n} (need exactly 1)", {}
    r = rs[0]
    if str(r.get("job")) != writers[0] or r.get("dest") != "hf" or r.get("PASS") is not True or not str(r.get("repo", "")).startswith("InternalCan/ppm-v1"):
        return False, f"hfb record mismatch (job {r.get('job')} vs writer {writers[0]}, dest {r.get('dest')}, PASS {r.get('PASS')}, repo {r.get('repo')})", {}
    try:
        d = json.load(open(f"{a.hfb}/done/w-{n}"))
    except Exception:
        return False, "no done/w-N record", {}
    if d.get("PASS") is not True:
        return False, "done/w-N not PASS", {}
    ok, why, info = bundle_match(w, n, r, d, live=not a.no_live, do_sha=True)
    return ok, (f"job {writers[0]} wrote it; " + why) if ok else why, info


def bundle_match(w, n, r, d, live, do_sha):
    """audit 2 R2-5: the hfb 'w' upload is a BUNDLE = the export dir + run_state/ + hfb's gen files + the repo-root RESTORE.md + its own
    _backup/MANIFEST.w-N.tsv; done/w-N local_files / local_bytes count all of it. So:
      records: the on-disk export is a strict subset of the verified totals
      live   : the bundle's MANIFEST (from HF) is the verified one (rows + itself == the verified files / bytes); its rows whose SOURCE is this
               export dir == the on-disk export exactly (names, count, every size); every MANIFEST row under w-N/ is on HF with that size and
               HF w-N/ holds nothing else; every LFS sha256 of the export part == the local file's. Anything else -> keep."""
    loc = {os.path.relpath(os.path.join(p, x), w): os.path.getsize(os.path.join(p, x)) for p, _, xs in os.walk(w) for x in xs}
    b, vf, vb = sum(loc.values()), int(d.get("local_files", -1)), int(d.get("local_bytes", -1))
    if not (0 < len(loc) < vf and 0 < b < vb):
        return False, f"on-disk export {len(loc)} files / {b} B is not a strict subset of the verified bundle {vf} / {vb}", {}
    if not live:
        return True, (f"CANDIDATE (hfb record PASS in {r['repo']}; export {len(loc)} files / {b} B within the verified bundle {vf} / {vb}); "
                      "live HF re-check + sha256 pending"), {"bytes": b, "repo": r["repo"]}
    rows, msz = hf_manifest(r["repo"], n)                      # LIVE: the bundle's own per-file listing
    mb = sum(s for s, _ in rows.values()) + msz
    if len(rows) + 1 != vf or mb != vb:
        return False, f"HF _backup/MANIFEST.w-{n}.tsv {len(rows) + 1} files / {mb} B != verified bundle {vf} / {vb}", {}
    src = os.path.realpath(w) + "/"
    exp = {}
    for p, (s, so) in rows.items():
        sp = os.path.realpath(so.split(":", 1)[1]) if ":" in so else ""
        if sp.startswith(src):
            if p != f"w-{n}/" + sp[len(src):]:
                return False, f"MANIFEST row {p} maps the export file {sp} elsewhere", {}
            exp[sp[len(src):]] = s
    if set(exp) != set(loc) or any(exp[k] != loc[k] for k in loc):
        return False, (f"MANIFEST export rows != on-disk export (MANIFEST {len(exp)} files, local {len(loc)}; diff {sorted(set(exp) ^ set(loc))[:3]}; "
                       f"size diffs {[k for k in loc if k in exp and exp[k] != loc[k]][:3]})"), {}
    hf = hf_listing(r["repo"], n)                              # LIVE re-check (every file under w-N/ now)
    mw = {p[len(f"w-{n}/"):]: s for p, (s, _) in rows.items() if p.startswith(f"w-{n}/")}
    if set(hf) != set(mw) or any(hf[k][0] != mw[k] for k in mw):
        return False, (f"live HF w-{n}/ != its MANIFEST rows (HF {len(hf)} files, MANIFEST {len(mw)}; diff {sorted(set(hf) ^ set(mw))[:3]}; "
                       f"size diffs {[k for k in mw if k in hf and hf[k][0] != mw[k]][:3]})"), {}
    if do_sha:
        for k in sorted(exp):
            h = hf[k][1]
            if h and sha(os.path.join(w, k)) != h:
                return False, f"sha256 mismatch {k}", {}
    return True, (f"hfb record PASS in {r['repo']}; MANIFEST == verified bundle ({vf} files / {vb} B); its export rows == the on-disk export "
                  f"({len(loc)} files / {b} B); live HF w-{n}/ == MANIFEST ({len(hf)} files, sizes{' + sha256' if do_sha else '; sha256 NOT run'})"), \
        {"bytes": b, "repo": r["repo"]}


def explain(n):
    """tests: run the record + bundle match for step n even if it is refused (hard refusal / <= pin), without deleting anything"""
    recs = [json.loads(l) for l in open(f"{a.hfb}/records.jsonl") if l.strip()] if os.path.isfile(f"{a.hfb}/records.jsonl") else []
    rs = [r for r in recs if r.get("kind") == "w" and int(r.get("step", -1)) == n]
    w = f"{a.root}/ppm-v1/w-{n}"
    if len(rs) != 1 or not os.path.isfile(f"{a.hfb}/done/w-{n}"):
        print(f"would NOT match: {len(rs)} records / done/w-{n} {'present' if os.path.isfile(f'{a.hfb}/done/w-{n}') else 'missing'}"); return 1
    d = json.load(open(f"{a.hfb}/done/w-{n}"))
    ok, why, _ = bundle_match(w, n, rs[0], d, live=not a.no_live, do_sha=a.sha)
    refused = n in SHARED or n <= PIN_STEP
    print((f"would match but REFUSED: {n}" if refused else f"would DROP: {n}") + f" :: {why}" if ok else f"would NOT match: {n} :: {why}")
    return 0 if ok else 1


def main():
    jobs = run_jobs()
    recs = []
    for l in open(f"{a.hfb}/records.jsonl") if os.path.isfile(f"{a.hfb}/records.jsonl") else []:
        try:
            recs.append(json.loads(l))
        except Exception:
            pass
    klog = a.klog or f"{K}/logs/DELETED-{time.strftime('%Y%m%d')}.log"
    rc = 0
    for w in sorted(glob.glob(f"{a.root}/ppm-v1/w-*")):
        try:
            ok, why, info = check(w, jobs, recs)
        except Exception as e:                                 # e.g. HF API / import error: KEEP + ALERT (fail closed, loud)
            ok, why, info = False, f"check raised {type(e).__name__}: {str(e)[:160]} [ALERT: export drop check crashed]", {}
            rc = 3
        n = os.path.basename(w)
        if not ok:
            stale = n[2:].isdigit() and int(n[2:]) > PIN_STEP and int(n[2:]) not in SHARED and os.path.isfile(f"{w}/COMPLETE") \
                and time.time() - os.path.getmtime(f"{w}/COMPLETE") > 3 * 3600
            print(f"KEEP {w}: {why}" + (" [ALERT: a COMPLETE pose-run export still not droppable after 3 h]" if stale else ""))
            if stale:
                rc = 3
            continue
        if a.dry_run or a.no_live:
            print(f"WOULD DROP {w} ({info['bytes'] / 1e9:.1f} GB): {why}"); continue
        shutil.rmtree(w)
        line = f"[{pt()}] pose5 export drop (main 10-27 = MoT policy): deleted /data copy {w} ({info['bytes']} B) :: {why}"
        for f in (klog, a.status if not a.fake_hf else klog, f"{a.hfb}/DELETED-1021-hfb-hold.log" if not a.fake_hf else klog):
            open(f, "a").write((line if not f.endswith("STATUS.md") else "- " + line) + "\n")
        print(f"DROPPED {w}")
    return rc


def selftest():
    import tempfile
    T = tempfile.mkdtemp(prefix="exdrop-", dir="/data/home/amartyavarma/ppm/pose5/kit/state")
    R, Hb, Lg = f"{T}/ckpt-weka", f"{T}/hfb", f"{T}/logs"
    for d in (f"{R}/ppm-v1", f"{R}/ppm-mot", f"{Hb}/done", Lg):
        os.makedirs(d)
    fake, recs, logs = {}, [], {"900": "", "800": ""}

    def mk(root, n, files, job, rec=True, done=True, logged=True, hfdiff=False, bytesdiff=False, repo="InternalCan/ppm-v1-d", hfextra=False,
           nobundle=False, extras_missing=False, nomanifest=False, localchanged=False):
        w = f"{root}/w-{n}"; os.makedirs(w)
        for k, c in files.items():
            open(f"{w}/{k}", "wb").write(c)
        open(f"{w}/COMPLETE", "w").write(str(n))
        loc = {k: len(c) for k, c in files.items()}; loc["COMPLETE"] = len(str(n))
        hf = {k: [s + (1 if hfdiff and k != "COMPLETE" else 0), hashlib.sha256(open(f"{w}/{k}", "rb").read()).hexdigest()
                  if k != "COMPLETE" else None] for k, s in loc.items()}
        # the hfb bundle as hfb_up.py builds it: w-N/<export>, w-N/run_state/..., w-N/RESTORE.md (gen), repo-root RESTORE.md, + its MANIFEST
        extra = {} if nobundle else {f"w-{n}/RESTORE.md": (700, f"slinky-15:{Hb}/gen/w-{n}/RESTORE.md"),
                                     f"w-{n}/run_state/config/cfg.json": (300, f"slinky-15:{Hb}/gen/w-{n}/run_state/config/cfg.json"),
                                     f"w-{n}/run_state/code/ppm-code/x.py": (50, "slinky-15:/data/home/amartyavarma/ppm/code/x.py"), "RESTORE.md": (900, "slinky-15:/x/R.md")}
        rows = {f"w-{n}/{k}": (s, f"slinky-15:{os.path.realpath(w)}/{k}") for k, s in loc.items()}
        rows.update(extra)
        man = "path\tsize\tsource\n" + "".join(f"{p}\t{s}\t{so}\n" for p, (s, so) in sorted(rows.items()))
        hfw = {**hf, **{p[len(f"w-{n}/"):]: [s, None] for p, (s, _) in extra.items() if p.startswith(f"w-{n}/") and not extras_missing}}
        if hfextra:
            hfw["stray.bin"] = [9, None]
        fake.setdefault(repo, {})[f"w-{n}"] = hfw
        if not nomanifest:
            fake[repo][f"_backup/MANIFEST.w-{n}.tsv"] = man
        if logged:
            logs[job] = logs.get(job, "") + f"[x] [export] {root}/w-{n}: consolidated bf16 weights\n"
        if rec:
            recs.append({"kind": "w", "step": n, "job": job, "dest": "hf", "PASS": True, "repo": repo})
        if done:                                               # the verified BUNDLE totals = every MANIFEST row + the MANIFEST itself
            vf, vb = (len(loc), sum(loc.values())) if nobundle else (len(rows) + 1, sum(s for s, _ in rows.values()) + len(man.encode()))
            json.dump({"PASS": True, "local_files": vf, "local_bytes": vb + (5 if bytesdiff else 0)}, open(f"{Hb}/done/w-{n}", "w"))
        if localchanged:
            open(f"{w}/a.distcp", "ab").write(b"!")
    mk(f"{R}/ppm-v1", 18100, {"a.distcp": b"x" * 100, ".metadata": b"m"}, "900")              # GOOD -> drop
    mk(f"{R}/ppm-v1", 15973, {"a.distcp": b"y" * 10}, "900")                                    # shared decoy -> keep
    mk(f"{R}/ppm-v1", 18200, {"a.distcp": b"z" * 10}, "800")                                    # another run's job -> keep
    mk(f"{R}/ppm-v1", 18300, {"a.distcp": b"q" * 10}, "900", rec=False)                         # no hfb record -> keep
    mk(f"{R}/ppm-v1", 18400, {"a.distcp": b"r" * 10}, "900", bytesdiff=True)                    # bytes != verified -> keep
    mk(f"{R}/ppm-v1", 18500, {"a.distcp": b"s" * 10}, "900", hfdiff=True)                       # live HF sizes differ -> keep
    mk(f"{R}/ppm-mot", 18600, {"a.distcp": b"t" * 10}, "900")                                   # MoT dir (never scanned) -> untouched
    mk(f"{R}/ppm-v1", 18700, {"a.distcp": b"u" * 10}, "900", hfextra=True)                      # HF w-N/ has a stray non-bundle file -> keep
    mk(f"{R}/ppm-v1", 18800, {"a.distcp": b"v" * 10}, "900", extras_missing=True)               # HF lacks the run_state/ part -> keep
    mk(f"{R}/ppm-v1", 18900, {"a.distcp": b"w" * 10}, "900", nobundle=True)                     # totals == export only (not a bundle) -> keep
    mk(f"{R}/ppm-v1", 19000, {"a.distcp": b"x" * 10}, "900", nomanifest=True)                   # no MANIFEST on HF -> KEEP + ALERT
    mk(f"{R}/ppm-v1", 19100, {"a.distcp": b"y" * 10}, "900", localchanged=True)                 # local file changed after upload -> keep
    for j, t in logs.items():
        open(f"{Lg}/ppm-{j}.log", "w").write(t)
    open(f"{Hb}/records.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in recs))
    json.dump(fake, open(f"{T}/fake_hf.json", "w"))
    import subprocess
    base = [sys.executable, "-I", __file__, "--root", R, "--hfb", Hb, "--logs", Lg, "--jobs", "900", "--fake-hf", f"{T}/fake_hf.json", "--klog", f"{T}/klog"]
    dry = subprocess.run(base + ["--dry-run"], capture_output=True, text=True).stdout
    real = subprocess.run(base, capture_output=True, text=True).stdout
    left = sorted(os.listdir(f"{R}/ppm-v1")) + [f"mot:{x}" for x in os.listdir(f"{R}/ppm-mot")]
    print("DRY:\n" + dry + "REAL:\n" + real + "left:", left)
    ok = sorted(left) == sorted(["mot:w-18600", "w-15973", "w-18200", "w-18300", "w-18400", "w-18500", "w-18700", "w-18800", "w-18900",
                                 "w-19000", "w-19100"]) and "export drop check crashed" in dry \
        and "WOULD DROP" in dry and "DROPPED" in real \
        and dry.count("WOULD DROP") == 1 and real.count("DROPPED") == 1
    shutil.rmtree(T)
    print("SELFTEST", "PASS" if ok else "FAIL"); return 0 if ok else 1


sys.exit(selftest() if a.selftest else explain(a.explain) if a.explain else main())
