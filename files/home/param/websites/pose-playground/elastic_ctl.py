#!/usr/bin/env python3
"""10-28 main / user ("why are multiple renders and generations not happening concurrently"): ELASTIC monoplex workers. Every 3 s: when a model's
queued runs exceed its free workers, an IDLE worker of a model with an EMPTY queue (and >= 1 other worker still on that model) reloads to the busy
model (genweb/UNIT_WEIGHTS/<job>.txt + restart of its worker python by its unit loop; same weights / maths per model). A worker on a foreign model
goes home once that model's queue has been empty for HOME_S. Never mid-run; one switch at a time; MIN_DWELL after each load."""
import glob, json, os, subprocess, time
CANOPY_ROOT = os.environ.get("CANOPY_ROOT") or os.path.expanduser("~")   # cluster home holding ppm/, venvs/, models/ (see README: paths)
HOME_DIR = os.path.expanduser("~")
G = os.path.dirname(os.path.abspath(__file__)) + ""
# worker id, slurm job, worker log name, home weights (10-28 main: static split 4 raw / 2 32B / 1 SAM while the user iterates on raw)
WORKERS = [("mono31177", "31177", "mono-31177", "base"), ("base31637", "31637", "base-31637", "mono17850raw"), ("base31640", "31640", "base-31640", "base"),
           ("mono17850raw31458", "31458", "mono17850raw-31458", "mono17850raw"), ("mono17850raw31639", "31639", "mono17850raw-31639", "mono17850raw"),
           ("mono17850sam31459", "31459", "mono17850sam-31459", "mono17850raw"), ("mono17850sam31638", "31638", "mono17850sam-31638", "mono17850sam")]
MODELS = ("base", "mono17850raw", "mono17850sam")
MIN_DWELL, HOME_S, ROUNDS = 30, 300, 2   # 10-28: with the fast loader a switch is ~40 s (< 1 round of a 60 s gen) -> switch for >= 2 rounds of backlog
st = {w[0]: {"loading_since": None, "n0": 0, "loaded_at": time.time(), "empty_since": None} for w in WORKERS}


def log(*a):
    print(time.strftime("[elastic %H:%M:%S]"), *a, flush=True)


def cur(job, home):
    try:
        return open(f"{G}/UNIT_WEIGHTS/{job}.txt").read().strip() or home
    except OSError:
        return home


def nloaded(lf):
    try:
        return sum(1 for ln in open(f"{G}/logs/worker-{lf}.log", errors="ignore") if "loaded + captured" in ln)
    except OSError:
        return 0


def busy(wid):
    for c in glob.glob(f"{G}/runs/*/CLAIMED.{wid}"):
        try:
            if json.load(open(os.path.join(os.path.dirname(c), "status.json"))).get("state") == "running":
                return True
        except Exception:
            pass
    return False


def queued():
    q = {m: 0 for m in MODELS}
    for r in glob.glob(f"{G}/runs/*/READY"):
        try:
            rq = json.load(open(os.path.join(os.path.dirname(r), "request.json")))
        except Exception:
            continue
        if rq.get("hidden"):                                            # 10-28: agent test runs never move workers (a hidden SAM test pulled 32B workers away)
            continue
        if (rq.get("model") or "monoplex") == "monoplex":
            m = rq.get("mono_weights") or "base"
            if m in q:
                q[m] += 1
    return q


def switch(w, to):
    wid, job, lf, home = w
    s = st[wid]
    s["n0"], s["loading_since"] = nloaded(lf), time.time()
    os.makedirs(f"{G}/UNIT_WEIGHTS", exist_ok=True)
    p = f"{G}/UNIT_WEIGHTS/{job}.txt"
    if to == home:
        if os.path.exists(p):
            os.replace(p, f"{G}/UNIT_WEIGHTS/.{job}.home-{int(time.time())}")
    else:
        open(p + ".tmp", "w").write(to); os.replace(p + ".tmp", p)
    subprocess.run(["srun", "--jobid", job, "--overlap", "-n1", "-c1", "--mem=256M", "--cpu-bind=none", "bash", "-c",
                    f"for p in $(pgrep -f -- 'worker[.]py {wid}'); do kill $p; done"], stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
    log(f"{wid}: switch -> {to}")


log("elastic controller up:", ", ".join(f"{w[0]}({w[3]})" for w in WORKERS))
while True:
    try:
        now = time.time(); q = queued()
        for w in WORKERS:                                               # finish loads
            s = st[w[0]]
            if s["loading_since"] and nloaded(w[2]) > s["n0"]:
                s["loading_since"], s["loaded_at"] = None, now; log(f"{w[0]}: loaded {cur(w[1], w[3])} ({q})")
        serving = {m: [w for w in WORKERS if cur(w[1], w[3]) == m] for m in MODELS}
        idle = {w[0]: (not st[w[0]]["loading_since"]) and not busy(w[0]) and not os.path.exists(f"{G}/flex/drain/{w[0]}") for w in WORKERS}
        free = {m: sum(1 for w in serving[m] if idle[w[0]]) for m in MODELS}
        loading = {m: sum(1 for w in serving[m] if st[w[0]]["loading_since"]) for m in MODELS}
        did = False
        for m in sorted(MODELS, key=lambda m: -(q[m] - free[m] - loading[m])):
            nm = len(serving[m])
            if q[m] - free[m] - loading[m] <= 0 or q[m] < ROUNDS * max(1, nm):   # backlog must outlast a reload
                continue
            for w in WORKERS:                                           # donor: idle, dwell done, its model has an empty queue and keeps >= 1 other worker
                n = cur(w[1], w[3])
                if n == m or not idle[w[0]] or now - st[w[0]]["loaded_at"] < MIN_DWELL or q[n] > 0 or len(serving[n]) < 2:
                    continue
                switch(w, m); did = True; break
            if did:
                break
        if not did:                                                     # go home after the foreign model's queue has been quiet for HOME_S
            for w in WORKERS:
                n = cur(w[1], w[3]); s = st[w[0]]
                if n == w[3]:
                    s["empty_since"] = None; continue
                s["empty_since"] = (s["empty_since"] or now) if q[n] == 0 else None
                if idle[w[0]] and s["empty_since"] and now - s["empty_since"] >= HOME_S and now - s["loaded_at"] >= MIN_DWELL:
                    switch(w, w[3]); break
    except Exception as e:  # noqa: BLE001
        log("error", repr(e)[:200])
    time.sleep(3)
