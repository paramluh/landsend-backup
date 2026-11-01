"""MONOFT (10-22): place this FT host's needed init files (init_need.py need -> JSON) into the node-local init dir.
Own files: hard link from this host's stage dir (same /scratch fs; the stage dir is deleted afterwards, so no extra space). Other hosts' files:
HTTP GET from that host's stage server (python -m http.server on the pod network; no ssh between compute nodes here), streamed to <f>.tmp with
sha256 + size checked against the source host's manifest, then renamed. Nothing on /data is written except the log.
  python init_fetch.py --need <need.json> --init <dir> --stage <own stage dir> --port P [--workers 4] [--mbps 0 = unlimited, per stream]"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import socket
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--need", required=True)
ap.add_argument("--init", required=True)
ap.add_argument("--stage", required=True)
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--mbps", type=float, default=0.0)
ap.add_argument("--tries", type=int, default=8)
a = ap.parse_args()
me = socket.gethostname().split(".")[0]
need = json.load(open(a.need))
assert a.init.startswith("/scratch/"), a.init
os.makedirs(a.init, exist_ok=True)


def log(*x):
    print(f"[{time.strftime('%H:%M:%S')} {me}]", *x, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                return h.hexdigest()
            h.update(b)


def one(e):
    f, dst = e["file"], f"{a.init}/{e['file']}"
    if os.path.isfile(dst) and os.path.getsize(dst) == e["size"] and sha256(dst) == e["sha256"]:
        return f, "present", 0.0
    if e["owner"] == me:
        src = f"{a.stage}/{f}"
        assert os.path.getsize(src) == e["size"], (f, "stage size")
        if os.path.exists(dst):
            os.remove(dst)
        os.link(src, dst)
        return f, "linked", 0.0
    url = f"http://{e['owner']}:{a.port}/{f}"
    for t in range(a.tries):
        try:
            t0 = time.time(); h = hashlib.sha256(); n = 0
            with urllib.request.urlopen(url, timeout=120) as r, open(dst + ".tmp", "wb") as w:
                while True:
                    b = r.read(1 << 23)
                    if not b:
                        break
                    w.write(b); h.update(b); n += len(b)
                    if a.mbps > 0:
                        lag = n / (a.mbps * 1e6) - (time.time() - t0)
                        if lag > 0:
                            time.sleep(lag)
            assert n == e["size"] and h.hexdigest() == e["sha256"], (f, "size / sha256 mismatch", n, e["size"])
            os.replace(dst + ".tmp", dst)
            return f, f"fetched from {e['owner']}", n / 1e6 / max(time.time() - t0, 1e-3)
        except Exception as x:   # noqa: BLE001 (server not up yet / reset: retry, then fail loudly)
            log(f"{f} from {e['owner']} attempt {t + 1}: {x!r}")
            if os.path.exists(dst + ".tmp"):
                os.remove(dst + ".tmp")
            time.sleep(min(60, 5 * 2 ** t))
    raise RuntimeError(f"{f}: could not fetch from {e['owner']} after {a.tries} tries")


t0 = time.time()
with cf.ThreadPoolExecutor(a.workers) as ex:
    res = list(ex.map(one, need))
for f, how, mbs in res:
    log(f"{f}: {how}" + (f" ({mbs:.0f} MB/s)" if mbs else ""))
gb = sum(e["size"] for e in need) / 1e9
log(f"== {len(need)} files, {gb:.1f} GB in {a.init} ({sum(1 for r in res if r[1].startswith('fetched'))} fetched, "
    f"{sum(1 for r in res if r[1] == 'linked')} linked, {sum(1 for r in res if r[1] == 'present')} present) in {time.time() - t0:.0f}s")
