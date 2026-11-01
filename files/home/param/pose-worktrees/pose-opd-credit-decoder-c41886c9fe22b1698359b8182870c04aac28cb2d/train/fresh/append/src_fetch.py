"""APPEND ingest step 3 (one task per host): fetch THIS host's placed files (<W>/placement/<host>.tsv) into <DEST>/<rel path>.
Source from <W>/LISTING.json: hf -> the bucket's /resolve/ endpoint -> signed CDN URL (= data/new-pose-llm-data-dl/assemble.py's 'hf' path,
token-bucket rate HF_RATE files/s per host, default 60: HF's edge 429-blocks the cluster above ~2-4k files/s, and the hub's resolver quota is
500k / 5 min); local -> threaded copy (THREADS_LOCAL, default 32). Resume-safe: a file is done when <DEST>/<rel> exists at its listed size;
writes <dst>.part then renames. Never prints the token. -> <W>/fetch/<host>.DONE | .FAILED (+ .json stats).
usage (voicebank venv, python -I):  src_fetch.py <W> <DEST>"""
import json
import os
import random
import shutil
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

W, DEST = sys.argv[1], sys.argv[2]
host = socket.gethostname().split(".")[0]
L = json.load(open(f"{W}/LISTING.json"))
SRC = L["src"]
HF_RATE = float(os.environ.get("HF_RATE", "60"))
T_HF, T_LOCAL = int(os.environ.get("THREADS_HF", "64")), int(os.environ.get("THREADS_LOCAL", "32"))
items = []
for ln in open(f"{W}/placement/{host}.tsv"):
    rel, s, xh = ln.rstrip("\n").split("\t")
    items.append((rel, int(s)))
os.makedirs(f"{DEST}/jobs", exist_ok=True); os.makedirs(f"{W}/fetch", exist_ok=True)   # (jobs/ even when this host got no files: the scan lists it)
for f in (f"{W}/fetch/{host}.DONE", f"{W}/fetch/{host}.FAILED"):
    if os.path.exists(f):
        os.remove(f)
lock = threading.Lock()
st = {"done": 0, "skip": 0, "bytes": 0, "fail": 0, "r429": 0}
t0 = time.time()
tl = threading.local()
hf_pause, bucket = [0.0], [time.time()]
if SRC["kind"] == "hf":
    import requests
    tokf = os.path.expanduser("~/.cache/huggingface/token")
    TOKEN = os.environ.get("HF_TOKEN") or open(tokf).read().strip()
    BASE = f"https://huggingface.co/buckets/{SRC['bucket']}/resolve/"
    PFX = (SRC["prefix"] + "/") if SRC["prefix"] else ""


def sess():
    if not hasattr(tl, "s"):
        tl.s = requests.Session(); tl.s.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=4, max_retries=0))
    return tl.s


def hf_slot():
    while True:
        with lock:
            now = time.time(); w = max(hf_pause[0] - now, bucket[0] - now)
            if w <= 0:
                bucket[0] = max(bucket[0], now) + 1.0 / HF_RATE; return
        time.sleep(min(w, 5) + random.random() * 0.05)


def one(item):
    rel, s = item
    dst = f"{DEST}/{rel}"
    if os.path.exists(dst) and os.path.getsize(dst) == s:
        with lock:
            st["skip"] += 1
        return
    tmp = dst + ".part"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(40):
        try:
            if SRC["kind"] == "local":
                shutil.copyfile(f"{SRC['root']}/{rel}", tmp)
            else:
                hf_slot()
                r = sess().get(BASE + quote(PFX + rel), headers={"Authorization": f"Bearer {TOKEN}"}, allow_redirects=False, timeout=60)
                if r.status_code == 429:
                    with lock:
                        hf_pause[0] = max(hf_pause[0], time.time() + 60); st["r429"] += 1
                    print(f"[{host}] 429 from the hub; pausing 60 s", flush=True); continue
                if r.status_code not in (301, 302, 303, 307, 308):
                    raise IOError(f"resolve status {r.status_code}")
                with sess().get(r.headers["location"], stream=True, timeout=120) as g:
                    if g.status_code != 200:
                        raise IOError(f"cdn status {g.status_code}")
                    with open(tmp, "wb") as o:
                        for c in g.iter_content(4 << 20):
                            o.write(c)
            if os.path.getsize(tmp) != s:
                raise IOError(f"size {os.path.getsize(tmp)} != {s}")
            os.replace(tmp, dst)
            with lock:
                st["done"] += 1; st["bytes"] += s
            return
        except Exception as e:  # noqa: BLE001
            if attempt % 5 == 4:
                print(f"[{host}] retry {attempt} {rel}: {type(e).__name__} {str(e)[:160]}", flush=True)
            time.sleep(min(30, 2 + attempt) + random.random())
    with lock:
        st["fail"] += 1
    print(f"[{host}] FAILED {rel}", flush=True)


def report(final=False):
    el = time.time() - t0
    print(f"[{host}] {'FINAL ' if final else ''}{time.strftime('%H:%M:%S')} el {el:.0f}s files {st['done'] + st['skip']}/{len(items)} (new {st['done']}, "
          f"present {st['skip']}) {st['bytes'] / 1e9:.2f} GB {st['bytes'] / 1e9 / max(el, 1):.3f} GB/s fail {st['fail']} 429s {st['r429']}", flush=True)


stop = [False]


def ticker():
    while not stop[0]:
        time.sleep(15); report()


threading.Thread(target=ticker, daemon=True).start()
print(f"[{host}] fetch {len(items):,} files {sum(s for _, s in items) / 1e9:.2f} GB from {SRC['kind']} -> {DEST}; "
      + (f"hf rate {HF_RATE}/s, {T_HF} threads" if SRC["kind"] == "hf" else f"{T_LOCAL} threads"), flush=True)
with ThreadPoolExecutor(T_HF if SRC["kind"] == "hf" else T_LOCAL) as ex:
    list(ex.map(one, sorted(items, key=lambda x: (not x[0].endswith(".parquet"), x[0]))))
stop[0] = True; report(final=True)
res = dict(st, host=host, files=len(items), seconds=round(time.time() - t0, 1))
json.dump(res, open(f"{W}/fetch/{host}.json", "w"))
open(f"{W}/fetch/{host}.{'DONE' if st['fail'] == 0 else 'FAILED'}", "w").write(json.dumps(res) + "\n")
sys.exit(0 if st["fail"] == 0 else 1)
