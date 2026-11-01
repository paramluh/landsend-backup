"""si20k podcast A-side raw FLAC fetch, ONE host (runs ON that host: srun --overlap --cpu-bind=none, nice). Read-only GCS (devstorage.read_only),
never writes to any bucket. Resumable (rows whose final file exists + are in the host done log are skipped).
1 live check: every planned (slice, node row) -> row_key in THIS node's live rows.parquet == the mapping (meta copy). Mismatch = abort.
2 parallel ranged-free whole-object GETs (each row window = one whole source chunk FLAC, ~5-11 MB), W threads, md5 vs x-goog-hash.
3 check per file: FLAC, 24,000 Hz, mono, 14,400,000 samples (= 7,500 Mimi frames x 1,920) -> atomic rename to the final path.
4 host manifest (the chan-A speaker per row) -> /data plan/podflac/manifest/<host>.jsonl + a copy next to the audio.
usage: fetch_host.py HOST [WORKERS]"""
import base64
import hashlib
import json
import os
import socket
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
import requests
import numpy as np
import soundfile as sf

HOST = sys.argv[1]
W = int(sys.argv[2]) if len(sys.argv) > 2 else 64
BASE = "/data/home/amartyavarma/ppm/si20k-vox-ppm/plan/podflac"
DEST = "/scratch/amartyavarma/si20k-podcast-flac"
KEY = "/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json"
BUCKET = "podcast-raw-audio-data-mass"
SR, NS = 24000, 14_400_000
MIN_FREE_GB = 30
assert socket.gethostname().startswith(HOST), (socket.gethostname(), HOST)
assert DEST.startswith("/scratch/amartyavarma/si20k-podcast-flac")
os.makedirs(f"{BASE}/status", exist_ok=True)
os.makedirs(f"{BASE}/manifest", exist_ok=True)
os.makedirs(DEST, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), HOST, *a, flush=True)


recs = [json.loads(l) for l in open(f"{BASE}/todo/{HOST}.jsonl")]
assert all(r["host"] == HOST for r in recs)
recs.sort(key=lambda r: (r["kind"] != "eval", r["stream_pos"], r["local_rank"]))   # eval first, then stream position t for all ranks before t+1
MAXPOS = int(os.environ.get("MAXPOS", "0"))           # tight hosts: only train stream positions < MAXPOS (eval always)
if MAXPOS:
    recs = [r for r in recs if r["kind"] == "eval" or r["stream_pos"] < MAXPOS]
FLOOR_GB = float(os.environ.get("FLOOR_GB", "0"))       # stop starting new downloads when /scratch free < FLOOR_GB
if int(os.environ.get("LIMIT", "0")):
    recs = recs[:int(os.environ["LIMIT"])]

# ---- 1 live check of the mapping against this node's own slices
by_root = {}
for r in recs:
    by_root.setdefault(r["live_root"], []).append(r)
live = {}
for root, rs in by_root.items():
    d = pq.read_table(f"{root}/rows.parquet", columns=["row", "row_key"]).to_pydict()
    assert d["row"] == list(range(len(d["row"]))), root
    bad = [(r["node_row"], r["row_key"]) for r in rs if d["row_key"][r["node_row"]] != r["row_key"]]
    assert not bad, ("live rows.parquet disagrees with the mapping", root, bad[:5])
    live[root] = len(rs)
log("live mapping check ok", live)

# ---- disk
need = 9.0e6 * len(recs)   # upper bound (rows already on disk are skipped below)
free = os.statvfs("/scratch").f_bavail * os.statvfs("/scratch").f_frsize
if not FLOOR_GB:
    FLOOR_GB = MIN_FREE_GB   # never take /scratch below MIN_FREE_GB (per-file check below)
log(f"/scratch free {free/1e9:.0f} GB, need ~{need/1e9:.0f} GB, rows {len(recs)}, workers {W}")

# ---- token (read-only scope), never printed
from google.oauth2 import service_account  # noqa: E402
import google.auth.transport.requests as gtr  # noqa: E402

cred = service_account.Credentials.from_service_account_file(KEY, scopes=["https://www.googleapis.com/auth/devstorage.read_only"])
tok_lock = threading.Lock()


def token(force=False):
    with tok_lock:
        if force or not cred.valid:
            cred.refresh(gtr.Request())
        return cred.token


tl = threading.local()


def sess():
    if getattr(tl, "s", None) is None:
        s = requests.Session()
        s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=2, max_retries=0))
        tl.s = s
    return tl.s


done_log = f"{DEST}/.done.{HOST}.jsonl"
done = {}
if os.path.exists(done_log):
    for l in open(done_log):
        try:
            e = json.loads(l)
            if os.path.exists(e["path"]) and os.path.getsize(e["path"]) == e["bytes"]:
                done[e["row_key"]] = e
        except Exception:  # noqa: BLE001
            pass
log("already done", len(done))
lock = threading.Lock()
st = {"done": len(done), "bytes": sum(e["bytes"] for e in done.values()), "fail": 0, "retries": 0}
fails = []
dl = open(done_log, "a")
T0 = time.time()
B0 = st["bytes"]


def fetch(r):
    if r["row_key"] in done:
        return
    if FLOOR_GB:
        sv = os.statvfs("/scratch")
        if sv.f_bavail * sv.f_frsize < FLOOR_GB * 1e9:
            with lock:
                st["skipped_floor"] = st.get("skipped_floor", 0) + 1
            return
    url = f"https://storage.googleapis.com/download/storage/v1/b/{BUCKET}/o/{urllib.parse.quote(r['obj'], safe='')}?alt=media"
    path = r["path"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    err = None
    for att in range(6):
        try:
            h = hashlib.md5()
            resp = sess().get(url, headers={"Authorization": "Bearer " + token(force=att > 0 and err == "401")}, stream=True, timeout=(10, 120))
            if resp.status_code == 401:
                err = "401"
                resp.close()
                continue
            if resp.status_code != 200:
                err = f"http {resp.status_code}"
                resp.close()
                time.sleep(min(2 ** att, 20))
                continue
            gh = dict(x.split("=", 1) for x in resp.headers.get("x-goog-hash", "").split(",") if "=" in x)
            n = 0
            with open(tmp, "wb") as f:
                for ch in resp.iter_content(1 << 20):
                    f.write(ch)
                    h.update(ch)
                    n += len(ch)
            md5 = base64.b64encode(h.digest()).decode()
            if "md5" in gh and gh["md5"].strip() != md5:
                err = "md5 mismatch"
                continue
            inf = sf.info(tmp)
            if not (inf.format == "FLAC" and inf.samplerate == SR and inf.channels == 1 and NS - 1920 < inf.frames <= NS):
                raise ValueError(f"audio check: {inf.format} sr {inf.samplerate} ch {inf.channels} frames {inf.frames}")
            pad = NS - inf.frames
            if pad:   # the episode's last full chunk ends inside Mimi frame 7,499 (streaming Mimi zero-pads the partial frame): zero-pad to 7,500 x 1,920
                x, _ = sf.read(tmp, dtype="int32", always_2d=False)
                x = np.concatenate([x, np.zeros(pad, x.dtype)])
                sf.write(tmp + ".pad.flac", x, SR, subtype=inf.subtype, format="FLAC")
                assert sf.info(tmp + ".pad.flac").frames == NS
                os.replace(tmp + ".pad.flac", tmp)
            n_out = os.path.getsize(tmp)
            os.replace(tmp, path)
            e = {"row_key": r["row_key"], "path": path, "bytes": n_out, "src_bytes": n, "md5_b64": md5, "padded_samples": pad, "src_samples": inf.frames}
            with lock:
                dl.write(json.dumps(e) + "\n")
                dl.flush()
                done[r["row_key"]] = e
                st["done"] += 1
                st["bytes"] += n_out
            return
        except ValueError as ex:
            err = repr(ex)
            break
        except Exception as ex:  # noqa: BLE001
            err = repr(ex)[:200]
            with lock:
                st["retries"] += 1
            time.sleep(min(2 ** att, 20))
    with lock:
        st["fail"] += 1
        fails.append({"row_key": r["row_key"], "obj": r["obj"], "err": err})
    try:
        os.remove(tmp)
    except OSError:
        pass


def status(final=False):
    el = time.time() - T0
    d = dict(st, host=HOST, rows=len(recs), elapsed_s=round(el, 1), rate_MBps=round((st["bytes"] - B0) / max(el, 1e-3) / 1e6, 1),
             updated=time.strftime("%Y-%m-%d %H:%M:%S"), final=final, workers=W, fails=fails[:20])
    json.dump(d, open(f"{BASE}/status/{HOST}.json.tmp", "w"))
    os.replace(f"{BASE}/status/{HOST}.json.tmp", f"{BASE}/status/{HOST}.json")


stop = [False]


def ticker():
    while not stop[0]:
        status()
        time.sleep(3)


th = threading.Thread(target=ticker, daemon=True)
th.start()
with ThreadPoolExecutor(W) as ex:
    list(ex.map(fetch, recs))
stop[0] = True
dl.close()
el = time.time() - T0
log(f"download done {st['done']}/{len(recs)} fail {st['fail']} {(st['bytes']-B0)/1e9:.1f} GB in {el:.0f}s ({(st['bytes']-B0)/max(el,1e-3)/1e9:.2f} GB/s)")

# ---- host manifest
LIC = "internal podcast pretraining corpus (same SAM source-separated rows the model pretrained on as Mimi codes); commercial licence not verified"
with open(f"{BASE}/manifest/{HOST}.jsonl.tmp", "w") as f:
    for r in recs:
        e = done.get(r["row_key"])
        if e is None:
            continue
        m = {"row_key": r["row_key"], "host": HOST, "path": r["path"], "kind": r["kind"], "local_rank": r["local_rank"],
             "prod_rank": r["prod_rank"], "stream_pos": r["stream_pos"], "code": r["code"],
             "side": r["chan_A_data_side"], "frame0": 0, "channel": "A", "audio_channels": 1, "audio_is": "model channel A only (the partner of the row's B side; goes through Voxtral)",
             "swap": r["swap"], "chan_A_data_side": r["chan_A_data_side"], "A_speaker_dir": r["A_speaker_dir"], "A_speaker": r["A_speaker"],
             "chan_B_data_side": r["chan_B_data_side"], "B_speaker_dir": r["B_speaker_dir"], "B_speaker": r["B_speaker"],
             "source": r["gcs"], "dataset": r["dataset"], "podcast_id": r["podcast_id"], "episode_id": r["episode_id"],
             "chunk_index": r["chunk_index"], "sample_rate": SR, "n_samples": NS,
             "episode_start_sample": r["chunk_index"] * NS, "episode_end_sample": (r["chunk_index"] + 1) * NS,
             "frame_start": r["frame_start"], "frame_end": r["frame_start"] + r["n_frames"], "n_frames": r["n_frames"],
             "row_offset_samples": 0, "samples_per_frame": 1920, "level": "raw SAM output (not loudness-normalised; Mimi codes were made at -23 LUFS)",
             "bytes": e["bytes"], "src_md5_b64": e["md5_b64"], "src_samples": e.get("src_samples", NS), "padded_samples": e.get("padded_samples", 0),
             "licence": LIC}
        f.write(json.dumps(m) + "\n")
os.replace(f"{BASE}/manifest/{HOST}.jsonl.tmp", f"{BASE}/manifest/{HOST}.jsonl")
os.system(f"cp {BASE}/manifest/{HOST}.jsonl {DEST}/MANIFEST.{HOST}.jsonl")
status(final=True)
log("ALLDONE", json.dumps({k: v for k, v in st.items()}))
