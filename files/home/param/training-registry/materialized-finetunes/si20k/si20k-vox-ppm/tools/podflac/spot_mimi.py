"""spot check (CPU, ON the host): re-encode a fetched A-side FLAC with Mimi and compare with the store's channel-A codes of that row.
Gain = the upstream Mimi preprocessing (voxtral_front.lufs_gain_db over the speaker's WHOLE episode: every chunk of that speaker is fetched
read-only to a temp dir on /scratch and deleted after). Encode window = the previous chunk's last 30 s (streaming context; none for chunk 0)
+ this row's 600 s. Controls: the row's B-channel codes, and the A codes shifted by +-1 frame.  usage: spot_mimi.py HOST PICK OUT_JSON
PICK: swap0_k | swap1_B | padded | chunk0 | eval"""
import json
import os
import shutil
import socket
import sys
import urllib.parse

import numpy as np
import pyarrow.parquet as pq
import requests
import soundfile as sf
import torch

sys.path.insert(0, "/data/home/amartyavarma/ppm/si20k-vox-ppm/code/lib")
import voxtral_front as VF  # noqa: E402

HOST, PICK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
assert socket.gethostname().startswith(HOST)
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
MAN = "/data/home/amartyavarma/ppm/si20k-vox-ppm/plan/podcast_flac_manifest.jsonl"
TODO = f"/data/home/amartyavarma/ppm/si20k-vox-ppm/plan/podflac/todo/{HOST}.jsonl"
KEY = "/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json"
BUCKET = "podcast-raw-audio-data-mass"
NS, SPF, CTX = 14_400_000, 1920, 375

rows = [json.loads(l) for l in open(MAN)]
rows = [m for m in rows if m["host"] == HOST]
todo = {json.loads(l)["row_key"]: json.loads(l) for l in open(TODO)}
sel = {"swap0_k": lambda m: m["swap"] == 0 and m["chunk_index"] >= 2 and m["dataset"] == "A",
       "swap1_B": lambda m: m["swap"] == 1 and m["dataset"] == "B",
       "padded": lambda m: m.get("padded_samples", 0) > 0,
       "chunk0": lambda m: m["chunk_index"] == 0,
       "eval": lambda m: m["kind"] == "eval"}[PICK]
m = next(x for x in rows if sel(x))
t = todo[m["row_key"]]
print("row", m["row_key"], "side", m["side"], "swap", m["swap"], "chunk", m["chunk_index"], "pad", m.get("padded_samples"), flush=True)

# store codes of this row (live node store, read-only)
d = pq.read_table(f"{t['live_root']}/rows.parquet", columns=["row", "row_key", "shard", "row_in_shard"]).to_pydict()
r = t["node_row"]
assert d["row_key"][r] == m["row_key"]
sh, ri = d["shard"][r], d["row_in_shard"][r]
cA = np.load(f"{t['live_root']}/{sh}.codes_{m['side']}.npy", mmap_mode="r")[ri].astype(np.int64)
cB = np.load(f"{t['live_root']}/{sh}.codes_{t['chan_B_data_side']}.npy", mmap_mode="r")[ri].astype(np.int64)

# every chunk of the A speaker (read-only GCS) for the whole-episode gain
from google.oauth2 import service_account  # noqa: E402
import google.auth.transport.requests as gtr  # noqa: E402
cred = service_account.Credentials.from_service_account_file(KEY, scopes=["https://www.googleapis.com/auth/devstorage.read_only"])
cred.refresh(gtr.Request())
H = {"Authorization": "Bearer " + cred.token}
spk_prefix = m["source"].split(f"gs://{BUCKET}/", 1)[1].rsplit("/", 1)[0] + "/"
lst = requests.get(f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o", params={"prefix": spk_prefix, "fields": "items(name,size)"},
                   headers=H, timeout=60).json()
chunks = sorted(x["name"] for x in lst.get("items", []) if x["name"].rsplit("/", 1)[1].startswith("chunk_") and x["name"].endswith(".flac"))
tmpd = f"/scratch/amartyavarma/si20k-podcast-flac/_spot/{PICK}"
os.makedirs(tmpd, exist_ok=True)
waves = []
for nm in chunks:
    p = f"{tmpd}/{nm.rsplit('/', 1)[1]}"
    rr = requests.get(f"https://storage.googleapis.com/download/storage/v1/b/{BUCKET}/o/{urllib.parse.quote(nm, safe='')}?alt=media",
                      headers=H, timeout=120)
    assert rr.status_code == 200, rr.status_code
    open(p, "wb").write(rr.content)
    waves.append(sf.read(p, dtype="float32")[0])
k = m["chunk_index"]
local, _ = sf.read(m["path"], dtype="float32")
assert len(local) == NS
src_k = waves[k]
same = bool(np.array_equal(local[:len(src_k)], src_k)) and not np.any(local[len(src_k):])
ep = np.concatenate(waves)
gain_db, lufs_in, status = VF.lufs_gain_db(ep)
g = np.float32(10.0 ** (gain_db / 20.0))
ctx = waves[k - 1][-CTX * SPF:] if k > 0 else np.zeros(0, np.float32)
x = np.concatenate([ctx, local]) * g
shutil.rmtree(tmpd)

from transformers import MimiModel  # noqa: E402
mm = MimiModel.from_pretrained("/data/home/amartyavarma/models/mimi").eval()
with torch.no_grad():
    codes = mm.encode(torch.from_numpy(x)[None, None], num_quantizers=32).audio_codes[0].numpy().astype(np.int64)
off = len(ctx) // SPF
c = codes[:, off:off + 7500]
n = min(c.shape[1], 7500)
c, A, B = c[:, :n], cA[:, :n], cB[:, :n]
w = slice(250 if k == 0 else 0, n)


def rate(a, b, cb=None):
    return float((a[cb] == b[cb]).mean()) if cb is not None else float((a == b).mean())


res = {"host": HOST, "pick": PICK, "row_key": m["row_key"], "side": m["side"], "swap": m["swap"], "chunk_index": k,
       "padded_samples": m.get("padded_samples", 0), "local_equals_source_chunk": same, "episode_chunks": len(chunks),
       "gain_db": round(gain_db, 2), "lufs_in": round(lufs_in, 2), "gain_status": status, "frames": n,
       "cb0_match_A": rate(c[:, w], A[:, w], 0), "all32_match_A": rate(c[:, w], A[:, w]), "cb0_match_B_control": rate(c[:, w], B[:, w], 0),
       "cb0_match_A_shift+1": float((c[0, 1:] == A[0, :-1]).mean()), "cb0_match_A_shift-1": float((c[0, :-1] == A[0, 1:]).mean()),
       "cb0_match_A_first250": rate(c[:, :250], A[:, :250], 0)}
print(json.dumps(res), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
