"""Download ONLY the Voxtral Realtime audio encoder (audio_tower.*) + config/feature-extractor files.

Reads the safetensors header with an HTTP range request, then range-fetches each audio_tower.* tensor and writes
a standalone safetensors file (keys keep the audio_tower. prefix). Run on a compute node.
usage: python fetch_voxtral_encoder.py [--out $ALOE_ROOT/models/voxtral-realtime]
"""
_ALOE_ROOT = __import__("os").environ["ALOE_ROOT"]        # release: every path comes from aloe.env (ALOE_ROOT)
import argparse, json, os, struct, hashlib
import requests
from huggingface_hub import hf_hub_download, hf_hub_url, list_repo_files, get_token

REPO = "mistralai/Voxtral-Mini-4B-Realtime-2602"
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.environ.get("VOXTRAL_DIR") or f"{_ALOE_ROOT}/models/voxtral-realtime")
ap.add_argument("--revision", default="2769294da9567371363522aac9bbcfdd19447add")   # RELEASE: the revision pinned in env/ASSETS_PINNED.json
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
files = list_repo_files(REPO, revision=a.revision); print("repo files:", files, flush=True)
for f in files:
    if f.endswith(".json") and f not in ("tekken.json",) or f in ("README.md",):
        hf_hub_download(REPO, f, revision=a.revision, local_dir=a.out); print("got", f, flush=True)
st = "model.safetensors" if "model.safetensors" in files else [f for f in files if f.endswith(".safetensors")][0]
url = hf_hub_url(REPO, st, revision=a.revision)
tok = get_token(); H = {"Authorization": f"Bearer {tok}"} if tok else {}
s = requests.Session()
def rng(lo, hi):  # inclusive
    r = s.get(url, headers={**H, "Range": f"bytes={lo}-{hi}"}, allow_redirects=True, timeout=600)
    assert r.status_code == 206, (r.status_code, r.text[:200]); return r.content
n = struct.unpack("<Q", rng(0, 7))[0]
hdr = json.loads(rng(8, 8 + n - 1)); base = 8 + n
keys = sorted(k for k in hdr if k.startswith("audio_tower."))
print(f"{st}: {len(hdr)-1} tensors, audio_tower {len(keys)}", flush=True)
spans = sorted((hdr[k]["data_offsets"][0], hdr[k]["data_offsets"][1], k) for k in keys)
lo, hi = spans[0][0], max(e for _, e, _ in spans)
covered = sum(e - b for b, e, _ in spans)
print(f"audio_tower bytes {covered/1e9:.3f} GB, span {lo}-{hi} ({(hi-lo)/1e9:.3f} GB)", flush=True)
# fetch the contiguous span if it is (nearly) all audio_tower, else per tensor
blobs = {}
if hi - lo <= covered * 1.05:
    CH = 256 << 20; buf = bytearray()
    for p in range(lo, hi, CH):
        buf += rng(base + p, base + min(hi, p + CH) - 1); print(f"  {len(buf)/1e9:.2f}/{(hi-lo)/1e9:.2f} GB", flush=True)
    for b, e, k in spans: blobs[k] = bytes(buf[b - lo:e - lo])
else:
    for b, e, k in spans: blobs[k] = rng(base + b, base + e - 1)
# write standalone safetensors
out_hdr, off = {"__metadata__": {"format": "pt", "source": f"{REPO}/{st} audio_tower.* only"}}, 0
for k in keys:
    out_hdr[k] = {"dtype": hdr[k]["dtype"], "shape": hdr[k]["shape"], "data_offsets": [off, off + len(blobs[k])]}; off += len(blobs[k])
hb = json.dumps(out_hdr, separators=(",", ":")).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
tmp = f"{a.out}/audio_tower.safetensors.tmp"
with open(tmp, "wb") as f:
    f.write(struct.pack("<Q", len(hb))); f.write(hb)
    for k in keys: f.write(blobs[k])
os.replace(tmp, f"{a.out}/audio_tower.safetensors")
json.dump({"repo": REPO, "file": st, "n_tensors": len(keys), "bytes": off}, open(f"{a.out}/audio_tower.source.json", "w"), indent=1)
print("wrote", f"{a.out}/audio_tower.safetensors", off / 1e9, "GB", flush=True)
