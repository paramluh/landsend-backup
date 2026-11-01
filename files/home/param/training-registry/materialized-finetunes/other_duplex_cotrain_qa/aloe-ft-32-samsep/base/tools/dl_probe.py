"""probe HF download speed for one dcp-17850 rank file (xet) into a node-local dir; prints MB/s and sha256 vs HF lfs sha."""
import hashlib, os, sys, time
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from huggingface_hub import hf_hub_download, HfApi
repo, path, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
info = HfApi().get_paths_info(repo, [path])[0]
t0 = time.time()
p = hf_hub_download(repo, path, local_dir=out)
dt = time.time() - t0
sz = os.path.getsize(p)
h = hashlib.sha256()
with open(p, "rb") as f:
    while True:
        b = f.read(1 << 24)
        if not b: break
        h.update(b)
print(f"{path}: {sz/1e9:.2f} GB in {dt:.1f}s = {sz/1e6/dt:.0f} MB/s; sha256 match {h.hexdigest() == info.lfs.sha256}", flush=True)
