"""speaker FT init staging for the 2-NODE run (10-24, main: Ashley, 2 nodes x 8 GPUs = world 16, PREP ONLY).
Builds /scratch/amartyavarma/aloespk-init/export-249 on THIS host = 29345's final node-local export-249 (ppm-local-bf16-shards-v1, world 64),
only the source rank files this host's ranks read at world 16 (subset json from tools/subset_w16.py = init_subset_hosts.py's rule; host index =
position in PPM_HOSTS), + rep.pt + COMPLETE + the NEW meta.json (export meta + the exact fp32 vgain theta, speaker/init-xfer/meta.json written by
tools/init_meta.py on slinky-1 from dcp-249).
  - a needed rank file this host already holds (its 8 ranks of 29345) -> HARDLINK of the node-local export file (same inode, no space);
  - every other file -> download from OUR private HF backup InternalCan/ppm-aloe-samsep-vox17127-final (export-249/<file>, uploaded + VERIFY PASS by
    29345's HF watcher);
  - EVERY staged .pt file (hardlinks too) sha256 == the HF LFS sha256; rep.pt also == slinky-1's source sha256; meta.json sha == the xfer copy.
Built in <dest>.part, COMPLETE written last, renamed; refuses below the /scratch floor. Nothing outside <dest>(.part/.dl) is written or deleted.
usage (compute node): python -I stage_w16.py <subset json> <host index> <dest> [workers]"""
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, hf_hub_download

REPO, TAG = "InternalCan/ppm-aloe-samsep-vox17127-final", "export-249"
X = "/scratch/amartyavarma/aloe-ft-32-samsep/exports-vox17127/export-249"
XF = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/init-xfer"
FLOOR_GB = 30
sj, hi, dest = sys.argv[1], sys.argv[2], sys.argv[3]
NW = int(sys.argv[4]) if len(sys.argv) > 4 else 16
if os.path.isfile(f"{dest}/COMPLETE"):
    print(f"{dest} already COMPLETE"); sys.exit(0)
need = sorted(json.load(open(sj))["hosts"][hi])
assert 0 in need
src_sha = {ln.split()[1]: ln.split()[0] for ln in open(f"{XF}/SRC_SHA256_slinky1.txt")}
api = HfApi()
want = [f"{TAG}/rank{s}.pt" for s in need] + [f"{TAG}/rep.pt", f"{TAG}/COMPLETE"]
info = {p.path: p for p in api.get_paths_info(REPO, want)}
assert sorted(info) == sorted(want), ("files missing on HF", sorted(set(want) - set(info)))
part, dl = dest + ".part", dest + ".dl"
os.makedirs(part, exist_ok=True); os.makedirs(dl, exist_ok=True)
local = {f"rank{s}.pt" for s in need if os.path.isfile(f"{X}/rank{s}.pt") and os.path.isfile(f"{X}/NODE_COMPLETE")}
dl_bytes = sum(int(info[f"{TAG}/{n}"].size) for n in [f"rank{s}.pt" for s in need] + ["rep.pt"] if n not in local)
free = shutil.disk_usage(os.path.dirname(dest)).free / 1e9
assert free - dl_bytes / 1e9 >= FLOOR_GB, ("/scratch floor", free, dl_bytes / 1e9)
assert not local or os.stat(X).st_dev == os.stat(os.path.dirname(dest)).st_dev, "export and dest on different filesystems (no hardlinks)"
t0 = time.time()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def one(f):
    n = f.rsplit("/", 1)[1]
    lfs = getattr(info[f], "lfs", None)
    if n in local:
        if not os.path.exists(f"{part}/{n}"):
            os.link(f"{X}/{n}", f"{part}/{n}")
        how = "hardlink"
        assert os.stat(f"{part}/{n}").st_ino == os.stat(f"{X}/{n}").st_ino
    else:
        p = hf_hub_download(REPO, f, local_dir=dl)
        shutil.move(p, f"{part}/{n}")
        how = "hf"
    if lfs is not None:
        h = sha256(f"{part}/{n}")
        assert h == lfs.sha256, ("sha256 != HF LFS", f, how)
    else:
        assert os.path.getsize(f"{part}/{n}") == int(info[f].size), ("size != HF", f)
        h = sha256(f"{part}/{n}")
    if n in src_sha:
        assert h == src_sha[n], ("sha256 != slinky-1 source", n)
    return n, how, h


with ThreadPoolExecutor(NW) as ex:
    got = list(ex.map(one, [w for w in want if not w.endswith("/COMPLETE")]))
shutil.copyfile(f"{XF}/meta.json", f"{part}/meta.json")
m = json.load(open(f"{part}/meta.json"))
assert m["format"] == "ppm-local-bf16-shards-v1" and int(m["world"]) == 64 and int(m["step"]) == 249 and isinstance(m["vgain_theta_fp32"], float)
got.append(("meta.json", "xfer (export meta + fp32 theta)", sha256(f"{part}/meta.json")))
json.dump({"repo": REPO, "tag": TAG, "host_index": hi, "subset": sj, "sources": need, "files": [{"name": n, "how": w, "sha256": h} for n, w, h in got],
           "check": "every .pt sha256 == HF LFS sha256 (hardlinks included); rep.pt == slinky-1 source sha256", "vgain_theta_fp32": m["vgain_theta_fp32"],
           "time_s": round(time.time() - t0)}, open(f"{part}/STAGE.json", "w"), indent=1)
pc = hf_hub_download(REPO, f"{TAG}/COMPLETE", local_dir=dl)
assert open(pc).read().strip() == "249" and sha256(pc) == src_sha["COMPLETE"]
shutil.move(pc, f"{part}/COMPLETE")
shutil.rmtree(dl, ignore_errors=True)
os.rename(part, dest)
nl = sum(1 for _, w, _ in got if w == "hardlink")
print(f"staged {dest}: {len(got) + 1} files ({nl} hardlinks of the node-local export, {len(got) - nl - 1} from HF), "
      f"{sum(os.path.getsize(f'{dest}/{n}') for n, _, _ in got) / 1e9:.1f} GB, sha256 == HF LFS, {time.time() - t0:.0f}s; "
      f"theta {m['vgain_theta_fp32']!r}; /scratch free {shutil.disk_usage(os.path.dirname(dest)).free / 1e9:.0f} GB")
