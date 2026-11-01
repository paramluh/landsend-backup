"""trim a host's podcast A-side FLAC margin (keeps /scratch >= 30 GB free): delete OUR files of train stream positions >= CUT on THIS host,
drop them from the host manifest and the done log. Only paths under /scratch/amartyavarma/si20k-podcast-flac/ (scratch_guard checked).
usage: trim_host.py HOST CUT"""
import json
import os
import socket
import sys

sys.path.insert(0, "/data/home/amartyavarma/ppm")
from scratch_guard import assert_deletable  # noqa: E402

HOST, CUT = sys.argv[1], int(sys.argv[2])
assert socket.gethostname().startswith(HOST) and CUT >= 198
BASE = "/data/home/amartyavarma/ppm/si20k-vox-ppm/plan/podflac"
DEST = "/scratch/amartyavarma/si20k-podcast-flac/"
man = [json.loads(l) for l in open(f"{BASE}/manifest/{HOST}.jsonl")]
keep, drop = [], []
for m in man:
    (drop if m["kind"] == "train" and m["stream_pos"] >= CUT else keep).append(m)
freed = 0
for m in drop:
    p = m["path"]
    assert p.startswith(DEST) and p.endswith(".A.flac"), p
    assert_deletable(p)
    if os.path.exists(p):
        freed += os.path.getsize(p)
        os.remove(p)
dk = {m["row_key"] for m in drop}
dl = f"{DEST}.done.{HOST}.jsonl"
lines = [l for l in open(dl) if json.loads(l)["row_key"] not in dk]
open(dl + ".tmp", "w").writelines(lines)
os.replace(dl + ".tmp", dl)
with open(f"{BASE}/manifest/{HOST}.jsonl.tmp", "w") as f:
    for m in keep:
        f.write(json.dumps(m) + "\n")
os.replace(f"{BASE}/manifest/{HOST}.jsonl.tmp", f"{BASE}/manifest/{HOST}.jsonl")
os.system(f"cp {BASE}/manifest/{HOST}.jsonl {DEST}MANIFEST.{HOST}.jsonl")
sv = os.statvfs("/scratch")
print(HOST, "trimmed", len(drop), "rows", round(freed / 1e9, 1), "GB; kept", len(keep), "; free now", round(sv.f_bavail * sv.f_frsize / 1e9, 1), "GB")
