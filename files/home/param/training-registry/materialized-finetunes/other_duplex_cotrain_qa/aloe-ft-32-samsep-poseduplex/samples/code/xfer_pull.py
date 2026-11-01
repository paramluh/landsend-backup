"""pdsamples host-to-host pull, DEST side: wait for the 8 pose hosts' <host>.serve + <host>.sha256.json, fetch every file (rank files from
each host, rep.pt / meta.json / COMPLETE from rank 0's host) in parallel into <dest>.part, sha256 each == the source manifest, rename to <dest>.
usage: python -I xfer_pull.py <xfer dir> <dest> <host,host,...>"""
import hashlib, json, os, shutil, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
X, DEST, HOSTS = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
if os.path.isfile(f"{DEST}/COMPLETE"):
    print(DEST, "already COMPLETE"); sys.exit(0)
t0 = time.time()
while not all(os.path.isfile(f"{X}/{h}.serve") for h in HOSTS):
    time.sleep(5)
print(f"all {len(HOSTS)} sources serving after {time.time() - t0:.0f}s", flush=True)
while not all(os.path.isfile(f"{X}/{h}.sha256.json") for h in HOSTS):
    time.sleep(3)
src = {h: open(f"{X}/{h}.serve").read().strip() for h in HOSTS}
jobs, seen = [], {}
for h in HOSTS:
    man = json.load(open(f"{X}/{h}.sha256.json"))
    for n, (sz, sh) in man.items():
        if n in seen:
            assert seen[n][1] == sh, ("same file differs between hosts", n); continue
        seen[n] = (sz, sh); jobs.append((h, n, sz, sh))
ranks = sorted(int(n[4:-3]) for n in seen if n.startswith("rank") and n.endswith(".pt"))
assert ranks == list(range(len(ranks))) and all(n in seen for n in ("rep.pt", "meta.json", "COMPLETE")), ("export incomplete", len(ranks))
part = DEST + ".part"; os.makedirs(part, exist_ok=True)
def one(j):
    h, n, sz, sh = j
    for att in range(4):
        try:
            hh = hashlib.sha256()
            with urllib.request.urlopen(f"http://{src[h]}/{n}", timeout=120) as r, open(f"{part}/{n}", "wb") as f:
                for b in iter(lambda: r.read(1 << 23), b""):
                    f.write(b); hh.update(b)
            assert os.path.getsize(f"{part}/{n}") == sz and hh.hexdigest() == sh, ("size / sha256 != source manifest", n)
            return sz
        except Exception as ex:                                       # noqa: BLE001
            print(f"retry {n} from {h}: {ex}", flush=True); time.sleep(3)
    raise RuntimeError(f"failed {n}")
t1 = time.time()
with ThreadPoolExecutor(24) as ex:
    tot = sum(ex.map(one, sorted(jobs, key=lambda j: -j[2])))
json.dump({"sources": src, "files": len(jobs), "bytes": tot, "sha256": "== source manifests", "time_s": round(time.time() - t1)}, open(f"{part}/PDSAMPLES_PULL.json", "w"))
os.rename(part, DEST)
print(f"pulled {len(jobs)} files {tot / 1e9:.1f} GB in {time.time() - t1:.0f}s ({tot / 1e9 / max(1, time.time() - t1):.2f} GB/s), sha256 == source manifests -> {DEST}", flush=True)
