"""APPEND ingest step 4 (one task per host): verify THIS host's fetched files against the listing (= data/new-pose-llm-data-dl/verify.py):
every file present at its listed size; hf source: xet content hash == the bucket listing's xet_hash (hf_xet.hash_files); local source: sha256 ==
the source file's. (The parquet payload sha256 vs each release's complete.json / release.json pins is checked again by the scan, which
rejects a release on any mismatch.) Also removes stray *.part files. -> <W>/verify/<host>.json; exit 1 on any problem.
usage (voicebank venv, python -I):  src_verify.py <W> <DEST>"""
import hashlib
import json
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor

W, DEST = sys.argv[1], sys.argv[2]
host = socket.gethostname().split(".")[0]
SRC = json.load(open(f"{W}/LISTING.json"))["src"]
rows = [ln.rstrip("\n").split("\t") for ln in open(f"{W}/placement/{host}.tsv")]
t0 = time.time()
ok = [r for r in rows if os.path.isfile(f"{DEST}/{r[0]}") and os.path.getsize(f"{DEST}/{r[0]}") == int(r[1])]
bad_size = [r[0] for r in rows if not (os.path.isfile(f"{DEST}/{r[0]}") and os.path.getsize(f"{DEST}/{r[0]}") == int(r[1]))]
bad = []


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


if SRC["kind"] == "hf":
    import hf_xet

    def chk(c):
        return [r[0] for r, h in zip(c, hf_xet.hash_files([f"{DEST}/{r[0]}" for r in c])) if h.hash != r[2]]
else:
    def chk(c):
        return [r[0] for r in c if sha(f"{DEST}/{r[0]}") != sha(f"{SRC['root']}/{r[0]}")]
with ThreadPoolExecutor(int(os.environ.get("VERIFY_THREADS", "32"))) as ex:
    for b in ex.map(chk, [ok[i:i + 500] for i in range(0, len(ok), 500)]):
        bad.extend(b)
parts = 0
for dp, _, fn in os.walk(DEST):
    for f in fn:
        if f.endswith(".part"):
            os.remove(os.path.join(dp, f)); parts += 1
res = {"host": host, "kind": SRC["kind"], "files": len(rows), "size_ok": len(ok), "missing_or_wrong_size": len(bad_size), "hash_mismatch": len(bad),
       "bytes": sum(int(r[1]) for r in ok), "stray_parts_removed": parts, "sec": round(time.time() - t0, 1), "examples_bad": bad[:20],
       "examples_missing": bad_size[:20]}
os.makedirs(f"{W}/verify", exist_ok=True)
json.dump(res, open(f"{W}/verify/{host}.json", "w"))
print(json.dumps(res), flush=True)
sys.exit(0 if not bad and not bad_size else 1)
