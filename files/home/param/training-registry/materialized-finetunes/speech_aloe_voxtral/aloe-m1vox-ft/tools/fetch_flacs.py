"""aloe-vox-ft: download the channel-A FLACs of ONE store (= one training node) from GCS to node-local scratch, verified.

List = PLAN.json conversations with store == K: every a_tracks entry (the partner track of each qualifying B track). Each object is fetched
AT THE MANIFEST GENERATION (gs://...?generation=G; a newer / deleted object fails loudly), written to <root>/<ses>/<seg>/SPEAKER_0N.flac via
a .part file, and verified: size == manifest bytes == GCS size, crc32c AND md5 of the local file == the GCS object's (google-cloud-storage
also validates crc32c during the download). Ledger <root>/_ledger/store<K>.tsv (rel, bytes, generation, crc32c, md5) + DONE_store<K>.json.
Already-present files are re-verified (size + crc32c + md5 vs the ledger / GCS metadata), not re-downloaded. Credentials: the key file by
path only (GOOGLE_APPLICATION_CREDENTIALS). Run niced / ionice'd as side work (fetch_flacs.sh).
usage: python -I fetch_flacs.py --store K --root /scratch/amartyavarma/aloe-voxall-ft/flac [--workers 8] [--only-eval] [--limit N]"""
import argparse
import base64
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import threading
import time

import google_crc32c

PLAN = "/data/home/amartyavarma/ppm/aloe-voxall-ft/plan/PLAN.json"
BUCKET = "aloe-processed-datasets"
PREFIX = "gs://aloe-processed-datasets/"


def hashes(path):
    c, m = google_crc32c.Checksum(), hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 << 20)
            if not b:
                break
            c.update(b); m.update(b)
    return base64.b64encode(c.digest()).decode(), base64.b64encode(m.digest()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=int, required=True); ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--only-eval", action="store_true")
    ap.add_argument("--stores", default="", help="comma list: fetch several stores' tracks (test nodes)")
    ap.add_argument("--limit", type=int, default=0, help="test: first N train conversations per store (+ every eval one)")
    a = ap.parse_args()
    P = json.load(open(PLAN))
    stores = [int(x) for x in a.stores.split(",")] if a.stores else [a.store]
    todo = []
    for k in stores:
        cs = [c for c in P["convs"] if c["store"] == k]
        if a.only_eval or a.limit:
            ev = [c for c in cs if c["split"] == "eval"]
            tr = [] if a.only_eval else [c for c in cs if c["split"] == "train"][:a.limit]
            cs = ev + tr
        for c in cs:
            todo += c["a_tracks"]
    assert len({t["rel"] for t in todo}) == len(todo), "an A track listed twice"
    from google.cloud import storage
    cli = storage.Client.from_service_account_json(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    bkt = cli.bucket(BUCKET)
    os.makedirs(f"{a.root}/_ledger", exist_ok=True)
    tag = f"store{a.store}" if not a.stores else "stores" + "-".join(map(str, stores))
    tag += "-eval" if a.only_eval else (f"-lim{a.limit}" if a.limit else "")
    ledger_p = f"{a.root}/_ledger/{tag}.tsv"
    done = {}
    if os.path.exists(ledger_p):
        for line in open(ledger_p):
            rel, nb, gen, crc, md5 = line.rstrip("\n").split("\t")
            done[rel] = (int(nb), int(gen), crc, md5)
    lock = threading.Lock()
    led = open(ledger_p, "a")
    st = {"n": 0, "bytes": 0, "fetched": 0, "reverified": 0}
    t0 = time.time()

    def one(t):
        rel, gen, nb = t["rel"], int(t["generation"]), int(t["bytes"])
        name = t["uri"][len(PREFIX):]
        assert t["uri"].startswith(PREFIX + "stage2-handover-20261008/wlt/") and name.endswith(rel), t["uri"]
        dst = f"{a.root}/{rel}"
        blob = bkt.blob(name, generation=gen)
        blob.reload()
        assert int(blob.generation) == gen and int(blob.size) == nb, (rel, blob.generation, gen, blob.size, nb)
        if os.path.exists(dst) and os.path.getsize(dst) == nb:
            crc, md5 = hashes(dst)
            if crc == blob.crc32c and md5 == blob.md5_hash:
                return rel, nb, gen, crc, md5, "reverified"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        part = dst + ".part"
        for att in range(5):
            try:
                blob.download_to_filename(part, checksum="crc32c")
                break
            except Exception as e:  # noqa: BLE001
                if att == 4:
                    raise
                print(f"[retry {att + 1}] {rel}: {e!r}", flush=True); time.sleep(5 * (att + 1))
        assert os.path.getsize(part) == nb, (rel, os.path.getsize(part), nb)
        crc, md5 = hashes(part)
        assert crc == blob.crc32c and md5 == blob.md5_hash, (rel, crc, blob.crc32c, md5, blob.md5_hash)
        os.replace(part, dst)
        return rel, nb, gen, crc, md5, "fetched"

    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, t) for t in todo]
        for f in cf.as_completed(futs):
            rel, nb, gen, crc, md5, how = f.result()
            with lock:
                led.write(f"{rel}\t{nb}\t{gen}\t{crc}\t{md5}\n"); led.flush()
                st["n"] += 1; st["bytes"] += nb; st[how] += 1
                if st["n"] % 100 == 0 or st["n"] == len(todo):
                    el = time.time() - t0
                    print(f"[{tag}] {st['n']}/{len(todo)} files {st['bytes'] / 1e9:.1f} GB ({st['bytes'] / 1e6 / max(el, 1e-9):.0f} MB/s) "
                          f"fetched {st['fetched']} reverified {st['reverified']} {el:.0f}s", flush=True)
    led.close()
    # final audit: every listed file present with the verified size
    bad = [t["rel"] for t in todo if not (os.path.exists(f"{a.root}/{t['rel']}") and os.path.getsize(f"{a.root}/{t['rel']}") == int(t["bytes"]))]
    assert not bad, ("missing / wrong-size files", bad[:5], len(bad))
    json.dump({"tag": tag, "stores": stores, "files": len(todo), "bytes": sum(int(t["bytes"]) for t in todo), "verified": "size+crc32c+md5 vs GCS @ manifest generation",
               "time_s": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(f"{a.root}/_ledger/DONE_{tag}.json", "w"), indent=1)
    print(f"[{tag}] DONE {len(todo)} files {st['bytes'] / 1e9:.1f} GB verified in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
