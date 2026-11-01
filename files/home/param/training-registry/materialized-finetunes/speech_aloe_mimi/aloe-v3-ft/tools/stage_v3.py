"""aloe-v3-ft: stage ONE training node (run on that node, 1 GPU for the Mimi decodes) = aloe-m1vox-ft's stage_m1vox.py for the v3 data:
 1. the 4 v3 whole-call stores (/data/home/amartyavarma/ppm/aloe-v3-ft/stores/v3nK) -> /scratch/amartyavarma/aloe-v3-ft/stores/v3nK (copied
    unless an identical copy is there; digest + every file size asserted against plan/STORES.json).
 2. plan/NEEDS.json[node] 'gcs' tracks (the partner WLT FLACs of the whole calls this node's ranks read) -> <root>/<ses>/<seg>/SPEAKER_0N.flac:
    hard-linked from an existing node-local copy (aloe-m1vox-ft / aloe-voxall-ft roots) when present, else downloaded AT THE MANIFEST
    GENERATION; EVERY file verified: size == manifest bytes == GCS size, crc32c AND md5 == the GCS object's.
 3. 'mimidec' tracks (purged calls: no FLAC left anywhere): the v3 row's A-side Mimi codes (frames 0..n, up to 13,500) -> transformers
    MimiModel decode on this node's GPU -> 24 kHz mono FLAC PCM_24 at <root>/_mimidec/<ses>/<seg>/c00.<spk>.flac (= m1vox's rule).
 Ledger <root>/_ledger/node<i>.tsv + DONE_node<i>.json.
usage: python -I stage_v3.py --node I --root /scratch/amartyavarma/aloe-v3-ft/flac [--workers 64]"""
import argparse
import base64
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import sys
import time

import google_crc32c
import numpy as np

V = "/data/home/amartyavarma/ppm/aloe-v3-ft"
os.environ.setdefault("S3_MIN_FRAMES", "125")
sys.path.insert(0, f"{V}/code/lib")
import s3rows as F  # noqa: E402

OLD = ["/scratch/amartyavarma/aloe-m1vox-ft/flac", "/scratch/amartyavarma/aloe-voxall-ft/flac"]
BUCKET, PREFIX = "aloe-processed-datasets", "gs://aloe-processed-datasets/"
SRC = f"{V}/stores"
DST = "/scratch/amartyavarma/aloe-v3-ft/stores"


def hashes(path):
    c, m = google_crc32c.Checksum(), hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 << 20)
            if not b:
                break
            c.update(b); m.update(b)
    return base64.b64encode(c.digest()).decode(), base64.b64encode(m.digest()).decode()


def store_ok(k, sj):
    d = f"{DST}/v3n{k}"
    try:
        n = F.load_node(d)
        return n["digest"] == sj["stores"][k]["digest"] and all(os.path.getsize(f"{d}/{f}") == b for f, b in sj["stores"][k]["files"].items())
    except Exception:  # noqa: BLE001
        return False


def stage_stores():
    sj = json.load(open(f"{V}/plan/STORES.json"))
    t0 = time.time()
    os.makedirs(DST, exist_ok=True)
    for k in range(4):
        how = "already staged"
        if not store_ok(k, sj):
            shutil.rmtree(f"{DST}/v3n{k}.tmp", ignore_errors=True); shutil.rmtree(f"{DST}/v3n{k}", ignore_errors=True)
            shutil.copytree(f"{SRC}/v3n{k}", f"{DST}/v3n{k}.tmp"); os.replace(f"{DST}/v3n{k}.tmp", f"{DST}/v3n{k}")
            how = "copied"
        assert store_ok(k, sj), ("staged store != the build", k)
        print(f"[stores] v3n{k} {how} + verified digest {sj['stores'][k]['digest']} rows {sj['stores'][k]['rows']}", flush=True)
    print(f"[stores] done {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, required=True); ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    t0 = time.time()
    stage_stores()
    need = json.load(open(f"{V}/plan/NEEDS.json"))[str(a.node)]
    gcs = [t for t in need if t["kind"] == "gcs"]
    mdec = [t for t in need if t["kind"] == "mimidec"]
    os.makedirs(f"{a.root}/_ledger", exist_ok=True)
    from google.cloud import storage
    cli = storage.Client.from_service_account_json(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    bkt = cli.bucket(BUCKET)
    st = {"linked": 0, "fetched": 0, "present": 0, "bytes": 0}

    def one(t):
        rel, gen, nb = t["rel"], int(t["generation"]), int(t["bytes"])
        name = t["uri"][len(PREFIX):]
        assert t["uri"].startswith(PREFIX + "stage2-handover-20261008/wlt/") and name.endswith(rel), t["uri"]
        dst = f"{a.root}/{rel}"
        blob = bkt.blob(name, generation=gen)
        blob.reload()
        assert int(blob.generation) == gen and int(blob.size) == nb, (rel, blob.generation, gen, blob.size, nb)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        how = "present"
        if not (os.path.exists(dst) and os.path.getsize(dst) == nb):
            for old in OLD:
                if os.path.exists(f"{old}/{rel}") and os.path.getsize(f"{old}/{rel}") == nb:
                    if os.path.exists(dst):
                        os.remove(dst)
                    os.link(f"{old}/{rel}", dst); how = "linked"
                    break
        if os.path.exists(dst) and os.path.getsize(dst) == nb:
            crc, md5 = hashes(dst)
            if crc == blob.crc32c and md5 == blob.md5_hash:
                return rel, nb, gen, crc, md5, how
        part = dst + ".part"
        for att in range(5):
            try:
                blob.download_to_filename(part, checksum="crc32c"); break
            except Exception as e:  # noqa: BLE001
                if att == 4:
                    raise
                print(f"[retry {att + 1}] {rel}: {e!r}", flush=True); time.sleep(5 * (att + 1))
        assert os.path.getsize(part) == nb, (rel, os.path.getsize(part), nb)
        crc, md5 = hashes(part)
        assert crc == blob.crc32c and md5 == blob.md5_hash, (rel, crc, blob.crc32c, md5, blob.md5_hash)
        os.replace(part, dst)
        return rel, nb, gen, crc, md5, "fetched"

    led = open(f"{a.root}/_ledger/node{a.node}.tsv", "w")
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for j, f in enumerate(cf.as_completed([ex.submit(one, t) for t in gcs])):
            rel, nb, gen, crc, md5, how = f.result()
            led.write(f"{rel}\t{nb}\t{gen}\t{crc}\t{md5}\t{how}\n")
            st[how] += 1; st["bytes"] += nb
            if (j + 1) % 250 == 0 or j + 1 == len(gcs):
                print(f"[gcs] {j + 1}/{len(gcs)} verified ({st}) {time.time() - t0:.0f}s", flush=True)
    import torch
    from transformers import MimiModel
    import soundfile as sf
    dev = torch.device("cuda", 0)
    mimi = MimiModel.from_pretrained("/data/home/amartyavarma/models/mimi").to(dev).eval()
    rows_ = {}
    clipped = 0
    for t in mdec:
        k, r, side, n = int(t["store"]), int(t["row"]), t["side"], int(t["n_frames"])
        if k not in rows_:
            rows_[k] = F.Rows(F.load_node(f"{DST}/v3n{k}"))
        row = rows_[k][r]
        assert int(row["chunk_index"]) == 0 and int(row["n_frames"]) == n, (t, row["row_key"])
        c = np.asarray(row[f"codes_{side}"])[:, :n].astype(np.int64)
        assert c.shape == (32, n) and int(c.min()) >= 0, (t, c.shape)
        with torch.no_grad():
            w = mimi.decode(torch.as_tensor(c, device=dev)[None]).audio_values[0, 0].float().cpu().numpy()[:n * 1920]
        assert w.shape == (n * 1920,) and np.isfinite(w).all(), (t, w.shape)
        clipped += int((np.abs(w) >= 1.0).sum())
        dst = f"{a.root}/{t['rel']}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        sf.write(dst + ".part.flac", np.clip(w, -1.0, 1.0 - 2 ** -23), 24000, subtype="PCM_24", format="FLAC")
        os.replace(dst + ".part.flac", dst)
        led.write(f"{t['rel']}\t{os.path.getsize(dst)}\tmimidec\t{k}/{r}/{side}\t-\tdecoded\n")
    led.close()
    print(f"[mimidec] {len(mdec)} rows decoded on GPU (clipped samples {clipped})", flush=True)
    bad = [t["rel"] for t in gcs if not (os.path.exists(f"{a.root}/{t['rel']}") and os.path.getsize(f"{a.root}/{t['rel']}") == int(t["bytes"]))]
    bad += [t["rel"] for t in mdec if not os.path.exists(f"{a.root}/{t['rel']}")]
    assert not bad, ("missing files", bad[:5], len(bad))
    json.dump({"node": a.node, "gcs_files": len(gcs), "gcs_bytes": st["bytes"], "mimidec": len(mdec), "stats": st, "clipped": clipped,
               "verified": "gcs: size + crc32c + md5 vs GCS @ manifest generation", "needs_sha": hashlib.sha256(open(f"{V}/plan/NEEDS.json", "rb").read()).hexdigest()[:16],
               "time_s": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(f"{a.root}/_ledger/DONE_node{a.node}.json", "w"), indent=1)
    print(f"[node {a.node}] DONE gcs {len(gcs)} ({st}) mimidec {len(mdec)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
