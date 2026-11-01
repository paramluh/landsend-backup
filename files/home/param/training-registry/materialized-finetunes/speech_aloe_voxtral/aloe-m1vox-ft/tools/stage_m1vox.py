"""aloe-m1vox-ft: stage ONE training node (run on that node, 1 GPU for the Mimi decodes).
 1. 21980's 4 Model-1 stores -> /scratch/amartyavarma/aloe-ft/stores/m1nK (= 21980's staging path; copied from /data unless an identical copy
    is there; digest + file sizes asserted, = aloe-ft/stage.sbatch's checks).
 2. plan/NEEDS.json[node] 'gcs' tracks -> <root>/<ses>/<seg>/SPEAKER_0N.flac: hard-linked from the aloe-voxall-ft root when present, else
    downloaded AT THE MANIFEST GENERATION; EVERY file verified: size == manifest bytes == GCS size, crc32c AND md5 == the GCS object's.
 3. 'mimidec' rows (purged conversations, build_m1vox_index doc): the row's A-side Mimi codes (staged store) -> transformers MimiModel decode
    on this node's GPU -> 24 kHz mono FLAC PCM_24 (clipped to [-1, 1), clipped-sample count logged) at <root>/_mimidec/...
 Ledger <root>/_ledger/node<i>.tsv + DONE_node<i>.json.
usage: python -I stage_m1vox.py --node I --root /scratch/amartyavarma/aloe-m1vox-ft/flac [--workers 64]"""
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

V = "/data/home/amartyavarma/ppm/aloe-m1vox-ft"
sys.path.insert(0, f"{V}/code"); sys.path.insert(0, f"{V}/code/lib")
OLD = "/scratch/amartyavarma/aloe-voxall-ft/flac"
BUCKET, PREFIX = "aloe-processed-datasets", "gs://aloe-processed-datasets/"


def hashes(path):
    c, m = google_crc32c.Checksum(), hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 << 20)
            if not b:
                break
            c.update(b); m.update(b)
    return base64.b64encode(c.digest()).decode(), base64.b64encode(m.digest()).decode()


def stage_stores():
    import data_aloe as AL
    import s3rows as F
    t0 = time.time()
    for k in range(4):
        src, dst = AL.SRC_STORES[k], AL.STORES[k]
        ok = False
        if os.path.isfile(f"{dst}/NODE.json"):
            try:
                ok = F.load_node(dst)["digest"] == AL.DIGESTS[k]
            except Exception:  # noqa: BLE001
                ok = False
        if not ok:
            shutil.rmtree(dst + ".tmp", ignore_errors=True); shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst + ".tmp"); os.replace(dst + ".tmp", dst)
        a, b = F.load_node(src), F.load_node(dst)
        assert a["digest"] == b["digest"] == AL.DIGESTS[k] and (a["length"] == b["length"]).all(), k
        for suf in ("codes_user.npy", "codes_assistant.npy", "words.arrow", "persona.arrow"):
            f = "rows-00000-of-00001." + suf
            assert os.path.getsize(f"{src}/{f}") == os.path.getsize(f"{dst}/{f}"), (k, f)
        print(f"[stores] m1n{k} {'copied' if not ok else 'already staged'} + verified digest {b['digest']} rows {len(b['length'])}", flush=True)
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
        if not (os.path.exists(dst) and os.path.getsize(dst) == nb) and os.path.exists(f"{OLD}/{rel}") and os.path.getsize(f"{OLD}/{rel}") == nb:
            if os.path.exists(dst):
                os.remove(dst)
            os.link(f"{OLD}/{rel}", dst); how = "linked"
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
    # mimidec rows
    import torch
    from transformers import MimiModel
    import soundfile as sf
    import data_aloe as AL
    dev = torch.device("cuda", 0)
    mimi = MimiModel.from_pretrained("/data/home/amartyavarma/models/mimi").to(dev).eval()
    import s3rows as F
    rows_ = {}
    clipped = 0
    for t in mdec:
        k, r, side, n = int(t["store"]), int(t["row"]), t["side"], int(t["n_frames"])
        if k not in rows_:
            rows_[k] = F.Rows(F.load_node(AL.STORES[k]))
        row = rows_[k][r]
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
               "verified": "gcs: size + crc32c + md5 vs GCS @ manifest generation", "time_s": round(time.time() - t0, 1),
               "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(f"{a.root}/_ledger/DONE_node{a.node}.json", "w"), indent=1)
    print(f"[node {a.node}] DONE gcs {len(gcs)} ({st}) mimidec {len(mdec)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
