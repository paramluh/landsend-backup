"""aloe-v2vox-ft: verify / complete ONE training node's channel-A audio for the v2 plan (run on that node; 1 GPU only if a Mimi decode is needed).
 1. the 4 Model-1 stores at /scratch/amartyavarma/aloe-ft/stores/m1nK: digest == Model 1's (re-copied from /data if not; = stage_m1vox).
 2. the v2 plan (data_aloe_v2.make_plan(64), first chunks only) -> every item node i's 8 ranks read (train incl. the dummy fallback item, eval)
    -> its A track from m1vox's M1VOX_INDEX.json (v2's rows are a subset of m1vox's rows; asserted: every item is in the index).
 3. 'gcs' tracks at <root>/<rel>: a file already there (m1vox staging) is re-hashed and its crc32c + md5 must equal the m1vox ledger's
    (= the GCS object's at the manifest generation); absent / mismatching files are downloaded AT THE MANIFEST GENERATION and verified
    (size == manifest bytes == GCS size, crc32c AND md5 == the GCS object's) = stage_m1vox's rule.
 4. 'mimidec' rows (purged conversations): present in the m1vox ledger -> kept; else decoded on this node's GPU exactly as stage_m1vox.
 Ledger <root>/_ledger/v2vox_node<i>.tsv + DONE_v2vox_node<i>.json.
usage: python -I stage_v2vox.py --node I --root /scratch/amartyavarma/aloe-m1vox-ft/flac [--workers 32]"""
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

V = "/data/home/amartyavarma/ppm/aloe-v2vox-ft"
sys.path.insert(0, f"{V}/code"); sys.path.insert(0, f"{V}/code/lib")
BUCKET, PREFIX = "aloe-processed-datasets", "gs://aloe-processed-datasets/"
WORLD, LW = 64, 8


def hashes(path):
    c, m = google_crc32c.Checksum(), hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 << 20)
            if not b:
                break
            c.update(b); m.update(b)
    return base64.b64encode(c.digest()).decode(), base64.b64encode(m.digest()).decode()


def check_stores():
    import data_aloe_v2vox as A
    import s3rows as F
    for k in range(4):
        src, dst = A.SRC_STORES[k], A.STORES[k]
        ok = False
        if os.path.isfile(f"{dst}/NODE.json"):
            try:
                ok = F.load_node(dst)["digest"] == A.DIGESTS[k]
            except Exception:  # noqa: BLE001
                ok = False
        if not ok:
            shutil.rmtree(dst + ".tmp", ignore_errors=True); shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst + ".tmp"); os.replace(dst + ".tmp", dst)
        a, b = F.load_node(src), F.load_node(dst)
        assert a["digest"] == b["digest"] == A.DIGESTS[k] and (a["length"] == b["length"]).all(), k
        for f in sorted(os.listdir(src)):
            if os.path.isfile(f"{src}/{f}"):
                assert os.path.getsize(f"{src}/{f}") == os.path.getsize(f"{dst}/{f}"), (k, f)
        print(f"[stores] m1n{k} {'re-copied' if not ok else 'already staged'} + verified digest {b['digest']} rows {len(b['length'])}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, required=True); ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    t0 = time.time()
    check_stores()
    import data_aloe_v2vox as A
    PL = A.make_plan(WORLD)
    dg = hashlib.sha256(json.dumps(PL["items"]).encode()).hexdigest()[:16]
    items = A.node_items(PL, a.node, WORLD, LW)
    idx = A.vox_index()
    need_g, need_m = {}, {}
    for k, r, sw in items:
        aside = "assistant" if int(sw) else "user"
        e = idx[int(k)][int(r)]
        assert e["row"] == int(r), (k, r)
        t = e["A"][aside]
        if t["kind"] == "gcs":
            need_g[t["flac"]] = {"rel": t["flac"], "bytes": int(t["flac_bytes"]), "generation": int(t["generation"]), "uri": t["uri"]}
        else:
            assert t["kind"] == "mimidec", t
            need_m[t["flac"]] = {"rel": t["flac"], "store": int(k), "row": int(r), "side": aside, "n_frames": int(t["n_frames"])}
    print(f"[plan] v2 plan {dg}: node {a.node} reads {len(items)} items -> {len(need_g)} gcs A tracks + {len(need_m)} mimidec rows", flush=True)
    led_old = {}
    lf = f"{a.root}/_ledger/node{a.node}.tsv"
    if os.path.exists(lf):
        for ln in open(lf):
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 6:
                led_old[p[0]] = p
    from google.cloud import storage
    cli = storage.Client.from_service_account_json(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    bkt = cli.bucket(BUCKET)
    st = {"present_ledger_ok": 0, "present_gcs_ok": 0, "fetched": 0, "bytes": 0}

    def one(t):
        rel, gen, nb = t["rel"], int(t["generation"]), int(t["bytes"])
        name = t["uri"][len(PREFIX):]
        assert t["uri"].startswith(PREFIX + "stage2-handover-20261008/wlt/") and name.endswith(rel), t["uri"]
        dst = f"{a.root}/{rel}"
        L = led_old.get(rel)
        if os.path.exists(dst) and os.path.getsize(dst) == nb:
            crc, md5 = hashes(dst)
            if L is not None and L[2] != "mimidec" and int(L[1]) == nb and int(L[2]) == gen and L[3] == crc and L[4] == md5:
                return rel, nb, gen, crc, md5, "present_ledger_ok"
            blob = bkt.blob(name, generation=gen); blob.reload()
            assert int(blob.generation) == gen and int(blob.size) == nb, (rel, blob.generation, gen, blob.size, nb)
            if crc == blob.crc32c and md5 == blob.md5_hash:
                return rel, nb, gen, crc, md5, "present_gcs_ok"
        blob = bkt.blob(name, generation=gen); blob.reload()
        assert int(blob.generation) == gen and int(blob.size) == nb, (rel, blob.generation, gen, blob.size, nb)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
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

    os.makedirs(f"{a.root}/_ledger", exist_ok=True)
    led = open(f"{a.root}/_ledger/v2vox_node{a.node}.tsv", "w")
    gl = list(need_g.values())
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for j, f in enumerate(cf.as_completed([ex.submit(one, t) for t in gl])):
            rel, nb, gen, crc, md5, how = f.result()
            led.write(f"{rel}\t{nb}\t{gen}\t{crc}\t{md5}\t{how}\n")
            st[how] += 1; st["bytes"] += nb
            if (j + 1) % 500 == 0 or j + 1 == len(gl):
                print(f"[gcs] {j + 1}/{len(gl)} verified ({st}) {time.time() - t0:.0f}s", flush=True)
    todo = []
    for t in need_m.values():
        dst = f"{a.root}/{t['rel']}"
        L = led_old.get(t["rel"])
        if L is not None and L[2] == "mimidec" and L[3] == f"{t['store']}/{t['row']}/{t['side']}" and os.path.exists(dst) \
                and os.path.getsize(dst) == int(L[1]) > 0:
            led.write(f"{t['rel']}\t{os.path.getsize(dst)}\tmimidec\t{t['store']}/{t['row']}/{t['side']}\t-\tpresent_ledger_ok\n")
        else:
            todo.append(t)
    clipped = 0
    if todo:
        import soundfile as sf
        import torch
        from transformers import MimiModel
        import s3rows as F
        dev = torch.device("cuda", 0)
        mimi = MimiModel.from_pretrained("/data/home/amartyavarma/models/mimi").to(dev).eval()
        rows_ = {}
        for t in todo:
            k, r, side, n = int(t["store"]), int(t["row"]), t["side"], int(t["n_frames"])
            if k not in rows_:
                rows_[k] = F.Rows(F.load_node(A.STORES[k]))
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
    print(f"[mimidec] {len(need_m)} rows: {len(need_m) - len(todo)} present (m1vox ledger), {len(todo)} decoded on GPU (clipped samples {clipped})", flush=True)
    bad = [t["rel"] for t in gl if not (os.path.exists(f"{a.root}/{t['rel']}") and os.path.getsize(f"{a.root}/{t['rel']}") == int(t["bytes"]))]
    bad += [t["rel"] for t in need_m.values() if not (os.path.exists(f"{a.root}/{t['rel']}") and os.path.getsize(f"{a.root}/{t['rel']}") > 0)]
    assert not bad, ("missing files", bad[:5], len(bad))
    json.dump({"node": a.node, "plan_digest": dg, "items": len(items), "gcs_files": len(gl), "gcs_bytes": st["bytes"], "mimidec": len(need_m),
               "mimidec_decoded": len(todo), "stats": st, "clipped": clipped,
               "verified": "gcs: crc32c + md5 == m1vox ledger (= GCS @ manifest generation) or == GCS object; fetched: size + crc32c + md5 vs GCS",
               "time_s": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(f"{a.root}/_ledger/DONE_v2vox_node{a.node}.json", "w"), indent=1)
    print(f"[node {a.node}] DONE items {len(items)} gcs {len(gl)} ({st}) mimidec {len(need_m)} (decoded {len(todo)}) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
