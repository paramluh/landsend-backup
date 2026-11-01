"""aloe-v2vox-raw: stage ONE physical training node for the LOGICAL nodes it emulates (22420 ran 8 nodes x 8 ranks = 64 logical ranks; this run
keeps those 64 logical ranks and runs logical node l's 8 ranks on physical node P's 8 GPUs by gradient accumulation: ALOE_LNODES).
= tools/stage_v2vox.py (job 22407) for each logical node l in --lnodes, on a fresh node:
 1. the 4 Model-1 stores at /scratch/amartyavarma/aloe-ft/stores/m1nK: digest == Model 1's (copied from /data if not) (= stage_v2vox).
 2. the v2 plan (data_aloe_v2.make_plan(64)) -> every item logical node l's 8 ranks read (train incl. the dummy fallback item, eval) -> its A
    track (m1vox's M1VOX_INDEX.json); union over --lnodes.
 3. 'gcs' tracks at <root>/<rel>: downloaded AT THE MANIFEST GENERATION; size == manifest bytes == GCS size, crc32c AND md5 == the GCS
    object's (a file already present is re-hashed and must match the GCS object, else re-downloaded) (= stage_v2vox's rule).
 4. 'mimidec' tracks: copied from the run's central decode (tools/build_rawgain_mimidec.py, mimidec/MANIFEST.json: size + md5 verified).
 Ledgers <root>/_ledger/v2vox_node<l>.tsv + DONE_v2vox_node<l>.json per logical node l (the trainer checks them for its logical nodes).
SHARED COPY (10-17): the A tracks of ALL 8 logical nodes are staged ONCE to /data/home/amartyavarma/ppm/aloe-v2vox-raw/flac (238 GB) and every
training node's ALOE_FLAC_ROOT (/scratch/amartyavarma/aloe-m1vox-ft/flac, = 22420's path) is a symlink to it (the spare nodes' /scratch is
nearly full; any 2-3 nodes can then run any logical nodes). The 4 stores are copied node-local (--stores-only, training pre-flight).
usage: python -I stage_raw.py --lnodes 0,1,2,3,4,5,6,7 --root /data/home/amartyavarma/ppm/aloe-v2vox-raw/flac [--workers 64] [--min-free-gb 700]
       python -I stage_raw.py --lnodes '' --root X --stores-only"""
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

R = "/data/home/amartyavarma/ppm/aloe-v2vox-raw"
sys.path.insert(0, f"{R}/code"); sys.path.insert(0, f"{R}/code/lib")
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
        if ok:
            ok = all(os.path.getsize(f"{src}/{f}") == os.path.getsize(f"{dst}/{f}") for f in os.listdir(src) if os.path.isfile(f"{src}/{f}"))
        if not ok:
            shutil.rmtree(dst + ".tmp", ignore_errors=True); shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst + ".tmp"); os.replace(dst + ".tmp", dst)
        a, b = F.load_node(src), F.load_node(dst)
        assert a["digest"] == b["digest"] == A.DIGESTS[k] and (a["length"] == b["length"]).all(), k
        for f in sorted(os.listdir(src)):
            if os.path.isfile(f"{src}/{f}"):
                assert os.path.getsize(f"{src}/{f}") == os.path.getsize(f"{dst}/{f}"), (k, f)
        print(f"[stores] m1n{k} {'copied' if not ok else 'already staged'} + verified digest {b['digest']} rows {len(b['length'])}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lnodes", required=True); ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=32); ap.add_argument("--mimidec-src", default=f"{R}/mimidec")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--stores-only", action="store_true", help="only stage + verify the 4 stores on this node (training pre-flight)")
    a = ap.parse_args()
    lnodes = [int(x) for x in a.lnodes.split(",") if x != ""]
    t0 = time.time()
    check_stores()
    if a.stores_only:
        return 0
    import data_aloe_v2vox as A
    PL = A.make_plan(WORLD)
    dg = hashlib.sha256(json.dumps(PL["items"]).encode()).hexdigest()[:16]
    assert dg == "ba2d70729ea437ab", dg
    idx = A.vox_index()
    dman = json.load(open(f"{a.mimidec_src}/MANIFEST.json"))["files"]
    per = {}
    need_g, need_m = {}, {}
    for l in lnodes:
        items = A.node_items(PL, l, WORLD, LW)
        g_, m_ = set(), set()
        for k, r, sw in items:
            aside = "assistant" if int(sw) else "user"
            e = idx[int(k)][int(r)]
            assert e["row"] == int(r), (k, r)
            t = e["A"][aside]
            if t["kind"] == "gcs":
                need_g[t["flac"]] = {"rel": t["flac"], "bytes": int(t["flac_bytes"]), "generation": int(t["generation"]), "uri": t["uri"]}
                g_.add(t["flac"])
            else:
                assert t["kind"] == "mimidec", t
                d = dman[t["flac"]]
                assert (d["store"], d["row"], d["side"]) == (int(k), int(r), aside), (t["flac"], d)
                need_m[t["flac"]] = {"rel": t["flac"], **d}
                m_.add(t["flac"])
        per[l] = {"items": len(items), "gcs": sorted(g_), "mimidec": sorted(m_)}
        print(f"[plan] v2 plan {dg}: logical node {l} reads {len(items)} items -> {len(g_)} gcs A tracks + {len(m_)} mimidec", flush=True)
    os.makedirs(a.root, exist_ok=True)
    gb_need = sum(t["bytes"] for t in need_g.values() if not (os.path.exists(f"{a.root}/{t['rel']}") and os.path.getsize(f"{a.root}/{t['rel']}") == t["bytes"])) / 1e9
    free = shutil.disk_usage(a.root).free / 1e9
    print(f"[disk] union {len(need_g)} gcs ({sum(t['bytes'] for t in need_g.values()) / 1e9:.1f} GB, {gb_need:.1f} GB to fetch) + {len(need_m)} mimidec; "
          f"/scratch free {free:.0f} GB", flush=True)
    assert free - gb_need >= a.min_free_gb, ("not enough /scratch for the A tracks", free, gb_need)
    from google.cloud import storage
    cli = storage.Client.from_service_account_json(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    bkt = cli.bucket(BUCKET)
    st = {"present_gcs_ok": 0, "fetched": 0, "bytes": 0}

    def one(t):
        rel, gen, nb = t["rel"], int(t["generation"]), int(t["bytes"])
        name = t["uri"][len(PREFIX):]
        assert t["uri"].startswith(PREFIX + "stage2-handover-20261008/wlt/") and name.endswith(rel), t["uri"]
        dst = f"{a.root}/{rel}"
        blob = bkt.blob(name, generation=gen); blob.reload()
        assert int(blob.generation) == gen and int(blob.size) == nb, (rel, blob.generation, gen, blob.size, nb)
        if os.path.exists(dst) and os.path.getsize(dst) == nb:
            crc, md5 = hashes(dst)
            if crc == blob.crc32c and md5 == blob.md5_hash:
                return rel, nb, gen, crc, md5, "present_gcs_ok"
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
    rec = {}
    gl = list(need_g.values())
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for j, f in enumerate(cf.as_completed([ex.submit(one, t) for t in gl])):
            rel, nb, gen, crc, md5, how = f.result()
            rec[rel] = f"{rel}\t{nb}\t{gen}\t{crc}\t{md5}\t{how}\n"
            st[how] += 1; st["bytes"] += nb
            if (j + 1) % 500 == 0 or j + 1 == len(gl):
                print(f"[gcs] {j + 1}/{len(gl)} verified ({st}) {time.time() - t0:.0f}s", flush=True)
    for t in need_m.values():
        src, dst = f"{a.mimidec_src}/{t['rel']}", f"{a.root}/{t['rel']}"
        if not (os.path.exists(dst) and os.path.getsize(dst) == t["bytes"] and hashes(dst)[1] == t["md5"]):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst + ".part"); os.replace(dst + ".part", dst)
        assert os.path.getsize(dst) == t["bytes"] and hashes(dst)[1] == t["md5"], ("mimidec copy differs from the central decode", t["rel"])
        rec[t["rel"]] = f"{t['rel']}\t{t['bytes']}\tmimidec\t{t['store']}/{t['row']}/{t['side']}\t{t['md5']}\tcopied_md5_ok\n"
    bad = [t["rel"] for t in gl if not (os.path.exists(f"{a.root}/{t['rel']}") and os.path.getsize(f"{a.root}/{t['rel']}") == int(t["bytes"]))]
    assert not bad, ("missing files", bad[:5], len(bad))
    for l in lnodes:
        with open(f"{a.root}/_ledger/v2vox_node{l}.tsv", "w") as led:
            for rel in per[l]["gcs"] + per[l]["mimidec"]:
                led.write(rec[rel])
        json.dump({"node": l, "plan_digest": dg, "items": per[l]["items"], "gcs_files": len(per[l]["gcs"]),
                   "gcs_bytes": sum(need_g[x]["bytes"] for x in per[l]["gcs"]), "mimidec": len(per[l]["mimidec"]), "mimidec_decoded": 0,
                   "stats": st, "physical_host": os.uname().nodename.split(".")[0], "lnodes_on_host": lnodes,
                   "verified": "gcs: size + crc32c + md5 == GCS object @ manifest generation; mimidec: md5 == central decode (build_rawgain_mimidec.py)",
                   "time_s": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  open(f"{a.root}/_ledger/DONE_v2vox_node{l}.json", "w"), indent=1)
    print(f"[host {os.uname().nodename}] DONE logical nodes {lnodes}: gcs {len(gl)} ({st}) mimidec {len(need_m)} in {time.time() - t0:.0f}s; "
          f"/scratch free {shutil.disk_usage(a.root).free / 1e9:.0f} GB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
