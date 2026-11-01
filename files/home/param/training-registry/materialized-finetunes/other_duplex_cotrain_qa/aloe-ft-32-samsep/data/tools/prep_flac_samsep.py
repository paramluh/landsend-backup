"""aloe-ft-32-samsep: channel-A (Voxtral) audio bookkeeping for the sam_sep stores. NO audio is fetched or re-made: every A track the new
plan reads is already in the reference's shared, GCS-verified copy /data/home/amartyavarma/ppm/aloe-v2vox-raw/flac (the kept rows are a
subset of the reference's rows; A audio = the raw WLT FLAC, unchanged by the code swap).
Writes (all new files, nothing of the reference touched):
  <data>/plan/M1VOX_INDEX_samsep.json   = M1VOX_INDEX.json (aloe-m1vox-index-v1) for the NEW stores: kept rows only, keyed by the NEW node row,
                                          'row' renumbered, 'stores' / 'digests' = the new ones; every A entry byte-identical
  <data>/plan/RAWGAIN_samsep.json       = RAWGAIN.json entries of the tracks the new plan reads (unchanged), plan_digest = the new plan's
  <data>/flac/                          shared FLAC root for the new plan: <session> -> symlink into the reference copy, _mimidec -> idem,
                                          _ledger/DONE_v2vox_node<l>.json + v2vox_node<l>.tsv for l = 0..7 under the NEW plan digest (each
                                          track: size == index flac_bytes and listed in the reference ledgers with the same size / md5)
  <data>/plan/PLAN_samsep.json          new plan summary (items digest, steps, counts, eval)
usage: python -I prep_flac_samsep.py"""
import hashlib
import json
import os
import sys
import time

CODE = "/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973/code"
for p in (os.path.join(CODE, "lib"), CODE):
    sys.path.insert(0, p)
import data_aloe_v2 as A2  # noqa: E402
import data_aloe_v2vox as V  # noqa: E402

DATA = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/data"
NEW = [f"/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores/m1n{k}" for k in range(4)]
REF_FLAC = "/data/home/amartyavarma/ppm/aloe-v2vox-raw/flac"
REF_INDEX = "/data/home/amartyavarma/ppm/aloe-m1vox-ft/plan/M1VOX_INDEX.json"
REF_RAWGAIN = "/data/home/amartyavarma/ppm/aloe-v2vox-raw/plan/RAWGAIN.json"
WORLD, LW = 64, 8


def main():
    import pyarrow.parquet as pq
    t0 = time.time()
    rm = json.load(open(f"{DATA}/row_map.json"))
    newdg = [rm["new_digest"][f"m1n{k}"] for k in range(4)]
    j = json.load(open(REF_INDEX))
    assert j["digests"] == A2.DIGESTS
    rows = {}
    for k in range(4):
        m = rm["map"][f"m1n{k}"]["new_of_old"]
        keys = pq.read_table(f"{NEW[k]}/rows.parquet", columns=["row", "row_key"]).to_pydict()
        out = [None] * len(keys["row"])
        for e in j["rows"][str(k)]:
            nr = m.get(str(e["row"]))
            if nr is None:
                continue
            assert keys["row_key"][nr] == e["row_key"], (k, e["row"], nr)
            out[nr] = dict(e, row=nr)
        assert all(x is not None for x in out), k
        rows[str(k)] = out
    ix = {kk: v for kk, v in j.items() if kk not in ("stores", "digests", "rows")}
    ix.update(stores=NEW, digests=newdg, rows=rows, samsep={"derived_from": REF_INDEX, "rule": "kept rows only, row renumbered; A entries unchanged"})
    os.makedirs(f"{DATA}/plan", exist_ok=True)
    json.dump(ix, open(f"{DATA}/plan/M1VOX_INDEX_samsep.json", "w"))
    idx = {int(k): v for k, v in rows.items()}
    PL = A2.make_plan(WORLD, roots=NEW)
    dg = hashlib.sha256(json.dumps(PL["items"]).encode()).hexdigest()[:16]
    ev_dg = hashlib.sha256(json.dumps(PL["eval"]).encode()).hexdigest()[:16]
    print(f"[plan] new plan {dg}: {len(PL['items'])} items, {PL['steps']} steps, train rows {PL['n_train_rows']}, eval rows {PL['n_eval_rows']} "
          f"(eval digest {ev_dg}) n_first {PL['n_first']} n_all {PL['n_all']}", flush=True)
    led_ref = {}
    for l in range(8):
        for ln in open(f"{REF_FLAC}/_ledger/v2vox_node{l}.tsv"):
            f = ln.rstrip("\n").split("\t")
            led_ref[f[0]] = f
    rg = json.load(open(REF_RAWGAIN))
    assert rg["plan_digest"] == "ba2d70729ea437ab"
    root = f"{DATA}/flac"
    os.makedirs(f"{root}/_ledger", exist_ok=True)
    used, per = {}, {}
    for l in range(8):
        items = V.node_items(PL, l, WORLD, LW)
        tr = {}
        for k, r, sw in items:
            aside = "assistant" if int(sw) else "user"
            e = idx[int(k)][int(r)]
            assert e["row"] == int(r)
            t = e["A"][aside]
            path = f"{REF_FLAC}/{t['flac']}"
            ref = led_ref.get(t["flac"])
            assert ref is not None, ("track not in the reference ledgers", t["flac"])
            if t["kind"] == "gcs":
                assert os.path.getsize(path) == int(t["flac_bytes"]) == int(ref[1]) and int(ref[2]) == int(t["generation"]), (t["flac"], ref)
            else:
                assert t["kind"] == "mimidec" and os.path.getsize(path) == int(ref[1]), (t["flac"], ref)
            tr[t["flac"]] = t; used[t["flac"]] = t
        per[l] = (items, tr)
        with open(f"{root}/_ledger/v2vox_node{l}.tsv", "w") as f:
            for rel in sorted(tr):
                f.write("\t".join(led_ref[rel]) + "\n")
        gc = [x for x in tr.values() if x["kind"] == "gcs"]
        json.dump({"node": l, "plan_digest": dg, "items": len(items), "gcs_files": len(gc), "gcs_bytes": sum(int(x["flac_bytes"]) for x in gc),
                   "mimidec": len(tr) - len(gc), "mimidec_decoded": 0, "physical_host": "shared /data (symlinks into the reference copy)",
                   "verified": f"size == index flac_bytes == reference ledger (crc32c + md5 verified vs GCS by {REF_FLAC}/_ledger); generation equal",
                   "reference_root": REF_FLAC, "time_s": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  open(f"{root}/_ledger/DONE_v2vox_node{l}.json", "w"), indent=1)
        print(f"[ledger] logical node {l}: {len(items)} items -> {len(gc)} gcs + {len(tr) - len(gc)} mimidec tracks", flush=True)
    tops = sorted({rel.split("/")[0] for rel in used})
    for tname in tops:
        dst = f"{root}/{tname}"
        if os.path.islink(dst):
            assert os.readlink(dst) == f"{REF_FLAC}/{tname}"
        else:
            os.symlink(f"{REF_FLAC}/{tname}", dst)
    for rel, t in used.items():
        assert os.path.getsize(f"{root}/{rel}") == os.path.getsize(f"{REF_FLAC}/{rel}")
    miss = [rel for rel in used if rel not in rg["rows"]]
    assert not miss, miss[:5]
    rg2 = {kk: v for kk, v in rg.items() if kk != "rows"}
    rg2.update(plan_digest=dg, rows={rel: rg["rows"][rel] for rel in sorted(used)}, samsep={"derived_from": REF_RAWGAIN, "ref_plan_digest": rg["plan_digest"]})
    json.dump(rg2, open(f"{DATA}/plan/RAWGAIN_samsep.json", "w"), indent=1)
    kinds = {}
    for t in used.values():
        kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
    summ = {"plan_digest": dg, "eval_digest": ev_dg, "items": len(PL["items"]), "steps": PL["steps"], "n_train_rows": PL["n_train_rows"],
            "n_eval_rows": PL["n_eval_rows"], "n_first": PL["n_first"], "n_all": PL["n_all"], "tracks": kinds, "symlinks": len(tops),
            "new_digests": newdg, "index": f"{DATA}/plan/M1VOX_INDEX_samsep.json", "rawgain": f"{DATA}/plan/RAWGAIN_samsep.json",
            "shared_flac": root, "rawgain_sha16": hashlib.sha256(open(f"{DATA}/plan/RAWGAIN_samsep.json", "rb").read()).hexdigest()[:16]}
    json.dump(summ, open(f"{DATA}/plan/PLAN_samsep.json", "w"), indent=1)
    print("[done] " + json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
