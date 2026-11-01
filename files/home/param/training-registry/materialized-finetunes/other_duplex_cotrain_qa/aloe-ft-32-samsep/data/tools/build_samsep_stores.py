"""aloe-ft-32-samsep DATA: the s2h-m1 handover stores with ONLY the Mimi codes swapped for the SAM-separated ISOLATED streams.

SOURCE  rows: /data/home/amartyavarma/qwen-1ch-data/s2h-m1/stores/m1n{0..3} (read-only; s3-duplex-rows-v1, 11,940 rows, 8,229 conversations)
        codes: /data/home/amartyavarma/data/aloe-sam-sep/hf (InternalCan/stage2-sam-sep-v1-mimi @ ecb72b2d; local copy, no download)
RULE    every row of a conversation present in sam_sep (all chunks, train + eval):
          codes_assistant[r] (= SPEAKER_00 = lane 0) <- SPEAKER_00_isolated[:, frame_start : frame_start + n_frames], frames >= n = -1
          codes_user[r]      (= SPEAKER_01 = lane 1) <- SPEAKER_01_isolated[:, frame_start : frame_start + n_frames], frames >= n = -1
        asserted per row: sam T == the row's src_frames, shape [32, T] int16, codes in 0..2047.
        rows of the conversations NOT in sam_sep (10-16 no-face purge) are DROPPED (not kept with old codes).
        each kept conversation stays on its original logical node m1nK, rows in original order; node row ids renumbered 0..R'-1.
NON-CODE FILES  words.arrow / persona.arrow / index.npz / rows.jsonl / rows.parquet = the original records of the kept rows, unchanged
        (rows.jsonl / rows.parquet: only 'row' and 'row_in_shard' renumbered, as the trainer asserts row == range(R)); DONE / NODE.json
        recomputed for the kept rows (rows, frames, ids_blake2b digest over row_key + n_frames, hours, split_rows) + a 'samsep' provenance block.
OUT     /data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores (same layout and file names) + <data>/row_stats.npz, row_map.json,
        dropped_122.txt, build_summary.json
usage   python -I build_samsep_stores.py [--workers 32]"""
import argparse
import io
import json
import os
import shutil
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/data/home/amartyavarma/qwen-1ch/stage3/data/rows_build/scripts")
import s3rows as F  # noqa: E402

SRC = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1/stores"
OUT = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores"
HF = "/data/home/amartyavarma/data/aloe-sam-sep/hf"
DATA = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/data"
SH = "rows-00000-of-00001"
NODES = [f"m1n{k}" for k in range(4)]
LANE = {"assistant": "SPEAKER_00", "user": "SPEAKER_01"}          # s2h_to_s3rows.SPK_SIDE
OTHER = {"assistant": "user", "user": "assistant"}
ROWB = F.DEPTH * F.ROW_FRAMES * 2
HF_REV = "ecb72b2de2347bfbbc3b71b0236e2986b0c44572"


def work(job):
    """one sam_sep token shard: every kept row of its pairs -> both lanes' new codes written in place; per-row stats."""
    import pyarrow.parquet as pq
    shard, pairs = job                                   # pairs: {conv: [(k, new_r, old_r, fs, n, src_frames, row_key, split, ci)]}
    fd = {(k, s): os.open(f"{OUT}/{NODES[k]}/{SH}.codes_{s}.npy", os.O_WRONLY) for k in range(4) for s in F.SIDES}
    hdr = {}
    for k in range(4):
        for s_ in F.SIDES:
            mm = np.load(f"{OUT}/{NODES[k]}/{SH}.codes_{s_}.npy", mmap_mode="r")
            assert mm.dtype == np.int16 and mm.shape[1:] == (F.DEPTH, F.ROW_FRAMES)
            assert hdr.setdefault(k, mm.offset) == mm.offset and os.path.getsize(mm.filename) == mm.offset + mm.shape[0] * ROWB
            del mm
    old = {(k, s): np.load(f"{SRC}/{NODES[k]}/{SH}.codes_{s}.npy", mmap_mode="r") for k in range(4) for s in F.SIDES}
    pf = pq.ParquetFile(f"{HF}/{shard}")
    cols = ["session", "seg", "T", "equal_T", "SPEAKER_00_isolated_npy", "SPEAKER_01_isolated_npy"]
    recs, seen = [], set()
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["session", "seg"]).to_pydict()
        if not any(f"{a}/{b}" in pairs for a, b in zip(t["session"], t["seg"])):
            continue
        for row in pf.read_row_group(rg, columns=cols).to_pylist():
            conv = f"{row['session']}/{row['seg']}"
            if conv not in pairs:
                continue
            seen.add(conv)
            T = int(row["T"])
            assert row["equal_T"], conv
            iso = {}
            for side, spk in LANE.items():
                x = np.load(io.BytesIO(row[f"{spk}_isolated_npy"]))
                assert x.dtype == np.int16 and x.shape == (F.DEPTH, T), (conv, spk, x.dtype, x.shape)
                assert int(x.min()) >= 0 and int(x.max()) < F.CBS, (conv, spk, int(x.min()), int(x.max()))
                iso[side] = x
            for (k, nr, orr, fs, n, srcT, rk, split, ci) in pairs[conv]:
                assert T == srcT, ("sam T != the row's src_frames", rk, T, srcT)
                assert fs + n <= T and 1 <= n <= F.ROW_FRAMES, (rk, fs, n, T)
                st = {"store": k, "new_row": nr, "old_row": orr, "row_key": rk, "split": split, "chunk_index": ci, "n": n}
                for side in F.SIDES:
                    cc = np.full((F.DEPTH, F.ROW_FRAMES), F.PAD_CODE, np.int16)
                    cc[:, :n] = iso[side][:, fs:fs + n]
                    b = cc.tobytes()
                    assert len(b) == ROWB
                    w = os.pwrite(fd[(k, side)], b, hdr[k] + nr * ROWB)
                    assert w == ROWB, (rk, side, w)
                    o_own = np.asarray(old[(k, side)][orr])
                    o_oth = np.asarray(old[(k, OTHER[side])][orr])
                    assert (o_own[:, n:] == F.PAD_CODE).all() and (o_own[:, :n] >= 0).all(), ("original row layout", rk, side)
                    st[f"diff_{side}"] = (cc[:, :n] != o_own[:, :n]).sum(1).astype(np.int64)        # [32] codes differing per CB (own lane)
                    st[f"cb0_own_{side}"] = float((cc[0, :n] == o_own[0, :n]).mean())
                    st[f"cb0_oth_{side}"] = float((cc[0, :n] == o_oth[0, :n]).mean())
                recs.append(st)
    for f in fd.values():
        os.fsync(f); os.close(f)
    missing = set(pairs) - seen
    assert not missing, (shard, sorted(missing)[:5])
    return shard, recs


def main():
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=32); a = ap.parse_args()
    t0 = time.time()
    assert not os.path.exists(OUT), ("output exists; remove it first (never the source)", OUT)
    pidx = pq.read_table(f"{HF}/index/pairs.parquet", columns=["session", "seg", "T", "token_shard", "token_row"]).to_pydict()
    sam = {f"{s}/{g}": (int(T), sh) for s, g, T, sh in zip(pidx["session"], pidx["seg"], pidx["T"], pidx["token_shard"])}
    assert len(sam) == 12681, len(sam)
    os.makedirs(OUT + ".building", exist_ok=False)
    tmp = OUT + ".building"
    plan, dropped, by_shard = {}, [], {}
    for k, nd in enumerate(NODES):
        t = pq.read_table(f"{SRC}/{nd}/rows.parquet").to_pydict()
        assert t["row"] == list(range(len(t["row"])))
        keep = [i for i, c in enumerate(t["src_conversation"]) if c in sam]
        plan[k] = {"keep": keep, "n_old": len(t["row"])}
        for i, c in enumerate(t["src_conversation"]):
            if c not in sam:
                dropped.append({"store": k, "old_row": i, "row_key": t["row_key"][i], "conv": c, "split": t["split"][i], "chunk_index": int(t["chunk_index"][i]),
                                "n_frames": int(t["n_frames"][i])})
        for nr, i in enumerate(keep):
            c = t["src_conversation"][i]
            by_shard.setdefault(sam[c][1], {}).setdefault(c, []).append(
                (k, nr, i, int(t["frame_start"][i]), int(t["n_frames"][i]), int(t["src_frames"][i]), t["row_key"][i], t["split"][i], int(t["chunk_index"][i])))
        os.makedirs(f"{tmp}/{nd}")
        for s in F.SIDES:                                                # header + full-size file; every row is written by a worker
            mm = np.lib.format.open_memmap(f"{tmp}/{nd}/{SH}.codes_{s}.npy", mode="w+", dtype=np.int16, shape=(len(keep), F.DEPTH, F.ROW_FRAMES))
            del mm
        print(f"[plan] {nd}: {len(t['row'])} rows -> keep {len(keep)}, drop {len(t['row']) - len(keep)}", flush=True)
    os.rename(tmp, OUT)                                                  # workers write into OUT (rows land at their final offsets)
    dconv = sorted({d["conv"] for d in dropped})
    print(f"[plan] dropped {len(dropped)} rows of {len(dconv)} conversations; {sum(len(v) for v in by_shard.values())} kept convs over {len(by_shard)} sam shards "
          f"({time.time() - t0:.0f}s)", flush=True)
    with Pool(a.workers) as pool:
        recs = []
        for j, (sh, r) in enumerate(pool.imap_unordered(work, sorted(by_shard.items()), chunksize=1)):
            recs += r
            print(f"[codes] {j + 1}/{len(by_shard)} {os.path.basename(sh)}: {len(r)} rows ({time.time() - t0:.0f}s)", flush=True)
    nkeep = sum(len(p["keep"]) for p in plan.values())
    assert len(recs) == nkeep and len({(r["store"], r["new_row"]) for r in recs}) == nkeep, (len(recs), nkeep)
    # ---------------------------------------------------------------- non-code files: kept records, unchanged
    old_dg, new_dg = {}, {}
    for k, nd in enumerate(NODES):
        s_, o_ = f"{SRC}/{nd}", f"{OUT}/{nd}"
        keep = plan[k]["keep"]; ki = pa.array(keep, pa.int64())
        old_dg[nd] = F.load_node(s_)["digest"]
        wt = ipc.open_file(pa.memory_map(f"{s_}/{SH}.words.arrow", "r"))
        tb = wt.read_all().take(ki).combine_chunks()
        with pa.OSFile(f"{o_}/{SH}.words.arrow", "wb") as sink, ipc.new_file(sink, wt.schema) as w:   # = write_node (no compression, 256-row batches)
            w.write_table(tb, max_chunksize=256)
        pr = ipc.open_file(pa.memory_map(f"{s_}/{SH}.persona.arrow", "r"))
        pt = pr.read_all().take(ki).combine_chunks()
        with pa.OSFile(f"{o_}/{SH}.persona.arrow", "wb") as sink, ipc.new_file(sink, pr.schema, options=ipc.IpcWriteOptions(compression="zstd")) as w:
            w.write_table(pt, max_chunksize=int(pr.schema.metadata[b"batch_rows"]))
        z = np.load(f"{s_}/{SH}.index.npz")
        ix = {kk: z[kk][keep] for kk in z.files}
        np.savez(f"{o_}/{SH}.index.npz.tmp.npz", **ix); os.replace(f"{o_}/{SH}.index.npz.tmp.npz", f"{o_}/{SH}.index.npz")
        keys = tb.column("row_key").to_pylist(); nf = ix["length"].astype(np.int32)
        ks = set(keep)
        assert keys == [json.loads(l)["row_key"] for i, l in enumerate(open(f"{s_}/rows.jsonl")) if i in ks]
        done = {"shard": SH, "rows": len(keep), "frames": int(nf.sum()), "ids_blake2b": F.shard_digest(keys, nf)}
        json.dump(done, open(f"{o_}/{SH}.DONE", "w"))
        lines = open(f"{s_}/rows.jsonl").read().splitlines()
        with open(f"{o_}/rows.jsonl", "w") as f:
            for nr, i in enumerate(keep):
                p = json.loads(lines[i]); assert p["row"] == i and p["row_in_shard"] == i
                p["row"] = nr; p["row_in_shard"] = nr
                f.write(json.dumps(p) + "\n")
        rp = pq.read_table(f"{s_}/rows.parquet").take(ki)
        for col in ("row", "row_in_shard"):
            j = rp.schema.get_field_index(col)
            rp = rp.set_column(j, rp.schema.field(j), pa.array(np.arange(len(keep)), rp.schema.field(j).type))
        pq.write_table(rp, f"{o_}/rows.parquet")
        meta = json.load(open(f"{s_}/NODE.json"))
        sp = Counter(rp.column("split").to_pylist())
        meta.update(rows=[len(keep)], frames=[int(nf.sum())], ids_blake2b=[done["ids_blake2b"]], total_rows=len(keep), total_frames=int(nf.sum()),
                    by_dataset={"ALOE": {"rows": len(keep), "frames": int(nf.sum()), "hours": round(nf.sum() / F.FPS / 3600, 2)}},
                    converter=os.path.abspath(__file__), version="s2h-m1-samsep-v1", built=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    split_rows={"eval": sp["eval"], "train": sp["train"]})
        meta["stats_note"] = "'stats' = the ORIGINAL s2h-m1 build stats (before the sam_sep drop), copied unchanged"
        meta["samsep"] = {"derived_from": s_, "derived_from_digest": old_dg[nd], "codes": f"HF InternalCan/stage2-sam-sep-v1-mimi @ {HF_REV} (local {HF})",
                          "codes_assistant": "SPEAKER_00_isolated_npy[:, frame_start:frame_start+n_frames]",
                          "codes_user": "SPEAKER_01_isolated_npy[:, frame_start:frame_start+n_frames]",
                          "rows_dropped": plan[k]["n_old"] - len(keep), "convs_dropped": len({d['conv'] for d in dropped if d['store'] == k}),
                          "drop_rule": "conversation not in sam_sep (10-16 no-face purge)", "unchanged": "every non-code record of the kept rows; row ids renumbered"}
        json.dump(meta, open(f"{o_}/NODE.json", "w"), indent=1)
        new_dg[nd] = F.load_node(o_)["digest"]
        print(f"[meta] {nd}: rows {len(keep)} frames {int(nf.sum())} digest {new_dg[nd]} (was {old_dg[nd]}) ({time.time() - t0:.0f}s)", flush=True)
    # ---------------------------------------------------------------- top-level files
    dset = set(dconv)
    ev_lines = open(f"{SRC}/eval_episodes.txt").read().splitlines()
    dropped_eps = {f"ALOE/s2handover-none/{c.replace('/', '_')}" for c in dconv}
    with open(f"{OUT}/eval_episodes.txt", "w") as f:
        for l in ev_lines:
            if l.startswith("#") or l not in dropped_eps:
                f.write(l + "\n")
    for nd in NODES:
        sj = json.load(open(f"{SRC}/SPLIT-{nd}.json"))
        sj["node_of"] = {c: h for c, h in sj["node_of"].items() if c not in dset}
        sj["eval_episodes"] = [e for e in sj["eval_episodes"] if e not in dropped_eps]
        sj["convs_with_rows"] = len(sj["node_of"])
        rpn = {"eval": {}, "train": {}}
        for k, n2 in enumerate(NODES):
            sp = Counter(pq.read_table(f"{OUT}/{n2}/rows.parquet", columns=["split"]).column("split").to_pylist())
            rpn["eval"][n2] = sp["eval"]; rpn["train"][n2] = sp["train"]
        sj["rows_per_node"] = rpn
        sj["samsep"] = {"derived_from": f"{SRC}/SPLIT-{nd}.json", "convs_dropped": dconv, "note": "eval_sessions / stats / convs_listed = the original's"}
        json.dump(sj, open(f"{OUT}/SPLIT-{nd}.json", "w"), indent=1)
    # PLAN.env (stage-3 MODEL 1 trainer only; the Aloe FT does not read it) is regenerated by verify_samsep_stores.py
    # ---------------------------------------------------------------- side outputs
    os.makedirs(DATA, exist_ok=True)
    order = sorted(recs, key=lambda r: (r["store"], r["new_row"]))
    np.savez(f"{DATA}/row_stats.npz", store=np.array([r["store"] for r in order], np.int8), new_row=np.array([r["new_row"] for r in order], np.int32),
             old_row=np.array([r["old_row"] for r in order], np.int32), n=np.array([r["n"] for r in order], np.int32),
             chunk_index=np.array([r["chunk_index"] for r in order], np.int16), split=np.array([r["split"] for r in order]),
             row_key=np.array([r["row_key"] for r in order]),
             **{f"diff_{s}": np.stack([r[f"diff_{s}"] for r in order]) for s in F.SIDES},
             **{f"cb0_{w}_{s}": np.array([r[f"cb0_{w}_{s}"] for r in order]) for s in F.SIDES for w in ("own", "oth")})
    json.dump({"format": "aloe-ft-32-samsep-row-map-v1", "src": SRC, "out": OUT,
               "map": {nd: {"old_rows": plan[k]["n_old"], "new_of_old": {str(i): nr for nr, i in enumerate(plan[k]["keep"])}} for k, nd in enumerate(NODES)},
               "old_digest": old_dg, "new_digest": new_dg}, open(f"{DATA}/row_map.json", "w"))
    with open(f"{DATA}/dropped_122.txt", "w") as f:
        f.write(f"# {len(dconv)} s2h-m1 conversations absent from sam_sep (10-16 no-face purge), dropped from {OUT}; {len(dropped)} rows\n")
        f.write("# conversation\trow_key\tstore\told_row\tsplit\tchunk_index\tn_frames\tclass\n")
        for d in sorted(dropped, key=lambda d: (d["conv"], d["chunk_index"])):
            cl = "later_chunk" if d["chunk_index"] else f"first_chunk_{d['split']}"
            f.write(f"{d['conv']}\t{d['row_key']}\tm1n{d['store']}\t{d['old_row']}\t{d['split']}\t{d['chunk_index']}\t{d['n_frames']}\t{cl}\n")
    cls = Counter(("later_chunk_" + d["split"]) if d["chunk_index"] else f"first_chunk_{d['split']}" for d in dropped)
    summ = {"out": OUT, "rows_kept": nkeep, "rows_dropped": len(dropped), "convs_dropped": len(dconv), "dropped_by_class": dict(cls),
            "kept_by_class": dict(Counter(("later_chunk_" + r["split"]) if r["chunk_index"] else f"first_chunk_{r['split']}" for r in recs)),
            "new_digests": new_dg, "old_digests": old_dg, "time_s": round(time.time() - t0, 1)}
    json.dump(summ, open(f"{DATA}/build_summary.json", "w"), indent=1)
    print("[done] " + json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
