"""D2 (user decision): the live PPM run sees ALL full podcast rows. The 16 training nodes carry 16 of the 20 s3-rows slices; the other 4
(slinky-4, -7, -15, -18) are split over the training nodes so that every rank ends with the same number of FULL train rows.

usage: plan_extra.py OUT.json NATIVE_COUNTS.jsonl SLICE_META_DIR [SLICE_META_DIR ...]
  NATIVE_COUNTS.jsonl  one line per training host: {"host", "node", "counts": [FULL train rows of local rank 0..7 under the live plan]}
  SLICE_META_DIR       a directory with the slice's NODE.json, rows.parquet, *.DONE, *.index.npz and LISTING.txt ("<bytes> <file>")

EXTRA ROWS of a slice = rows with n_frames == 7500 whose EPISODE is not an eval episode (data_s3q.is_eval_episode, the live run's
eval_frac / eval_salt) = exactly the rows the slice's own node would train on (data_ppm.speech_plan's FULL filter of DS.plan's train rows).
TARGET: T = (sum native + sum extra) // world rows per rank; rank deficit d_r = T - native_r; host deficit D_h = sum of its ranks' d_r.
ASSIGNMENT: the extra pool in (slice order, node row) order is cut into consecutive host ranges of D_h rows (hosts in sorted order), so a
host's rows sit in a contiguous shard range (of at most a couple of slices). A host's rows are shuffled (seed, host) and cut into its local
ranks' d_r blocks. HOST FILES: every slice it touches gets NODE.json / NODE.set1.json / rows.parquet / all *.DONE / all *.index.npz
(load_node's contract, tiny) + for the NEEDED shards codes_user / codes_assistant / words / persona. needed = the shards of its rows plus
every shard holding any row of the same RECORDING (sysprompt.rec_key) as one of its rows: the voice clips come from later chunks of the same
recording and rec_chunks counts its chunks, so the recording must be complete on the host (SibIndex then answers exactly as on the native
node)."""
import json
import os
import sys
import zlib

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))
import data_s3q as DS  # noqa: E402
import sysprompt as SPR  # noqa: E402

SEED, FRAC, SALT, LW = 20261011, 0.0005, "ppm-v1", 8
EXTRA_ROOT = "/scratch/amartyavarma/s3-rows-extra"
SMALL = ("NODE.json", "NODE.set1.json", "rows.parquet")


def slice_info(d):
    meta = json.load(open(f"{d}/NODE.json"))
    t = pq.read_table(f"{d}/rows.parquet", columns=["row", "shard", "row_key", "dataset", "podcast_id", "episode_id", "n_frames"]).to_pydict()
    assert t["row"] == list(range(len(t["row"])))
    assert sorted(set(t["shard"])) == list(meta["shards"]), "rows.parquet shards != NODE.json shards"
    ev = np.array([DS.is_eval_episode(DS.episode_key(a, b, c), FRAC, SALT) for a, b, c in zip(t["dataset"], t["podcast_id"], t["episode_id"])])
    nf = np.asarray(t["n_frames"])
    extra = np.flatnonzero((nf == 7500) & ~ev)
    recs = [SPR.rec_key(k) for k in t["row_key"]]
    rec_shards = {}
    for r, s in zip(recs, t["shard"]):
        rec_shards.setdefault(r, set()).add(s)
    sizes = {}
    for ln in open(f"{d}/LISTING.txt"):
        b, f = ln.split()
        sizes[f] = int(b)
    return {"node": meta["node"], "dir": d, "shards": list(meta["shards"]), "shard": t["shard"], "recs": recs, "rec_shards": rec_shards,
            "extra": extra, "sizes": sizes, "n_rows": len(nf)}


def main():
    out, ncpath, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
    nat = [json.loads(l) for l in open(ncpath)]
    hosts = sorted((n["host"] for n in nat), key=lambda h: int(h.split("-")[1]))
    cnt = {n["host"]: n["counts"] for n in nat}
    assert all(n["host"] == n["node"] for n in nat), "native counts must come from each host's own slice"
    W = LW * len(hosts)
    S = [slice_info(d) for d in dirs]
    n_nat = sum(sum(c) for c in cnt.values())
    n_ext = sum(len(s["extra"]) for s in S)
    T = (n_nat + n_ext) // W
    d = {h: [T - c for c in cnt[h]] for h in hosts}
    assert min(min(v) for v in d.values()) >= 0, "a rank already has more native rows than the target"
    pool = [(k, int(r)) for k, s in enumerate(S) for r in s["extra"]]          # (slice, node row) in slice / node-row order
    plan = {"format": "ppm-extra-plan-v1", "seed": SEED, "eval_frac": FRAC, "eval_salt": SALT, "world": W, "lw": LW, "target_rows_per_rank": T,
            "native_rows": n_nat, "extra_rows": n_ext, "unassigned_extra": 0, "slices": [s["node"] for s in S], "hosts": {}}
    pos = 0
    for h in hosts:
        need = sum(d[h])
        mine = pool[pos:pos + need]; pos += need
        assert len(mine) == need
        rng = np.random.default_rng(np.random.SeedSequence([SEED, zlib.crc32(h.encode()), 0xD2]))
        order = rng.permutation(len(mine))
        sl_used = sorted({k for k, _ in mine})
        slots = []
        for k in sl_used:
            s = S[k]
            rows = sorted(r for kk, r in mine if kk == k)
            shards = set(s["shard"][r] for r in rows)
            for r in rows:
                shards |= s["rec_shards"][s["recs"][r]]
            shards = sorted(shards)
            files = list(SMALL) + [f"{x}.{e}" for x in s["shards"] for e in ("DONE", "index.npz")] + \
                [f"{x}.{e}" for x in shards for e in ("codes_user.npy", "codes_assistant.npy", "words.arrow", "persona.arrow")]
            files = [f for f in files if f in s["sizes"]]
            slots.append({"node": s["node"], "root": f"{EXTRA_ROOT}/{s['node']}", "src": s["dir"], "shards_present": shards, "rows": rows,
                          "files": files, "bytes": int(sum(s["sizes"][f] for f in files)), "rows_shards": len({s["shard"][r] for r in rows})})
        slot_of = {k: i for i, k in enumerate(sl_used)}
        ranks, at = [], 0
        for l in range(LW):
            blk = [mine[j] for j in order[at:at + d[h][l]]]; at += d[h][l]
            ranks.append([[slot_of[k], r] for k, r in blk])
        plan["hosts"][h] = {"slots": slots, "ranks": ranks, "native": cnt[h], "extra_per_rank": d[h], "bytes": int(sum(x["bytes"] for x in slots))}
    plan["unassigned_extra"] = len(pool) - pos
    json.dump(plan, open(out, "w"))
    print(f"native {n_nat:,} + extra {n_ext:,} = {n_nat + n_ext:,} FULL train rows; world {W}; target {T:,} rows/rank -> {T // 2:,} steps (2 speech "
          f"micros/step); unassigned extra {len(pool) - pos}")
    for h in hosts:
        P = plan["hosts"][h]
        print(f"  {h}: extra {sum(P['extra_per_rank']):,} rows ({min(P['extra_per_rank'])}..{max(P['extra_per_rank'])}/rank) from "
              + ", ".join(f"{x['node']} {len(x['rows']):,} rows {x['rows_shards']}+{len(x['shards_present']) - x['rows_shards']} shards" for x in P["slots"])
              + f"; {P['bytes'] / 1e9:.1f} GB")
    print(f"total bytes to place {sum(P['bytes'] for P in plan['hosts'].values()) / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
