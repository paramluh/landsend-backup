"""aloe-m1vox-ft SPEED FIX (10-16, SPEED_FIX.md): decode node I's channel-A rows ONCE into the node-local row cache (code/rowcache_vox.py),
with the trainer's exact function (audio_s4q.ffmpeg_row_audio via data_aloe_m1vox's arguments: FLAC_ROOT/<flac>, audio_frame0, n_frames,
recorded gain_db). Rows = the items of node I's 8 ranks (data_aloe plan = 21980's, built from the /data source stores = what the trainer
asserts), in training order (step-major over the 8 ranks), then their eval items; deduplicated by cache key.
DISK GUARD: a full node needs ~2,900 rows x 57.6 MB (~170 GB); building stops once /scratch free would drop below --min-free-gb (default
ROWCACHE_MIN_FREE_GB or 120; the user's rule keeps >= 100 GB free for the run's local exports). Rows not cached decode on the fly in the
loader (bit-identical, just slower) -- coverage is logged and written to <cache>/LEDGER_node<I>.json.
--verify N: re-decode N cached rows (spread over the list) on the fly and assert bit equality.
usage: python -I stage_rowcache.py --node I [--cache /scratch/amartyavarma/aloe-m1vox-ft/rowcache] [--workers 16] [--verify 4]"""
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import sys
import time

V = "/data/home/amartyavarma/ppm/aloe-m1vox-ft"
sys.path.insert(0, f"{V}/code"); sys.path.insert(0, f"{V}/code/lib")
WORLD, LW = 64, 8


def entries(node):
    import data_aloe_m1vox as A
    PL = A.make_plan(WORLD, roots=A.AL.SRC_STORES)
    idx = A.vox_index()
    per_rank = [A.rank_items(PL, r, WORLD) for r in range(node * LW, (node + 1) * LW)]
    order = [tr[t] for t in range(PL["steps"]) for tr, _ in per_rank if tr[t] is not None] + [it for _, ev in per_rank for it in ev]
    out, seen = [], set()
    for k, r, sw in order:
        e = idx[int(k)][int(r)]
        aside = "assistant" if int(sw) else "user"                  # = data_aloe_m1vox.Stores.conv
        a = e["A"][aside]
        ent = {"flac": f"{A.FLAC_ROOT}/{a['flac']}", "frame0": int(a["audio_frame0"]), "n_frames": int(a["n_frames"]), "row_key": e["row_key"]}
        tag = (ent["flac"], ent["frame0"], ent["n_frames"], float(a["gain_db"]))
        if tag not in seen:
            seen.add(tag); out.append((ent, float(a["gain_db"])))
    return out


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, required=True)
    ap.add_argument("--cache", default=os.environ.get("ALOE_ROWCACHE") or "/scratch/amartyavarma/aloe-m1vox-ft/rowcache")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-free-gb", type=float, default=float(os.environ.get("ROWCACHE_MIN_FREE_GB", "120")))
    ap.add_argument("--verify", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="TEST ONLY: first N rows")
    a = ap.parse_args()
    import rowcache_vox as RC
    os.makedirs(a.cache, exist_ok=True)
    t0 = time.time()
    ents = entries(a.node)
    if a.limit:
        ents = ents[:a.limit]
    row_gb = max((x[0]["n_frames"] for x in ents), default=0) * 1920 * 4 / 1e9
    tot_gb = sum(x[0]["n_frames"] for x in ents) * 1920 * 4 / 1e9
    print(f"[rowcache] node {a.node}: {len(ents)} distinct A rows (train step-major + eval); <= {row_gb * 1e3:.1f} MB each, "
          f"{tot_gb:.0f} GB for all; cache {a.cache}; /scratch free {free_gb(a.cache):.0f} GB, keep >= {a.min_free_gb:.0f}", flush=True)
    st = {"built": 0, "hit": 0, "skipped_disk": 0}
    todo = []
    for ent, g in ents:
        if os.path.exists(RC.path_for(ent, g, a.cache)):
            st["hit"] += 1
        else:
            todo.append((ent, g))
    run = todo                                                     # per-row disk guard below (keeps >= min_free_gb free)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(lambda x: RC.build(x[0], x[1], a.cache) if free_gb(a.cache) - row_gb > a.min_free_gb else "skipped_disk", run)):
            st["built" if r == "built" else ("hit" if r == "hit" else "skipped_disk")] += 1
            if (i + 1) % 200 == 0:
                print(f"[rowcache] {i + 1}/{len(run)} ({time.time() - t0:.0f}s; free {free_gb(a.cache):.0f} GB)", flush=True)
    cached = [x for x in ents if os.path.exists(RC.path_for(x[0], x[1], a.cache))]
    ver = []
    if a.verify and cached:
        step = max(1, len(cached) // a.verify)
        for ent, g in cached[::step][:a.verify]:
            ok = RC.verify(ent, g, a.cache)
            ver.append([ent["row_key"], ent["frame0"], ok])
            assert ok, ("row cache != on-the-fly decode", ent)
    led = {"node": a.node, "rows": len(ents), "cached": len(cached), "coverage": len(cached) / max(len(ents), 1), **st, "verify": ver,
           "cache": a.cache, "free_gb_after": round(free_gb(a.cache), 1), "min_free_gb": a.min_free_gb, "secs": round(time.time() - t0, 1)}
    json.dump(led, open(f"{a.cache}/LEDGER_node{a.node}.json", "w"), indent=1)
    print(f"[rowcache] node {a.node} DONE: {json.dumps(led)}", flush=True)


if __name__ == "__main__":
    main()
