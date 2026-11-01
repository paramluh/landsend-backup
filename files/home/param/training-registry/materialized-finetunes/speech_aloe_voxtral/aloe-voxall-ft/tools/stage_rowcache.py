"""aloe-voxall-ft SPEED FIX (10-16, SPEED_FIX.md): decode node I's channel-A rows ONCE into the node-local row cache (code/rowcache_vox.py),
with the trainer's exact function (audio_s4q.ffmpeg_row_audio via data_aloe_vox.Stores.conv's arguments: FLAC_ROOT/<flac>, frame0, n_frames,
recorded gain_db). Rows = every train + eval item of node I's stores (data_aloe_vox.node_stores / local_items from the node-local staged
stores + their vox_index.json; A = the other side of the item's B), train items in make_plan's order (seed (20261011, I)), then eval;
deduplicated by cache key. Run AFTER the node's store + FLACs are staged (prep).
DISK GUARD: a full node needs ~3,000+ rows x 57.6 MB (~175+ GB); building stops once /scratch free would drop below --min-free-gb (default
ROWCACHE_MIN_FREE_GB or 120; the user's rule keeps >= 100 GB free for the run's local exports). Rows not cached decode on the fly in the
loader (bit-identical, just slower) -- coverage is logged and written to <cache>/LEDGER_node<I>.json.
--verify N: re-decode N cached rows (spread over the list) on the fly and assert bit equality.
usage: python -I stage_rowcache.py --node I [--nnodes 8] [--cache <ALOE_ROWCACHE>] [--workers 16] [--verify 4]"""
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(os.path.dirname(HERE), "code")
sys.path.insert(0, CODE); sys.path.insert(0, f"{CODE}/lib")
RUN = os.path.basename(os.path.dirname(HERE))                       # aloe-voxall-ft / aloe-vox-ft


def entries(node, nnodes):
    import numpy as np
    import data_aloe_vox as A
    tr, ev, vis = [], [], {}
    for k in A.node_stores(node, nnodes):
        t_, e_ = A.local_items(k)
        tr += [(k, int(r), int(sw)) for r, sw in t_]
        ev += [(k, int(r), int(sw)) for r, sw in e_]
        vis[k] = json.load(open(f"{A.store_root(k)}/vox_index.json"))["rows"]
    perm = np.random.default_rng([A.PLAN_SEED, node]).permutation(len(tr))
    order = [tr[j] for j in perm] + ev
    out, seen = [], set()
    for k, r, sw in order:
        e = vis[k][r]
        aside = "assistant" if sw else "user"                       # = data_aloe_vox.Stores.conv
        a = e["A"][aside]
        ent = {"flac": f"{A.FLAC_ROOT}/{a['flac']}", "frame0": int(a["frame0"]), "n_frames": int(a["n_frames"]), "row_key": e["row_key"]}
        tag = (ent["flac"], ent["frame0"], ent["n_frames"], float(a["gain_db"]))
        if tag not in seen:
            seen.add(tag); out.append((ent, float(a["gain_db"])))
    return out


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, required=True)
    ap.add_argument("--nnodes", type=int, default=8)
    ap.add_argument("--cache", default=os.environ.get("ALOE_ROWCACHE") or f"/scratch/amartyavarma/{RUN}/rowcache")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-free-gb", type=float, default=float(os.environ.get("ROWCACHE_MIN_FREE_GB", "120")))
    ap.add_argument("--verify", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="TEST ONLY: first N rows")
    a = ap.parse_args()
    import rowcache_vox as RC
    os.makedirs(a.cache, exist_ok=True)
    t0 = time.time()
    ents = entries(a.node, a.nnodes)
    if a.limit:
        ents = ents[:a.limit]
    row_gb = max((x[0]["n_frames"] for x in ents), default=0) * 1920 * 4 / 1e9
    tot_gb = sum(x[0]["n_frames"] for x in ents) * 1920 * 4 / 1e9
    print(f"[rowcache] {RUN} node {a.node}: {len(ents)} distinct A rows (train in plan order + eval); <= {row_gb * 1e3:.1f} MB each, "
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
    led = {"run": RUN, "node": a.node, "rows": len(ents), "cached": len(cached), "coverage": len(cached) / max(len(ents), 1), **st, "verify": ver,
           "cache": a.cache, "free_gb_after": round(free_gb(a.cache), 1), "min_free_gb": a.min_free_gb, "secs": round(time.time() - t0, 1)}
    json.dump(led, open(f"{a.cache}/LEDGER_node{a.node}.json", "w"), indent=1)
    print(f"[rowcache] node {a.node} DONE: {json.dumps(led)}", flush=True)


if __name__ == "__main__":
    main()
