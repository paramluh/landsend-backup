"""aloe-m1vox-ft DATA INDEX (10-16): job 21980's EXACT rows (Model 1's Aloe handover 'none' bucket stores m1n0..m1n3, data_aloe plan
UNCHANGED: every train row x both orientations, seed 20261011, 1 item / rank / step, 240 eval rows x 2) + channel A's AUDIO reference per
row and data side (A = the other speaker of the item: swap 0 -> A = data 'user' (SPEAKER_01), swap 1 -> A = data 'assistant' (SPEAKER_00)).

Per row and side: the speaker's WLT FLAC in the Aloe handover set (local manifest aloe-handover-full/DATA/_meta/manifest.jsonl: uri /
bytes / generation) + the recorded Mimi LUFS gain (aloe-handover-full w00.mimi.json mimi[spk].lufs_gain_db; = aloe-vox-ft's rule).
ALIGNMENT CHECK (every row, both sides): the m1 store's Mimi codes of the row == aloe-handover-full's w00.<spk>.codes.npy[:, frame_start :
frame_start + n] (same clip origin -> the FLAC timeline frame0 = frame_start, as aloe-vox-ft's rows) and mimi n_frames == the row's src_frames.
PURGED CONVERSATIONS: 122 of Model 1's 8,229 conversations (163 rows: 156 train, 7 eval) were purged from the handover after 21980's rows
were built (aloe-handover-full _meta/noface_exclusion_manifest.jsonl; no WLT FLAC / Mimi left in GCS, current or noncurrent). Their rows stay in
the plan UNCHANGED (21980's exact rows / order); channel A's audio for them = the Mimi DECODE of the A side's codes stored in the row itself
(transformers MimiModel, /data/home/amartyavarma/models/mimi; decoded at staging on the node that reads it -> <root>/_mimidec/<ses>/<seg>/
c<k>.<spk>.flac, 24 kHz FLAC PCM_24; gain 0 dB: the codes were made from the already gain-normalised audio). Entry kind 'mimidec'.
NEEDS: per training node i (ranks 8i..8i+7): every A track its ranks read (train items idx -> rank idx % 64; eval item j -> rank j % 64;
the fallback dummy item = plan item 0 on every node).
usage: python -I build_m1vox_index.py --out PLANDIR [--workers 48]"""
import argparse
import collections
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lib"))
import data_aloe as AL  # noqa: E402   (21980's data module, unchanged copy)

DATA = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
PREFIX = "gs://aloe-processed-datasets/stage2-handover-20261008/wlt/"
SIDE_SPK = {"assistant": "SPEAKER_00", "user": "SPEAKER_01"}
WORLD, LW = 64, 8


def load_maps():
    man = {}
    for r in csv.DictReader(open(f"{DATA}/manifest.csv")):
        man[(r["session_id"], r["seg"])] = r
    fm = {}
    for line in open(f"{DATA}/_meta/manifest.jsonl"):
        r = json.loads(line)
        for spk in ("SPEAKER_00", "SPEAKER_01"):
            k = spk.lower()
            if r.get(f"wlt_{k}_flac_uri"):
                fm[(r["session_id"], r["seg"], spk)] = {"uri": r[f"wlt_{k}_flac_uri"], "bytes": int(r[f"wlt_{k}_flac_bytes"]),
                                                        "generation": int(r[f"wlt_{k}_flac_generation"])}
    return man, fm


_G = {}


def check_store(job):
    """one (store k, row list) chunk -> index entries; asserts the code alignment of every row / side."""
    k, rows = job
    if "man" not in _G:
        _G["man"], _G["fm"] = load_maps()
        _G["st"] = {}
    man, fm = _G["man"], _G["fm"]
    if k not in _G["st"]:
        node = AL.D.R3.load_node(AL.SRC_STORES[k])
        assert node["digest"] == AL.DIGESTS[k], (k, node["digest"])
        _G["st"][k] = AL.D.R3.Rows(node)
    st = _G["st"][k]
    out, cache = [], {}
    for r in rows:
        row = st[r]
        rk = row["row_key"]
        _, _, ses, seg, ck = rk.split("/")
        fs, n = int(row["frame_start"]), int(row["n_frames"])
        if (ses, seg) not in man:                                  # purged conversation (module doc): A audio = Mimi decode of the row's codes
            ent = {}
            for side, spk in SIDE_SPK.items():
                assert row[f"{side}_speaker_dir"] == spk, (rk, side)
                ent[side] = {"kind": "mimidec", "flac": f"_mimidec/{ses}/{seg}/{ck}.{spk}.flac", "gain_db": 0.0, "frame0": fs, "audio_frame0": 0,
                             "n_frames": n, "a_spk": spk, "store": int(k), "row": int(r), "side": side}
            out.append({"row": int(r), "row_key": rk, "A": ent, "purged": 1})
            continue
        m = man[(ses, seg)]
        d = f"{DATA}/{m['relative_dir']}"
        assert fs == 7500 * int(ck[1:]), (rk, fs)
        if d not in cache:
            mj = json.load(open(f"{d}/w00.mimi.json"))
            assert float(mj["clip"]["start_sec"]) == 0.0, (rk, "clip start")
            cache = {d: (mj, {spk: np.load(f"{d}/w00.{spk}.codes.npy", mmap_mode="r") for spk in ("SPEAKER_00", "SPEAKER_01")})}
        mj, codes = cache[d]
        ent = {}
        for side, spk in SIDE_SPK.items():
            assert row[f"{side}_speaker_dir"] == spk, (rk, side, row[f"{side}_speaker_dir"])
            mm = mj["mimi"][spk]
            T = int(mm["n_frames"])
            assert int(mm["audio_sr"]) == 24000 and T == int(m["mimi_T"]) and fs + n <= T, (rk, spk, T, fs, n)
            src = np.asarray(row[f"codes_{side}"])[:, :n]
            ref = np.asarray(codes[spk][:, fs:fs + n])
            assert src.shape == ref.shape and np.array_equal(src, ref), (rk, side, "m1 codes != aloe-handover-full codes")
            f = fm[(ses, seg, spk)]
            rel = f"{ses}/{seg}/{spk}.flac"
            assert f["uri"] == PREFIX + rel, f["uri"]
            ent[side] = {"kind": "gcs", "audio_frame0": fs, "flac": rel, "gain_db": float(mm["lufs_gain_db"]), "lufs_status": mm.get("lufs_status"), "frame0": fs, "n_frames": n,
                         "flac_bytes": f["bytes"], "generation": f["generation"], "a_spk": spk, "uri": f["uri"]}
        out.append({"row": int(r), "row_key": rk, "A": ent})
    return k, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    t0 = time.time()
    PL = AL.make_plan(WORLD, roots=AL.SRC_STORES)
    assert PL["steps"] == 366 and len(PL["items"]) == 23400 and len(PL["eval"]) == 480, (PL["steps"], len(PL["items"]), len(PL["eval"]))
    import pyarrow.parquet as pq
    nrows = [len(pq.read_table(f"{AL.SRC_STORES[k]}/rows.parquet", columns=["row"])) for k in range(4)]
    jobs = []
    for k in range(4):
        for a0 in range(0, nrows[k], 50):
            jobs.append((k, list(range(a0, min(a0 + 50, nrows[k])))))
    import multiprocessing as mp
    idx = {k: [] for k in range(4)}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for j, (k, out) in enumerate(pool.imap_unordered(check_store, jobs)):
            idx[k] += out
            if (j + 1) % 40 == 0 or j + 1 == len(jobs):
                print(f"[index] {j + 1}/{len(jobs)} chunks checked ({time.time() - t0:.0f}s)", flush=True)
    for k in range(4):
        idx[k].sort(key=lambda e: e["row"])
        assert [e["row"] for e in idx[k]] == list(range(nrows[k])), k
    # per-node needs (A side of every item a node's ranks read)
    need = collections.defaultdict(dict)

    def add(node, item):
        k, r, sw = item
        aside = "assistant" if int(sw) else "user"
        e = idx[int(k)][int(r)]["A"][aside]
        if e["kind"] == "gcs":
            need[node][e["flac"]] = {"kind": "gcs", "rel": e["flac"], "uri": e["uri"], "bytes": e["flac_bytes"], "generation": e["generation"]}
        else:
            need[node][e["flac"]] = {"kind": "mimidec", "rel": e["flac"], "store": e["store"], "row": e["row"], "side": e["side"], "n_frames": e["n_frames"]}
    for i, it in enumerate(PL["items"]):
        add((i % WORLD) // LW, it)
    for j, it in enumerate(PL["eval"]):
        add((j % WORLD) // LW, it)
    for node in range(WORLD // LW):
        add(node, PL["items"][0])
    os.makedirs(a.out, exist_ok=True)
    json.dump({"format": "aloe-m1vox-index-v1", "stores": [AL.SRC_STORES[k] for k in range(4)], "digests": AL.DIGESTS,
               "rows": {str(k): idx[k] for k in range(4)}}, open(f"{a.out}/M1VOX_INDEX.json", "w"))
    nd = {str(n): sorted(v.values(), key=lambda t: t["rel"]) for n, v in sorted(need.items())}
    json.dump(nd, open(f"{a.out}/NEEDS.json", "w"))
    alltr = {t["rel"] for v in nd.values() for t in v}
    print(f"[index] rows per store {nrows}; code alignment verified on every row x both sides; plan steps {PL['steps']} items {len(PL['items'])} "
          f"eval {len(PL['eval'])}; distinct A tracks {len(alltr)} "
          f"({sum(t.get('bytes', 0) for v in nd.values() for t in v) / 1e9:.1f} GB summed over nodes); purged rows {sum(e.get('purged', 0) for k in range(4) for e in idx[k])}; "
          f"per node (gcs files, GB, mimidec rows) "
          + str({n: (sum(t['kind'] == 'gcs' for t in v), round(sum(t.get('bytes', 0) for t in v) / 1e9, 1), sum(t['kind'] == 'mimidec' for t in v)) for n, v in nd.items()})
          + f" ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
