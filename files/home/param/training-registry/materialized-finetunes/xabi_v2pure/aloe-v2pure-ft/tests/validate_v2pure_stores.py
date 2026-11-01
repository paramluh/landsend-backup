"""aloe-v2pure-ft STORE VALIDATION (10-17; compute nodes only). Checks the stores built by code/build_v2pure_rows.py against the source cache
/data/home/amartyavarma/aloe-pose/sam-cache-active-v2pure, through the TRAINER's readers (code/lib/s3rows.py, code/data_aloe_pose.py).
Env forced before any import: S3_MIN_FRAMES=1 PPM_CD=2 QQA_A_EMPTY=1 QQA_POSE_B=1.

modes (each prints one line 'RESULT <json>'):
  rows    --store K --part j --nparts m   every row of the slice: lib/s3rows.check_row(strict) through lib/s3rows.Rows; B Mimi [0, n) and B pose
                                          [0, n) == the cache item's arrays exactly; B pose -1 after n, partner pose all -1; partner codes [0, n)
                                          == the silence sequence; B words' texts == the cache item's words (mismatches counted with the cause);
                                          partner words / pieces empty; b_ok / split / row_key / rec_key / chunk_index / speakers / provenance.
  store   --store K                       store-level: pose_digest recomputed from the files on disk == POSE.json == STORES.json; load_node
                                          digest == STORES.json node_digest; DONE ids_blake2b recomputed; per recording k = 0..nk-1 with
                                          clip starts non-decreasing; rows.parquet / index / pose_index consistency.
  global                                  all stores: every cache item exactly once, counts per split == expected, split == held-out rule
                                          for every recording, recordings never span stores, frames balanced.
  loader  --ntrain 1000 --neval -1        data_aloe_pose (stores root passed explicitly, DIGESTS_FILE monkeypatched): splits(), Stores.conv(item,
                                          aloe_cfg(), 0), as_micro() on random train items + eval items (-1 = all) + every n < 2 item; micro
                                          length T / prefix P distributions, voice-clip fraction, failures with messages.
usage: python -I validate_v2pure_stores.py MODE --root /data/home/amartyavarma/ppm/aloe-v2pure-ft/stores [...]"""
import argparse
import collections
import hashlib
import json
import os
import sys
import time

for k_, v_ in (("S3_MIN_FRAMES", "1"), ("PPM_CD", "2"), ("QQA_A_EMPTY", "1"), ("QQA_POSE_B", "1")):
    os.environ[k_] = v_

import numpy as np  # noqa: E402

CODE = "/data/home/amartyavarma/ppm/aloe-v2pure-ft/code"
sys.path.insert(0, f"{CODE}/lib")
sys.path.insert(0, CODE)
import s3rows as R3  # noqa: E402   (the trainer's frozen copy)
import sysprompt as SPR  # noqa: E402

CACHE = "/data/home/amartyavarma/aloe-pose/sam-cache-active-v2pure"
SIL_CODES = "/data/home/amartyavarma/setup-logs/s3hestia/ourrows/silence_codes.json"
EXPECT = {"train": {"items": 367496, "frames": 92132615, "recordings": 12384}, "eval": {"items": 3482, "frames": 868932, "recordings": 113}}
SIDE_OF = {"speaker_1": "assistant", "speaker_2": "user"}
NST = 8


def is_eval(rid):
    return int.from_bytes(hashlib.blake2b(("aloe-pose-v1|" + rid).encode(), digest_size=8).digest(), "little") % 10000 < 100


def result(d):
    print("RESULT " + json.dumps(d), flush=True)


def prov_table(root, cols=None):
    import pyarrow.parquet as pq
    return pq.read_table(f"{root}/rows.parquet", columns=cols).to_pydict()


# ------------------------------------------------------------------------------------------------ rows
def mode_rows(a):
    import pyarrow.parquet as pq
    assert R3.MIN_FRAMES == 1
    t0 = time.time()
    root = f"{a.root}/p{a.store}"
    node = R3.load_node(root)
    rows = R3.Rows(node)
    N = len(node["length"])
    lo, hi = N * a.part // a.nparts, N * (a.part + 1) // a.nparts
    pv = prov_table(root, ["row", "row_key", "split", "src_key", "src_cache_shard", "src_item", "src_codes_off", "b_side", "n_frames",
                           "episode_id", "chunk_index", "recording_id", "active_speaker", "segment_id", "a_present", "partner"])
    assert pv["row"] == list(range(N))
    name = node["shards"][0]
    pz = {s: np.load(f"{root}/{name}.pose_{s}.npy", mmap_mode="r") for s in R3.SIDES}
    z = np.load(f"{root}/pose_index.npz")
    b_ok, split = z["b_ok"], z["split"]
    q = np.asarray(json.load(open(SIL_CODES))["seq"], np.int16)
    st = collections.Counter()
    bad = []
    by_shard = collections.defaultdict(list)
    for i in range(lo, hi):
        by_shard[pv["src_cache_shard"][i]].append(i)
    for sh in sorted(by_shard):
        d = f"{CACHE}/{sh}"
        it = pq.read_table(f"{d}/items.parquet", columns=["key", "rid", "speaker", "segment_id", "n", "words"]).to_pydict()
        off = np.load(f"{d}/codes_off.npy")
        cc = np.load(f"{d}/codes.npy", mmap_mode="r"); cp = np.load(f"{d}/pose.npy", mmap_mode="r")
        for i in by_shard[sh]:
            try:
                row = rows[i]
                R3.check_row(row, strict=True)
                n = int(row["n_frames"]); j = int(pv["src_item"][i]); rid = it["rid"][j]
                bs = pv["b_side"][i]; ps = "user" if bs == "assistant" else "assistant"
                assert it["key"][j] == pv["src_key"][i] == f"{rid}/{it['speaker'][j]}/{it['segment_id'][j]}", "src key"
                assert SIDE_OF[it["speaker"][j]] == bs and pv["active_speaker"][i] == it["speaker"][j] and pv["segment_id"][i] == it["segment_id"][j]
                assert int(it["n"][j]) == n == pv["n_frames"][i] and int(off[j + 1] - off[j]) == n and int(off[j]) == pv["src_codes_off"][i]
                k = int(row["chunk_index"])
                assert row["row_key"] == pv["row_key"][i] == f"ALOE/aloe-v2pure/{rid}/c{k:05d}" and k == pv["chunk_index"][i], "row_key"
                assert SPR.rec_key(row["row_key"]) == f"ALOE/aloe-v2pure/{rid}", "rec_key"
                assert row["episode_id"] == rid == pv["recording_id"][i] == pv["episode_id"][i] and row["podcast_id"] == "aloe-v2pure"
                assert row["dataset"] == "ALOE" and row["frame_start"] == 7500 * k and row["frame_end"] == 7500 * k + n
                assert row["assistant_speaker"] == f"{rid}/speaker_1" and row["user_speaker"] == f"{rid}/speaker_2"
                assert row["assistant_speaker_dir"] == "speaker_1" and row["user_speaker_dir"] == "speaker_2"
                assert pv["a_present"][i] is False and pv["partner"][i] == "absent"
                # codes / pose vs the cache, exactly
                assert np.array_equal(np.asarray(row[f"codes_{bs}"])[:, :n], np.asarray(cc[off[j]:off[j + 1]]).T), "B Mimi != cache"
                assert np.array_equal(np.asarray(row[f"codes_{ps}"])[:, :n], q[:, np.arange(n) % q.shape[1]]), "partner != silence codes"
                pb, pp = np.asarray(pz[bs][i]), np.asarray(pz[ps][i])
                assert np.array_equal(pb[:, :n], np.asarray(cp[off[j]:off[j + 1]]).T), "B pose != cache"
                assert (pb[:, n:] == -1).all() and (pp == -1).all() and (pb[:, :n] >= 0).all(), "pose padding / partner pose"
                assert b_ok[i].tolist() == [int(bs == "user"), int(bs == "assistant")], "b_ok"
                sp = "eval" if is_eval(rid) else "train"
                assert pv["split"][i] == sp and int(split[i]) == int(sp == "eval") and ((sh == "eval") == (sp == "eval")), "split"
                # words
                assert row[f"words_{ps}"] == [] and len(row[f"piece_ids_{ps}"]) == 0 and list(row[f"piece_offsets_{ps}"]) == [0], "partner words"
                cw = json.loads(it["words"][j])
                ct = [str(x[2]).strip() for x in cw]
                bt = [w["text"] for w in row[f"words_{bs}"]]
                st["words_cache"] += len(ct); st["words_store"] += len(bt)
                if bt != ct:
                    kept = [str(x[2]).strip() for x in cw if float(x[0]) >= -0.5 and not R3_CJK(str(x[2]))]
                    st["rows_words_differ"] += 1
                    if bt == kept:
                        st["rows_words_differ_explained_by_drop_rules"] += 1
                    else:
                        bad.append([row["row_key"], "words differ beyond the drop rules", bt[:5], ct[:5]])
                st["w_start_clamped"] += sum(1 for x in cw if -0.5 <= float(x[0]) < 0)
                st["rows_" + sp] += 1; st["frames_" + sp] += n; st[f"brows_{sp}_{bs}"] += 1
            except Exception as e:  # noqa: BLE001
                bad.append([pv["row_key"][i], repr(e)[:300]])
            st["rows_checked"] += 1
    result({"mode": "rows", "store": a.store, "part": a.part, "nparts": a.nparts, "rows": [lo, hi], "stats": dict(st), "n_bad": len(bad),
            "bad": bad[:20], "sec": round(time.time() - t0, 1)})


_CJK = [None]


def R3_CJK(s):
    if _CJK[0] is None:
        import re
        _CJK[0] = re.compile(r"[　-鿿가-힯]")          # = s2h_to_s3rows.CJK
    return bool(_CJK[0].search(s))


# ------------------------------------------------------------------------------------------------ store
def mode_store(a):
    t0 = time.time()
    root = f"{a.root}/p{a.store}"
    S = json.load(open(f"{a.root}/STORES.json"))[f"p{a.store}"]
    pj = json.load(open(f"{root}/POSE.json"))
    nj = json.load(open(f"{root}/NODE.json"))
    node = R3.load_node(root)
    name = node["shards"][0]
    N = len(node["length"])
    out = {"mode": "store", "store": a.store, "rows": N}
    z = np.load(f"{root}/pose_index.npz")
    b_ok, split = z["b_ok"], z["split"]
    h = hashlib.blake2b(digest_size=16)
    for s in R3.SIDES:
        with open(f"{root}/{name}.pose_{s}.npy", "rb") as f:
            while True:
                b = f.read(1 << 26)
                if not b:
                    break
                h.update(b)
    h.update(b_ok.tobytes()); h.update(split.tobytes())
    out["pose_digest_ok"] = bool(h.hexdigest() == pj["pose_digest"] == S["pose_digest"])
    out["node_digest_ok"] = bool(node["digest"] == S["node_digest"] and S["rows"] == N == pj["rows"] == nj["total_rows"])
    import pyarrow as pa
    import pyarrow.ipc as ipc
    tb = ipc.open_file(pa.memory_map(f"{root}/{name}.words.arrow", "r")).read_all()
    keys = tb.column("row_key").to_pylist()
    out["ids_digest_ok"] = bool(R3.shard_digest(keys, node["length"]) == nj["ids_blake2b"][0] == pj["node_digest"])
    pv = prov_table(root, ["row", "row_key", "split", "n_frames", "recording_id", "chunk_index", "clip_start_sec", "n_clips_rec", "b_side"])
    out["prov_keys_ok"] = bool(pv["row_key"] == keys and pv["n_frames"] == node["length"].tolist())
    out["split_ok"] = bool([("eval" if x else "train") for x in split] == pv["split"] and len(b_ok) == N and (b_ok.sum(1) == 1).all())
    by = collections.defaultdict(list)
    for i in range(N):
        by[pv["recording_id"][i]].append((pv["chunk_index"][i], pv["clip_start_sec"][i], pv["n_clips_rec"][i]))
    ok = True
    for rid, L in by.items():
        ks = [x[0] for x in L]; ss = [x[1] for x in L]
        ok &= ks == list(range(len(L))) and all(b >= a_ for a_, b in zip(ss, ss[1:])) and all(x[2] == len(L) for x in L)
    out["ordinals_ok"] = bool(ok)
    ps = __import__("persona_s3q").PersonaSidecar(node)
    rng = np.random.default_rng(a.store)
    pk = True
    for i in rng.choice(N, min(N, 2000), replace=False):
        p = ps.get(int(i), keys[int(i)])
        pk &= all(p[s]["status"] == "missing" and p[s]["prompt_tokens"] == -1 for s in R3.SIDES)
    out["persona_missing_ok"] = bool(pk)
    out["recordings"] = len(by)
    out["frames"] = {sp: int(sum(n for n, s_ in zip(pv["n_frames"], pv["split"]) if s_ == sp)) for sp in ("train", "eval")}
    out["rows_split"] = {sp: int(sum(1 for s_ in pv["split"] if s_ == sp)) for sp in ("train", "eval")}
    out["sizes_gb"] = round(sum(os.path.getsize(f"{root}/{f}") for f in os.listdir(root)) / 1e9, 2)
    out["sec"] = round(time.time() - t0, 1)
    out["ok"] = all(bool(out[k]) for k in ("pose_digest_ok", "node_digest_ok", "ids_digest_ok", "prov_keys_ok", "split_ok", "ordinals_ok", "persona_missing_ok"))
    result(out)


# ------------------------------------------------------------------------------------------------ global
def mode_global(a):
    import pyarrow.parquet as pq
    keys, rec_store, out = collections.Counter(), {}, {"mode": "global"}
    tot = {sp: {"items": 0, "frames": 0, "recordings": set()} for sp in ("train", "eval")}
    span = 0
    per = {}
    for k in range(NST):
        pv = prov_table(f"{a.root}/p{k}", ["src_key", "split", "n_frames", "recording_id"])
        keys.update(pv["src_key"])
        fr = {"train": 0, "eval": 0}
        for sk, sp, n, rid in zip(pv["src_key"], pv["split"], pv["n_frames"], pv["recording_id"]):
            tot[sp]["items"] += 1; tot[sp]["frames"] += n; tot[sp]["recordings"].add(rid); fr[sp] += n
            if rec_store.setdefault(rid, k) != k:
                span += 1
            assert sp == ("eval" if is_eval(rid) else "train"), (rid, sp)
        per[f"p{k}"] = fr
    cache = collections.Counter()
    names = sorted(json.load(open(f"{CACHE}/plan.json"))["shards"])
    for sh in names:
        cache.update(pq.read_table(f"{CACHE}/{sh}/items.parquet", columns=["key"]).column("key").to_pylist())
    out["cache_items"] = sum(cache.values())
    out["every_cache_item_once"] = keys == cache and max(keys.values()) == 1
    out["recordings_spanning_stores"] = span
    got = {sp: {"items": v["items"], "frames": v["frames"], "recordings": len(v["recordings"])} for sp, v in tot.items()}
    out["totals"] = got
    out["counts_ok"] = got == EXPECT
    out["frames_per_store"] = per
    out["ok"] = out["every_cache_item_once"] and span == 0 and out["counts_ok"]
    result(out)


# ------------------------------------------------------------------------------------------------ loader
def mode_loader(a):
    import importlib
    t0 = time.time()
    DA = importlib.import_module("data_aloe_pose")
    roots = [f"{a.root}/p{k}" for k in range(a.nstores)]
    DA.DIGESTS_FILE = f"{a.root}/STORES.json"                       # monkeypatch (module constant; data_aloe_pose.py itself untouched)
    assert DA.A_EMPTY and DA.POSE_B and DA.D.CD == 2 and DA.D.R3.MIN_FRAMES == 1, (DA.A_EMPTY, DA.POSE_B, DA.D.CD, DA.D.R3.MIN_FRAMES)
    sp, nrows = DA.splits(roots)
    out = {"mode": "loader", "splits_items": {k: len(v) for k, v in sp.items()}, "splits_rows": nrows}
    import pyarrow.parquet as pq
    nf = {}
    for k, r in enumerate(roots):
        for i, n in enumerate(pq.read_table(f"{r}/rows.parquet", columns=["n_frames"]).column("n_frames").to_pylist()):
            nf[(k, i)] = n
    rng = np.random.default_rng(20261017)
    tr = [sp["train"][i] for i in rng.choice(len(sp["train"]), min(a.ntrain, len(sp["train"])), replace=False)]
    ev = sp["eval"] if a.neval < 0 else [sp["eval"][i] for i in rng.choice(len(sp["eval"]), min(a.neval, len(sp["eval"])), replace=False)]
    short = [it for s_ in ("train", "eval") for it in sp[s_] if nf[(it[0], it[1])] < 2]
    cfg = DA.aloe_cfg()
    st = DA.Stores(roots)
    res = {}
    for nm, items in (("train", tr), ("eval", ev), ("n_lt2", short)):
        T, P, voice, fails, swaps = [], [], 0, [], collections.Counter()
        for it in items:
            try:
                m = st.conv(it, cfg, 0)
                mm = DA.as_micro(m)
                T.append(len(mm["pid"])); P.append(int(m["P"])); swaps[int(m["swap"])] += 1
                pvv = json.loads(m["prov"])
                voice += int(bool(pvv.get("clips")))
                assert mm["kind"] == "pose" and mm["pdepth"].shape == (len(mm["pid"]), 32)
                assert len(mm["pid"]) == int(m["P"]) + nf[(it[0], it[1])] + DA.D.CD
            except Exception as e:  # noqa: BLE001
                fails.append([list(map(int, it)), nf[(it[0], it[1])], repr(e)[:200]])
        q = lambda x: {p: (float(np.percentile(x, p)) if x else None) for p in (0, 5, 25, 50, 75, 95, 99, 100)}  # noqa: E731
        res[nm] = {"n": len(items), "ok": len(T), "fail": len(fails), "fails": fails[:10],
                   "fail_n_frames": dict(collections.Counter(f[1] for f in fails)) if fails else {},
                   "T": q(T), "T_mean": float(np.mean(T)) if T else None, "P": q(P), "P_values": dict(collections.Counter(P).most_common(8)),
                   "voice_clip_frac": round(voice / max(len(T), 1), 4), "swap": dict(swaps)}
    out["items"] = res
    if hasattr(DA, "make_plan_v2"):                                  # the trainer's v2 plan (being edited in parallel; informational only)
        try:
            pl = DA.make_plan_v2(64, cfg, roots)
            out["plan_v2"] = {k: pl[k] for k in ("steps", "n_rows", "n_train_items", "n_eval_items", "train_frames", "eval_frames", "max_micro_tok_ub")}
        except Exception as e:  # noqa: BLE001
            out["plan_v2"] = {"error": repr(e)[:300]}
    out["sec"] = round(time.time() - t0, 1)
    out["ok"] = all(int(n) < 2 for nm in ("train", "eval") for n in res[nm]["fail_n_frames"]) and \
        res["train"]["fail"] == sum(1 for it in tr if nf[(it[0], it[1])] < 2) and res["eval"]["fail"] == sum(1 for it in ev if nf[(it[0], it[1])] < 2)
    out["only_n_lt2_fail"] = out["ok"]
    result(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("rows", "store", "global", "loader"))
    ap.add_argument("--root", default="/data/home/amartyavarma/ppm/aloe-v2pure-ft/stores")
    ap.add_argument("--store", type=int, default=0); ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--ntrain", type=int, default=1000); ap.add_argument("--neval", type=int, default=-1)
    ap.add_argument("--nstores", type=int, default=NST)
    a = ap.parse_args()
    {"rows": mode_rows, "store": mode_store, "global": mode_global, "loader": mode_loader}[a.mode](a)


if __name__ == "__main__":
    main()
