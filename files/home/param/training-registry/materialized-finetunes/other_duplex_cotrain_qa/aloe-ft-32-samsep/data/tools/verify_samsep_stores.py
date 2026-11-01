"""aloe-ft-32-samsep DATA: independent verification of the sam_sep stores vs the original s2h-m1 stores + the sam_sep HF parquet.

A  per store: every non-code file of the kept rows == the original's records (Arrow-level equality of words.arrow / persona.arrow tables
   incl. schema + metadata, index.npz arrays + dtypes, rows.jsonl lines byte-identical up to the renumbered row ids, rows.parquet columns),
   DONE / NODE.json consistent (s3rows.load_node, digest recomputed), codes files dtype / shape / header as the originals'.
B  per sam shard (re-read independently): every kept row's new codes == the isolated stream slice of its OWN lane (frames < n) and -1 after,
   in 0..2047, T == src_frames; difference rates vs our original codes (all CB / per CB); lane check: CB0 match of each new lane vs our
   same-lane and other-lane codes (flipped = other >= own).
C  the trainer's own data path on CPU (aloe-v2vox-asr-w15973/code: data_aloe_v2.Stores / speech_micro_v2, data_aloe_v2vox.speech_micro_v2vox):
   every first-chunk EVAL row x both swaps + 400 seeded first-chunk TRAIN rows x both swaps, built from the new AND the original stores:
   ids / present / pid / sel / y / word / pad / eob / nw / light / P / voice-clip provenance identical, depth differs only on B (and A in
   the non-vox micro), v2vox micro: channel A depth all -1 (no A Mimi fed).
D  PLAN.env regenerated with validate_rows' stage-3 plan rule (MODEL 1 trainer bookkeeping only; the Aloe FT does not read it).
-> <data>/VERIFY.json
usage: python -I verify_samsep_stores.py [--workers 32] [--n-train 400]"""
import argparse
import hashlib
import io
import json
import os
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
CODE = "/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973/code"
SH = "rows-00000-of-00001"
NODES = [f"m1n{k}" for k in range(4)]
LANE = {"assistant": "SPEAKER_00", "user": "SPEAKER_01"}
OTHER = {"assistant": "user", "user": "assistant"}


def rowmap():
    j = json.load(open(f"{DATA}/row_map.json"))
    out = {}
    for k, nd in enumerate(NODES):
        m = {int(o): n for o, n in j["map"][nd]["new_of_old"].items()}
        keep = [o for o, _ in sorted(m.items(), key=lambda x: x[1])]
        assert [m[o] for o in keep] == list(range(len(keep)))
        out[k] = keep
    return out, j


# ------------------------------------------------------------------------------------------------ A
def check_noncode(k, keep):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq
    nd = NODES[k]; s_, o_ = f"{SRC}/{nd}", f"{OUT}/{nd}"
    ki = pa.array(keep, pa.int64()); r = {}
    for fn in ("words", "persona"):
        a = ipc.open_file(pa.memory_map(f"{s_}/{SH}.{fn}.arrow", "r")); b = ipc.open_file(pa.memory_map(f"{o_}/{SH}.{fn}.arrow", "r"))
        ta, tb = a.read_all().take(ki), b.read_all()
        r[f"{fn}_schema_eq"] = a.schema.equals(b.schema, check_metadata=True)
        r[f"{fn}_records_eq"] = ta.equals(tb)
        r[f"{fn}_batches"] = [a.num_record_batches, b.num_record_batches]
        if fn == "words":
            keys = tb.column("row_key").to_pylist(); win = {c: tb.column(c).to_pylist() for c in ("frame_start", "frame_end", "n_frames", "chunk_index")}
    za, zb = np.load(f"{s_}/{SH}.index.npz"), np.load(f"{o_}/{SH}.index.npz")
    r["index_keys_eq"] = list(za.files) == list(zb.files)
    r["index_arrays_eq"] = all(np.array_equal(za[x][keep], zb[x]) and za[x].dtype == zb[x].dtype for x in za.files)
    la, lb = open(f"{s_}/rows.jsonl").read().splitlines(), open(f"{o_}/rows.jsonl").read().splitlines()
    bad = 0
    for nr, (i, l) in enumerate(zip(keep, lb)):
        p = json.loads(l)
        bad += not (p["row"] == nr and p["row_in_shard"] == nr and json.dumps(dict(p, row=i, row_in_shard=i)) == la[i])
    r["rows_jsonl_lines"] = len(lb); r["rows_jsonl_bad"] = bad + abs(len(lb) - len(keep))
    pa_, pb_ = pq.read_table(f"{s_}/rows.parquet").take(ki), pq.read_table(f"{o_}/rows.parquet")
    r["rows_parquet_schema_eq"] = pa_.schema.equals(pb_.schema)
    r["rows_parquet_cols_eq_except_row_ids"] = all(pa_.column(c).equals(pb_.column(c)) for c in pa_.column_names if c not in ("row", "row_in_shard"))
    r["rows_parquet_row_ids_ok"] = pb_.column("row").to_pylist() == list(range(len(keep))) == pb_.column("row_in_shard").to_pylist()
    r["rows_parquet_keys_eq_words"] = pb_.column("row_key").to_pylist() == keys
    r["windows_eq_parquet"] = all(pb_.column(c).to_pylist() == win[c] for c in win)
    na, nb = F.load_node(s_), F.load_node(o_)
    r["load_node_ok"] = True
    r["length_eq"] = bool(np.array_equal(na["length"][keep], nb["length"]))
    dg = F.shard_digest(keys, nb["length"])
    r["digest_recomputed_eq"] = dg == nb["meta"]["ids_blake2b"][0] == json.load(open(f"{o_}/{SH}.DONE"))["ids_blake2b"]
    r["digest"] = nb["digest"]; r["digest_old"] = na["digest"]
    r["node_meta_totals_ok"] = nb["meta"]["total_rows"] == len(keep) and nb["meta"]["total_frames"] == int(nb["length"].sum())
    for s in F.SIDES:
        ca = np.load(f"{s_}/{SH}.codes_{s}.npy", mmap_mode="r"); cb = np.load(f"{o_}/{SH}.codes_{s}.npy", mmap_mode="r")
        r[f"codes_{s}_dtype_shape_ok"] = ca.dtype == cb.dtype == np.int16 and ca.shape[1:] == cb.shape[1:] == (32, 7500) and cb.shape[0] == len(keep)
        r[f"codes_{s}_header_bytes"] = [int(ca.offset), int(cb.offset)]
        r[f"codes_{s}_fortran"] = [bool(ca.flags.f_contiguous and not ca.flags.c_contiguous), bool(cb.flags.f_contiguous and not cb.flags.c_contiguous)]
    tl = [x for x in open(f"{OUT}/eval_episodes.txt").read().splitlines() if not x.startswith("#")]
    r["n_rows"] = len(keep); r["n_rows_old"] = len(na["length"])
    return r, tl


# ------------------------------------------------------------------------------------------------ B
def check_codes(job):
    import pyarrow.parquet as pq
    shard, pairs = job
    new = {(k, s): np.load(f"{OUT}/{NODES[k]}/{SH}.codes_{s}.npy", mmap_mode="r") for k in range(4) for s in F.SIDES}
    old = {(k, s): np.load(f"{SRC}/{NODES[k]}/{SH}.codes_{s}.npy", mmap_mode="r") for k in range(4) for s in F.SIDES}
    pf = pq.ParquetFile(f"{HF}/{shard}")
    st = Counter(); diff = {s: np.zeros(32, np.int64) for s in F.SIDES}; frames = 0; rows = []
    cols = ["session", "seg", "T", "SPEAKER_00_isolated_npy", "SPEAKER_01_isolated_npy", "mix_SPEAKER_00_npy", "mix_SPEAKER_01_npy"]
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["session", "seg"]).to_pydict()
        if not any(f"{a}/{b}" in pairs for a, b in zip(t["session"], t["seg"])):
            continue
        for row in pf.read_row_group(rg, columns=cols).to_pylist():
            conv = f"{row['session']}/{row['seg']}"
            if conv not in pairs:
                continue
            T = int(row["T"])
            iso = {s: np.load(io.BytesIO(row[f"{LANE[s]}_isolated_npy"])) for s in F.SIDES}
            mix = {s: np.load(io.BytesIO(row[f"mix_{LANE[s]}_npy"])) for s in F.SIDES}
            for (k, nr, orr, fs, n, srcT, rk, split, ci) in pairs[conv]:
                st["rows"] += 1; frames += n
                st["T_eq_src_frames"] += T == srcT
                rec = {"row_key": rk, "store": k, "new_row": nr, "split": split, "chunk_index": ci, "n": n}
                for s in F.SIDES:
                    c = np.asarray(new[(k, s)][nr]); o = np.asarray(old[(k, s)][orr]); oo = np.asarray(old[(k, OTHER[s])][orr])
                    st[f"{s}_eq_iso_own"] += bool(np.array_equal(c[:, :n], iso[s][:, fs:fs + n]))
                    st[f"{s}_eq_iso_other"] += bool(np.array_equal(c[:, :n], iso[OTHER[s]][:, fs:fs + n]))
                    st[f"{s}_eq_mix"] += bool(np.array_equal(c[:, :n], mix[s][:, fs:fs + n]))
                    st[f"{s}_pad_ok"] += bool((c[:, n:] == F.PAD_CODE).all())
                    st[f"{s}_range_ok"] += bool(c[:, :n].min() >= 0 and c[:, :n].max() < F.CBS)
                    st[f"{s}_identical_to_old"] += bool(np.array_equal(c, o))
                    d = (c[:, :n] != o[:, :n]).sum(1); diff[s] += d
                    own, oth = float((c[0, :n] == o[0, :n]).mean()), float((c[0, :n] == oo[0, :n]).mean())
                    rec[f"cb0_own_{s}"], rec[f"cb0_oth_{s}"], rec[f"diff_all_{s}"], rec[f"diff_cb0_{s}"] = own, oth, float(d.sum() / (32 * n)), float(d[0] / n)
                    st[f"{s}_flipped"] += oth >= own
                rows.append(rec)
    return shard, dict(st), {s: diff[s].tolist() for s in F.SIDES}, frames, rows


# ------------------------------------------------------------------------------------------------ C
_ST = {}


def _stores(which):
    for p in (os.path.join(CODE, "lib"), CODE):
        if p not in sys.path:
            sys.path.insert(0, p)
    import data_aloe_v2 as A2
    if which not in _ST:
        if which == "new":
            new_dg = [F.load_node(f"{OUT}/{nd}")["digest"] for nd in NODES]
            st = A2.Stores(roots=[f"{OUT}/{nd}" for nd in NODES])
            saved = list(A2.DIGESTS); A2.DIGESTS[:] = new_dg
            for k in range(4):
                st.open(k)                                               # digest asserted == the new stores' (patched list), then restored
            A2.DIGESTS[:] = saved
        else:
            st = A2.Stores(roots=[f"{SRC}/{nd}" for nd in NODES])
            for k in range(4):
                st.open(k)                                               # digest asserted == Model 1's (the reference's hard-coded DIGESTS)
        _ST[which] = st
    return A2, _ST[which]


def micro_pair(job):
    """(k, new_r, old_r, swap, split) -> comparison of the trainer's micros built from the new vs the original store."""
    k, nr, orr, sw, split = job
    A2, sn = _stores("new"); _, so = _stores("old")
    import data_aloe_v2vox as V
    cfg = A2.eval_cfg() if split == "eval" else A2.aloe_cfg()
    out = {}
    for which, st, r in (("new", sn, nr), ("old", so, orr)):
        node, rows, sib = st.s[k]
        row = rows[int(r)]
        A2.D.R3.check_row(row)
        m2 = A2.speech_micro_v2(row, int(r), sib, cfg, 0, sw)
        mv = V.speech_micro_v2vox(row, int(r), sib, cfg, 0, sw)
        out[which] = (row, m2, mv)
    (rn, mn, vn), (ro, mo, vo) = out["new"], out["old"]
    res = {"row_key": rn["row_key"], "swap": sw, "split": split}
    res["row_key_eq"] = rn["row_key"] == ro["row_key"]
    same = ("ids", "present", "pid", "sel", "y", "word", "pad", "eob", "nw", "light")
    res["v2_same_fields"] = all(np.array_equal(mn[f], mo[f]) for f in same) and mn["P"] == mo["P"]
    pn, po = json.loads(mn["prov"]), json.loads(mo["prov"])
    res["prov_eq"] = pn == po
    res["voice_clip"] = int(pn.get("voice_clip", 0))
    P = int(mn["P"]); n = int(rn["n_frames"])
    dn, do = np.asarray(mn["depth"]), np.asarray(mo["depth"])
    res["depth_mask_eq"] = bool(np.array_equal(dn < 0, do < 0))
    res["B_conv_codes_differ"] = bool((dn[1, P:] != do[1, P:]).any())
    res["B_prefix_codes_differ"] = bool((dn[1, :P] != do[1, :P]).any()) if res["voice_clip"] else None
    res["v2vox_same_fields"] = all(np.array_equal(vn[f], vo[f]) for f in ("ids", "present", "pid")) and vn["vin"] == vo["vin"]
    res["v2vox_A_depth_all_neg"] = bool((np.asarray(vn["depth"])[0] < 0).all())
    res["v2vox_A_present_conv_zero"] = bool((np.asarray(vn["present"])[0, P:] == 0).all())
    # B's conversation codes in the micro = this row's new codes of the B data side (CD-delayed lanes: compare as sets of frames per CB0)
    bside = "user" if sw else "assistant"
    c0 = np.asarray(rn[f"codes_{bside}"])[0, :n]
    dcol = dn[1, P:, 0]; dcol = dcol[dcol >= 0]
    res["B_cb0_seq_eq_new_codes"] = bool(len(dcol) == n and np.array_equal(dcol, c0))
    return res


# ------------------------------------------------------------------------------------------------ D
def plan_env():
    Q = "/data/home/amartyavarma/qwen-1ch"
    for p in (f"{Q}/stage3/data/rows_build/scripts", f"{Q}/stage2", f"{Q}/stage3", Q):
        if p not in sys.path:
            sys.path.insert(0, p)
    import data_s3q as DS
    out = {}
    for root in (SRC, OUT):
        steps_min, ev_total = 10 ** 9, 0
        for nd in NODES:
            node = DS.R3.load_node(f"{root}/{nd}"); idx = DS.load_index(f"{root}/{nd}")
            for local in range(8):
                pl = DS.plan(node, idx, 20261009, 8, local, 1, 1, 0.0, "m1", row_exclude=None, eval_list=f"{root}/eval_episodes.txt")
                steps_min = min(steps_min, len(pl["rows"]))
            ev_total += pl["stats"]["eval_episode_rows_node"]
        out[root] = {"M1_STEPS": steps_min, "M1_EVAL_ROWS_PER_RANK": ev_total // 4 // 8}
    return out


def main():
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=32); ap.add_argument("--n-train", type=int, default=400)
    a = ap.parse_args()
    t0 = time.time()
    V = {"stores": OUT, "src": SRC, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    keep, rm = rowmap()
    # A
    V["A_noncode"] = {}
    for k, nd in enumerate(NODES):
        V["A_noncode"][nd], tl = check_noncode(k, keep[k])
        print(f"[A] {nd} {V['A_noncode'][nd]} ({time.time() - t0:.0f}s)", flush=True)
    ev_old = [x for x in open(f"{SRC}/eval_episodes.txt").read().splitlines() if not x.startswith("#")]
    V["A_eval_episodes"] = {"old": len(ev_old), "new": len(tl), "new_subset_of_old": set(tl) <= set(ev_old)}
    # B
    pidx = pq.read_table(f"{HF}/index/pairs.parquet", columns=["session", "seg", "token_shard"]).to_pydict()
    sam = {f"{s}/{g}": sh for s, g, sh in zip(pidx["session"], pidx["seg"], pidx["token_shard"])}
    by_shard, allrows = {}, []
    for k, nd in enumerate(NODES):
        t = pq.read_table(f"{SRC}/{nd}/rows.parquet").to_pydict()
        for nr, i in enumerate(keep[k]):
            c = t["src_conversation"][i]
            tup = (k, nr, i, int(t["frame_start"][i]), int(t["n_frames"][i]), int(t["src_frames"][i]), t["row_key"][i], t["split"][i], int(t["chunk_index"][i]))
            by_shard.setdefault(sam[c], {}).setdefault(c, []).append(tup); allrows.append(tup)
        ks = set(keep[k])
        V.setdefault("dropped_not_in_sam", 0)
        V["dropped_not_in_sam"] += sum(1 for i, c in enumerate(t["src_conversation"]) if i not in ks and c not in sam)
        V.setdefault("dropped_in_sam_BAD", 0)
        V["dropped_in_sam_BAD"] += sum(1 for i, c in enumerate(t["src_conversation"]) if i not in ks and c in sam)
        V.setdefault("kept_not_in_sam_BAD", 0)
        V["kept_not_in_sam_BAD"] += sum(1 for i in keep[k] if t["src_conversation"][i] not in sam)
    st, diff, frames, rows = Counter(), {s: np.zeros(32, np.int64) for s in F.SIDES}, 0, []
    with Pool(a.workers) as pool:
        for sh, s_, d_, f_, r_ in pool.imap_unordered(check_codes, sorted(by_shard.items())):
            st.update(s_); frames += f_; rows += r_
            for s in F.SIDES:
                diff[s] += np.asarray(d_[s])
    print(f"[B] {dict(st)} ({time.time() - t0:.0f}s)", flush=True)
    rate = {s: (diff[s] / frames) for s in F.SIDES}
    flipped = [r for r in rows if r["cb0_oth_user"] >= r["cb0_own_user"] or r["cb0_oth_assistant"] >= r["cb0_own_assistant"]]
    marg = {s: np.array([r[f"cb0_own_{s}"] - r[f"cb0_oth_{s}"] for r in rows]) for s in F.SIDES}
    V["B_codes"] = {"rows": st["rows"], "frames": frames, "counts": dict(st),
                    "diff_frac_all_cb": {s: round(float(diff[s].sum() / (32 * frames)), 4) for s in F.SIDES},
                    "diff_frac_cb0": {s: round(float(rate[s][0]), 4) for s in F.SIDES},
                    "diff_frac_cb1_3": {s: [round(float(x), 4) for x in rate[s][1:4]] for s in F.SIDES},
                    "diff_frac_cb4_31_range": {s: [round(float(rate[s][4:].min()), 4), round(float(rate[s][4:].max()), 4)] for s in F.SIDES},
                    "row_diff_cb0_pctl_5_50_95": {s: [round(float(np.percentile([r[f"diff_cb0_{s}"] for r in rows], q)), 4) for q in (5, 50, 95)] for s in F.SIDES},
                    "lane_check": {"rows_any_lane_flipped": len(flipped), "flipped_rows": [r["row_key"] for r in flipped][:50],
                                   "cb0_match_own_median": {s: round(float(np.median([r[f"cb0_own_{s}"] for r in rows])), 4) for s in F.SIDES},
                                   "cb0_match_other_median": {s: round(float(np.median([r[f"cb0_oth_{s}"] for r in rows])), 4) for s in F.SIDES},
                                   "own_minus_other_min": {s: round(float(marg[s].min()), 4) for s in F.SIDES},
                                   "own_minus_other_p1": {s: round(float(np.percentile(marg[s], 1)), 4) for s in F.SIDES},
                                   "rows_margin_lt_0.2": {s: int((marg[s] < 0.2).sum()) for s in F.SIDES}}}
    lows = sorted(rows, key=lambda r: min(marg_ for marg_ in (r["cb0_own_user"] - r["cb0_oth_user"], r["cb0_own_assistant"] - r["cb0_oth_assistant"])))[:10]
    V["B_codes"]["lane_check"]["lowest_margin_rows"] = [{k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in r.items()} for r in lows]
    # C
    import random
    first = [(t[0], t[1], t[2], t[7]) for t in allrows if t[8] == 0]
    ev = [x for x in first if x[3] == "eval"]
    tr = [x for x in first if x[3] == "train"]
    random.Random(20261024).shuffle(tr)
    jobs = [(k, nr, orr, sw, sp) for (k, nr, orr, sp) in ev + tr[:a.n_train] for sw in (0, 1)]
    with Pool(min(a.workers, 24)) as pool:
        res = pool.map(micro_pair, jobs, chunksize=4)
    keysC = ("row_key_eq", "v2_same_fields", "prov_eq", "depth_mask_eq", "B_conv_codes_differ", "v2vox_same_fields", "v2vox_A_depth_all_neg",
             "v2vox_A_present_conv_zero", "B_cb0_seq_eq_new_codes")
    V["C_trainer_path"] = {"micros": len(res), "rows_eval_first": len(ev), "rows_train_sampled": min(a.n_train, len(tr)),
                           **{f"n_{x}": sum(bool(r[x]) for r in res) for x in keysC},
                           "voice_clip_micros": sum(r["voice_clip"] for r in res),
                           "voice_clip_B_prefix_codes_differ": sum(bool(r["B_prefix_codes_differ"]) for r in res if r["voice_clip"]),
                           "fails": [r for r in res if not all(r[x] for x in keysC)][:10]}
    print(f"[C] {json.dumps({k_: v for k_, v in V['C_trainer_path'].items() if k_ != 'fails'})} ({time.time() - t0:.0f}s)", flush=True)
    sys.path.insert(0, os.path.join(CODE, "lib")); sys.path.insert(0, CODE)
    import data_aloe_v2 as A2
    spn = A2.splits(roots=[f"{OUT}/{nd}" for nd in NODES]); spo = A2.splits(roots=A2.SRC_STORES)
    V["C_splits"] = {"new": {"n_first": spn["n_first"], "n_all": spn["n_all"]}, "old": {"n_first": spo["n_first"], "n_all": spo["n_all"]}}
    # D
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(1) as p1:                         # fresh process: stage3's data_s3q, not the trainer's frozen copy
        pe = p1.apply(plan_env)
    old_env = open(f"{SRC}/PLAN.env").read()
    V["D_plan_env"] = {"recomputed_old": pe[SRC], "old_file": old_env, "new": pe[OUT],
                       "old_reproduced": old_env == f"M1_STEPS={pe[SRC]['M1_STEPS']}\nM1_EVAL_ROWS_PER_RANK={pe[SRC]['M1_EVAL_ROWS_PER_RANK']}\n"}
    if V["D_plan_env"]["old_reproduced"]:
        open(f"{OUT}/PLAN.env", "w").write(f"M1_STEPS={pe[OUT]['M1_STEPS']}\nM1_EVAL_ROWS_PER_RANK={pe[OUT]['M1_EVAL_ROWS_PER_RANK']}\n")
    # verdict
    A_ok = all(all(v for v in r.values() if isinstance(v, bool)) and r["rows_jsonl_bad"] == 0 for r in V["A_noncode"].values())
    A_ok = A_ok and V["A_eval_episodes"]["new_subset_of_old"]
    n = st["rows"]
    B_ok = (n == len(allrows) and st["T_eq_src_frames"] == n and all(st[f"{s}_{x}"] == n for s in F.SIDES for x in ("eq_iso_own", "pad_ok", "range_ok"))
            and all(st[f"{s}_identical_to_old"] == 0 for s in F.SIDES) and len(flipped) == 0)
    C_ok = all(V["C_trainer_path"][f"n_{x}"] == len(res) for x in keysC)
    V["verdict"] = {"A_noncode_identical": A_ok, "B_codes_ok": B_ok, "C_trainer_path_ok": C_ok, "dropped_ok": V["dropped_in_sam_BAD"] == 0 == V["kept_not_in_sam_BAD"],
                    "plan_env_ok": V["D_plan_env"]["old_reproduced"]}
    V["verdict"]["ALL_OK"] = bool(all(V["verdict"].values()))
    V["time_s"] = round(time.time() - t0, 1)
    json.dump(V, open(f"{DATA}/VERIFY.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("[verdict] " + json.dumps(V["verdict"]), flush=True)


if __name__ == "__main__":
    main()
