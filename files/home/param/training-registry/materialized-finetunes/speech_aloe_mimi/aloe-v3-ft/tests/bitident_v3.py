"""v3 row vs Model 1 chunk-0 row (spec item 1): on N calls, the first 7,500 frames (= Model 1's chunk 0, n1 = min(T, 7500)) of the v3 whole-call
row must be bit-identical to Model 1's chunk-0 row in codes, words and trainer targets.
  codes    both sides, frames [0, n1).
  words    both sides: words starting before 600 s: text, start, start_frame, in_overlap, piece ids identical; end / end_frame identical except
           words that cross 600 s (Model 1 clamps their end to its row end; v3 keeps the true end) -> counted, listed.
  targets  the trainer micro (v2 data path data_aloe_v2.speech_micro_v2 on the Model 1 row vs v3 path data_aloe_v3.speech_micro_v3 (vox
           False) on the v3 row), BOTH orientations, train cfg (light 0.1) and eval cfg (light 1.0): conversation part aligned at P:
           channel A + B ids / depth at conversation frames [0, n1), B targets (y, word, pad, eob, nw) at scored frames f < n1 - 1, cb0 labels,
           light frames < n1. Prefix compared too (identical unless the voice clip source differs: v2 = a later Model-1 chunk, v3 = the tail).
  vox      the v3 vox index entry == aloe-m1vox-ft's M1VOX_INDEX entry for the same chunk-0 row (flac, bytes, generation, gain) both sides.
usage: python -I bitident_v3.py [N]"""
import json
import os
import sys

os.environ.setdefault("S3_MIN_FRAMES", "125")
os.environ.setdefault("PPM_CD", "2")
os.environ["ALOE_STORE_ROOT"] = "/data/home/amartyavarma/ppm/aloe-v3-ft/stores"
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-v3-ft/code")
import numpy as np  # noqa: E402
import data_aloe_v3 as A3  # noqa: E402
import data_aloe_v2 as A2  # noqa: E402   (v2's data module, byte-identical copy)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
D = A3.D
M1 = A2.SRC_STORES
assert open(A2.__file__, "rb").read() == open("/data/home/amartyavarma/ppm/aloe-v2-ft/code/data_aloe_v2.py", "rb").read(), "data_aloe_v2 copy != v2's"
s2, s3 = A2.Stores(M1), A3.Stores(A3.SRC_STORES)
c2, c3 = A2.aloe_cfg(), A3.aloe_cfg()
e2, e3 = A2.eval_cfg(), A3.eval_cfg()
assert {k: v for k, v in c3.items() if k != "clip_src"} == c2, "v3 data cfg != v2's except clip_src"
mv = json.load(open("/data/home/amartyavarma/ppm/aloe-m1vox-ft/plan/M1VOX_INDEX.json"))["rows"]

# pick N calls: spread over stores and lengths (short < 7500, 7500..13500, > 13500), incl. 2 purged calls
import pyarrow.parquet as pq  # noqa: E402
cands = []
for k in range(4):
    t3 = pq.read_table(f"{A3.SRC_STORES[k]}/rows.parquet", columns=["row", "row_key", "chunk_index", "n_frames", "call_frames", "src_kind", "m1_rows"]).to_pydict()
    for i in range(len(t3["row"])):
        if t3["chunk_index"][i] == 0:
            m1r = json.loads(t3["m1_rows"][i])
            cands.append((k, t3["row"][i], m1r[0][1], t3["call_frames"][i], t3["src_kind"][i], t3["row_key"][i]))
rng = np.random.default_rng(7)
pick = []
for lo, hi, m in ((0, 7500, 5), (7500, 13500, 8), (13500, 10 ** 9, 5)):
    pool = [c for c in cands if lo <= c[3] < hi and c[4] == "handover-full"]
    pick += [pool[j] for j in rng.choice(len(pool), m, replace=False)]
pool = [c for c in cands if c[4] == "m1chunks" and c[3] > 7500]
pick += [pool[j] for j in rng.choice(len(pool), 2, replace=False)]
pick = pick[:N]
res = {"calls": 0, "codes_ok": 0, "words_ok": 0, "targets_ok": 0, "prefix_same": 0, "prefix_diff_clip": 0, "vox_ok": 0, "vox_n": 0,
       "words_end_clamped_m1": 0, "fails": []}
for k, r3, r1, T, kind, rk in pick:
    res["calls"] += 1
    n2, st2, sib2 = s2.open(k)
    n3, st3, sib3 = s3.open(k)
    row1, row3 = st2[r1], st3[r3]
    assert row1["row_key"] == row3["row_key"] == rk and int(row1["chunk_index"]) == 0
    n1 = int(row1["n_frames"])
    assert n1 == min(T, 7500) or kind == "m1chunks", (rk, n1, T)
    ok = all(np.array_equal(np.asarray(row1[f"codes_{s}"])[:, :n1], np.asarray(row3[f"codes_{s}"])[:, :n1]) for s in ("user", "assistant"))
    res["codes_ok"] += ok
    if not ok:
        res["fails"].append((rk, "codes"))
    wok = True
    for s in ("user", "assistant"):
        w1, w3 = row1[f"words_{s}"], [w for w in row3[f"words_{s}"] if w["start"] < n1 / 12.5]
        p1, o1 = np.asarray(row1[f"piece_ids_{s}"]), np.asarray(row1[f"piece_offsets_{s}"])
        p3, o3 = np.asarray(row3[f"piece_ids_{s}"]), np.asarray(row3[f"piece_offsets_{s}"])
        if len(w1) != len(w3):
            wok = False; res["fails"].append((rk, s, "n words", len(w1), len(w3))); continue
        for j, (a, b) in enumerate(zip(w1, w3)):
            same = all(a[f] == b[f] for f in ("text", "start", "start_frame", "in_overlap")) and np.array_equal(p1[o1[j]:o1[j + 1]], p3[o3[j]:o3[j + 1]])
            if a["end"] != b["end"] or a["end_frame"] != b["end_frame"]:
                if abs(a["end"] - n1 / 12.5) < 1e-9 and b["end"] > a["end"]:
                    res["words_end_clamped_m1"] += 1
                else:
                    same = False
            if not same:
                wok = False; res["fails"].append((rk, s, "word", j, a["text"], b["text"])); break
    res["words_ok"] += wok
    tok_ = True
    for cfg2, cfg3, tag in ((c2, c3, "train"), (e2, e3, "eval")):
        for sw in (0, 1):
            m1 = A2.speech_micro_v2(row1, r1, sib2, cfg2, 0, sw)
            m3 = A3.speech_micro_v3(row3, r3, sib3, cfg3, 0, sw, k=k, stores=s3, vox=False)
            P1, P3 = int(m1["P"]), int(m3["P"])
            if P1 == P3 and np.array_equal(m1["ids"][:, :P1], m3["ids"][:, :P3]) and np.array_equal(m1["depth"][:, :P1], m3["depth"][:, :P3]):
                res["prefix_same"] += 1
            else:
                res["prefix_diff_clip"] += 1
            chk = {"ids": np.array_equal(m1["ids"][:, P1:P1 + n1], m3["ids"][:, P3:P3 + n1]),
                   "depth": np.array_equal(m1["depth"][:, P1:P1 + n1 + D.CD], m3["depth"][:, P3:P3 + n1 + D.CD]),
                   "cb0": np.array_equal(np.asarray(m1["cb0"])[P1:P1 + n1 + D.CD], np.asarray(m3["cb0"])[P3:P3 + n1 + D.CD])}
            f1, f3 = m1["sel"] - P1, m3["sel"] - P3
            k1, k3 = f1 < n1 - 1, f3 < n1 - 1
            chk["sel"] = np.array_equal(f1[k1], f3[k3])
            for key in ("y", "word", "pad", "eob", "nw"):
                chk[key] = np.array_equal(np.asarray(m1[key])[k1], np.asarray(m3[key])[k3])
            l1, l3 = m1["light"] - P1, m3["light"] - P3
            chk["light"] = np.array_equal(l1, l3[l3 < n1 + D.CD])
            if not all(chk.values()):
                # where: conversation frames whose ids differ (expected only frame n1 - 1: NEW_WORD of the first word at / after 600 s, which
                # Model 1's chunk-0 row cannot see since it ends there) and which values
                dif = np.flatnonzero((m1["ids"][:, P1:P1 + n1] != m3["ids"][:, P3:P3 + n1]).any(0))
                vals = [(int(f), [int(m1["ids"][c, P1 + f]) for c in (0, 1)], [int(m3["ids"][c, P3 + f]) for c in (0, 1)]) for f in dif[:4]]
                boundary = bool(len(dif)) and set(dif.tolist()) == {n1 - 1} and all(
                    m3["ids"][c, P3 + n1 - 1] == D.NW_ID and m1["ids"][c, P1 + n1 - 1] == D.PAD_ID
                    for c in (0, 1) if m1["ids"][c, P1 + n1 - 1] != m3["ids"][c, P3 + n1 - 1])
                ok_rest = all(v for a, v in chk.items() if a not in ("ids", "y", "pad", "nw"))
                kk = (f1 < n1 - 2); kk3 = (f3 < n1 - 2)
                ok_rest = ok_rest and all(np.array_equal(np.asarray(m1[key])[kk], np.asarray(m3[key])[kk3]) for key in ("y", "word", "pad", "eob", "nw"))
                if boundary and ok_rest:
                    res["boundary_nw_only"] = res.get("boundary_nw_only", 0) + 1
                else:
                    tok_ = False
                res["fails"].append((rk, tag, sw, {a: b for a, b in chk.items() if not b}, "diff frames", vals, "boundary-NW-only" if boundary and ok_rest else "OTHER"))
    res["targets_ok"] += tok_
    if kind == "handover-full":
        res["vox_n"] += 1
        e = s3.vox(k)[r3]
        m = mv[str(k)][r1]
        assert m["row_key"] == rk
        res["vox_ok"] += all(all(e["A"][s][f] == m["A"][s][f] for f in ("flac", "gain_db", "flac_bytes", "generation", "a_spk")) and m["A"][s]["frame0"] == 0
                             for s in ("user", "assistant"))
print("JSON " + json.dumps(res))
print(f"boundary-only differences (frame n1 - 1 = NEW_WORD of the first word after 600 s, PAD in Model 1): {res.get('boundary_nw_only', 0)} micros; "
      f"OTHER differences: {sum(1 for f in res['fails'] if f[-1] == 'OTHER' or f[1] in ('codes',) or len(f) < 5)}")
print(f"bit-identical first 7,500 frames on {res['calls']} calls ({sum(c[4] == 'm1chunks' for c in pick)} purged): codes {res['codes_ok']}, words "
      f"{res['words_ok']} (words crossing 600 s whose end Model 1 clamped: {res['words_end_clamped_m1']}), targets {res['targets_ok']} (both "
      f"orientations x train/eval cfg; prefix identical {res['prefix_same']} / {res['prefix_same'] + res['prefix_diff_clip']}), vox index == m1vox "
      f"{res['vox_ok']} / {res['vox_n']}; fails {res['fails'][:6]}")
