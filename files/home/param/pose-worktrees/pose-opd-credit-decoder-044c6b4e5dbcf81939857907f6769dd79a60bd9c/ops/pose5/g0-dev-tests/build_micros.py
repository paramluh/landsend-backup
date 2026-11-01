#!/usr/bin/env python3
"""pose5 G0COND tests: REAL 8,1,1 training micros of each row type with the G0COND copy's data code -> MICD/<name>.pkl + out/micros.json.

Same sources and row choice as workstream A's build_micros.py (code-811-s2p/tests/s2p), so the micros are comparable byte for byte:
  pose      a FRESH pose row (plan-1 segments of slinky-0 from plan-1's Weka copy, fresh_pose's own packer / seed; the fullest of the rows)
            built exactly as FreshPoseDS does: FreshRows -> data_ppm.pose_micro_ppm -> packed, with the pose datasets' cfg
            dict(CFG['data'], g0cond=1) (= the trainer under PPM_G0COND=1) -> + g0pos / g0f. pose_off.pkl = the same row with CFG['data']
            (flag off).
  speech    evalsets/speech_s3/B-000.pkl (speech_micro path, untouched by G0COND)
  gemma     data_ppm.TextDS on data/gemma-txtdup-v1 train pack 0            chat   chat_ppm.ChatDS, split 2 pack 0
Checks (asserted, recorded in out/micros.json):
  * flag off: the pose micro == A's pose.pkl (built by the ORIGINAL data code: code-811-s2p/data_ppm.py md5 == code-811's) in every key,
    dtype, shape and byte; gemma / chat == A's (built by the original code); flag on: the same micro + exactly {g0pos, g0f}.
  * every one of the packed rows (11 rows, 400 segments): g0pos = the clips' SLOT positions (kind SLOT, = clip start - 1, one per clip, in
    order), g0f == data_ppm.g0_feat(clip record) recomputed from the raw parquet frame-0 fields, valid count.
Run with the 8,1,1 env sourced (prod_811_fresh_append3.env) + RANK=0 WORLD_SIZE=1 LOCAL_RANK=0."""
import hashlib
import json
import os
import pickle
import sys
import zlib

import numpy as np

sys.path.insert(0, "/data/home/amartyavarma/ppm/pose5/code-811-g0/tests/g0")
import g0_cpu as H  # noqa: E402

sys.path.insert(0, H.COPY)
import train_ppm as T  # noqa: E402

D, CH = T.D, T.CH
dcfg = T.CFG["data"]
pcfg = dict(dcfg, g0cond=1)                                          # = train_ppm.main()'s pcfg under PPM_G0COND=1
assert "g0cond" not in dcfg
os.makedirs(H.MICD, exist_ok=True)
os.makedirs(H.OUTD, exist_ok=True)
OUT = {}


def same(a, b, path="", diffs=None):
    """deep byte equality of two micro dicts (numpy arrays: dtype + shape + bytes)."""
    diffs = [] if diffs is None else diffs
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)) or set(a) != set(b):
            diffs.append((path, "keys", sorted(set(a) ^ set(b)) if isinstance(a, dict) and isinstance(b, dict) else type(b).__name__))
            return diffs
        for k in a:
            same(a[k], b[k], f"{path}.{k}", diffs)
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a_, b_ = np.asarray(a), np.asarray(b)
        if a_.dtype != b_.dtype or a_.shape != b_.shape or a_.tobytes() != b_.tobytes():
            diffs.append((path, str(a_.dtype), str(b_.dtype), a_.shape, b_.shape))
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b) or type(a) is not type(b):
            diffs.append((path, "len/type"))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                same(x, y, f"{path}[{i}]", diffs)
    elif not (type(a) is type(b) and (a == b)):
        diffs.append((path, repr(a)[:60], repr(b)[:60]))
    return diffs


# ---- pose: A's fresh row (same selection, same builder)
FP = T._fresh_pose_module()
import pyarrow.parquet as pq  # noqa: E402

SRC = "/data/home/amartyavarma/data/new-pose-llm-data/new-pose-llm-data/jobs"
SCAN = f"{H.ORIG}/fresh/scan/slinky-0.parquet"
t = pq.ParquetFile(SCAN).read_row_group(0, columns=["file", "row", "segment_id", "video_id", "parent_clip_id", "segment_index", "n", "sam_ok"]).to_pydict()
_ex = {}
k = []
for i in range(len(t["file"])):
    if t["file"][i] not in _ex:
        _ex[t["file"][i]] = os.path.exists(f"{SRC}/{t['file'][i]}")
    if t["sam_ok"][i] and _ex[t["file"][i]]:
        k.append(i)
        if len(k) == 400:
            break
files = sorted({t["file"][i] for i in k})
fidx = {f: j for j, f in enumerate(files)}
clips = [{"video_id": t["video_id"][i], "parent_clip_id": t["parent_clip_id"][i], "segment_index": int(t["segment_index"][i]),
          "segment_id": t["segment_id"][i], "n": int(t["n"][i])} for i in k]
rng = np.random.default_rng(np.random.SeedSequence([FP.PACK_SEED, zlib.crc32(b"slinky-0"), 0]))
rows = [[c for c in cl] for cl in FP.B.pack(FP.B.units_of(clips), rng)]
row_seg = np.array([j for r in rows for j in r], np.int32)
row_off = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int64)
z = {"seg_file": np.array([fidx[t["file"][i]] for i in k], np.int32), "seg_row": np.array([t["row"][i] for i in k], np.int32),
     "seg_n": np.array([t["n"][i] for i in k], np.int32), "row_seg": row_seg, "row_off": row_off,
     "row_len": np.array([sum(int(t["n"][k[j]]) + FP.B.SLOT + FP.B.GAP for j in r) for r in rows], np.int64),
     "row_shard": np.zeros(len(rows), np.int32)}
plan = {"files": files, "segment_ids": [t["segment_id"][i] for i in k], "src_root": SRC, "host": "slinky-0", "n_rows": len(rows), "z": z}
fr = FP.FreshRows(plan, cache_files=8)


def raw_frame0(seg_j):
    """the clip's frame-0 fields straight from the raw parquet pose struct (independent of decode_new): root, width, H, W, frame0_ok."""
    r = fr._table(int(z["seg_file"][seg_j])).slice(int(z["seg_row"][seg_j]), 1).to_pylist()[0]
    P = r["pose"]
    gs = np.asarray(P["global_state"], np.float32).reshape(-1, 3)
    f0 = np.asarray(P["frame0_pose"], np.float32).reshape(-1, 3)
    Hh, Ww = float(P["orig_h"]), float(P["orig_w"])
    j55 = f0[FP.B.JOINTS] / np.array([Hh, Ww, 1.0], np.float32)
    root, width = (j55[5, :2] + j55[6, :2]) / 2, abs(j55[5, 1] - j55[6, 1])
    return {"frame0_root": [float(root[0]), float(root[1])], "frame0_width": float(width), "orig_h": Hh, "orig_w": Ww,
            "frame0_ok": bool(width > 0) and bool(np.allclose(gs[0], [*root, width], atol=1e-4)), "global_state0": gs[0].tolist()}


rowchk = []
for ri in range(len(rows)):
    row = fr[ri]
    m_on = D.packed(D.pose_micro_ppm(row, pcfg, 0), pcfg)
    gpos, gf = np.asarray(m_on["g0pos"]), np.asarray(m_on["g0f"])
    segs = z["row_seg"][z["row_off"][ri]:z["row_off"][ri + 1]]
    assert len(gpos) == len(row["clips"]) == len(segs) and gf.shape == (len(segs), D.G0_DIM) and gf.dtype == np.float32
    kind = np.asarray(row["kind"])
    assert (kind[gpos] == FP.B.KIND_SLOT).all() and (gpos == np.array([c["slot"] for c in row["clips"]])).all()
    assert (gpos + 1 == np.array([c["start"] for c in row["clips"]])).all() and (np.diff(gpos) > 0).all()
    nv, maxdiff = 0, 0.0
    for c, j, f in zip(row["clips"], segs, gf):
        rf = raw_frame0(int(j))
        f2, ok2 = D.g0_feat(rf)
        f3, ok3 = D.g0_feat(c)
        assert ok2 == ok3 and np.array_equal(f3, f), ("g0f != g0_feat(clip record)", ri)
        maxdiff = max(maxdiff, float(np.abs(f2 - f).max()))
        assert abs(c["orig_h"] - rf["orig_h"]) == 0 and abs(c["orig_w"] - rf["orig_w"]) == 0 and c["frame0_ok"] == rf["frame0_ok"]
        nv += int(ok3)
    assert maxdiff == 0.0, ("g0f from the clip record != from the raw parquet fields", maxdiff)
    rowchk.append({"row": ri, "n_clips": len(gpos), "valid": nv, "N": int(np.asarray(m_on["ids"]).shape[1]), "gpos_head": gpos[:6].tolist()})
OUT["rows_checked"] = rowchk
H.log("rows:", json.dumps(rowchk))

ri = int(np.argmax(z["row_len"]))                                      # = A's row (the fullest)
row = fr[ri]
pose_off = D.packed(D.pose_micro_ppm(row, dcfg, 0), dcfg)
pose = D.packed(D.pose_micro_ppm(row, pcfg, 0), pcfg)
OUT["pose_source"] = {"scan": SCAN, "src_root": SRC, "segments_considered": len(k), "rows_packed": len(rows), "row_index": ri,
                      "row_len": int(z["row_len"][ri]), "row_key": row["row_key"], "n_clips": len(row["clips"])}
mA = pickle.load(open(f"{H.MICD_A}/pose.pkl", "rb"))
dA = same(mA, pose_off)
OUT["pose_off_vs_A_original_code_diffs"] = dA[:10]
assert not dA, ("flag-off pose micro != A's (original data code)", dA[:5])
d_on = same({k_: v for k_, v in pose.items() if k_ not in ("g0pos", "g0f")}, pose_off)
assert set(pose) - set(pose_off) == {"g0pos", "g0f"} and not d_on, (sorted(set(pose) ^ set(pose_off)), d_on[:5])
OUT["pose_on_extra_keys"] = sorted(set(pose) - set(pose_off))
OUT["pose_g0"] = {"g0pos": np.asarray(pose["g0pos"]).tolist(), "g0f": np.round(np.asarray(pose["g0f"]), 5).tolist(),
                  "slot_ids_B": np.asarray(pose["ids"])[1, pose["g0pos"]].tolist(), "slot_present_B": np.asarray(pose["present"])[1, pose["g0pos"]].tolist(),
                  "slot_depth_B_cb0": np.asarray(pose["depth"])[1, pose["g0pos"], 0].tolist(), "slot_pdepth_cb0": np.asarray(pose["pdepth"])[pose["g0pos"], 0].tolist(),
                  "slot_text_target_sel": bool(np.isin(pose["g0pos"], pose["sel"]).all())}

# ---- speech / gemma / chat (as A)
speech = pickle.load(open("/data/home/amartyavarma/ppm/evalsets/speech_s3/B-000.pkl", "rb"))
gst = D.TXD.Store("/data/home/amartyavarma/data/gemma-txtdup-v1")
gcfg = dict(dcfg, txt_cap=int(gst.cap))
gemma = D.TextDS("/data/home/amartyavarma/data/gemma-txtdup-v1", "train", [0], gcfg)[0]
cp = CH.multi_split_packs([2], dcfg["txt_cap"], T.CFG["chat_seed"])[0]
chat = CH.ChatDS([cp], dcfg)[0]
for nm, m in (("speech", speech), ("gemma", gemma), ("chat", chat)):
    dd = same(pickle.load(open(f"{H.MICD_A}/{nm}.pkl", "rb")), m)
    OUT[f"{nm}_vs_A_diffs"] = dd[:10]
    assert not dd, (nm, dd[:5])
for nm, m in (("pose", pose), ("pose_off", pose_off), ("speech", speech), ("gemma", gemma), ("chat", chat)):
    pickle.dump(m, open(f"{H.MICD}/{nm}.pkl", "wb"), protocol=4)
    OUT[f"{nm}_sha"] = hashlib.sha256(open(f"{H.MICD}/{nm}.pkl", "rb").read()).hexdigest()[:32]
OUT["dcfg"] = dcfg
OUT["PASS"] = True
json.dump(OUT, open(f"{H.OUTD}/micros.json", "w"), indent=1, default=str)
H.log("PASS micros ->", H.MICD, "| pose g0:", json.dumps(OUT["pose_g0"])[:1500])
