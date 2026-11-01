#!/usr/bin/env python3
"""pose5 s2p tests: build REAL 8,1,1 training micros of each row type with the trainer's own data code (read-only sources on Weka) ->
SCR/micros/<name>.pkl (node-local) + out/micros.json (summary + the channel / row-type facts the s2p port relies on).

  pose    a FRESH pose row (new-pose-llm-data, the 8,1,1 pose stream's format) built exactly as fresh_pose.FreshPoseDS does
          (FreshRows -> data_ppm.pose_micro_ppm -> packed) from plan-1 segments of slinky-0 (fresh/scan/slinky-0.parquet, the first
          sam_ok segments in host order) read from plan-1's Weka copy data/new-pose-llm-data/new-pose-llm-data/jobs, packed with
          fresh_pose's own packer and seed rule (shard 0 of host slinky-0). Not one of the planned training rows (those plans are
          node-local on the production hosts), but the same builder / segments / format.
  speech  a podcast speech row: evalsets/speech_s3/B-000.pkl (built through data_ppm.MixedSpeechDS.get = speech_micro, the training rows'
          path; the 811 eval set on Weka).
  gemma   a Gemma text-duplex pack: data_ppm.TextDS (the training class) on data/gemma-txtdup-v1 'train' pack 0 (Weka copy of the format;
          the training root txtdup-full is node-local).
  chat    a corpus15m chat pack: chat_ppm.ChatDS (the training class) on split 2 pack 0 of multi_split_packs (Weka).
Run with the 8,1,1 env sourced (prod_811_fresh_append3.env) + RANK=0 WORLD_SIZE=1 LOCAL_RANK=0."""
import json
import os
import pickle
import sys
import zlib

import numpy as np

sys.path.insert(0, "/data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p")
import s2p_cpu as H  # noqa: E402

sys.path.insert(0, H.COPY)
import train_ppm as T  # noqa: E402

D, CH = T.D, T.CH
dcfg = T.CFG["data"]
os.makedirs(H.MICD, exist_ok=True)
os.makedirs(H.OUTD, exist_ok=True)
OUT = {}


def facts(m):
    """the s2p-relevant facts of a micro."""
    f = {"kind": m["kind"], "src": m.get("src", m["kind"]), "N": int(np.asarray(m["ids"]).shape[1]), "segs": [int(x) for x in m["segs"]][:8],
         "n_segs": len(m["segs"]), "has_depth": "depth" in m, "has_pdepth": "pdepth" in m}
    if "depth" in m:
        dep = np.asarray(m["depth"])
        pres = np.asarray(m["present"])
        f.update(chanA_codes=int((dep[0, :, 0] >= 0).sum()), chanB_codes=int((dep[1, :, 0] >= 0).sum()),
                 chanA_present=float(pres[0].sum()), chanB_present=float(pres[1].sum()))
    if "pdepth" in m:
        pdep, dep, pcb0 = np.asarray(m["pdepth"]), np.asarray(m["depth"]), np.asarray(m["pcb0"])
        hs, hp, lab = dep[1, :, 0] >= 0, pdep[:, 0] >= 0, pcb0 >= 0
        f.update(pose_code_positions=int(hp.sum()), pose_label_positions=int(lab.sum()), speechB_code_positions=int(hs.sum()),
                 pose_labels_with_speechB_same_T=int((lab & hs).sum()), pose_labels_without_speechB=int((lab & ~hs).sum()),
                 speechB_without_pose=int((hs & ~hp).sum()),
                 label_is_pdepth_cb0=bool((pcb0[lab] == pdep[lab, 0]).all()), cb0_is_depthB_cb0=bool((np.asarray(m["cb0"])[hs] == dep[1, hs, 0]).all()),
                 frame_check="depth[1][T] and pdepth[T] are frames T - CD of the same clip (pose_s3q.segment_arrays: src = s0 + q - CD, "
                             f"pose_delay = cfg pose_delay {dcfg['pose_delay']} = CD {D.CD})")
        f["chanA_ids_all_PAD"] = bool((np.asarray(m["ids"])[0] == D.PAD_ID).all())
        f["prov"] = json.loads(m["prov"]) if isinstance(m.get("prov"), str) else None
    return f


# ---- pose: a fresh row by fresh_pose's own builder
FP = T._fresh_pose_module()
import pyarrow.parquet as pq  # noqa: E402

SRC = "/data/home/amartyavarma/data/new-pose-llm-data/new-pose-llm-data/jobs"
SCAN = f"{H.ORIG}/fresh/scan/slinky-0.parquet"
t = pq.ParquetFile(SCAN).read_row_group(0, columns=["file", "row", "segment_id", "video_id", "parent_clip_id", "segment_index", "n", "sam_ok"]).to_pydict()
_ex = {}                                                               # (the Weka copy lacks some releases' parquet: use segments it holds)
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
ri = int(np.argmax(z["row_len"]))                                      # the fullest of these rows
row = fr[ri]
pose = D.packed(D.pose_micro_ppm(row, dcfg, 0), dcfg)
OUT["pose_source"] = {"scan": SCAN, "src_root": SRC, "segments_considered": len(k), "files_checked": len(_ex), "files_missing_on_weka": sum(not v for v in _ex.values()), "rows_packed": len(rows), "row_index": ri,
                      "row_len": int(z["row_len"][ri]), "row_key": row["row_key"], "n_clips": len(row["clips"])}

# ---- speech: the 811 speech_s3 eval micro (speech_micro path)
speech = pickle.load(open("/data/home/amartyavarma/ppm/evalsets/speech_s3/B-000.pkl", "rb"))

# ---- gemma: TextDS (training class) on the Weka gemma-txtdup-v1 store
gst = D.TXD.Store("/data/home/amartyavarma/data/gemma-txtdup-v1")
gcfg = dict(dcfg, txt_cap=int(gst.cap))
gemma = D.TextDS("/data/home/amartyavarma/data/gemma-txtdup-v1", "train", [0], gcfg)[0]
OUT["gemma_source"] = {"store": "/data/home/amartyavarma/data/gemma-txtdup-v1", "split": "train", "pack": 0, "cap": int(gst.cap)}

# ---- chat: ChatDS (training class), split 2 pack 0
cp = CH.multi_split_packs([2], dcfg["txt_cap"], T.CFG["chat_seed"])[0]
chat = CH.ChatDS([cp], dcfg)[0]
OUT["chat_source"] = {"split": 2, "pack": 0, "rows": len(cp[1])}

for nm, m in (("pose", pose), ("speech", speech), ("gemma", gemma), ("chat", chat)):
    pickle.dump(m, open(f"{H.MICD}/{nm}.pkl", "wb"), protocol=4)
    OUT[nm] = facts(m)
    H.log(nm, json.dumps({k_: v_ for k_, v_ in OUT[nm].items() if k_ != "prov"}))
OUT["dcfg"] = dcfg
OUT["CD"] = int(D.CD)
json.dump(OUT, open(f"{H.OUTD}/micros.json", "w"), indent=1, default=str)
H.log("micros ->", f"{H.MICD}", "| summary", f"{H.OUTD}/micros.json")
