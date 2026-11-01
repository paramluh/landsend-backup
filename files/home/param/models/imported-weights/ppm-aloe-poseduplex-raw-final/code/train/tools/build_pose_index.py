"""poseduplex DATA (10-25): channel-B pose index for the sam-sep speech-duplex rows (job 29345's stores, FIRST chunks only, train + eval).
NO new pose store is written (/data is at 99 %): the trainer reads B's pose tokens IN PLACE from the handover copy
  /data/home/amartyavarma/data/aloe-handover-full/DATA/files/none/<session>/<seg>/w00.SPEAKER_0N.pose_tokens.npy  (int16 [32, T], 12.5 Hz)
and this index pins every file: sha256 (recomputed here == the pose json's pose_tokens_sha256), byte size (re-checked per read), T (== the
manifest's mimi_T == the sam-sep row's src_frames), tokenizer InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f0 (= PPM's / v5's),
normalisation frame0-root-frame0-horizontal-shoulder-width-v1 (= PPM's), 12.5 Hz, codes in [0, 2048).
A side is pose-OK iff pose_manifest pose_status == 'ok' and every check above passes (anything else: ok 0 + the reason; nothing is repaired).
Sides: data 'assistant' = SPEAKER_00 (B on swap 0), data 'user' = SPEAKER_01 (B on swap 1) (asserted from the store row).
usage (CPU node): python -I build_pose_index.py <out json> [workers]"""
import csv
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

STORES = [f"/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores/m1n{k}" for k in range(4)]
DIGESTS = ["7318ffb3ea75cbcc3ff48b63b6cf8b88", "31ecb2273542310096936c0baf49efa8", "1759200f7818c6017dfb6a46e37fa5df", "611b2e7edb1503cc600a6489e49dcf40"]
DATA = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
TOKENIZER = "InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f0"
NORM = "frame0-root-frame0-horizontal-shoulder-width-v1"
SIDE_SPK = {"assistant": "SPEAKER_00", "user": "SPEAKER_01"}


def check(job):
    """one (conversation dir, speaker) -> entry dict."""
    rel, spk, T, man_status, man_reason = job
    d = f"{DATA}/{rel}"
    e = {"spk": spk, "ok": 0}
    if man_status != "ok":
        e["why"] = f"manifest {man_status}: {man_reason[:80]}"
        return (rel, spk), e
    try:
        pj = json.load(open(f"{d}/w00.{spk}.pose.json"))
        f = f"{d}/{pj['pose_tokens_file']}"
        raw = open(f, "rb").read()
        sha = hashlib.sha256(raw).hexdigest()
        a = np.load(f, mmap_mode="r")
        errs = []
        if pj.get("status") != "ok" or pj.get("speaker") != spk:
            errs.append(f"json status {pj.get('status')} speaker {pj.get('speaker')}")
        if not str(pj.get("tokenizer", "")).startswith(TOKENIZER):
            errs.append(f"tokenizer {pj.get('tokenizer')}")
        if not str((pj.get("normalization") or {}).get("recipe", "")).startswith(NORM):
            errs.append("normalization")
        if float(pj.get("token_rate_hz", 0)) != 12.5:
            errs.append(f"rate {pj.get('token_rate_hz')}")
        if sha != pj.get("pose_tokens_sha256"):
            errs.append("sha256 != json")
        if a.shape != (32, T) or tuple(pj.get("pose_tokens_shape", [])) != (32, T) or a.dtype != np.int16:
            errs.append(f"shape {a.shape} {a.dtype} vs T {T}")
        elif int(a.min()) < 0 or int(a.max()) >= 2048:
            errs.append(f"codes out of range [{int(a.min())}, {int(a.max())}]")
        if errs:
            e["why"] = "; ".join(errs)
            return (rel, spk), e
        e.update(ok=1, path=f"{rel}/{pj['pose_tokens_file']}", sha256=sha, bytes=len(raw), T=T)
    except Exception as ex:                                        # noqa: BLE001  (missing file etc.: reported, never repaired)
        e["why"] = f"{type(ex).__name__}: {str(ex)[:120]}"
    return (rel, spk), e


def main(out, workers=8):
    t0 = time.time()
    man = {(r["session_id"], r["seg"]): r for r in csv.DictReader(open(f"{DATA}/manifest.csv"))}
    pman = {(r["session_id"], r["seg"], r["speaker"]): r for r in csv.DictReader(open(f"{DATA}/pose_manifest.csv"))}
    rows, jobs = {}, {}
    for k, root in enumerate(STORES):
        nd = json.load(open(f"{root}/NODE.json"))
        t = pq.read_table(f"{root}/rows.parquet").to_pydict()
        assert t["row"] == list(range(len(t["row"]))), root
        rows[str(k)] = {}
        for i in range(len(t["row"])):
            if int(t["chunk_index"][i]) != 0:
                continue
            ses, seg = t["src_conversation"][i].split("/")
            m = man[(ses, seg)]
            assert m["relative_dir"] == f"files/none/{ses}/{seg}", (ses, seg, m["relative_dir"])
            T = int(m["mimi_T"])
            assert T == int(t["src_frames"][i]) and int(t["frame_start"][i]) == 0 and int(t["n_frames"][i]) <= min(T, 7500), (ses, seg)
            ent = {"row_key": t["row_key"][i], "conv": f"{ses}/{seg}", "split": t["split"][i], "n": int(t["n_frames"][i]), "T": T, "sides": {}}
            for side, spk in SIDE_SPK.items():
                assert t[f"{side}_speaker_dir"][i] == spk, (ses, seg, side, t[f"{side}_speaker_dir"][i])
                p = pman.get((ses, seg, spk))
                jobs[(m["relative_dir"], spk)] = (m["relative_dir"], spk, T, p["pose_status"] if p else "absent", (p or {}).get("reason", "") or "")
                ent["sides"][side] = (m["relative_dir"], spk)
            rows[str(k)][str(int(t["row"][i]))] = ent
        assert len(nd.get("shards", [])) == 1
    with Pool(workers) as pool:
        res = dict(pool.map(check, list(jobs.values()), chunksize=16))
    cnt = {"rows": 0, "rows_train": 0, "rows_eval": 0, "sides_ok": 0, "sides_not_ok": 0, "micros_ok_train": 0, "micros_ok_eval": 0}
    why = {}
    for k in rows:
        for r, ent in rows[k].items():
            cnt["rows"] += 1; cnt[f"rows_{ent['split']}"] += 1
            for side in SIDE_SPK:
                e = res[ent["sides"][side]]
                ent["sides"][side] = e
                if e["ok"]:
                    cnt["sides_ok"] += 1; cnt[f"micros_ok_{ent['split']}"] += 1
                else:
                    cnt["sides_not_ok"] += 1
                    w = e["why"].split(":")[0] if e["why"].startswith("manifest") else e["why"][:40]
                    why[w] = why.get(w, 0) + 1
    j = {"format": "posedup-pose-index-v1", "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "data_root": DATA, "stores": STORES, "store_digests": DIGESTS,
         "pose_tokenizer": TOKENIZER, "normalization": NORM, "token_rate_hz": 12.5, "pose_fps": 25, "side_speaker": SIDE_SPK,
         "rule": "B pose OK iff pose_manifest status ok + json status / speaker / tokenizer / normalisation / 12.5 Hz / sha256 / shape (32, T) / "
                 "codes in [0, 2048) all pass; nothing repaired",
         "counts": cnt, "not_ok_reasons": dict(sorted(why.items(), key=lambda x: -x[1])), "rows": rows}
    tmp = out + ".tmp"
    json.dump(j, open(tmp, "w"))
    os.replace(tmp, out)
    dg = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(json.dumps({"counts": cnt, "not_ok_top": list(j["not_ok_reasons"].items())[:12], "sha256": dg, "files_checked": len(jobs),
                      "time_s": round(time.time() - t0)}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
