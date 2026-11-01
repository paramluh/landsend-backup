"""poseduplex PHASE 1 coverage (CPU, read-only): which sam-sep speech-duplex micros (first chunks x 2 orientations, the 29345 plan's
rows) have the channel-B speaker's pose in the handover copy (aloe-handover-full/DATA/files/none/<session>/<seg>/w00.SPEAKER_0N.pose_tokens.npy).
B = data 'assistant' (SPEAKER_00) on swap 0, data 'user' (SPEAKER_01) on swap 1 (data_aloe_v2 rule).
usage: python -I coverage.py OUT.json"""
import collections
import csv
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

STORES = [f"/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores/m1n{k}" for k in range(4)]
DATA = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
FPS = 12.5


def main(out):
    man = {}
    for r in csv.DictReader(open(f"{DATA}/manifest.csv")):
        man[(r["session_id"], r["seg"])] = r
    pose = {}
    for r in csv.DictReader(open(f"{DATA}/pose_manifest.csv")):
        pose[(r["session_id"], r["seg"], r["speaker"])] = r
    st = collections.Counter()
    hrs = collections.Counter()
    reasons = collections.Counter()
    per_conv = {}
    hdr_bad = []
    for k, root in enumerate(STORES):
        t = pq.read_table(f"{root}/rows.parquet").to_pydict()
        for i in range(len(t["row"])):
            if int(t["chunk_index"][i]) != 0:
                continue
            sp = t["split"][i]
            ses, seg = t["src_conversation"][i].split("/")
            n = int(t["n_frames"][i])
            m = man[(ses, seg)]
            assert m["relative_dir"].startswith("files/none/"), (ses, seg, m["relative_dir"])
            assert int(m["mimi_T"]) == int(t["src_frames"][i]), (ses, seg, m["mimi_T"], t["src_frames"][i])
            st[f"rows_{sp}"] += 1
            for sw in (0, 1):
                spk = t["user_speaker_dir"][i] if sw else t["assistant_speaker_dir"][i]
                assert spk == ("SPEAKER_01" if sw else "SPEAKER_00"), (ses, seg, sw, spk)
                p = pose.get((ses, seg, spk))
                st[f"micros_{sp}"] += 1
                hrs[f"micros_{sp}"] += n / FPS / 3600
                ok = p is not None and p["pose_status"] == "ok"
                if ok:
                    assert int(p["pose_T"]) == int(p["mimi_T"]) == int(m["mimi_T"]), (ses, seg, spk, p["pose_T"], p["mimi_T"])
                    f = f"{DATA}/{m['relative_dir']}/{p['pose_tokens_file']}"
                    a = np.load(f, mmap_mode="r")
                    if a.shape != (32, int(m["mimi_T"])) or a.dtype != np.int16:
                        hdr_bad.append([f, list(a.shape), str(a.dtype)])
                        ok = False
                    st[f"pose_ok_{sp}"] += 1
                    hrs[f"pose_ok_{sp}"] += n / FPS / 3600
                else:
                    reasons[(p["pose_status"] if p else "absent") + ": " + ((p["reason"] or "")[:28] if p else "")] += 1
                per_conv.setdefault(f"{ses}/{seg}", {})[spk] = int(ok)
    both = collections.Counter(sum(v.values()) for v in per_conv.values())
    res = {"counts": dict(st), "hours": {k: round(v, 1) for k, v in hrs.items()}, "convs_by_n_ok_sides": dict(both),
           "not_ok_reasons_top": reasons.most_common(25), "header_bad": hdr_bad[:20], "n_header_bad": len(hdr_bad)}
    for sp in ("train", "eval"):
        res[f"frac_micros_pose_ok_{sp}"] = round(st[f"pose_ok_{sp}"] / max(st[f"micros_{sp}"], 1), 4)
        res[f"frac_hours_pose_ok_{sp}"] = round(hrs[f"pose_ok_{sp}"] / max(hrs[f"micros_{sp}"], 1e-9), 4)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
