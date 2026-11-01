"""fraction of the trained B-track frames that lie in a SAM chunk flagged 'swapped' (chunk_consistency.swapped[<speaker>]) or 'ambiguous'."""
import json
import numpy as np
import pyarrow.parquet as pq
SRC, SAM = "/data/home/amartyavarma/ppm/aloe-pose-ft/stores", "/data/home/amartyavarma/data/aloe-sam-sep/hf"
P = {f"{r['session']}/{r['seg']}": json.loads(r["chunk_consistency"]) for r in pq.read_table(f"{SAM}/index/pairs.parquet", columns=["session", "seg", "chunk_consistency"]).to_pylist()}
tot = {"train": 0, "eval": 0}; sw = {"train": 0, "eval": 0}; amb = {"train": 0, "eval": 0}; convs_sw = set(); ex = []
for k in range(8):
    rows = pq.read_table(f"{SRC}/p{k}/rows.parquet", columns=["src_conversation", "frame_start", "n_frames"]).to_pylist()
    pi = np.load(f"{SRC}/p{k}/pose_index.npz")
    for i, r in enumerate(rows):
        cc = P[r["src_conversation"]]; sp = "eval" if pi["split"][i] else "train"
        for j, spk in ((0, "SPEAKER_01"), (1, "SPEAKER_00")):          # b_ok columns (user, assistant)
            if not pi["b_ok"][i, j]:
                continue
            a, b = r["frame_start"], r["frame_start"] + r["n_frames"]; tot[sp] += b - a
            for ch in cc["chunks"]:
                s, e = int(round(ch["start_s"] * 12.5)), int(round(ch["end_s"] * 12.5)); ov = max(0, min(b, e) - max(a, s))
                if ch["chunk"] in cc["swapped"].get(spk, []):
                    sw[sp] += ov; convs_sw.add((r["src_conversation"], spk))
                if ch["chunk"] in cc.get("ambiguous", []) or any(isinstance(x, dict) and x.get("chunk") == ch["chunk"] for x in cc.get("ambiguous", [])):
                    amb[sp] += ov
print(json.dumps({"B_frames": tot, "B_frames_in_swapped_chunks": sw, "frac_swapped": {s: round(sw[s] / tot[s], 5) for s in tot},
                  "B_frames_in_ambiguous_chunks": amb, "frac_ambiguous": {s: round(amb[s] / tot[s], 5) for s in tot}, "B_tracks_with_swapped": len(convs_sw)}))
