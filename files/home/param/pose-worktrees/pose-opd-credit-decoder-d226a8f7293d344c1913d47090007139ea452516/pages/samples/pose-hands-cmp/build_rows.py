"""pose-hands-cmp ROWS (10-17): 16 held-out aloe-pose-ft conversations = the 8 of ppm/samples/aloe-pose-ft-430/rows (B-000..007, copied
unchanged) + 8 more from the SAME candidate list (c00, both tracks qualify, >= 60.8 s): cand[(2 i + 1) * n // 16], i = 0..7 (the midpoints
between the existing picks cand[i * n // 8], so never the same conversation).
  duo  : python build_rows.py duo  <out>   (aloe-pose-ft code)   -> B-iii.pkl (swap 0) + A-iii.pkl (swap 1) + rows.json (= build_rows.py of
         aloe-pose-ft-430/work, rows 008..015 new)
  mono : python build_rows.py mono <out>   (aloe-uncond-ft code; source aloe-uncond-ft/aloe_uncond.env first) -> B-000..015 = swap 0
         (B = SPEAKER_00 = s1 of the duo row; 000..007 copied from aloe-uncond-ft-430/rows), B-016..031 = swap 1 of conversation i - 16
         (B = SPEAKER_01 = s2 of the duo row), each with its own speaker's normalisation in norm['s1'] (the mono renderer reads 's1').
CPU only."""
import ast
import json
import os
import pickle
import shutil
import sys

import numpy as np
import pyarrow.parquet as pq

MODE, OUT = sys.argv[1], sys.argv[2]
assert MODE in ("duo", "mono"), MODE
if MODE == "mono":
    assert os.environ.get("QQA_A_EMPTY") == "1" and os.environ.get("QQA_POSE_B") == "1", "source aloe-uncond-ft/aloe_uncond.env first"
    sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-uncond-ft/code")
else:
    sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-pose-ft/code")
import data_aloe_pose as A  # noqa: E402

os.makedirs(OUT, exist_ok=True)
REFD = "/data/home/amartyavarma/ppm/samples/aloe-pose-ft-430/rows"
REFM = "/data/home/amartyavarma/ppm/samples/aloe-uncond-ft-430/rows"
REF = json.load(open(f"{REFD}/rows.json"))
sp, _ = A.splits(A.SRC_STORES)
ev = sp["eval"]
both = {}
for k, r, sw in ev:
    both.setdefault((k, r), set()).add(sw)
tabs = [pq.read_table(f"{root}/rows.parquet", columns=["row_key", "n_frames", "chunk_index", "src_dir"]).to_pylist() for root in A.SRC_STORES]
cand = sorted(kr for kr, s in both.items() if s == {0, 1} and tabs[kr[0]][kr[1]]["chunk_index"] == 0 and tabs[kr[0]][kr[1]]["n_frames"] >= 760)
n = len(cand)
old = [cand[i * n // 8] for i in range(8)]
assert old == [(int(r["store"]), int(r["node_row"])) for r in REF], "candidate list differs from the existing page's"
new = [cand[(2 * i + 1) * n // 16] for i in range(8)]
assert not set(new) & set(old) and len(set(new)) == 8
convs = old + new
print(MODE, "candidates", n, "convs", convs, flush=True)
cfg = A.aloe_cfg()
st = A.Stores(A.SRC_STORES)


def norm_of(t, S):
    pj = json.load(open(f"{t['src_dir']}/w00.{S}.pose.json"))
    nm = pj["normalization"]
    nm = nm if isinstance(nm, dict) else ast.literal_eval(nm)
    return {"frame0_root": nm["root0_yx_normalized"], "frame0_width": nm["shoulder_width0_normalized"],
            "orig_w": nm.get("orig_w"), "orig_h": nm.get("orig_h"), "lip_lag_ms": pj.get("alignment", {}).get("lip_sync_best_lag_ms")}


def dump(m, fn):
    pickle.dump(m, open(fn + ".tmp", "wb"))
    os.replace(fn + ".tmp", fn)


rows = []
if MODE == "duo":
    for i, (k, r) in enumerate(convs):
        t = tabs[k][r]
        if i < 8:
            for pre in ("A", "B"):
                shutil.copyfile(f"{REFD}/{pre}-{i:03d}.pkl", f"{OUT}/{pre}-{i:03d}.pkl")
            row = dict(REF[i])
        else:
            for sw, pre in ((1, "A"), (0, "B")):
                m = st.conv((k, r, sw), cfg, 0)
                assert m["swap"] == sw
                pv = json.loads(m["prov"])
                assert pv["chan_B_data_side"] == ("assistant" if sw == 0 else "user"), pv["chan_B_data_side"]
                dump(m, f"{OUT}/{pre}-{i:03d}.pkl")
            row = {"row": f"B-{i:03d}", "store": int(k), "node_row": int(r), "row_key": m["row_key"], "P": int(m["P"]), "T": int(len(m["pid"])),
                   "norm": {"s1": norm_of(t, "SPEAKER_00"), "s2": norm_of(t, "SPEAKER_01")}}
        row["conv"] = i
        rows.append(row)
        print(row, flush=True)
else:
    REFU = json.load(open(f"{REFM}/rows.json"))
    tr = {(k, r) for k, r, _ in sp["train"]}
    evs = set(ev)
    for sw in (0, 1):
        for i, (k, r) in enumerate(convs):
            j = i + 16 * sw
            t = tabs[k][r]
            S = "SPEAKER_00" if sw == 0 else "SPEAKER_01"
            if sw == 0 and i < 8:
                shutil.copyfile(f"{REFM}/B-{i:03d}.pkl", f"{OUT}/B-{j:03d}.pkl")
                row = dict(REFU[i])
                assert row["row_key"] == REF[i]["row_key"]
            else:
                assert (k, r, sw) in evs and (k, r) not in tr, ("not a held-out item of the uncond run", k, r, sw)
                m = st.conv((k, r, sw), cfg, 0)
                assert m["swap"] == sw
                pv = json.loads(m["prov"])
                assert pv["chan_B_data_side"] == ("assistant" if sw == 0 else "user") and pv["chan_A"] == "empty", pv
                P = int(m["P"])
                assert (m["ids"][0] == A.D.PAD_ID).all() and (m["depth"][0] < 0).all() and (m["present"][0] == 0).all(), "channel A not empty"
                assert (m["pdepth"][P + A.D.CD:P + A.D.CD + 752] >= 0).all(), "B pose missing in the 60 s window"
                dump(m, f"{OUT}/B-{j:03d}.pkl")
                row = {"row": f"B-{j:03d}", "store": int(k), "node_row": int(r), "row_key": m["row_key"], "P": P, "T": int(len(m["pid"])),
                       "voice_clip": pv["clips"], "persona": pv["persona"], "dropped_A": pv["dropped_A"], "norm": {"s1": norm_of(t, S)}}
            row.update(conv=i, swap=sw, duo_spk="s1" if sw == 0 else "s2")
            rows.append(row)
            print(row, flush=True)
json.dump(rows, open(f"{OUT}/rows.json", "w"), indent=1)
print("done", len(rows))
